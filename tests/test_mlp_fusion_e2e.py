"""Phase 7 Stage 1 端到端測試：真實 vLLM Qwen2MLP 融合驗證。

流程：
1. 建立小型 Qwen2MLP（hidden=64, intermediate=256）
2. 模擬 dispatch_cpu_unquantized_gemm 的權重 stash（_dml_gate_fp32/_dml_up_fp32/_dml_w_fp32）
3. patch Qwen2MLP.forward（mlp_fusion.patch_qwen2_mlp）
4. 對比 fused 輸出 vs 原始 forward 輸出（數值 + 統計）
"""
import sys
import os
import tempfile
import torch

# 必須在 import vllm 之前設定：
# 1. VLLM_TARGET_DEVICE=cpu → vLLM 偵測 CpuPlatform（否則 UnspecifiedPlatform，
#    CustomOp.dispatch_forward 會走 forward_cuda 而炸）
# 2. VLLM_USE_DML=0 → 停用插件對 F.linear 的全域攔截，讓「原始 forward 參考」是純 CPU
os.environ["VLLM_TARGET_DEVICE"] = "cpu"
os.environ["VLLM_USE_DML"] = "0"

sys.path.insert(0, ".")

# 坑 4：vLLM 模型建構子需要分散式環境已初始化（Qwen2MLP 建構會呼叫 get_tp_group）
# vLLM 0.31.0 的 initialize_model_parallel 需要 get_current_vllm_config()，
# 測試環境沒有 VllmConfig，用最小 fake config 取代。
import vllm.config as vllm_config_mod  # noqa: F401  （確保 package 已載入）


class _FakeParallelConfig:
    data_parallel_size = 1
    enable_elastic_ep = False
    data_parallel_master_ip = None
    _coord_store_port = None
    use_all2all = False
    enable_eplb = False


class _FakeCompilationConfig:
    enabled_custom_ops = set()
    disabled_custom_ops = set()
    custom_ops = ["all"]


class _FakeVllmConfig:
    parallel_config = _FakeParallelConfig()
    engram_config = None
    model_config = None
    compilation_config = _FakeCompilationConfig()


# 注意：vllm.config 是 package，get_current_vllm_config 定義在 vllm/config/vllm.py 模組層。
# 需要 patch 兩層：
#   1. 模組層（vllm.config.vllm）— vllm.py 內部的全域查詢（get_cached_compilation_config 用）
#   2. package 層（vllm.config）— `from vllm.config import ...` 的 re-export（parallel_state 用）
import vllm.config as vllm_config_mod
import vllm.config.vllm as vllm_config_vllm_mod

vllm_config_vllm_mod.get_current_vllm_config = lambda: _FakeVllmConfig()
vllm_config_mod.get_current_vllm_config = lambda: _FakeVllmConfig()

from vllm.distributed.parallel_state import (
    init_distributed_environment,
    ensure_model_parallel_initialized,
)

_init_file = os.path.join(tempfile.mkdtemp(), "init")
init_distributed_environment(
    world_size=1, rank=0, distributed_init_method=f"file://{_init_file}",
    local_rank=0, backend="gloo",
)
ensure_model_parallel_initialized(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

from vllm.model_executor.models.qwen2 import Qwen2MLP
from vllm.model_executor.layers.linear import MergedColumnParallelLinear, RowParallelLinear

torch.manual_seed(7)

H, I = 64, 256
mlp = Qwen2MLP(hidden_size=H, intermediate_size=I, hidden_act="silu")
mlp.eval()

# 模擬 process_weights_after_loading 的 cpu_linear 設定（但保留權重，測試用）
import torch.nn.functional as F
mlp.gate_up_proj.cpu_linear = F.linear
mlp.down_proj.cpu_linear = F.linear

# vLLM 未載入權重是 1e37 標記值，覆蓋為真實隨機權重（模擬 load_weights）
with torch.no_grad():
    mlp.gate_up_proj.weight.data.normal_(0, 0.02)
    mlp.down_proj.weight.data.normal_(0, 0.02)

# 模擬 dispatch_cpu_unquantized_gemm 的 stash（權重移除前擷取）
gu = mlp.gate_up_proj
w_fp32 = gu.weight.data.to(torch.float32)
i = gu.output_sizes[0]
gu._dml_gate_fp32 = w_fp32[:i].t().contiguous()
gu._dml_up_fp32 = w_fp32[i:].t().contiguous()
mlp.down_proj._dml_w_fp32 = mlp.down_proj.weight.data.to(torch.float32).t().contiguous()

# 原始 forward 參考（stash 前先算一次）
x = torch.randn(3, H, dtype=torch.float32)
with torch.no_grad():
    ref = mlp.forward(x)

# patch 融合路徑
import vllm_xpu_minimal.device.mlp_fusion as mlp_fusion_mod
ok_patch = mlp_fusion_mod.patch_qwen2_mlp()
print(f"patch_qwen2_mlp: {ok_patch}")

with torch.no_grad():
    out = mlp.forward(x)

diff = (out - ref).abs().max().item()
denom = ref.abs().max().item()
print(f"fused vs original: max_diff={diff:.6e} (rel={diff/denom:.2e})")
print(f"fused_calls={mlp_fusion_mod.g_mlp_fused_calls} (fallback={mlp_fusion_mod.g_mlp_fallback_calls})")
mlp_fusion_mod.log_mlp_stats()

assert mlp_fusion_mod.g_mlp_fused_calls > 0, "融合路徑未被呼叫！"
assert diff < max(1e-3, 1e-5 * denom), "數值不符！"
print("✓ Qwen2MLP 融合路徑端到端驗證通過")
