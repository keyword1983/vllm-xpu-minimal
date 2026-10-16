"""Phase 8 Stage A 測試：圖捕獲與 IR（torch.export → fx → IR → interpret 對拍）。

1. 簡單 SwiGLU MLP：快速驗證 pipeline（不依賴 vLLM）
2. 真實 vLLM Qwen2MLP：驗證高層 op 保留（linear/slice/silu/mul）與對拍
"""
import sys
import os
import tempfile
import torch

sys.path.insert(0, ".")

from vllm_xpu_minimal.compile.fx_to_ir import fx_to_ir, bind_weights
from vllm_xpu_minimal.compile.ir import interpret, IRGraph

torch.manual_seed(0)


def _bind_from_module(ep, graph: IRGraph, module: torch.nn.Module) -> IRGraph:
    """依 placeholder 名與 module 參數名匹配，綁定權重。"""
    pmap = {}
    for n in ep.graph.nodes:
        if n.op == "placeholder":
            for pname, pt in module.named_parameters():
                if "weight" in n.name and pname.replace(".", "_") in n.name:
                    pmap[n.name] = pt
    bind_weights(graph, pmap)
    return graph


def _interpret_with(ep, graph: IRGraph, x: torch.Tensor) -> torch.Tensor:
    in_name = next(
        n.name for n in ep.graph.nodes
        if n.op == "placeholder" and "x" in n.name
    )
    return interpret(graph, {in_name: x})


def test_simple_mlp():
    """簡單 SwiGLU MLP：export → IR → interpret 對拍。"""
    from torch.export import export

    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up = torch.nn.Linear(64, 256, bias=False)
            self.down = torch.nn.Linear(128, 64, bias=False)

        def forward(self, x):
            gu = self.gate_up(x)
            g, u = gu[..., :128], gu[..., 128:]
            a = torch.nn.functional.silu(g) * u
            return self.down(a)

    m = MLP().eval()
    x = torch.randn(3, 64)
    ep = export(m, (x,))
    graph = _bind_from_module(ep, fx_to_ir(ep), m)

    # 高層 op 保留檢查
    ops = {n.op for n in graph.nodes.values()}
    assert {"ARG", "GEMM", "MUL", "SIGMOID", "SLICE", "OUT"} <= ops, ops

    ref = m(x)
    out = _interpret_with(ep, graph, x)
    diff = (out - ref).abs().max().item()
    print(f"  [simple MLP] diff={diff:.6e} {'PASS' if diff < 1e-5 else 'FAIL'}")
    assert diff < 1e-5
    return True


def test_real_vllm_qwen2_mlp():
    """真實 vLLM Qwen2MLP：export → IR → 對拍（A1 驗證高層 op 保留）。"""
    # 環境初始化（坑 4 + VLLM_TARGET_DEVICE=cpu，見 test_mlp_fusion_e2e.py）
    os.environ["VLLM_TARGET_DEVICE"] = "cpu"
    os.environ["VLLM_USE_DML"] = "0"

    import vllm.config as vllm_config_mod
    import vllm.config.vllm as vllm_config_vllm_mod

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
    ensure_model_parallel_initialized(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

    from vllm.model_executor.models.qwen2 import Qwen2MLP
    import torch.nn.functional as F

    H, I = 64, 256
    mlp = Qwen2MLP(hidden_size=H, intermediate_size=I, hidden_act="silu")
    mlp.eval()

    # 模擬 process_weights_after_loading（保留權重，測試用）
    mlp.gate_up_proj.cpu_linear = F.linear
    mlp.down_proj.cpu_linear = F.linear

    # 權重覆蓋為隨機值（vLLM 未載入是 1e37 標記值）
    with torch.no_grad():
        mlp.gate_up_proj.weight.data.normal_(0, 0.02)
        mlp.down_proj.weight.data.normal_(0, 0.02)

    x = torch.randn(2, H)
    from torch.export import export

    ep = export(mlp, (x,))

    # 高層 op 名稱保留檢查（A1 核心驗證）
    targets = {
        getattr(n.target, "__name__", str(n.target))
        for n in ep.graph.nodes if n.op == "call_function"
    }
    print(f"  [vLLM Qwen2MLP] fx call_function targets: {sorted(targets)}")
    assert {"linear.default", "slice.Tensor", "silu.default",
            "mul.Tensor"} <= targets, targets

    graph = _bind_from_module(ep, fx_to_ir(ep), mlp)
    print(f"  [vLLM Qwen2MLP] IR: {graph}")
    assert len(graph.weights) == 2, graph.weights   # gate_up + down

    with torch.no_grad():
        ref = mlp(x)
    out = _interpret_with(ep, graph, x)
    diff = (out - ref).abs().max().item()
    print(f"  [vLLM Qwen2MLP] diff={diff:.6e} {'PASS' if diff < 1e-5 else 'FAIL'}")
    assert diff < 1e-5
    return True


if __name__ == "__main__":
    print("【1】簡單 SwiGLU MLP（速驗 pipeline）")
    ok1 = test_simple_mlp()
    print("【2】真實 vLLM Qwen2MLP（A1 高層 op 保留 + 對拍）")
    ok2 = test_real_vllm_qwen2_mlp()
    print("=" * 50)
    print(f"結果: {sum([ok1, ok2])}/2 通過")
    sys.exit(0 if (ok1 and ok2) else 1)