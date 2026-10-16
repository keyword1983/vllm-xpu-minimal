"""MLP 閉環融合：Gate/Up GEMM -> SiluAndMul -> Down GEMM 全程在 iGPU 顯存池流轉。

Phase 7 Stage 1（Task 7.1.1）：
- 每層 MLP 由 2 次 CPU-iGPU 來回（gate_up 進出 + down 進出）降為 1 次（x 進、out 出）
- 權重（gate/up/down）首次上傳後常駐 WeightCache，推論期零重傳
- 防禦式：結構不符或任何例外 → fallback 原 forward（正確性優先）

【掛載點】fake_model_runner.py 的 _patch_linear_for_directml 會在
dispatch_cpu_unquantized_gemm 時 stash 權重（_dml_gate_fp32/_dml_up_fp32/_dml_w_fp32），
本模組的 patch_qwen2_mlp() 在 forward 時取用。
"""
import logging

import torch

from vllm_xpu_minimal.device.txp_backend import (
    TxpTensor,
    WeightCache,
    dml_gemm_out,
    dml_elementwise_mul_out,
    dml_silu_out,
)

logger = logging.getLogger(__name__)

# 全域權重快取（跨層共用；key = data_ptr）
_weight_cache = WeightCache()

# 統計計數器（Stage 0 驗證用）
g_mlp_fused_calls = 0
g_mlp_fallback_calls = 0


def patch_qwen2_mlp() -> bool:
    """Patch Qwen2MLP.forward 為 DirectML 閉環融合路徑。

    回傳 True 表示 patch 成功（或已 patch 過）。
    """
    try:
        from vllm.model_executor.models.qwen2 import Qwen2MLP
    except ImportError:
        logger.warning("[MLP Fusion] 無法 import Qwen2MLP（vLLM 版本漂移？），跳過融合 patch。")
        return False

    if getattr(Qwen2MLP, "_dml_mlp_fused", False):
        return True

    orig_forward = Qwen2MLP.forward

    def dml_mlp_forward(self, x):
        global g_mlp_fused_calls, g_mlp_fallback_calls

        gate_w = getattr(self.gate_up_proj, "_dml_gate_fp32", None)
        up_w = getattr(self.gate_up_proj, "_dml_up_fp32", None)
        down_w = getattr(self.down_proj, "_dml_w_fp32", None)
        if gate_w is None or up_w is None or down_w is None:
            g_mlp_fallback_calls += 1
            return orig_forward(self, x)

        try:
            # 1 H2D：輸入進裝置
            x_t = TxpTensor.from_cpu(x.to(torch.float32))

            # 權重常駐（首次上傳後零重傳）
            gate_t = _weight_cache.get(gate_w)   # [H, I]
            up_t = _weight_cache.get(up_w)       # [H, I]
            down_t = _weight_cache.get(down_w)   # [I, H]

            # 全裝置端閉環：silu(x@Wg) * (x@Wu) @ Wd
            gate = dml_gemm_out(x_t, gate_t)          # [M, I]
            up = dml_gemm_out(x_t, up_t)              # [M, I]
            gate = dml_silu_out(gate)                 # silu(gate)，裝置端
            act = dml_elementwise_mul_out(gate, up)   # [M, I]
            out = dml_gemm_out(act, down_t)           # [M, H]

            # 1 D2H：輸出回 CPU
            result = out.to_cpu().to(x.dtype)
            g_mlp_fused_calls += 1
            if g_mlp_fused_calls <= 3 or g_mlp_fused_calls % 50 == 0:
                logger.info(
                    "[MLP Fusion] 第 %d 次閉環執行成功 (x=%s, I=%d, 裝置端 4 算子零往返)",
                    g_mlp_fused_calls, tuple(x.shape), gate_w.shape[-1],
                )
            return result
        except Exception as e:
            g_mlp_fallback_calls += 1
            logger.warning("[MLP Fusion] 執行失敗，fallback 原 forward: %s", e)
            return orig_forward(self, x)

    Qwen2MLP.forward = dml_mlp_forward
    Qwen2MLP._dml_mlp_fused = True
    logger.info("✓ [MLP Fusion] Qwen2MLP.forward 已 patch 為 DirectML 閉環融合路徑")
    return True


def log_mlp_stats() -> None:
    """印出 MLP 融合統計（Stage 0 驗證用）。"""
    logger.info(
        "[MLP Fusion Stats] fused_calls=%d | fallback_calls=%d",
        g_mlp_fused_calls, g_mlp_fallback_calls,
    )