"""Phase 7 Stage 1 測試：裝置常駐張量池 + handle-based GEMM + elementwise mul + sigmoid。

驗證：
1. TxpTensor alloc/upload/readback 正確性
2. handle-based GEMM 數值 vs CPU
3. elementwise mul 數值 vs CPU
4. silu 分解（sigmoid + mul）數值 vs CPU
5. 權重常駐（WeightCache 重複使用零重傳）
6. 剖析統計（Stage 0）
"""
import sys
import time

import torch

sys.path.insert(0, "vllm-xpu-minimal")

from vllm_xpu_minimal.device.txp_backend import (
    TxpTensor,
    WeightCache,
    dml_elementwise_mul_out,
    dml_gemm_out,
    dml_silu_out,
    get_gemm_stats,
    log_gemm_stats,
)

torch.manual_seed(42)


def check(name, a, b, tol=1e-4, rtol=None):
    diff = (a - b).abs().max().item()
    if rtol is not None:
        denom = a.abs().max().item()
        ok = diff < max(tol, rtol * denom)
    else:
        ok = diff < tol
    status = "✓" if ok else "✗"
    print(f"  {status} {name}: max_diff={diff:.6e}")
    return ok


def test_tensor_pool():
    print("【1】TxpTensor alloc/upload/readback")
    t = torch.randn(64, 128)
    xt = TxpTensor.from_cpu(t)
    back = xt.to_cpu()
    ok = check("roundtrip", back, t)
    xt.free()
    return ok


def test_gemm():
    print("【2】handle-based GEMM（裝置常駐，零 CPU 往返）")
    a = torch.randn(8, 256)
    b = torch.randn(256, 512)
    a_t = TxpTensor.from_cpu(a)
    b_t = TxpTensor.from_cpu(b)
    c_t = dml_gemm_out(a_t, b_t)
    c = c_t.to_cpu()
    ref = a @ b
    ok = check("C = A @ B", c, ref, tol=1e-3)
    a_t.free(); b_t.free(); c_t.free()
    return ok


def test_elementwise_mul():
    print("【3】elementwise mul")
    a = torch.randn(16, 256)
    b = torch.randn(16, 256)
    a_t = TxpTensor.from_cpu(a)
    b_t = TxpTensor.from_cpu(b)
    c_t = dml_elementwise_mul_out(a_t, b_t)
    c = c_t.to_cpu()
    ok = check("C = A * B", c, a * b, tol=1e-5)
    a_t.free(); b_t.free(); c_t.free()
    return ok


def test_silu():
    print("【4】silu 分解（sigmoid + mul，全裝置端）")
    a = torch.randn(16, 256)
    a_t = TxpTensor.from_cpu(a)
    c_t = dml_silu_out(a_t)
    c = c_t.to_cpu()
    ref = a * torch.sigmoid(a)
    ok = check("silu(x) = x*sigmoid(x)", c, ref, tol=1e-5)
    a_t.free(); c_t.free()
    return ok


def test_weight_cache():
    print("【5】WeightCache 權重常駐（重複使用零重傳）")
    w = torch.randn(256, 512)
    cache = WeightCache()
    t1 = cache.get(w)
    t2 = cache.get(w)
    ok = (t1.handle == t2.handle)
    print(f"  {'✓' if ok else '✗'} 同一權重重複 get 回傳同一 handle: {t1.handle}")
    cache.clear()
    return ok


def test_mlp_chain():
    print("【6】MLP 閉環鏈（gate/up GEMM + silu + mul + down GEMM 全裝置端）")
    M, H, I = 4, 256, 1024
    x = torch.randn(M, H)
    w_gate = torch.randn(H, I)
    w_up = torch.randn(H, I)
    w_down = torch.randn(I, H)

    x_t = TxpTensor.from_cpu(x)
    gate_t = TxpTensor.from_cpu(w_gate)
    up_t = TxpTensor.from_cpu(w_up)
    down_t = TxpTensor.from_cpu(w_down)

    gate = dml_gemm_out(x_t, gate_t)
    up = dml_gemm_out(x_t, up_t)
    gate = dml_silu_out(gate)
    act = dml_elementwise_mul_out(gate, up)
    out = dml_gemm_out(act, down_t)
    result = out.to_cpu()

    ref = torch.nn.functional.silu(x @ w_gate) * (x @ w_up) @ w_down
    ok = check("MLP 閉環", result, ref, tol=1e-3, rtol=1e-5)

    for t in (x_t, gate_t, up_t, down_t, gate, up, act, out):
        t.free()
    return ok


def test_stats():
    print("【7】剖析統計（Stage 0）")
    s = get_gemm_stats()
    print(f"  count={s['count']} | H2D={s['h2d_ms']:.2f}ms | dispatch={s['dispatch_ms']:.2f}ms | "
          f"D2H={s['d2h_ms']:.2f}ms | fence={s['fence_ms']:.2f}ms")
    log_gemm_stats("test")
    return s["count"] > 0


if __name__ == "__main__":
    results = [
        test_tensor_pool(),
        test_gemm(),
        test_elementwise_mul(),
        test_silu(),
        test_weight_cache(),
        test_mlp_chain(),
        test_stats(),
    ]
    print("=" * 50)
    print(f"結果: {sum(results)}/{len(results)} 通過")
    sys.exit(0 if all(results) else 1)