import ctypes
import os
import time
import torch

# 1. 載入編譯好的 DirectML 原生算子庫
so_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../vllm_xpu_minimal/device/lib/libdml_gemm.so"))
lib = ctypes.CDLL(so_path)

lib.dml_gemm_init.restype = ctypes.c_int
lib.dml_gemm_forward.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32
]
lib.dml_gemm_forward.restype = ctypes.c_int
lib.dml_gemm_shutdown.restype = None

# 2. 測試初始化
ret = lib.dml_gemm_init()
assert ret == 0, f"dml_gemm_init failed: {ret}"
print("✓ DirectML 原生硬體加速引擎初始化成功！")

# 3. 測試不同矩陣維度
test_shapes = [
    (1, 896, 896),      # LLM 解碼 (Decode) 典型單 token 尺寸 (Qwen2.5-0.5B Hidden)
    (1, 896, 4864),     # LLM 解碼 MLP Gate/Up 投影
    (64, 896, 896),     # 短 Prefill (64 tokens)
    (512, 512, 512)     # 標準 Benchmark 尺寸
]

for M, K, N in test_shapes:
    # 建立 PyTorch 張量 (隨機數據)
    a = torch.randn(M, K, dtype=torch.float32)
    b = torch.randn(K, N, dtype=torch.float32)
    c_dml = torch.empty(M, N, dtype=torch.float32)

    # 執行 DirectML GEMM 算子
    t0 = time.perf_counter()
    ret = lib.dml_gemm_forward(
        ctypes.c_void_p(a.data_ptr()),
        ctypes.c_void_p(b.data_ptr()),
        ctypes.c_void_p(c_dml.data_ptr()),
        M, K, N
    )
    t1 = time.perf_counter()
    assert ret == 0, f"dml_gemm_forward failed for ({M},{K},{N})"

    # CPU 參考真值
    c_ref = torch.matmul(a, b)

    # 精度與耗時比對
    max_err = torch.max(torch.abs(c_dml - c_ref)).item()
    print(f"✓ GEMM ({M:3d}x{K:4d}x{N:4d}) -> 耗時: {(t1 - t0)*1000:6.2f} ms | 最大絕對誤差: {max_err:.2e}")
    assert max_err < 1e-3, f"Accuracy validation failed: {max_err}"

print("==========================================================")
print("🎉 PyTorch 與 DirectML C-API 橋接全面驗證通過！")
print("==========================================================")
lib.dml_gemm_shutdown()
