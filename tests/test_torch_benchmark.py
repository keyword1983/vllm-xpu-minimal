import ctypes
import os
import time
import torch

so_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../vllm_xpu_minimal/device/lib/libdml_gemm.so"))
lib = ctypes.CDLL(so_path)
lib.dml_gemm_init()

M, K, N = 1, 896, 896
a = torch.randn(M, K, dtype=torch.float32)
b = torch.randn(K, N, dtype=torch.float32)
c = torch.empty(M, N, dtype=torch.float32)

# Warmup (trigger JIT compilation)
lib.dml_gemm_forward(ctypes.c_void_p(a.data_ptr()), ctypes.c_void_p(b.data_ptr()), ctypes.c_void_p(c.data_ptr()), M, K, N)

latencies = []
for i in range(10):
    t0 = time.perf_counter()
    lib.dml_gemm_forward(ctypes.c_void_p(a.data_ptr()), ctypes.c_void_p(b.data_ptr()), ctypes.c_void_p(c.data_ptr()), M, K, N)
    t1 = time.perf_counter()
    latencies.append((t1 - t0) * 1000)

avg_ms = sum(latencies) / len(latencies)
print(f"🔥 Intel UHD 630 iGPU 單 Token 解碼延遲 (Decode Latency): 平均 {avg_ms:.2f} ms (Min: {min(latencies):.2f} ms)")
lib.dml_gemm_shutdown()
