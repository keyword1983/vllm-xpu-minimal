"""插件自己的 env vars。

真插件（tpu-inference/vllm-rbln）都有這個檔案：
- 集中管理所有 VLLM_XPU_* 開關
- additional_env_vars 要回報給 Platform（Ray 跨 process 傳遞用）
"""
import os


def env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes")


# 假裝的裝置數量（教學用：模擬多卡）
XPU_FAKE_NUM_DEVICES = int(os.getenv("XPU_FAKE_NUM_DEVICES", "1"))
# 假裝的每卡記憶體（bytes）——影響 KV cache 大小
XPU_FAKE_DEVICE_MEMORY_BYTES = int(
    os.getenv("XPU_FAKE_DEVICE_MEMORY_BYTES", str(8 * 1024**3)))
