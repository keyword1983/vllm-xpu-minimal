"""Minimal fake platform plugin.

鐵律（vllm-rbln 踩過的坑）：
1. 這個模組不得 top-level import vllm——vLLM 在半初始化狀態下
   解析 current_platform，re-enter 會讓 getattr 失敗、靜默 fallback 到 CPU。
2. register() 必須便宜且不炸——exception 會被吞掉變成偵測失敗。
3. 本插件會在多個 process 載入多次，所有副作用必須冪等。
"""

import os


def register():
    """Platform plugin 偵測函式。

    vLLM 會呼叫它並期待回傳平台類別的 qualified name，或 None（偵測失敗）。
    真插件在這裡探測硬體（如 import libtpu / rebel.get_npu_name()）；
    假插件用 env var 開關，避免污染正常環境。
    """
    if os.environ.get("VLLM_XPU_FAKE", "0") != "1":
        return None
    return "vllm_xpu_minimal.platforms.fake_platform.XPUFakePlatform"


def register_ops():
    """General plugin：每個 process 啟動時執行。

    真插件在這裡：
    - 註冊模型（vllm.ModelRegistry.register_model）
    - 觸發 @register_oot decorator（import custom_ops 模組）
    - 註冊 dummy torch ops（torch.library.define）
    假插件只印一行證明它被執行了。
    """
    import logging
    logging.getLogger(__name__).info(
        "[XPUFakePlugin] general_plugins hook executed (idempotent).")
