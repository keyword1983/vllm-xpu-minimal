"""XPUFakeWorker：vLLM executor 與「Fake XPU 裝置世界」之間的橋樑。

繼承 vLLM 原生 CPUWorker（以 CPU 模擬 Fake XPU 硬體後端接入）：
- 繼承體系：WorkerBase -> Worker (gpu_worker) -> CPUWorker -> XPUFakeWorker
- 職責：模擬 XPU 硬體接入、裝置初始化、記憶體探測、模型載入、Warmup 與 Serving 迴圈。

【硬體接入 Demo 重點】
1. 模擬硬體探測（Probing PCIe slot / NUMA / 假晶片架構）
2. 模擬硬體記憶體（LPDDR / HBM 統一定址空間配額）
3. 繼承 CPUWorker 的分散式環境與執行管線，模型運算安全落在 CPU 上
"""
import logging
from typing import TYPE_CHECKING, Optional

import torch
from vllm.v1.worker.cpu_worker import CPUWorker

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.worker.worker_base import CompilationTimes

logger = logging.getLogger(__name__)


class XPUFakeWorker(CPUWorker):
    """Fake XPU Worker 實作（繼承 CPUWorker，以 CPU 模擬硬體接入）。"""

    def __init__(self, *args, **kwargs):
        # 嘗試以 CPUWorker 初始化；若處於特殊環境（如無 NUMA C++ 擴展），提供平滑容錯
        try:
            super().__init__(*args, **kwargs)
        except Exception as e:
            logger.warning(
                "[XPUFakeWorker] CPUWorker.__init__ encountered an issue (%s), "
                "falling back to base Worker initialization for fake XPU demo.",
                e,
            )
            from vllm.v1.worker.gpu_worker import Worker
            Worker.__init__(self, *args, **kwargs)

        # 【真插件】RBLN / TPU 在此關閉 custom all-reduce（走自家 CCL 或 Gloo）
        self.parallel_config.disable_custom_all_reduce = True
        logger.info(
            "[XPUFakeWorker] Worker initialized (inheriting CPUWorker). "
            "Simulating Fake XPU accelerator attached to Host."
        )

    # ---- 生命週期 1：裝置初始化 (Hardware Attach Demo) ----

    def init_device(self) -> None:
        """模擬 Fake XPU 硬體接入與裝置初始化。

        在真實插件中：
        - TPU：設定 TPU_VISIBLE_CHIPS -> jax.devices() -> 初始化分散式環境
        - RBLN：設定 RBLN_VISIBLE_DEVICES -> 初始化 rbln-ccl
        - TranXform XPU：探測 PCIe 裝置 -> 初始化 XPU Runtime/Driver -> 綁定晶片

        Demo 模式：
        - 走 CPUWorker.init_device() 建立分散式環境 (gloo backend)
        - 替換 self.model_runner 為繼承 CPUModelRunner 的 XPUFakeModelRunner
        """
        logger.info(
            "=================================================================="
        )
        logger.info(
            "[XPUFakeWorker] [Hardware Demo] Connecting to Fake XPU Accelerator..."
        )
        logger.info(
            "[XPUFakeWorker] [Hardware Demo] Device: TranXform XPU-v1 (Simulated on CPU)"
        )
        logger.info(
            "[XPUFakeWorker] [Hardware Demo] Bus Info: PCIe Gen5 x16 @ NUMA Node 0"
        )
        logger.info(
            "[XPUFakeWorker] [Hardware Demo] Driver Version: fake-xpu-drv-v1.0.0 (active)"
        )
        logger.info(
            "=================================================================="
        )

        # 呼叫 CPUWorker 的初始化（負責分散式環境建立、執行緒設定、種子設定等）
        super().init_device()

        # 確保裝置為 cpu（符合目前模擬層）
        self.device = torch.device("cpu")

        # ★ 關鍵：將 model_runner 替換為我們自訂的 XPUFakeModelRunner
        from vllm_xpu_minimal.v1.worker.fake_model_runner import (
            XPUFakeModelRunner,
        )

        self.model_runner = XPUFakeModelRunner(self.vllm_config, self.device)

        logger.info(
            "[XPUFakeWorker] Fake XPU hardware attached successfully. "
            "ModelRunner instantiated: %s",
            type(self.model_runner).__name__,
        )

    # ---- 生命週期 2：模型載入 ----

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        """載入模型至 Fake XPU 記憶體空間。"""
        logger.info(
            "[XPUFakeWorker] [Hardware Demo] Loading model weights into simulated XPU memory..."
        )
        super().load_model(load_dummy_weights=load_dummy_weights)
        logger.info(
            "[XPUFakeWorker] [Hardware Demo] Model loaded successfully."
        )

    # ---- 生命週期 3：記憶體探測 (KV Cache 記憶體協議) ----

    def determine_available_memory(self) -> int:
        """回傳可用於 KV cache 的 bytes。

        【真插件對照】
        - 模擬硬體晶片上的獨立記憶體空間（如 LPDDR / HBM），受 XPU_FAKE_DEVICE_MEMORY_BYTES 控制
        - 若未特別設定或模型過大，回退至 CPUWorker 依主機記憶體計算的邏輯
        """
        from vllm_xpu_minimal.envs import XPU_FAKE_DEVICE_MEMORY_BYTES

        logger.info(
            "[XPUFakeWorker] [Hardware Demo] Querying Fake XPU onboard memory..."
        )
        try:
            # 先執行暖機 (Warmup)
            if hasattr(self.model_runner, "warming_up_model"):
                self.model_runner.warming_up_model()
            elif hasattr(self.model_runner, "dummy_warmup"):
                self.model_runner.dummy_warmup()

            util = self.cache_config.gpu_memory_utilization
            model_bytes = sum(
                p.numel() * p.element_size()
                for p in self.model_runner.model.parameters()
            )
            simulated_total = XPU_FAKE_DEVICE_MEMORY_BYTES
            available = int(simulated_total * util - model_bytes)

            logger.info(
                "[XPUFakeWorker] Fake XPU Onboard Memory: Total=%.2f GiB, "
                "Model=%.2f GiB, Available for KV=%.2f GiB (util=%.2f)",
                simulated_total / 2**30,
                model_bytes / 2**30,
                available / 2**30,
                util,
            )

            if available <= 0:
                logger.warning(
                    "[XPUFakeWorker] Model size exceeds simulated fake memory (%.2f GiB). "
                    "Falling back to host CPU memory calculation.",
                    simulated_total / 2**30,
                )
                return super().determine_available_memory()

            return available
        except Exception as e:
            logger.info(
                "[XPUFakeWorker] Custom memory calculation fallback: %s. "
                "Using CPUWorker default.",
                e,
            )
            return super().determine_available_memory()

    # ---- 生命週期 4：KV Cache 配置 ----

    def initialize_from_config(self, kv_cache_config: "KVCacheConfig") -> None:
        """KV Cache 配置通知與實體配置。"""
        logger.info(
            "[XPUFakeWorker] [Hardware Demo] Initializing %d KV cache blocks on Fake XPU...",
            kv_cache_config.num_blocks,
        )
        super().initialize_from_config(kv_cache_config)
        logger.info(
            "[XPUFakeWorker] [Hardware Demo] KV cache allocation complete."
        )

    # ---- 生命週期 5：編譯與暖機 ----

    def compile_or_warm_up_model(self) -> "CompilationTimes":
        """編譯與暖機模型（回傳 CompilationTimes）。"""
        logger.info(
            "[XPUFakeWorker] [Hardware Demo] Warming up Fake XPU execution graph..."
        )
        times = super().compile_or_warm_up_model()
        logger.info(
            "[XPUFakeWorker] [Hardware Demo] Warmup finished successfully."
        )
        return times

    # ---- 裝置健康檢查 ----

    def check_health(self) -> None:
        """探測 Fake XPU 裝置狀態（模擬硬體心跳檢查）。"""
        # 真插件在此查詢硬體 status / PCIe link / DRAM ECC / 溫度等
        return
