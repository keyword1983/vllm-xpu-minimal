"""XPUFakeModelRunner：插件的核心執行層（主廚）。

職責：SchedulerOutput（純 CPU dict）→ device 輸入 → 執行 → ModelRunnerOutput。

【繼承架構】
繼承 vLLM 原生 CPUModelRunner（別名 CPURunner）：
- 內建 _torch_cuda_wrapper（Event/Stream placeholder）
- 已處理 _init_device_properties / _sync_device override
- 已整合 _postprocess_tensors（將 CpuGpuBuffer 導向 CPU）
- 原生適配 CPU attention backend

【真插件對照】
- 真 XPU 插件在此負責：
  1. 輸入準備（DeviceBuffer / H2D 串流）
  2. 算子執行（呼叫 XPU Runtime 或 torch.compile 捕捉的靜態圖）
  3. 取樣（在裝置端執行或搬回 Host 取樣）
- 本 demo 繼承 CPUModelRunner，以 CPU 模擬硬體執行，展示外掛接入點。
"""
import logging
from typing import TYPE_CHECKING, Optional

import torch
from vllm.v1.worker.cpu_model_runner import CPUModelRunner

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import ModelRunnerOutput

logger = logging.getLogger(__name__)


class XPUFakeModelRunner(CPUModelRunner):
    """繼承 vLLM 官方 CPUModelRunner，模擬 Fake XPU 運算執行。"""

    def __init__(self, vllm_config: "VllmConfig", device: torch.device):
        super().__init__(vllm_config, device)
        logger.info(
            "[XPUFakeModelRunner] Initialized Fake XPU ModelRunner "
            "(inheriting CPUModelRunner/CPURunner, device=%s).",
            device,
        )

    # ---- 模型載入 ----

    def load_model(self, load_dummy_weights: bool = False) -> None:
        logger.info(
            "[XPUFakeModelRunner] [FakeXPU Hardware] Loading model weights into simulated XPU memory..."
        )
        super().load_model(load_dummy_weights=load_dummy_weights)
        logger.info(
            "[XPUFakeModelRunner] [FakeXPU Hardware] Weights successfully loaded."
        )

    # ---- 暖機與預編譯 (AOT / JIT Warmup) ----

    def warming_up_model(self) -> None:
        """暖機：跑 dummy forward（模擬 XPU AOT precompile / warm-up）。"""
        logger.info(
            "[XPUFakeModelRunner] [FakeXPU Hardware] Executing warmup pass on simulated XPU pipelines..."
        )
        super().warming_up_model()
        logger.info(
            "[XPUFakeModelRunner] [FakeXPU Hardware] Warmup pass completed."
        )

    def dummy_warmup(self) -> None:
        """向下相容的 warmup 入口。"""
        self.warming_up_model()

    # ---- 執行模型 (Inference Execution) ----

    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors=None,
    ) -> Optional["ModelRunnerOutput"]:
        """模型推理執行進入點。

        【真插件在此做】：
        1. Host-to-Device 串流張量拷貝
        2. 觸發 XPU 靜態計算圖（Graph Execution）或 Dispatch Kernel
        3. 收集輸出張量
        """
        return super().execute_model(scheduler_output, intermediate_tensors)


# 別名導出（支持 CPURunner / XPUFakeCPURunner 命名）
CPURunner = CPUModelRunner
XPUFakeCPURunner = XPUFakeModelRunner
