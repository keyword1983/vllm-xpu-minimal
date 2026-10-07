"""XPUFakeModelRunner：插件 80% 的程式碼在這一層。

職責：SchedulerOutput（純 CPU dict）→ device 輸入 → 執行 → ModelRunnerOutput。

【真插件對照】
- tpu-inference: runner/tpu_runner.py（3285 行：bucketing、DeviceBuffer、
  persistent batch manager 群、async placeholder、continue_decode）
- vllm-rbln:     v1/worker/rbln_model_runner.py（3793 行：decode bucket、
  InputStager、DP 形狀協調、mega-cache）
- 上游原版:      vllm/v1/worker/gpu_model_runner.py

假插件策略：**繼承 GPUModelRunner 但把 device 換成 CPU**——
這是 vLLM 自己 CPU 後端（CPUModelRunner(GPUModelRunner)）的官方做法，
讓我們專注在「插件骨架」而非「runner 邏輯」。
"""
import contextlib
import logging

import torch
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _torch_cuda_wrapper():
    """★ 抄自 vllm/v1/worker/cpu_model_runner.py::_torch_cuda_wrapper。

    GPUModelRunner.__init__ 有一整排 torch.cuda.* 呼叫：
      torch.cuda.Stream()（comm_stream / async_output_copy_stream）
      torch.cuda.Event()（transfer_event 等）
      torch.cuda.get_device_properties()（經 _init_device_properties）
    在沒有可用 CUDA driver 的機器上，第一個呼叫就炸：
      torch.AcceleratorError: CUDA error: CUDA driver version is insufficient
    vLLM 官方 CPU 後端的解法：__init__ 期間用 placeholder 換掉
    Event/Stream，裝置查詢用 override 蓋掉（見下方兩個方法）。

    【真插件對照】這就是「盤點上游 CUDA 接觸點」的具體作業——
    tpu-inference 用 monkeypatch torch.accelerator.*、
    RBLN 用 USE_DEVICE_TENSOR 條件式分流，本質相同。
    """

    class _EventPlaceholder:

        def __init__(self, *args, **kwargs) -> None:
            self.record = lambda: None
            self.synchronize = lambda: None

    class _StreamPlaceholder:

        def __init__(self, *args, **kwargs) -> None:
            pass

    cuda_event = torch.cuda.Event
    cuda_stream = torch.cuda.Stream
    try:
        torch.cuda.Event = _EventPlaceholder
        torch.cuda.Stream = _StreamPlaceholder
        yield
    finally:
        torch.cuda.Event = cuda_event
        torch.cuda.Stream = cuda_stream


class XPUFakeModelRunner(GPUModelRunner):
    """繼承上游 GPU runner，全部運算落在 CPU。

    【真插件不會這樣做】——真插件會自己寫 runner（因為要控制
    輸入準備、執行方式、取樣位置）。這裡繼承是教學取捨：
    讓最小插件能真的跑通，同時保留「要替換的接縫」註解。
    """

    def __init__(self, vllm_config, device: torch.device):
        # GPUModelRunner 會用 current_platform 判斷裝置；
        # 假平台回 CPU，所以直接可用。
        # ★ __init__ 必須包在 _torch_cuda_wrapper 內——
        #   否則 torch.cuda.Stream()/Event() 在無 CUDA driver 機器上炸。
        with _torch_cuda_wrapper():
            super().__init__(vllm_config, device)
        logger.info(
            "[XPUFakeModelRunner] initialized (inheriting GPUModelRunner, "
            "device=%s). 真插件會自寫 runner：見 tpu_runner.py / "
            "rbln_model_runner.py 的接縫註解。", device)

    # ---- ★ CUDA 接觸點 override（抄自 CPUModelRunner）----

    def _init_device_properties(self) -> None:
        """GPUModelRunner.__init__ 呼叫 torch.cuda.get_device_properties()。
        CPU 上無意義，pass（CPUModelRunner 同款）。"""
        pass

    def _sync_device(self) -> None:
        """GPUModelRunner 在 profile_run/execute 後呼叫 torch.cuda.synchronize()。
        CPU 上無意義，pass（CPUModelRunner 同款）。"""
        pass

    # ---- 接縫 1：KV cache spec（真插件必寫）----
    # get_kv_cache_spec() 繼承自 GPUModelRunner：從 static_forward_context
    # 的 Attention 模組收集 spec。真插件的差異點：
    #   - TPU：head_size 對齊 128、page size 由 SMEM 反推、hybrid 補償
    #   - RBLN：大 block（1k-4k）+ sub-block caching

    # ---- 接縫 2：KV cache 配置（真插件必寫）----
    # initialize_kv_cache() 繼承自 GPUModelRunner：torch.zeros 配置。
    # 真插件的差異點：
    #   - TPU：jax.Array + NamedSharding 分片 + 一次性配置
    #   - RBLN：graph input tensor + kv_cache_bases 共享基底

    # ---- 接縫 3：執行（真插件必寫）----
    # execute_model()/sample_tokens() 繼承自 GPUModelRunner。
    # 真插件的差異點：
    #   - TPU：model_fn 是 jit 過的（donate KV cache）、
    #     _prepare_inputs 用 DeviceBuffer 一次 H2D、
    #     async placeholder substitution
    #   - RBLN：model_executable 是 torch.compile 過的、
    #     InputStager 暫存輸入、單相 batch

    def dummy_warmup(self) -> None:
        """最小 warmup：跑一次 dummy forward（真插件 bucketing 迴圈的縮影）。

        ★ 失敗不擋啟動（教學插件寬容；真插件 fail-fast）。
        已知的 warmup 炸點（都與 CUDA op 有關）：
        - FlexAttention 的 block mask 建構走 torch.compile 過的
          create_block_mask（CPU 上可跑但慢）
        - _dummy_sampler_run 內的 rejection sampler 用 Triton kernel
          （CPU 上炸）——但我們禁用了 spec decode，不會走到
        """
        try:
            self._dummy_run(self.max_num_tokens)
            logger.info("[XPUFakeModelRunner] dummy warmup ok (num_tokens=%d)",
                        self.max_num_tokens)
        except Exception as e:
            logger.warning("[XPUFakeModelRunner] dummy warmup failed: %s", e)
