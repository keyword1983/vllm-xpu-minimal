"""XPUFakeModelRunner：插件 80% 的程式碼在這一層。

職責：SchedulerOutput（純 CPU dict）→ device 輸入 → 執行 → ModelRunnerOutput。

【真插件對照】
- tpu-inference: runner/tpu_runner.py（3285 行：bucketing、DeviceBuffer、
  persistent batch manager 群、async placeholder、continue_decode）
- vllm-rbln:     v1/worker/rbln_model_runner.py（3793 行：decode bucket、
  InputStager、DP 形狀協調、mega-cache）
- 上游原版:      vllm/v1/worker/gpu_model_runner.py

★ 坑 9 的教訓：先查上游有沒有現成的同類後端！
原本繼承 GPUModelRunner + 自補 CUDA workaround（坑 2/6/8 的三個補丁），
後來發現 vLLM 官方 CPU 後端的 CPUModelRunner 已經解完全部問題：
  - _torch_cuda_wrapper（Event/Stream placeholder）內建在 __init__
  - _init_device_properties / _sync_device 已 override 成 pass
  - _postprocess_tensors 已把 CpuGpuBuffer 的 device 側換成 cpu
  - warming_up_model 已用 _set_global_compilation_settings 包好
改繼承 CPUModelRunner 後，本檔只剩「教學用接縫註解」與 thin wrapper。

【真插件不會這樣做】——真插件（TPU/RBLN）都是從零自寫 runner
（只 mixin KVConnectorModelRunnerMixin 等功能模組），
繼承 CPUModelRunner 是教學取捨。
"""
import logging

import torch
from vllm.v1.worker.cpu_model_runner import CPUModelRunner

logger = logging.getLogger(__name__)


class XPUFakeModelRunner(CPUModelRunner):
    """繼承 vLLM 官方 CPU runner（它自己繼承 GPUModelRunner 並處理好
    所有 CUDA 接觸點），全部運算落在 CPU。

    【真插件不會這樣做】——真插件會自己寫 runner（因為要控制
    輸入準備、執行方式、取樣位置）。這裡繼承是教學取捨：
    讓最小插件能真的跑通，同時保留「要替換的接縫」註解。
    """

    def __init__(self, vllm_config, device: torch.device):
        # CPUModelRunner 會 assert device == cpu——假平台正合適。
        super().__init__(vllm_config, device)
        logger.info(
            "[XPUFakeModelRunner] initialized (inheriting CPUModelRunner, "
            "device=%s). 真插件會自寫 runner：見 tpu_runner.py / "
            "rbln_model_runner.py 的接縫註解。", device)

    # ---- 接縫 1：KV cache spec（真插件必寫）----
    # get_kv_cache_spec() 繼承自 GPUModelRunner（CPUModelRunner 未覆蓋）：
    # 從 static_forward_context 的 Attention 模組收集 spec。
    # 真插件的差異點：
    #   - TPU：head_size 對齊 128、page size 由 SMEM 反推、hybrid 補償
    #   - RBLN：大 block（1k-4k）+ sub-block caching

    # ---- 接縫 2：KV cache 配置（真插件必寫）----
    # initialize_kv_cache() 繼承自 GPUModelRunner（CPUModelRunner 未覆蓋）。
    # 真插件的差異點：
    #   - TPU：jax.Array + NamedSharding 分片 + 一次性配置
    #   - RBLN：graph input tensor + kv_cache_bases 共享基底

    # ---- 接縫 3：執行（真插件必寫）----
    # execute_model()/sample_tokens() 繼承自 GPUModelRunner
    # （CPUModelRunner 未覆蓋，device=cpu 所以 tensor 操作可沿用）。
    # 真插件的差異點：
    #   - TPU：model_fn 是 jit 過的（donate KV cache）、
    #     _prepare_inputs 用 DeviceBuffer 一次 H2D、
    #     async placeholder substitution
    #   - RBLN：model_executable 是 torch.compile 過的、
    #     InputStager 暫存輸入、單相 batch

    def dummy_warmup(self) -> None:
        """最小 warmup：跑一次 dummy forward（真插件 bucketing 迴圈的縮影）。

        CPUModelRunner 有自己的 warming_up_model()（含 inductor 設定），
        但它走 torch.compile 路徑；教學插件用 GPUModelRunner 的
        _dummy_run 即可。失敗不擋啟動（教學插件寬容；真插件 fail-fast）。
        """
        try:
            self._dummy_run(self.max_num_tokens)
            logger.info("[XPUFakeModelRunner] dummy warmup ok (num_tokens=%d)",
                        self.max_num_tokens)
        except Exception as e:
            logger.warning("[XPUFakeModelRunner] dummy warmup failed: %s", e)
