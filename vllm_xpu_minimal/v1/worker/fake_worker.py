"""XPUFakeWorker：vLLM executor 與「裝置世界」之間的橋。

WorkerBase 的必答題（vLLM 會依序呼叫）：
  init_device() → load_model() → determine_available_memory()
  → initialize_from_config(kv_cache_config) → compile_or_warm_up_model()
  → execute_model() / sample_tokens()（serving 迴圈）

【真插件對照】
- tpu-inference: worker/tpu_worker.py（JAX 裝置隔離 env、假 torch.distributed）
- vllm-rbln:     v1/worker/rbln_worker.py（RBLN_VISIBLE_DEVICES、rbln-ccl）
"""
import logging
from typing import TYPE_CHECKING, Optional

import torch
from vllm.v1.worker.worker_base import WorkerBase

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.core.sched.output import SchedulerOutput

logger = logging.getLogger(__name__)


class XPUFakeWorker(WorkerBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 【真插件】RBLN 在這裡關掉 custom all-reduce（走自家 CCL）
        self.parallel_config.disable_custom_all_reduce = True

    # ---- 生命週期 1：裝置 ----

    def init_device(self) -> None:
        """【真插件】這裡是裝置初始化的重頭戲：
        - TPU：設 TPU_VISIBLE_CHIPS 等 env → jax.devices() →
          假的 torch.distributed init（world_size=1, gloo file://）
        - RBLN：RBLN_VISIBLE_DEVICES → init_worker_distributed_environment
          （真分散式，rbln-ccl backend）
        假插件：直接用 CPU。

        ★ 分散式初始化不可省——模型建構子（如 Qwen2Model）會呼叫
        get_pp_group().is_first_rank，未初始化就炸：
          AssertionError: pipeline model parallel group is not initialized
        抄 GPUWorker.init_device 的 init_worker_distributed_environment
        （vllm/v1/worker/gpu_worker.py:691）。
        """
        self.device = torch.device("cpu")

        # 分散式環境（world_size=1 的 singleton，gloo backend）
        from vllm.distributed import (ensure_model_parallel_initialized,
                                      init_distributed_environment,
                                      set_custom_all_reduce)
        from vllm.distributed.kv_transfer import ensure_kv_transfer_initialized
        import tempfile
        import os
        # 對照 GPUWorker.init_device 的 init_worker_distributed_environment
        # （vllm/v1/worker/gpu_worker.py:691）：set_custom_all_reduce +
        # init_distributed_environment + ensure_model_parallel_initialized +
        # ensure_kv_transfer_initialized
        set_custom_all_reduce(not self.parallel_config.
                              disable_custom_all_reduce)
        # 用 file:// init method 避免需要 TCP port（TPU 同款做法）
        fd, temp_file = tempfile.mkstemp()
        os.close(fd)
        init_distributed_environment(
            world_size=self.parallel_config.world_size,
            rank=self.rank,
            local_rank=self.local_rank,
            distributed_init_method=f"file://{temp_file}",
            backend="gloo",
        )
        ensure_model_parallel_initialized(
            tensor_model_parallel_size=self.parallel_config.
            tensor_parallel_size,
            pipeline_model_parallel_size=self.parallel_config.
            pipeline_parallel_size,
        )
        # ★ ensure_kv_transfer_initialized 不在這裡呼叫——
        #   它需要 kv_cache_config（本版 vLLM 簽名），而 KV cache 要到
        #   initialize_from_config 才存在。TPU 同款：tpu_worker.py 在
        #   initialize_from_config 內才呼叫（tpu_worker.py:725）。
        #   GPUWorker 在 init_device 呼叫是因為它的簽名只要 vllm_config。

        torch.manual_seed(self.model_config.seed or 0)
        logger.info("[XPUFakeWorker] init_device: using CPU (fake device).")

        # 建 ModelRunner（真插件在這裡傳入 devices/rank 等資訊）
        from vllm_xpu_minimal.v1.worker.fake_model_runner import (
            XPUFakeModelRunner)
        self.model_runner = XPUFakeModelRunner(self.vllm_config, self.device)

    # ---- 生命週期 2：模型 ----

    def load_model(self) -> None:
        """【真插件】TPU：get_model()（兩條路徑分岔）→ jax.clear_caches()。
        假插件：直接用 vLLM 原生 loader（CPU 上）。"""
        from vllm.config import set_current_vllm_config
        with set_current_vllm_config(self.vllm_config):
            self.model_runner.load_model()
        logger.info("[XPUFakeWorker] load_model done.")

    # ---- 生命週期 3：記憶體探測（KV cache 協議的第一步）----

    def determine_available_memory(self) -> int:
        """回傳「可用於 KV cache 的 bytes」。vLLM 用它算 num_blocks。

        【真插件】三種做法：
        - TPU：HBM limit × gpu_memory_utilization − used（靜態查詢）
        - RBLN：手動數 runtime 數量 + 量化感知 byte 估算 +
          Dynamic KV Sizer（編譯後量測）
        - GPU：profile_run() 量測峰值
        假插件：總記憶體 × util − 模型大小（粗估）。"""
        from vllm_xpu_minimal.envs import XPU_FAKE_DEVICE_MEMORY_BYTES
        util = self.cache_config.gpu_memory_utilization
        model_bytes = sum(p.numel() * p.element_size()
                          for p in self.model_runner.model.parameters())
        available = int(XPU_FAKE_DEVICE_MEMORY_BYTES * util - model_bytes)
        logger.info(
            "[XPUFakeWorker] determine_available_memory: %.2f GiB "
            "(total=%.2f GiB, model=%.2f GiB)", available / 2**30,
            XPU_FAKE_DEVICE_MEMORY_BYTES / 2**30, model_bytes / 2**30)
        if available <= 0:
            raise ValueError(
                f"Model ({model_bytes / 2**30:.2f} GiB) does not fit in "
                "fake device memory. Reduce model size or increase "
                "XPU_FAKE_DEVICE_MEMORY_BYTES.")
        return available

    # ---- 生命週期 4：KV cache 配置（協議的第二步）----

    def get_kv_cache_spec(self) -> dict:
        """告訴 vLLM 每層要什麼 cache（形狀/大小宣告）。
        【真插件】從 attention 模組的 get_kv_cache_spec() 收集。"""
        return self.model_runner.get_kv_cache_spec()

    def initialize_from_config(self, kv_cache_config) -> None:
        """vLLM 算好 num_blocks 後呼叫——一次性配置 KV cache 實體。
        【真插件】TPU：先 precompile 大 vocab 函數（防 OOM）→ 配 jax.Array。
        【真插件】RBLN：Dynamic KV Sizer 可能在此後 resize。"""
        # KV transfer connector 需要 kv_cache_config，在這裡初始化
        # （TPU 同款：tpu_worker.py:725）
        from vllm.distributed.kv_transfer import ensure_kv_transfer_initialized
        ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)

        self.model_runner.initialize_kv_cache(kv_cache_config)
        logger.info("[XPUFakeWorker] KV cache initialized: %d blocks",
                    kv_cache_config.num_blocks)

    def initialize_cache(self, num_gpu_blocks: int,
                         num_cpu_blocks: int) -> None:
        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

    # ---- 生命週期 5：編譯/warmup ----

    def compile_or_warm_up_model(self):
        """【真插件】TPU：capture_model()（26 個 _precompile_*，bucketing 全組合）。
        【真插件】RBLN：run_model_graphs()（每個 decode bucket × query len 的
        dummy run）+ mega-cache load/save。
        假插件：無編譯，只 warmup 一次 dummy run。

        ★ 必須回傳 CompilationTimes——executor 會讀
        `t.language_model`（vllm/v1/executor/abstract.py），回 None 就炸：
          AttributeError: 'NoneType' object has no attribute 'language_model'
        （TPU 同款：tpu_worker.py:675）
        """
        from vllm.v1.worker.worker_base import CompilationTimes
        self.model_runner.dummy_warmup()
        logger.info("[XPUFakeWorker] warmup done.")
        return CompilationTimes(
            language_model=self.compilation_config.compilation_time,
            encoder=0.0,
        )

    # ---- Serving 迴圈 ----

    def execute_model(self, scheduler_output: "SchedulerOutput",
                      intermediate_tensors=None
                      ) -> Optional["ModelRunnerOutput"]:
        """【真插件】PP 時負責中間 tensor 的 send/recv
        （TPU 用 step_counter 算 uuid）。"""
        return self.model_runner.execute_model(scheduler_output,
                                               intermediate_tensors)

    def sample_tokens(self, grammar_output) -> "ModelRunnerOutput":
        return self.model_runner.sample_tokens(grammar_output)

    def get_model(self):
        return self.model_runner.get_model()

    def check_health(self) -> None:
        """【真插件】探測裝置狀態（RBLN 查 DRAM used）。"""
        return
