"""XPUFakePlatform：插件的「硬體身分證」+「config 主控台」。

每個方法都標注了【真插件會做什麼】，對照閱讀：
- tpu-inference: tpu_inference/platforms/tpu_platform.py
- vllm-rbln:     vllm_rbln/platform/__init__.py + vllm_impl.py
"""
import logging
from typing import TYPE_CHECKING, Optional

import torch
from vllm.platforms import Platform, PlatformEnum

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = logging.getLogger(__name__)


class XPUFakePlatform(Platform):
    _enum = PlatformEnum.OOT          # OOT 插件固定用這個（is_out_of_tree()=True）

    # ---- 身分證欄位 ----
    # ★★ OOT 平台的 device_name 與 device_type 都會被拿去建 torch.device！
    #   兩個炸點（同一類問題，兩個位置）：
    #   1. DeviceConfig.__post_init__（vllm/config/device.py）：
    #      torch.device(device_type)——只有 "tpu" 在硬編碼白名單
    #   2. GroupCoordinator.__init__（vllm/distributed/parallel_state.py:245）：
    #      is_out_of_tree() 時 → torch.device(f"{device_name}:{local_rank}")
    #      （TPU 是 in-tree，走 else 分支的 "cpu"，所以它的 device_name="tpu" 沒事）
    #   【真插件】RBLN 兩個都條件式："rbln" if USE_DEVICE_TENSOR else "cpu"
    #   ——SDK 註冊了 torch 裝置才用自訂字串，否則一律 "cpu"。
    #   顯示用的名稱放 get_device_name()（classmethod），別放這兩個屬性。
    device_name: str = "cpu"
    device_type: str = "cpu"
    dispatch_key: str = "CPU"         # 【真插件】"PrivateUse1"（自訂裝置）或 "XLA"（TPU）
    ray_device_key: str = "XPU_FAKE"  # 【真插件】Ray 資源標記
    dist_backend: str = "gloo"        # 【真插件】"nccl"/"ccl"/"rbln-ccl"/TPU 不用
    device_control_env_var: str = "XPU_FAKE_VISIBLE_DEVICES"
    simple_compile_backend: str = "eager"  # 【真插件】TPU 用 "eager"（繞過 torch.compile）

    supported_quantization: list[str] = []  # 【真插件】量化白名單

    # ---- 裝置查詢 ----

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        """【真插件】查硬體型號：TPU 用 tpu_info、RBLN 用 rebel.get_npu_name()。
        失敗要有 fallback（RBLN 用 RBLN_FORCE_NPU_NAME env）。
        ★ 顯示用的自訂名稱放這裡（classmethod），不要放 device_name 屬性——
        屬性會被 GroupCoordinator 拿去建 torch.device（見上方註解）。"""
        return "TranXform-XPU-Fake-Device (CPU-Emulated)"

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        """【真插件】單卡總記憶體。注意：KV cache 大小實際由
        Worker.determine_available_memory() 決定，這個只是部分呼叫點用。"""
        from vllm_xpu_minimal.envs import XPU_FAKE_DEVICE_MEMORY_BYTES
        return XPU_FAKE_DEVICE_MEMORY_BYTES

    @classmethod
    def mem_get_info(cls) -> tuple[int, int]:
        """(free, total) bytes。【真插件】TPU 聚合 jax.local_devices() 的
        memory_stats()；RBLN 用 torch.rbln.mem_get_info_per_chiplet()。"""
        from vllm_xpu_minimal.envs import XPU_FAKE_DEVICE_MEMORY_BYTES
        return XPU_FAKE_DEVICE_MEMORY_BYTES // 2, XPU_FAKE_DEVICE_MEMORY_BYTES

    @classmethod
    def get_current_memory_usage(cls, device=None) -> float:
        """★ GPUModelRunner.load_model() 的 DeviceMemoryProfiler 會呼叫
        （vllm/utils/mem_utils.py → current_platform.get_current_memory_usage）。
        Platform 基底類 raise NotImplementedError——不實作就炸在 load_model。
        【真插件】CUDA：empty_cache + reset_peak + max_memory_allocated；
        XPU：torch.xpu.max_memory_allocated；RBLN：torch.rbln 記憶體統計。
        假插件（CPU）：回 0——CPU 上沒有「裝置記憶體峰值」概念，
        profiler 的數字只是日誌用途。"""
        return 0.0

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        """【真插件】TPU/RBLN 都是 no-op（runtime 自己管裝置）。"""
        logger.info("[XPUFakePlatform] set_device is a no-op (like TPU/RBLN).")

    @classmethod
    def current_device(cls) -> torch.device:
        """【真插件】TPU 回 torch.device("cpu") 佔位（tensor 手動放上裝置）。"""
        return torch.device("cpu")

    @classmethod
    def inference_mode(cls):
        """【真插件】RBLN 回 torch.no_grad()；TPU 回 True。"""
        return torch.no_grad()

    @classmethod
    def manual_seed_all(cls, seed: int) -> None:
        """【真插件】TPU 用 random/numpy seed（JAX key 另管）。"""
        torch.manual_seed(seed)

    # ---- 能力宣告 ----

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        """【真插件】TPU/RBLN 都回 False 並 warning——如實回報，
        否則 vLLM 會用 pinned memory 然後炸。"""
        logger.warning("[XPUFakePlatform] Pin memory is not supported.")
        return False

    @classmethod
    def supports_v1(cls, model_config) -> bool:
        return True

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        """【真插件】TPU 回 False。決定 vLLM 能不能用 async scheduling。"""
        return False

    @classmethod
    def can_update_inplace(cls) -> bool:
        """【真插件】TPU/RBLN 都回 False（權重 in-place 更新不支援）。"""
        return False

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        """【真插件】TPU 回 True（mamba hybrid）。"""
        return False

    @classmethod
    def use_all_gather(cls) -> bool:
        """【真插件】TPU 回 True（TP 收集策略）。"""
        return False

    @classmethod
    def get_punica_wrapper(cls) -> str:
        """【真插件】LoRA kernel 的 qualified name：
        TPU 回 tpu_inference.lora.torch_punica_tpu.PunicaWrapperTPU"""
        raise NotImplementedError("LoRA is not supported on XPU-Fake.")

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        """【真插件】分散式原語的實作類別：
        TPU 回 vllm...tpu_communicator.TpuCommunicator
        RBLN 回 vllm_rbln...RblnCommunicator"""
        return ("vllm.distributed.device_communicators.cpu_communicator."
                "CpuCommunicator")

    # ---- Attention backend ----

    @classmethod
    def get_attn_backend_cls(cls, selected_backend, attn_selector_config,
                             **kwargs) -> str:
        """【真插件】強制選自己的 backend：
        TPU 強制 FLASH_ATTN（MLA 模型 → FLASH_ATTN_MLA）；
        RBLN 只允許 FLASH_ATTN/FLASH_ATTN_MLA，其他 raise。"""
        logger.info("[XPUFakePlatform] Forcing FAKE attention backend.")
        return ("vllm_xpu_minimal.v1.attention.fake_attention."
                "XPUFakeAttentionBackend")

    # ---- ★ 主控台：config 改寫（最重要的 hook）----

    @classmethod
    def check_and_update_config(cls, vllm_config: "VllmConfig") -> None:
        """config 建好後、任何東西初始化前呼叫。config 是 pass by reference。

        真插件在這裡做（照 tpu-inference 的順序）：
        1. 驗證不支援的組合 → raise（fail-fast）
        2. 改寫 dtype / block_size / 關閉不支援功能
        3. 強制 worker_cls 與 executor backend
        4. 初始化平台自己的 config 物件（如 ShardingConfigManager）
        """
        pc = vllm_config.parallel_config
        cc = vllm_config.cache_config
        mc = vllm_config.model_config

        # 1. fail-fast 驗證（抄 RBLN 的 _validate 風格）
        if vllm_config.lora_config is not None:
            raise ValueError("LoRA is not supported on XPU-Fake.")
        if vllm_config.speculative_config is not None:
            raise ValueError(
                "Speculative decoding is not supported on XPU-Fake.")

        # 2. dtype 改寫（抄 RBLN 的 _override）
        if mc.dtype not in (torch.float32, torch.bfloat16):
            logger.info("[XPUFakePlatform] Overriding dtype to bfloat16.")
            mc.dtype = torch.bfloat16

        # 3. block_size（抄 TPU：由 attention backend 的硬體限制反推）
        if cc and not cc.user_specified_block_size:
            cc.block_size = 16
            logger.info("[XPUFakePlatform] Default block_size=16.")

        # 4. ★ 強制 worker_cls——vLLM 就是用這個字串建 Worker 的
        if pc.worker_cls == "auto":
            pc.worker_cls = ("vllm_xpu_minimal.v1.worker.fake_worker."
                             "XPUFakeWorker")
            logger.info("[XPUFakePlatform] [Hardware Demo] Assigned worker_cls -> XPUFakeWorker (inheriting CPUWorker)")

        # 5. executor backend（抄 TPU：單機強制 uni）
        if pc.distributed_executor_backend in (None, "auto"):
            pc.distributed_executor_backend = "uni"
            logger.info("[XPUFakePlatform] executor -> UniProcExecutor")

        # 6. 關閉不支援的功能（抄 RBLN：cascade attention）
        if mc and not mc.disable_cascade_attn:
            logger.warning("[XPUFakePlatform] Disabling cascade attention.")
            mc.disable_cascade_attn = True

        # 7. ★ 關閉 torch.compile 與 CUDA graph（抄 RBLN 的 _override）。
        #    GPUModelRunner 的 cudagraph 捕獲路徑會呼叫 torch.cuda.graph，
        #    在無 CUDA driver 的機器上炸。mode=NONE → 純 eager。
        #    【真插件】RBLN 同款：compilation_config.mode = CompilationMode.NONE
        #    （因為 @support_torch_compile 不支援）；TPU 則用
        #    simple_compile_backend="eager" 達成同樣效果。
        try:
            from vllm.config import CompilationMode, CUDAGraphMode
            if vllm_config.compilation_config.mode != CompilationMode.NONE:
                logger.info(
                    "[XPUFakePlatform] Overriding compilation mode -> NONE "
                    "(eager only, like RBLN).")
                vllm_config.compilation_config.mode = CompilationMode.NONE
            vllm_config.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
        except (ImportError, AttributeError) as e:
            logger.warning("[XPUFakePlatform] Could not force eager mode "
                           "(vLLM version drift?): %s", e)

    @classmethod
    def pre_register_and_update(cls, parser=None) -> None:
        """CLI 解析前（parser 非 None）與 config 建立前（parser None）各呼叫一次。
        【真插件】RBLN 在這裡加 --rbln-* CLI 參數、改 gpu_memory_utilization 預設值。"""
        if parser is not None:
            # CLI 階段：可以加自訂參數
            pass

    @classmethod
    def validate_request(cls, processed_inputs, params) -> None:
        """【真插件】TPU 在這裡拒絕 per-request seed（JAX 不支援）。"""
        pass
