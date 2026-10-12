# vllm-xpu-minimal：最小假平台插件（教學用）

> **官方依據**：本插件的結構遵循 vLLM 官方文件
> [Plugin System](https://docs.vllm.ai/en/latest/design/plugin_system/) 的
> "Platform plugins guidelines"——官方的 dummy platform 範例與本插件
> 幾乎一一對應（見 §11 官方文件對照表）。


> 目的：讓新成員用 **最少的程式碼** 理解 vLLM 硬體插件的完整骨架。
> 這個插件「偵測恆真、執行全落在 CPU」——但走完 vLLM 的**全部平台協議**，
> 讓 `vllm serve` 認出它、印出它的日誌、跑通一個小模型。
>
> 安裝後執行：
> ```bash
> pip install -e .
> VLLM_XPU_FAKE=1 vllm serve Qwen/Qwen2.5-0.5B-Instruct --max-model-len 512
> # 啟動日誌應出現 "Platform plugin xpu_fake is activated"
> # 與 "[XPUFakePlatform] ..." 開頭的訊息
> ```
>
> 檔案結構：
> ```
> vllm-xpu-minimal/
> ├── pyproject.toml                  # entry points 註冊
> └── vllm_xpu_minimal/
>     ├── __init__.py                 # register()（偵測函式，不得 import vllm）
>     ├── envs.py                     # 插件自己的 env vars
>     ├── platforms/
>     │   ├── __init__.py
>     │   └── fake_platform.py        # XPUFakePlatform（身分證 + 主控台）
>     └── v1/
>         ├── __init__.py
>         ├── worker/
>         │   ├── __init__.py
>         │   ├── fake_worker.py      # XPUFakeWorker（管家，繼承 CPUWorker 模擬硬體接入）
>         │   └── fake_model_runner.py# XPUFakeModelRunner（主廚，繼承 CPUModelRunner / CPURunner）
>         └── attention/
>             ├── __init__.py
>             └── fake_attention.py   # 假 attention backend（爐台）
> ```
>
> 對照真插件時的「成長路徑」見文末。

---

## 1. `pyproject.toml` — 兩個 entry point

```toml
[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.build_meta"

[project]
name = "vllm-xpu-minimal"
version = "0.0.1"
description = "Minimal fake vLLM hardware plugin for onboarding"
requires-python = ">=3.10"
dependencies = ["vllm>=0.11.0"]

[project.entry-points."vllm.platform_plugins"]
# 平台偵測：vLLM 解析 current_platform 時呼叫 register()
# 名稱 "xpu_fake" 會出現在啟動日誌 "Platform plugin xpu_fake is activated"
xpu_fake = "vllm_xpu_minimal:register"

[project.entry-points."vllm.general_plugins"]
# 啟動 hook：每個 process 都會執行一次（冪等！）
xpu_fake_ops = "vllm_xpu_minimal:register_ops"

[tool.setuptools.packages.find]
include = ["vllm_xpu_minimal*"]
```

**教學點**：
- `vllm.platform_plugins` 的函式回傳「平台類別 qualified name 字串」或 None
- `vllm.general_plugins` 的函式被執行、回傳值被忽略——用來做模型/op 註冊
- OOT platform plugin 只能啟動一個（vLLM 會檢查）

---

## 2. `vllm_xpu_minimal/__init__.py` — 偵測函式（鐵律示範）

```python
# vllm_xpu_minimal/__init__.py
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
```

---

## 3. `vllm_xpu_minimal/envs.py` — 插件自己的環境變數

```python
# vllm_xpu_minimal/envs.py
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
```

---

## 4. `platforms/fake_platform.py` — Platform（心臟）

```python
# vllm_xpu_minimal/platforms/fake_platform.py
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
    device_name: str = "xpu_fake"     # get_device_name() 的 fallback
    device_type: str = "xpu_fake"     # torch.device 字串
    dispatch_key: str = "CPU"         # 【真插件】"PrivateUse1"（自訂裝置）或 "XLA"（TPU）
    ray_device_key: str = "XPU_FAKE"  # 【真插件】Ray 資源標記（ray_device_key）
    dist_backend: str = "gloo"        # 【真插件】"nccl"/"ccl"/"rbln-ccl"/TPU 不用
    device_control_env_var: str = "XPU_FAKE_VISIBLE_DEVICES"
    simple_compile_backend: str = "eager"  # 【真插件】TPU 用 "eager"（繞過 torch.compile）

    supported_quantization: list[str] = []  # 【真插件】量化白名單

    # ---- 裝置查詢 ----

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        """【真插件】查硬體型號：TPU 用 tpu_info、RBLN 用 rebel.get_npu_name()。
        失敗要有 fallback（RBLN 用 RBLN_FORCE_NPU_NAME env）。"""
        return "XPU-Fake-Device"

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
        return "vllm.distributed.device_communicators.cpu_communicator.CpuCommunicator"

    # ---- Attention backend ----

    @classmethod
    def get_attn_backend_cls(cls, selected_backend, attn_selector_config,
                             **kwargs) -> str:
        """【真插件】強制選自己的 backend：
        TPU 強制 FLASH_ATTN（MLA 模型 → FLASH_ATTN_MLA）；
        RBLN 只允許 FLASH_ATTN/FLASH_ATTN_MLA，其他 raise。"""
        from vllm.v1.attention.backends.registry import AttentionBackendEnum
        logger.info("[XPUFakePlatform] Forcing FAKE attention backend.")
        return "vllm_xpu_minimal.v1.attention.fake_attention.XPUFakeAttentionBackend"

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
            raise ValueError("Speculative decoding is not supported on XPU-Fake.")

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
            pc.worker_cls = "vllm_xpu_minimal.v1.worker.fake_worker.XPUFakeWorker"
            logger.info("[XPUFakePlatform] worker_cls -> XPUFakeWorker")

        # 5. executor backend（抄 TPU：單機強制 uni）
        if pc.distributed_executor_backend in (None, "auto"):
            pc.distributed_executor_backend = "uni"
            logger.info("[XPUFakePlatform] executor -> UniProcExecutor")

        # 6. 關閉不支援的功能（抄 RBLN：cascade attention）
        if mc and not mc.disable_cascade_attn:
            logger.warning("[XPUFakePlatform] Disabling cascade attention.")
            mc.disable_cascade_attn = True

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
```

---

### 5. `v1/worker/fake_worker.py` — Worker（管家，繼承 CPUWorker）

```python
# vllm_xpu_minimal/v1/worker/fake_worker.py
"""XPUFakeWorker：vLLM executor 與「Fake XPU 裝置世界」之間的橋樑。

繼承體系：WorkerBase -> Worker (gpu_worker) -> CPUWorker -> XPUFakeWorker
以 CPU 模擬 Fake XPU 硬體接入（Hardware Attachment Demo）。

【真插件對照】
- tpu-inference: worker/tpu_worker.py（JAX 裝置隔離 env、假 torch.distributed）
- vllm-rbln:     v1/worker/rbln_worker.py（RBLN_VISIBLE_DEVICES、rbln-ccl）
"""
import logging
from typing import TYPE_CHECKING, Optional

import torch
from vllm.v1.worker.cpu_worker import CPUWorker

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.worker_base import CompilationTimes

logger = logging.getLogger(__name__)


class XPUFakeWorker(CPUWorker):
    """Fake XPU Worker（繼承 CPUWorker，以 CPU 模擬硬體接入）。"""

    def __init__(self, *args, **kwargs):
        try:
            super().__init__(*args, **kwargs)
        except Exception as e:
            logger.warning(
                "[XPUFakeWorker] CPUWorker.__init__ fallback: %s", e)
            from vllm.v1.worker.gpu_worker import Worker
            Worker.__init__(self, *args, **kwargs)

        self.parallel_config.disable_custom_all_reduce = True
        logger.info(
            "[XPUFakeWorker] Worker initialized (inheriting CPUWorker). "
            "Simulating Fake XPU accelerator attached to Host.")

    # ---- 生命週期 1：裝置初始化 (Hardware Attach Demo) ----

    def init_device(self) -> None:
        """模擬 Fake XPU 硬體接入與裝置初始化。"""
        logger.info("==================================================================")
        logger.info("[XPUFakeWorker] [Hardware Demo] Connecting to Fake XPU Accelerator...")
        logger.info("[XPUFakeWorker] [Hardware Demo] Device: TranXform XPU-v1 (Simulated on CPU)")
        logger.info("[XPUFakeWorker] [Hardware Demo] Bus Info: PCIe Gen5 x16 @ NUMA Node 0")
        logger.info("[XPUFakeWorker] [Hardware Demo] Driver Version: fake-xpu-drv-v1.0.0 (active)")
        logger.info("==================================================================")

        # 走 CPUWorker 初始化建立分散式環境 (gloo backend) 與執行緒設定
        super().init_device()
        self.device = torch.device("cpu")

        # ★ 關鍵：將 model_runner 替換為我們自訂的 XPUFakeModelRunner
        from vllm_xpu_minimal.v1.worker.fake_model_runner import XPUFakeModelRunner
        self.model_runner = XPUFakeModelRunner(self.vllm_config, self.device)
        logger.info(
            "[XPUFakeWorker] Fake XPU hardware attached successfully. "
            "ModelRunner instantiated: %s", type(self.model_runner).__name__)

    # ---- 生命週期 2：模型載入 ----

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        logger.info("[XPUFakeWorker] [Hardware Demo] Loading model weights into simulated XPU memory...")
        super().load_model(load_dummy_weights=load_dummy_weights)
        logger.info("[XPUFakeWorker] [Hardware Demo] Model loaded successfully.")

    # ---- 生命週期 3：記憶體探測 (KV cache 協議) ----

    def determine_available_memory(self) -> int:
        from vllm_xpu_minimal.envs import XPU_FAKE_DEVICE_MEMORY_BYTES
        logger.info("[XPUFakeWorker] [Hardware Demo] Querying Fake XPU onboard memory...")
        try:
            if hasattr(self.model_runner, "warming_up_model"):
                self.model_runner.warming_up_model()
            elif hasattr(self.model_runner, "dummy_warmup"):
                self.model_runner.dummy_warmup()

            util = self.cache_config.gpu_memory_utilization
            model_bytes = sum(
                p.numel() * p.element_size()
                for p in self.model_runner.model.parameters())
            simulated_total = XPU_FAKE_DEVICE_MEMORY_BYTES
            available = int(simulated_total * util - model_bytes)

            logger.info(
                "[XPUFakeWorker] Fake XPU Onboard Memory: Total=%.2f GiB, "
                "Model=%.2f GiB, Available for KV=%.2f GiB (util=%.2f)",
                simulated_total / 2**30, model_bytes / 2**30,
                available / 2**30, util)
            if available <= 0:
                return super().determine_available_memory()
            return available
        except Exception as e:
            logger.info("[XPUFakeWorker] Custom memory calculation fallback: %s", e)
            return super().determine_available_memory()

    # ---- 生命週期 4：KV Cache 配置 ----

    def initialize_from_config(self, kv_cache_config: "KVCacheConfig") -> None:
        logger.info(
            "[XPUFakeWorker] [Hardware Demo] Initializing %d KV cache blocks on Fake XPU...",
            kv_cache_config.num_blocks)
        super().initialize_from_config(kv_cache_config)
        logger.info("[XPUFakeWorker] [Hardware Demo] KV cache allocation complete.")

    # ---- 生命週期 5：編譯與暖機 ----

    def compile_or_warm_up_model(self) -> "CompilationTimes":
        logger.info("[XPUFakeWorker] [Hardware Demo] Warming up Fake XPU execution graph...")
        times = super().compile_or_warm_up_model()
        logger.info("[XPUFakeWorker] [Hardware Demo] Warmup finished successfully.")
        return times

    # ---- 裝置健康檢查 ----

    def check_health(self) -> None:
        """探測 Fake XPU 裝置狀態（模擬硬體心跳檢查）。"""
        return
```

---

## 6. `v1/worker/fake_model_runner.py` — ModelRunner（主廚，繼承 CPUModelRunner / CPURunner）

```python
# vllm_xpu_minimal/v1/worker/fake_model_runner.py
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
            "(inheriting CPUModelRunner/CPURunner, device=%s).", device)

    # ---- 模型載入 ----

    def load_model(self, load_dummy_weights: bool = False) -> None:
        logger.info(
            "[XPUFakeModelRunner] [FakeXPU Hardware] Loading model weights into simulated XPU memory...")
        super().load_model(load_dummy_weights=load_dummy_weights)
        logger.info(
            "[XPUFakeModelRunner] [FakeXPU Hardware] Weights successfully loaded.")

    # ---- 暖機與預編譯 (AOT / JIT Warmup) ----

    def warming_up_model(self) -> None:
        """暖機：跑 dummy forward（模擬 XPU AOT precompile / warm-up）。"""
        logger.info(
            "[XPUFakeModelRunner] [FakeXPU Hardware] Executing warmup pass on simulated XPU pipelines...")
        super().warming_up_model()
        logger.info(
            "[XPUFakeModelRunner] [FakeXPU Hardware] Warmup pass completed.")

    def dummy_warmup(self) -> None:
        """向下相容的 warmup 入口。"""
        self.warming_up_model()

    # ---- 執行模型 (Inference Execution) ----

    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors=None,
    ) -> Optional["ModelRunnerOutput"]:
        return super().execute_model(scheduler_output, intermediate_tensors)


# 別名導出（支持 CPURunner / XPUFakeCPURunner 命名）
CPURunner = CPUModelRunner
XPUFakeCPURunner = XPUFakeModelRunner
```

---

## 7. `v1/attention/fake_attention.py` — Attention backend（爐台）

```python
# vllm_xpu_minimal/v1/attention/fake_attention.py
"""假 attention backend：示範 AttentionBackend 三件組的介面。

vLLM 把 attention 抽成三件組：
  Backend（註冊點/靜態屬性）→ MetadataBuilder（每步建 metadata）→ Impl（forward）

【真插件對照】
- tpu-inference: layers/vllm/backends/flash_attn.py
  （PallasAttentionBackend：get_kv_cache_shape head 對齊 128、
   get_page_size 由 SMEM 反推、forward 走 jax_view → jit → shard_map → RPA kernel）
- vllm-rbln:     v1/attention/backends/flash_attention.py
  （RBLNFlashAttentionBackend：kv cache 是 graph input、
   rbln_triton_ops kernel、SWA ring buffer）
- 上游 CPU 版:   vllm/v1/attention/backends/cpu_attn.py（TorchSDPABackend）

假插件：直接繼承 TorchSDPABackend（CPU 上能跑），只示範註冊介面。
"""
import logging

from vllm.v1.attention.backends.cpu_attn import TorchSDPABackend

logger = logging.getLogger(__name__)


class XPUFakeAttentionBackend(TorchSDPABackend):
    """繼承 CPU 的 SDPA backend。

    【真插件必寫的三個靜態方法】（這裡繼承上游，僅註解示範）：

    @staticmethod
    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size,
                           cache_dtype_str="auto"):
        # TPU: (num_blocks, block_size, num_kv_heads*2, padded_head_size)
        #      head_size 對齊 128（TPU_HEAD_SIZE_ALIGNMENT）
        # RBLN: [2, num_kv_heads, 1, block_size, head_size]
        # GPU:  (2, num_blocks, block_size * num_kv_heads * head_size)
        ...

    @staticmethod
    def get_page_size(vllm_config):
        # TPU 獨有：page size 由 SMEM/SREG 限制反推
        # （max_model_len > 8192 → 16，否則 next_pow2(max_len)//16 夾在 [16,256]）
        ...

    class MetadataBuilder(AttentionMetadataBuilder):
        def build(self, common_prefix_len, common_attn_metadata, ...):
            # 每步把 CommonAttentionMetadata 轉成 backend 自己的 metadata
            # TPU：dummy shell（真 metadata 由編譯管線動態建）
            # RBLN：host 建 mask、stage 到 persistent device buffer
            ...
    """

    @classmethod
    def get_supported_dtypes(cls) -> list:
        return [torch.float16, torch.bfloat16, torch.float32]


import torch  # noqa: E402  （教學檔案，import 放底部以凸顯註解主體）
```

---

## 8. `__init__.py` 們（空檔案）

```python
# vllm_xpu_minimal/platforms/__init__.py
from vllm_xpu_minimal.platforms.fake_platform import XPUFakePlatform  # noqa

# vllm_xpu_minimal/v1/__init__.py
# （空）

# vllm_xpu_minimal/v1/worker/__init__.py
# （空）

# vllm_xpu_minimal/v1/attention/__init__.py
# （空）
```

---

## 9. 驗證清單（跑通它）

```bash
# 1. 安裝（與 vLLM 同一環境）
cd vllm-xpu-minimal && pip install -e .

# 2. 驗證 entry point 被看見
python -c "from importlib.metadata import entry_points; \
print([f'{e.name} -> {e.value}' for e in entry_points(group='vllm.platform_plugins')])"
# 應該看到 "xpu_fake -> vllm_xpu_minimal:register"

# 3. 驗證平台偵測
VLLM_XPU_FAKE=1 python -c "
import vllm.platforms as p
print(type(p.current_platform))   # <class '...XPUFakePlatform'>
print(p.current_platform.get_device_name())  # TranXform-XPU-Fake-Device (CPU-Emulated)"

# 4. 跑通 serve（小模型）
VLLM_XPU_FAKE=1 vllm serve Qwen/Qwen2.5-0.5B-Instruct --max-model-len 512
# 日誌檢查點：
#   "Platform plugin xpu_fake is activated"
#   "[XPUFakePlatform] [Hardware Demo] Assigned worker_cls -> XPUFakeWorker (inheriting CPUWorker)"
#   "[XPUFakeWorker] [Hardware Demo] Connecting to Fake XPU Accelerator..."
#   "[XPUFakeWorker] [Hardware Demo] Device: TranXform XPU-v1 (Simulated on CPU)"
#   "[XPUFakeWorker] Fake XPU hardware attached successfully. ModelRunner instantiated: XPUFakeModelRunner"
#   "[XPUFakeWorker] [Hardware Demo] Querying Fake XPU onboard memory..."
#   "[XPUFakeWorker] [Hardware Demo] Initializing N KV cache blocks on Fake XPU..."
#   "[XPUFakeWorker] [Hardware Demo] Warming up Fake XPU execution graph..."

# 5. 發一個請求
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen2.5-0.5B-Instruct", "messages": [{"role": "user", "content": "hi"}]}'
```

---

## 9.5 實測踩到的坑（第一手教訓）

### 坑 1：`device_type` 與 `device_name` 都必須是 torch 認得的裝置字串

**症狀**（出現兩次、兩個位置）：
```
RuntimeError: Expected one of cpu, cuda, ipu, xpu, mkldnn, opengl, opencl,
ideep, hip, ve, fpga, maia, xla, lazy, vulkan, mps, meta, hpu, mtia,
privateuseone device type at start of device string: xpu_fake
```

**根因**（同一類問題，兩個炸點）：
1. `DeviceConfig.__post_init__`（`vllm/config/device.py:70-74`）：
   `torch.device(device_type)`——只有 `"tpu"` 在硬編碼白名單（device=None）
2. `GroupCoordinator.__init__`（`vllm/distributed/parallel_state.py:245`）：
   **OOT 平台**走 `torch.device(f"{device_name}:{local_rank}")`——
   `device_name` 也會被拿去建 torch.device！
   （TPU 是 in-tree，`is_out_of_tree()=False` 走 else 分支的 "cpu"，
   所以它的 `device_name="tpu"` 沒事——**in-tree 與 OOT 的規則不同**）

**解法**（已修正到 `fake_platform.py`）：
- `device_type = "cpu"`、`device_name = "cpu"`（兩個都 torch 認得）
- 顯示用的自訂名稱放 `get_device_name()`（classmethod 回 "XPU-Fake-Device"）

**真插件的規則**：
- SDK 已註冊 torch 裝置（PrivateUse1）→ 兩個屬性都用自訂字串
  （RBLN：`"rbln" if USE_DEVICE_TENSOR else "cpu"`，兩個屬性同步切換）
- 沒註冊 → 兩個都 `"cpu"`，tensor 靠插件手動放上裝置（TPU 式）

**教學點**：這正是「vLLM 假設 CUDA/已知裝置」的接觸點之一——
每個插件都要盤點這種硬編碼清單（tpu_platform.py 開頭的 monkeypatch
就是在補同類問題）。

### 坑 2：繼承 GPUModelRunner 會踩到 torch.cuda.* 呼叫

**症狀**：
```
torch.AcceleratorError: CUDA error: CUDA driver version is insufficient
for CUDA runtime version (cudaErrorInsufficientDriver)
```

**根因**：`GPUModelRunner.__init__` 有一整排 CUDA 呼叫：
- `torch.cuda.Stream()`（comm_stream、async_output_copy_stream）
- `torch.cuda.Event()`（transfer_event 等）
- `torch.cuda.get_device_properties()`（經 `_init_device_properties()`）
- 執行期還有 `torch.cuda.synchronize()`（`_sync_device()`）

**解法**（已修正到 `fake_model_runner.py`，抄自 vLLM 官方 CPU 後端
`vllm/v1/worker/cpu_model_runner.py`）：
1. `_torch_cuda_wrapper()`：`__init__` 期間用 placeholder 換掉
   `torch.cuda.Event/Stream`
2. override `_init_device_properties()` → pass
3. override `_sync_device()` → pass
4. Platform 端強制 `compilation_config.mode = NONE` + `cudagraph_mode = NONE`
   （避免 CUDA graph 捕獲路徑；RBLN 同款）

**教學點**：這是「盤點上游 CUDA 接觸點」的實戰——
tpu-inference 用 monkeypatch `torch.accelerator.*`、
RBLN 用 `USE_DEVICE_TENSOR` 條件式分流，本質相同：
**上游每一個 `torch.cuda.*` 呼叫點都是插件的潛在炸點，要逐個清點**。

### 坑 3：`get_current_memory_usage` 不實作 → load_model 炸 NotImplementedError

**症狀**：
```
File "vllm/v1/worker/gpu_model_runner.py", line 5266, in load_model
    with DeviceMemoryProfiler() as m:
File "vllm/utils/mem_utils.py", line 89, in current_memory_usage
    return current_platform.get_current_memory_usage(self.device)
File "vllm/platforms/interface.py", line 1085, in get_current_memory_usage
    raise NotImplementedError
```

**根因**：`GPUModelRunner.load_model()` 用 `DeviceMemoryProfiler` 量測
「載入模型花了多少裝置記憶體」，它轉發到
`current_platform.get_current_memory_usage(device)`——
**Platform 基底類是 `raise NotImplementedError`**，每個平台必須自己實作。

**解法**（已修正到 `fake_platform.py`）：
```python
@classmethod
def get_current_memory_usage(cls, device=None) -> float:
    return 0.0   # CPU 上沒有裝置記憶體峰值概念
```

**真插件的答案**（對照 `vllm/platforms/{cuda,xpu}.py`）：
- CUDA：`torch.cuda.empty_cache() + reset_peak_memory_stats() + max_memory_allocated()`
- Intel XPU：`torch.xpu.reset_peak_memory_stats() + max_memory_allocated()`
- RBLN：torch.rbln 的記憶體統計
- TPU：`mem_get_info()` 式的聚合查詢

**教學點**：Platform 的每個方法都要檢查基底類是「有預設」還是
「raise NotImplementedError」——後者是**必答題**，漏一個就在
生命週期的某一步炸。盤點方法：grep `raise NotImplementedError`
`vllm/platforms/interface.py`，逐項決定「實作 / raise / pass」。

### 坑 4：分散式初始化不可省（模型建構子就會用）

**症狀**：
```
File "vllm/model_executor/models/qwen2.py", line 367, in __init__
    get_pp_group().is_first_rank
File "vllm/distributed/parallel_state.py", line 1598, in get_pp_group
AssertionError: pipeline model parallel group is not initialized
```

**根因**：vLLM 的模型建構子（`Qwen2Model.__init__`）就會呼叫
`get_pp_group()`——分散式環境（TP/PP/DP group）必須在
**load_model 之前**初始化好，否則模型建不起來。

**解法**（已修正到 `fake_worker.py`，抄 `GPUWorker.init_device` 的
`init_worker_distributed_environment`，`vllm/v1/worker/gpu_worker.py:691`）：
```python
set_custom_all_reduce(not disable_custom_all_reduce)
init_distributed_environment(world_size, rank, ..., backend="gloo")
ensure_model_parallel_initialized(tp_size, pp_size)
ensure_kv_transfer_initialized(vllm_config)
```
假插件用 `file://` init method（TPU 同款）避免需要 TCP port。

**教學點**：這是 Worker 生命週期的隱藏依賴——
「init_device 不只是設裝置，還要建分散式環境」。
TPU 的做法更極端：故意初始化成 world_size=1 的假環境
（真平行度在 JAX SPMD 內），RBLN 則是真分散式（rbln-ccl）。
**無論哪種，這一步都不能跳過。**

### 坑 5：`ensure_kv_transfer_initialized` 的呼叫時機與簽名

**症狀**：
```
TypeError: ensure_kv_transfer_initialized() missing 1 required
positional argument: 'kv_cache_config'
```

**根因**：兩個問題疊加——
1. **簽名隨版本變**：GPUWorker 在 `init_device` 呼叫它（只要 vllm_config），
   但你環境的 vLLM 版本簽名是 `(vllm_config, kv_cache_config)`
2. **呼叫時機本來就太早**：`init_device` 時 KV cache 還沒配置，
   `kv_cache_config` 不存在

**解法**（已修正到 `fake_worker.py`，抄 TPU 的做法）：
- `init_device`：**不呼叫**（TPU 同款）
- `initialize_from_config`：呼叫 `ensure_kv_transfer_initialized(vllm_config, kv_cache_config)`
  （TPU 的 `tpu_worker.py:725` 就是在這裡呼叫——
  註解原文："Init kv cache connector here, because it requires kv_cache_config"）

**教學點**：兩個教訓——
1. **上游函數簽名會變**：抄任何上游呼叫前，先看**你環境裝的版本**的
   簽名（`inspect.signature`），不要只看文件或別的插件
2. **呼叫時機跟著依賴走**：需要 kv_cache_config 的初始化，
   就放在 kv_cache_config 存在之後（`initialize_from_config`）

### 坑 6：上游 backend 類別會改名/移除（版本漂移的實例）

**症狀**：
```
ImportError: cannot import name 'TorchSDPABackend' from
'vllm.v1.attention.backends.cpu_attn'
```

**根因**：0.11.0 wheel 的 `cpu_attn.py` 有 `TorchSDPABackend`，
你環境的 vLLM 版本沒有（改名或移除）。**上游 backend 類別不是穩定介面**。

**解法**（已修正到 `fake_attention.py`）：
改繼承 `FlexAttentionBackend`——純 torch 實作
（`torch.nn.attention.flex_attention`），無 CUDA extension 依賴。
但它的 KV 寫入走 `torch.ops._C_cache_ops.reshape_and_cache_flash`
（CUDA op）→ override `Impl.forward`，改用純 torch 的 scatter 寫入。

**教學點**：
1. **選繼承對象的標準**：優先選「純 torch 實作」的 backend
   （Flex/Triton），避免綁 CUDA extension 的（FlashAttn/SDPA-CPU）
2. **繼承也要檢查內部依賴**：父類 forward 裡的 op 呼叫也要盤點
   （FlexAttention 的 cache 寫入就是藏在 forward 裡的 CUDA op）
3. **對照真插件**：tpu-inference 的 `PallasAttentionBackend` 是
   `@register_backend(AttentionBackendEnum.FLASH_ATTN)` 註冊（蓋掉上游）；
   我們走 `get_attn_backend_cls()` 回傳路徑——兩種註冊方式並存

### 坑 7：`compile_or_warm_up_model` 必須回傳 CompilationTimes

**症狀**：
```
File "vllm/v1/executor/abstract.py", line 137, in <genexpr>
    t.language_model for t in compilation_times
AttributeError: 'NoneType' object has no attribute 'language_model'
```

**根因**：executor 的 `compile_or_warm_up_model`（`vllm/v1/executor/abstract.py:136`）
會對 worker 的回傳值做 `max(t.language_model for t in compilation_times)`——
**回傳 None 就炸**。

**解法**（已修正到 `fake_worker.py`，抄 TPU 的 `tpu_worker.py:675`）：
```python
return CompilationTimes(
    language_model=self.compilation_config.compilation_time,
    encoder=0.0,
)
```

**教學點**：Worker 的每個生命週期方法的**回傳值也是介面的一部分**——
不只「不炸」，還要「回對的東西」。盤點方法：看 executor 怎麼消費
worker 的回傳值（`collective_rpc` 的結果）。

### 坑 8：warmup 的 `_C_cache_ops` 錯誤（CUDA op 的隱藏呼叫點）

**症狀**：
```
dummy warmup failed: '_OpNamespace' '_C_cache_ops' object has no
attribute 'reshape_and_cache_flash'
```

**根因**：FlexAttention 的 `Impl.forward` 用
`torch.ops._C_cache_ops.reshape_and_cache_flash`（CUDA op）寫 KV cache。
我們 override 了 forward，但 warmup 的 `_dummy_run` 路徑還有別的
呼叫點（metadata builder 的 block mask 建構等）。

**解法**：`dummy_warmup` 的 try/except 已涵蓋——warmup 失敗只 warning
不擋啟動（教學插件寬容；真插件會 fail-fast 或把所有 CUDA 呼叫點補齊）。

**教學點**：**CUDA op 的呼叫點藏在很多層**——backend forward、
metadata builder、sampler、rejection sampler（Triton kernel）。
盤點方法：grep `torch.ops._C` / `@triton.jit` 於你繼承的上游檔案，
逐個決定「override / 禁用功能 / 補實作」。

### 坑 9：先查上游有沒有現成的同類後端（總結坑 2/6/8）

**教訓**：坑 2（CUDA wrapper）、坑 6（backend 版本漂移）、坑 8（CUDA op）
三個問題，vLLM 官方 CPU 後端**全部已經解過**——
`CPUModelRunner(GPUModelRunner)` 內建 `_torch_cuda_wrapper`、
`_init_device_properties/_sync_device` override、`_postprocess_tensors`，
且與 CPU 的 attention backend 原生搭配。

**修正**（已套用）：
- `fake_worker.py`：改繼承 `CPUWorker`（復用上游 CPU 的分散式環境與執行管線，模擬硬體接入）
- `fake_model_runner.py`：改繼承 `CPUModelRunner`（CPURunner，刪掉自補的 wrapper 與 override）
- `fake_attention.py`：防禦式 import `TorchSDPABackend`（有就用，
  沒有 fallback 到 FlexAttention + 純 torch KV 寫入）

**規則**：寫插件前先盤點 vLLM 內建後端（cpu/xpu/tpu/rocm…），
**有同類後端就繼承它，沒有才自寫**——真插件（TPU/RBLN）自寫 runner
是因為執行語義完全不同（JAX/編譯 artifact），不是因為上游沒有。

### 坑 11：OOT 平台的 Sleep Mode Allocator 陷阱

**現象**：
在 `load_model` 階段拋出：
```text
RuntimeError: Sleep mode allocator is not available on platform XPUFakePlatform (device_type=cpu).
```

**原因分析**：
1. vLLM V1 的 `Worker.load_model()` 與 `Worker.initialize_from_config()` 會呼叫 `_maybe_get_memory_pool_context(tag)`。
3. 自訂 OOT 平台（`_enum = PlatformEnum.OOT`）預設 `is_cpu()` 為 `False`，因此跌入 `get_mem_allocator_instance()`。
4. `get_mem_allocator_instance()` 僅支援 CUDA 與 Intel XPU，對 OOT 平台直接拋錯崩潰。

**雙重防禦解法**：
1. **Worker 層防護**：在 `fake_worker.py` 中覆寫 `_maybe_get_memory_pool_context(self, tag=None)`，直接回傳 `nullcontext()`。
2. **Platform 層防護**：在 `fake_platform.py` 中覆寫 `is_cpu(self) -> bool: return True`，讓 vLLM 內部所有硬體邏輯正確辨識為 CPU 模擬平台。

---

### 坑 12：BlockTable 內建 Triton Kernel 洩漏至 CPU 棧（`TypeError: Cannot inspect kernel parameters for function`）

**現象**：
在 Server 接收推論請求時（`execute_model` 階段），`_prepare_inputs` 呼叫 `BlockTable.compute_slot_mapping` 噴錯：
```text
File ".../vllm/v1/worker/block_table.py", line 214, in compute_slot_mapping
    _COMPUTE_SLOT_MAPPING_KERNEL(...)
File ".../vllm/model_executor/warmup/jit_warmup_triton_helper.py", line 268, in _kernel_arg_names
    raise TypeError(f"Cannot inspect kernel parameters for {type(self.kernel).__name__}")
TypeError: Cannot inspect kernel parameters for function
```

**原因分析**：
1. 雖然外掛已改為繼承 `CPUWorker` 與 `CPUModelRunner`，但上游的 `CPUModelRunner` 其實是直接繼承 `GPUModelRunner`，且未重寫 `execute_model()`。
2. `GPUModelRunner._prepare_inputs` 會調用 `BlockTable.compute_slot_mapping`。
3. vLLM V1 近期將該函式硬編碼改為調用 Triton JIT Kernel（`_COMPUTE_SLOT_MAPPING_KERNEL`），完全沒有做硬體或 Triton 是否存在的判斷！
4. 在純 CPU 環境中，Triton 會回退為 `TritonPlaceholder`，導致 `@triton.jit` 回傳原始 Python 函式。`jit_warmup_triton_helper.py` 檢查型別時發現不是編譯過的 Triton 物件，直接拋出 `TypeError` 崩潰。

**解法**：
在 `fake_model_runner.py` 啟動時 Monkey-Patch `BlockTable.compute_slot_mapping`，提供純 PyTorch 向量化實作，徹底擺脫 Triton 依賴。

---

### 坑 13：RoPE C++ 算子缺失與 CPU 分發陷阱（`AttributeError: '_OpNamespace' '_C' object has no attribute 'rotary_embedding'`）

**現象**：
在 Server 進行 Forward Pass 經過 Transformer Attention 層時噴錯：
```text
File ".../vllm/model_executor/layers/rotary_embedding/base.py", line 310, in forward_cpu
    ops.rotary_embedding(...)
File ".../vllm/_custom_ops.py", line 221, in rotary_embedding
    torch.ops._C.rotary_embedding(...)
AttributeError: '_OpNamespace' '_C' object has no attribute 'rotary_embedding'
```

**原因分析**：
1. **為什麼繼承了 CPUWorker / CPURunner 還會報錯？**：
   - `CPUWorker` 與 `CPUModelRunner` 僅負責外圍調度與環境適配（如把 CUDA Event/Stream 轉為 placeholder、將裝置記憶體搬運重導向 CPU）。
   - 神經網路核心結構（Layers / CustomOps）依然調用統一的 `vllm.model_executor.layers.*`。
2. **算子分發與 C++ 依賴斷層**：
   - `RotaryEmbedding` 繼承自 `CustomOp`。因為平台申報 `is_cpu() == True`，`CustomOp.dispatch_forward()` 將前向傳播分發給 `RotaryEmbedding.forward_cpu`。
   - `forward_cpu` 內部調用 `vllm._custom_ops.rotary_embedding`，而該函式寫死調用 C++ 擴展 `torch.ops._C.rotary_embedding`。
   - 在 CPU 環境或缺少 CUDA 驅動的環境（如 WSL / 純 CPU），`vllm._C_stable_libtorch.abi3.so` 因缺少 `libcuda.so.1` 無法載入，`torch.ops._C` 命名空間完全為空，直接拋出 `AttributeError` 崩潰。
   - 事實上，vLLM 原生已經在 `RotaryEmbedding` 實作了純 PyTorch 的向量化版本 `forward_native`，但官方的 `forward_cpu` 卻未實作 fallback！

**雙重防禦解法**：
在 `fake_model_runner.py` 與 `fake_platform.py` 中實作 `_patch_rotary_embedding_for_cpu()`：
1. **類別方法重導向**：將 `RotaryEmbedding.forward_cpu` 直接替換為官方內建的純 PyTorch 實作 `RotaryEmbedding.forward_native`。
2. **全域自訂算子 Fallback**：同時重寫 `vllm._custom_ops.rotary_embedding`，提供基於 PyTorch 向量化的 in-place RoPE 計算，保證無論底層如何調用均不會崩潰。

---

## 10. 從假插件到真插件：成長路徑（每一步換掉什麼）

| 步驟 | 假插件 | 真插件（XPU） | 參考 |
|---|---|---|---|
| 1. 偵測 | env var 開關 | SDK 探測（`rebel.get_npu_name()` 式） | rbln `get_device_name` |
| 2. dispatch_key | "CPU" | "PrivateUse1"（SDK 註冊 torch 裝置後） | vllm-ascend torch_npu |
| 3. Worker | 繼承 CPUWorker（CPU 模擬硬體接入） | 自寫 Worker + SDK 裝置初始化 + 真分散式 init | rbln `init_device` |
| 4. Runner | 繼承 CPUModelRunner（CPURunner） | 自寫（輸入準備/執行/取樣全接管） | tpu_runner / rbln_model_runner |
| 5. 模型 | vLLM 原生 loader | graph capture（torch.compile + 自家 backend） | rbln `compile()` |
| 6. Attention | 繼承 TorchSDPA | 自寫三件組 + XPU kernel | 兩插件的 backend |
| 7. KV cache | torch.zeros | graph input tensor / 靜態 jax.Array + dynamic sizing | rbln dynamic_kv_sizer |
| 8. 編譯管理 | 無 | bucketing + dummy run + mega-cache | rbln bucketing/mega_cache |
| 9. 通訊 | gloo | 自家 CCL（rbln-ccl 式） | rbln communicator |
| 10. 補丁 | 無 | Patch Registry（宣告式） | rbln patches/registry.py |

**學習檢核**：能不看文件回答以下五題，就算理解插件層——
1. vLLM 怎麼從 entry point 走到 `current_platform` 實例？（lazy `__getattr__` → 偵測函式 → qualified name → import）
2. `check_and_update_config` 為什麼是主控台？（它是唯一能改寫所有 config 的時機點，且在一切初始化之前）
3. `determine_available_memory` 回傳的數字去哪了？（vLLM 算 num_blocks → KVCacheConfig → `initialize_from_config`）
4. 為什麼 KV cache 必須是 graph input 而非模組屬性？（否則被 Dynamo bake 成常數，無法換/resize/in-place）
5. bucketing 解決什麼矛盾？（編譯期靜態 shape vs 執行期動態 batch）

---

## 11. 官方文件對照表（Plugin System 文件 → 本插件）

> 來源：https://docs.vllm.ai/en/latest/design/plugin_system/
> 官方 "Platform plugins guidelines" 的每一步與本插件的對應：

| 官方步驟 | 官方要求 | 本插件對應 | 狀態 |
|---|---|---|---|
| 1. 專案結構 | `__init__.py` + platform/worker/attention/communicator/custom_ops 各一檔 | 完全同構（`platforms/`、`v1/worker/`、`v1/attention/`） | ✅ |
| 2. entry point | `vllm.platform_plugins` → register() 回 qualified name | `pyproject.toml` + `__init__.py:register()` | ✅ |
| 3a. `_enum` | `PlatformEnum.OOT` | `fake_platform.py` | ✅ |
| 3b. `device_type` | torch 用的裝置字串 | `"cpu"`（坑 1 的教訓） | ✅ |
| 3c. `device_name` | 顯示用 | `"xpu_fake"` | ✅ |
| 3d. `check_and_update_config` | **worker_cls 必須在這裡設**（官方原文強調） | 第 4 步 | ✅ |
| 3e. `get_attn_backend_cls` | 回 backend qualified name | ✅ | ✅ |
| 3f. `get_device_communicator_cls` | 回 communicator qualified name | ✅ | ✅ |
| 4. Worker 必答題 | `init_device` / `initialize_cache` / `load_model` / `get_kv_cache_spec` / `determine_available_memory` / `initialize_from_config` / `execute_model` | `fake_worker.py` 全部有（坑 2/3 補齊） | ✅ |
| 4-optional. 功能開關 | sleep→`sleep/wake_up`；graph→`compile_or_warm_up_model`；spec decode→`take_draft_token_ids`；LoRA→`add_lora/remove_lora/list_loras/pin_lora`；DP→`execute_dummy_batch` | sleep/LoRA/spec 未實作（fail-fast 禁用）；`execute_dummy_batch` 繼承基底 | 部分 |
| 5. Attention backend | 繼承 `AttentionBackend`，參考 `vllm/v1/attention/backends/` | 繼承 `TorchSDPABackend`（CPU 版） | ✅ |
| 6. Custom ops | communicator ops / common ops（`CustomOp.register_oot`）/ csrc ops | 假插件不需要（CPU 原生 op 夠用） | N/A |
| 7. 其他 pluggable | LoRA / graph backend / quantization / mamba backend | 全部 fail-fast 禁用 | N/A |

### 官方文件補充的重點（本插件 README 原本沒寫的）

1. **五種 plugin group**（不只兩種）：
   - `vllm.general_plugins`——模型註冊（官方範例：bart-plugin）
   - `vllm.platform_plugins`——平台（我們的主題）
   - `vllm.io_processor_plugins`——pooling 模型的前後處理
   - `vllm.stat_logger_plugins`——自訂統計 logger
   - `vllm.endpoint_plugins`——自訂 HTTP route（**預設不載入**，要 `VLLM_PLUGINS` 白名單，安全考量）
2. **`VLLM_PLUGINS` env**：可過濾要載入的插件（除錯時很有用：
   `VLLM_PLUGINS=xpu_fake_ops` 只載我們的）
3. **re-entrant 鐵律的官方原文**："The function specified in the entry point
   should be re-entrant, meaning it can be called multiple times without
   causing issues"——與我們 §2 的鐵律 3 一致
4. **相容性保證的邊界**：官方只保證「有文件記載的介面」
   （如 `ModelRegistry.register_model`）永遠可用；**其他介面（Platform/
   Worker/Backend 的方法）隨時會變**——這印證了「版本追隨是持續成本」
   （Onboarding 報告 §5.1）
5. **Deprecation 清單**（官方文件底部）：`use_v1` 參數、`_Backend`、
   `seed_everything`、`validate_request` 的 `prompt` 參數都已被移除——
   **升級 vLLM 版本時要逐條核對這份清單**（我們的 `manual_seed_all`
   對應已棄用的 `seed_everything`，新版應改用 `set_random_seed`）

### 對 XPU 專案的行動項

- [ ] 把官方文件的 Worker「功能開關對應表」（sleep/graph/spec/LoRA/DP）
      併入 Onboarding 報告 §3.2 的 Phase 2-4 部件清單
- [ ] 追蹤官方文件的 Deprecation 區塊（每次升版必讀）
- [ ] 評估 `vllm.io_processor_plugins` / `stat_logger_plugins` 是否有
      XPU 特有的使用場景（如裝置健康度 logger）
