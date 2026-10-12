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


def _patch_block_table_for_cpu() -> None:
    """坑 12：避開 BlockTable 硬編碼 Triton Kernel 洩漏至 CPU 棧。

    vLLM V1 的 BlockTable.compute_slot_mapping 原生硬編碼調用了 _COMPUTE_SLOT_MAPPING_KERNEL (Triton)。
    當在 CPU 環境（無 Triton 驅動）下執行時，會觸發：
      TypeError: Cannot inspect kernel parameters for function
    在此將其替換為純 PyTorch 向量化實作，徹底擺脫 Triton 依賴。
    """
    try:
        from vllm.v1.worker.block_table import BlockTable, SlotMappingMode
        from vllm.v1.attention.backends.utils import PAD_SLOT_ID

        def cpu_compute_slot_mapping(
            self,
            num_reqs: int,
            query_start_loc: torch.Tensor,
            positions: torch.Tensor,
        ) -> None:
            num_tokens = positions.shape[0]
            if self.slot_mapping_mode == SlotMappingMode.NONE:
                return

            block_table = self.block_table.gpu
            slot_mapping = self.slot_mapping.gpu
            block_size = self.block_size
            kv_cache_block_size = self.kv_cache_block_size
            blocks_per_kv_block = self.blocks_per_kv_block
            dcp_world_size = self.dcp_world_size
            dcp_rank = self.dcp_rank
            cp_interleave = self.cp_kv_cache_interleave_size

            # 填充其餘 slot 為 PAD_SLOT_ID (-1)
            if num_tokens < self.max_num_batched_tokens:
                slot_mapping[num_tokens:self.max_num_batched_tokens].fill_(PAD_SLOT_ID)

            query_start = query_start_loc.cpu().tolist()
            virtual_block_size = kv_cache_block_size * dcp_world_size

            for req_idx in range(num_reqs):
                s = query_start[req_idx]
                e = query_start[req_idx + 1]
                if s >= e:
                    continue
                pos = positions[s:e]
                v_block_idx = pos // virtual_block_size
                v_block_offset = pos - v_block_idx * virtual_block_size

                is_local = ((v_block_offset // cp_interleave) % dcp_world_size) == dcp_rank
                local_block_offset = (
                    (v_block_offset // (dcp_world_size * cp_interleave)) * cp_interleave
                    + (v_block_offset % cp_interleave)
                )
                block_idx = v_block_idx * blocks_per_kv_block + local_block_offset // block_size
                block_numbers = block_table[req_idx, block_idx].to(torch.int64)
                slot_offsets = local_block_offset % block_size
                slot_ids = block_numbers * block_size + slot_offsets
                slot_ids = torch.where(is_local, slot_ids, PAD_SLOT_ID)
                slot_mapping[s:e] = slot_ids

        BlockTable.compute_slot_mapping = cpu_compute_slot_mapping
        logger.info(
            "[XPUFakeModelRunner] [Hardware Demo] Successfully patched BlockTable.compute_slot_mapping with pure PyTorch CPU impl."
        )
    except Exception as e:
        logger.warning(
            "[XPUFakeModelRunner] Failed to patch BlockTable.compute_slot_mapping: %s",
            e,
        )


def _patch_rotary_embedding_for_cpu() -> None:
    """坑 13：避開 C++ 缺少 torch.ops._C.rotary_embedding 符號。

    在 CPU 或未編譯 C++ RoPE 內核環境下，torch.ops._C 缺少 rotary_embedding。
    vLLM 原生 RotaryEmbedding.forward_cpu 與 vllm._custom_ops.rotary_embedding
    會直接噴 AttributeError: '_OpNamespace' '_C' object has no attribute 'rotary_embedding'。
    在此提供雙重保護：
    1. 將 RotaryEmbedding.forward_cpu 指向官方已內建的純 PyTorch 實作 forward_native。
    2. 同時在 vllm._custom_ops.rotary_embedding 提供純 PyTorch 向量化 fallback 實作（確保 in-place 更新語意）。
    """
    try:
        import vllm._custom_ops as ops
        from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
        from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

        def cpu_rotary_embedding(
            positions: torch.Tensor,
            query: torch.Tensor,
            key: torch.Tensor | None,
            head_size: int,
            cos_sin_cache: torch.Tensor,
            is_neox: bool,
            rope_dim_offset: int = 0,
            inverse: bool = False,
        ) -> None:
            positions = positions.flatten()
            num_tokens = positions.shape[0]
            cos_sin = cos_sin_cache.index_select(0, positions)
            cos, sin = cos_sin.chunk(2, dim=-1)
            rotary_dim = cos.shape[-1] * 2

            # Query in-place RoPE
            q_shape = query.shape
            q_view = query.reshape(num_tokens, -1, head_size)
            q_rot = q_view[..., rope_dim_offset : rope_dim_offset + rotary_dim]
            q_rot_out = ApplyRotaryEmb.forward_static(q_rot, cos, sin, is_neox)

            if q_view.data_ptr() == query.data_ptr():
                q_view[..., rope_dim_offset : rope_dim_offset + rotary_dim] = q_rot_out
            else:
                q_pass_pre = q_view[..., :rope_dim_offset]
                q_pass_post = q_view[..., rope_dim_offset + rotary_dim :]
                q_full = torch.cat([q_pass_pre, q_rot_out, q_pass_post], dim=-1).reshape(q_shape)
                query.copy_(q_full)

            # Key in-place RoPE
            if key is not None:
                k_shape = key.shape
                k_view = key.reshape(num_tokens, -1, head_size)
                k_rot = k_view[..., rope_dim_offset : rope_dim_offset + rotary_dim]
                k_rot_out = ApplyRotaryEmb.forward_static(k_rot, cos, sin, is_neox)

                if k_view.data_ptr() == key.data_ptr():
                    k_view[..., rope_dim_offset : rope_dim_offset + rotary_dim] = k_rot_out
                else:
                    k_pass_pre = k_view[..., :rope_dim_offset]
                    k_pass_post = k_view[..., rope_dim_offset + rotary_dim :]
                    k_full = torch.cat([k_pass_pre, k_rot_out, k_pass_post], dim=-1).reshape(k_shape)
                    key.copy_(k_full)

        # 1. 替換全域 ops 進入點
        ops.rotary_embedding = cpu_rotary_embedding

        # 2. 替換 RotaryEmbedding.forward_cpu 方法
        RotaryEmbedding.forward_cpu = RotaryEmbedding.forward_native

        logger.info(
            "[XPUFakeModelRunner] [Hardware Demo] Successfully patched RotaryEmbedding / ops.rotary_embedding for CPU execution."
        )
    except Exception as e:
        logger.warning(
            "[XPUFakeModelRunner] Failed to patch RotaryEmbedding: %s",
            e,
        )


class XPUFakeModelRunner(CPUModelRunner):
    """繼承 vLLM 官方 CPUModelRunner，模擬 Fake XPU 運算執行。"""

    def __init__(self, vllm_config: "VllmConfig", device: torch.device):
        _patch_block_table_for_cpu()
        _patch_rotary_embedding_for_cpu()
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

# 模組載入時立即套用 Patch，確保任何 BlockTable 與 RoPE 建立前已生效
_patch_block_table_for_cpu()
_patch_rotary_embedding_for_cpu()

