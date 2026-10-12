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

★ 坑 9 的教訓：先查上游有沒有現成的同類後端！
版本漂移（坑 6）後改繼承 FlexAttentionBackend，但它的 KV 寫入走
CUDA op（torch.ops._C_cache_ops.reshape_and_cache_flash），
要自己 override。改回 CPU 後端原生搭配的 TorchSDPABackend——
但注意：你環境的 vLLM 版本可能沒有這個類別（坑 6 的教訓），
所以這裡用防禦式 import：有就用、沒有就 fallback 到 FlexAttention
+ 純 torch KV 寫入 override。
"""
import logging

import torch

logger = logging.getLogger(__name__)

# 防禦式 import（坑 6 教訓：上游 backend 類別不是穩定介面）
try:
    from vllm.v1.attention.backends.cpu_attn import TorchSDPABackend

    _HAS_TORCH_SDPA = True
except ImportError:
    _HAS_TORCH_SDPA = False

if _HAS_TORCH_SDPA:

    class XPUFakeAttentionBackend(TorchSDPABackend):
        """繼承 CPU 的 SDPA backend（純 torch，與 CPUModelRunner 原生搭配）。

        【真插件必寫的三個靜態方法】（這裡繼承上游，僅註解示範）：

        @staticmethod
        def get_kv_cache_shape(num_blocks, block_size, num_kv_heads,
                               head_size, cache_dtype_str="auto"):
            # TPU: (num_blocks, block_size, num_kv_heads*2, padded_head_size)
            #      head_size 對齊 128（TPU_HEAD_SIZE_ALIGNMENT）
            # RBLN: [2, num_kv_heads, 1, block_size, head_size]
            # GPU:  (2, num_blocks, block_size * num_kv_heads * head_size)
            ...

        @staticmethod
        def get_page_size(vllm_config):
            # TPU 獨有：page size 由 SMEM/SREG 限制反推
            # （max_model_len > 8192 → 16，否則 next_pow2(max_len)//16
            #   夾在 [16,256]）
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

else:
    # Fallback：FlexAttention（純 torch）+ 純 torch KV 寫入 override
    from vllm.v1.attention.backends.flex_attention import (
        FlexAttentionBackend, FlexAttentionImpl)

    try:
        from vllm.v1.attention.backend import AttentionType
    except ImportError:
        try:
            from vllm.attention.backends.abstract import AttentionType
        except ImportError:
            AttentionType = None

    def _register_cpu_cache_ops() -> None:
        """坑 14：註冊純 PyTorch 版 _C_cache_ops.reshape_and_cache_flash。

        在純 CPU / 無 CUDA 擴展環境下，torch.ops._C_cache_ops 不存在。
        使用 torch.library 註冊純 PyTorch fallback。
        注意：必須使用 4D 切片索引進行真實 in-place 寫入，禁止用 .reshape()
        （non-contiguous tensor 的 reshape 會產生 copy，導致寫入丟失）。
        """
        try:
            def cpu_reshape_and_cache_flash(
                key: torch.Tensor,
                value: torch.Tensor,
                key_cache: torch.Tensor,
                value_cache: torch.Tensor,
                slot_mapping: torch.Tensor,
                kv_cache_dtype: str = "auto",
                k_scale: float = 1.0,
                v_scale: float = 1.0,
            ) -> None:
                # key_cache shape: (num_blocks, block_size, num_kv_heads, head_size)
                num_blocks = key_cache.shape[0]
                block_size = key_cache.shape[1]
                num_kv_heads = key_cache.shape[2]
                head_size = key_cache.shape[3]

                slot_mapping = slot_mapping.flatten()
                valid = (slot_mapping >= 0) & (slot_mapping < num_blocks * block_size)
                if valid.any():
                    valid_slots = slot_mapping[valid].to(torch.int64)
                    b_idx = valid_slots // block_size
                    b_off = valid_slots % block_size

                    k_val = key[valid].reshape(-1, num_kv_heads, head_size).to(key_cache.dtype)
                    v_val = value[valid].reshape(-1, num_kv_heads, head_size).to(value_cache.dtype)

                    key_cache[b_idx, b_off, :, :] = k_val
                    value_cache[b_idx, b_off, :, :] = v_val

            lib = torch.library.Library("_C_cache_ops", "FRAGMENT")
            lib.define(
                "reshape_and_cache_flash(Tensor key, Tensor value, Tensor(a!) key_cache, "
                "Tensor(b!) value_cache, Tensor slot_mapping, str kv_cache_dtype, "
                "float? k_scale, float? v_scale) -> ()"
            )
            lib.impl("reshape_and_cache_flash", cpu_reshape_and_cache_flash, "CPU")
            logger.info(
                "[XPUFakeAttention] Successfully registered pure PyTorch _C_cache_ops::reshape_and_cache_flash."
            )
        except Exception as e:
            logger.debug("[XPUFakeAttention] _C_cache_ops registration note: %s", e)

    _register_cpu_cache_ops()

    class XPUFakeAttentionImpl(FlexAttentionImpl):
        """Override KV cache 寫入：FlexAttention 用 CUDA op
        （torch.ops._C_cache_ops.reshape_and_cache_flash），CPU 上不存在。
        改用純 torch 4D 切片 in-place 寫入。"""

        def do_kv_cache_update(
            self,
            layer: torch.nn.Module,
            key: torch.Tensor,
            value: torch.Tensor,
            kv_cache: torch.Tensor,
            slot_mapping: torch.Tensor,
        ) -> None:
            if AttentionType is not None and getattr(self, "attn_type", None) == AttentionType.ENCODER_ONLY:
                return
            if "ENCODER_ONLY" in str(getattr(self, "attn_type", "")):
                return

            # kv_cache shape: (num_blocks, num_kv_heads, block_size, 2 * head_size)
            num_blocks = kv_cache.shape[0]
            num_kv_heads = kv_cache.shape[1]
            block_size = kv_cache.shape[2]
            head_size = kv_cache.shape[3] // 2

            slot_mapping = slot_mapping.flatten()
            valid = (slot_mapping >= 0) & (slot_mapping < num_blocks * block_size)
            if valid.any():
                valid_slots = slot_mapping[valid].to(torch.int64)
                b_idx = valid_slots // block_size
                b_off = valid_slots % block_size

                k_val = key[valid].reshape(-1, num_kv_heads, head_size).to(kv_cache.dtype)
                v_val = value[valid].reshape(-1, num_kv_heads, head_size).to(kv_cache.dtype)

                kv_cache[b_idx, :, b_off, :head_size] = k_val
                kv_cache[b_idx, :, b_off, head_size:] = v_val

        def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                    output=None, output_scale=None, output_block_scale=None):
            return super().forward(layer, query, key, value, kv_cache,
                                   attn_metadata, output, output_scale,
                                   output_block_scale)

    class XPUFakeAttentionBackend(FlexAttentionBackend):

        @staticmethod
        def get_impl_cls():
            return XPUFakeAttentionImpl

    logger.warning(
        "[XPUFakeAttention] TorchSDPABackend not found in this vLLM "
        "version (坑 6: version drift); falling back to FlexAttention.")

