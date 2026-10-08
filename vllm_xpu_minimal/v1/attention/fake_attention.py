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

    class XPUFakeAttentionImpl(FlexAttentionImpl):
        """Override KV cache 寫入：FlexAttention 用 CUDA op
        （torch.ops._C_cache_ops.reshape_and_cache_flash），CPU 上不存在。
        改用純 torch 的 scatter 寫入（教學用，效能不追求）。"""

        def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                    output=None, output_scale=None, output_block_scale=None):
            if attn_metadata is None:
                return output

            num_actual_tokens = attn_metadata.num_actual_tokens

            # KV cache 寫入：純 torch 版的 reshape_and_cache_flash
            # kv_cache shape: (2, num_blocks, block_size, num_kv_heads,
            #                  head_size)
            if self.kv_sharing_target_layer_name is None:
                key_cache, value_cache = kv_cache.unbind(0)
                slot_mapping = attn_metadata.slot_mapping
                flat_k = key_cache.view(-1, key_cache.shape[-2],
                                        key_cache.shape[-1])
                flat_v = value_cache.view(-1, value_cache.shape[-2],
                                          value_cache.shape[-1])
                valid = slot_mapping >= 0
                slots = slot_mapping[valid]
                flat_k[slots] = key[valid]
                flat_v[slots] = value[valid]

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
