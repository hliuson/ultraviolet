"""Triton decode kernels for the math LM (sm75-safe: fp16 tensor cores, no bf16, no cp.async needed).

- `decode_attention`: split-K (FlashDecoding-style) single-query attention over a static KV cache, MQA/GQA, ragged
  per-row positions, full-context or ring-buffer sliding-window caches, fp16 or int8 (per-token-per-head fp16
  scales, dequant fused).
- `quantize_kv` / `kv_write`: int8 quantization and cache writes.
- `fast_decode`: drop-in DecodeState / decode_step / prefill replacements that use the above.
"""
from .decode_attn import decode_attention, decode_attention_ref, default_splits  # noqa: F401
from .kv_quant import dequantize_kv, kv_write, kv_write_rows, quantize_kv  # noqa: F401
