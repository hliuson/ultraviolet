"""int8 KV cache: symmetric per-token-per-head quantization (absmax / 127) with fp16 scales.

Cache layout (matches DecodeState): values (B, K, S, hd) int8, scales (B, K, S) fp16. Flat index of slot s of row b,
kv head h is (b * K + h) * S + s for both tensors, so the model's existing `index_copy_` write indices are reused.
"""
from __future__ import annotations

import torch
from torch import Tensor


def quantize_kv(x: Tensor) -> tuple[Tensor, Tensor]:
    """x (..., hd) float -> (int8 (..., hd), fp16 scale (...)). Quantizes with the fp16-rounded scale so that
    dequant(q, s) uses exactly the scale the rounding assumed."""
    xf = x.float()
    s16 = (xf.abs().amax(-1) / 127.0).to(torch.float16)
    sf = s16.float().clamp_min(1e-30)  # all-zero rows -> q = 0
    q = torch.round(xf / sf.unsqueeze(-1)).clamp_(-127, 127).to(torch.int8)
    return q, s16


def dequantize_kv(q: Tensor, s: Tensor, dtype=torch.float16) -> Tensor:
    return (q.float() * s.float().unsqueeze(-1)).to(dtype)


def kv_write(cache: Tensor, scale: Tensor | None, flat_idx: Tensor, x: Tensor) -> None:
    """Decode-time write of one token per (row, kv head). cache (B, K, S, hd) fp16 or int8; scale (B, K, S) fp16 or
    None; flat_idx (B*K,) = (b*K + h)*S + slot; x (B*K, hd) or anything reshapeable to it."""
    hd = cache.size(-1)
    x = x.reshape(-1, hd)
    if scale is None:
        cache.view(-1, hd).index_copy_(0, flat_idx, x.to(cache.dtype))
    else:
        q, s = quantize_kv(x)
        cache.view(-1, hd).index_copy_(0, flat_idx, q)
        scale.view(-1).index_copy_(0, flat_idx, s)


def kv_write_rows(cache: Tensor, scale: Tensor | None, new: Tensor, slots: Tensor | slice,
                  rows: Tensor | None = None) -> None:
    """Prefill-time write. new (n, K, T', hd) float goes to cache[rows][:, :, slots] (rows None: all rows, n == B).
    Quantizes when scale is given."""
    if scale is None:
        vals = [(cache, new.to(cache.dtype))]
    else:
        q, s = quantize_kv(new)
        vals = [(cache, q), (scale, s)]
    for dst, src in vals:
        if rows is None:
            dst[:, :, slots] = src
        else:
            blk = dst[rows]
            blk[:, :, slots] = src
            dst[rows] = blk
