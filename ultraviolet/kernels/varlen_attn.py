"""Varlen (packed) causal attention with an optional sliding window, for training on sm75.

Wraps PyTorch's CUTLASS memory-efficient attention (`aten._efficient_attention_forward/backward`, tensor cores on
Turing) with `cu_seqlens` (packed sequences, no padding) and `window_size` (the kernel skips key blocks outside the
window; same convention as MathLM: query i sees keys i - window < j <= i). MQA/GQA K/V are broadcast to the H query
heads (a stride-0 expand when K == 1, no copy) and their gradients summed back.

Compared with the default training path (SDPA with a dense (T, T) boolean window mask on local layers, K/V
repeat_interleave'd to H heads) the outputs are identical up to the kernel's accumulation order, while local layers
read only the keys in the window (docs/perf_opt.md).
"""
from __future__ import annotations

import torch
from torch import Tensor


def _expand_kv(x: Tensor, H: int) -> Tensor:
    """(1, N, K, hd) -> (1, N, H, hd)."""
    K = x.size(2)
    if K == H:
        return x
    if K == 1:
        return x.expand(x.size(0), x.size(1), H, x.size(3))
    return x.repeat_interleave(H // K, dim=2)


def _reduce_kv(g: Tensor, K: int) -> Tensor:
    B, N, H, hd = g.shape
    if K == H:
        return g
    return g.view(B, N, K, H // K, hd).sum(3)


# Opaque custom ops (torch.compile treats them as single nodes): the aten backward op's fake/meta registration
# rejects the keyword-only `scale` / `window_size` arguments in this torch build, which graph-breaks a compiled
# trunk; the forward's meta function works and gives the logsumexp / philox shapes.
@torch.library.custom_op("mathk::varlen_attn_fwd", mutates_args=())
def _fwd(q: Tensor, k: Tensor, v: Tensor, cu: Tensor, max_seqlen: int, window: int, scale: float
         ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    H = q.size(2)
    o, lse, seed, off, _, _ = torch.ops.aten._efficient_attention_forward(
        q, _expand_kv(k, H), _expand_kv(v, H), None, cu, cu, max_seqlen, max_seqlen, 0.0, 1, True, scale=scale,
        window_size=window if window > 0 else None)
    return o, lse, seed, off


@_fwd.register_fake
def _(q, k, v, cu, max_seqlen, window, scale):
    H = q.size(2)
    o, lse, seed, off, _, _ = torch.ops.aten._efficient_attention_forward(
        q, _expand_kv(k, H), _expand_kv(v, H), None, cu, cu, max_seqlen, max_seqlen, 0.0, 1, True, scale=scale)
    return o, lse, seed, off


@torch.library.custom_op("mathk::varlen_attn_bwd", mutates_args=())
def _bwd(go: Tensor, q: Tensor, k: Tensor, v: Tensor, o: Tensor, lse: Tensor, cu: Tensor, seed: Tensor, off: Tensor,
         max_seqlen: int, window: int, scale: float) -> tuple[Tensor, Tensor, Tensor]:
    H, K = q.size(2), k.size(2)
    dq, dk, dv, _ = torch.ops.aten._efficient_attention_backward(
        go.contiguous(), q, _expand_kv(k, H), _expand_kv(v, H), None, o, cu, cu, max_seqlen, max_seqlen, lse, 0.0,
        seed, off, 1, False, scale=scale, window_size=window if window > 0 else None)
    return dq, _reduce_kv(dk, K), _reduce_kv(dv, K)


@_bwd.register_fake
def _(go, q, k, v, o, lse, cu, seed, off, max_seqlen, window, scale):
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _setup(ctx, inputs, output):
    q, k, v, cu, max_seqlen, window, scale = inputs
    o, lse, seed, off = output
    ctx.save_for_backward(q, k, v, o, lse, cu, seed, off)
    ctx.ms, ctx.window, ctx.scale = max_seqlen, window, scale


def _backward(ctx, go, glse, gseed, goff):
    q, k, v, o, lse, cu, seed, off = ctx.saved_tensors
    dq, dk, dv = _bwd(go, q, k, v, o, lse, cu, seed, off, ctx.ms, ctx.window, ctx.scale)
    return dq, dk, dv, None, None, None, None


_fwd.register_autograd(_backward, setup_context=_setup)


def varlen_attention(q: Tensor, k: Tensor, v: Tensor, cu_seqlens: Tensor, max_seqlen: int, scale: float,
                     window: int | None = None) -> Tensor:
    """q (1, N, H, hd), k/v (1, N, K, hd) fp16, packed along N; cu_seqlens (S+1,) int32 segment boundaries
    (cu[0] = 0, cu[-1] = N, every segment <= max_seqlen). Causal within each segment, keys restricted to the last
    `window` positions (None / 0: no window). Returns (1, N, H, hd)."""
    return _fwd(q.contiguous(), k, v, cu_seqlens, int(max_seqlen), int(window or 0), float(scale))[0]


def varlen_attention_ref(q, k, v, cu_seqlens, scale, window=None):
    """fp32 reference (loops over segments)."""
    H, K = q.size(2), k.size(2)
    outs = []
    cu = cu_seqlens.tolist()
    for a, b in zip(cu[:-1], cu[1:]):
        L = b - a
        if L == 0:
            continue
        qq = q[0, a:b].float().transpose(0, 1)
        kk = _expand_kv(k[:, a:b], H)[0].float().transpose(0, 1)
        vv = _expand_kv(v[:, a:b], H)[0].float().transpose(0, 1)
        s = qq @ kk.transpose(1, 2) * scale
        i = torch.arange(L, device=q.device)
        d = i[:, None] - i[None, :]
        m = d >= 0
        if window:
            m = m & (d < window)
        s = s.masked_fill(~m, float("-inf"))
        outs.append((s.softmax(-1) @ vv).transpose(0, 1))
    return torch.cat(outs)[None]
