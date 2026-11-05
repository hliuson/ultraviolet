"""perf flag 'fz': fused softcap + categorical sample + behaviour log-prob for the decode step (one kernel).

Replaces the compiled sampling chain (fp32 softcapped logits written as `out_logits`, one Philox uniform + two logs
per vocab entry for Gumbel-max, argmax, then logsumexp + gather for the behaviour log-prob) with one Triton kernel
per row that reads the fp16 head output (B, V) and one uniform per row (hierarchical inverse-CDF draw):
  pass 1: z = a * sigmoid((x + b) / c) in (0, a) (softcap, fp32), e = exp((z - a) / T) in (exp(-a / T), 1] (the
          softcap bounds z, so no max pass), summed per BLOCK entries;
  pick the block whose cumulative mass crosses u' * S, u' = 1 - u in (0, 1];
  pass 2: re-read only that block (cache-hot), cumsum inside it, take the first entry whose cumsum >= the target.
The token is an exact draw from softmax(z / T) up to fp32 summation rounding (a different random stream from
Gumbel-max, so tokens are not bitwise those of the default path); log p(token) = (z_tok - a) / T - log S is the same
quantity the default path records (log_softmax(z / T)[token]), to fp32 rounding.
Plain Triton (loads, reductions, cumsum; no tl.dot), so it runs on sm75 Turing and newer. `fused_sample_ok` says
when it applies (CUDA, Triton importable, V % BLOCK == 0, T > 0 with a / T small enough for fp32); otherwise the
caller keeps the default sampler.
"""
from __future__ import annotations

import torch
from torch import Tensor

BLOCK = 1024

try:  # Triton is optional (CPU-only installs): the reference below still works
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None

if triton is not None:
    @triton.jit
    def _fz_sample_kernel(X, U, Out, LogP, stride, inv_t, A: tl.constexpr, Bc: tl.constexpr, C: tl.constexpr,
                          V: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        NB: tl.constexpr = V // BLOCK
        xrow = X + row.to(tl.int64) * stride
        offs = tl.arange(0, BLOCK)
        bidx = tl.arange(0, NB)
        sums = tl.zeros([NB], tl.float32)
        for i in tl.static_range(NB):
            x = tl.load(xrow + i * BLOCK + offs).to(tl.float32)
            z = A * tl.sigmoid((x + Bc) / C)
            e = tl.exp((z - A) * inv_t)
            sums = tl.where(bidx == i, tl.sum(e, 0), sums)
        S = tl.sum(sums, 0)
        u = 1.0 - tl.load(U + row)  # (0, 1]
        target = u * S
        csum = tl.cumsum(sums, 0)
        blk = tl.minimum(tl.sum((csum < target).to(tl.int32), 0), NB - 1)  # first block with csum >= target
        before = tl.sum(tl.where(bidx < blk, sums, 0.0), 0)
        x = tl.load(xrow + blk * BLOCK + offs).to(tl.float32)
        z = A * tl.sigmoid((x + Bc) / C)
        e = tl.exp((z - A) * inv_t)
        c = tl.cumsum(e, 0) + before
        j = tl.minimum(tl.sum((c < target).to(tl.int32), 0), BLOCK - 1)
        tl.store(Out + row, (blk * BLOCK + j).to(tl.int64))
        zj = tl.sum(tl.where(offs == j, z, 0.0), 0)
        tl.store(LogP + row, (zj - A) * inv_t - tl.log(S))


def fused_sample_ok(x_or_dev, V: int, softcap, temperature: float, block: int = BLOCK) -> bool:
    dev = x_or_dev.device if isinstance(x_or_dev, Tensor) else torch.device(x_or_dev)
    a = float(softcap[0])
    return (triton is not None and dev.type == "cuda" and V % block == 0 and V // block <= 64 and temperature > 0
            and a / temperature <= 80.0)


def fused_sample(x: Tensor, u: Tensor, softcap, temperature: float = 1.0, block: int = BLOCK) -> tuple[Tensor, Tensor]:
    """x (B, V) fp16/fp32 raw head output (before the softcap), u (B,) fp32 uniforms in [0, 1) -> (tokens int64 (B,),
    log softmax(softcap(x) / T)[token] fp32 (B,)). Graph-capturable (no host syncs)."""
    B, V = x.shape
    assert x.stride(1) == 1 and V % block == 0
    out = torch.empty(B, dtype=torch.int64, device=x.device)
    lp = torch.empty(B, dtype=torch.float32, device=x.device)
    a, b, c = (float(v) for v in softcap)
    _fz_sample_kernel[(B,)](x, u, out, lp, x.stride(0), 1.0 / float(temperature), A=a, Bc=b, C=c, V=V, BLOCK=block,
                            num_warps=8)
    return out, lp


def fused_sample_ref(x: Tensor, u: Tensor, softcap, temperature: float = 1.0) -> tuple[Tensor, Tensor]:
    """Torch reference of fused_sample (flat fp32 cumsum instead of the per-block one; any device)."""
    a, b, c = (float(v) for v in softcap)
    z = a * torch.sigmoid((x.float() + b) / c)
    e = torch.exp((z - a) / temperature)
    cs = e.cumsum(-1)
    S = cs[:, -1]
    target = (1.0 - u.float()) * S
    tok = torch.searchsorted(cs, target[:, None]).squeeze(1).clamp_(max=x.size(1) - 1)
    lp = (z.gather(1, tok[:, None])[:, 0] - a) / temperature - torch.log(S)
    return tok, lp
