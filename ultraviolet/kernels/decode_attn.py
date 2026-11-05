"""Split-K single-query decode attention (FlashDecoding style) in Triton, tuned for RTX 2080 Ti (sm75, fp16).

One program (a single warp) handles (row b, kv head h, split s) for all G = H/K query heads of the group, so every K/V
byte is read once per group. Each row's valid keys are the absolute positions
[lo_b, pos_b] with lo_b = max(start_b, pos_b - window + 1) (window 0: no window); a row's range is cut into
`num_splits` equal chunks, so ragged rows only read what they need (inactive splits exit immediately). Position t
lives in cache slot t % S when `ring` (sliding-window ring buffer of S slots), else slot t.

int8 caches: K/V int8 (B, K, S, hd) with fp16 per-token-per-head scales (B, K, S). Dequant is fused: the K scale
multiplies the QK^T columns, the V scale multiplies the probabilities before P@V.

sm75: Triton emits no tensor-core MMA for Turing, so the kernel avoids tl.dot (see _split_kernel).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch import Tensor

LOG2E = 1.4426950408889634


@triton.jit
def _split_kernel(Q, Kc, Vc, Ks, Vs, Pos, Start, Out, Ob, Mb, Lb,
                  KH, S, window, scale_log2,
                  sqb, sqh, skb, skh, sks, svb, svh, svs, ssb, ssh, stb, sth, sob, soh,
                  G: tl.constexpr, GP: tl.constexpr, HD: tl.constexpr, BLOCK_N: tl.constexpr,
                  SPLITS: tl.constexpr, RING: tl.constexpr, INT8: tl.constexpr):
    # Turing note: Triton does not emit tensor-core MMA on sm75 (tl.dot lowers to FMA with heavy layout shuffles,
    # ~100 GB/s here), so QK^T and PV are written as per-query-head multiply-reduce over a small key tile, run by a
    # single warp (no cross-warp reductions). Per-head softmax state lives in rows of (GP, .) tensors.
    pid = tl.program_id(0)
    sid = tl.program_id(1)
    b = pid // KH
    h = pid % KH
    p = tl.load(Pos + b).to(tl.int32)
    lo = tl.maximum(tl.load(Start + b).to(tl.int32), 0)
    if window > 0:
        lo = tl.maximum(lo, p - window + 1)
    n = p + 1 - lo
    chunk = tl.cdiv(tl.cdiv(n, SPLITS), BLOCK_N) * BLOCK_N
    s0 = lo + sid * chunk
    s1 = tl.minimum(s0 + chunk, p + 1)

    gi = tl.arange(0, GP)
    d = tl.arange(0, HD)
    gm = gi < G
    m_i = tl.full([GP], float("-inf"), tl.float32)
    l_i = tl.zeros([GP], tl.float32)
    acc = tl.zeros([GP, HD], tl.float32)
    b64 = b.to(tl.int64)
    qbase = Q + b64 * sqb + h * G * sqh
    kbase = Kc + b64 * skb + h * skh
    vbase = Vc + b64 * svb + h * svh
    for start in range(s0, s1, BLOCK_N):
        t = start + tl.arange(0, BLOCK_N)
        tm = t < s1
        if RING:
            slot = t % S
        else:
            slot = t
        k = tl.load(kbase + slot[:, None] * sks + d[None, :], mask=tm[:, None], other=0).to(tl.float32)
        v = tl.load(vbase + slot[:, None] * svs + d[None, :], mask=tm[:, None], other=0).to(tl.float32)
        qk = tl.full([GP, BLOCK_N], float("-inf"), tl.float32)
        for g in tl.static_range(G):
            qg = tl.load(qbase + g * sqh + d).to(tl.float32)
            qk = tl.where(gi[:, None] == g, tl.sum(k * qg[None, :], 1)[None, :], qk)
        if INT8:
            ksc = tl.load(Ks + b64 * ssb + h * ssh + slot, mask=tm, other=0.0).to(tl.float32)
            qk = qk * ksc[None, :]
        qk = tl.where(tm[None, :], qk * scale_log2, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp2(m_i - m_new)
        pr = tl.exp2(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(pr, 1)
        if INT8:
            vsc = tl.load(Vs + b64 * stb + h * sth + slot, mask=tm, other=0.0).to(tl.float32)
            pr = pr * vsc[None, :]
        acc = acc * alpha[:, None]
        for g in tl.static_range(G):
            pg = tl.sum(tl.where(gi[:, None] == g, pr, 0.0), 0)
            acc = tl.where(gi[:, None] == g, acc + tl.sum(pg[:, None] * v, 0)[None, :], acc)
        m_i = m_new
    if SPLITS == 1:
        o = acc / l_i[:, None]
        tl.store(Out + b64 * sob + (h * G + gi)[:, None] * soh + d[None, :], o.to(Out.dtype.element_ty),
                 mask=gm[:, None])
    else:
        r = (pid * SPLITS + sid) * G + gi  # partial buffers (B*K, SPLITS, G[, HD])
        tl.store(Ob + r[:, None] * HD + d[None, :], acc, mask=gm[:, None])
        tl.store(Mb + r, m_i, mask=gm)
        tl.store(Lb + r, l_i, mask=gm)


@triton.jit
def _combine_kernel(Out, Ob, Mb, Lb, KH, sob, soh,
                    G: tl.constexpr, HD: tl.constexpr, SPLITS: tl.constexpr, SP2: tl.constexpr):
    pid = tl.program_id(0)  # (b * KH + h) * G + g
    bk = pid // G
    gi = pid % G
    b = bk // KH
    h = bk % KH
    s = tl.arange(0, SP2)
    d = tl.arange(0, HD)
    sm = s < SPLITS
    r = (bk * SPLITS + s) * G + gi
    m = tl.load(Mb + r, mask=sm, other=float("-inf"))
    l = tl.load(Lb + r, mask=sm, other=0.0)
    mx = tl.max(m, 0)
    w = tl.where(m == float("-inf"), 0.0, tl.exp2(m - mx))
    o = tl.load(Ob + r[:, None] * HD + d[None, :], mask=sm[:, None], other=0.0)
    num = tl.sum(o * w[:, None], 0)
    den = tl.sum(l * w, 0)
    tl.store(Out + b * sob + (h * G + gi) * soh + d, (num / den).to(Out.dtype.element_ty))


_N_SM: dict = {}


def default_splits(n_bk: int, ctx: int, device=None, min_chunk: int = 128, waves: int = 8) -> int:
    """Splits per (row, kv head): enough programs (warps) for ~`waves` per SM, each chunk >= min_chunk keys.
    2080 Ti sweeps: Triton waves=8/min_chunk=128; CUDA waves=4/min_chunk=64."""
    dev = torch.device(device or "cuda")
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    if idx not in _N_SM:
        _N_SM[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    want = -(-waves * _N_SM[idx] // max(n_bk, 1))
    return int(max(1, min(want, max(1, ctx // min_chunk), 64)))


def default_block_n(hd: int, int8: bool, n_bk: int) -> int:
    """Keys per loop iteration (2080 Ti sweep): ~1-2 KB K tile."""
    if hd >= 128:
        return 16 if int8 else 8
    if int8:
        return 32
    return 32 if n_bk >= 128 else 16


def _launch(q, k_cache, v_cache, pos, starts, scale, window, ring, k_scale, v_scale, num_splits, block_n,
            num_warps, out):
    B, H, hd = q.shape
    KH, S = k_cache.shape[1], k_cache.shape[2]
    G = H // KH
    GP = max(2, triton.next_power_of_2(G))
    int8 = k_cache.dtype == torch.int8
    assert q.stride(-1) == 1 and k_cache.stride(-1) == 1 and v_cache.stride(-1) == 1
    if int8:
        assert k_scale is not None and v_scale is not None and k_scale.stride(-1) == 1 and v_scale.stride(-1) == 1
        ks, vs = k_scale, v_scale
        ssb, ssh, stb, sth = ks.stride(0), ks.stride(1), vs.stride(0), vs.stride(1)
    else:
        ks, vs = k_cache, v_cache  # unused
        ssb = ssh = stb = sth = 0
    if num_splits > 1:
        ob = torch.empty(B * KH * num_splits * G, hd, device=q.device, dtype=torch.float32)
        mb = torch.empty(B * KH * num_splits * G, device=q.device, dtype=torch.float32)
        lb = torch.empty_like(mb)
    else:
        ob = mb = lb = out
    _split_kernel[(B * KH, num_splits)](
        q, k_cache, v_cache, ks, vs, pos, starts, out, ob, mb, lb,
        KH, S, window, scale * LOG2E,
        q.stride(0), q.stride(1), k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), ssb, ssh, stb, sth, out.stride(0), out.stride(1),
        G=G, GP=GP, HD=hd, BLOCK_N=block_n, SPLITS=num_splits, RING=ring, INT8=int8, num_warps=num_warps)
    if num_splits > 1:
        _combine_kernel[(B * KH * G,)](out, ob, mb, lb, KH, out.stride(0), out.stride(1), G=G, HD=hd,
                                       SPLITS=num_splits, SP2=triton.next_power_of_2(num_splits), num_warps=4)
    return out


@torch.library.custom_op("mathk::decode_attn", mutates_args=())
def _decode_attn_op(q: Tensor, k_cache: Tensor, v_cache: Tensor, pos: Tensor, starts: Tensor, scale: float,
                    window: int, ring: bool, k_scale: Tensor | None, v_scale: Tensor | None, num_splits: int,
                    block_n: int, num_warps: int, backend: str) -> Tensor:
    out = torch.empty(q.shape, device=q.device, dtype=q.dtype)
    if backend == "cuda":
        from . import decode_attn_cuda
        return decode_attn_cuda.launch(q, k_cache, v_cache, pos, starts, scale, window, ring, k_scale, v_scale,
                                       num_splits, out)
    return _launch(q, k_cache, v_cache, pos, starts, scale, window, ring, k_scale, v_scale, num_splits, block_n,
                   num_warps, out)


@_decode_attn_op.register_fake
def _(q, k_cache, v_cache, pos, starts, scale, window, ring, k_scale, v_scale, num_splits, block_n, num_warps,
      backend):
    return torch.empty_like(q)


def decode_attention(q: Tensor, k_cache: Tensor, v_cache: Tensor, pos: Tensor, starts: Tensor, scale: float, *,
                     window: int = 0, ring: bool | None = None, k_scale: Tensor | None = None,
                     v_scale: Tensor | None = None, ctx_hint: int | None = None, num_splits: int | None = None,
                     block_n: int | None = None, num_warps: int = 1, backend: str = "auto") -> Tensor:
    """Single-query attention for every row of a static KV cache.

    q        (B, H, hd) fp16 (any (B, 1, H, hd) view is accepted); H = K * G, query head h*G+g uses kv head h.
    k_cache, v_cache  (B, K, S, hd) fp16, or int8 with k_scale / v_scale (B, K, S) fp16.
    pos      (B,) int: position of the current (last) key, already written to the cache. Keys attended:
             positions max(starts[b], pos[b] - window + 1, 0) .. pos[b] (window 0 = unlimited).
    starts   (B,) int: first valid (non-pad) position of each row.
    ring     slot of position t is t % S (default: window > 0); else slot t (needs pos < S).
    ctx_hint static upper bound on keys per row (e.g. g_len, or the window) for choosing num_splits.
    backend  "cuda" (NVRTC tensor-core kernel, sm75+, hd 64/128, G <= 8), "triton", or "auto" (cuda when supported).
    Returns (B, H, hd) in q.dtype. Graph-capturable; registered as a torch custom op (opaque to torch.compile).
    """
    if q.dim() == 4:
        q = q.reshape(q.size(0), q.size(-2), q.size(-1))
    B, H, hd = q.shape
    KH, S = k_cache.shape[1], k_cache.shape[2]
    if ring is None:
        ring = window > 0
    if backend == "auto":
        from . import decode_attn_cuda
        backend = "cuda" if decode_attn_cuda.supported(q, k_cache) else "triton"
    if num_splits is None:
        ctx = ctx_hint or S
        if window > 0:
            ctx = min(ctx, window)
        if backend == "cuda":
            num_splits = default_splits(B * KH, ctx, q.device, min_chunk=64, waves=4)
        else:
            num_splits = default_splits(B * KH, ctx, q.device)
    if block_n is None:
        block_n = default_block_n(hd, k_cache.dtype == torch.int8, B * KH)
    q = q.contiguous() if q.stride(-1) != 1 else q
    return _decode_attn_op(q, k_cache, v_cache, pos, starts, float(scale), int(window), bool(ring), k_scale,
                           v_scale, int(num_splits), int(block_n), int(num_warps), backend)


def decode_attention_ref(q: Tensor, k_cache: Tensor, v_cache: Tensor, pos: Tensor, starts: Tensor, scale: float, *,
                         window: int = 0, ring: bool | None = None, k_scale: Tensor | None = None,
                         v_scale: Tensor | None = None) -> Tensor:
    """fp32 PyTorch reference with the same semantics (dequantizes int8 caches)."""
    if q.dim() == 4:
        q = q.reshape(q.size(0), q.size(-2), q.size(-1))
    B, H, hd = q.shape
    KH, S = k_cache.shape[1], k_cache.shape[2]
    G = H // KH
    if ring is None:
        ring = window > 0
    kf, vf = k_cache.float(), v_cache.float()
    if k_cache.dtype == torch.int8:
        kf = kf * k_scale.float().unsqueeze(-1)
        vf = vf * v_scale.float().unsqueeze(-1)
    dev = q.device
    slot = torch.arange(S, device=dev)[None, :]
    p = pos.long()[:, None]
    if ring:
        tpos = p - torch.remainder(p - slot, S)  # absolute position held in each slot
    else:
        tpos = slot.expand(B, S)
    lo = torch.maximum(starts.long()[:, None], torch.zeros_like(p))
    if window > 0:
        lo = torch.maximum(lo, p - window + 1)
    valid = (tpos >= lo) & (tpos <= p)  # (B, S)
    qf = q.float().view(B, KH, G, hd)
    s = torch.einsum("bkgd,bksd->bkgs", qf, kf) * scale
    s = s.masked_fill(~valid[:, None, None, :], float("-inf"))
    w = torch.softmax(s, -1)
    return torch.einsum("bkgs,bksd->bkgd", w, vf).reshape(B, H, hd).to(q.dtype)
