"""Reference integration of the Triton decode kernel (+ optional int8 KV cache) with MathLM, without editing model.py.

`KernelDecodeState` is a DecodeState whose caches are fp16 or int8 (+ fp16 scales); `decode_step_fast` and
`prefill_fast` mirror MathLM.decode_step / MathLM.prefill (prefill chunking, conv/GDN mixer layers) line for line
except for the cache write and the attention call. `patch_model(model, kv_dtype)` binds them onto a model instance so
generate / generate_continuous / DecodeRunner (graphs + compile) use them unchanged.
"""
from __future__ import annotations

import types

import numpy as np
import torch
from torch import Tensor

from ..model import DecodeState, MathConfig
from .decode_attn import decode_attention
from .kv_quant import kv_write, kv_write_rows, quantize_kv


class KernelDecodeState(DecodeState):
    """Same fields as DecodeState (k, v lists of (B, K, S, hd) or None for non-attention layers; mix; pos, starts,
    host_pos) plus k_scale / v_scale lists of (B, K, S) fp16 when kv_dtype == "int8" (else None entries). `dtype`
    stays the compute dtype (fp16). Non-attention mixer state (st.mix) is filled by the patched new_state."""

    def __init__(self, cfg: MathConfig, B: int, max_len: int, dtype, device, kv_dtype: str = "fp16"):
        self.cfg, self.B, self.max_len, self.dtype = cfg, B, max_len, dtype
        self.kv_dtype = kv_dtype
        self.local_len = min(cfg.window, max_len)
        cdt = torch.int8 if kv_dtype == "int8" else dtype
        self.k, self.v, self.k_scale, self.v_scale = [], [], [], []
        self.mix: dict[int, dict] = {}
        for l in range(cfg.n_layers):
            if _mixer(cfg, l) != "attn":
                for lst in (self.k, self.v, self.k_scale, self.v_scale):
                    lst.append(None)
                continue
            S = max_len if cfg.is_global(l) else self.local_len
            shp = (B, cfg.n_kv_heads, S, cfg.head_dim)
            self.k.append(torch.zeros(shp, dtype=cdt, device=device))
            self.v.append(torch.zeros(shp, dtype=cdt, device=device))
            if kv_dtype == "int8":
                self.k_scale.append(torch.zeros(shp[:3], dtype=torch.float16, device=device))
                self.v_scale.append(torch.zeros(shp[:3], dtype=torch.float16, device=device))
            else:
                self.k_scale.append(None)
                self.v_scale.append(None)
        self.pos = torch.zeros(B, dtype=torch.long, device=device)
        self.starts = torch.zeros(B, dtype=torch.long, device=device)
        self.host_pos = np.zeros(B, dtype=np.int64)
        # hashed bigram prev-token buffer (as DecodeState)
        self.prev = torch.full((B,), -1, dtype=torch.long, device=device) if getattr(cfg, "bigram_rows", 0) else None

    def nbytes(self) -> int:
        ts = [t for t in self.k + self.v + self.k_scale + self.v_scale if t is not None]
        ts += [t for d in self.mix.values() for t in d.values()]
        return sum(t.numel() * t.element_size() for t in ts)


def _mixer(cfg: MathConfig, l: int) -> str:
    return cfg.mixer(l) if hasattr(cfg, "mixer") else "attn"


def kv_bytes_per_seq(cfg: MathConfig, ctx: int, kv_dtype: str = "fp16") -> int:
    per_pos = 2 * cfg.n_kv_heads * (cfg.head_dim * 2 if kv_dtype == "fp16" else cfg.head_dim + 2)
    return sum(per_pos * (ctx if cfg.is_global(l) else min(ctx, cfg.window)) for l in range(cfg.n_layers)
               if _mixer(cfg, l) == "attn")


@torch.no_grad()
def prefill_fast(self, idx: Tensor, starts: Tensor, st: KernelDecodeState, rows: Tensor | None = None) -> Tensor:
    """MathLM.prefill with (optionally quantizing) cache writes. Attention itself stays on SDPA."""
    n, T = idx.shape
    assert T <= st.max_len
    chunk = max(1, getattr(self, "prefill_chunk_tokens", 1 << 62) // T)
    if n > chunk:  # bound activation memory, as MathLM.prefill does
        if rows is None:
            assert n == st.B
            rows = torch.arange(n, device=idx.device)
        return torch.cat([prefill_fast(self, idx[i:i + chunk], starts[i:i + chunk], st, rows[i:i + chunk])
                          for i in range(0, n, chunk)])
    if getattr(self, "prefill_packed", False):
        return _prefill_packed(self, idx, starts, st, rows)
    key_valid = torch.arange(T, device=idx.device)[None, :] >= starts[:, None]
    h, kvs = self.trunk(idx, key_valid=key_valid, collect_kv=True)
    full = rows is None
    if full:
        assert n == st.B
        rows = torch.arange(n, device=idx.device)
    for l, kv in enumerate(kvs):
        if isinstance(kv, dict):  # recurrent / conv mixer state
            for name, val in kv.items():
                buf = st.mix[l][name]
                if full:
                    buf.copy_(val.to(buf.dtype))
                else:
                    buf[rows] = val.to(buf.dtype)
            continue
        k, v = kv
        kt, vt = k.transpose(1, 2), v.transpose(1, 2)  # (n, K, T, hd)
        for cache, sc, new in ((st.k[l], st.k_scale[l], kt), (st.v[l], st.v_scale[l], vt)):
            S = cache.size(2)
            if self.cfg.is_global(l):
                kv_write_rows(cache, sc, new, slice(0, T), None if full else rows)
            else:
                lo = max(0, T - S)
                slots = torch.arange(lo, T, device=idx.device) % S
                kv_write_rows(cache, sc, new[:, :, lo:], slots, None if full else rows)
    st.starts[rows] = starts
    st.pos[rows] = T
    st.host_pos[rows.cpu().numpy()] = T
    if getattr(st, "prev", None) is not None:
        st.prev[rows] = idx[:, -1]
    return self.logits(h[:, -1])


def _prefill_packed(self, idx: Tensor, starts: Tensor, st: KernelDecodeState, rows: Tensor | None) -> Tensor:
    """Perf flag `model.prefill_packed = True` (docs/perf_opt.md): the same prefill without computing the left
    padding. The real tokens of all rows are packed back to back (1, N) and run through the varlen attention kernel
    (causal per row, sliding window on local layers); token j of row b keeps its left-padded position
    starts[b] + j, so RoPE, the cache slots and st.pos / st.starts are exactly those of the padded prefill. Pad
    slots are not written (they are never read: rows read keys >= start)."""
    n, T = idx.shape
    dev = idx.device
    assert all(b.attn.kind == "attn" for b in self.blocks), "packed prefill needs attention-only mixers"
    ar = torch.arange(T, device=dev)
    valid = ar[None, :] >= starts[:, None]                     # (n, T)
    lens = T - starts
    cu = torch.zeros(n + 1, dtype=torch.int32, device=dev)
    cu[1:] = torch.cumsum(lens, 0)
    flat = valid.reshape(-1).nonzero().squeeze(1)              # packed token -> flat (b * T + t)
    x = idx.reshape(-1).index_select(0, flat)[None]           # (1, N)
    pos = flat % T                                             # left-padded position of every packed token
    rowb = flat // T
    if getattr(self, "prefill_compile", False):
        fn = self.__dict__.get("_prefill_trunk_c")
        if fn is None:
            m = self

            def core(x, pos, cu, T):
                h, kvs = m.trunk(x, pos=pos, collect_kv=True, cu_seqlens=cu, max_seqlen=T)
                return h, [kv for kv in kvs]
            fn = self.__dict__["_prefill_trunk_c"] = torch.compile(core, dynamic=True)
        h, kvs = fn(x, pos, cu, T)
    else:
        h, kvs = self.trunk(x, pos=pos, collect_kv=True, cu_seqlens=cu, max_seqlen=T)
    full = rows is None
    if full:
        assert n == st.B
        rows = torch.arange(n, device=dev)
    K = self.cfg.n_kv_heads
    bk = rows[rowb][:, None] * K + torch.arange(K, device=dev)[None, :]   # (N, K) cache row of every token
    if getattr(self, "prefill_kv_batched", False) and st.kv_dtype == "int8" and \
            2 * len(kvs) * kvs[0][0].numel() * 4 <= (256 << 20):  # bound the stacked fp32 copy (else per layer)
        _write_kv_batched(self, st, kvs, pos, bk, T)
    else:
        _write_kv_layers(self, st, kvs, pos, bk, T)
    st.starts[rows] = starts
    st.pos[rows] = T
    st.host_pos[rows.cpu().numpy()] = T
    if getattr(st, "prev", None) is not None:  # packed trunk resets the bigram at every row start (cu)
        st.prev[rows] = idx[:, -1]
    return self.logits(h[0].index_select(0, (cu[1:] - 1).long()))


def _write_kv_layers(self, st, kvs, pos, bk, T, sels=None):
    for l, (k, v) in enumerate(kvs):
        S = st.k[l].size(2)
        if self.cfg.is_global(l):
            sel, slot = None, pos
        elif sels is not None:  # perf flag rfh: (pos >= T - S).nonzero() computed on the host
            sel = sels[S]
            slot = pos.index_select(0, sel) % S
        else:  # only the last S positions survive in the ring
            keep = pos >= T - S
            sel = keep.nonzero().squeeze(1)
            slot = pos.index_select(0, sel) % S
        ix = (bk if sel is None else bk.index_select(0, sel)) * S + slot[:, None]
        for cache, sc, new in ((st.k[l], st.k_scale[l], k[0]), (st.v[l], st.v_scale[l], v[0])):
            new = new if sel is None else new.index_select(0, sel)
            kv_write(cache, sc, ix.reshape(-1), new)


def _write_kv_batched(self, st, kvs, pos, bk, T, sels=None):
    """perf flag 'pkv' (model.prefill_kv_batched): _write_kv_layers with the int8 quantization of every layer's k and
    v done in one quantize_kv call over the stacked (2L, N, K, hd) tensor instead of 2L calls (quantize_kv is per
    (token, head) row, so selecting rows before or after quantizing gives the same bytes); the slot indices of the
    local layers are computed once (all local layers share the window). Same cache contents bit for bit."""
    hd = self.cfg.head_dim
    L = len(kvs)
    q, s = quantize_kv(torch.stack([t[0] for kv in kvs for t in kv]))   # (2L, N, K, hd) int8, (2L, N, K) fp16
    loc = {}
    for l in range(L):
        S = st.k[l].size(2)
        if self.cfg.is_global(l):
            sel, ix = None, (bk * S + pos[:, None]).reshape(-1)
        else:
            if S not in loc:  # only the last S positions survive in the ring
                sel_ = sels[S] if sels is not None else (pos >= T - S).nonzero().squeeze(1)
                loc[S] = (sel_, (bk.index_select(0, sel_) * S + pos.index_select(0, sel_)[:, None] % S).reshape(-1))
            sel, ix = loc[S]
        for j, (cache, sc) in enumerate(((st.k[l], st.k_scale[l]), (st.v[l], st.v_scale[l]))):
            qj, sj = q[2 * l + j], s[2 * l + j]
            if sel is not None:
                qj, sj = qj.index_select(0, sel), sj.index_select(0, sel)
            cache.view(-1, hd).index_copy_(0, ix, qj.reshape(-1, hd))
            sc.view(-1).index_copy_(0, ix, sj.reshape(-1))


def prefill_packed_host(self, prompts, rows, st: KernelDecodeState, dev) -> Tensor:
    """perf flag 'rfh': MathLM.prefill(*_left_pad(prompts), st, rows) for a kernel state with prefill_packed, with
    every index tensor built on the host from the prompt lengths (no left-padded idx, no device nonzero / .cpu()
    syncs). Same chunking (rows in chunks of prefill_chunk_tokens // T, T = longest prompt), the same packed trunk
    inputs (x, pos, cu, T) to the same compiled function, the same cache slots / indices and the same KV writes, so
    the cache, st.pos / starts / host_pos and the returned logits are bit-identical. Caller sets autocast (and the
    acc16 scope)."""
    T = max(len(p) for p in prompts)
    assert T <= st.max_len
    n = len(prompts)
    chunk = max(1, getattr(self, "prefill_chunk_tokens", 1 << 62) // T)
    out = [_prefill_host_chunk(self, prompts[i:i + chunk], rows[i:i + chunk], T, st, dev) for i in range(0, n, chunk)]
    return out[0] if len(out) == 1 else torch.cat(out)


def _prefill_host_chunk(self, prompts, rows, T, st, dev):
    assert all(b.attn.kind == "attn" for b in self.blocks), "packed prefill needs attention-only mixers"
    lens = np.fromiter((len(p) for p in prompts), dtype=np.int64, count=len(prompts))
    starts = T - lens
    cu = np.zeros(len(prompts) + 1, dtype=np.int32)
    cu[1:] = np.cumsum(lens)
    x = np.fromiter((t for p in prompts for t in p), dtype=np.int64, count=int(cu[-1]))
    rowb = np.repeat(np.arange(len(prompts), dtype=np.int64), lens)
    pos = np.arange(int(cu[-1]), dtype=np.int64) - np.repeat(cu[:-1].astype(np.int64), lens) + np.repeat(starts, lens)
    K = self.cfg.n_kv_heads
    rows_np = np.asarray(rows, dtype=np.int64)
    bk = rows_np[rowb][:, None] * K + np.arange(K, dtype=np.int64)[None, :]
    sels = {}
    for l in range(len(self.blocks)):
        S = st.k[l].size(2)
        if not self.cfg.is_global(l) and S not in sels:
            sels[S] = np.nonzero(pos >= T - S)[0]
    tt = lambda a: torch.from_numpy(a).to(dev)
    xt, post, cut = tt(x)[None], tt(pos), tt(cu)
    if getattr(self, "prefill_compile", False):
        fn = self.__dict__.get("_prefill_trunk_c")
        if fn is None:
            m = self

            def core(x, pos, cu, T):
                h, kvs = m.trunk(x, pos=pos, collect_kv=True, cu_seqlens=cu, max_seqlen=T)
                return h, [kv for kv in kvs]
            fn = self.__dict__["_prefill_trunk_c"] = torch.compile(core, dynamic=True)
        h, kvs = fn(xt, post, cut, T)
    else:
        h, kvs = self.trunk(xt, pos=post, collect_kv=True, cu_seqlens=cut, max_seqlen=T)
    sels_t = {S: tt(v) for S, v in sels.items()}
    bkt = tt(bk)
    if getattr(self, "prefill_kv_batched", False) and st.kv_dtype == "int8" and \
            2 * len(kvs) * kvs[0][0].numel() * 4 <= (256 << 20):
        _write_kv_batched(self, st, kvs, post, bkt, T, sels_t)
    else:
        _write_kv_layers(self, st, kvs, post, bkt, T, sels_t)
    rt = tt(rows_np)
    st.starts[rt] = tt(starts)
    st.pos[rt] = T
    st.host_pos[rows_np] = T
    if getattr(st, "prev", None) is not None:  # last prompt token of every row (as _prefill_packed)
        st.prev[rt] = tt(np.fromiter((p[-1] for p in prompts), dtype=np.int64, count=len(prompts)))
    return self.logits(h[0].index_select(0, (cut[1:] - 1).long()))


def decode_step_fast(self, tok: Tensor, st: KernelDecodeState, g_len: int, skip_attn: bool = False,
                     num_splits: dict | None = None, backend: str = "auto", return_hidden: bool = False) -> Tensor:
    """MathLM.decode_step with the Triton kernel. g_len only sizes the split count (the kernel reads per-row
    positions, so no masks are built and rows only read their own valid keys). return_hidden (perf flag 'fz'):
    return the final normed hidden (B, d) instead of the logits (the caller runs the head + fused sampler)."""
    cfg = self.cfg
    B = tok.size(0)
    p = st.pos
    cos, sin = self.rotary.cos_sin(p)
    cos, sin = cos[:, None, :], sin[:, None, :]
    rp = self.rotary.n_pairs
    idx = tok.view(B, 1)
    ve = self._value_embeds(idx)
    xb = self._bigram_step(tok, st) if getattr(self.cfg, "bigram_rows", 0) else None
    x = rms_norm_(self.embed(idx)).to(st.dtype)
    hist = {0: x} if 0 in self.retained else {}
    dev = tok.device
    K, hd = cfg.n_kv_heads, cfg.head_dim
    pc = p[:, None]
    W = st.local_len
    bk = (torch.arange(B, device=dev)[:, None] * K + torch.arange(K, device=dev)[None, :])
    gidx = (bk * st.max_len + pc).reshape(-1)
    lidx = (bk * W + torch.remainder(pc, W)).reshape(-1)
    ns = num_splits or {}
    for l, blk in enumerate(self.blocks):
        at = blk.attn
        if xb is not None:
            x = self._inject(x, xb, l)
        if self.mudd is not None and l in cfg.mudd_sites:
            taps = [hist[t] if t != l else x for t in cfg.mudd_taps(l)]
            a_in, base = self.mudd(taps, x, l)
            xn = rms_norm_(a_in)
        else:
            xn, base = rms_norm_(x), x
        if getattr(at, "kind", "attn") != "attn":
            x = blk.finish(base, at.step(xn, st.mix[l]))
            if l + 1 in self.retained:
                hist[l + 1] = x
            continue
        q, k, v = at.qkv(xn, ve, cos, sin, rp)
        if skip_attn:
            z = q.reshape(B, 1, at.H, at.hd)
        else:
            ix = gidx if at.is_global else lidx
            kv_write(st.k[l], st.k_scale[l], ix, k)
            kv_write(st.v[l], st.v_scale[l], ix, v)
            if at.is_global:
                z = decode_attention(q.reshape(B, at.H, hd), st.k[l], st.v[l], p, st.starts, cfg.attn_scale,
                                     window=0, ring=False, k_scale=st.k_scale[l], v_scale=st.v_scale[l],
                                     ctx_hint=g_len, num_splits=ns.get("global"), backend=backend)
            else:
                z = decode_attention(q.reshape(B, at.H, hd), st.k[l], st.v[l], p, st.starts, cfg.attn_scale,
                                     window=cfg.window, ring=True, k_scale=st.k_scale[l], v_scale=st.v_scale[l],
                                     ctx_hint=W, num_splits=ns.get("local"), backend=backend)
            z = z.reshape(B, 1, at.H, at.hd)
        a = at.post(z, xn, v)
        x = blk.finish(base, a)
        if l + 1 in self.retained:
            hist[l + 1] = x
    st.pos.add_(1)
    if return_hidden:
        return rms_norm_(x)[:, 0]
    return self.logits(rms_norm_(x)[:, 0])


def rms_norm_(x: Tensor) -> Tensor:
    return torch.nn.functional.rms_norm(x, (x.size(-1),))


def patch_model(model, kv_dtype: str = "fp16", num_splits: dict | None = None, backend: str = "auto"):
    """Bind the kernel decode path onto `model` (instance attributes only). kv_dtype: "fp16" | "int8";
    backend: "auto" (CUDA tensor-core kernel when supported, else Triton) | "cuda" | "triton"."""
    def new_state(self, batch, max_len, dtype=torch.float16, device=None):
        dev = device or self.embed.weight.device
        st = KernelDecodeState(self.cfg, batch, max_len, dtype, dev, kv_dtype)
        for l, b in enumerate(self.blocks):
            if getattr(b.attn, "kind", "attn") != "attn":
                st.mix[l] = b.attn.new_state(batch, dtype, dev)
        return st

    def decode_step(self, tok, st, g_len, skip_attn=False):
        return decode_step_fast(self, tok, st, g_len, skip_attn, num_splits, backend)

    model.new_state = types.MethodType(new_state, model)
    model.prefill = types.MethodType(prefill_fast, model)
    model.decode_step = types.MethodType(decode_step, model)
    return model


def unpatch_model(model):
    for n in ("new_state", "prefill", "decode_step"):
        model.__dict__.pop(n, None)
    return model
