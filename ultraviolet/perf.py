"""Pure-win performance paths for the RL training step and rollout generation (docs/perf_opt.md).

Everything here is opt-in. Nothing in model.py / rl_grpo.py / rl_stream.py changes behaviour unless these are used.

Flags (comma-separated, `bench_perf.py --perf` / `--variants`):
  varlen   training attention through the CUTLASS varlen kernel with window_size (model.train_attn = "varlen")
  pack     pack the step's sequences back to back into fixed-size token micro-batches (cu_seqlens) instead of
           padding each to a 512-multiple and padding the row count to tok_budget // L (implies varlen)
  head     policy-gradient head (softcap logits / CE / TIS) only on target rows, torch.compile'd per fixed-size
           chunk (fused elementwise), with recomputation
  lp       (generation) behaviour log-prob computed inside the compiled / CUDA-graphed decode step instead of an
           eager log_softmax over (B, V) after every step
  h16f     (with head16) head gradient computed in the forward pass from the logits already in hand (no backward
           recompute GEMM, no second pass); bit-identical loss / grads
  rfh      (generation; with pf/pfc and the int8 kernel KV state) StreamGen refill builds the packed prefill indices on
           the host (no left-padded idx, no device nonzero / .cpu() syncs); bit-identical rows / log-probs / cache
  pkv      (generation; with pf/pfc) one int8 quantize call for every layer's prefill K/V; bit-identical
  acc16    (generation) fp16-accumulate GEMMs in prefill / decode only; changes sampler numerics
  fz       (generation; int8/fp16 kernel KV state, T > 0, no top-k / top-p) StreamGen decode step samples with one
           fused Triton kernel (softcap + inverse-CDF draw + behaviour log-prob, kernels/fused_sample.py) from the
           fp16 head output instead of the compiled Gumbel-max chain (no fp32 (B, V) logits, one uniform per row);
           implies 'lp'. Same distribution, different random stream (not bit-identical)
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn.functional as F

EOS_DEFAULT = 0


# ---------------------------------------------------------------------------------------------- packing
def pack_bins(lens: list[int], budget: int) -> list[list[int]]:
    """First-fit decreasing: indices grouped so that sum(lens) <= budget per bin."""
    order = sorted(range(len(lens)), key=lambda i: -lens[i])
    bins, room = [], []
    for i in order:
        L = lens[i]
        assert L <= budget, f"sequence of {L} tokens > budget {budget}"
        for j, r in enumerate(room):
            if r >= L:
                bins[j].append(i)
                room[j] -= L
                break
        else:
            bins.append([i])
            room.append(budget - L)
    return bins


def build_packed(items, budget: int, pad_id: int, seg_max: int = 512):
    """items: [(tokens np int64, prompt_len s, adv float, blp np float | None)]. Returns one dict per micro-batch:
    x (1, budget), y (1, budget) (-1 = no target), A, OL (float32), pos (budget,), cu (S+1,) int32, max_seqlen,
    tgt (n_t,) indices of target positions. The unused tail is split into pad segments of <= seg_max tokens (every
    query position must lie in a segment <= max_seqlen so the kernel writes it)."""
    lens = [len(t) - 1 for t, _, _, _ in items]
    out = []
    for b in pack_bins(lens, budget):
        b = sorted(b)  # deterministic order within a bin
        x = np.full(budget, pad_id, dtype=np.int64)
        y = np.full(budget, -1, dtype=np.int64)
        A = np.zeros(budget, dtype=np.float32)
        OL = np.zeros(budget, dtype=np.float32)
        pos = np.zeros(budget, dtype=np.int64)
        cu = [0]
        o = 0
        has_ol = True
        for i in b:
            t, s, adv, bl = items[i]
            n = len(t) - 1
            x[o:o + n] = t[:-1]
            y[o + s - 1:o + n] = t[s:]
            A[o:o + n] = adv
            if bl is None:
                has_ol = False
            else:
                OL[o + s - 1:o + n] = bl
            pos[o:o + n] = np.arange(n)
            o += n
            cu.append(o)
        ms = max(lens[i] for i in b)
        while o < budget:
            n = min(seg_max, budget - o)
            pos[o:o + n] = np.arange(n)
            o += n
            cu.append(o)
        ms = max(ms, min(seg_max, budget))
        out.append({"x": x[None], "y": y[None], "A": A[None], "OL": OL[None] if has_ol else None, "pos": pos,
                    "cu": np.asarray(cu, dtype=np.int32), "max_seqlen": int(ms),
                    "tgt": np.nonzero(y != -1)[0].astype(np.int64)})
    return out


def round_up(x, m):
    return ((x + m - 1) // m) * m


# ---------------------------------------------------------------------------------------------- fused head loss
def _make_head_fn(softcap, cap, temp, ent_coef, has_old):
    a_, b_, c_ = softcap

    def f(hc, w, yc, ac, oc):
        z = F.linear(hc, w.to(hc.dtype)).float()
        z = a_ * torch.sigmoid((z + b_) / c_) / temp
        ce = F.cross_entropy(z, yc.clamp_min(0), reduction="none")
        m = (yc != -1).float()
        if has_old:
            r = torch.exp((-ce.detach() - oc).clamp(max=20.0))
            wt = torch.clamp(r, max=cap) * m
            nclip = ((wt >= cap) * m).sum()
        else:
            wt = m
            nclip = torch.zeros((), device=hc.device)
        l = (ce * ac * wt).sum()
        if ent_coef > 0:
            lsm = F.log_softmax(z, -1)
            l = l - ent_coef * (-(lsm.exp() * lsm).sum(-1) * m).sum()
        return l, torch.stack([wt.sum(), nclip, m.sum()])
    return f


_HEAD_FNS: dict = {}


def pg_loss_targets(h, w, y, adv, old_lp, tgt, softcap, cap, temp=1.0, chunk=4096, ent_coef=0.0, compile=True):
    """rl_grpo.pg_loss restricted to target rows `tgt` (indices into the flattened (N,) positions): rows without a
    target contribute exactly 0 to the loss and to the stats there, so the result is the same sum. Rows are
    processed in fixed-size chunks (last one padded with masked rows) so the compiled chunk function has one
    static shape. Returns (loss sum, stats [tis sum, n clipped, n tokens])."""
    d = h.size(-1)
    h, y, adv = h.reshape(-1, d), y.reshape(-1), adv.reshape(-1)
    n = tgt.numel()
    ntot = round_up(max(n, 1), chunk)
    hs = h.index_select(0, tgt)
    ys, As = y.index_select(0, tgt), adv.index_select(0, tgt)
    Os = old_lp.reshape(-1).index_select(0, tgt) if old_lp is not None else None
    if ntot > n:
        p = ntot - n
        hs = torch.cat([hs, hs.new_zeros(p, d)])
        ys = torch.cat([ys, ys.new_full((p,), -1)])
        As = torch.cat([As, As.new_zeros(p)])
        if Os is not None:
            Os = torch.cat([Os, Os.new_zeros(p)])
    key = (tuple(softcap), cap, temp, ent_coef, Os is not None, compile)
    fn = _HEAD_FNS.get(key)
    if fn is None:
        fn = _make_head_fn(softcap, cap, temp, ent_coef, Os is not None)
        if compile:
            fn = torch.compile(fn, dynamic=False)
        _HEAD_FNS[key] = fn
    tot = h.new_zeros((), dtype=torch.float32)
    stats = torch.zeros(3, device=h.device)
    for i in range(0, ntot, chunk):
        oc = None if Os is None else Os[i:i + chunk]
        l, s = torch.utils.checkpoint.checkpoint(fn, hs[i:i + chunk], w, ys[i:i + chunk], As[i:i + chunk], oc,
                                                 use_reentrant=False)
        tot = tot + l
        stats += s.detach()
    return tot, stats


# ---------------------------------------------------------------------------------------------- head16
def _mm32(a, b):
    """fp16 x fp16 -> fp32 output (fp32 accumulate, no fp16 rounding of the result)."""
    return torch.mm(a, b, out_dtype=torch.float32)


def _make_h16_fns(softcap, cap, temp, has_old):
    a_, b_, c_ = softcap

    def fwd(z, yc, ac, oc):
        u = torch.sigmoid((z + b_) / c_)
        zc = a_ * u / temp
        lse = torch.logsumexp(zc, -1)
        zt = zc.gather(1, yc.clamp_min(0)[:, None])[:, 0]
        ce = lse - zt
        m = (yc != -1).float()
        if has_old:
            r = torch.exp((-ce - oc).clamp(max=20.0))
            wt = torch.clamp(r, max=cap) * m
            nclip = ((wt >= cap) * m).sum()
        else:
            wt = m
            nclip = torch.zeros((), device=z.device)
        coef = ac * wt
        return (ce * coef).sum(), lse, coef, torch.stack([wt.sum(), nclip, m.sum()])

    def bwd(z, yc, lse, coef):
        # d loss / d z = coef * (softmax(zc) - onehot) * dzc/dz,  dzc/dz = a/(c T) u (1 - u)
        u = torch.sigmoid((z + b_) / c_)
        zc = a_ * u / temp
        p = torch.exp(zc - lse[:, None])
        oh = torch.zeros_like(p).scatter_(1, yc.clamp_min(0)[:, None], 1.0)
        g = coef[:, None] * (p - oh) * (a_ / (c_ * temp)) * u * (1 - u)
        s = coef.abs().amax().clamp_min(1e-30) * (a_ / (4 * c_ * temp))  # |g| <= s
        return (g / s).to(torch.float16), s
    return fwd, bwd


_H16: dict = {}


class _Head16(torch.autograd.Function):
    """Sum over rows of coef * CE(softcap(h W^T) / T, y) with TIS weights, chunked; GEMMs in fp16 with fp32
    accumulate/output; logits recomputed in backward (never stored)."""

    @staticmethod
    def forward(ctx, hs, w, ys, As, Os, key, chunk):
        fwd, bwd = _H16[key]
        h16, w16 = hs.to(torch.float16), w.to(torch.float16)
        n = hs.size(0)
        tot = torch.zeros((), device=hs.device)
        stats = torch.zeros(3, device=hs.device)
        lses, coefs = [], []
        for i in range(0, n, chunk):
            z = _mm32(h16[i:i + chunk], w16.t())
            l, lse, coef, st = fwd(z, ys[i:i + chunk], As[i:i + chunk], None if Os is None else Os[i:i + chunk])
            tot += l
            stats += st
            lses.append(lse)
            coefs.append(coef)
        ctx.save_for_backward(h16, w16, ys, torch.cat(lses), torch.cat(coefs))
        ctx.key, ctx.chunk, ctx.hdt = key, chunk, hs.dtype
        ctx.mark_non_differentiable(stats)
        return tot, stats

    @staticmethod
    def backward(ctx, gl, gs):
        h16, w16, ys, lse, coef = ctx.saved_tensors
        _, bwd = _H16[ctx.key]
        n, chunk = h16.size(0), ctx.chunk
        dh = torch.empty(h16.shape, dtype=torch.float32, device=h16.device)
        dw = torch.zeros(w16.shape, dtype=torch.float32, device=h16.device)
        for i in range(0, n, chunk):
            z = _mm32(h16[i:i + chunk], w16.t())
            g16, s = bwd(z, ys[i:i + chunk], lse[i:i + chunk], coef[i:i + chunk])
            dh[i:i + chunk] = _mm32(g16, w16) * s
            dw += _mm32(g16.t().contiguous(), h16[i:i + chunk]) * s
        return (dh * gl).to(ctx.hdt), dw * gl, None, None, None, None, None


class _Head16F(torch.autograd.Function):
    """perf flag 'h16f' (with head16): _Head16 with the gradient computed in the forward pass, chunk by chunk, from
    the logits already in hand (the loss is a sum over rows, so each chunk's d loss / d logits needs only that
    chunk). Same GEMMs, same compiled fwd / bwd functions on the same inputs in the same chunk order as _Head16, so
    loss, stats, dh and dw are bit-identical; the backward-pass logit recompute GEMM is gone (3 instead of 4 head
    GEMMs per chunk). Holds dh (n, d) fp32 and dw (V, d) fp32 from forward to backward instead of h16 / lse / coef."""

    @staticmethod
    def forward(ctx, hs, w, ys, As, Os, key, chunk):
        fwd, bwd = _H16[key]
        h16, w16 = hs.to(torch.float16), w.to(torch.float16)
        n = hs.size(0)
        tot = torch.zeros((), device=hs.device)
        stats = torch.zeros(3, device=hs.device)
        dh = torch.empty(h16.shape, dtype=torch.float32, device=h16.device)
        dw = torch.zeros(w16.shape, dtype=torch.float32, device=h16.device)
        for i in range(0, n, chunk):
            hc, yc = h16[i:i + chunk], ys[i:i + chunk]
            z = _mm32(hc, w16.t())
            l, lse, coef, st = fwd(z, yc, As[i:i + chunk], None if Os is None else Os[i:i + chunk])
            tot += l
            stats += st
            g16, s = bwd(z, yc, lse, coef)
            del z
            dh[i:i + chunk] = _mm32(g16, w16) * s
            dw += _mm32(g16.t().contiguous(), hc) * s
        ctx.save_for_backward(dh, dw)
        ctx.hdt = hs.dtype
        ctx.mark_non_differentiable(stats)
        return tot, stats

    @staticmethod
    def backward(ctx, gl, gs):
        dh, dw = ctx.saved_tensors
        return (dh * gl).to(ctx.hdt), dw * gl, None, None, None, None, None


def pg_loss_targets16(h, w, y, adv, old_lp, tgt, softcap, cap, temp=1.0, chunk=4096, fused=False):
    """head16 variant of pg_loss_targets (no entropy term). fused (perf flag 'h16f'): _Head16F (bit-identical)."""
    d = h.size(-1)
    h, y, adv = h.reshape(-1, d), y.reshape(-1), adv.reshape(-1)
    n = tgt.numel()
    ntot = round_up(max(n, 1), chunk)
    hs = h.index_select(0, tgt)
    ys, As = y.index_select(0, tgt), adv.index_select(0, tgt)
    Os = old_lp.reshape(-1).index_select(0, tgt) if old_lp is not None else None
    if ntot > n:
        p = ntot - n
        hs = torch.cat([hs, hs.new_zeros(p, d)])
        ys = torch.cat([ys, ys.new_full((p,), -1)])
        As = torch.cat([As, As.new_zeros(p)])
        if Os is not None:
            Os = torch.cat([Os, Os.new_zeros(p)])
    key = (tuple(softcap), cap, temp, Os is not None)
    if key not in _H16:
        f, b = _make_h16_fns(softcap, cap, temp, Os is not None)
        _H16[key] = (torch.compile(f, dynamic=False), torch.compile(b, dynamic=False))
    return (_Head16F if fused else _Head16).apply(hs, w, ys, As, Os, key, chunk)


# ---------------------------------------------------------------------------------------------- generation
def scope_fp16_acc():
    """perf flag 'acc16': fp16-accumulate cuBLAS GEMMs only inside MathLM.prefill and
    DecodeRunner.step (rollouts + in-run evals; graphs captured in step keep the h1688 kernels). Training stays fp32."""
    import functools
    from . import model as M

    def wrap(fn):
        @functools.wraps(fn)
        def w(*a, **k):
            old = torch.backends.cuda.matmul.allow_fp16_accumulation
            torch.backends.cuda.matmul.allow_fp16_accumulation = True
            try:
                return fn(*a, **k)
            finally:
                torch.backends.cuda.matmul.allow_fp16_accumulation = old
        return w
    M.MathLM.prefill = wrap(M.MathLM.prefill)
    M.DecodeRunner.step = wrap(M.DecodeRunner.step)


# ---------------------------------------------------------------------------------------------- trainer
def make_trainer(base, flags, rl):
    """Subclass of rl_grpo.Trainer whose step() is the reference forward/backward (a verbatim copy of
    Trainer.step's 1gpu / stream path) or the flagged variant, followed by the unchanged optimizer tail.
    `rl` is the rl_grpo module (for DIST, pg_loss, make_xy, micro_batches)."""
    flags = set(flags)
    if "pack" in flags:
        flags.add("varlen")

    class PerfTrainer(base):
        perf_flags = frozenset(flags)

        def __init__(self, raw, dev, args):
            super().__init__(raw, dev, args)
            if "varlen" in flags:
                raw.train_attn = "varlen"
            if "pack" in flags:
                assert all(b.attn.kind == "attn" for b in raw.blocks), "packing needs attention-only mixers"
                self.ptrunk = torch.compile(
                    lambda x, pos, cu, ms: raw.trunk(x, pos=pos, cu_seqlens=cu, max_seqlen=ms)[0])

        # ---- forward + backward over the batch (accumulates scaled grads); returns (n_ex, st, loss_sum)
        def fwd_bwd(self, k, seqs, advs, blps):
            a = self.a
            DIST = rl.DIST
            if getattr(a, "grpo_norm", 0):
                advs = [adv * a.max_new / max(1, len(t) - s) for (t, s), adv in zip(seqs, advs)]
            ent_coef = getattr(a, "entropy_coef", 0.0)
            ex = [(t, s, adv) for (t, s), adv in zip(seqs, advs) if adv != 0.0 or ent_coef > 0]
            n_seq, n_ex = len(seqs), len(ex)
            if DIST["on"]:
                c = torch.tensor([n_seq, n_ex], device=self.dev, dtype=torch.float64)
                torch.distributed.all_reduce(c)
                n_seq, n_ex = int(c[0].item()), int(c[1].item())
            norm = max(n_seq, 1) * a.max_new
            st = torch.zeros(3, device=self.dev)
            ltot = 0.0
            if "pack" in flags:
                bmap = {id(t): b for (t, s), b in zip(seqs, blps)} if blps is not None else None
                items = [(t, s, adv, None if bmap is None else np.asarray(bmap[id(t)], dtype=np.float32))
                         for t, s, adv in ex]
                for it in items:
                    if it[3] is not None:
                        assert len(it[3]) == len(it[0]) - it[1]
                mbs = build_packed(items, a.tok_budget, rl.EOS)
                for mb in mbs:
                    x = torch.from_numpy(mb["x"]).to(self.dev, non_blocking=True)
                    y = torch.from_numpy(mb["y"]).to(self.dev, non_blocking=True)
                    A = torch.from_numpy(mb["A"]).to(self.dev, non_blocking=True)
                    OL = None if mb["OL"] is None else torch.from_numpy(mb["OL"]).to(self.dev, non_blocking=True)
                    pos = torch.from_numpy(mb["pos"]).to(self.dev, non_blocking=True)
                    cu = torch.from_numpy(mb["cu"]).to(self.dev, non_blocking=True)
                    tgt = torch.from_numpy(mb["tgt"]).to(self.dev, non_blocking=True)
                    with torch.autocast("cuda", dtype=torch.float16, enabled="fp32" not in flags):
                        h = self.ptrunk(x, pos, cu, round_up(mb["max_seqlen"], 512))
                    loss, s = self._loss(h, y, A, OL, tgt, ent_coef)
                    self.scaler.scale(loss / norm).backward()
                    st += s
                    ltot += float(loss.detach())
                return n_ex, norm, st, ltot, len(mbs)
            mbs = rl.micro_batches([(t, s) for t, s, _ in ex], a.tok_budget, a.pad_to)
            amap = {id(t): adv for t, s, adv in ex}
            bmap = {id(t): b for (t, s), b in zip(seqs, blps)} if blps is not None else None
            for mb in mbs:
                x, y = rl.make_xy(mb, a.pad_to, a.tok_budget)
                A = torch.zeros(y.shape, dtype=torch.float32)
                for i, (t, s) in enumerate(mb):
                    A[i] = amap[id(t)]
                OL = None
                if bmap is not None:
                    OL = torch.zeros(y.shape, dtype=torch.float32)
                    for i, (t, s) in enumerate(mb):
                        b = bmap[id(t)]
                        OL[i, s - 1:len(t) - 1] = torch.from_numpy(np.asarray(b, dtype=np.float32))
                    OL = OL.to(self.dev)
                x, y, A = x.to(self.dev), y.to(self.dev), A.to(self.dev)
                with torch.autocast("cuda", dtype=torch.float16, enabled="fp32" not in flags):
                    h = self.trunk(x) if "fp32" not in flags else self.raw.trunk(x)[0]
                tgt = (y.reshape(-1) != -1).nonzero().squeeze(1) if flags & {"head", "head16"} else None
                loss, s = self._loss(h, y, A, OL, tgt, ent_coef)
                self.scaler.scale(loss / norm).backward()
                st += s
                ltot += float(loss.detach())
            return n_ex, norm, st, ltot, len(mbs)

        @torch.no_grad()
        def logps(self, seqs, temp=1.0):
            """Per-sequence log-probs of the response tokens (np arrays) under this variant's forward numerics."""
            a = self.a
            sc = self.raw.cfg.softcap
            out = {}
            fp32 = "fp32" in flags
            wt = self.raw.embed.weight

            def head(hr):
                if "head16" in flags:
                    z = _mm32(hr.to(torch.float16), wt.to(torch.float16).t())
                else:
                    z = F.linear(hr, wt.to(hr.dtype)).float()
                z = sc[0] * torch.sigmoid((z + sc[1]) / sc[2]) / temp
                return torch.log_softmax(z, -1)
            if "pack" in flags:
                items = [(t, s_, 1.0, None) for t, s_ in seqs]
                for mb, bins in zip(build_packed(items, a.tok_budget, rl.EOS), pack_bins([len(t) - 1 for t, _ in seqs], a.tok_budget)):
                    x = torch.from_numpy(mb["x"]).to(self.dev)
                    with torch.autocast("cuda", dtype=torch.float16, enabled=not fp32):
                        h = self.ptrunk(x, torch.from_numpy(mb["pos"]).to(self.dev), torch.from_numpy(mb["cu"]).to(self.dev),
                                        round_up(mb["max_seqlen"], 512))[0]
                    o = 0
                    for i in sorted(bins):
                        t, s_ = seqs[i]
                        n = len(t) - 1
                        lp = head(h[o + s_ - 1:o + n])
                        out[i] = lp.gather(1, torch.from_numpy(t[s_:]).to(self.dev)[:, None])[:, 0].cpu().numpy()
                        o += n
                return [out[i] for i in range(len(seqs))]
            for i, (t, s_) in enumerate(seqs):
                L = round_up(len(t) - 1, a.pad_to)
                x = torch.full((1, L), rl.EOS, dtype=torch.long)
                x[0, :len(t) - 1] = torch.from_numpy(t[:-1])
                x = x.to(self.dev)
                with torch.autocast("cuda", dtype=torch.float16, enabled=not fp32):
                    h = (self.raw.trunk(x)[0] if fp32 else self.trunk(x))[0]
                lp = head(h[s_ - 1:len(t) - 1])
                out[i] = lp.gather(1, torch.from_numpy(t[s_:]).to(self.dev)[:, None])[:, 0].cpu().numpy()
            return [out[i] for i in range(len(seqs))]

        def _loss(self, h, y, A, OL, tgt, ent_coef):
            a = self.a
            if "head16" in flags and ent_coef == 0:
                if tgt is None:
                    tgt = (y.reshape(-1) != -1).nonzero().squeeze(1)
                return pg_loss_targets16(h, self.raw.embed.weight, y, A, OL, tgt, self.raw.cfg.softcap, a.tis_cap,
                                         a.temperature, fused="h16f" in flags)
            if "head" in flags or "head16" in flags:
                if tgt is None:
                    tgt = (y.reshape(-1) != -1).nonzero().squeeze(1)
                return pg_loss_targets(h, self.raw.embed.weight, y, A, OL, tgt, self.raw.cfg.softcap, a.tis_cap,
                                       a.temperature, ent_coef=ent_coef)
            if "pack" in flags:  # packed rows: plain pg_loss over all positions (pad rows have y = -1)
                return rl.pg_loss(h, self.raw.embed.weight, y, A, OL, self.raw.cfg.softcap, a.tis_cap,
                                  a.temperature, ent_coef=ent_coef)
            return rl.pg_loss(h, self.raw.embed.weight, y, A, OL, self.raw.cfg.softcap, a.tis_cap, a.temperature,
                              ent_coef=ent_coef)

        def step(self, k, seqs, advs, blps=None):
            if not flags or getattr(self.a, "clip_cov", 0.0) > 0 or self.old is not None:
                return super().step(k, seqs, advs, blps)
            a = self.a
            DIST = rl.DIST
            f = min(1.0, (k + 1) / a.warmup) if a.warmup else 1.0
            for o in self.opts:
                for g in o.param_groups:
                    g["lr"] = g["base_lr"] * f
            self.raw.train()
            n_ex, norm, st, _, _ = self.fwd_bwd(k, seqs, advs, blps)
            if n_ex == 0:
                return {"train_seqs": 0, "grad_norm": 0.0, "lr_f": f, "grad_scale": self.scaler.get_scale(),
                        "tis_mean": 1.0, "tis_clip_frac": 0.0}
            # ---- optimizer tail: verbatim from rl_grpo.Trainer.step
            if DIST["on"]:
                gs = []
                for p in self.params:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    gs.append(p.grad)
                buf = torch._utils._flatten_dense_tensors(gs)
                torch.distributed.all_reduce(buf)
                for g_, v_ in zip(gs, torch._utils._unflatten_dense_tensors(buf, gs)):
                    g_.copy_(v_)
                torch.distributed.all_reduce(st)
            for o in self.opts:
                self.scaler.unscale_(o)
            gn = torch.nn.utils.clip_grad_norm_(self.params, a.grad_clip if a.grad_clip > 0 else float("inf"))
            finite = math.isfinite(float(gn)) and self.params_finite()
            if finite:
                for o in self.opts:
                    self.scaler.step(o)
                self.nonfinite_consec = 0
            else:
                self.nonfinite_consec += 1
                self.nonfinite_total += 1
            self.scaler.update()
            for o in self.opts:
                o.zero_grad(set_to_none=True)
            n_abort = getattr(a, "nonfinite_abort", 0)
            if n_abort and self.nonfinite_consec >= n_abort:
                raise FloatingPointError(f"NONFINITE_ABORT: {self.nonfinite_consec} consecutive non-finite steps "
                                         f"(step {k + 1}, grad_norm {float(gn)}, params finite {self.params_finite()})")
            st = st.tolist()
            return {"train_seqs": n_ex, "grad_norm": float(gn), "lr_f": f, "grad_scale": self.scaler.get_scale(),
                    "tis_mean": st[0] / max(st[2], 1), "tis_clip_frac": st[1] / max(st[2], 1),
                    "nonfinite_skip": int(not finite), "nonfinite_total": self.nonfinite_total}

    return PerfTrainer


def grads_only(tr, k, seqs, advs, blps):
    """Run tr.fwd_bwd and return (unscaled grads as fp32 CPU tensors, stats, seconds); grads are cleared after."""
    for o in tr.opts:
        o.zero_grad(set_to_none=True)
    tr.raw.train()
    torch.cuda.synchronize()
    t = time.time()
    n_ex, norm, st, ltot, nmb = tr.fwd_bwd(k, seqs, advs, blps)
    torch.cuda.synchronize()
    dt = time.time() - t
    sc = tr.scaler.get_scale()
    g = [(p.grad.float() / sc).cpu() if p.grad is not None else torch.zeros(p.shape) for p in tr.params]
    for o in tr.opts:
        o.zero_grad(set_to_none=True)
    st = st.tolist()
    return g, {"loss": ltot, "tis": st[0] / max(st[2], 1), "n_tok": st[2], "n_mb": nmb}, dt


# ---------------------------------------------------------------------------------------------- generation
def make_streamgen(base, flags):
    """StreamGen variant. 'lp': behaviour log-probs computed inside the decode graph (no eager (B, V) log_softmax
    per step). 'hv': harvest copies only the finished rows' token / log-prob buffers to the host (instead of the
    whole (slots, max_new) buffers every chunk). Outputs are the same rows / tokens / log-probs."""
    if not ({"lp", "hv", "rfh", "fz"} & set(flags)):
        return base
    lp, hv, rfh, acc16, fz = "lp" in flags, "hv" in flags, "rfh" in flags, "acc16" in flags, "fz" in flags

    class FastGen(base):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.runner.want_logp = lp
            self._rfh = rfh and self.st.kv_dtype is not None and getattr(self.m, "prefill_packed", False)
            self._fz = fz and install_fz(self.runner)
            if self._fz:
                self.runner.want_logp = True

        def chunk(self):
            if not (lp or hv or self._fz):
                return super().chunk()
            return _chunk_fast(self, lp or self._fz, hv)

        def _refill(self, rows):
            if not self._rfh:
                return super()._refill(rows)
            return _refill_host(self, rows, acc16)
    return FastGen


def install_fz(runner) -> bool:
    """perf flag 'fz': replace runner._body (one decode step + sample) by the trunk (compiled when runner.compile) up
    to the final normed hidden, the fp16 head GEMM and the fused sampler (kernels/fused_sample.py), writing
    runner.out_tok / out_logp (out_logits becomes the raw fp16 head output, not the softcapped fp32 logits). Returns
    False (runner unchanged, default sampler) when it does not apply: non-kernel KV state, top-k / top-p, or
    fused_sample_ok fails (no CUDA / Triton, vocab not a multiple of the block, extreme temperature)."""
    import types
    from .kernels.fused_sample import fused_sample_ok
    m, st = runner.m, runner.st
    if st.kv_dtype is None or runner.top_k or runner.top_p < 1.0:
        return False
    if not fused_sample_ok(st.pos.device, m.cfg.vocab_size, m.cfg.softcap, runner.temperature):
        return False
    runner._body = types.MethodType(_fz_body, runner)
    runner.want_logp = True
    return True


def _fz_body(self, g_len: int):
    from .kernels.fused_sample import fused_sample
    fn = _fz_trunk_compiled() if self.compile else _fz_trunk
    h = fn(self.m, self.st, self.tok, g_len, self.skip_attn)
    x = F.linear(h, self.m.embed.weight.to(h.dtype))  # (B, V) fp16: the GEMM MathLM.logits runs before .float()
    u = torch.rand(x.size(0), device=x.device, dtype=torch.float32)
    nxt, lp = fused_sample(x, u, self.m.cfg.softcap, self.temperature)
    self.out_logp.copy_(lp)
    self.out_tok.copy_(nxt)
    return x


@torch.no_grad()
def _fz_trunk(m, st, tok, g_len: int, skip_attn: bool = False):
    from .kernels.fast_decode import decode_step_fast
    return decode_step_fast(m, tok, st, g_len, skip_attn, m.decode_num_splits, m.decode_backend, return_hidden=True)


_FZ_COMPILED = None


def _fz_trunk_compiled():
    global _FZ_COMPILED
    if _FZ_COMPILED is None:
        import torch._dynamo
        torch._dynamo.config.cache_size_limit = max(64, torch._dynamo.config.cache_size_limit)
        _FZ_COMPILED = torch.compile(_fz_trunk)
    return _FZ_COMPILED


@torch.no_grad()
def _refill_host(self, rows, acc16):
    """perf flag 'rfh' (needs pf/pfc and the kernel KV state): rl_stream.StreamGen._refill with the packed prefill's
    index tensors built on the host (kernels.fast_decode.prefill_packed_host). Same rows / prompts / RNG draws and
    the same device writes in the same order; bit-identical."""
    from .kernels.fast_decode import prefill_packed_host
    from .model import _autocast, sample
    rows = rows[:len(self.queue)]
    if not rows:
        return
    items = [self.queue.popleft() for _ in rows]
    rt = torch.tensor(rows, device=self.dev)
    self.buf[rt] = -1
    old = torch.backends.cuda.matmul.allow_fp16_accumulation
    if acc16:  # scope_fp16_acc wraps MathLM.prefill, which this path does not call
        torch.backends.cuda.matmul.allow_fp16_accumulation = True
    try:
        with _autocast(self.dev, self.dtype):
            logits = prefill_packed_host(self.m, [p for p, _ in items], rows, self.st, self.dev)
    finally:
        torch.backends.cuda.matmul.allow_fp16_accumulation = old
    first = sample(logits, self.T)
    self.lp[rt, 0] = F.log_softmax(logits.float() / self.T, -1).gather(1, first[:, None])[:, 0]
    self.tok[rt] = first
    self.buf[rt, 0] = first
    self.cnt[rt] = 1
    for r, (_, meta) in zip(rows, items):
        self.req[r] = (meta, self.version)
        self.host_cnt[r] = 1


@torch.no_grad()
def _chunk_fast(self, lp_in_graph, hv):
    """rl_stream.StreamGen.chunk with the two optional changes above (same harvest / refill / parking logic)."""
    idle = [r for r in range(self.B) if self.req[r] is None]
    if idle and self.queue:
        self._refill(idle)
    if self.n_active() == 0:
        return []
    B = self.B
    for _ in range(self.ce):
        if lp_in_graph:
            tok = self.runner.step(self.tok)
            lpt = self.runner.out_logp.view(B, 1)
        else:
            gl = self.runner.g_len(int(self.st.host_pos.max()))
            tok = self.runner.step(self.tok)
            lg = self.bucket_logits.setdefault(gl, self.runner.out_logits) if self.runner.use_graph \
                else self.runner.out_logits
            lpt = F.log_softmax(lg.float() / self.T, -1).gather(1, tok.view(B, 1))
        self.buf.scatter_(1, self.cnt.view(B, 1), tok.view(B, 1))
        self.lp.scatter_(1, self.cnt.view(B, 1), lpt)
        self.cnt.add_(1).clamp_(max=self.buf.size(1) - 1)
        self.tok.copy_(tok)
        self.n_decode_steps += 1
    self.host_cnt += self.ce
    hit = (self.buf == self.eos).any(1).cpu().numpy()
    fin = [r for r in range(B) if self.req[r] is not None and (self.host_cnt[r] >= self.max_new or hit[r])]
    out, done = [], []
    if fin:
        if hv:
            ri = torch.tensor(fin, device=self.buf.device)
            rb, rl = self.buf.index_select(0, ri).cpu().numpy(), self.lp.index_select(0, ri).cpu().numpy()
            where = {r: j for j, r in enumerate(fin)}
        else:
            rb, rl = self.buf.cpu().numpy(), self.lp.cpu().numpy()
            where = {r: r for r in fin}
        for r in fin:
            j = where[r]
            n = min(self.host_cnt[r], self.max_new)
            row = rb[j, :n].tolist()
            if self.eos in row:
                n = row.index(self.eos) + 1
                row = row[:n]
            meta, v0 = self.req[r]
            out.append((meta, row, rl[j, :n].copy(), v0))
            self.n_tokens += n
            self.req[r] = None
            done.append(r)
    if done:
        self._refill(done)
        self._park([r for r in done if self.req[r] is None])
    return out
