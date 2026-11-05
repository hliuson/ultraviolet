"""GRPO-family RL for the math LM (Dr.GRPO advantages, truncated IS for the one-step-stale 2-GPU pipeline).

Per step: n_prompts problems from the guess-resistant pool x G samples (T, max_new, continuous batching, fused fp16
+ int8-KV decode) -> reward 1 iff </think> closed and extract(text) == ref -> advantage r - mean_group (no std) ->
one policy-gradient update: loss = -sum_t A * min(pi/pi_behav, tis_cap).detach() * log pi / (n_seq * max_new)
(Dr.GRPO constant normaliser), fp16 autocast + GradScaler, grad clip, NorMuon (matrices) + AdamW (rest) or AdamW
for everything -> sync into the fp16 generation copy.

--mode 2gpu: rollout k+1 (cuda:0, weights W_k) overlaps update k (cuda:1); rollout k was sampled by W_{k-1}, so the
behaviour log-probs come from a stale copy of the previous weights (teacher-forced, same numerics as training).
--mode 1gpu: sequential on-policy (IS ratio == 1, no extra forward).

Eval (every --eval-every steps, in-process on the generation model) on eval_problems.jsonl: conditions a_normal
(think), c_answer_only, [b_nothink]; subsets all / guess-resistant (filters.json) and per source.
"""
import argparse
import collections
import copy
import json
import math
import os
import random
import sys
import threading
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from eval_math import ANS, EOS, ans_extract, arith_claims, extract, grader, last_boxed, norm  # noqa: E402
from prep_traces import prompt_text  # noqa: E402
from rl_sampler import DEFAULT_WEIGHTS, POOLS, make_sampler  # noqa: E402
from sft import make_xy, micro_batches  # noqa: E402
from ultraviolet.model import MathConfig, MathLM, generate_continuous  # noqa: E402
from ultraviolet.optim import build_optimizers  # noqa: E402


def softcap_logits(h, w, softcap):
    a, b, c = softcap
    z = F.linear(h, w.to(h.dtype)).float()
    return a * torch.sigmoid((z + b) / c)


@torch.no_grad()
def token_logp(h, w, y, softcap, temp=1.0, chunk=4096):
    h, y = h.reshape(-1, h.size(-1)), y.reshape(-1)
    out = torch.zeros(y.shape, dtype=torch.float32, device=h.device)
    for i in range(0, h.size(0), chunk):
        z = softcap_logits(h[i:i + chunk], w, softcap) / temp
        out[i:i + chunk] = -F.cross_entropy(z, y[i:i + chunk].clamp_min(0), reduction="none")
    return out


def pg_loss(h, w, y, adv, old_lp, softcap, cap, temp=1.0, chunk=4096, ent_coef=0.0, ppo_lp=None, clip=(0.2, 0.28)):
    """log pi is the SAMPLING distribution softmax(logits / temp) (the rollout temperature). Returns (sum_t adv*tis*CE, sum of tis weights, n clipped, n tokens). old_lp None -> tis = 1.
    ent_coef > 0 (--entropy-coef): entropy bonus, - ent_coef * H(softmax(z / temp)) per response token is added to
    the summed loss (same normaliser as the policy-gradient term, i.e. verl's token-level entropy loss).
    ppo_lp (--ppo-schedule): the trainer's log-probs before this update (pi_old). The weight becomes
    r * tis with r = pi / pi_old, zeroed where the PPO clip is active (A > 0 and r > 1 + clip[1], A < 0 and
    r < 1 - clip[0]); tis = min(pi_old / pi_behav, cap) is the decoupled behaviour correction (1 without old_lp). The
    gradient is that of -min(r A, clip(r) A) * tis. Stats gain n low-clipped, n high-clipped and sum(pi_old lp - lp)."""
    h, y, adv = h.reshape(-1, h.size(-1)), y.reshape(-1), adv.reshape(-1)
    stats = torch.zeros(3 if ppo_lp is None else 6, device=h.device)

    def f(hc, w, yc, ac, oc, pc):
        z = softcap_logits(hc, w, softcap) / temp
        ce = F.cross_entropy(z, yc.clamp_min(0), reduction="none")
        m = (yc != -1).float()
        zero = torch.zeros((), device=hc.device)
        if pc is not None:
            r = torch.exp((-ce.detach() - pc).clamp(-20.0, 20.0))
            tis = torch.ones_like(r) if oc is None else torch.clamp(torch.exp((pc - oc).clamp(max=20.0)), max=cap)
            lo, hi = (ac < 0) & (r < 1 - clip[0]), (ac > 0) & (r > 1 + clip[1])
            wt = r * tis * (~(lo | hi)).float() * m
            st = [(tis * m).sum(), ((tis >= cap) * m).sum() if oc is not None else zero, m.sum(),
                  (lo * m).sum(), (hi * m).sum(), ((pc + ce.detach()) * m).sum()]
        elif oc is None:
            wt = m
        else:
            r = torch.exp((-ce.detach() - oc).clamp(max=20.0))
            wt = torch.clamp(r, max=cap) * m
        l = (ce * ac * wt).sum()
        if ent_coef > 0:
            lsm = F.log_softmax(z, -1)
            l = l - ent_coef * (-(lsm.exp() * lsm).sum(-1) * m).sum()
        if pc is not None:
            return l, torch.stack(st)
        return l, torch.stack([(wt).sum(), ((wt >= cap) * m).sum() if oc is not None else zero, m.sum()])

    tot = h.new_zeros((), dtype=torch.float32)
    for i in range(0, h.size(0), chunk):
        oc = None if old_lp is None else old_lp.reshape(-1)[i:i + chunk]
        pc = None if ppo_lp is None else ppo_lp.reshape(-1)[i:i + chunk]
        l, s = torch.utils.checkpoint.checkpoint(f, h[i:i + chunk], w, y[i:i + chunk], adv[i:i + chunk], oc, pc,
                                                 use_reentrant=False)
        tot = tot + l
        stats += s.detach()
    return tot, stats


def parse_ppo_schedule(s):
    """--ppo-schedule "2x4:50,2x2:150,2x1" -> [(minibatches, epochs, last update or None)]: updates 1-50 take 2
    minibatches x 4 passes, 51-150 2 x 2, then 2 x 1 to the end. "" -> [] (one step per batch, no PPO clip)."""
    out = []
    for i, tok in enumerate(x for x in s.split(",") if x.strip()):
        me, _, until = tok.strip().partition(":")
        mb, ep = (int(v) for v in me.split("x"))
        assert mb >= 1 and ep >= 1, tok
        if until:
            assert not out or out[-1][2] < int(until), f"--ppo-schedule: bounds must increase ({s})"
        else:
            assert i == len([x for x in s.split(",") if x.strip()]) - 1, f"--ppo-schedule: only the last rung is open ({s})"
        out.append((mb, ep, int(until) if until else None))
    return out


def group_starts(seqs):
    """Indices where a group (a run of sequences with the same prompt) starts in seqs [(tokens, prompt_len)]."""
    return [i for i in range(len(seqs)) if i == 0 or seqs[i][1] != seqs[i - 1][1] or
            not np.array_equal(seqs[i][0][:seqs[i][1]], seqs[i - 1][0][:seqs[i - 1][1]])]


def probe_stats(ga, gb, na, nb, g0=None):
    """--grad-probe-every: gradient-noise statistics from two disjoint halves of a batch (na, nb groups; each half's
    gradient a per-group mean). E[ga.gb] = |G|^2 (the true gradient's squared norm) and E|ga - gb|^2 = tr(Sigma)
    (1/na + 1/nb) (Sigma = the per-group gradient covariance), so B_noise = tr(Sigma) / |G|^2 is the critical batch size
    in groups (McCandlish et al. 2018): minibatches well above it make extra steps nearly free, below it each step is
    mostly noise. Single-update estimates are noisy: average g2 and trs over updates before dividing.
    cos0: cosine of this full gradient (ga + gb) with g0, the same batch's full gradient before the update's first step.
    Returns (stats, ga + gb)."""
    ab, a2, b2 = float(ga @ gb), float(ga @ ga), float(gb @ gb)
    g = ga + gb
    out = {"cos": ab / max(math.sqrt(max(a2 * b2, 0.0)), 1e-30), "g2": ab,
           "trs": (a2 + b2 - 2 * ab) / (1 / max(na, 1) + 1 / max(nb, 1))}
    out["cos0"] = 1.0 if g0 is None else float(g @ g0) / max(float(g.norm() * g0.norm()), 1e-30)
    return out, g


def ppo_cuts(seqs, M):
    """M + 1 cut indices splitting seqs [(tokens, prompt_len)] into M contiguous minibatches of whole groups (a group =
    a run of sequences with the same prompt), cutting at the group starts nearest n * p / M; minibatches may be empty
    but there are always M of them (every rank takes the same number of optimizer steps)."""
    n = len(seqs)
    starts = group_starts(seqs)
    if len(starts) < M:  # fewer groups than minibatches: plain split
        return [n * p // M for p in range(M + 1)]
    return [0] + [min(starts, key=lambda b: abs(b - n * p / M)) for p in range(1, M)] + [n]


def ppo_rung(sched, k):
    """(minibatches, epochs) for update k (0-based) under parse_ppo_schedule's list; past the last bound, the last rung."""
    for mb, ep, until in sched:
        if until is None or k + 1 <= until:
            return mb, ep
    return sched[-1][:2]


class RLScaler(torch.amp.GradScaler):
    """fp16 GradScaler with two RL additions (both inert by default):
    - max_scale (--scaler-max-scale): update() never grows the scale above this cap (with rare overflows and a short
      growth interval the scale otherwise grows without bound).
    - min_scale (--scaler-min-scale): update() never backs the scale off below this floor. For isolated non-finite
      gradients that occur at any scale (docs/perf_opt.md (fp16 loss scaling in RL)), backing off only drives the
      scale toward gradient underflow; the step is still skipped by the non-finite guard.
    - on a step whose unscaled gradients are non-finite, records the names of the parameters with non-finite grads
      (self.bad, reset by update()), to localise the source. Costs one scan only on overflow steps."""
    def __init__(self, names, min_scale=0.0, max_scale=0.0, **kw):
        super().__init__("cuda", **kw)
        self.names, self.min_scale, self.max_scale, self.bad = names, float(min_scale or 0.0), float(max_scale or 0.0), []

    def unscale_(self, optimizer):
        super().unscale_(optimizer)
        st = self._per_optimizer_states[id(optimizer)]
        if sum(float(v.item()) for v in st["found_inf_per_device"].values()) > 0:
            for g in optimizer.param_groups:
                for p in g["params"]:
                    if p.grad is not None and not bool(torch.isfinite(p.grad).all()):
                        self.bad.append(self.names.get(id(p), "?"))

    def update(self, new_scale=None):
        self.last_bad, self.bad = self.bad, []
        super().update(new_scale)
        if self.min_scale and self._scale is not None and float(self._scale.item()) < self.min_scale:
            self._scale.fill_(self.min_scale)
        if self.max_scale and self._scale is not None and float(self._scale.item()) > self.max_scale:
            self._scale.fill_(self.max_scale)


class Trainer:
    def __init__(self, raw, dev, args):
        self.raw, self.dev, self.a = raw, dev, args
        import torch._dynamo as dynamo
        dynamo.config.cache_size_limit = 64
        self.trunk = torch.compile(lambda x: raw.trunk(x)[0], dynamic=False)
        self.old = None
        if args.mode == "2gpu":
            self.old = copy.deepcopy(raw).eval()
            for p in self.old.parameters():
                p.requires_grad_(False)
            old = self.old
            self.old_trunk = torch.compile(lambda x: old.trunk(x)[0], dynamic=False)
        if args.optim == "normuon":
            self.mu, self.adam = build_optimizers(raw, muon_lr=0.018, lr_mult=args.muon_lr / 0.018, muon_wd=0.0)
            for g in self.adam.param_groups:
                g["lr"] = g["base_lr"] = args.adam_lr
                g["weight_decay"] = 0.0
            self.opts = [self.mu, self.adam]
        else:
            ps = [p for p in raw.parameters() if p.requires_grad]
            self.adam = torch.optim.AdamW(ps, lr=args.adam_lr, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0,
                                          fused=True)
            for g in self.adam.param_groups:
                g["base_lr"] = g["lr"]
            self.mu = None
            self.opts = [self.adam]
        # growth_interval: torch default 2000; RL arms with sporadic fp16 overflows use ~100 so one overflow does not
        # halve the loss scale for the rest of the run (docs/perf_opt.md (fp16 loss scaling in RL))
        self.scaler = RLScaler({id(p): n for n, p in raw.named_parameters()}, getattr(args, "scaler_min_scale", 0.0),
                               getattr(args, "scaler_max_scale", 0.0), growth_interval=getattr(args, "scaler_growth_interval", 2000))
        self.params = [p for p in raw.parameters() if p.requires_grad]
        self._cc_calls = 0
        self.nonfinite_consec = self.nonfinite_total = 0
        self.ppo = parse_ppo_schedule(getattr(args, "ppo_schedule", ""))

    def step(self, k, seqs, advs, blps=None):
        """blps: optional per-sequence behaviour log-probs of the response tokens (len(t) - prompt_len floats, the
        sampler's log softmax(logits / T)); with them, truncated IS (cap --tis-cap) uses these instead of a stale
        copy (streaming rollouts)."""
        a = self.a
        f = min(1.0, (k + 1) / a.warmup) if a.warmup else 1.0
        for o in self.opts:
            for g in o.param_groups:
                g["lr"] = g["base_lr"] * f
        self.raw.train()
        if getattr(a, "grpo_norm", 0):  # GRPO 1/|o_i|: with the n_seq * max_new normaliser -> mean_i (1/|o_i|) sum_t
            advs = [adv * a.max_new / max(1, len(t) - s) for (t, s), adv in zip(seqs, advs)]
        ent_coef = getattr(a, "entropy_coef", 0.0)
        if self.ppo:
            return self._ppo_step(k, f, seqs, advs, blps, ent_coef)
        ex = [(t, s, adv) for (t, s), adv in zip(seqs, advs) if adv != 0.0 or ent_coef > 0]
        n_seq, n_ex = len(seqs), len(ex)
        if DIST["on"]:  # global Dr.GRPO normaliser and global "any gradient" decision
            c = torch.tensor([n_seq, n_ex], device=self.dev, dtype=torch.float64)
            dist.all_reduce(c)
            n_seq, n_ex = int(c[0].item()), int(c[1].item())
        norm = max(n_seq, 1) * a.max_new  # Dr.GRPO constant normaliser (all ranks' sequences)
        mbs = micro_batches([(t, s) for t, s, _ in ex], a.tok_budget, a.pad_to)
        amap = {id(t): adv for t, s, adv in ex}
        bmap = {id(t): b for (t, s), b in zip(seqs, blps)} if blps is not None else None
        built = []
        for mb in mbs:
            x, y = make_xy(mb, a.pad_to, a.tok_budget)
            A = torch.zeros(y.shape, dtype=torch.float32)
            for i, (t, s) in enumerate(mb):
                A[i] = amap[id(t)]
            if bmap is not None:
                OL = torch.zeros(y.shape, dtype=torch.float32)
                for i, (t, s) in enumerate(mb):
                    b = bmap[id(t)]
                    assert len(b) == len(t) - s, (len(b), len(t), s)
                    OL[i, s - 1:len(t) - 1] = torch.from_numpy(np.asarray(b, dtype=np.float32))
                built.append((x.to(self.dev), y.to(self.dev), A.to(self.dev), OL.to(self.dev)))
            else:
                built.append((x.to(self.dev), y.to(self.dev), A.to(self.dev), None))
        olds = [b[3] for b in built]
        built = [b[:3] for b in built]
        cc_stats = {}
        if getattr(a, "clip_cov", 0.0) > 0 and built:
            cc_stats = self.clip_cov(built)
        if not built and n_ex == 0:  # every group (on every rank) had equal rewards: no gradient this step
            if self.old is not None:
                with torch.no_grad():
                    for po, pr in zip(self.old.parameters(), self.raw.parameters()):
                        po.copy_(pr)
            return {"train_seqs": 0, "grad_norm": 0.0, "lr_f": f, "grad_scale": self.scaler.get_scale(),
                    "tis_mean": 1.0, "tis_clip_frac": 0.0}
        if self.old is not None:
            for j, (x, y, A) in enumerate(built):
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    h = self.old_trunk(x)
                olds[j] = token_logp(h, self.old.embed.weight, y, self.raw.cfg.softcap, a.temperature)
            with torch.no_grad():  # old <- W_k (the weights that sample the next rollout's successor)
                for po, pr in zip(self.old.parameters(), self.raw.parameters()):
                    po.copy_(pr)
        st = torch.zeros(3, device=self.dev)
        for (x, y, A), ol in zip(built, olds):
            with torch.autocast("cuda", dtype=torch.float16):
                h = self.trunk(x)
            loss, s = pg_loss(h, self.raw.embed.weight, y, A, ol, self.raw.cfg.softcap, a.tis_cap, a.temperature,
                              ent_coef=ent_coef)
            self.scaler.scale(loss / norm).backward()
            st += s
        gn, finite = self._apply(k, st)
        st = st.tolist()
        return {"train_seqs": n_ex, "grad_norm": float(gn), "lr_f": f, "grad_scale": self.scaler.get_scale(),
                "tis_mean": st[0] / max(st[2], 1), "tis_clip_frac": st[1] / max(st[2], 1),
                "nonfinite_skip": int(not finite), "nonfinite_total": self.nonfinite_total, **cc_stats}

    def _apply(self, k, st):
        """All-reduce the gradients (and the stats st, in place), unscale, clip, step (unless non-finite), update the
        scaler, zero the gradients. Returns (grad norm, finite)."""
        a = self.a
        if DIST["on"]:  # sum of per-rank gradients = gradient of the global loss (identical scalers on all ranks)
            gs = []
            for p in self.params:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                gs.append(p.grad)
            buf = torch._utils._flatten_dense_tensors(gs)
            dist.all_reduce(buf)
            for g_, v_ in zip(gs, torch._utils._unflatten_dense_tensors(buf, gs)):
                g_.copy_(v_)
            dist.all_reduce(st)
        for o in self.opts:
            self.scaler.unscale_(o)
        gn = torch.nn.utils.clip_grad_norm_(self.params, a.grad_clip if a.grad_clip > 0 else float("inf"))
        # non-finite guard: skip the optimizer step if the (unscaled, all-reduced) gradient or any parameter is
        # non-finite. A finite step makes exactly the same calls as before; an inf/NaN gradient was already skipped
        # by GradScaler.step, so the guard only adds the parameter check, the counter and the abort.
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
        return gn, finite

    def _ppo_step(self, k, f, seqs, advs, blps, ent_coef):
        """--ppo-schedule: the batch is cut into M minibatches of whole groups (a group = a run of sequences with the
        same prompt; cuts at the group starts nearest n * p / M) and swept E times, one optimizer step per minibatch per pass, with the
        PPO clip against pi_old = the trainer's log-probs before the first step. Each step's normaliser is its
        minibatch's sequence count (all ranks) x max_new, so a step has the scale of a single-step update."""
        a = self.a
        M, E = ppo_rung(self.ppo, k)
        cuts = ppo_cuts(seqs, M)
        parts, cnt = [], []
        for p in range(M):
            ex = [(seqs[i][0], seqs[i][1], advs[i], None if blps is None else blps[i]) for i in range(cuts[p], cuts[p + 1])
                  if advs[i] != 0.0 or ent_coef > 0]
            parts.append(ex)
            cnt += [cuts[p + 1] - cuts[p], len(ex)]
        if DIST["on"]:
            c = torch.tensor(cnt, device=self.dev, dtype=torch.float64)
            dist.all_reduce(c)
            cnt = [int(v) for v in c.tolist()]
        sc = self.raw.cfg.softcap

        def build(ex):  # micro-batches of one minibatch, with pi_old = the trainer's log-probs now (before any step)
            amap = {id(t): (adv, b) for t, s, adv, b in ex}
            bp = []
            for mb in micro_batches([(t, s) for t, s, _, _ in ex], a.tok_budget, a.pad_to):
                x, y = make_xy(mb, a.pad_to, a.tok_budget)
                A = torch.zeros(y.shape, dtype=torch.float32)
                OL = torch.zeros(y.shape, dtype=torch.float32) if blps is not None else None
                for i, (t, s) in enumerate(mb):
                    adv, b = amap[id(t)]
                    A[i] = adv
                    if OL is not None:
                        assert len(b) == len(t) - s, (len(b), len(t), s)
                        OL[i, s - 1:len(t) - 1] = torch.from_numpy(np.asarray(b, dtype=np.float32))
                x, y, A = x.to(self.dev), y.to(self.dev), A.to(self.dev)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    h = self.trunk(x)
                PL = token_logp(h, self.raw.embed.weight, y, sc, a.temperature).reshape(y.shape)
                bp.append((x, y, A, None if OL is None else OL.to(self.dev), PL))
            return bp

        def mbx(ix):
            return [(seqs[i][0], seqs[i][1], advs[i], None if blps is None else blps[i]) for i in ix
                    if advs[i] != 0.0 or ent_coef > 0]

        built = [build(ex) for ex in parts]
        every = getattr(a, "grad_probe_every", 0)
        gp, g0 = [], None
        gst = group_starts(seqs) + [len(seqs)] if every and k % every == 0 else []
        if len(gst) > 2:  # gradient-noise probe: two random halves' gradients at the start and after each pass
            # (random halves of the groups: the batch order is not exchangeable, e.g. streamed groups arrive in completion order)
            grp = list(zip(gst[:-1], gst[1:]))
            perm = np.random.default_rng(1000003 + k).permutation(len(grp))
            hg = [sorted(perm[:len(grp) // 2]), sorted(perm[len(grp) // 2:])]
            hb = [build(mbx([i for g in h for i in range(*grp[g])])) for h in hg]
            hng = [len(h) for h in hg]
            hnorm = [max(sum(grp[g][1] - grp[g][0] for g in h), 1) * a.max_new for h in hg]

            def probe():
                nonlocal g0
                for o in self.opts:
                    o.zero_grad(set_to_none=True)
                gs = []
                for p in range(2):
                    for x, y, A, OL, PL in hb[p]:
                        with torch.autocast("cuda", dtype=torch.float16):
                            h = self.trunk(x)
                        loss, _ = pg_loss(h, self.raw.embed.weight, y, A, OL, sc, a.tis_cap, a.temperature,
                                          ent_coef=ent_coef, ppo_lp=PL, clip=(a.ppo_clip_low, a.ppo_clip_high))
                        self.scaler.scale(loss / hnorm[p]).backward()
                    gs.append(torch.cat([(q.grad if q.grad is not None else torch.zeros_like(q)).reshape(-1).float()
                                         for q in self.params]) / self.scaler.get_scale())
                    for o in self.opts:
                        o.zero_grad(set_to_none=True)
                st, g = probe_stats(gs[0], gs[1], hng[0], hng[1], g0)
                g0 = g if g0 is None else g0
                gp.append(st)

            probe()
        tot, kl_last, gns = torch.zeros(6, device=self.dev), torch.zeros(2, device=self.dev), []
        skipped = 0
        for e in range(E):
            for p in range(M):
                if cnt[2 * p + 1] == 0:  # no nonzero advantage in this minibatch on any rank
                    continue
                norm = max(cnt[2 * p], 1) * a.max_new
                st = torch.zeros(6, device=self.dev)
                for x, y, A, OL, PL in built[p]:
                    with torch.autocast("cuda", dtype=torch.float16):
                        h = self.trunk(x)
                    loss, s = pg_loss(h, self.raw.embed.weight, y, A, OL, sc, a.tis_cap, a.temperature,
                                      ent_coef=ent_coef, ppo_lp=PL, clip=(a.ppo_clip_low, a.ppo_clip_high))
                    self.scaler.scale(loss / norm).backward()
                    st += s
                gn, finite = self._apply(k, st)
                gns.append(float(gn))
                skipped += int(not finite)
                tot += st
                if e == E - 1:
                    kl_last += torch.stack([st[5], st[2]])
            if gp:  # after pass e
                probe()
        gpd = {}
        for i, st in enumerate(gp):  # probe i: before pass i (i = E: after the last pass)
            gpd.update({f"gp_{n}_{i}": v for n, v in st.items()})
        if gp:
            gpd.update(gp_groups=sum(hng), gp_bnoise_0=gp[0]["trs"] / gp[0]["g2"] if gp[0]["g2"] > 0 else float("nan"))
        if not gns:
            return {"train_seqs": 0, "grad_norm": 0.0, "lr_f": f, "grad_scale": self.scaler.get_scale(),
                    "tis_mean": 1.0, "tis_clip_frac": 0.0, "ppo_steps": 0, "ppo_minibatches": M, "ppo_epochs": E}
        t, kl = tot.tolist(), kl_last.tolist()
        return {"train_seqs": sum(cnt[1::2]), "grad_norm": float(np.mean(gns)), "grad_norm_max": float(np.max(gns)),
                "lr_f": f, "grad_scale": self.scaler.get_scale(),
                "tis_mean": t[0] / max(t[2], 1), "tis_clip_frac": t[1] / max(t[2], 1),
                "ppo_clip_low_frac": t[3] / max(t[2], 1), "ppo_clip_high_frac": t[4] / max(t[2], 1),
                "ppo_kl_last": kl[0] / max(kl[1], 1), "ppo_steps": len(gns), "ppo_minibatches": M, "ppo_epochs": E,
                "nonfinite_skip": skipped, "nonfinite_total": self.nonfinite_total, **gpd}

    @torch.no_grad()
    def params_finite(self):
        return bool(torch.stack([p.isfinite().all() for p in self.params]).all())

    def state_dict(self):
        """Resume state: optimizer(s), GradScaler and the non-finite counters (weights are saved separately)."""
        return {"opts": [o.state_dict() for o in self.opts], "scaler": self.scaler.state_dict(),
                "nonfinite_consec": self.nonfinite_consec, "nonfinite_total": self.nonfinite_total}

    def load_state_dict(self, d):
        assert len(d["opts"]) == len(self.opts), "resume: optimizer layout differs (--optim)"
        for o, sd in zip(self.opts, d["opts"]):
            o.load_state_dict(sd)
        self.scaler.load_state_dict(d["scaler"])
        # the flag wins over the saved growth_interval (it may be changed on a resume)
        self.scaler.set_growth_interval(getattr(self.a, "scaler_growth_interval", 2000))
        self.nonfinite_consec, self.nonfinite_total = d["nonfinite_consec"], d["nonfinite_total"]

    @torch.no_grad()
    def clip_cov(self, built):
        """Clip-Cov (Cui et al. arXiv 2505.22617 eq. 10-12): over this rank's response tokens, Cov_i =
        (log pi(y_i) - mean log pi)(A_i - mean A) with log pi from the CURRENT weights (one extra no-grad forward);
        floor(r * N) tokens drawn uniformly from those with cov_lb < Cov_i < cov_ub are detached from the policy
        gradient (their advantage is zeroed in place). Returns log stats."""
        a = self.a
        lps = []
        for x, y, A in built:
            with torch.autocast("cuda", dtype=torch.float16):
                h = self.trunk(x)
            lps.append(token_logp(h, self.raw.embed.weight, y, self.raw.cfg.softcap, a.temperature).reshape(y.shape))
        ms = [y != -1 for _, y, _ in built]
        lp_all = torch.cat([lp[m] for lp, m in zip(lps, ms)])
        A_all = torch.cat([A[m] for (_, _, A), m in zip(built, ms)])
        N = lp_all.numel()
        cov = (lp_all - lp_all.mean()) * (A_all - A_all.mean())
        cand = torch.nonzero((cov > a.cov_lb) & (cov < a.cov_ub)).flatten()
        n_sel = min(int(a.clip_cov * N), cand.numel())
        g = torch.Generator(device=cov.device)
        g.manual_seed(int(a.seed) * 1000003 + int(self._cc_calls))
        self._cc_calls += 1
        sel = cand[torch.randperm(cand.numel(), generator=g, device=cov.device)[:n_sel]]
        keep = torch.ones(N, device=cov.device)
        keep[sel] = 0.0
        o = 0
        for (_, y, A), m in zip(built, ms):
            n = int(m.sum())
            A[m] = A[m] * keep[o:o + n]
            o += n
        return {"clipcov_tokens": N, "clipcov_candidates": int(cand.numel()), "clipcov_detached": int(n_sel),
                "cov_mean": float(cov.mean()), "cov_max": float(cov.max()) if N else 0.0}


def sync(gen_m, raw):
    with torch.no_grad():
        for pg, pt in zip(gen_m.parameters(), raw.parameters()):
            pg.copy_(pt)
    gen_m.fuse_for_inference(torch.float16)
    torch.cuda.synchronize(gen_m.embed.weight.device)


def gen(gen_m, prompts, max_new, temp, slots, top_p=1.0):
    with torch.cuda.device(gen_m.embed.weight.device):
        outs = generate_continuous(gen_m, prompts, max_new, slots, temperature=temp, eos_id=EOS, compile=True,
                                   **({"top_p": top_p} if top_p < 1.0 else {}))
    return [[t for t in o if t != EOS] for o in outs], outs


NOTHINK = "</think>\n\n"


def extract_direct(text):
    """--answer-direct: the answer is whatever follows </think> (a \\boxed{} there still wins; else the first
    non-empty line, trailing period / $ stripped). Without a closed think block only a \\boxed{} counts."""
    if "</think>" not in text:
        return last_boxed(text)
    tail = text.split("</think>", 1)[1]
    b = last_boxed(tail)
    if b is not None:
        return b
    lines = [l.strip() for l in tail.strip().splitlines() if l.strip()]
    if not lines:
        return None
    return lines[0].strip("$ ").rstrip(".").strip() or None


def item_prompt(p):
    return p.get("prompt") or prompt_text(p["problem"])


def item_ref(p):
    r = p.get("ref")
    return p.get("answer") if r is None else r


def answer_suffix(answer_only, answer_direct):
    """Text appended to the prompt for the answer-only condition (and generated-text prefix for scoring)."""
    if not answer_only:
        return ""
    return NOTHINK if answer_direct else ANS


# Methods that read rewards as 0/1 (r > 0 = success). On a pool with graded (partial-credit) tasks they would
# silently count any partial credit as a success, so main() refuses them unless --allow-binary-methods. Graded-safe:
# --adv grpo (r - mean, optional std norm) / tailrl, --dyn-sampling (zero reward range), samplers uniform / pool
# (buckets by group-mean reward, i.e. mean score on graded tasks).
BINARY_ONLY_ADV = ("maxrl", "psr", "nsr", "wreinforce")


def binary_only(a):
    """The selected methods that assume binary rewards (empty list = graded-safe configuration)."""
    out = []
    if a.adv in BINARY_ONLY_ADV:
        out.append(f"--adv {a.adv}")
    if getattr(a, "resample_zero", 0):
        out.append("--resample-zero")
    return out


def graded_tasks(rows):
    """Sorted names of the graded tasks among rows (rows without a `task` field use the binary math checker)."""
    from ultraviolet.tasks.score import reward_kind
    return sorted({r["task"] for r in rows if r.get("task") and reward_kind(r["task"]) != "binary"})  # graded / ratio


def task_score(item, full_text):
    from ultraviolet.tasks import score
    return float(score(item["task"], full_text, item))


GRADER = "v1"  # --grader; v1 (eval_math.correct) is the default, v2 (grader_v2) is opt-in


def set_grader(version):
    """Select the math answer checker used by correct() (reward, in-run evals, reward_pool): v1 or v2."""
    global GRADER
    grader(version)
    GRADER = version


def correct(pred, ref):
    return grader(GRADER)(pred, ref)


def reward(tk, close_id, o, item, answer_only=False, answer_direct=False):
    """1/0 reward. Items with a `task` field are scored by ultraviolet.tasks.score(task, completion, item) on the
    completion text (prefixed with the forced suffix in answer-only mode); others by the math extract/correct. The
    normal (thinking) condition requires a closed </think>."""
    txt = tk.decode(o, skip_special_tokens=False)
    if answer_only:
        full = answer_suffix(True, answer_direct) + txt
        if item.get("task"):
            return task_score(item, full), True
        pred = extract_direct(full) if answer_direct else ans_extract(txt)
        return float(correct(pred, item_ref(item))), True
    if close_id not in o:
        return 0.0, False
    if item.get("task"):
        return task_score(item, txt), True
    return float(correct((extract_direct if answer_direct else extract)(txt), item_ref(item))), True


def strict_ok(row, close_id, a):
    """--strict-format (R1-style well-formed response): the rollout (generated tokens, EOS included when it stopped)
    has exactly one </think> and ends with EOS (not truncated at max-new). Always True when the flag is off or in
    answer-only mode."""
    if not getattr(a, "strict_format", 0) or a.answer_only:
        return True
    return wellformed_row(row, close_id)


def wellformed_row(row, close_id):
    """Exactly one </think> and stopped by EOS (logged as wellformed_frac whether or not --strict-format is on)."""
    row = list(row)
    return row.count(close_id) == 1 and len(row) > 0 and row[-1] == EOS


def format_ok(tk, close_id, o, answer_direct=False):
    """--reward-mode format: 1 iff the rollout closes </think> and puts a non-empty \\boxed{...} after it (the
    answer-direct format; otherwise anywhere extract finds one), regardless of correctness."""
    if close_id not in o:
        return 0.0
    txt = tk.decode(o, skip_special_tokens=False)
    b = last_boxed(txt.split("</think>", 1)[1]) if answer_direct else last_boxed(txt)
    return float(b is not None and str(b).strip() != "")


def train_reward(true_r, closed, o, tk, close_id, a, rng):
    """Reward the policy is trained on: correct (the true 0/1 reward), format (format_ok), or random
    (Bernoulli(--random-p) per rollout, independent of the content)."""
    mode = getattr(a, "reward_mode", "correct")
    if mode == "correct":
        return true_r
    if mode == "format":
        return format_ok(tk, close_id, o, bool(getattr(a, "answer_direct", 0)))
    if mode == "random":
        return float(rng.random() < a.random_p)
    raise ValueError(mode)


def overlong_pen(L, closed, max_new, L_start):
    """DAPO soft overlong punishment (arXiv 2503.14476 eq. 13), added to the 0/1 reward before the group baseline:
    0 for L <= L_start, -(L - L_start) / (max_new - L_start) for L_start < L <= max_new (-1 at the cap); an unclosed
    (truncated) trace gets the full -1. L_start <= 0 disables it (returns 0)."""
    if L_start <= 0:
        return 0.0
    if not closed:
        return -1.0
    return -min(1.0, max(0.0, (L - L_start) / max(1, max_new - L_start)))


def shaped_rewards(rew, pen, mixed_only=True):
    """Reward + overlong penalty for one group. mixed_only: groups without a correct rollout keep their raw rewards
    (all 0, so zero advantage), i.e. length is only shaped where there is a success to compare against."""
    rew = np.asarray(rew, dtype=np.float64)
    if mixed_only and rew.max() <= 0:
        return rew
    return rew + np.asarray(pen, dtype=np.float64)


def len_only(rew, shaped):
    """1 if the group's raw rewards carry no signal but the shaped ones do (signal from length alone)."""
    rew, shaped = np.asarray(rew), np.asarray(shaped)
    return int(np.ptp(rew) == 0 and np.ptp(shaped) > 0)


def tail_weights(r):
    """TailRL weights w(r_i) (arXiv 2609.02987 Eq. 27) for one group's rewards r >= 0; sum(w) = max(r) (Eq. 107)."""
    r = np.maximum(np.asarray(r, dtype=np.float64), 0.0)
    o = np.argsort(r, kind="stable")
    rs = r[o]
    gaps = np.diff(np.concatenate([[0.0], rs]))
    w = np.empty_like(r)
    w[o] = np.cumsum(gaps / (len(r) - np.arange(len(r))))
    return w


def group_adv(r, kind="grpo", std_norm=False, lam=0.1):
    """Advantages for one group's rewards r (1-D). grpo: Dr.GRPO r - mean (no std). maxrl (arXiv 2602.02710 Alg. 1,
    = prime-rl max_rl): (r - mean) / mean, 0 when mean <= 0 -- with the batch-constant 1/(n_seq * max_new) loss
    normaliser this is sum_j (r_j/K - 1/N) S_j per problem (K successes of N), the order-N MaxRL estimator.
    std_norm (--grpo-norm, grpo only): original GRPO (r - mean) / (std + 1e-6), std with ddof=1 as in verl/OpenRLHF.
    tailrl (arXiv 2609.02987 Eqs. 26-27, 30): w(r) = int_0^r dtau / #{j: r_j > tau} (sorted r_(1) <= ... <= r_(N):
    w_(i) = w_(i-1) + (r_(i) - r_(i-1)) / (N - i + 1), w_(0) = 0), advantage N (w - mean w) (App. C.5: the
    mean-reduced loss); graded-safe (any rewards >= 0), = maxrl on 0/1 rewards (Cor. 14), no hyperparameters.
    psr / nsr / wreinforce (Zhu et al. arXiv 2506.01347, rewards mapped to +-1, no baseline): psr +1 on correct and 0
    on wrong; nsr -1 on wrong and 0 on correct; wreinforce +lam on correct and -1 on wrong (their eq. 9). Binary
    rewards (r > 0 = correct)."""
    r = np.asarray(r, dtype=np.float64)
    m = r.mean()
    if kind == "grpo":
        if std_norm:
            sd = r.std(ddof=1) if len(r) > 1 else 0.0
            return (r - m) / (sd + 1e-6)
        return r - m
    if kind == "maxrl":
        return (r - m) / m if m > 0 else np.zeros_like(r)
    if kind == "tailrl":
        return len(r) * tail_weights(r) - len(r) * tail_weights(r).mean() if r.max() > 0 else np.zeros_like(r)
    pos = r > 0
    if kind == "psr":
        return pos.astype(np.float64)
    if kind == "nsr":
        return -(~pos).astype(np.float64)
    if kind == "wreinforce":
        return np.where(pos, lam, -1.0)
    raise ValueError(kind)


def ds_drop(g, a, totals=None):
    """--dyn-sampling filter for one completed group (True = drop and refill). reward (default): zero reward range.
    adv: every per-sample training advantage is zero (|a| <= 1e-12), computed exactly as for training (shaped rewards,
    --adv) but before --adv-norm batch scaling, which is the only cross-group step; the advantages are cached in
    g["adv"] and reused at batch assembly.
    totals["ds_kept_zero_var_groups"] counts kept groups the reward filter would have dropped."""
    if getattr(a, "dyn_sampling_on", "reward") == "reward":
        return bool(np.ptp(g["rew"]) == 0)
    g["adv"] = np.asarray(group_adv(shaped_rewards(g["rew"], g["pen"], getattr(a, "overlong_mixed_only", 1)),
                                    getattr(a, "adv", "grpo"), **adv_args(a)), dtype=np.float64)
    drop = not bool(np.any(np.abs(g["adv"]) > 1e-12))
    if not drop and totals is not None and np.ptp(g["rew"]) == 0:
        totals["ds_kept_zero_var_groups"] += 1
    return drop


def adv_args(a):
    """group_adv keyword arguments from the run args (old arg namespaces without the new flags -> defaults)."""
    return {"std_norm": bool(getattr(a, "grpo_norm", 0)), "lam": getattr(a, "wr_lambda", 0.1)}


def scale_batch_adv(adv, a):
    """--adv-norm batch: the step's advantages / their std over all ranks' samples (no-op by default)."""
    if getattr(a, "adv_norm", "none") != "batch":
        return adv
    x = np.asarray(adv, dtype=np.float64)
    parts = gather_obj((len(x), float(x.sum()), float((x * x).sum())))  # per-rank (n, sum, sum of squares)
    n = sum(p[0] for p in parts)
    m = sum(p[1] for p in parts) / max(n, 1)
    sd = float(np.sqrt(max(0.0, sum(p[2] for p in parts) / max(n, 1) - m * m)))
    return (x / sd if sd > 1e-8 else x).tolist()  # uniform batches (std ~ 0) unchanged


def encode_prompt(tk, p, a):
    suf = tk.encode(answer_suffix(a.answer_only, a.answer_direct)).ids if a.answer_only else []
    return [EOS] + tk.encode(item_prompt(p)).ids + suf


def group_stats(groups):
    """groups: list of reward arrays (variable size)."""
    allr = np.concatenate([np.asarray(r, dtype=float) for r in groups]) if groups else np.zeros(0)
    # share of groups per difficulty bucket of their own mean reward (rl_sampler pool boundaries 0.25 / 0.75)
    mb = [float(np.mean(r)) for r in groups]
    bk = {"zero": [m <= 0 for m in mb], "low": [0 < m <= 0.25 for m in mb], "mid": [0.25 < m <= 0.75 for m in mb],
          "high": [0.75 < m < 1 for m in mb], "solved": [m >= 1 for m in mb]}
    return {**{f"bucket_{k}": float(np.mean(v)) if groups else 0.0 for k, v in bk.items()},"reward": float(allr.mean()) if allr.size else 0.0,
            "frac_groups_nonzero_adv": float(np.mean([np.std(r) > 0 for r in groups])) if groups else 0.0,
            "frac_groups_all_correct": float(np.mean([np.min(r) > 0 for r in groups])) if groups else 0.0}


def rollout(gen_m, tk, close_id, probs, Gs, a):
    """probs: problems; Gs: group size per problem (int -> same for all). --resample-zero K: groups whose G
    samples all score 0 get K extra samples (one extra generation pass) before their advantages are computed."""
    if isinstance(Gs, int):
        Gs = [Gs] * len(probs)
    enc = [encode_prompt(tk, p, a) for p in probs]
    owner = [j for j, G in enumerate(Gs) for _ in range(G)]
    torch.cuda.synchronize(gen_m.embed.weight.device)
    t0 = time.time()
    clean, raw_outs = gen(gen_m, [enc[j] for j in owner], a.max_new, a.temperature, a.slots)
    t_gen = time.time() - t0
    t0 = time.time()
    rew = [[] for _ in probs]
    pen = [[] for _ in probs]  # --overlong-soft shaping (0 when off); rew stays the raw 0/1 reward
    rew_true = []  # true correctness reward (rew is the --reward-mode training reward)
    seqs_by = [[] for _ in probs]
    closed, n_gen, n_out = [], 0, 0

    def score(owners, cl, ro_):
        nonlocal n_gen, n_out
        for j, o, ro in zip(owners, cl, ro_):
            r, c = reward(tk, close_id, o, probs[j], a.answer_only, a.answer_direct)
            rew_true.append(r)
            r = train_reward(r, c, o, tk, close_id, a, RRNG)
            rew[j].append(r)
            pen[j].append(overlong_pen(len(ro), c, a.max_new, getattr(a, "overlong_soft", 0)))
            closed.append(c)
            n_gen += len(ro)
            n_out += 1
            seqs_by[j].append((np.array(enc[j] + ro, dtype=np.int64), len(enc[j])))

    score(owner, clean, raw_outs)
    n_resampled = 0
    K = getattr(a, "resample_zero", 0)
    if K > 0:
        zero = [j for j in range(len(probs)) if max(rew[j]) == 0]
        if zero:
            own2 = [j for j in zero for _ in range(K)]
            t1 = time.time()
            c2, r2 = gen(gen_m, [enc[j] for j in own2], a.max_new, a.temperature, a.slots)
            t_gen += time.time() - t1
            score(own2, c2, r2)
            n_resampled = len(zero)
    seqs, adv = [], []
    n_len_only = 0
    for j in range(len(probs)):
        seqs += seqs_by[j]
        sh = shaped_rewards(rew[j], pen[j], getattr(a, "overlong_mixed_only", 1))
        n_len_only += len_only(rew[j], sh)
        adv += group_adv(sh, a.adv, **adv_args(a)).tolist()
    adv = scale_batch_adv(adv, a)
    pf = [x for p_ in pen for x in p_]
    g = {"_group_means": [float(np.mean(r)) for r in rew],
         "overlong_frac": float(np.mean([x < 0 for x in pf])), "overlong_pen_mean": float(np.mean(pf)),
         "frac_groups_len_only": n_len_only / max(1, len(probs)), "reward_true": float(np.mean(rew_true)),
         "t_gen": t_gen, "t_reward": time.time() - t0,
         "gen_tok_s": n_gen / max(t_gen, 1e-9), "close_rate": float(np.mean(closed)), "mean_len": n_gen / n_out,
         "n_rollouts": n_out, "n_resampled_groups": n_resampled,
         "frac_groups_rescued": float(np.mean([max(rew[j]) > 0 for j in range(len(probs))
                                               if len(rew[j]) > Gs[j]])) if n_resampled else 0.0}
    g.update(group_stats([r[:G] for r, G in zip(rew, Gs)]))  # reward stats on the first G samples (comparable)
    return seqs, adv, g


# ------------------------------------------------------------------------------------------------------- eval
def run_eval(gen_m, tk, close_id, evals, gr_ids, a, full, big=None):
    """full: all eval problems, conditions a/b/c at T x full_samples. light: guess-resistant eval subset, a and c at
    T x eval_samples. `big` (held-out guess-resistant pool slice, gr_big) is appended to either and gets all of a/b/c
    at T x k. Sampled rows report acc (= mean pass@1 over the k samples), pass@k and maj@k (majority vote over the
    extracted answers). Greedy (k=1) runs only with a.greedy (diagnostic; off by default)."""
    ps = list(evals) if full else [e for e in evals if e["id"] in gr_ids]
    n_e = len(ps)
    if big:
        ps = ps + list(big)
    P = len(ps)
    is_big = np.arange(P) >= n_e
    k = a.full_samples if (full or big) else a.eval_samples
    pre = [[EOS] + tk.encode(item_prompt(e)).ids for e in ps]
    direct = bool(getattr(a, "answer_direct", 0))
    c_suf = answer_suffix(True, direct)
    enc_ans, enc_nt = tk.encode(c_suf).ids, tk.encode(NOTHINK).ids
    ext_t = extract_direct if direct else extract
    both = ([0.0] if getattr(a, "greedy", 0) else []) + [a.eval_temp]
    # (prompts, max_new, wrap -> text for the math extractor, extractor, temps, full-text wrap for task scoring)
    conds = {"a_normal": (pre, a.max_think, lambda t: t, ext_t, both, lambda t: t),
             "c_answer_only": ([p + enc_ans for p in pre], 16, (lambda t: c_suf + t) if direct else (lambda t: t),
                               ext_t if direct else ans_extract, both, lambda t: c_suf + t)}
    if full or big:
        conds["b_nothink"] = ([p + enc_nt for p in pre], 512, lambda t: NOTHINK + t, ext_t, both, lambda t: NOTHINK + t)
    subsets = {"all": ~is_big, "gr": np.array([e["id"] in gr_ids for e in ps]) & ~is_big}
    nomm = np.array([e["source"] != "metamath" for e in ps]) & ~is_big  # metamath refs unreliable
    subsets["nomm"], subsets["gr_nomm"] = nomm, nomm & subsets["gr"]
    for src in sorted({e["source"] for e in ps[:n_e]}):
        subsets[f"src_{src}"] = np.array([e["source"] == src for e in ps]) & ~is_big
        subsets[f"gr_src_{src}"] = subsets[f"src_{src}"] & subsets["gr"]
    if big:
        subsets["gr_big"] = is_big.copy()
        for src in sorted({e["source"] for e in big}):
            subsets[f"gr_big_src_{src}"] = np.array([e["source"] == src for e in ps]) & is_big
    main_sets = ("all", "gr", "nomm", "gr_nomm", "gr_big")
    res = {}
    t0 = time.time()
    for c, (prompts, mx, wrap, ext, temps, fwrap) in conds.items():
        for temp in temps:
            n = 1 if temp == 0 else k
            # light eval: b/c only on gr_big rows when the 500-set part is light
            outs, _ = gen(gen_m, prompts * n, mx, temp, 512)
            ok = np.zeros((n, P), dtype=bool)
            clo = np.zeros((n, P), dtype=bool)
            lens = np.zeros((n, P))
            cl_n = np.zeros(P)
            cl_ok = np.zeros(P)
            preds = [[None] * P for _ in range(n)]
            for i, o in enumerate(outs):
                raw_t = tk.decode(o, skip_special_tokens=False)
                txt = wrap(raw_t)
                s_, j = divmod(i, P)
                if c != "a_normal" or close_id in o:
                    try:
                        pv = ext(txt)
                    except Exception:  # noqa: BLE001
                        pv = None
                    preds[s_][j] = None if pv is None else norm(str(pv))
                if ps[j].get("task"):
                    hit = task_score(ps[j], fwrap(raw_t)) >= 1.0  # graded wb_ tasks: solved = full reward
                else:
                    hit = correct(ext(txt), item_ref(ps[j]))
                ok[s_, j] = hit and (c != "a_normal" or close_id in o)
                clo[s_, j] = close_id in o
                lens[s_, j] = len(o)
                if c == "a_normal":
                    x, y = arith_claims(txt.split("</think>", 1)[0])
                    cl_n[j] += x
                    cl_ok[j] += y
            tag = "greedy" if temp == 0 else f"t{temp}x{n}"
            for sn, msk in subsets.items():
                if not msk.any():
                    continue
                d = res.setdefault(f"{c}/{tag}/{sn}", {})
                d["acc"] = float(ok[:, msk].mean())
                if n > 1:
                    d[f"pass@{n}"] = float(ok[:, msk].any(0).mean())
                    d[f"maj@{n}"] = float(np.mean([maj_ok([preds[s_][j] for s_ in range(n)], ok[:, j])
                                                   for j in np.nonzero(msk)[0]]))
                if sn in main_sets:
                    d["mean_len"] = float(lens[:, msk].mean())
                    if c == "a_normal":
                        d["close_rate"] = float(clo[:, msk].mean())
                        d["arith_correct_frac"] = float(cl_ok[msk].sum() / max(cl_n[msk].sum(), 1))
                        d["arith_claims_per_trace"] = float(cl_n[msk].sum() / (n * msk.sum()))
    for sn in main_sets:
        for tag in ("greedy", f"t{a.eval_temp}x{k}", f"t{a.eval_temp}x{a.eval_samples}"):
            ka, kb, kc = f"a_normal/{tag}/{sn}", f"b_nothink/{tag}/{sn}", f"c_answer_only/{tag}/{sn}"
            if ka in res and kc in res:
                res[f"delta_a_minus_c/{tag}/{sn}"] = {"acc": res[ka]["acc"] - res[kc]["acc"]}
            if ka in res and kb in res:
                res[f"delta_a_minus_b/{tag}/{sn}"] = {"acc": res[ka]["acc"] - res[kb]["acc"]}
    res["eval_seconds"] = {"s": time.time() - t0}
    return res


def parse_repl_evals(spec):
    """--repl-evals "name:path:every:k,...": held-out sets evaluated every `every` steps (and at step 0 and the last
    step) with k samples per problem at --eval-temp. Returns [(name, path, every, k)]."""
    out = []
    for item in filter(None, spec.split(",")):
        name, path, every, k = item.split(":")
        out.append((name, path, int(every), int(k)))
    return out


_RPOOL = {}


def reward_pool(a):
    """--reward-workers N > 0: a shared thread pool that grades rollouts off the decode loop (code tasks run in a
    sandboxed subprocess, so threads parallelize them). None when off (the default: grading inline, unchanged)."""
    n = getattr(a, "reward_workers", 0)
    if not n:
        return None
    if n not in _RPOOL:
        import concurrent.futures
        _RPOOL[n] = concurrent.futures.ThreadPoolExecutor(n)
    return _RPOOL[n]


def repl_eval_run(gen_m, tk, close_id, sets, a, step, force=False):
    """Replication evals (held-out sets for the paper replications): the thinking condition only, graded exactly like the training
    reward (closed </think>, answer extracted per --answer-direct), k samples at --eval-temp (no greedy). Per set:
    repl_eval.summary (pass@1 = mean accuracy, unbiased pass@k, maj@k, length of correct / incorrect, close rate)."""
    from repl_eval import hash_pred, summary, wellformed
    ext = extract_direct if a.answer_direct else extract
    res = {}
    for name, items, every, k in sets:
        if not (force or step == 0 or step % every == 0):
            continue
        t0 = time.time()
        P = len(items)
        pre = [[EOS] + tk.encode(item_prompt(e)).ids for e in items]
        outs, raw = gen(gen_m, pre * k, a.max_think, a.eval_temp, 1024)
        ok = np.zeros((P, k), bool)
        cl = np.zeros((P, k), bool)
        ln = np.zeros((P, k), np.int64)
        pr = np.zeros((P, k), np.int64)
        wf = np.zeros((P, k), bool)
        rp = reward_pool(a)
        pre_rc = list(rp.map(lambda x: reward(tk, close_id, x[0], items[x[1] % P], False, a.answer_direct),
                             zip(outs, range(len(outs))))) if rp is not None else None
        for i, (o, ro) in enumerate(zip(outs, raw)):
            s_, j = divmod(i, P)
            r, c = pre_rc[i] if pre_rc is not None else reward(tk, close_id, o, items[j], False, a.answer_direct)
            ok[j, s_], cl[j, s_], ln[j, s_] = r >= 1.0, c, len(ro)
            wf[j, s_] = wellformed(ro, close_id, EOS)
            if c:
                try:
                    pr[j, s_] = hash_pred(ext(tk.decode(o, skip_special_tokens=False)))
                except Exception:  # noqa: BLE001
                    pass
        res[name] = summary(ok, ln, cl, pr, wf)
        res[name]["seconds"] = time.time() - t0
    return res


def repl_keys(res):
    """repl_eval_run result -> tr_log keys: sum/<set>_pass1, sum/<set>_maj<k>, sum/<set>_pass<k> (largest k)
    and diag/repl/<set>/<metric>."""
    out = {}
    for name, d in res.items():
        out[f"sum/{name}_pass1"] = d["pass@1"]
        ks = sorted(int(x[5:]) for x in d if x.startswith("pass@") and x[5:].isdigit() and x != "pass@1")
        if ks:
            out[f"sum/{name}_pass{ks[-1]}"] = d[f"pass@{ks[-1]}"]
        for x, v in d.items():
            if x.startswith("maj@"):
                out[f"sum/{name}_maj{x[4:]}"] = v
            if x == "pass@1_strict":
                out[f"sum/{name}_pass1_strict"] = v
            if isinstance(v, (int, float)):
                out[f"diag/repl/{name}/{x.replace('@', '')}"] = v
    return out


RRNG = np.random.default_rng(0)  # --reward-mode random draws (re-seeded in main: seed + 11 + 1000 * rank)


def maj_ok(preds, oks):
    """Majority vote over k samples' extracted answers (None = no answer, never voted; ties -> first seen): correct
    iff a sample carrying the winning answer is correct. 0 when no sample has an answer."""
    cnt = collections.Counter(p for p in preds if p is not None)
    if not cnt:
        return 0.0
    top = cnt.most_common(1)[0][0]
    return float(any(o for p, o in zip(preds, oks) if p == top))


def parse_pool_G(spec, default):
    out = {p: default for p in POOLS}
    for kv in filter(None, spec.split(",")):
        k, v = kv.split("=")
        assert k in out, f"unknown pool {k}"
        out[k] = int(v)
    return out


def group_composition(batch, filtered):
    """Per-step reward accounting on the true reward (this rank; 0/1, or [0, 1] for graded tasks: true_reward is then
    the mean score and true_solved the share of samples with reward 1). true_*: every group completed this step, trained
    plus --dyn-sampling filtered (stale-dropped groups excluded), i.e. the policy's actual reward on what it sampled.
    ds_filtered_*: filtered groups by kind. train_group_p_*: distribution of the per-group pass rate over the trained
    groups only (batch composition, what the gradient sees)."""
    p_tr = np.array([np.mean(g["rew_true"]) for g in batch], dtype=float)
    p_fi = np.array([np.mean(g["rew_true"]) for g in filtered], dtype=float)
    allr = [x for g in list(batch) + list(filtered) for x in g["rew_true"]]
    p_all = np.concatenate([p_tr, p_fi])
    q = lambda v, f: float(np.quantile(v, f)) if v.size else float("nan")
    return {"true_reward": float(np.mean(allr)) if allr else float("nan"),
            "true_solved": float(np.mean(np.asarray(allr) >= 1.0)) if allr else float("nan"), "true_groups": int(p_all.size),
            "true_frac_all_wrong": float(np.mean(p_all == 0)) if p_all.size else float("nan"),
            "true_frac_all_correct": float(np.mean(p_all == 1)) if p_all.size else float("nan"),
            "ds_filtered_all_wrong": int(np.sum(p_fi == 0)), "ds_filtered_all_correct": int(np.sum(p_fi == 1)),
            "ds_filtered_other": int(np.sum((p_fi > 0) & (p_fi < 1))),
            "train_group_p_mean": float(p_tr.mean()) if p_tr.size else float("nan"),
            "train_group_p_q10": q(p_tr, .1), "train_group_p_q50": q(p_tr, .5), "train_group_p_q90": q(p_tr, .9),
            "train_frac_p_lt10": float(np.mean(p_tr < .1)) if p_tr.size else float("nan"),
            "train_frac_p_gt90": float(np.mean(p_tr > .9)) if p_tr.size else float("nan")}


def dump_rollouts(f, step, batch, tk, pool, n):
    """--dump-rollouts: n trained rollouts of this rank's batch (half rewarded when available, seeded by step) with
    item id / family / tier, true and training reward, pre-batch-norm advantage, answer key and decoded response."""
    rng = random.Random(step)
    cand = [(gi, j) for gi, g in enumerate(batch) for j in range(len(g["rew"]))]
    pos = [c for c in cand if batch[c[0]]["rew"][c[1]] > 0]
    pick = rng.sample(pos, min(len(pos), n // 2))
    rest = [c for c in cand if c not in set(pick)]
    pick += rng.sample(rest, min(len(rest), n - len(pick)))
    for gi, j in pick:
        g = batch[gi]
        it = pool[g["i"]]
        seq, n_pre = g["seqs"][j]
        f.write(json.dumps({"step": step, "id": it.get("id"), "family": it.get("family"), "tier": it.get("tier"),
                            "rew_true": float(g["rew_true"][j]), "rew": float(g["rew"][j]),
                            "adv": float(g["adv"][j]) if "adv" in g else None,
                            "group_mean": float(np.mean(g["rew"])),
                            "n_tokens": int(len(seq) - n_pre), "text": tk.decode(seq[n_pre:].tolist(), skip_special_tokens=False)},
                           default=str) + "\n")
    f.flush()


def stream_loop(args, gen_m, raw, tr, tk, close_id, pool, sampler, group_sizes, wb, logf, save, maybe_eval):
    """--stream training loop (1 GPU): generation and training alternate, but the generator's slots never drain.
    Groups are enqueued in batches of n_prompts whenever fewer than slots + n_prompts*group requests are outstanding;
    each optimizer step takes the first n_prompts groups to complete (all their samples finished, after any
    --resample-zero extras) whose oldest sample started at most --max-staleness steps ago."""
    from rl_stream import StreamGen, sync_weights_inplace
    if getattr(args, "perf", ""):  # opt-in speed flags (docs/perf_opt.md); same outputs up to fp tolerance
        from ultraviolet.perf import make_streamgen
        StreamGen = make_streamgen(StreamGen, set(f for f in args.perf.split(",") if f))
    enc_cache = {}

    def enc(i):
        if i not in enc_cache:
            enc_cache[i] = encode_prompt(tk, pool[i], args)
        return enc_cache[i]

    max_prompt = max(len(encode_prompt(tk, p, args)) for p in pool)
    slots = args.slots // DIST["world"]  # --slots is global: same in-flight rollouts per step as one GPU
    gen = StreamGen(gen_m, slots, max_prompt, args.max_new, args.temperature, EOS,
                    check_every=args.stream_check_every)
    gen.version = RS["start"]  # resumed run: in-flight rollouts are not saved, the pipeline restarts empty
    pending, ready = {}, collections.deque()
    gid_item = {}  # group id -> pool index (read by --reward-workers threads; outlives the pending entry)
    rpool = reward_pool(args)
    rfuts = []  # --reward-workers: (future, gid, row, lp, v0) in submission order
    nid = 0
    totals = RS["totals"]

    def enqueue(n):
        nonlocal nid
        idx = sampler.sample(n)
        for i, G in zip(idx, group_sizes(idx)):
            pending[nid] = {"i": i, "G": G, "need": G, "rew": [], "seqs": [], "blp": [], "v0": [], "closed": [],
                            "lens": [], "pen": [], "rew_true": [], "extra": 0}
            gid_item[nid] = i
            totals["groups_started"] += 1
            for _ in range(G):
                gen.add(enc(i), nid)
            nid += 1

    def grade(gid, row):  # --reward-workers: runs in a worker thread; returns (reward, closed)
        return reward(tk, close_id, [t for t in row if t != EOS], pool[gid_item[gid]], args.answer_only,
                      args.answer_direct)

    def finish(gid, row, lp, v0, pre=None):
        totals["rollouts"] += 1  # every generated rollout, incl. ones of dropped groups (compute accounting)
        g = pending.get(gid)
        if g is None:  # group already dropped (stale)
            totals["orphan_rollouts"] += 1
            return
        o = [t for t in row if t != EOS]
        if pre is None:
            r, c = reward(tk, close_id, o, pool[g["i"]], args.answer_only, args.answer_direct)
        else:
            r, c = pre[0], pre[1]
        g["rew_true"].append(r)
        r = train_reward(r, c, o, tk, close_id, args, RRNG)
        wf = wellformed_row(row, close_id)
        g.setdefault("wf", []).append(wf)
        if not strict_ok(row, close_id, args):  # --strict-format: a malformed / truncated response scores 0
            r = 0.0
        e = enc(g["i"])
        g["rew"].append(r)
        g["closed"].append(c)
        g["lens"].append(len(row))
        g["pen"].append(overlong_pen(len(row), c, args.max_new, args.overlong_soft))
        g["seqs"].append((np.array(e + row, dtype=np.int64), len(e)))
        g["blp"].append(lp)
        g["v0"].append(v0)
        if len(g["rew"]) == g["need"]:
            if args.resample_zero > 0 and g["extra"] == 0 and max(g["rew"]) == 0:
                g["extra"] = args.resample_zero
                g["need"] += args.resample_zero
                for _ in range(args.resample_zero):
                    gen.add(e, gid)
                totals["resampled_groups"] += 1
                return
            ready.append(gid)

    def drop_stats(g, acc):
        acc["n"] += 1
        acc["len"] += g["lens"]
        acc["unfinished"] += g["need"] - len(g["rew"])
        acc["rew"] += g["rew"]

    def stale(g):
        return bool(g["v0"]) and gen.version - min(g["v0"]) > args.max_staleness

    groups_f = open(os.path.join(args.out, "groups.jsonl"), "a") if getattr(args, "log_groups", 0) and \
        DIST["rank"] == 0 else None
    roll_f = open(os.path.join(args.out, "rollouts.jsonl"), "a") if getattr(args, "dump_rollouts", 0) and \
        DIST["rank"] == 0 else None

    t_start = time.time() - RS["hours"] * 3600
    prev_tok = 0 if RS["start"] == 0 else None  # --save-at-tokens crossing detection (unknown for the first resumed step)
    npl = args.n_prompts // DIST["world"]  # groups per rank per step (global batch = --n-prompts)
    rpers = npl * args.group
    for k in range(RS["start"], args.steps):
        t0 = time.time()
        tok0, steps0 = gen.n_tokens, gen.n_decode_steps
        n_stale = 0
        dropped = {"n": 0, "len": [], "unfinished": 0, "rew": []}
        batch, filtered = [], []  # filtered: groups dropped by --dyn-sampling
        t_gen = 0.0
        while len(batch) < npl:
            while ready and len(batch) < npl:
                gid = ready.popleft()
                g = pending.pop(gid)
                if stale(g):
                    n_stale += 1
                    drop_stats(g, dropped)
                    continue
                if getattr(args, "dyn_sampling", 0) and ds_drop(g, args, totals):  # DAPO dynamic sampling
                    filtered.append(g)
                    totals["ds_filtered_groups"] += 1
                    continue
                batch.append(g)
            if len(batch) >= npl:
                break
            # rollouts waiting in the --reward-workers queue count as in flight: when decoding outpaces grading
            # (short responses), enqueueing on decode occupancy alone grows the backlog without bound
            if gen.n_outstanding() + len(rfuts) < slots + rpers:
                enqueue(npl)
            t1 = time.time()
            if rpool is None:
                for gid, row, lp, v0 in gen.chunk():
                    finish(gid, row, lp, v0)
            else:  # grade off the decode loop; finish in submission order as results arrive
                for gid, row, lp, v0 in gen.chunk():
                    rfuts.append((rpool.submit(grade, gid, row), gid, row, lp, v0))
                if rfuts and not rfuts[0][0].done() and gen.n_outstanding() == 0:
                    import concurrent.futures
                    concurrent.futures.wait([rfuts[0][0]], timeout=0.05)
                while rfuts and rfuts[0][0].done():
                    f, gid, row, lp, v0 = rfuts.pop(0)
                    finish(gid, row, lp, v0, pre=f.result())
            t_gen += time.time() - t1
            # drop incomplete groups that can no longer meet the staleness bound
            for gid in [gid for gid, g in pending.items() if gid not in ready and stale(g)]:
                drop_stats(pending.pop(gid), dropped)
                gen.drop_queued(lambda m, gid=gid: m == gid)
                n_stale += 1
        totals["stale_groups"] += n_stale
        seqs, adv, blps = [], [], []
        n_len_only = 0
        for g in batch:
            seqs += g["seqs"]
            blps += g["blp"]
            sh = shaped_rewards(g["rew"], g["pen"], args.overlong_mixed_only)
            n_len_only += len_only(g["rew"], sh)
            a_g = g["adv"] if "adv" in g else group_adv(sh, args.adv, **adv_args(args))
            adv += a_g.tolist()  # g["adv"]: --dyn-sampling-on adv computed it at filter time (same values)
        adv = scale_batch_adv(adv, args)  # --adv-norm batch (no-op by default)
        if roll_f is not None and (k + 1) % args.dump_rollouts_every == 0:
            dump_rollouts(roll_f, k + 1, batch, tk, pool, args.dump_rollouts)
        upd = gather_obj(([g["i"] for g in batch], [float(np.mean(g["rew"])) for g in batch],
                          [(int(g["i"]), float(np.mean(g["rew"]))) for g in filtered]))
        sampler.update([i for u in upd for i in u[0]], [m for u in upd for m in u[1]])  # same order on every rank
        if getattr(args, "log_groups", 0) and DIST["rank"] == 0:  # per-problem training accuracy history
            fl = [x for u in upd for x in u[2]]  # --dyn-sampling filtered groups (empty without it)
            groups_f.write(json.dumps({"step": k + 1, "i": [int(i) for u in upd for i in u[0]],
                                       "r": [round(m, 4) for u in upd for m in u[1]],
                                       **({"f_i": [i for i, _ in fl], "f_r": [round(m, 4) for _, m in fl]}
                                          if fl else {})}) + "\n")
            groups_f.flush()
        lens = [x for g in batch for x in g["lens"]]
        pens = [x for g in batch for x in g["pen"]]
        age = [gen.version - v for g in batch for v in g["v0"]]
        dtok = gen.n_tokens - tok0
        totals["gen_tokens"] += dtok  # every generated token incl. dropped / filtered groups (matched-compute axis)
        # policy entropy estimate: -mean log mu(sampled token) at the rollout temperature over all batch tokens (an
        # unbiased estimate of the mean per-token entropy of the sampler); lengths split by true correctness
        nlp = [float(-np.sum(b)) for g in batch for b in g["blp"]]
        ntk = [len(b) for g in batch for b in g["blp"]]
        rt = np.array([x for g in batch for x in g["rew_true"]]) > 0
        la = np.array(lens, dtype=float)
        ent_rec = {"entropy_tok": sum(nlp) / max(1, sum(ntk)),
                   "entropy_seq": float(np.mean([a_ / max(1, b_) for a_, b_ in zip(nlp, ntk)])),
                   "len_correct": float(la[rt].mean()) if rt.any() else float("nan"),
                   "len_incorrect": float(la[~rt].mean()) if (~rt).any() else float("nan")}
        rec_g = {"t_gen": t_gen, "gen_tok_s": dtok / max(t_gen, 1e-9), "decode_steps": gen.n_decode_steps - steps0,
                 **ent_rec,
                 "close_rate": float(np.mean([c for g in batch for c in g["closed"]])), "mean_len": float(np.mean(lens)),
                 "overlong_frac": float(np.mean([x < 0 for x in pens])), "overlong_pen_mean": float(np.mean(pens)),
                 "frac_groups_len_only": n_len_only / max(1, len(batch)),
                 "reward_true": float(np.mean([x for g in batch for x in g["rew_true"]])),
                 "wellformed_frac": float(np.mean([x for g in batch for x in g.get("wf", [True])])),
                 "multi_close_frac": float(np.mean([s_.count(close_id) > 1 for g in batch for s_, _ in
                                                    ((t[n:].tolist(), n) for t, n in g["seqs"])])),
                 "n_rollouts": len(seqs), "stale_dropped": n_stale, "sample_age_mean": float(np.mean(age)),
                 "sample_age_max": int(max(age)), "frac_offpolicy": float(np.mean([x > 0 for x in age])),
                 **{f"age_frac_{x}": float(np.mean([y == x for y in age])) for x in range(4)},
                 "age_frac_4plus": float(np.mean([y >= 4 for y in age])),
                 "dropped_finished_len_mean": float(np.mean(dropped["len"])) if dropped["len"] else float("nan"),
                 "dropped_unfinished": dropped["unfinished"], "dropped_finished": len(dropped["len"]),
                 "dropped_finished_reward": float(np.mean(dropped["rew"])) if dropped["rew"] else float("nan"),
                 "queue": len(gen.queue), "pending_groups": len(pending), **sampler.stats(),
                 **{f"total_{x}": v for x, v in totals.items()}}
        rec_g["rollouts_total"] = totals["rollouts"]
        rec_g.update(group_composition(batch, filtered))
        allr = gather_obj((rec_g, [list(g["rew"][:g["G"]]) for g in batch]))
        rec_g = merge_recs([x[0] for x in allr])
        rec_g.update(group_stats([np.asarray(r) for x in allr for r in x[1]]))
        t1 = time.time()
        ts = tr.step(k, seqs, adv, blps)
        torch.cuda.synchronize()
        if ts.get("nonfinite_skip") and DIST["rank"] == 0:  # which parameters had non-finite grads (RLScaler)
            bad = getattr(tr.scaler, "last_bad", [])
            with open(os.path.join(args.out, "nonfinite.jsonl"), "a") as f_:
                f_.write(json.dumps({"step": k + 1, "grad_scale": ts.get("grad_scale"), "n_bad": len(bad),
                                     "bad": bad}) + "\n")
            print(f"NONFINITE step {k + 1}: {len(bad)} params with non-finite grads, e.g. {bad[:6]}", flush=True)
        t_train = time.time() - t1
        t1 = time.time()
        if DIST["on"] and ((k + 1) % 10 == 0 or k < 3 or (RS["world_change"] and k < RS["start"] + 3)):  # replicated-weight check: spread of a parameter checksum
            cs = torch.stack([p_.detach().double().sum() for p_ in raw.parameters()]).sum().reshape(1)
            cl = [torch.zeros_like(cs) for _ in range(DIST["world"])]
            dist.all_gather(cl, cs)
            ts["ddp_param_spread"] = float(torch.cat(cl).max() - torch.cat(cl).min())
        if DIST["on"] and (k + 1) % 50 == 0:  # guard against slow numerical drift of the replicated weights
            for p_ in raw.parameters():
                dist.broadcast(p_.data, 0)
        sync_weights_inplace(gen_m, raw)
        gen.version += 1
        rec = {"step": k + 1, **rec_g, **ts, "t_train": t_train, "t_sync": time.time() - t1,
               "t_step": time.time() - t0, "hours": (time.time() - t_start) / 3600}
        if k == RS["start"] and RS.get("world_change"):  # elastic resume: marks the first step on the new world size
            rec["resume_world_old"], rec["resume_world_new"] = RS["world_change"]
        RS["train_s"] += rec["t_step"]  # cumulative step time excluding evals (compute axis; --max-hours)
        RS["hours"] = rec["hours"]
        rec["train_hours"] = RS["train_s"] / 3600
        if DIST["on"]:  # identical stop decision on every rank
            th = torch.tensor([rec["train_hours"]], device=torch.device("cuda", torch.cuda.current_device()))
            dist.all_reduce(th, op=dist.ReduceOp.MAX)
            rec["train_hours"] = float(th.item())
        wb.train(rec, k + 1)
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        if DIST["rank"] == 0 and ((k + 1) % 10 == 0 or k < 3):
            print(json.dumps({x: round(v, 4) if isinstance(v, float) else v for x, v in rec.items()}), flush=True)
        last = k + 1 == args.steps or (args.max_rollouts > 0 and rec_g["rollouts_total"] >= args.max_rollouts) or \
            (getattr(args, "max_hours", 0) > 0 and rec["train_hours"] >= args.max_hours)
        cur_tok = rec_g.get("total_gen_tokens", 0)  # --max-gen-tokens stop + --save-at-tokens checkpoints
        last = last or (getattr(args, "max_gen_tokens", 0) > 0 and cur_tok >= args.max_gen_tokens)
        marks = [float(x) for x in filter(None, getattr(args, "save_at_tokens", "").split(","))]
        hit = [m_ for m_ in marks if prev_tok is not None and prev_tok < m_ <= cur_tok]
        if hit and DIST["rank"] == 0:
            with open(os.path.join(args.out, "budget_marks.jsonl"), "a") as f_:
                f_.write(json.dumps({"marks": hit, "step": k + 1, "gen_tokens": cur_tok,
                                     "rollouts": rec_g["rollouts_total"]}) + "\n")
        prev_tok = cur_tok
        if (k + 1) % args.ckpt_every == 0 or last or hit:
            save(k + 1)
        maybe_eval(k + 1, force=last)
        if last:
            break


# ---------------------------------------------------------------------------------------------- data parallel
DIST = {"on": False, "rank": 0, "world": 1}  # set by main() under torchrun (--stream only)
# run state that survives a --resume (stream loop): first step, rollout counters, compute clock
RS = {"start": 0, "totals": collections.Counter(), "train_s": 0.0, "hours": 0.0, "world_change": None}


def rank_share(n, world, rank):
    """rank's share of an integer count n split over world ranks (n // W, plus one for the first n % W ranks), so
    the shares sum exactly to n."""
    return n // world + int(rank < n % world)


def rank_state(sampler):
    """This rank's resume state: sampler (pool / permutation + its rng), every other rng it draws from, and this
    rank's own counters (RS totals; only read by an elastic resume onto a different world size)."""
    return {"sampler": sampler.state_dict(), "rrng": RRNG.bit_generator.state, "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
            "totals": dict(RS["totals"])}


def load_rank_state(d, sampler):
    sampler.load_state_dict(d["sampler"])
    RRNG.bit_generator.state = d["rrng"]
    torch.set_rng_state(d["torch"])
    if d["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(d["cuda"])


def resume_state(step, tr, ranks):
    """--save-state: everything a --resume needs besides the weights (step<N>.pt), as one dict."""
    return {"step": step, "world": len(ranks), "trainer": tr.state_dict(), "ranks": ranks,
            "totals": dict(RS["totals"]), "train_s": RS["train_s"], "hours": RS["hours"]}


def find_resume(spec, out):
    """--resume auto -> the newest state<N>.pt in the run dir whose step<N>.pt exists (None if there is none);
    otherwise the given path."""
    if spec != "auto":
        return spec
    import glob
    import re
    c = []
    for f in glob.glob(os.path.join(out, "state*.pt")):
        m = re.fullmatch(r"state(\d+)\.pt", os.path.basename(f))
        if m and os.path.exists(os.path.join(out, f"step{m.group(1)}.pt")):
            c.append((int(m.group(1)), f))
    return max(c)[1] if c else None


def global_totals(R):
    """All-rank totals of a resume_state: the sum of the per-rank counters when every rank saved its own (rank_state
    "totals"), else the saved master-rank counters x world (exact for world 1; states written before the per-rank
    counters were saved). Returns (totals dict, exact flag)."""
    rk = R.get("ranks") or []
    if rk and all("totals" in d for d in rk):
        c = collections.Counter()
        for d in rk:
            c.update(d["totals"])
        return dict(c), True
    W = int(R["world"])
    return {x: v * W for x, v in R["totals"].items()}, W == 1


def split_totals(tot, world, rank):
    """This rank's share of all-rank counters (ints: rank_share, so the ranks sum exactly to tot; floats: tot / W).
    The stream loop logs per-rank counters summed over ranks (merge_recs), so the logged totals continue exactly."""
    return {x: rank_share(int(v), world, rank) if float(v).is_integer() else float(v) / world for x, v in tot.items()}


def reseed_rank_rngs(seed, rank, step):
    """Elastic resume: per-rank rngs re-derived with the fresh-start convention (RRNG seed + 11 + 1000 r, torch
    seed + r), each moved off its step-0 stream (numpy: PCG64.jumped(step); torch: seeded from (seed + r, step)) so
    the resumed ranks do not replay the start of the run."""
    from rl_sampler import jumped_state
    RRNG.bit_generator.state = jumped_state(np.random.default_rng(seed + 11 + 1000 * rank).bit_generator.state, step)
    torch.manual_seed(int(np.random.default_rng([seed + rank, step]).integers(2 ** 62)))  # cpu + every cuda device


def apply_resume(R, raw, tr, sampler, rank, world=None, seed=0):
    """Restore a resume_state dict (weights must already be loaded into raw). world (the current world size) !=
    R["world"]: elastic resume (resume_elastic); otherwise exactly the per-rank restore."""
    tr.load_state_dict(R["trainer"])
    if world is None or int(world) == int(R["world"]):
        load_rank_state(R["ranks"][rank], sampler)
        RS.update(start=int(R["step"]), totals=collections.Counter(R["totals"]), train_s=float(R["train_s"]),
                  hours=float(R.get("hours", 0.0)))
        return
    resume_elastic(R, sampler, rank, int(world), seed)


def resume_elastic(R, sampler, rank, world, seed):
    """Resume a state saved with W_old ranks onto `world` ranks (1 -> N, N -> M). Carried over (by the caller /
    here): weights, optimizers + GradScaler + trainer counters (replicated, identical on every rank), step (hence the
    LR warmup position), the train clock, and the sampler's shared state (from saved
    rank 0; rl_sampler.rewidth_sampler_state). Counters: all-rank totals (global_totals) split over the new ranks
    (split_totals). Re-derived: per-rank rngs (reseed_rank_rngs) and the sampler's per-rank rng; shards (i % W == r)
    come from the sampler built for the new world. In-flight StreamGen rows are never saved (pipeline restarts
    empty)."""
    from rl_sampler import rewidth_sampler_state
    step, W_old = int(R["step"]), int(R["world"])
    sampler.load_state_dict(rewidth_sampler_state(R["ranks"][0]["sampler"], sampler.state_dict(), step))
    reseed_rank_rngs(seed, rank, step)
    tot, exact = global_totals(R)
    RS.update(start=step, totals=collections.Counter(split_totals(tot, world, rank)), train_s=float(R["train_s"]),
              hours=float(R.get("hours", 0.0)), world_change=(W_old, world))
    return exact


def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def gather_obj(x):
    """all_gather_object -> list over ranks (rank order); [x] when not distributed."""
    if not DIST["on"]:
        return [x]
    out = [None] * DIST["world"]
    dist.all_gather_object(out, x)
    return out


SUM_KEYS = ("n_rollouts", "stale_dropped", "dropped_unfinished", "dropped_finished", "queue", "pending_groups",
            "rollouts_total")


def merge_recs(recs):
    """Per-rank step records -> one: counts summed (SUM_KEYS, total_*), everything else averaged over ranks
    (NaNs ignored); group-level reward stats are recomputed by the caller from all ranks' groups."""
    if len(recs) == 1:
        return recs[0]
    out = {}
    for k in recs[0]:
        v = [r.get(k) for r in recs if isinstance(r.get(k), (int, float))]
        if not v:
            continue
        if k in SUM_KEYS or k.startswith("total_"):
            out[k] = sum(v)
        elif k.endswith("_max"):
            out[k] = max(v)
        else:
            f = [x for x in v if x == x]
            out[k] = float(np.mean(f)) if f else float("nan")
    return out


class TRLog:
    """wandb logging through tr_log (README.md (wandb)); master rank only (args=None: no-op).
    Run id --tr-name (default: --name with mrl- -> rl-)."""
    def __init__(self, args, cfg, world):
        self.on = args is not None
        if not self.on:
            return
        import tr_log
        self.tl = tr_log
        name = args.tr_name or "rl-" + args.name.replace("mrl-", "")
        mix = {"pool": args.tr_pool_kind or args.pool, "pool_file": args.pool, "sampler": args.sampler, "pool_weights": args.pool_weights,
               "pool_G": args.pool_G, "include": args.tr_include or None}
        tr_log.init(name, "rl", parent=args.tr_parent or args.ckpt, data=args.pool, mix=mix, key=bool(args.tr_key),
                    tags=[t for t in args.tr_tags.split(",") if t],
                    ckpt_path=args.out, dir=args.out, run_id=args.tr_run_id or None,
                    config={**cfg, **({"replication": load_meta(args.repl_meta)} if args.repl_meta else {}),
                            **({"recipe": load_meta(args.run_meta)} if args.run_meta else {}),
                            "sampler": args.sampler, "gpus": world,
                            "reward_mode": args.reward_mode,
                            "max_rollouts": args.max_rollouts, "n_prompts": args.n_prompts, "group": args.group})

    def train(self, rec, step):
        if self.on:
            self.tl.log(self.tl.rl_train_keys(rec), step=step)

    def eval(self, res, step):
        if self.on:
            self.tl.log(self.tl.math_keys(res), step=step)

    def repl(self, res, step):
        if self.on and res:
            self.tl.log(repl_keys(res), step=step)

    def raw(self, d, step):
        if self.on and d:
            self.tl.log(d, step=step)

    def finish(self):
        if self.on:
            self.tl.finish()


def load_meta(m):
    """--repl-meta: a JSON string, or @path to a JSON file."""
    return json.load(open(m[1:])) if m.startswith("@") else json.loads(m)


def main():
    ap = argparse.ArgumentParser()
    a = ap.add_argument
    a("--ckpt", required=True)
    a("--tok", required=True)
    a("--pool", required=True)
    a("--filters", required=True)
    a("--eval-problems", required=True)
    a("--out", required=True, help="run dir")
    a("--name", required=True, help="run dir name")
    a("--tr-run-id", default="", help="wandb run id if it must differ from --tr-name (e.g. a restarted run: wandb "
      "never reuses an id)")
    a("--scaler-max-scale", type=float, default=0.0, help="cap for the fp16 loss scale (0 = none; see RLScaler)")
    a("--scaler-min-scale", type=float, default=0.0, help="floor for the fp16 loss scale (0 = none; see RLScaler)")
    a("--scaler-growth-interval", type=int, default=2000, help="fp16 GradScaler growth_interval (torch default 2000)")
    a("--tr-name", default="", help="tr_log wandb id (default: --name with mrl- -> rl-)")
    a("--tr-parent", default="", help="tr_log parent run id (default: --ckpt path)")
    a("--tr-include", default="", help="broad-RL include file (recorded in the run's mix)")
    a("--tr-pool-kind", default="", help="recorded as mix.pool (e.g. plain | gr)")
    a("--tr-tags", default="", help="extra tr_log tags, comma-separated (e.g. control)")
    a("--tr-key", type=int, default=0, help="tag the tr_log run as a headline (key) run")
    a("--mode", default="2gpu", choices=["1gpu", "2gpu"])
    a("--steps", type=int, default=1000)
    a("--n-prompts", type=int, default=64)
    a("--group", type=int, default=16)
    a("--temperature", type=float, default=1.0)
    a("--max-new", type=int, default=2048)
    a("--slots", type=int, default=1024)
    a("--optim", default="normuon", choices=["normuon", "adam"])
    a("--muon-lr", type=float, default=9e-4, help="NorMuon LR (SFT used 0.018*0.15 = 2.7e-3)")
    a("--adam-lr", type=float, default=1e-5, help="AdamW LR for every Adam group (all params with --optim adam)")
    a("--warmup", type=int, default=10)
    a("--tis-cap", type=float, default=2.0)
    a("--ppo-schedule", default="", help="multiple optimizer steps per batch with the PPO clip: comma-separated "
      "MxE[:last_update] rungs, M group-aligned minibatches x E passes, e.g. 2x4:50,2x2:150,2x1 (\"\" = one step "
      "per batch, no clip; see Trainer._ppo_step). Needs --mode 1gpu")
    a("--grad-probe-every", type=int, default=0, help="--ppo-schedule: every N updates, measure the batch's gradient "
      "noise (two random halves' (whole groups) gradients at the start and after each pass, no step): logs gp_cos_i, gp_g2_i, "
      "gp_trs_i, gp_cos0_i (probe i before pass i; see probe_stats) and gp_bnoise_0. 0 = off")
    a("--ppo-clip-low", type=float, default=0.2)
    a("--ppo-clip-high", type=float, default=0.28)
    a("--grad-clip", type=float, default=1.0)
    a("--tok-budget", type=int, default=16384)
    a("--pad-to", type=int, default=512)
    a("--eval-every", type=int, default=100)
    a("--eval-full", type=int, default=1, help="1: all 500 + per source, a/b/c at T x full-samples; 0: GR subset, "
      "a and c at T x eval-samples (gr_big always a/b/c at T x full-samples)")
    a("--full-eval-every", type=int, default=0, help="steps between full evals (0: never; overrides the light eval)")
    a("--eval-samples", type=int, default=16)
    a("--full-samples", type=int, default=16, help="T samples per problem in full / gr_big evals (pass@1 = mean, "
      "maj@k)")
    a("--greedy", type=int, default=0, help="also run greedy (k=1) evals (diagnostic only)")
    a("--gr-big", default=None, help="extra held-out guess-resistant eval set (eval_problems format), evaluated "
      "at every eval with a/b/c")
    a("--answer-only", type=int, default=0, help="RL on the answer-only condition: prompt + ANS, reward = "
      "ans_extract vs ref (use --max-new 16)")
    a("--max-think", type=int, default=2048, help="a_normal eval generation cap")
    a("--eval-temp", type=float, default=1.0)
    a("--ckpt-every", type=int, default=200)
    a("--seed", type=int, default=0)
    a("--sampler", default="uniform", choices=["uniform", "pool"],
      help="uniform: epoch permutations; pool: difficulty pools by latest group-mean reward (prime-rl style)")
    a("--pool-thresholds", default="0.25,0.75", help="lo,hi: pools zero | (0,lo] | (lo,hi] | (hi,1) | solved")
    a("--pool-weights", default=",".join(map(str, DEFAULT_WEIGHTS)),
      help="sampling weights for unseen,zero,low,mid,high,solved")
    a("--dyn-sampling", type=int, default=0, help="--stream: DAPO dynamic sampling, zero-variance groups are dropped "
      "after rollout and the batch is refilled (their rollouts count in rollouts_total / gen_tokens)")
    a("--dyn-sampling-on", default="reward", choices=["reward", "adv"], help="--dyn-sampling filter: reward (default) "
      "drops groups with zero reward range; adv drops a group only if all its training advantages (--adv, before "
      "--adv-norm batch) are zero, e.g. keeps all-correct groups whose --adv still assigns nonzero advantages")
    a("--max-gen-tokens", type=float, default=0, help="--stream: stop once total generated tokens (all groups) reach this")
    a("--save-at-tokens", default="", help="--stream: comma-separated generated-token marks; checkpoint at the first "
      "step at or past each (budget_marks.jsonl)")
    a("--answer-direct", type=int, default=0, help="answer follows </think> directly (no \\boxed{} needed): "
      "extract/reward/eval use the text after </think>; the answer-only condition appends just </think>\\n\\n")
    a("--overlong-soft", type=int, default=0, help="L_start for DAPO soft overlong shaping (0 = off): reward "
      "-(L - L_start)/(max_new - L_start) for longer rollouts (-1 at max_new; unclosed -1), before the group baseline")
    a("--reward-mode", default="correct", choices=["correct", "format", "random"],
      help="training reward: correct (true 0/1), format (closed </think> + non-empty \\boxed{} after it, correctness "
      "ignored), random (Bernoulli(--random-p)). The true reward is always logged as reward_true. Controls: use "
      "--sampler uniform (the pool sampler would key on the fake reward)")
    a("--random-p", type=float, default=0.5)
    a("--overlong-mixed-only", type=int, default=1, help="with --overlong-soft: shape only groups with >= 1 correct "
      "rollout (1, default); 0 also shapes all-wrong groups (length-only signal: favours shorter wrong answers)")
    a("--adv", default="grpo", choices=["grpo", "maxrl", "psr", "nsr", "wreinforce", "tailrl"],
      help="grpo: Dr.GRPO r - mean; maxrl: (r - mean) / mean (arXiv 2602.02710), 0 if mean == 0; tailrl: "
      "N (w - mean w), TailRL tail-likelihood weights (arXiv 2609.02987), graded-safe")
    a("--adv-norm", default="none", choices=["none", "batch"], help="batch: divide the step's advantages by their "
      "std over the whole batch, all ranks; any --adv, so every arm gets the same scaling")
    a("--pool-G", default="", help="per-pool group size with --sampler pool, e.g. low=32 (others: --group); "
      "pools: " + ",".join(POOLS))
    a("--allow-binary-methods", type=int, default=0, help="run binary-only methods (binary_only) on a pool with "
      "graded-reward tasks anyway (they then treat any partial credit as a success)")
    a("--resample-zero", type=int, default=0, help="K > 0: a group whose samples all score 0 gets K extra samples "
      "(once) before its advantages are computed (DAPO-style dynamic sampling)")
    a("--stream", type=int, default=0, help="persistent generator (rl_stream.StreamGen): slots stay busy across "
      "steps; a step trains on the first n_prompts complete groups; stragglers finish under newer weights with "
      "per-token behaviour log-probs for truncated IS (--tis-cap). 1gpu mode only")
    a("--max-staleness", type=int, default=8, help="--stream: drop a group whose oldest sample started more than "
      "this many optimizer steps ago (8: low values act as a hidden length/difficulty filter, since long rollouts "
      "are the ones that go stale)")
    a("--stream-check-every", type=int, default=16)
    a("--max-rollouts", type=int, default=0, help="--stream: stop once this many rollouts were generated "
      "(rollouts_total, incl. dropped groups); 0 = --steps only")
    # paper-replication flags; all off by default (behaviour unchanged)
    a("--grpo-norm", type=int, default=0, help="original GRPO normalisation (Liu et al. 2503.20783 baseline): group "
      "advantage / (std + 1e-6) and per-response 1/|o_i| loss weighting instead of Dr.GRPO's constant normaliser")
    a("--wr-lambda", type=float, default=0.1, help="--adv wreinforce: weight of the positive samples (Zhu et al.)")
    a("--entropy-coef", type=float, default=0.0, help="entropy bonus coefficient (verl entropy loss, alpha = -coef)")
    a("--clip-cov", type=float, default=0.0, help="Clip-Cov ratio r (Cui et al. 2505.22617; paper 2e-4); 0 = off")
    a("--cov-lb", type=float, default=1.0)
    a("--cov-ub", type=float, default=5.0)
    a("--log-groups", type=int, default=0, help="write groups.jsonl: per step, trained problem indices and their "
      "group-mean reward (training-accuracy history)")
    a("--dump-rollouts", type=int, default=0, help="--stream: write N trained rollouts (half rewarded when "
      "available) of rank 0's batch to rollouts.jsonl every --dump-rollouts-every steps, for qualitative inspection")
    a("--dump-rollouts-every", type=int, default=10)
    a("--repl-evals", default="", help="name:path:every:k,... held-out thinking evals (repl_eval_run)")
    a("--max-hours", type=float, default=0.0, help="--stream: stop once the summed step time (evals excluded, "
      "logged as train_hours) reaches this many hours; 0 = off")
    a("--repl-meta", default="", help="JSON (or @file.json) recorded as config.replication (paper, arXiv id, deviations)")
    a("--run-meta", default="", help="JSON (or @file.json) recorded as config.recipe (datasets + revisions, decontam "
      "counts, sampler, objective)")
    a("--reward-workers", type=int, default=0, help="--stream and in-run evals: grade rollouts in a pool of this many "
      "threads (per rank) off the decode loop; 0 = inline (unchanged). For code tasks, whose sandboxed grading is slow")
    a("--perf", default="", help="opt-in performance flags (docs/perf_opt.md), e.g. pack,head,pfc,lp,hv (exact "
      "precision; head16 changes the head's arithmetic; acc16 = fp16-accumulate GEMMs in generation only)")
    a("--strict-format", type=int, default=0, help="training reward 0 unless the rollout has exactly one </think> and "
      "stops with EOS within --max-new (R1-style well-formed response; truncated responses score 0). --stream only; "
      "reward_true and evals keep the plain grader, wellformed_frac / pass@1_strict are logged")
    a("--save-state", type=int, default=0, help="with every step<N>.pt also write state<N>.pt (optimizers, GradScaler, "
      "sampler state, every rank's rngs, rollout counters) for --resume")
    a("--resume", default="", help="--stream: resume from a state<N>.pt (+ the step<N>.pt beside it), or 'auto' = the "
      "newest one in --out (fresh start if none). In-flight rollouts are not saved; the pipeline restarts empty")
    a("--nonfinite-abort", type=int, default=0, help="raise after this many consecutive optimizer steps skipped for "
      "a non-finite gradient or parameter (logged as nonfinite_skip); 0 = never")
    a("--grader", default="v1", choices=["v1", "v2"], help="math answer checker for the reward and in-run evals: v1 "
      "(eval_math.correct, default) or v2 (grader_v2: canonicalised answers)")
    args = ap.parse_args()
    set_grader(args.grader)
    assert not (args.stream and args.mode != "1gpu"), "--stream needs --mode 1gpu"
    if parse_ppo_schedule(args.ppo_schedule):
        assert args.mode == "1gpu", "--ppo-schedule needs --mode 1gpu (pi_old is recomputed, not a stale copy)"
        assert not getattr(args, "clip_cov", 0.0), "--ppo-schedule with --clip-cov is not supported"
    if "RANK" in os.environ:  # torchrun: data-parallel --stream (each rank: full model + own StreamGen)
        import datetime
        assert args.stream, "data parallel needs --stream 1"
        dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=120))
        DIST.update(on=True, rank=dist.get_rank(), world=dist.get_world_size())
        assert args.n_prompts % DIST["world"] == 0, "--n-prompts must divide by the number of ranks"
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    master = DIST["rank"] == 0
    assert not (args.pool_G and args.sampler != "pool"), "--pool-G needs --sampler pool"
    assert args.dyn_sampling_on == "reward" or args.dyn_sampling, "--dyn-sampling-on adv needs --dyn-sampling 1"
    assert not (args.dyn_sampling and (not args.stream or args.resample_zero)), \
        "--dyn-sampling needs --stream without --resample-zero"
    assert args.overlong_soft < args.max_new, "--overlong-soft must be below --max-new"
    assert not (args.max_rollouts and not args.stream), "--max-rollouts needs --stream"
    assert not (args.strict_format and not args.stream), "--strict-format is implemented for --stream"
    assert not (args.adv in ("psr", "nsr", "wreinforce") and args.grpo_norm), "--grpo-norm applies to --adv grpo"
    os.makedirs(args.out, exist_ok=True)
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(args.tok)
    close_id = tk.token_to_id("</think>")
    pool = [json.loads(l) for l in open(args.pool)]
    filt = json.load(open(args.filters))
    gr_ids = set(filt["eval_guess_resistant_ids"])
    evals = [json.loads(l) for l in open(args.eval_problems)]
    graded = graded_tasks(pool + evals)
    if graded and binary_only(args):
        msg = (f"graded-reward tasks {graded} with binary-only methods {binary_only(args)} (they read r > 0 as success; "
               f"see BINARY_ONLY_ADV)")
        if not args.allow_binary_methods:
            raise SystemExit("refusing to run: " + msg + "; pass --allow-binary-methods 1 to override")
        print("WARNING: " + msg, flush=True)
    big = [json.loads(l) for l in open(args.gr_big)] if args.gr_big else None
    repl_sets = [(n_, [json.loads(l) for l in open(p_)], ev_, k_) for n_, p_, ev_, k_ in parse_repl_evals(args.repl_evals)]
    R_, W_ = DIST["rank"], DIST["world"]
    global RRNG
    RRNG = np.random.default_rng(args.seed + 11 + 1000 * R_)
    if args.reward_mode != "correct":
        print(f"NOTE: --reward-mode {args.reward_mode} (control): the policy is trained on a non-correctness reward; "
              f"reward_true logs real accuracy. Sampler: {args.sampler}"
              + (" -- WARNING: the pool sampler keys on the fake reward; use --sampler uniform" if args.sampler == "pool"
                 else ""), flush=True)
    # uniform: one shared permutation stream, rank r takes positions r, r+W, ...; pool: identical pool state on every
    # rank (updates all-gathered each step), draws from a per-rank rng
    rng = np.random.default_rng(args.seed if args.sampler == "uniform" else args.seed + 1000 * R_)
    torch.manual_seed(args.seed + R_)
    lo, hi = map(float, args.pool_thresholds.split(","))
    sampler = make_sampler(args.sampler, len(pool), rng, lo, hi, tuple(map(float, args.pool_weights.split(","))),
                           shard=(R_, W_))
    poolG = parse_pool_G(args.pool_G, args.group)

    def group_sizes(idx):
        if not args.pool_G:
            return [args.group] * len(idx)
        return [poolG[POOLS[sampler.pool[i]]] for i in idx]

    s = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    raw = MathLM(MathConfig(**s["cfg"]))
    raw.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in s["model"].items()})
    d_tr = torch.device("cuda", 1 if args.mode == "2gpu" else torch.cuda.current_device() if DIST["on"] else 0)
    d_gen = torch.device("cuda", d_tr.index if DIST["on"] else 0)
    raw = raw.to(d_tr)
    gen_m = copy.deepcopy(raw).to(d_gen).half()
    gen_m.fuse_for_inference(torch.float16)
    torch.cuda.set_device(d_tr)
    PERF = set(f for f in args.perf.split(",") if f)
    if PERF & {"pf", "pfc"}:
        gen_m.prefill_packed = True
        gen_m.prefill_compile = "pfc" in PERF
        gen_m.prefill_kv_batched = "pkv" in PERF  # one int8 quantize call for all layers' prefill KV (bit-identical)
    if "acc16" in PERF:  # fp16-accumulate GEMMs in generation only
        from ultraviolet.perf import scope_fp16_acc
        scope_fp16_acc()
    assert not (parse_ppo_schedule(args.ppo_schedule) and PERF & {"pack", "varlen", "head", "head16"}), \
        "--ppo-schedule runs on the reference trainer: drop the trainer --perf flags (pack, varlen, head, head16)"
    if PERF & {"pack", "varlen", "head", "head16"}:
        from ultraviolet.perf import make_trainer
        tr = make_trainer(Trainer, PERF, sys.modules[__name__])(raw, d_tr, args)
    else:
        tr = Trainer(raw, d_tr, args)
    cfg = {"args": vars(args), "pool_size": len(pool), "eval_gr": len(gr_ids), "init": args.ckpt, "graded_tasks": graded}
    cfg["world"] = W_
    rpath = find_resume(args.resume, args.out) if args.resume else None
    R = None
    if rpath:
        assert args.stream, "--resume needs --stream"
        R = torch.load(rpath, map_location="cpu", weights_only=False)
        if int(R["world"]) != W_:  # elastic resume: recorded in config.json / the wandb config
            cfg["resume_world_change"] = f"{R['world']}->{W_}"
            cfg["resume_world_change_step"] = int(R["step"])
    if master:
        json.dump(cfg, open(os.path.join(args.out, "config.json"), "w"), indent=1)
        print(json.dumps(cfg), flush=True)
        wb = TRLog(args, cfg, W_)
        logf = open(os.path.join(args.out, "log.jsonl"), "a")
    else:
        wb = TRLog(None, None, W_)
        logf = open(os.devnull, "w")

    def do_eval(step, full):
        res = run_eval(gen_m, tk, close_id, evals, gr_ids, args, full, big)
        res["eval_full"] = {"v": float(full)}
        wb.eval(res, step)
        rec = {"step": step, "eval": res}
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        key = [k for k in res if k.startswith(("a_normal/", "c_answer_only/", "b_nothink/", "delta"))
               and k.rsplit("/", 1)[1] in ("gr", "gr_nomm", "nomm", "all", "gr_big")]
        print("EVAL", step, json.dumps({k: {k2: round(v, 4) for k2, v in res[k].items()} for k in key}), flush=True)

    def save(step):
        ranks = gather_obj(rank_state(sampler)) if args.save_state else None  # collective: every rank
        if not master:
            return
        if not tr.params_finite():  # never write a checkpoint a --resume would restart from into NaN
            print(f"SAVE SKIPPED at step {step}: non-finite parameters", flush=True)
            return
        atomic_save({"model": raw.state_dict(), "cfg": s["cfg"], "step": step, "init": args.ckpt},
                    os.path.join(args.out, f"step{step}.pt"))
        if args.save_state:
            atomic_save(resume_state(step, tr, ranks), os.path.join(args.out, f"state{step}.pt"))

    def maybe_eval(step, force=False):  # rank 0 evaluates; the other ranks wait
        F_ = args.full_eval_every
        rdue = any(step == 0 or step % ev_ == 0 for _, _, ev_, _ in repl_sets) or (force and repl_sets)
        if master and rdue:
            rres = repl_eval_run(gen_m, tk, close_id, repl_sets, args, step, force)
            wb.repl(rres, step)
            logf.write(json.dumps({"step": step, "repl_eval": rres}) + "\n")
            logf.flush()
            print("REPL", step, json.dumps({n_: {k_: round(v_, 4) for k_, v_ in d_.items() if k_ in ("pass@1", "maj@4",
                  "maj@16", "mean_len_correct", "mean_len_incorrect", "close_rate")} for n_, d_ in rres.items()}),
                  flush=True)
        if DIST["on"] and rdue:
            dist.barrier()
        due = (F_ and step % F_ == 0) or step % args.eval_every == 0 or force
        if master and F_ and step % F_ == 0:
            do_eval(step, True)
        elif master and due:
            do_eval(step, bool(args.eval_full))
        if DIST["on"] and due:
            dist.barrier()

    if rpath:
        w = torch.load(os.path.join(os.path.dirname(rpath), f"step{R['step']}.pt"), map_location="cpu",
                       weights_only=False)
        raw.load_state_dict(w["model"])
        sync(gen_m, raw)
        apply_resume(R, raw, tr, sampler, R_, W_, args.seed)
        if master and RS["world_change"]:
            gt, exact = global_totals(R)
            print(f"WARNING resume_world_change: {R['world']}->{W_} ranks at step {RS['start']}: shared sampler state "
                  f"from saved rank 0, per-rank rngs re-derived, all-rank counters (rollouts {gt.get('rollouts', 0)}, "
                  f"gen_tokens {gt.get('gen_tokens', 0)}) split over the new ranks "
                  f"{'exactly' if exact else '(approx: saved master counters x old world)'}", flush=True)
        if master:
            print(f"RESUMED from {rpath} at step {RS['start']} (rollouts_total {RS['totals'].get('rollouts', 0)}"
                  f"{' on rank 0' if RS['world_change'] else ''}, "
                  f"train_hours {RS['train_s'] / 3600:.3f})", flush=True)
    if RS["start"] == 0:
        maybe_eval(0)
    t_start = time.time()
    nxt = {}

    def do_roll(k, idx):  # idx drawn on the main thread (the sampler is not thread-safe)
        nxt[k] = (idx,) + rollout(gen_m, tk, close_id, [pool[i] for i in idx], group_sizes(idx), args)

    def take(k):
        idx, seqs, adv, g = nxt.pop(k)
        sampler.update(idx, g.pop("_group_means"))
        g.update(sampler.stats())
        return seqs, adv, g

    if args.stream:
        stream_loop(args, gen_m, raw, tr, tk, close_id, pool, sampler, group_sizes, wb, logf, save, maybe_eval)
        if master:
            print("DONE", flush=True)
        wb.finish()
        if DIST["on"]:
            dist.barrier()
            dist.destroy_process_group()
        return
    if args.mode == "2gpu":
        do_roll(0, sampler.sample(args.n_prompts))
    for k in range(args.steps):
        t0 = time.time()
        if args.mode == "2gpu":
            seqs, adv, g = take(k)  # rollout k+1 below samples after this update (1-step-stale pools)
            th = threading.Thread(target=do_roll, args=(k + 1, sampler.sample(args.n_prompts))) \
                if k + 1 < args.steps else None
            if th:
                th.start()
            ts = tr.step(k, seqs, adv)
            torch.cuda.synchronize(d_tr)
            t_train = time.time() - t0
            if th:
                th.join()
        else:
            do_roll(k, sampler.sample(args.n_prompts))
            seqs, adv, g = take(k)
            t1 = time.time()
            ts = tr.step(k, seqs, adv)
            torch.cuda.synchronize(d_tr)
            t_train = time.time() - t1
        t1 = time.time()
        sync(gen_m, raw)
        rec = {"step": k + 1, **g, **ts, "t_train": t_train, "t_sync": time.time() - t1, "t_step": time.time() - t0,
               "hours": (time.time() - t_start) / 3600}
        wb.train(rec, k + 1)
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        if (k + 1) % 10 == 0 or k < 3:
            print(json.dumps({x: round(v, 4) if isinstance(v, float) else v for x, v in rec.items()}), flush=True)
        if (k + 1) % args.ckpt_every == 0 or k + 1 == args.steps:
            save(k + 1)
        maybe_eval(k + 1, force=k + 1 == args.steps)
    print("DONE", flush=True)
    wb.finish()


if __name__ == "__main__":
    main()
