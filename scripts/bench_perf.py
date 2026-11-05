"""Performance profiling / benchmarking for the math LM RL loop and pretraining (docs/perf_opt.md).

Modes
  rl      run rl_grpo.main() unchanged (stream mode) with logging / evals / checkpoint saving stubbed out, plus
          timers: per-step t_gen / t_train, CPU reward time, refill (prefill) time; optionally captures the training
          batch of one step (--capture PATH) and profiles one train step / one gen chunk (--profile-dir).
          Perf flags (--perf a,b,...) switch the optimizations on (see docs/perf_opt.md).
  train   replay a captured batch through Trainer.step variants: phase timings, profiler table, and equivalence
          (loss, grads) between the baseline and --perf flags.
  decode  steady-state StreamGen decode profile (per-kernel table).
  pt      pretrain step (model(x, y) fwd+bwd, train_pt-style) timing / profile.

Everything writes under --out.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

M = os.path.expanduser("~/vt/math-pt")
DEF_CKPT = f"{M}/sft/mid/mid-alpha/epoch2.pt"
DEF_TOK = f"{M}/ctok16384/tokenizer.json"
DEF_POOL = f"{M}/repl/data/math_train.jsonl"


def sync_t():
    torch.cuda.synchronize()
    return time.time()


def prof_table(prof, path, n=60):
    tab = prof.key_averages().table(sort_by="cuda_time_total", row_limit=n)
    with open(path, "w") as f:
        f.write(tab)
    return tab


# ------------------------------------------------------------------------------------------------ perf flags
def apply_perf(flags, raw=None, gen_m=None):
    """Model-level flags (instance attributes; defaults unchanged when a flag is absent)."""
    for m in (raw, gen_m):
        if m is None:
            continue
        if "varlen" in flags or "pack" in flags:
            m.train_attn = "varlen"
    if gen_m is not None and ("pf" in flags or "pfc" in flags):
        gen_m.prefill_packed = True
        gen_m.prefill_compile = "pfc" in flags
        gen_m.prefill_kv_batched = "pkv" in flags
    if gen_m is not None and "acc16" in flags and not getattr(apply_perf, "_acc16", False):  # as rl_grpo.main
        from ultraviolet.perf import scope_fp16_acc
        scope_fp16_acc()
        apply_perf._acc16 = True


# ------------------------------------------------------------------------------------------------ rl mode
def run_rl(a):
    import rl_grpo
    import rl_stream
    flags = set(x for x in a.perf.split(",") if x)
    T = {"reward": 0.0, "refill": 0.0, "n_reward": 0}

    class NoLog:
        def __init__(self, *x, **k): pass
        def train(self, rec, step): pass
        def eval(self, *x): pass
        def repl(self, *x): pass
        def finish(self): pass

    rl_grpo.TRLog = NoLog
    rl_grpo.run_eval = lambda *x, **k: {}
    rl_grpo.atomic_save = lambda *x, **k: None
    _reward = rl_grpo.reward

    def reward(*x, **k):
        t = time.time()
        r = _reward(*x, **k)
        T["reward"] += time.time() - t
        T["n_reward"] += 1
        return r
    rl_grpo.reward = reward

    base_gen = rl_stream.StreamGen
    from ultraviolet import perf as P
    Gen = P.make_streamgen(base_gen, flags)

    class TGen(Gen):
        def __init__(self, model, *x, **k):
            apply_perf(flags, gen_m=model)
            super().__init__(model, *x, **k)

        def _refill(self, rows):
            t = sync_t()
            super()._refill(rows)
            T["refill"] += sync_t() - t
    rl_stream.StreamGen = TGen

    base_tr = rl_grpo.Trainer
    TrCls = P.make_trainer(base_tr, flags, rl_grpo)
    prof_dir = a.profile_dir

    class TTr(TrCls):
        def __init__(self, raw, dev, args):
            apply_perf(flags, raw)
            super().__init__(raw, dev, args)
            self._k = 0

        def step(self, k, seqs, advs, blps=None):
            if a.capture and k == a.capture_step:
                with open(a.capture, "wb") as f:
                    pickle.dump({"k": k, "seqs": seqs, "advs": advs, "blps": blps,
                                 "state": {n: p.detach().cpu() for n, p in self.raw.state_dict().items()}}, f)
            if prof_dir and k == a.profile_step:
                from torch.profiler import ProfilerActivity, profile
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as pr:
                    r = super().step(k, seqs, advs, blps)
                    torch.cuda.synchronize()
                prof_table(pr, os.path.join(prof_dir, f"train_step{k}.txt"))
                pr.export_chrome_trace(os.path.join(prof_dir, f"train_step{k}.json"))
                return r
            return super().step(k, seqs, advs, blps)
    rl_grpo.Trainer = TTr

    # per-step record: add our timers (stream_loop writes rec to logf; we wrap logf via json in out/log.jsonl)
    os.makedirs(a.out, exist_ok=True)
    if prof_dir:
        os.makedirs(prof_dir, exist_ok=True)
    _dumps = json.dumps
    last = {"reward": 0.0, "refill": 0.0}

    def dumps(obj, *x, **k):
        if isinstance(obj, dict) and "t_step" in obj:
            obj = dict(obj)
            obj["t_reward_cpu"] = T["reward"] - last["reward"]
            obj["t_refill"] = T["refill"] - last["refill"]
            last.update(reward=T["reward"], refill=T["refill"])
            obj["perf"] = a.perf
        return _dumps(obj, *x, **k)
    rl_grpo.json.dumps = dumps
    argv = ["rl_grpo.py", "--ckpt", a.ckpt, "--tok", a.tok, "--pool", a.pool,
            "--filters", f"{M}/rl/pool_v0/filters.json", "--eval-problems", f"{M}/vtr_v0/eval_problems.jsonl",
            "--out", a.out, "--name", "perf", "--mode", "1gpu", "--stream", "1", "--max-staleness", "8",
            "--sampler", "uniform", "--answer-direct", "1", "--optim", "normuon", "--muon-lr", "9e-4",
            "--temperature", "1.0", "--max-new", str(a.max_new), "--eval-full", "0", "--eval-every", "100000000",
            "--steps", str(a.steps), "--n-prompts", str(a.n_prompts), "--group", str(a.group), "--slots",
            str(a.slots), "--ckpt-every", "100000000", "--seed", str(a.seed)] + a.extra.split()
    sys.argv = argv
    torch.manual_seed(a.seed)
    rl_grpo.main()


# ------------------------------------------------------------------------------------------------ train mode
def load_trainer(a, flags, state=None):
    import rl_grpo
    from ultraviolet import perf as P
    from ultraviolet.model import MathConfig, MathLM
    s = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    raw = MathLM(MathConfig(**s["cfg"]))
    raw.load_state_dict(state if state is not None else
                        {k.replace("_orig_mod.", ""): v for k, v in s["model"].items()})
    raw = raw.cuda()
    apply_perf(flags, raw)
    ns = argparse.Namespace(mode="1gpu", optim="normuon", muon_lr=9e-4, adam_lr=1e-5, warmup=10, max_new=a.max_new,
                            tok_budget=4096 if "fp32" in flags else a.tok_budget, pad_to=512, tis_cap=2.0, temperature=1.0, grad_clip=1.0,
                            seed=0, entropy_coef=0.0, clip_cov=0.0, grpo_norm=0, nonfinite_abort=0)
    Tr = P.make_trainer(rl_grpo.Trainer, flags, rl_grpo)
    return Tr(raw, torch.device("cuda", 0), ns), raw


def run_train(a):
    """Replay a captured RL batch through each variant (';'-separated flag sets; 'fp32' = full-fp32 reference):
    fwd+bwd and full-step time, peak memory, grads vs the first variant and vs fp32 (if present), per-token
    response log-probs vs fp32 / first variant / the sampler's behaviour log-probs."""
    from torch.profiler import ProfilerActivity, profile
    from ultraviolet import perf as P
    with open(a.capture, "rb") as f:
        cap = pickle.load(f)
    seqs, advs, blps = cap["seqs"], cap["advs"], cap["blps"]
    sig = [i for i, x in enumerate(advs) if x != 0]
    res = {"n_seq": len(seqs), "n_signal": len(sig),
           "tokens_signal": int(sum(len(seqs[i][0]) - 1 for i in sig)),
           "resp_tokens_signal": int(sum(len(seqs[i][0]) - seqs[i][1] for i in sig))}
    lp_seqs = [seqs[i] for i in sig[:a.lp_seqs]]
    out = {}
    for variant in a.variants.split(";"):
        name = variant or "base"
        flags = set(x for x in variant.split(",") if x and x != "base")
        torch.manual_seed(0)
        tr, raw = load_trainer(a, flags, cap["state"])
        st0 = {n: p.detach().clone() for n, p in raw.state_dict().items()}
        r = {}
        g, stats, _ = P.grads_only(tr, cap["k"], seqs, advs, blps)  # also compiles
        g, stats, _ = P.grads_only(tr, cap["k"], seqs, advs, blps)
        torch.cuda.reset_peak_memory_stats()
        tfb = []
        for _ in range(a.reps):
            _, _, dt = P.grads_only(tr, cap["k"], seqs, advs, blps)
            tfb.append(dt)
        r["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
        r["t_fwd_bwd"] = float(np.median(tfb))
        ts = []
        for _ in range(a.reps):
            raw.load_state_dict(st0)
            t = sync_t()
            tr.step(cap["k"], seqs, advs, blps)
            ts.append(sync_t() - t)
        raw.load_state_dict(st0)
        r["t_step"] = float(np.median(ts))
        r.update(loss=stats["loss"], tis=stats["tis"], n_tok=stats["n_tok"], n_mb=stats["n_mb"])
        lps = tr.logps(lp_seqs)
        if a.profile_dir and "fp32" not in flags:
            os.makedirs(a.profile_dir, exist_ok=True)
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as pr:
                tr.step(cap["k"], seqs, advs, blps)
                torch.cuda.synchronize()
            raw.load_state_dict(st0)
            prof_table(pr, os.path.join(a.profile_dir, f"train_{name}.txt"))
        out[name] = {"g": g, "loss": stats["loss"], "lps": lps}
        res[name] = r
        print(name, json.dumps(r), flush=True)
        del tr, raw
        torch.cuda.empty_cache()

    def gcmp(gs, ref):
        num = sum(float(((x - y) ** 2).sum()) for x, y in zip(gs, ref))
        den = sum(float((y ** 2).sum()) for y in ref)
        dot = sum(float((x * y).sum()) for x, y in zip(gs, ref))
        nx = sum(float((x ** 2).sum()) for x in gs) ** 0.5
        per = [float(torch.nn.functional.cosine_similarity(x.flatten(), y.flatten(), 0)) for x, y in zip(gs, ref)
               if y.norm() > 0]
        return {"grad_rel_err": (num / den) ** 0.5, "grad_cos": dot / (nx * den ** 0.5), "grad_cos_min_param": min(per)}

    def lcmp(l1, l2):
        d = np.concatenate([np.abs(x - y) for x, y in zip(l1, l2)])
        return {"lp_max_abs": float(d.max()), "lp_mean_abs": float(d.mean())}
    names = list(out)
    beh = [np.asarray(blps[i], dtype=np.float64) for i in sig[:a.lp_seqs]]
    refs = [names[0]] + (["fp32"] if "fp32" in out and names[0] != "fp32" else [])
    for n in names:
        for rn in refs:
            if n == rn:
                continue
            c = gcmp(out[n]["g"], out[rn]["g"])
            c.update(lcmp(out[n]["lps"], out[rn]["lps"]))
            c["loss_rel_err"] = abs(out[n]["loss"] - out[rn]["loss"]) / abs(out[rn]["loss"])
            res[n][f"vs_{rn}"] = c
        res[n]["vs_behaviour"] = lcmp(out[n]["lps"], beh)
        print(n, {k: v for k, v in res[n].items() if k.startswith("vs_")}, flush=True)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "train_bench.json"), "w") as f:
        json.dump(res, f, indent=1)


# ------------------------------------------------------------------------------------------------ decode mode
def load_models(a, flags=()):
    import copy
    from ultraviolet.model import MathConfig, MathLM
    s = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    raw = MathLM(MathConfig(**s["cfg"]))
    raw.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in s["model"].items()})
    raw = raw.cuda()
    gen_m = copy.deepcopy(raw).half()
    gen_m.fuse_for_inference(torch.float16)
    return raw, gen_m


def run_decode(a):
    """Steady-state StreamGen: fill all slots from the pool (G samples per prompt), warm up, then time `--steps`
    chunks and profile 4 more. Reports ms per decode step (graph replay + per-step eager work) and per chunk."""
    import rl_grpo
    import rl_stream
    from tokenizers import Tokenizer
    from torch.profiler import ProfilerActivity, profile
    from ultraviolet import perf as P
    flags = set(x for x in a.perf.split(",") if x)
    tk = Tokenizer.from_file(a.tok)
    pool = [json.loads(l) for l in open(a.pool)]
    ns = argparse.Namespace(answer_only=0, answer_direct=1)
    enc = [rl_grpo.encode_prompt(tk, p, ns) for p in pool[:2000]]
    torch.manual_seed(a.seed)
    raw, gen_m = load_models(a)
    del raw
    apply_perf(flags, gen_m=gen_m)
    Gen = P.make_streamgen(rl_stream.StreamGen, flags)
    gen = Gen(gen_m, a.slots, max(len(e) for e in enc), a.max_new, 1.0, rl_grpo.EOS)
    rng = np.random.default_rng(a.seed)

    def topup():
        while len(gen.queue) < 2 * a.slots:
            i = int(rng.integers(len(enc)))
            for _ in range(a.group):
                gen.add(enc[i], i)
    res = {}
    for _ in range(a.warm_chunks):
        topup()
        gen.chunk()
    t = sync_t()
    s0, n0 = gen.n_decode_steps, gen.n_tokens
    for _ in range(a.steps):
        topup()
        gen.chunk()
    dt = sync_t() - t
    res["ms_per_decode_step_incl_harvest"] = dt / (gen.n_decode_steps - s0) * 1e3
    res["tok_s_returned"] = (gen.n_tokens - n0) / dt
    # pure graph replay (no harvest / logprob): one runner.step at the current positions, 50x
    st = gen.st
    pos0, hp0 = st.pos.clone(), st.host_pos.copy()
    t = sync_t()
    for _ in range(50):
        gen.runner.step(gen.tok)
        st.pos.copy_(pos0)
        st.host_pos[:] = hp0
    res["ms_replay_only"] = (sync_t() - t) / 50 * 1e3
    res["mean_pos"] = float(st.host_pos.mean())
    if a.profile_dir:
        os.makedirs(a.profile_dir, exist_ok=True)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as pr:
            for _ in range(4):
                topup()
                gen.chunk()
            torch.cuda.synchronize()
        prof_table(pr, os.path.join(a.profile_dir, f"decode_{a.perf or 'base'}.txt"), 80)
        pr.export_chrome_trace(os.path.join(a.profile_dir, f"decode_{a.perf or 'base'}.json"))
    print(json.dumps(res), flush=True)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"decode_{a.perf or 'base'}.json"), "w") as f:
        json.dump(res, f, indent=1)


# ------------------------------------------------------------------------------------------------ pretrain mode
def run_pt(a):
    """train_pt-style step on random tokens: micro_bs x T, fp16 autocast + GradScaler, compiled trunk, chunked CE,
    NorMuon + AdamW. Reports tok/s for fwd+bwd and for the full step, peak memory; --variants compares losses/grads."""
    from torch.profiler import ProfilerActivity, profile
    from ultraviolet.model import MathConfig, MathLM
    from ultraviolet.optim import build_optimizers
    T, B = a.seq_len, a.micro_bs
    out = {}
    gref = None
    for variant in (a.variants or "base").split(";"):
        flags = set(x for x in variant.split(",") if x and x != "base")
        torch.manual_seed(0)
        s = torch.load(a.ckpt, map_location="cpu", weights_only=False)
        m = MathLM(MathConfig(**s["cfg"]))
        m.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in s["model"].items()})
        m = m.cuda().train()
        if "varlen" in flags:
            m.train_attn = "varlen"
        m.trunk = torch.compile(m.trunk)
        mu, adam = build_optimizers(m)
        scaler = torch.amp.GradScaler("cuda", init_scale=2.0 ** 14)
        g = torch.Generator().manual_seed(1)
        xb = torch.randint(0, m.cfg.vocab_size, (B, T + 1), generator=g).cuda()
        if a.pt_data:  # real tokens: a contiguous slice of a uint16 token file
            arr = np.memmap(a.pt_data, dtype=np.uint16, mode="r")
            xb = torch.from_numpy(np.asarray(arr[:B * (T + 1)], dtype=np.int64).reshape(B, T + 1)).cuda()

        if "doc" in flags:  # SEMANTIC CHANGE (not a pure win): attention masked at document (EOS) boundaries
            from ultraviolet.model import chunked_softcap_ce
            m.train_attn = "varlen"
            xin = xb[:, :-1].reshape(-1)
            eos = (xin == 0).nonzero().squeeze(1).cpu().numpy()  # EOS ends a doc; next token starts a new one
            bnd = sorted(set([0, B * T] + [int(e) + 1 for e in eos if (int(e) + 1) % T != 0] +
                             [b * T for b in range(B)]))
            cu = torch.tensor(bnd, dtype=torch.int32, device=xb.device)
            pos = torch.cat([torch.arange(b - a_) for a_, b in zip(bnd[:-1], bnd[1:])]).to(xb.device)
            ms = int(np.diff(bnd).max())

            def fb():
                with torch.autocast("cuda", dtype=torch.float16):
                    h = m.trunk(xin[None], pos=pos, cu_seqlens=cu, max_seqlen=ms)[0]
                    loss = chunked_softcap_ce(h.reshape(-1, h.size(-1)), m.embed.weight, xb[:, 1:].reshape(-1),
                                              m.cfg.softcap, 2048)
                scaler.scale(loss).backward()
                return loss
        else:
            def fb():
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = m(xb[:, :-1], xb[:, 1:])
                scaler.scale(loss).backward()
                return loss

        def opt():
            scaler.unscale_(mu)
            scaler.unscale_(adam)
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            scaler.step(mu)
            scaler.step(adam)
            scaler.update()
            mu.zero_grad(set_to_none=True)
            adam.zero_grad(set_to_none=True)
        for _ in range(3):
            fb()
            mu.zero_grad(set_to_none=True)
            adam.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        loss = fb()
        grads = [p.grad.float().cpu() / scaler.get_scale() for p in m.parameters() if p.grad is not None]
        mu.zero_grad(set_to_none=True)
        adam.zero_grad(set_to_none=True)
        t = sync_t()
        for _ in range(a.reps):
            fb()
            mu.zero_grad(set_to_none=True)
            adam.zero_grad(set_to_none=True)
        tfb = (sync_t() - t) / a.reps
        st0 = {k: v.clone() for k, v in m.state_dict().items()}
        fb()
        t = sync_t()
        opt()
        topt = sync_t() - t
        m.load_state_dict(st0)
        r = {"loss": float(loss), "t_fwd_bwd": tfb, "tok_s_fwd_bwd": B * T / tfb, "t_opt": topt,
             "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30}
        if gref is None:
            gref = grads
        else:
            num = sum(float(((x - y) ** 2).sum()) for x, y in zip(grads, gref))
            den = sum(float((y ** 2).sum()) for y in gref)
            dot = sum(float((x * y).sum()) for x, y in zip(grads, gref))
            nx = sum(float((x ** 2).sum()) for x in grads) ** 0.5
            r.update(grad_rel_err=(num / den) ** 0.5, grad_cos=dot / (nx * den ** 0.5),
                     loss_diff=float(loss) - out["base"]["loss"])
        if a.profile_dir:
            os.makedirs(a.profile_dir, exist_ok=True)
            with profile(activities=[ProfilerActivity.CUDA]) as pr:
                fb()
                opt()
                torch.cuda.synchronize()
            prof_table(pr, os.path.join(a.profile_dir, f"pt_{variant}.txt"))
            m.load_state_dict(st0)
        out[variant] = r
        print(variant, json.dumps(r), flush=True)
        del m, mu, adam
        torch.cuda.empty_cache()
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "pt_bench.json"), "w") as f:
        json.dump(out, f, indent=1)


# ------------------------------------------------------------------------------------------------ prefill mode
def run_prefill(a):
    """Refill-style prefill: n rows of real prompts into a KernelDecodeState of --slots rows (as StreamGen._refill),
    wall time per call (synced) vs GPU kernel time (profiler)."""
    import rl_grpo
    from tokenizers import Tokenizer
    from torch.profiler import ProfilerActivity, profile
    from ultraviolet.model import _autocast, _left_pad, sample
    tk = Tokenizer.from_file(a.tok)
    pool = [json.loads(l) for l in open(a.pool)]
    ns = argparse.Namespace(answer_only=0, answer_direct=1)
    enc = [rl_grpo.encode_prompt(tk, p, ns) for p in pool[:2000]]
    raw, gen_m = load_models(a)
    del raw
    apply_perf(set(a.perf.split(",")), gen_m=gen_m)
    st = gen_m.new_state(a.slots, max(len(e) for e in enc) + a.max_new + 16, torch.float16, torch.device("cuda"))
    rng = np.random.default_rng(0)
    res = {"mean_prompt_len": float(np.mean([len(e) for e in enc]))}
    for n in [int(x) for x in a.prefill_rows.split(",")]:
        calls = []
        for _ in range(12):
            items = [enc[int(i)] for i in rng.integers(len(enc), size=n)]
            rows = torch.tensor(sorted(rng.choice(a.slots, n, replace=False).tolist()), device="cuda")
            calls.append((items, rows))

        def one(items, rows):
            idx, starts = _left_pad(items, 0, torch.device("cuda"))
            with _autocast(torch.device("cuda"), torch.float16):
                lg = gen_m.prefill(idx, starts, st, rows=rows)
            return sample(lg, 1.0)
        for c in calls[:2]:
            one(*c)
        t = sync_t()
        for c in calls[2:]:
            one(*c)
        wall = (sync_t() - t) / 10
        with profile(activities=[ProfilerActivity.CUDA]) as pr:
            for c in calls[2:]:
                one(*c)
            torch.cuda.synchronize()
        gpu = sum(e.self_device_time_total for e in pr.key_averages()) / 10 / 1e3
        res[f"n{n}"] = {"wall_ms": wall * 1e3, "gpu_ms": gpu}
        if a.profile_dir:
            os.makedirs(a.profile_dir, exist_ok=True)
            prof_table(pr, os.path.join(a.profile_dir, f"prefill_n{n}.txt"), 50)
        print(n, res[f"n{n}"], flush=True)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"prefill_{a.perf or 'base'}.json"), "w") as f:
        json.dump(res, f, indent=1)


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["rl", "train", "decode", "pt", "prefill"])
    ap.add_argument("--ckpt", default=DEF_CKPT)
    ap.add_argument("--tok", default=DEF_TOK)
    ap.add_argument("--pool", default=DEF_POOL)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--n-prompts", type=int, default=64)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--slots", type=int, default=512)
    ap.add_argument("--max-new", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--extra", default="")
    ap.add_argument("--perf", default="")
    ap.add_argument("--capture", default="")
    ap.add_argument("--capture-step", type=int, default=10)
    ap.add_argument("--profile-dir", default="")
    ap.add_argument("--profile-step", type=int, default=12)
    ap.add_argument("--variants", default="")
    ap.add_argument("--tok-budget", type=int, default=16384)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warm-chunks", type=int, default=40)
    ap.add_argument("--lp-seqs", type=int, default=64)
    ap.add_argument("--prefill-rows", default="4,16,64")
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--micro-bs", type=int, default=4)
    ap.add_argument("--pt-data", default="")
    a = ap.parse_args()
    {"rl": run_rl, "train": run_train, "decode": run_decode, "pt": run_pt, "prefill": run_prefill}[a.mode](a)


if __name__ == "__main__":
    main()
