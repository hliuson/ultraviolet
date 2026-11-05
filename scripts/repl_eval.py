"""Sampled pass@k / maj@k eval for the RL paper replications. No greedy decoding.

  python scripts/repl_eval.py --ckpt C.pt --tok TOK --sets math500=A.jsonl,gsm8k=B.jsonl --k 64 --seed 0 \
      --out-dir D --name NAME [--answer-direct 1] [--temperature 1.0] [--max-new 2048]

Per set it writes D/NAME.<set>.s<seed>.npz with (P, K) arrays: ok (bool), length (int16, generated tokens), closed
(bool), pred (int64 hash of the normalised extracted answer; 0 = none). Grading is exactly the RL reward
(rl_grpo.reward): correct iff </think> is closed and the extracted answer (extract_direct with --answer-direct, else
extract) equals the reference under eval_math.correct. Shards with different --seed are merged by --merge (concatenate
along K) into D/NAME.<set>.npz and summarised (summary(): unbiased pass@k for k = 1, 2, 4, ..., K; maj@k averaged
over disjoint blocks of k samples; mean length of correct / incorrect samples; close rate) in D/NAME.<set>.json.
"""
import argparse
import collections
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))


def pass_at_k(n, c, k):
    """Unbiased pass@k (Chen et al. 2021): 1 - C(n-c, k) / C(n, k), per problem; n, c arrays."""
    n, c = np.asarray(n), np.asarray(c)
    out = np.ones(len(c), dtype=np.float64)
    for i, (ni, ci) in enumerate(zip(n, c)):
        if ni - ci >= k:
            out[i] = 1.0 - np.prod(1.0 - k / np.arange(ni - ci + 1, ni + 1))
    return out


def maj_at_k(ok, pred, k):
    """Majority vote over disjoint blocks of k samples (pred 0 = no answer, never voted; ties -> first seen); per
    problem, mean over the K // k blocks."""
    P, K = ok.shape
    nb = K // k
    res = np.zeros(P)
    for j in range(P):
        s = 0.0
        for b in range(nb):
            pr, o = pred[j, b * k:(b + 1) * k], ok[j, b * k:(b + 1) * k]
            cnt = collections.Counter(int(x) for x in pr if x != 0)
            if cnt:
                top = cnt.most_common(1)[0][0]
                s += float(any(o_ for p_, o_ in zip(pr, o) if p_ == top))
        res[j] = s / max(nb, 1)
    return res


def wellformed(raw, close_id, eos):
    """Well-formed thinking response (--strict-format): exactly one </think> and stopped by EOS within max-new."""
    return raw.count(close_id) == 1 and len(raw) > 0 and raw[-1] == eos


def summary(ok, length, closed, pred, wf=None):
    P, K = ok.shape
    c = ok.sum(1)
    out = {"P": int(P), "K": int(K), "pass@1": float(ok.mean())}
    if wf is not None:  # strict grading = correct and well-formed
        out["wellformed"] = float(wf.mean())
        out["pass@1_strict"] = float((ok & wf).mean())
        cs = (ok & wf).sum(1)
        for k in (16, 64):
            if k <= K:
                out[f"pass@{k}_strict"] = float(pass_at_k(np.full(P, K), cs, k).mean())
    k = 2
    while k <= K:
        out[f"pass@{k}"] = float(pass_at_k(np.full(P, K), c, k).mean())
        k *= 2
    for k in (4, 16, 64):
        if k <= K:
            out[f"maj@{k}"] = float(maj_at_k(ok, pred, k).mean())
    out["mean_len"] = float(length.mean())
    out["mean_len_correct"] = float(length[ok].mean()) if ok.any() else float("nan")
    out["mean_len_incorrect"] = float(length[~ok].mean()) if (~ok).any() else float("nan")
    out["close_rate"] = float(closed.mean())
    out["frac_unsolved"] = float((c == 0).mean())
    return out


def hash_pred(p):
    if p is None:
        return 0
    from eval_math import norm
    h = int.from_bytes(hashlib.blake2b(norm(str(p)).encode(), digest_size=8).digest(), "little", signed=True)
    return h or 1


def run(a):
    import torch
    from tokenizers import Tokenizer
    from eval_math import EOS, extract, load
    from rl_grpo import NOTHINK, extract_direct, gen, reward
    tk = Tokenizer.from_file(a.tok)
    close_id = tk.token_to_id("</think>")
    torch.manual_seed(a.seed)
    m, _ = load(a.ckpt)
    m = m.half()
    m.fuse_for_inference(torch.float16)
    ext = extract_direct if a.answer_direct else extract
    os.makedirs(a.out_dir, exist_ok=True)
    for spec in a.sets.split(","):
        name, path = spec.split("=")
        ps = [json.loads(l) for l in open(path)]
        if a.limit:
            ps = ps[:a.limit]
        P = len(ps)
        # --cond nothink / ao: prompt + "</think>\n\n" (answer-direct no-think and answer-only probes, rl_grpo.run_eval
        # b_nothink / c_answer_only), 512 / 16 new tokens, graded on NOTHINK + continuation like the answer-only reward
        suf = tk.encode(NOTHINK).ids if a.cond != "think" else []
        max_new = {"think": a.max_new, "nothink": 512, "ao": 16}[a.cond]
        pre = [[EOS] + tk.encode(p["prompt"]).ids + suf for p in ps]
        ok = np.zeros((P, a.k), bool)
        closed = np.zeros((P, a.k), bool)
        length = np.zeros((P, a.k), np.int16)
        pred = np.zeros((P, a.k), np.int64)
        wf = np.zeros((P, a.k), bool)
        for k0 in range(0, a.k, a.chunk_k):
            kk = min(a.chunk_k, a.k - k0)
            outs, raw = gen(m, pre * kk, max_new, a.temperature, a.slots, a.top_p)
            for i, (o, ro) in enumerate(zip(outs, raw)):
                s_, j = divmod(i, P)
                r, c = reward(tk, close_id, o, ps[j], a.cond != "think", bool(a.answer_direct))
                ok[j, k0 + s_] = r > 0
                closed[j, k0 + s_] = c
                wf[j, k0 + s_] = wellformed(ro, close_id, EOS) if a.cond == "think" else (len(ro) > 0 and ro[-1] == EOS)
                length[j, k0 + s_] = min(len(ro), 32767)
                if c:
                    try:
                        pv = ext(("" if a.cond == "think" else NOTHINK) + tk.decode(o, skip_special_tokens=False))
                    except Exception:  # noqa: BLE001
                        pv = None
                    pred[j, k0 + s_] = hash_pred(pv)
            print(name, f"k {k0 + kk}/{a.k}", f"pass@1 {ok[:, :k0 + kk].mean():.4f}", flush=True)
        np.savez_compressed(os.path.join(a.out_dir, f"{a.name}.{name}.s{a.seed}.npz"), ok=ok, length=length,
                            closed=closed, pred=pred, wellformed=wf)
        print(name, json.dumps(summary(ok, length, closed, pred, wf)), flush=True)


def merge(a):
    import glob
    res = {}
    for spec in a.sets.split(","):
        name = spec.split("=")[0]
        fs = sorted(glob.glob(os.path.join(a.out_dir, f"{a.name}.{name}.s*.npz")))
        assert fs, (a.name, name)
        d = [np.load(f) for f in fs]
        keys = ("ok", "length", "closed", "pred") + (("wellformed",) if all("wellformed" in x for x in d) else ())
        arr = {k: np.concatenate([x[k] for x in d], axis=1) for k in keys}
        np.savez_compressed(os.path.join(a.out_dir, f"{a.name}.{name}.npz"), **arr)
        s = summary(arr["ok"], arr["length"], arr["closed"], arr["pred"], arr.get("wellformed"))
        s.update(ckpt=a.ckpt, shards=[os.path.basename(f) for f in fs], temperature=a.temperature, top_p=a.top_p,
                 max_new=a.max_new)
        json.dump(s, open(os.path.join(a.out_dir, f"{a.name}.{name}.json"), "w"), indent=1)
        print(name, json.dumps(s), flush=True)
        res[name] = s
    if a.tr_run:  # tr_log: sum/<set>_pass1, _maj16, _pass<K> + diag/repl/<set>/* at --tr-step and in the summary
        import tr_log
        from rl_grpo import repl_keys
        tr_log.init(a.tr_run, a.tr_stage, ckpt_path=a.ckpt or None)
        kv = repl_keys({n: {k: v for k, v in d.items() if isinstance(v, (int, float))} for n, d in res.items()})
        if a.tr_step >= 0:
            tr_log.log(kv, step=a.tr_step)
        tr_log.summary(kv)
        tr_log.finish()


def main():
    ap = argparse.ArgumentParser()
    a_ = ap.add_argument
    a_("--ckpt", default="")
    a_("--tok", default="")
    a_("--sets", required=True, help="name=path.jsonl,... (rows need prompt and answer/ref)")
    a_("--k", type=int, default=64)
    a_("--chunk-k", type=int, default=16, help="samples per problem per generate call")
    a_("--seed", type=int, default=0)
    a_("--temperature", type=float, default=1.0)
    a_("--top-p", type=float, default=1.0, help="nucleus sampling (Yue et al.: T=0.6, top-p 0.95)")
    a_("--max-new", type=int, default=2048)
    a_("--slots", type=int, default=1024)
    a_("--answer-direct", type=int, default=1)
    a_("--cond", default="think", choices=["think", "nothink", "ao"], help="think (default); nothink / ao: prompt + "
       "'</think>\\n\\n' with 512 / 16 new tokens (needs --answer-direct 1)")
    a_("--limit", type=int, default=0)
    a_("--out-dir", required=True)
    a_("--name", required=True)
    a_("--merge", type=int, default=0, help="merge the seed shards of --name and summarise (CPU only)")
    a_("--tr-run", default="", help="with --merge: tr_log run id to attach the summaries to")
    a_("--tr-stage", default="rl")
    a_("--tr-step", type=int, default=-1)
    a = ap.parse_args()
    merge(a) if a.merge else run(a)


if __name__ == "__main__":
    main()
