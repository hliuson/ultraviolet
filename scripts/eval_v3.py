"""Headline eval. Sampled only (no greedy): T=1.0, top-p 1, K samples per
problem, max-new 2048, bf16 weights + bf16 KV cache. Prompt `[EOS]Problem:\\n{p}\\n\\n<think>\\n`.

Math sets (rows: prompt, answer; MATH-500 rows carry meta.level): graded exactly like repl_eval / the RL reward
(rl_grpo.reward, answer-direct): ok = </think> closed and the answer after it correct. wf = exactly one </think> and
EOS within max-new; strict = ok & wf. answered = closed and a parseable answer after </think>. trunc = no EOS within
max-new. MATH-500 is also split by reference answer type (`m5-atype-v1`, anstype.py classify): numfrac
(int / decimal / plain fraction) vs scalar_other (sqrt, pi, units, ...) vs nonscalar (everything else).

  python eval_v3.py --ckpt C.pt --tok TOK --math math500=a.jsonl,... --k 16 --out-dir D --name N
"""
import argparse, collections, gzip, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def pass_at_k(n, c, k):
    out = np.ones(len(c))
    for i, (ni, ci) in enumerate(zip(n, c)):
        if ni - ci >= k:
            out[i] = 1.0 - np.prod(1.0 - k / np.arange(ni - ci + 1, ni + 1))
    return out


def maj(strict, pred, k):
    P, K = strict.shape
    res = np.zeros(P)
    for j in range(P):
        v = []
        for b in range(K // k):
            cnt = collections.Counter()
            okh = {}
            for s in range(b * k, (b + 1) * k):
                if pred[j, s] != 0:
                    cnt[pred[j, s]] += 1
                    okh[pred[j, s]] = okh.get(pred[j, s], False) or bool(strict[j, s])
            v.append(okh[cnt.most_common(1)[0][0]] if cnt else False)
        res[j] = np.mean(v)
    return res


def summ(d, max_new, mask=None):
    ok, wf, ans, ln, cl = d["ok"], d["wf"], d["answered"], d["length"], d["closed"]
    if mask is not None:
        ok, wf, ans, ln, cl = ok[mask], wf[mask], ans[mask], ln[mask], cl[mask]
    P, K = ok.shape
    st = ok & wf
    trunc = ln >= max_new
    o = {"P": int(P), "K": int(K), "pass@1_strict": float(st.mean()), "pass@1_plain": float(ok.mean()),
         f"pass@{K}_strict": float(pass_at_k(np.full(P, K), st.sum(1), K).mean()),
         "answer_rate": float(ans.mean()), "acc_given_answer": float(st.sum() / max(1, ans.sum())),
         "wellformed": float(wf.mean()), "close_rate": float(cl.mean()), "trunc": float(trunc.mean()),
         "len_med": float(np.median(ln)), "len_mean": float(ln.mean()),
         "len_mean_correct": float(ln[st].mean()) if st.any() else None}
    if "pred" in d:
        pr = d["pred"] if mask is None else d["pred"][mask]
        o[f"maj@{K}"] = float(maj(st, pr, K).mean())
    if "rew" in d:  # graded tasks: mean reward (strict = 0 for malformed rollouts), and the best-of-K reward
        rw = d["rew"] if mask is None else d["rew"][mask]
        o["reward_strict"] = float((rw * wf).mean())
        o["reward_plain"] = float(rw.mean())
        o[f"reward_best@{K}"] = float((rw * wf).max(1).mean())
    return o


def main():
    ap = argparse.ArgumentParser()
    a_ = ap.add_argument
    a_("--ckpt", required=True); a_("--tok", required=True); a_("--math", default="")
    a_("--k", type=int, default=16); a_("--max-new", type=int, default=2048); a_("--temperature", type=float, default=1.0)
    a_("--slots", type=int, default=1024); a_("--dtype", default="bf16"); a_("--seed", type=int, default=0)
    a_("--limit", type=int, default=0); a_("--out-dir", required=True); a_("--name", required=True)
    a_("--save-texts", type=int, default=2)
    a_("--compile", type=int, default=1, help="torch.compile the decode step (0: eager)")
    a_("--grader", default="v1", choices=["v1", "v2"], help="math answer checker for `ok` (rl_grpo.set_grader): v1 "
       "(default, eval_math.correct) or v2 (grader_v2); the npz also stores ok_v1 / ok_v2 for both")
    a = ap.parse_args()
    import torch
    from tokenizers import Tokenizer
    from eval_math import EOS, load
    from rl_grpo import extract_direct, reward, set_grader
    from eval_math import grader
    set_grader(a.grader)
    from repl_eval import hash_pred, wellformed
    from ultraviolet.model import generate_continuous
    from anstype import classify
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16}[a.dtype]
    tk = Tokenizer.from_file(a.tok)
    close_id = tk.token_to_id("</think>")
    torch.manual_seed(a.seed)
    m, _ = load(a.ckpt)
    m = m.to(dt)
    m.fuse_for_inference(dt)
    os.makedirs(a.out_dir, exist_ok=True)
    res = {}
    for spec in [s for s in a.math.split(",") if s]:
        name, path = spec.split("=")
        ps = [json.loads(l) for l in open(path)]
        if a.limit:
            ps = ps[:a.limit]
        P, K = len(ps), a.k
        pre = [[EOS] + tk.encode(p.get("prompt") or f"Problem:\n{p['problem'].strip()}\n\n<think>\n").ids for p in ps]
        t0 = time.time()
        raw = generate_continuous(m, pre * K, a.max_new, a.slots, temperature=a.temperature, eos_id=EOS, dtype=dt,
                                  compile=bool(a.compile))
        gen_s = time.time() - t0
        d = {k: np.zeros((P, K), bool) for k in ("ok", "wf", "answered", "closed")}
        d["length"] = np.zeros((P, K), np.int32)
        d["rew"] = np.zeros((P, K), np.float32)  # graded reward (docs/wb_envs.md); == ok for binary tasks
        d["pred"] = np.zeros((P, K), np.int64)
        d["ok_v1"] = np.zeros((P, K), bool); d["ok_v2"] = np.zeros((P, K), bool)
        texts = []
        for i, ro in enumerate(raw):
            s_, j = divmod(i, P)
            o = [t for t in ro if t != EOS]
            d["wf"][j, s_] = wellformed(ro, close_id, EOS)
            d["closed"][j, s_] = close_id in o
            d["length"][j, s_] = len(ro)
            txt = tk.decode(o, skip_special_tokens=False)
            r, c = reward(tk, close_id, o, ps[j], False, True)
            d["ok"][j, s_] = r >= 1.0  # graded wb_ tasks: solved = full reward
            d["rew"][j, s_] = r
            pv = extract_direct(txt) if c else None
            d["pred"][j, s_] = hash_pred(pv)
            d["answered"][j, s_] = c and pv is not None
            for gv in ("v1", "v2"):  # both checkers on the same extraction (reward: ok = closed & correct)
                d[f"ok_{gv}"][j, s_] = bool(c) and bool(grader(gv)(pv, ps[j].get("answer") if ps[j].get("ref") is None
                                                                    else ps[j]["ref"]))
            if s_ < a.save_texts:
                texts.append({"i": j, "s": s_, "text": txt})
        np.savez_compressed(os.path.join(a.out_dir, f"{a.name}.{name}.npz"), **d,
                            grader=np.array(a.grader))
        with gzip.open(os.path.join(a.out_dir, f"{a.name}.{name}.texts.jsonl.gz"), "wt") as f:
            for t in texts:
                f.write(json.dumps(t) + "\n")
        r = {"all": summ(d, a.max_new), "gen_seconds": gen_s, "grader": a.grader}
        for gv in ("v1", "v2"):
            r["all"][f"pass@1_strict_{gv}"] = float((d[f"ok_{gv}"] & d["wf"]).mean())
        if name == "math500":
            lv = np.array([p["meta"]["level"] for p in ps])
            r["by_level"] = {int(l): summ(d, a.max_new, lv == l) for l in sorted(set(lv))}
            at = np.array([{"scalar_numfrac": "numfrac", "scalar_other": "scalar_other"}.get(classify(p["answer"]),
                                                                                             "nonscalar") for p in ps])
            r["by_atype"] = {t: summ(d, a.max_new, at == t) for t in ("numfrac", "scalar_other", "nonscalar")}
        for fld in ("tier", "family"):  # workbench envs (docs/wb_envs.md): per-tier / per-family breakdown + chance
            if all(fld in p for p in ps):
                vals = np.array([str(p[fld]) for p in ps])
                r[f"by_{fld}"] = {v: summ(d, a.max_new, vals == v) for v in sorted(set(vals))}
        if all("chance" in p.get("meta", {}) for p in ps):
            r["all"]["chance"] = float(np.mean([p["meta"]["chance"] for p in ps]))
        res[name] = r
        print(name, json.dumps(r["all"]), f"gen {gen_s:.0f}s", flush=True)
        json.dump(res, open(os.path.join(a.out_dir, f"{a.name}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
