"""Checkpoint evals for the math LM.

1. Val bpb on one or more token-shard val splits (--val NAME=DIR ...; fixed evenly spaced windows, as in training).
2. Held-out teacher-trace NLL (--heldout heldout_traces.jsonl from prep_traces.py). Mean nats/token, reported
   separately for the thinking span (after `<think>` through `</think>`) and for the written solution (after
   `</think>` through EOS). The problem prefix is context only and is not scored.
3. Accuracy on held-out problems (--problems eval_problems.jsonl). The prompt is "Problem:\\n...\\n\\n<think>\\n"; the
   model generates up to --max-new tokens. The answer is the last \\boxed{} after </think> (else anywhere, else the
   last number after </think>) and is compared to the reference (string-normalised, or numerically). Greedy, plus --samples
   samples at --temperature.
4. --int8-check: teacher-forced next-token NLL and top-1 agreement of the decode path, int8 KV kernel vs fp16 SDPA,
   on held-out traces (prefill + step-by-step decode).

  python scripts/eval_math.py --ckpt RUN/final.pt --val v1=CORPUS_V1_TOK v2=CORPUS_V2_TOK --heldout H.jsonl \
      --problems P.jsonl --tok tokenizer.json --out RUN/eval.json [--wandb-name NAME]
"""
import argparse
import json
import math
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from ultraviolet.data import TokenShards  # noqa: E402
from ultraviolet.model import MathConfig, MathLM, generate_continuous  # noqa: E402

EOS = 0


def load(path):
    s = torch.load(path, map_location="cpu", weights_only=False)
    m = MathLM(MathConfig(**s["cfg"]))
    m.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in s["model"].items()})
    return m.cuda().eval(), s


@torch.no_grad()
def val_bpb(m, d, n_tokens, T, bs):
    v = TokenShards(d, "val")
    w = v.fixed_windows(n_tokens, T)
    tot = n = 0.0
    for i in range(0, len(w), bs):
        xb = w[i:i + bs].cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            loss = m(xb[:, :-1], xb[:, 1:])
        tot += loss.item() * xb[:, 1:].numel()
        n += xb[:, 1:].numel()
    return tot / n / math.log(2) / v.meta["val"]["bytes_per_token"]


@torch.no_grad()
def trace_nll(m, tk, recs, T, bs):
    think_open, think_close = tk.token_to_id("<think>"), tk.token_to_id("</think>")
    sums = {"thinking": [0.0, 0], "solution": [0.0, 0]}
    docs = []
    for r in recs:
        ids = [EOS] + tk.encode(r["text"]).ids + [EOS]
        docs.append(ids[:T + 1])
    for i in range(0, len(docs), bs):
        ch = docs[i:i + bs]
        L = max(len(d) for d in ch)
        x = torch.full((len(ch), L), EOS, dtype=torch.long)
        for j, d in enumerate(ch):
            x[j, :len(d)] = torch.tensor(d)
        x = x.cuda()
        with torch.autocast("cuda", dtype=torch.float16):
            lg = m(x[:, :-1])
        nll = F.cross_entropy(lg.float().transpose(1, 2), x[:, 1:], reduction="none").cpu().numpy()
        for j, d in enumerate(ch):
            d = np.array(d)
            o = np.flatnonzero(d == think_open)
            c = np.flatnonzero(d == think_close)
            if not len(o):
                continue
            o = o[0]
            c = c[0] if len(c) else len(d) - 1
            # target index t predicts token d[t+1]
            th = nll[j, o:c]                      # tokens o+1 .. c (thinking incl. </think>)
            so = nll[j, c:len(d) - 1]             # tokens c+1 .. end (solution incl. EOS)
            sums["thinking"][0] += th.sum()
            sums["thinking"][1] += len(th)
            sums["solution"][0] += so.sum()
            sums["solution"][1] += len(so)
    out = {k: round(float(s) / max(n, 1), 4) for k, (s, n) in sums.items()}
    out["tokens"] = {k: int(n) for k, (_, n) in sums.items()}
    tot = sum(s for s, _ in sums.values()) / max(1, sum(n for _, n in sums.values()))
    out["all"] = round(float(tot), 4)
    return out


# ------------------------------------------------------------------------------------------------ answer checking
def last_boxed(s):
    i = s.rfind("\\boxed")
    if i < 0:
        i = s.rfind("\\fbox")
        if i < 0:
            return None
    j = s.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(s)):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                return s[j + 1:k]
    return None



ANS = "</think>\n\nThe answer is $\\boxed{"  # answer-only suffix: the model completes the boxed answer


def ans_extract(t):
    """Answer from a completion of ANS: the boxed content, else the text up to the first } or $."""
    b = last_boxed("\\boxed{" + t)
    if b is not None:
        return b
    return t.split("}")[0].split("$")[0].strip() or None

def norm(a):
    a = str(a).strip()
    a = re.sub(r"\\text\{([^}]*)\}", r"\1", a)
    a = re.sub(r"\\(mathrm|mathbf)\{([^}]*)\}", r"\2", a)
    for t in ("\\!", "\\,", "\\;", "\\ ", "$", "\\left", "\\right", "^\\circ", "^{\\circ}", "\\%", "%", "dfrac",
              "tfrac"):
        a = a.replace(t, "frac" if t in ("dfrac", "tfrac") else "")
    a = a.replace(" ", "").rstrip(".").lower()
    return a


def to_num(a):
    a = norm(a).replace(",", "")
    m = re.fullmatch(r"-?\\frac\{(-?[\d.]+)\}\{(-?[\d.]+)\}", a)
    try:
        if m:
            v = float(m.group(1)) / float(m.group(2))
            return -v if a.startswith("-") else v
        if re.fullmatch(r"-?[\d.]+/[\d.]+", a):
            p, q = a.split("/")
            return float(p) / float(q)
        return float(a)
    except (ValueError, ZeroDivisionError):
        return None


def extract(text):
    """Last \\boxed{} after </think>; with no closed think block, only a \\boxed{} counts (no bare-number fallback,
    which would score random digit strings)."""
    if "</think>" not in text:
        return last_boxed(text)
    tail = text.split("</think>", 1)[1]
    b = last_boxed(tail)
    if b is not None:
        return b
    nums = re.findall(r"-?\d[\d,]*\.?\d*", tail)
    return nums[-1] if nums else None


def correct(pred, ref):
    if pred is None:
        return False
    if norm(pred) == norm(ref):
        return True
    a, b = to_num(pred), to_num(ref)
    return a is not None and b is not None and abs(a - b) <= 1e-6 * max(1.0, abs(b))


GRADERS = ("v1", "v2")


def grader(version="v1"):
    """Answer checker by version: v1 = correct above (the default everywhere); v2 = grader_v2.correct
    (canonicalised: brace-less \\frac / \\sqrt, units, \\$, MC letters, `x =`; see grader_v2.py)."""
    if version == "v1":
        return correct
    if version == "v2":
        from grader_v2 import correct as correct_v2
        return correct_v2
    raise ValueError(f"unknown grader {version!r}; expected one of {GRADERS}")


NUM = r"-?\d+(?:\.\d+)?"
OPS = {"+": "+", "-": "-", "*": "*", "\\times": "*", "\\cdot": "*", "\u00d7": "*", "x": "*", "/": "/",
       "\\div": "/", "\u00f7": "/", "^": "^"}
CLAIM = re.compile(r"(?<![\d.\w])(" + NUM + r")\s*(\\times|\\cdot|\\div|[+\-*/^\u00d7\u00f7x])\s*(" + NUM +
                   r")\s*=\s*(" + NUM + r")(?![\d.])")


def arith_claims(text):
    """Simple `a op b = c` claims (integers/decimals; + - * / ^). Returns (n_claims, n_correct). A claim is correct
    if c matches a op b to the precision c is written in (half a unit in its last decimal place, or 1e-9)."""
    n = ok = 0
    text = text.replace(",", "")
    for mt in CLAIM.finditer(text):
        a_, op, b_, c_ = mt.groups()
        before = text[:mt.start()].rstrip()
        if before.endswith(("+", "-", "*", "/", "^", "\u00d7", "\u00f7", "times", "cdot", "div", "(")):
            continue  # part of a longer expression (e.g. 2 + 3 + 4 = 9): not a simple binary claim
        try:
            a, b, c = float(a_), float(b_), float(c_)
            o = OPS.get(op, op)
            v = {"+": a + b, "-": a - b, "*": a * b, "/": a / b if b else float("nan"),
                 "^": a ** b if abs(b) <= 64 and abs(a) < 1e6 else float("nan")}[o]
        except (OverflowError, KeyError, ValueError, ZeroDivisionError):
            continue
        if v != v:
            continue
        n += 1
        dec = len(c_.split(".")[1]) if "." in c_ else 0
        tol = max(0.5 * 10 ** (-dec), 1e-9) if dec else 1e-9
        ok += abs(v - c) <= tol * max(1.0, 1.0 if dec else abs(c))
    return n, ok


def problem_number_coverage(problem, trace):
    nums = set(re.findall(r"\d+(?:\.\d+)?", problem.replace(",", "")))
    if not nums:
        return None
    t = trace.replace(",", "")
    return sum(re.search(r"(?<![\d.])" + re.escape(x) + r"(?![\d])", t) is not None for x in nums) / len(nums)


@torch.no_grad()
def accuracy(m, tk, probs, max_new, temperature, batch, n_samples=1):
    prompts = [[EOS] + tk.encode(p["prompt"]).ids for p in probs] * n_samples
    t0 = time.time()
    outs = generate_continuous(m, prompts, max_new, batch, temperature=temperature, eos_id=EOS, compile=True)
    dt = time.time() - t0
    n_ok, closed, lens, ex, texts = 0, 0, [], [], []
    per_prob = np.zeros(len(probs), dtype=np.int64)
    claims = ok_claims = 0
    claims_per, cover = [], []
    for i, o in enumerate(outs):
        o = [t for t in o if t != EOS]
        txt = tk.decode(o, skip_special_tokens=False)
        p = probs[i % len(probs)]
        pred = extract(txt)
        ok = correct(pred, p["answer"])
        n_ok += ok
        per_prob[i % len(probs)] += ok
        closed += "</think>" in txt
        lens.append(len(o))
        think = txt.split("</think>", 1)[0]
        c, k = arith_claims(think)
        claims += c
        ok_claims += k
        claims_per.append(c)
        cv = problem_number_coverage(p["problem"], think)
        if cv is not None:
            cover.append(cv)
        texts.append({"id": p["id"], "answer": p["answer"], "pred": pred, "correct": bool(ok),
                      "problem": p["problem"], "text": txt})
        if i < 3:
            ex.append({"id": p["id"], "answer": p["answer"], "pred": pred, "text": txt[:600]})
    n = len(outs)
    r = {"acc": round(n_ok / n, 4), "n": n, "closed_think": round(closed / n, 3),
         "mean_len": round(float(np.mean(lens)), 1), "seconds": round(dt, 1),
         "arith_claims_per_trace": round(float(np.mean(claims_per)), 3),
         "arith_claims_correct_frac": round(ok_claims / claims, 4) if claims else None,
         "problem_number_coverage": round(float(np.mean(cover)), 4) if cover else None, "examples": ex}
    if n_samples > 1:
        r[f"pass@{n_samples}"] = round(float((per_prob > 0).mean()), 4)
        r[f"maj_frac@{n_samples}"] = round(float((per_prob * 2 > n_samples).mean()), 4)
    return r, texts


@torch.no_grad()
def trace_use(m, tk, recs, T, bs):
    """Teacher-forced solution-span NLL (tokens after </think> through EOS) given (a) the correct thinking,
    (b) another problem's thinking (next record with a different problem_id), (c) empty thinking."""
    n = len(recs)
    other = []
    for i in range(n):
        j = (i + 1) % n
        while recs[j]["problem_id"] == recs[i]["problem_id"] and j != i:
            j = (j + 1) % n
        other.append(j)
    close = tk.token_to_id("</think>")
    out = {}
    for mode in ("correct", "other", "empty"):
        docs = []
        for i, r in enumerate(recs):
            th = {"correct": r["thinking"], "other": recs[other[i]]["thinking"], "empty": None}[mode]
            p = tk.encode(f"Problem:\n{r['problem'].strip()}\n\n<think>\n").ids
            mid = tk.encode(f"{th}\n</think>\n\n" if th is not None else "</think>\n\n").ids
            sol = tk.encode(r["solution"]).ids + [EOS]
            ids = [EOS] + p + mid + sol
            if len(ids) > T + 1:  # keep the solution: trim thinking from the left end of mid
                cut = len(ids) - (T + 1)
                mid = mid[cut:]
                ids = [EOS] + p + mid + sol
            docs.append((ids, len(ids) - len(sol)))
        tot = cnt = 0.0
        for i in range(0, n, bs):
            ch = docs[i:i + bs]
            L = max(len(d) for d, _ in ch)
            x = torch.full((len(ch), L), EOS, dtype=torch.long)
            y = torch.full((len(ch), L - 1), -1, dtype=torch.long)
            for j, (d, s0) in enumerate(ch):
                x[j, :len(d)] = torch.tensor(d)
                y[j, s0 - 1:len(d) - 1] = torch.tensor(d[s0:])
            x, y = x.cuda(), y.cuda()
            with torch.autocast("cuda", dtype=torch.float16):
                lg = m(x[:, :-1])
            nll = F.cross_entropy(lg.float().transpose(1, 2), y, reduction="sum", ignore_index=-1)
            tot += nll.item()
            cnt += (y != -1).sum().item()
        out[mode] = round(tot / cnt, 4)
    out["gain_vs_empty"] = round(out["empty"] - out["correct"], 4)
    out["gain_vs_other"] = round(out["other"] - out["correct"], 4)
    _ = close
    return out


@torch.no_grad()
def int8_check(m, tk, recs, n_docs, T):
    """Teacher-forced decode over held-out traces: SDPA fp16 cache vs int8 kernel cache."""
    docs = [[EOS] + tk.encode(r["text"]).ids for r in recs[:n_docs]]
    L = min(T, min(len(d) for d in docs))
    x = torch.tensor([d[:L] for d in docs]).cuda()
    B = x.size(0)
    m16 = m.half()
    res = {}
    lgs = {}
    for impl in ("sdpa", "kernel-int8"):
        m16.decode_impl = impl
        st = m16.new_state(B, L + 1, torch.float16, "cuda")
        first = m16.prefill(x[:, :1], torch.zeros(B, dtype=torch.long, device="cuda"), st)
        out = [first.float()]
        for t in range(1, L - 1):
            out.append(m16.decode_step(x[:, t], st, g_len=L + 1).float())
        lgs[impl] = torch.stack(out, 1)  # (B, L-1, V): predicts x[:, 1:L]
        res[impl] = round(F.cross_entropy(lgs[impl].transpose(1, 2), x[:, 1:L]).item(), 5)
    m16.decode_impl = "auto"
    agree = (lgs["sdpa"].argmax(-1) == lgs["kernel-int8"].argmax(-1)).float().mean().item()
    kl = F.kl_div(lgs["kernel-int8"].log_softmax(-1), lgs["sdpa"].log_softmax(-1), log_target=True,
                  reduction="none").sum(-1).mean().item()
    return {"nll_sdpa_fp16": res["sdpa"], "nll_kernel_int8": res["kernel-int8"],
            "rel_change": round(res["kernel-int8"] / res["sdpa"] - 1, 6), "top1_agree": round(agree, 5),
            "kl": kl, "tokens": int(B * (L - 1))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--val", nargs="*", default=[])
    ap.add_argument("--val-tokens", type=float, default=4e6)
    ap.add_argument("--heldout", default=None)
    ap.add_argument("--problems", default=None)
    ap.add_argument("--max-new", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--trace-use", type=int, default=1)
    ap.add_argument("--n-dump", type=int, default=5, help="sample generations dumped to <out>.samples.txt")
    ap.add_argument("--wandb-group", default="math-pt")
    ap.add_argument("--wandb-prefix", default="mpt")
    ap.add_argument("--gen-batch", type=int, default=256)
    ap.add_argument("--int8-check", type=int, default=0, help="number of held-out traces for the int8 check")
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wandb-name", default=None)
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(a.tok)
    m, s = load(a.ckpt)
    res = {"ckpt": os.path.abspath(a.ckpt), "step": s.get("step")}
    for spec in a.val:
        nm, d = spec.split("=", 1)
        res[f"bpb_{nm}"] = round(val_bpb(m, d, int(a.val_tokens), a.seq, a.bs), 5)
        print(nm, res[f"bpb_{nm}"], flush=True)
    recs = [json.loads(l) for l in open(a.heldout)] if a.heldout else []
    if recs:
        res["heldout_trace_nll"] = trace_nll(m, tk, recs, a.seq, a.bs)
        print("trace nll", res["heldout_trace_nll"], flush=True)
        if a.trace_use:
            res["trace_use"] = trace_use(m, tk, recs, a.seq, a.bs)
            print("trace use", res["trace_use"], flush=True)
    if a.problems:
        probs = [json.loads(l) for l in open(a.problems)]
        mh = m.half()
        mh.fuse_for_inference(torch.float16)
        res["acc_greedy"], gtexts = accuracy(mh, tk, probs, a.max_new, 0.0, a.gen_batch)
        print("greedy", {k: v for k, v in res["acc_greedy"].items() if k != "examples"}, flush=True)
        key = f"acc_t{a.temperature}x{a.samples}"
        res[key], stexts = accuracy(mh, tk, probs, a.max_new, a.temperature, a.gen_batch, a.samples)
        print("sampled", {k: v for k, v in res[key].items() if k != "examples"}, flush=True)
        with open(os.path.splitext(a.out)[0] + ".samples.txt", "w") as fh:
            for tag, tx in (("greedy", gtexts), (f"T={a.temperature}", stexts)):
                for t in tx[:a.n_dump]:
                    fh.write(f"===== [{tag}] {t['id']}  ref={t['answer']}  pred={t['pred']}  correct={t['correct']}\n"
                             f"PROBLEM: {t['problem']}\n---\n{t['text']}\n\n")
        with open(os.path.splitext(a.out)[0] + ".generations.jsonl", "w") as fh:
            for tag, tx in (("greedy", gtexts), ("sampled", stexts)):
                for t in tx:
                    fh.write(json.dumps({"mode": tag, **t}) + "\n")
        mh.train(False)
    if a.int8_check and recs:
        res["int8_check"] = int8_check(m, tk, recs, a.int8_check, min(a.seq, 2048))
        print("int8", res["int8_check"], flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    if a.wandb_name:
        import wandb
        run = wandb.init(project=os.environ.get("WANDB_PROJECT", "ultraviolet"), group=a.wandb_group, name=a.wandb_name, id=f"{a.wandb_prefix}-{a.wandb_name}",
                         resume="allow", dir=os.path.dirname(os.path.abspath(a.out)),
                         settings=wandb.Settings(silent=True))
        flat = {}
        for k, v in res.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    if isinstance(v2, (int, float)):
                        flat[f"eval/{k}/{k2}"] = v2
            elif isinstance(v, (int, float)):
                flat[f"eval/{k}"] = v
        run.summary.update(flat)
        run.finish()


if __name__ == "__main__":
    main()
