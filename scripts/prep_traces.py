"""Prepare teacher (Violetto) traces for high-repetition pretraining mixing, plus held-out trace and problem evals.

Each correct trace is rendered in the corpus R1/QwQ format:
    "Problem:\\n{problem}\\n\\n<think>\\n{thinking}\\n</think>\\n\\n{solution}"   (+ EOS when tokenized)
`<think>`/`</think>` are the tokenizer's reserved special tokens.

About --n-heldout traces are held out by problem (every trace of a chosen problem goes to the held-out set). The
rest are tokenized into a TraceStream dir (tokens.npy, offsets.npy). Separately, --n-eval "easy" problems
(--easy-sources) are drawn from the held-out pool, excluding every problem that has any trace.

  python scripts/prep_traces.py --traces 'DIR/*.jsonl' --pool heldout_pool.jsonl --tok tokenizer.json --out OUT
"""
import argparse
import glob
import json
import os

import numpy as np

EOS = 0


def truthy(v):
    return v is True or str(v).lower() in ("true", "1")


def render(problem: str, trace: str):
    """-> (text, thinking, solution) or None if the trace has no closed think block."""
    t = trace
    if "</think>" not in t:
        return None
    if "<think>" in t:
        t = t.split("<think>", 1)[1]
    thinking, solution = t.split("</think>", 1)
    thinking, solution = thinking.strip(), solution.strip()
    if not thinking or not solution:
        return None
    return f"Problem:\n{problem.strip()}\n\n<think>\n{thinking}\n</think>\n\n{solution}", thinking, solution


def prompt_text(problem: str) -> str:
    return f"Problem:\n{problem.strip()}\n\n<think>\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True, help="glob(s), comma-separated")
    ap.add_argument("--pool", required=True)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-heldout", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=500)
    ap.add_argument("--easy-sources", default="orca_math,metamath,synthetic_math")
    ap.add_argument("--all", action="store_true", help="keep incorrect traces too (default: correct only)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(a.tok)
    assert tk.token_to_id("<think>") is not None and tk.token_to_id("</think>") is not None

    files = sorted(f for g in a.traces.split(",") for f in glob.glob(g))
    recs, seen_pid, n_in, n_bad = [], set(), 0, 0
    for f in files:
        for line in open(f):
            r = json.loads(line)
            n_in += 1
            seen_pid.add(r.get("problem_id"))
            if not (a.all or truthy(r.get("correct"))):
                continue
            out = render(r.get("prompt") or r.get("problem"), r["trace"])
            if out is None:
                n_bad += 1
                continue
            recs.append({"problem_id": r.get("problem_id"), "source": r.get("source"), "text": out[0],
                         "problem": (r.get("prompt") or r.get("problem")).strip(), "thinking": out[1],
                         "solution": out[2], "ref": r.get("ref")})
    # hold out by problem
    rng = np.random.default_rng(a.seed)
    pids = sorted({r["problem_id"] for r in recs})
    rng.shuffle(pids)
    by = {}
    for r in recs:
        by.setdefault(r["problem_id"], []).append(r)
    held, n_h = set(), 0
    for p in pids:
        if n_h >= a.n_heldout:
            break
        held.add(p)
        n_h += len(by[p])
    train = [r for r in recs if r["problem_id"] not in held]
    heldout = [r for r in recs if r["problem_id"] in held]
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "heldout_traces.jsonl"), "w") as fh:
        for r in heldout:
            fh.write(json.dumps(r) + "\n")
    # tokenize train traces (epoch order is shuffled at training time)
    enc = tk.encode_batch([r["text"] for r in train])
    lens = np.array([len(e.ids) + 1 for e in enc], dtype=np.int64)
    off = np.concatenate([[0], np.cumsum(lens)])
    tok = np.empty(off[-1], dtype=np.uint16)
    for e, o in zip(enc, off[:-1]):
        tok[o:o + len(e.ids)] = e.ids
        tok[o + len(e.ids)] = EOS
    np.save(os.path.join(a.out, "tokens.npy"), tok)
    np.save(os.path.join(a.out, "offsets.npy"), off)
    nbytes = sum(len(r["text"].encode()) for r in train)
    # eval problems: easy sources, never traced
    easy = set(a.easy_sources.split(","))
    pool = [json.loads(l) for l in open(a.pool)]
    cand = [p for p in pool if p.get("source") in easy and p["id"] not in seen_pid and p.get("answer")]
    rng2 = np.random.default_rng(a.seed + 1)
    pick = rng2.choice(len(cand), size=min(a.n_eval, len(cand)), replace=False)
    with open(os.path.join(a.out, "eval_problems.jsonl"), "w") as fh:
        for i in sorted(pick):
            p = cand[i]
            fh.write(json.dumps({"id": p["id"], "source": p["source"], "problem": p["problem"],
                                 "answer": p["answer"], "prompt": prompt_text(p["problem"])}) + "\n")
    meta = {"files": files, "records_in": n_in, "kept": len(recs), "unclosed_or_empty": n_bad,
            "train_traces": len(train), "train_problems": len({r["problem_id"] for r in train}),
            "heldout_traces": len(heldout), "heldout_problems": len(held), "train_tokens": int(off[-1]),
            "train_bytes": nbytes, "bytes_per_token": nbytes / max(1, int(off[-1])), "eval_problems": len(pick),
            "eval_sources": sorted(easy), "tokenizer": os.path.abspath(a.tok), "correct_only": not a.all}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
