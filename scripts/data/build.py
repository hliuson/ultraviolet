"""Build math-corpus-v1: raw parquet/json -> decontaminated jsonl shards {text, source} + held-out problem pool.

Usage: python -I build.py RAW_DIR OUT_DIR [--workers 48] [--pool 20000]
Decontamination: word 13-grams (8-12 for short statements, <8 words skipped) of eval problems and pool
problems; any doc containing one is dropped. Val = 0.5% of docs by text hash.
"""
import argparse, glob, hashlib, json, os, random, re, sys, collections
from multiprocessing import Pool
import pyarrow.parquet as pq

N = 13
MIN_WORDS = 8
VAL_FRAC = 0.005
SHARD_BYTES = 512 << 20

LICENSES = {
    "finemath4": "ODC-BY", "cosmo_automath": "Apache-2.0", "cosmo_khan": "Apache-2.0",
    "numina_cot": "Apache-2.0", "numina15": "Apache-2.0",
    "omi2": "CC-BY-4.0", "metamath": "MIT", "openr1": "Apache-2.0", "omr_cot": "CC-BY-4.0",
}
KIND = {
    "finemath4": "web", "cosmo_automath": "textbook", "cosmo_khan": "textbook",
    "numina_cot": "qa_short", "numina15": "qa_short", "omi2": "qa_short", "metamath": "qa_short",
    "openr1": "trace_long_R1", "omr_cot": "trace_long_R1_QwQ",
}


def words(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).split()


def grams(ws, n):
    return {" ".join(ws[i:i + n]) for i in range(len(ws) - n + 1)}


def qa(problem, solution):
    return f"Problem:\n{problem.strip()}\n\nSolution:\n{solution.strip()}"


def first_correct_gen(row):
    gens = row.get("generations") or []
    ok = row.get("correctness_math_verify") or []
    done = row.get("is_reasoning_complete") or []
    for i, g in enumerate(gens):
        if i < len(ok) and ok[i] and (i >= len(done) or done[i]) and g:
            return g
    return None


def docs_from(source, path):
    """Yield text docs for one raw file."""
    if path.endswith(".json"):
        for r in json.load(open(path)):
            yield qa(r["query"], r["response"])
        return
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=2048):
        for r in batch.to_pylist():
            if source in ("finemath4", "cosmo_automath", "cosmo_khan", "cosmo_openstax", "nemo4plus", "nemomind"):
                t = r.get("text")
            elif source in ("numina_cot", "numina15"):
                if source == "numina15" and (not r.get("solution") or r.get("solution_is_valid") == "No"):
                    continue
                t = qa(r["problem"], r["solution"])
            elif source == "omi2":
                t = qa(r["problem"], r["generated_solution"])
            elif source == "openr1":
                g = first_correct_gen(r)
                t = f"Problem:\n{r['problem'].strip()}\n\n{g.strip()}" if g else None
            elif source == "omr_cot":
                t = f"Problem:\n{r['problem'].strip()}\n\n{r['generated_solution'].strip()}"
            else:
                raise ValueError(source)
            if t and len(t) > 50:
                yield t


# ---- decontamination sets (built in parent, inherited by forked workers)
BAD = {"eval": {}, "pool": {}, "pool2": {}}  # kind -> n -> set of n-grams


def load_eval(raw):
    probs = []
    for d in glob.glob(os.path.join(raw, "ev_*")):
        for f in glob.glob(os.path.join(d, "**", "*"), recursive=True):
            if f.endswith(".parquet"):
                rows = pq.read_table(f).to_pylist()
            elif f.endswith(".jsonl"):
                rows = [json.loads(l) for l in open(f)]
            else:
                continue
            for r in rows:
                p = r.get("problem") or r.get("question") or r.get("Problem") or r.get("prompt")
                if p:
                    probs.append((os.path.basename(d), p))
    return probs


def add_bad(text, kind="eval"):
    ws = words(text)
    if len(ws) < MIN_WORDS:
        return False
    n = min(N, len(ws))
    BAD[kind].setdefault(n, set()).update(grams(ws, n))
    return True


def contaminated(text):
    """Return 'eval', 'pool' or None."""
    ws = words(text)
    for kind in ("eval", "pool", "pool2"):
        for n, s in BAD[kind].items():
            for i in range(len(ws) - n + 1):
                if " ".join(ws[i:i + n]) in s:
                    return kind
    return None


def work(args):
    source, path, out_dir, idx, k, K = args  # process doc blocks (i // 1000) % K == k of this file
    stats = collections.Counter()
    outs = {}
    for split in ("train", "val"):
        os.makedirs(os.path.join(out_dir, split), exist_ok=True)
        outs[split] = open(os.path.join(out_dir, split, f"{source}-{idx:04d}.jsonl"), "w")
    for i, t in enumerate(docs_from(source, path)):
        if (i // 1000) % K != k:
            continue
        stats["docs_in"] += 1
        c = contaminated(t)
        if c:
            stats[f"dropped_{c}"] += 1
            continue
        h = int(hashlib.md5(t.encode()).hexdigest()[:8], 16)
        split = "val" if h < VAL_FRAC * 0xFFFFFFFF else "train"
        line = json.dumps({"text": t, "source": source}, ensure_ascii=False)
        outs[split].write(line + "\n")
        stats[f"{split}_docs"] += 1
        stats[f"{split}_bytes"] += len(t.encode())
    for f in outs.values():
        f.close()
    return source, path, dict(stats)


def build_pool(raw, n_pool, seed=0):
    INT = re.compile(r"^-?\d{1,7}$")
    rows = []
    for f in sorted(glob.glob(os.path.join(raw, "numina15", "**", "*.parquet"), recursive=True)):
        for r in pq.read_table(f).to_pylist():
            a = (r.get("answer") or "").strip().strip("$")
            p = r.get("problem") or ""
            if (r.get("question_type") == "math-word-problem" and r.get("problem_is_valid") != "No"
                    and r.get("solution_is_valid") != "No" and INT.match(a) and 40 <= len(p) <= 800
                    and "[asy]" not in p and "figure" not in p.lower()):
                rows.append({"problem": p.strip(), "answer": a, "source": r.get("source"),
                             "problem_type": r.get("problem_type"), "synthetic": r.get("synthetic")})
    rng = random.Random(seed)
    by = collections.defaultdict(list)
    for r in rows:
        by[r["source"]].append(r)
    for v in by.values():
        rng.shuffle(v)
    # round-robin across sources so no single source dominates
    pool, i = [], 0
    while len(pool) < n_pool and any(i < len(v) for v in by.values()):
        for v in by.values():
            if i < len(v) and len(pool) < n_pool:
                pool.append(v[i])
        i += 1
    return pool, {k: len(v) for k, v in by.items()}


NUM = re.compile(r"^-?\d+(\.\d+)?$|^-?\\\\?d?frac\{\d+\}\{\d+\}$|^-?\d+/\d+$")


def build_pool_bigmath(path, n_pool, seed=0):
    rows = []
    for r in pq.read_table(path).to_pylist():
        a = (r.get("answer") or "").strip().strip("$").replace("\\\\", "\\").replace(",", "")
        p = (r.get("problem") or "").strip()
        if NUM.match(a) and 40 <= len(p) <= 800 and "[asy]" not in p and "figure" not in p.lower():
            rows.append({"problem": p, "answer": a, "source": r.get("source"), "problem_type": r.get("domain"),
                         "synthetic": str(r.get("source") == "big_math"), "llama8b_solve_rate": r.get("llama8b_solve_rate")})
    rng = random.Random(seed)
    by = collections.defaultdict(list)
    for r in rows:
        by[r["source"]].append(r)
    for v in by.values():
        rng.shuffle(v)
    pool, i = [], 0
    while len(pool) < n_pool and any(i < len(v) for v in by.values()):
        for v in by.values():
            if i < len(v) and len(pool) < n_pool:
                pool.append(v[i])
        i += 1
    return pool, {k: len(v) for k, v in by.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw"); ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--pool", type=int, default=20000)
    ap.add_argument("--pool-only", action="store_true")
    ap.add_argument("--spec", help="JSON {source: {raw, max_files, kind, license}}; default = v1 mix from RAW")
    ap.add_argument("--wandb-id", default="math-corpus-v1")
    ap.add_argument("--bigmath", help="Big-Math parquet -> heldout_pool_bigmath.jsonl, excluded from pretraining")
    ap.add_argument("--pool2", type=int, default=30000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    ev = load_eval(a.raw)
    ev_used = collections.Counter(); ev_short = collections.Counter()
    for name, p in ev:
        (ev_used if add_bad(p) else ev_short)[name] += 1
    print("eval problems used", dict(ev_used), "skipped(short)", dict(ev_short), flush=True)

    pool, avail = build_pool(a.raw, a.pool * 2)
    pool = [r for r in pool if not contaminated(r["problem"])][: a.pool]
    for r in pool:
        add_bad(r["problem"], "pool")
    for i, r in enumerate(pool):
        r["id"] = f"pool-{i:05d}"
    with open(os.path.join(a.out, "heldout_pool.jsonl"), "w") as f:
        for r in pool:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("pool", len(pool), "by source", collections.Counter(r["source"] for r in pool), "avail", avail, flush=True)
    pool2 = []
    if a.bigmath:
        cand, avail2 = build_pool_bigmath(a.bigmath, a.pool2 * 2)
        n_eval = sum(1 for r in cand if contaminated(r["problem"]) == "eval")
        pool2 = [r for r in cand if contaminated(r["problem"]) != "eval"][: a.pool2]
        for i, r in enumerate(pool2):
            r["id"] = f"bigmath-{i:05d}"
            add_bad(r["problem"], "pool2")
        with open(os.path.join(a.out, "heldout_pool_bigmath.jsonl"), "w") as f:
            for r in pool2:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("pool2", len(pool2), "by source", collections.Counter(r["source"] for r in pool2),
              "avail(numeric)", avail2, "eval-dropped candidates", n_eval, flush=True)
    if a.pool_only:
        return

    spec = json.load(open(a.spec)) if a.spec else {s: {"raw": a.raw} for s in LICENSES}
    for s_, v in spec.items():
        v["raw"] = os.path.expanduser(v["raw"])
        LICENSES[s_] = v.get("license", LICENSES.get(s_)); KIND[s_] = v.get("kind", KIND.get(s_))
    jobs = []
    for source, v in spec.items():
        files = sorted(glob.glob(os.path.join(v["raw"], source, "**", "*.parquet"), recursive=True)
                       + glob.glob(os.path.join(v["raw"], source, "**", "*.json"), recursive=True))
        files = files[: v["max_files"]] if v.get("max_files") else files
        K = v.get("split", 1)
        jobs += [(source, f, a.out, i * K + k, k, K) for i, f in enumerate(files) for k in range(K)]
    agg = collections.defaultdict(collections.Counter)
    with Pool(a.workers) as p:
        for source, path, st in p.imap_unordered(work, jobs):
            agg[source].update(st)
            print(source, os.path.basename(path), st, flush=True)

    dl = {}
    for v in spec.values():
        dl.update(json.load(open(os.path.join(v["raw"], "download_manifest.json"))))
    n_files = collections.Counter(s_ for s_, _ in {(j[0], j[1]) for j in jobs})
    man = {}
    for s, st in agg.items():
        man[s] = dict(st, kind=KIND[s], license=LICENSES[s], repo=dl[s]["repo"], sha=dl[s]["sha"],
                      files=dl[s]["files"][: n_files[s]], est_tokens=(st["train_bytes"] + st["val_bytes"]) // 4)
    man["_eval_used"] = dict(ev_used); man["_eval_short_skipped"] = dict(ev_short)
    man["_pool"] = {"n": len(pool), "path": "heldout_pool.jsonl", "source_repo": "AI-MO/NuminaMath-1.5"}
    if pool2:
        man["_pool_bigmath"] = {"n": len(pool2), "path": "heldout_pool_bigmath.jsonl",
                                "source_repo": "SynthLabsAI/Big-Math-RL-Verified"}
    json.dump(man, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    tot = sum(v["est_tokens"] for k, v in man.items() if not k.startswith("_"))
    print("TOTAL est tokens", tot)
    for k, v in man.items():
        if not k.startswith("_"):
            print(f"{k:16s} {v['kind']:18s} {v['est_tokens']/1e9:6.3f}B  {100*v['est_tokens']/tot:5.1f}%  "
                  f"docs {v.get('train_docs',0)+v.get('val_docs',0):9d}  drop_eval {v.get('dropped_eval',0):6d} drop_pool {v.get('dropped_pool',0):6d} drop_pool2 {v.get('dropped_pool2',0):6d}  {v['license']}")
    try:
        import wandb
        run = wandb.init(project=os.environ.get("WANDB_PROJECT", "ultraviolet"), group="math-corpus", id=a.wandb_id, name=a.wandb_id,
                         resume="allow", job_type="data", config={"out": a.out, "n_gram": N, "val_frac": VAL_FRAC})
        tbl = wandb.Table(columns=["source", "kind", "license", "est_tokens", "docs", "dropped_eval", "dropped_pool"])
        for k, v in man.items():
            if not k.startswith("_"):
                tbl.add_data(k, v["kind"], v["license"], v["est_tokens"], v.get("train_docs", 0) + v.get("val_docs", 0),
                             v.get("dropped_eval", 0), v.get("dropped_pool", 0) + v.get("dropped_pool2", 0))
        run.log({"sources": tbl}); run.summary["total_est_tokens"] = tot; run.summary["pool"] = len(pool)
        run.finish()
    except Exception as e:
        print("wandb failed:", e)


if __name__ == "__main__":
    main()
