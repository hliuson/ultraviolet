"""Shared helpers for the workbench RL envs (docs/wb_envs.md): answer parsing for list / set / pair / grid answers,
the graded reward functions, the pool row format, and the `build` CLI every env module exposes.

Env module contract (see resid.py for the pattern):
    FORMAT = "Give the ... separated by commas."      # format sentence, placed right before INSTR
    gen(rng, tier) -> (body, answer, meta, family) or None   # body = problem text without the format sentences
    sc_<env>(pred, item) -> float in [0, 1]              # pred = content of the last \\boxed{}
    TASKS["wb_<env>"] = {"kind": "list", "scorer": sc_<env>, "reward": "graded"}
    if __name__ == "__main__": build_cli("wb_<env>", gen, FORMAT)
Rewards are graded: a vector answer scores the fraction of components correct (wrong length = 0), a set answer the F1
against the gold set, a grid the fraction of blank cells correct. `meta["chance"]` is the expected reward of a random
well-formed answer; `tier` (0 / 1 / 2) and `family` are top-level row fields.
"""
import argparse
import json
import os
import random
import re
import zlib
from fractions import Fraction

from ultraviolet.tasks.score import parse_number, score, strip_latex

INSTR = "Put the final answer in \\boxed{}."
SEED_BASE = {"eval": 30_000_000, "rl": 20_000_000, "sft": 10_000_000}  # same convention as ultraviolet/tasks/build.py
SPLIT_ORDER = ("eval", "rl", "sft")  # sft: problems for a teacher-trace SFT split; built last
NONE_WORDS = {"none", "nosolution", "nosolutions", "nosolutionexists", "emptyset", "empty", "\\emptyset",
              "\\varnothing", "∅", "{}", "\\{\\}", "nointegersolutions", "noroots", "nosuchx"}


def prompt_text(problem):
    return f"Problem:\n{problem}\n\n<think>\n"


# ----------------------------------------------------------------------------------------------- parsing
def clean(s):
    """Normalise a boxed answer: drop LaTeX wrappers, set braces, 'and', trailing period."""
    s = str(s).replace("\\\\", ",").replace("{,}", ",")  # LaTeX row breaks / thin-space commas "4{,}9{,}2"
    s = re.sub(r"\\(?:begin|end)\{[a-z*]*\}(?:\{[^{}]*\})?", " ", s)  # \begin{array}{ccc} ... \end{array}
    s = strip_latex(s).replace("\u2212", "-").replace("\u00d7", "*")
    s = s.replace("&", ",").replace("\\{", "").replace("\\}", "").replace("\\;", " ")
    s = re.sub(r"\\(?:text|mathrm)\s*\{\s*(?:and|or)\s*\}", ",", s)
    s = re.sub(r"\s+(?:and|or)\s+", ",", s)
    s = s.strip().rstrip(".").strip()
    # "(x, y) = (3, 4)" / "x, y = 3, 4": keep the right-hand side when the left is only variable names
    m = re.fullmatch(r"\(?\s*[A-Za-z]\w*(?:\s*,\s*[A-Za-z]\w*)*\s*\)?\s*=\s*(.*)", s, re.S)
    if m:
        s = m.group(1).strip()
    # outer wrapper that encloses the whole answer: {1, 2}, [1, 2]; (3, 4) only when nothing nested is inside
    for o, c in (("{", "}"), ("[", "]"), ("(", ")")):
        if s.startswith(o) and s.endswith(c) and _balanced_outer(s, o, c):
            if o == "(" and re.search(r"[()\[\]]", s[1:-1]):
                break
            s = s[1:-1].strip()
    return s


def _balanced_outer(s, o, c):
    depth = 0
    for i, ch in enumerate(s):
        if ch == o:
            depth += 1
        elif ch == c:
            depth -= 1
            if depth == 0 and i != len(s) - 1:
                return False
    return depth == 0


def split_top(s):
    """Split on commas / semicolons outside parentheses. Falls back to whitespace when there is no separator and every
    whitespace-separated token is numeric ('3 5 7')."""
    out, depth, cur = [], 0, []
    for ch in s:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch in ",;" and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    out = [t.strip() for t in out if t.strip()]
    if len(out) == 1 and "(" not in out[0]:
        toks = out[0].split()
        if len(toks) > 1 and all(parse_number(t, allow_units=False) is not None for t in toks):
            return toks
    return out


def is_none(s):
    return re.sub(r"[\s$]", "", clean(s)).lower() in NONE_WORDS


def parse_num(tok):
    """Exact rational of one token ('x = 3', '-7/2', '\\frac{1}{2}', '0.5'); None if not a number."""
    return parse_number(tok, allow_units=False)


def parse_int(tok):
    v = parse_num(tok)
    return int(v) if v is not None and v.denominator == 1 else None


def parse_pair(tok):
    """'(x, y)' / 'x, y' / '(x=1, y=2)' -> (Fraction, Fraction) or None."""
    t = tok.strip()
    if t.startswith("(") and t.endswith(")"):
        t = t[1:-1]
    parts = [p.strip() for p in t.split(",")]
    if len(parts) != 2:
        return None
    a, b = parse_num(parts[0]), parse_num(parts[1])
    return None if a is None or b is None else (a, b)


def parse_list(pred, kind="int"):
    """Boxed content -> list of parsed items (ints / Fractions / pairs) or None if any item fails to parse. 'none' ->
    []."""
    if pred is None:
        return None
    if is_none(pred):
        return []
    f = {"int": parse_int, "num": parse_num, "pair": parse_pair}[kind]
    c = clean(pred)
    toks = split_top(c)
    if not toks:
        return None
    out = [f(t) for t in toks]
    if kind == "pair" and any(x is None for x in out) and len(toks) == 2 and "(" not in c:
        one = parse_pair(c)  # a single pair whose outer parentheses clean() removed: "(3, 4)" -> "3, 4"
        return None if one is None else [one]
    return None if any(x is None for x in out) else out


# ----------------------------------------------------------------------------------------------- rewards
def frac_correct(pred, gold):
    """Vector reward: fraction of positions equal; wrong length (or unparsed) = 0."""
    if pred is None or len(pred) != len(gold) or not gold:
        return 0.0
    return sum(1 for p, g in zip(pred, gold) if p == g) / len(gold)


def vec_credit(pred, gold):
    """Vector reward, prefix-aligned: matches at aligned positions / max(len(pred), len(gold)). A single correct first
    component earns 1/k, so a policy that still boxes one number (the midtrain habit) is pulled toward the full
    vector; extra components dilute the score. Equals frac_correct when the lengths agree."""
    if pred is None or not pred or not gold:
        return 0.0
    n = min(len(pred), len(gold))
    return sum(1 for i in range(n) if pred[i] == gold[i]) / max(len(pred), len(gold))


def f1_set(pred, gold):
    """Set reward: F1 of the deduplicated prediction against the gold set; gold = {} scores 1 only for 'none'."""
    if pred is None:
        return 0.0
    p, g = set(pred), set(gold)
    if not g:
        return 1.0 if not p else 0.0
    if not p:
        return 0.0
    tp = len(p & g)
    if tp == 0:
        return 0.0
    prec, rec = tp / len(p), tp / len(g)
    return 2 * prec * rec / (prec + rec)


def frac_cells(pred, gold, blank):
    """Grid reward: fraction of blank cells (indices into the flattened grid) equal to the gold; wrong length = 0."""
    if pred is None or len(pred) != len(gold) or not blank:
        return 0.0
    return sum(1 for i in blank if pred[i] == gold[i]) / len(blank)


def mc_chance(rng, sample, reward, n=200):
    """Monte Carlo chance level: mean reward of n random well-formed answers drawn by sample(rng)."""
    return sum(reward(sample(rng)) for _ in range(n)) / n


# ----------------------------------------------------------------------------------------------- items
def qkey(q):
    return re.sub(r"\s+", " ", q.strip().lower())


def make_item(task, family, tier, split, k, body, fmt, answer, meta):
    """fmt: the format sentence, or a callable (family, meta) -> sentence (e.g. a different example per answer kind)."""
    if callable(fmt):
        fmt = fmt(family, meta)
    problem = body.strip() + "\n" + (fmt.strip() + " " if fmt else "") + INSTR
    return {"id": f"{task}-{split}-{k:06d}", "task": task, "source": task, "family": family, "tier": int(tier),
            "split": split, "problem": problem, "answer": answer, "ref": answer, "prompt": prompt_text(problem),
            "meta": dict(meta)}


def build_items(task, gen, fmt, sizes, seed=0, tier_weights=(1, 1, 1), max_tries_mult=6, check=True):
    """sizes: {split: n}. Items are tier-stratified (cycling through tiers by weight), disjoint across splits by
    normalised problem text, and every gold answer is self-checked to score 1.0. Returns ({split: [items]}, stats)."""
    seen, out, stats = set(), {}, {}
    tiers = [t for t, w in enumerate(tier_weights) for _ in range(int(w))]
    for split in SPLIT_ORDER:
        n = sizes.get(split, 0)
        items, tries, dup, rej = [], 0, 0, 0
        cap = max_tries_mult * n + 100
        for i in range(cap):
            if len(items) >= n:
                break
            tries += 1
            rng = random.Random(SEED_BASE[split] + 1_000_003 * seed + i)
            tier = tiers[len(items) % len(tiers)]  # the tier advances only on an accepted item: exact stratification
            r = gen(rng, tier)
            if r is None:
                rej += 1
                continue
            body, ans, meta, family = r
            key = qkey(body)
            if key in seen:
                dup += 1
                continue
            seen.add(key)
            it = make_item(task, family, tier, split, len(items), body, fmt, ans, meta)
            it["meta"]["seed"] = SEED_BASE[split] + 1_000_003 * seed + i
            if check and score(task, f"</think>\n\n\\boxed{{{ans}}}", it) != 1.0:
                raise AssertionError(f"gold does not score 1: {task} {it}")
            items.append(it)
        out[split] = items
        stats[split] = {"n": len(items), "tries": tries, "dup": dup, "rejected": rej}
    return out, stats


def summarize(items):
    by = {}
    for it in items:
        for key in (f"tier{it['tier']}", it["family"]):
            d = by.setdefault(key, {"n": 0, "chance": 0.0})
            d["n"] += 1
            d["chance"] += it["meta"].get("chance", 0.0)
    for d in by.values():
        d["chance"] = round(d["chance"] / max(1, d["n"]), 4)
    return {"n": len(items), "chance": round(sum(it["meta"].get("chance", 0.0) for it in items) / max(1, len(items)), 4),
            "by": by}


def write_pools(out_dir, task, split_items, stats, gr_chance=0.1):
    os.makedirs(out_dir, exist_ok=True)
    for split, items in split_items.items():
        if not items:
            continue
        with open(os.path.join(out_dir, f"{split}.jsonl"), "w") as f:
            for it in items:
                f.write(json.dumps(it) + "\n")
    ev = split_items.get("eval", [])
    filt = {"eval_guess_resistant_ids": [it["id"] for it in ev if it["meta"].get("chance", 0.0) <= gr_chance],
            "gr_chance": gr_chance}
    json.dump(filt, open(os.path.join(out_dir, "filters.json"), "w"))
    summ = {"task": task, "stats": stats, **{s: summarize(v) for s, v in split_items.items()},
            "eval_gr": len(filt["eval_guess_resistant_ids"])}
    json.dump(summ, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    return summ


def build_cli(task, gen, fmt, argv=None, tier_weights=(1, 1, 1)):
    ap = argparse.ArgumentParser(description=f"build {task} pools (docs/wb_envs.md)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--n-rl", type=int, default=5000)
    b.add_argument("--n-eval", type=int, default=300)
    b.add_argument("--n-sft", type=int, default=0, help="problems for a teacher-trace SFT split")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--tier-weights", default=",".join(map(str, tier_weights)))
    b.add_argument("--gr-chance", type=float, default=0.1)
    s = sub.add_parser("show", help="print a few sample problems per tier")
    s.add_argument("--n", type=int, default=2)
    s.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.cmd == "show":
        for tier in range(3):
            k = 0
            for i in range(1000):
                r = gen(random.Random(a.seed * 1000 + 7 * i + tier), tier)
                if r is None:
                    continue
                body, ans, meta, fam = r
                f = fmt(fam, meta) if callable(fmt) else fmt
                print(f"--- tier {tier} [{fam}] chance={meta.get('chance', 0):.3f}\n{body}\n{f} {INSTR}\n=> {ans}\n")
                k += 1
                if k >= a.n:
                    break
        return
    tw = tuple(int(x) for x in a.tier_weights.split(","))
    items, stats = build_items(task, gen, fmt, {"eval": a.n_eval, "rl": a.n_rl, "sft": a.n_sft}, seed=a.seed, tier_weights=tw)
    summ = write_pools(a.out, task, items, stats, a.gr_chance)
    print(json.dumps(summ, indent=1))


def crc(s, mod=1000):
    return zlib.crc32(s.encode()) % mod


__all__ = ["INSTR", "Fraction", "prompt_text", "clean", "split_top", "is_none", "parse_num", "parse_int", "parse_pair",
           "parse_list", "frac_correct", "vec_credit", "f1_set", "frac_cells", "mc_chance", "make_item", "build_items",
           "write_pools", "build_cli", "qkey", "crc"]
