"""Build deterministic, split-disjoint task items (docs/tasks.md).

Item row: {id, task, source, split, problem, answer, ref, prompt, meta}
  problem: question text ending in exactly one answer-format instruction (reasoning-gym's own format instructions
           stripped); prompt = "Problem:\\n{problem}\\n\\n<think>\\n" (rl_grpo format); answer == ref is the
           normalised gold; meta holds what the scorer needs (countdown numbers/target, sudoku puzzle, ...).
Splits: eval, rl, sft. Built in that order; every later split drops items whose normalised question already occurs in
an earlier split (so RL/eval never overlap SFT, and eval never overlaps RL). reasoning-gym tasks draw each split from
its own seed range (RG item i of a dataset with seed s uses Random(s + i)): eval 30_000_000, rl 20_000_000,
sft 10_000_000. HF datasets split by their own train/test split, then by a crc32 hash of the row id.

reasoning-gym is needed only here (build time), not by the scorer.
"""
import json
import os
import random
import re
import zlib

INSTR = "Put the final answer in \\boxed{}."
SEED_BASE = {"eval": 30_000_000, "rl": 20_000_000, "sft": 10_000_000}
SPLIT_ORDER = ("eval", "rl", "sft")


def prompt_text(problem):
    return f"Problem:\n{problem.strip()}\n\n<think>\n"


def qkey(q):
    return re.sub(r"\s+", " ", q.strip().lower())


def _sentences_strip(q, pats):
    for p in pats:
        q = re.sub(p, "", q, flags=re.S)
    return q.strip()


# --------------------------------------------------------------------------------------------- reasoning-gym
# Each spec: rg dataset name, constructor knobs (the difficulty knobs), and rewrite(entry) -> (question, answer, meta)
# or None (reject). Knobs can be overridden from the CLI with --knobs task:json.

def rw_chain_sum(e):
    return f"Compute {e['metadata']['expression']}.", e["answer"], {}


def gen_products(rng, min_digits=2, max_digits=3):
    """2x2 to 3x2-digit products (reasoning-gym's products draws one digit count for both factors, so 2x3 never
    occurs and 2x2 alone is only ~8k questions)."""
    da = rng.randint(min_digits, max_digits)
    a = rng.randint(10 ** (da - 1), 10 ** da - 1)
    b = rng.randint(10, 99)
    if rng.random() < 0.5:
        a, b = b, a
    return {"question": "", "answer": str(a * b), "metadata": {"expression": f"{a} * {b}"}}


def rw_products(e):
    expr = e["metadata"]["expression"]
    a, b = [x.strip() for x in expr.split("*")]
    da, db = sorted((len(a.lstrip("-")), len(b.lstrip("-"))))
    if (da, db) not in ((2, 2), (2, 3)):
        return None
    return f"Compute {a} * {b}.", e["answer"], {"digits": [da, db]}


def rw_basic_arithmetic(e):
    if not re.fullmatch(r"-?\d+", e["answer"]):
        return None
    q = e["question"].strip()
    q = re.sub(r"^Calculate\s+", "", q).rstrip(".")
    q = re.sub(r"\s*Ensure.*$", "", q, flags=re.S)
    return f"Calculate {q}.", e["answer"], {}


def rw_simple_equations(e):
    m = e["metadata"]
    return f"Solve for {m['variable']}: {m['equation']}", e["answer"], {}


def rw_gcd(e):
    a, b = e["metadata"]["numbers"]
    g = e["metadata"]["result"]
    if g == 1 and (zlib.crc32(f"{a},{b}".encode()) % 100) >= 15:   # rebalance: P(gcd = 1) ~ 0.6 -> ~0.2
        return None
    return f"Find the greatest common divisor of {a} and {b}.", str(g), {}


def rw_lcm(e):
    a, b = e["metadata"]["numbers"]
    return f"Find the least common multiple of {a} and {b}.", str(e["metadata"]["result"]), {}


def rw_fraction(e):
    m = e["metadata"]
    n, d = m["numerator"], m["denominator"]
    sn, sd = m["simplified_numerator"], m["simplified_denominator"]
    if sd == 1 or m["reduction_factor"] == 1:
        return None
    return f"Simplify the fraction {n}/{d} to lowest terms.", f"{sn}/{sd}", {}


def _base_name(b):
    return {2: "base-2 (binary)", 16: "base-16 (hexadecimal)", 8: "base-8"}.get(b, f"base-{b}")


def rw_base(e):
    m = e["metadata"]
    sb, tb = m["source_base"], m["target_base"]
    q = f"Convert the {_base_name(sb)} number {m['source_repr']} to {_base_name(tb)}."
    if tb > 10:
        q += " Use lowercase letters a-z for digits above 9."
    return q, m["target_repr"], {"source_base": sb, "target_base": tb}


def rw_count_bits(e):
    n = e["metadata"]["number"]
    return f"How many 1 bits are there in the binary representation of {n}?", e["answer"], {}


def rw_time(e):
    q = e["question"].strip()
    fmt = "HH:MM:SS" if "HH:MM:SS" in q else "HH:MM"
    if "mmm" in q:
        return None
    sents = re.split(r"(?<=[.?])\s+", q)
    sents = [s for s in sents if "HH:MM" not in s]
    q = " ".join(sents)
    return q + f" Give the duration in {fmt} format.", e["answer"], {"format": fmt}


_CAL_FMT = re.compile(r"^(Respond|Provide|Give|Answer|Reply|State|Write|Express|Please|Format|Return|Just|Only)\b"
                      r"|^Name the day of the week in full")


def rw_calendar(e):
    q = e["question"].strip()
    if e["metadata"].get("task") == "is_leap_year" or e["answer"] in ("Yes", "No"):
        return None
    sents = re.split(r"(?<=[.?])\s+", q)
    while len(sents) > 1 and _CAL_FMT.match(sents[-1]):
        sents.pop()
    q = " ".join(sents)
    if e["answer"].lower() in ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"):
        q += " Give the full weekday name."
    return q, e["answer"], {"cal_task": e["metadata"].get("task")}


def rw_legs(e):
    """reasoning-gym's leg table is idiosyncratic (crab 10, woodlouse 14, ...), so each animal's leg count is stated:
    the task is the multi-step arithmetic, not zoology trivia."""
    from reasoning_gym.arithmetic.leg_counting import ANIMALS
    animals = e["metadata"]["animals"]
    parts = [f"{n} {a}{'' if n == 1 else 's'} ({ANIMALS[a]} leg{'' if ANIMALS[a] == 1 else 's'} each)"
             for a, n in animals.items()]
    total = sum(ANIMALS[a] * n for a, n in animals.items())
    assert str(total) == str(e["answer"])
    return f"How many legs are there in total if you have {', '.join(parts)}?", e["answer"], {}


_GSMS_PATS = [r"\s*Give the result as your final answer\.?", r"\s*Do not include units\.?",
              r"\s*Give your answer as .*?\.$", r"\s*Provide .*final answer.*?\.$"]


def rw_gsm_symbolic(e):
    q = _sentences_strip(e["question"], _GSMS_PATS)
    if not re.fullmatch(r"-?\d+", str(e["answer"])):
        return None
    return q, str(e["answer"]), {}


def rw_countdown(e):
    m = e["metadata"]
    nums = list(m["numbers"])
    ns = ", ".join(str(x) for x in nums)
    q = (f"Using each of the numbers {ns} exactly once, together with +, -, * and / and parentheses, write an "
         f"arithmetic expression that equals {m['target']}.")
    return q, m["expression"], {"numbers": nums, "target": m["target"]}


def rw_letters(e):
    m = e["metadata"]
    return (f"How many times does the letter \"{m['target_letter']}\" appear in the text: \"{' '.join(m['span'])}\"?",
            e["answer"], {})


def rw_spell(e):
    w = e["metadata"]["word"]
    return f"Spell the word \"{w}\" backward.", e["answer"], {}


def rw_sudoku(e):
    m = e["metadata"]
    grid = "\n".join(" ".join(str(x) if x else "_" for x in row) for row in m["puzzle"])
    q = ("Solve this 4x4 Mini Sudoku. Each row, each column and each 2x2 box must contain the numbers 1-4 exactly "
         f"once. Empty cells are shown as _.\n{grid}\nGive the completed grid as four rows of digits separated by "
         "spaces, e.g. 1234 3412 2143 4321.")
    ans = " ".join("".join(str(x) for x in row) for row in m["solution"])
    return q, ans, {"puzzle": m["puzzle"]}


def rw_knights(e):
    m = e["metadata"]
    q = re.sub(r"\s*\(Format your answer like:.*\)\s*$", "", e["question"].strip(), flags=re.S)
    t = m["knight_knave_terms"]
    n0, n1 = m["names"]
    n0, n1 = str(n0), str(n1)
    opt = f"{t['a_knight']}/{t['a_knave']}"
    q += f" Answer in the form \"{n0} is {opt}, {n1} is {opt}\", choosing one for each."
    ans = ", ".join(f"{n} is {t['a_knight'] if s else t['a_knave']}" for n, s in zip(m["names"], m["solution"]))
    return q, ans, {"names": [str(x) for x in m["names"]], "solution": [bool(x) for x in m["solution"]],
                    "knight": str(t["knight"]), "knave": str(t["knave"])}


RG = {
    "chain_sum": ("chain_sum", dict(min_terms=3, max_terms=5, min_digits=2, max_digits=3), rw_chain_sum),
    "products": (gen_products, dict(min_digits=2, max_digits=3), rw_products),
    "basic_arithmetic": ("basic_arithmetic", dict(min_terms=3, max_terms=4, min_digits=1, max_digits=2),
                         rw_basic_arithmetic),
    "simple_equations": ("simple_equations", dict(min_terms=2, max_terms=3, min_value=1, max_value=50),
                         rw_simple_equations),
    "gcd": ("gcd", dict(min_numbers=2, max_numbers=2, min_value=2, max_value=300), rw_gcd),
    "lcm": ("lcm", dict(min_numbers=2, max_numbers=2, min_value=2, max_value=100), rw_lcm),
    "fraction_simplification": ("fraction_simplification", dict(min_value=1, max_value=60, min_factor=2,
                                                                max_factor=20, styles=("plain",)), rw_fraction),
    "base_conversion": ("base_conversion", dict(min_base=2, max_base=16, min_value=0, max_value=300), rw_base),
    "count_bits": ("count_bits", dict(min_n=1, max_n=65535), rw_count_bits),
    "time_intervals": ("time_intervals", dict(task_types=["time", "time_seconds"]), rw_time),
    "calendar_arithmetic": ("calendar_arithmetic", dict(tasks=["weekday_offset", "count_days"], offset_upper_bound=30),
                            rw_calendar),
    "leg_counting": ("leg_counting", dict(min_animals=2, max_animals=4, min_instances=1, max_instances=10), rw_legs),
    "gsm_symbolic": ("gsm_symbolic", dict(difficulty=1.0), rw_gsm_symbolic),
    "countdown": ("countdown", dict(min_numbers=3, max_numbers=4, min_value=1, max_value=50, min_target=10,
                                    max_target=100), rw_countdown),
    "letter_counting": ("letter_counting", dict(min_words=3, max_words=8), rw_letters),
    "spell_backward": ("spell_backward", dict(min_word_len=3, max_word_len=8), rw_spell),
    "mini_sudoku": ("mini_sudoku", dict(min_empty=4, max_empty=8), rw_sudoku),
    "knights_knaves": ("knights_knaves", dict(n_people=2, depth_constraint=2, width_constraint=2), rw_knights),
}


def make_item(task, split, k, q, ans, meta, source=None, instr=INSTR):
    problem = q.strip() + "\n" + instr
    return {"id": f"{task}-{split}-{k:06d}", "task": task, "source": source or task, "split": split,
            "problem": problem, "answer": ans, "ref": ans, "prompt": prompt_text(problem), "meta": meta}


def build_rg(task, sizes, knobs=None, max_tries_mult=4, check=True):
    """sizes: {split: n}. Returns {split: [items]} disjoint by normalised question."""
    import reasoning_gym as rg
    from ultraviolet.tasks.score import score
    name, cfg, rw = RG[task]
    cfg = dict(cfg, **(knobs or {}))
    seen, out, stats = set(), {}, {}
    for split in SPLIT_ORDER:
        n = sizes.get(split, 0)
        items, tries, dup, rej = [], 0, 0, 0
        if n:
            cap = max_tries_mult * n + 100
            if callable(name):
                ds = None
            else:
                ds = rg.create_dataset(name, size=cap, seed=SEED_BASE[split], **cfg)
            for i in range(cap):
                if len(items) >= n:
                    break
                tries += 1
                e = ds[i] if ds is not None else name(random.Random(SEED_BASE[split] + i), **cfg)
                r = rw(e)
                if r is None:
                    rej += 1
                    continue
                q, ans, meta = r
                key = qkey(q)
                if key in seen:
                    dup += 1
                    continue
                seen.add(key)
                it = make_item(task, split, len(items), q, ans, meta)
                it["meta"]["rg_seed"] = SEED_BASE[split] + i
                if check and score(task, f"</think>\n\n\\boxed{{{ans}}}", it) != 1.0:
                    raise AssertionError(f"gold does not score 1: {task} {it}")
                items.append(it)
        out[split] = items
        stats[split] = {"n": len(items), "tries": tries, "dup": dup, "rejected": rej}
    return out, {"knobs": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()}, "splits": stats}


# --------------------------------------------------------------------------------------------- math datasets
def _h(s, mod=1000):
    return zlib.crc32(s.encode()) % mod


def _hf_split(rows, keyf, sizes, rl_frac):
    """rows of a train split -> rl (hash < rl_frac) / sft."""
    rl, sft = [], []
    for r in rows:
        (rl if _h(keyf(r)) < rl_frac * 1000 else sft).append(r)
    return {"rl": rl[:sizes.get("rl", len(rl))], "sft": sft[:sizes.get("sft", len(sft))]}


def build_gsm8k(dl, sizes, rl_frac=0.2):
    def conv(r):
        ans = r["answer"].split("####")[-1].strip().replace(",", "")
        return r["question"].strip(), ans
    train = [json.loads(l) for l in open(os.path.join(dl, "gsm8k", "train.jsonl"))]
    test = [json.loads(l) for l in open(os.path.join(dl, "gsm8k", "test.jsonl"))]
    parts = _hf_split(train, lambda r: r["question"], sizes, rl_frac)
    parts["eval"] = test[:sizes.get("eval", len(test))]
    return _finish("gsm8k", parts, conv, lambda r: {"solution": r["answer"]})


def build_math(dl, sizes, levels=("Level 1", "Level 2"), rl_frac=0.25):
    from ultraviolet.tasks.score import last_boxed

    def conv(r):
        a = last_boxed(r["solution"])
        return (r["problem"].strip(), a.strip()) if a else None
    tr = [r for r in map(json.loads, open(os.path.join(dl, "math", "train.jsonl"))) if r["level"] in levels]
    te = [r for r in map(json.loads, open(os.path.join(dl, "math", "test.jsonl"))) if r["level"] in levels]
    parts = _hf_split(tr, lambda r: r["problem"], sizes, rl_frac)
    random.Random(0).shuffle(te)
    parts["eval"] = te[:sizes.get("eval", len(te))]
    return _finish("math", parts, conv, lambda r: {"level": r["level"], "subject": r["subject"],
                                                   "solution": r["solution"]})


def _finish(task, parts, conv, metaf, instr=INSTR):
    seen, out, stats = set(), {}, {}
    for split in SPLIT_ORDER:
        items, dup, rej = [], 0, 0
        for r in parts.get(split, []):
            c = conv(r)
            if c is None:
                rej += 1
                continue
            q, ans = c[0], c[1]
            meta = metaf(r)
            if len(c) > 2:
                meta.update(c[2])
            key = qkey(q)
            if key in seen:
                dup += 1
                continue
            seen.add(key)
            items.append(make_item(task, split, len(items), q, ans, meta, instr=instr))
        out[split] = items
        stats[split] = {"n": len(items), "dup": dup, "rejected": rej}
    return out, {"splits": stats}


def write_items(out_dir, task, parts, stats):
    d = os.path.join(out_dir, task)
    os.makedirs(d, exist_ok=True)
    for sp, items in parts.items():
        with open(os.path.join(d, f"{sp}.jsonl.tmp"), "w") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        os.replace(os.path.join(d, f"{sp}.jsonl.tmp"), os.path.join(d, f"{sp}.jsonl"))
    json.dump(stats, open(os.path.join(d, "stats.json"), "w"), indent=1)
