"""Shared core of the param-object tasks (docs/wb_paramobj.md): answer parsing, the Family base class, verification
and scoring. An item asks for one valid answer out of a finite, exactly enumerated set. Stdlib only.

  from workbench.envs.paramobj import core
  core.score(row, completion_text)  # {"correct", "key", "wellformed", "correct_strict", "pred"}
  core.verify(row, answer_str) / core.canon(row, answer_str) / core.enumerate_valid(row)

Answer extraction follows docs/tasks.md (text after the last </think>, last balanced \\boxed{}; no </think> -> wrong).
"""
import json
import math
import re

from ultraviolet.tasks.score import TASKS as _TASK_REGISTRY
from ultraviolet.tasks.score import THINK_CLOSE, answer_tail, last_boxed, strip_latex

BOX = "Put the final answer in \\boxed{}."
TASK_PREFIX = "wb_"             # task = TASK_PREFIX + family (family names start with po_)
BUCKETS = (("2-3", 2, 3), ("4-8", 4, 8), ("9-20", 9, 20))
MAX_VALID_LISTED = 400          # valid_answers stored in the row only up to this many
FAMILIES = {}                   # filled by paramobj.py


def bucket_of(n):
    for name, lo, hi in BUCKETS:
        if lo <= n <= hi:
            return name
    return None


# ----------------------------------------------------------------------------------------------- number helpers
def is_prime(n):
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def digit_sum(n):
    return sum(int(c) for c in str(abs(n)))


def smallest_factor(n):
    for i in range(2, math.isqrt(n) + 1):
        if n % i == 0:
            return i
    return n


# ----------------------------------------------------------------------------------------------- answer parsing
_LABEL = re.compile(r"([A-Za-z])\s*=\s*([-+]?\d+)")
_INT = re.compile(r"[-+]?\d+")


def _clean(s):
    s = strip_latex(s).replace("−", "-")
    s = s.replace("\\{", " ").replace("\\}", " ")
    for ch in "()[]{}":
        s = s.replace(ch, " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s.rstrip(".").strip()


def parse_ints(s, k, names=None):
    """Exactly k comma-separated integers ("3, 4", "(3, 4)", "x = 3, y = 4", "3 and 4"). With all parts labelled by
    `names` the values are reordered by name. Returns a tuple of ints or None."""
    if s is None:
        return None
    s = _clean(s)
    if k == 1 and re.fullmatch(r"\d{1,3}(?:,\d{3})+", s):    # thousands separator for single integers
        s = s.replace(",", "")
    s = re.sub(r"\s+and\s+", ",", s)
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != k:
        return None
    vals, labels = [], []
    for p in parts:
        m = _LABEL.fullmatch(p)
        if m:
            labels.append(m.group(1))
            vals.append(int(m.group(2)))
        elif _INT.fullmatch(p):
            labels.append(None)
            vals.append(int(p))
        else:
            return None
    if names and all(labels):
        if sorted(labels) != sorted(names):
            return None
        vals = [vals[labels.index(n)] for n in names]
    elif any(labels) and names and any(lb not in names for lb in labels if lb):
        return None
    return tuple(vals)


def parse_frac(s):
    """(p, q) as written from p/q or \\frac{p}{q} (sign moved to p); None for decimals or anything else."""
    if s is None:
        return None
    s = strip_latex(s).replace("−", "-").replace(" ", "").rstrip(".")
    m = (re.fullmatch(r"([-+]?)\\frac\{([-+]?\d+)\}\{([-+]?\d+)\}", s)
         or re.fullmatch(r"([-+]?)\(?([-+]?\d+)\)?/\(?([-+]?\d+)\)?", s))
    if not m:
        return None
    p, q = int(m.group(2)), int(m.group(3))
    if m.group(1) == "-":
        p = -p
    if q == 0:
        return None
    if q < 0:
        p, q = -p, -q
    return p, q


# ----------------------------------------------------------------------------------------------- families
class Family:
    """Subclasses define: parse(params, s) -> candidate | None; valid(params, cand) -> bool (bounds checked before any
    work that depends on the answer's size); key(params, cand) -> hashable canon; all_keys(params) -> set; fmt(params,
    key) -> answer string."""
    name = None

    def param_key(self, p):
        return json.dumps([self.name, p], sort_keys=True)


# ----------------------------------------------------------------------------------------------- verify / canon
def _fam(row):
    meta = row.get("meta") or {}
    return FAMILIES[row.get("family") or meta["family"]], meta["params"]


def verify(row, answer_str):
    """True iff answer_str (the extracted box content) is a valid answer for the row's family / params."""
    try:
        fam, p = _fam(row)
        c = None if answer_str is None else fam.parse(p, answer_str)
        return bool(c is not None and fam.valid(p, c))
    except (ValueError, TypeError, OverflowError, KeyError):
        return False


def canon(row, answer_str):
    """Canonical key of a parseable answer (valid or not; equivalent answers share one key), None if unparseable."""
    try:
        fam, p = _fam(row)
        c = None if answer_str is None else fam.parse(p, answer_str)
        return None if c is None else fam.key(p, c)
    except (ValueError, TypeError, OverflowError, KeyError):
        return None


def enumerate_valid(row):
    """Exact set of canon keys of all valid answers (every family's answer space is finite)."""
    fam, p = _fam(row)
    return set(fam.all_keys(p))


def extract(completion_text, require_close=True):
    tail = answer_tail(completion_text, require_close)
    return None if tail is None else last_boxed(tail)


def score(row, completion_text, require_close=True, ends_eos=None):
    """docs/tasks.md convention. correct: last \\boxed{} after the last </think> verifies (no </think> -> False).
    key: canon key when correct, else None. wellformed: exactly one </think> (and ends_eos, when given: EOS is a
    token, so the caller passes whether generation stopped on it). correct_strict = correct and wellformed."""
    pred = extract(completion_text, require_close)
    ok = pred is not None and verify(row, pred)
    wf = completion_text is not None and completion_text.count(THINK_CLOSE) == 1 and (ends_eos is None or bool(ends_eos))
    return {"correct": ok, "key": canon(row, pred) if ok else None, "wellformed": wf,
            "correct_strict": ok and wf, "pred": pred}


def register_task_suite():
    """Add wb_<family> scorers to ultraviolet.tasks.score.TASKS (additive; existing names untouched)."""
    for name in FAMILIES:
        _TASK_REGISTRY.setdefault(f"{TASK_PREFIX}{name}", {"kind": "math", "scorer": lambda pred, item: verify(item, pred)})
