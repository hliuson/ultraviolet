"""wb_grid: line-sum grid completion (docs/wb_envs.md). A partially filled grid of positive integers; the model fills
the blanks and answers with the MISSING numbers only, in reading order (left to right, top to bottom). Reward =
vec_credit (prefix-aligned): matches at aligned positions / max(len(pred), len(gold_blank)), so a partially right or
truncated answer earns partial credit and extra values dilute it. The body states how many numbers are missing.

Families / tiers:
    magic3   3x3 magic square in the general form (centre c, parameters a, b):
                 [c+a, c-a-b, c+b] / [c-a+b, c, c+a-b] / [c-b, c+a+b, c-a]
             all 9 entries positive and distinct, magic sum S = 3c stated, 3 cells given (never all in one line) whose
             coefficient matrix over (c, a, b) has rank 3.
             tier 0: centre given + 2 others, c <= 12; tier 1: centre hidden, c <= 20; tier 2: centre hidden, c <= 40,
             at most one given cell per row.
    sumgrid  R x C grid of positive integers with all row and column sums stated. Unique completion iff the
             (row + column) incidence matrix over the blanks has full column rank (exact Fraction elimination), i.e.
             the blanks form a forest in the row/column bipartite graph, so at most R + C - 1 blanks.
             tier 0: 3x3, 3-4 blanks, entries <= 20; tier 1: 3x4, 5-6 blanks, entries <= 50, some line starts with
             >= 2 blanks and the "fill lines with one blank" cascade needs >= 2 rounds; tier 2: 4x4, 7 blanks (the
             maximum; a spanning forest), entries <= 50, cascade needs >= 3 rounds.
chance: Monte Carlo (n=200) over blank-value lists uniform in [1, max] (same length as gold_blank). Stdlib + common.py only.
"""
import random
import re
from fractions import Fraction

from ultraviolet.tasks.score import TASKS
from workbench.envs.common import build_cli, clean, mc_chance, parse_int, parse_list, vec_credit

TASK = "wb_grid"
FORMAT = ("List the missing numbers in reading order (left to right, top to bottom), separated by commas, "
          "e.g. \\boxed{4, 9, 2}.")

# magic3 cell coefficients over (c, a, b), row-major
MAGIC_COEF = ((1, 1, 0), (1, -1, -1), (1, 0, 1),
              (1, -1, 1), (1, 0, 0), (1, 1, -1),
              (1, 0, -1), (1, 1, 1), (1, -1, 0))
MAGIC_LINES = ((0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6))


# ----------------------------------------------------------------------------------------------- linear algebra
def rank(rows):
    """Rank of an integer / rational matrix (list of rows) by exact Fraction Gaussian elimination."""
    m = [[Fraction(x) for x in r] for r in rows]
    if not m:
        return 0
    nc, r = len(m[0]), 0
    for c in range(nc):
        p = next((i for i in range(r, len(m)) if m[i][c] != 0), None)
        if p is None:
            continue
        m[r], m[p] = m[p], m[r]
        for i in range(len(m)):
            if i != r and m[i][c] != 0:
                f = m[i][c] / m[r][c]
                m[i] = [x - f * y for x, y in zip(m[i], m[r])]
        r += 1
        if r == len(m):
            break
    return r


def line_matrix(R, C, blank):
    """Equations (R row sums, then C column sums) restricted to the blank cells' columns."""
    rows = [[1 if b // C == i else 0 for b in blank] for i in range(R)]
    rows += [[1 if b % C == j else 0 for b in blank] for j in range(C)]
    return rows


def cascade_rounds(R, C, blank):
    """Rounds of 'fill every line with exactly one remaining blank'. Returns (rounds, all_filled)."""
    left, rounds = set(blank), 0
    while left:
        lines = [[b for b in left if b // C == i] for i in range(R)] + [[b for b in left if b % C == j] for j in range(C)]
        fill = {ln[0] for ln in lines if len(ln) == 1}
        if not fill:
            return rounds, False
        left -= fill
        rounds += 1
    return rounds, True


# ----------------------------------------------------------------------------------------------- rendering
def render(given, R, C):
    w = max(len(str(v)) for v in given if v is not None)
    return "\n".join(" ".join(("_" if v is None else str(v)).rjust(w) for v in given[i * C:(i + 1) * C])
                     for i in range(R))


def _n_missing(n):
    return f"There are {n} missing numbers."


def _magic_body(rng, grid_txt, S, nb):
    if rng.random() < 0.5:
        return (f"The grid below is to be completed to a $3 \\times 3$ magic square of distinct positive integers: "
                f"every row, every column and both diagonals have the same sum. Magic sum {S}. "
                f"Blank cells are marked _.\n\n{grid_txt}\n\n{_n_missing(nb)} Find them.")
    return (f"Complete this magic square. Its entries are distinct positive integers, and each of its rows, columns "
            f"and two main diagonals adds up to the same total. Magic sum {S}. Each _ is a missing entry.\n\n{grid_txt}"
            f"\n\n{_n_missing(nb)}")


def _sum_body(rng, grid_txt, R, C, rs, cs, nb):
    sums = f"Row sums: {', '.join(map(str, rs))}. Column sums: {', '.join(map(str, cs))}."
    if rng.random() < 0.5:
        return (f"A ${R} \\times {C}$ grid of positive integers has some cells blank (marked _). The sums of the rows "
                f"(top to bottom) and of the columns (left to right) are given.\n\n{grid_txt}\n\n{sums}\n\n"
                f"{_n_missing(nb)} Find them.")
    return (f"Each _ in the grid below hides a positive integer. {sums} Rows are listed top to bottom and columns "
            f"left to right. {_n_missing(nb)}\n\n{grid_txt}")


# ----------------------------------------------------------------------------------------------- families
def magic_grid(c, a, b):
    return [c * k0 + a * k1 + b * k2 for k0, k1, k2 in MAGIC_COEF]


def _magic3(rng, tier):
    cmax = (12, 20, 40)[tier]
    c = rng.randint(5 if tier == 0 else 6, cmax)
    for _ in range(100):
        a, b = rng.randint(-(c - 1), c - 1), rng.randint(-(c - 1), c - 1)
        if a == 0 or b == 0 or a in (b, -b, 2 * b, -2 * b) or 2 * a in (b, -b):
            continue
        gold = magic_grid(c, a, b)
        if min(gold) >= 1 and len(set(gold)) == 9:
            break
    else:
        return None
    for _ in range(50):
        if tier == 0:
            cells = [4] + rng.sample([i for i in range(9) if i != 4], 2)
        elif tier == 1:
            cells = rng.sample([i for i in range(9) if i != 4], 3)
        else:
            cells = [3 * r + rng.choice([j for j in range(3) if 3 * r + j != 4]) for r in range(3)]
        if any(set(cells) == set(ln) for ln in MAGIC_LINES):
            continue
        if rank([MAGIC_COEF[i] for i in cells]) == 3:
            break
    else:
        return None
    cells = sorted(cells)
    blank = [i for i in range(9) if i not in cells]
    return gold, blank, 3, 3, 2 * c - 1, {"S": 3 * c, "c": c, "a": a, "b": b}


def _sumgrid(rng, tier):
    R, C = ((3, 3), (3, 4), (4, 4))[tier]
    vmax = 20 if tier == 0 else 50
    nb = rng.choice(((3, 4), (5, 6), (7,))[tier])
    gold = [rng.randint(1, vmax) for _ in range(R * C)]
    for _ in range(200):
        blank = sorted(rng.sample(range(R * C), nb))
        if rank(line_matrix(R, C, blank)) != nb:
            continue
        rounds, ok = cascade_rounds(R, C, blank)
        if not ok:
            continue
        if tier == 1:
            two = any(sum(1 for b in blank if b // C == i) >= 2 for i in range(R)) or \
                any(sum(1 for b in blank if b % C == j) >= 2 for j in range(C))
            if rounds < 2 or not two:
                continue
        if tier == 2 and rounds < 3:
            continue
        break
    else:
        return None
    rs = [sum(gold[i * C:(i + 1) * C]) for i in range(R)]
    cs = [sum(gold[j::C]) for j in range(C)]
    return gold, blank, R, C, vmax, {"row_sums": rs, "col_sums": cs, "rounds": rounds}


def gen(rng, tier):
    fam = "magic3" if rng.random() < 0.5 else "sumgrid"
    r = (_magic3 if fam == "magic3" else _sumgrid)(rng, tier)
    if r is None:
        return None
    gold, blank, R, C, vmax, extra = r
    given = [None if i in blank else v for i, v in enumerate(gold)]
    txt = render(given, R, C)
    if fam == "magic3":
        body = _magic_body(rng, txt, extra["S"], len(blank))
    else:
        body = _sum_body(rng, txt, R, C, extra["row_sums"], extra["col_sums"], len(blank))
    gold_blank = [gold[i] for i in blank]  # blank is sorted, so this is reading order
    crng = random.Random(rng.getrandbits(32))

    def sample(cr):
        return [cr.randint(1, vmax) for _ in gold_blank]

    chance = mc_chance(crng, sample, lambda p: vec_credit(p, gold_blank), n=200)
    meta = {"gold": gold, "blank": blank, "gold_blank": gold_blank, "shape": [R, C], "given": given, "max": vmax,
            "chance": chance, **extra}
    return body, ", ".join(map(str, gold_blank)), meta, fam


# ----------------------------------------------------------------------------------------------- scoring
def parse_grid(pred):
    """Boxed content -> flat list of ints. Accepts commas / semicolons / newlines / spaces / LaTeX row breaks (\\\\)
    and array environments; None if any token is not an integer."""
    out = parse_list(pred, "int")
    if out is not None or pred is None:
        return out
    s = re.sub(r"\\begin\{[a-z*]*\}(?:\{[lcr| ]*\})?|\\end\{[a-z*]*\}", " ", pred).replace("\\\\", ",")
    toks = [t for t in re.split(r"[,;\s]+", clean(s)) if t]
    vals = [parse_int(t) for t in toks]
    return None if not vals or any(v is None for v in vals) else vals


def sc_grid(pred, item):
    meta = item.get("meta", {})
    gold = meta.get("gold_blank")
    if gold is None:
        gold = parse_list(item["answer"], "int")
    return vec_credit(parse_grid(pred), gold or [])


TASKS[TASK] = {"kind": "grid", "scorer": sc_grid, "reward": "graded"}

if __name__ == "__main__":
    build_cli(TASK, gen, FORMAT)
