"""wb_resid: residue vectors (docs/wb_envs.md). Given an integer N (written out, a single power, or a two-term
expression), find its remainders modulo 2-3 distinct moduli in a stated order. Reward = fraction of residues correct.

Families / tiers:
    explicit  tier 0: N with 5-6 digits, 2 moduli with digit rules {3, 4, 5, 8, 9, 10, 11}
              tier 1 (25%): N with 7-12 digits, 2-3 moduli <= 13 including one of {7, 12, 13} (chunking, CRT)
    power     tier 1 (75%): a^e, 2 <= a <= 12, 5 <= e <= 30, 2-3 moduli <= 13
    expr      tier 2: c1 a^e1 +/- c2 b^e2, a^e1 b^e2, or explicit products / sums (e <= 100), 3 moduli from the full set
Moduli sets never contain a pair where one divides the other (so no residue is read off another). At most one modulus
is "trivial" (N = 0 mod m, or for a power, base = 0 / 1 mod m). chance = mean(1 / m_i).
Stdlib + common.py only.
"""
from ultraviolet.tasks.score import TASKS
from workbench.envs.common import build_cli, parse_list, vec_credit

TASK = "wb_resid"
FORMAT = "Give the remainders in the order asked, separated by commas, e.g. \\boxed{3, 5, 7}."
ALL_MODS = (3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 16, 25)
DIGIT_MODS = (3, 4, 5, 8, 9, 10, 11)
SMALL_MODS = tuple(m for m in ALL_MODS if m <= 13)


# ----------------------------------------------------------------------------------------------- rendering
def _mlist(mods):
    s = [f"${m}$" for m in mods]
    return s[0] if len(s) == 1 else ", ".join(s[:-1]) + " and " + s[-1]


def _phrase(rng, nexpr, mods):
    """2-4 templates; the order of the moduli is always explicit."""
    k = len(mods)
    t = rng.randrange(4)
    if t == 0:
        return f"Let $N = {nexpr}$. Find the remainders when $N$ is divided by {_mlist(mods)}, in that order."
    if t == 1:
        return (f"What are the remainders of ${nexpr}$ upon division by {_mlist(mods)}, respectively?")
    if t == 2:
        parts = [f"$N \\bmod {m}$" for m in mods]
        lst = parts[0] + " and " + parts[1] if k == 2 else ", ".join(parts[:-1]) + " and " + parts[-1]
        return f"Let $N = {nexpr}$. Compute {lst}, in that order."
    rs = ", ".join(f"r_{i + 1}" for i in range(k))
    conds = [f"$N \\equiv r_{i + 1} \\pmod{{{m}}}$" for i, m in enumerate(mods)]
    conds = ", ".join(conds[:-1]) + " and " + conds[-1]
    return (f"Let $N = {nexpr}$. Find ${rs}$ such that {conds}, where each $r_i$ is the least non-negative "
            f"remainder. List ${rs}$ in that order.")


def _pw(a, e):
    return f"{a}^{{{e}}}"


def _term(c, a, e):
    return (f"{c} \\cdot " if c > 1 else "") + _pw(a, e)


# ----------------------------------------------------------------------------------------------- moduli
def _antichain(mods):
    return all(a % b and b % a for i, a in enumerate(mods) for b in mods[i + 1:])


def _pick_mods(rng, pool, k, need=None):
    for _ in range(50):
        mods = rng.sample(pool, k)
        if _antichain(mods) and (need is None or any(m in need for m in mods)):
            return mods
    return None


# ----------------------------------------------------------------------------------------------- families
def _explicit(rng, tier):
    if tier == 0:
        nd, k, mods = rng.choice((5, 6, 6, 6)), 2, _pick_mods(rng, list(DIGIT_MODS), 2)
    else:
        nd, k = rng.randint(7, 12), rng.choice((2, 3))
        mods = _pick_mods(rng, list(SMALL_MODS), k, need=(7, 12, 13))
    if mods is None:
        return None
    n = rng.randrange(10 ** (nd - 1), 10 ** nd)
    return n, str(n), mods, sum(1 for m in mods if n % m == 0)


def _power(rng, tier):
    a, e, k = rng.randint(2, 12), rng.randint(5, 30), rng.choice((2, 3))
    mods = _pick_mods(rng, list(SMALL_MODS), k)
    if mods is None:
        return None
    n = a ** e
    triv = sum(1 for m in mods if a % m in (0, 1) or n % m == 0)
    return n, _pw(a, e), mods, triv


def _expr(rng, tier):
    form = rng.choice(("sum", "sum", "diff", "prod", "prod", "num"))
    a, b = rng.sample(range(2, 13), 2)
    e1, e2 = rng.randint(10, 100), rng.randint(5, 100)
    if form == "sum":
        c1, c2 = rng.choice((1, 1, rng.randint(2, 9))), rng.choice((1, rng.randint(2, 9)))
        n, s = c1 * a ** e1 + c2 * b ** e2, f"{_term(c1, a, e1)} + {_term(c2, b, e2)}"
    elif form == "diff":
        c2 = rng.choice((1, 1, rng.randint(2, 9)))
        n, s = a ** e1 - c2 * b ** e2, f"{_pw(a, e1)} - {_term(c2, b, e2)}"
        if n <= 0:
            return None
    elif form == "prod":
        e1, e2 = rng.randint(5, 60), rng.randint(5, 60)
        n, s = a ** e1 * b ** e2, f"{_pw(a, e1)} \\cdot {_pw(b, e2)}"
    else:
        if rng.random() < 0.5:
            x, y = rng.randrange(10 ** 5, 10 ** 6), rng.randrange(100, 10 ** rng.randint(3, 4))
            n, s = x * y, f"{x} \\cdot {y}"
        else:
            x = rng.randrange(10 ** 5, 10 ** 7)
            n, s = x + a ** e1, f"{x} + {_pw(a, e1)}"
    mods = _pick_mods(rng, list(ALL_MODS), 3)
    if mods is None:
        return None
    triv = sum(1 for m in mods if n % m == 0)
    return n, s, mods, triv


def gen(rng, tier):
    """Retries degenerate draws internally (same rng stream) so build_items' tier cycling stays balanced."""
    for _ in range(30):
        r = _gen_once(rng, tier)
        if r is not None:
            return r
    return None


def _gen_once(rng, tier):
    if tier == 0:
        fam, r = "explicit", _explicit(rng, 0)
    elif tier == 1:
        fam = "explicit" if rng.random() < 0.25 else "power"
        r = _explicit(rng, 1) if fam == "explicit" else _power(rng, 1)
    else:
        fam, r = "expr", _expr(rng, 2)
    if r is None:
        return None
    n, nexpr, mods, triv = r
    gold = [n % m for m in mods]
    if triv > 1 or all(g == 0 for g in gold) or (len(gold) == 3 and len(set(gold)) == 1):
        return None
    body = _phrase(rng, nexpr, mods)
    answer = ", ".join(map(str, gold))
    meta = {"gold": gold, "moduli": list(mods), "expr": nexpr, "n_digits": len(str(n)),
            "chance": sum(1 / m for m in mods) / len(mods)}
    return body, answer, meta, fam


# ----------------------------------------------------------------------------------------------- scoring
def sc_resid(pred, item):
    gold = item.get("meta", {}).get("gold")
    if gold is None:
        gold = parse_list(item["answer"], "int")
    return vec_credit(parse_list(pred, "int"), gold)


TASKS[TASK] = {"kind": "list", "scorer": sc_resid, "reward": "graded"}

if __name__ == "__main__":
    build_cli(TASK, gen, FORMAT)
