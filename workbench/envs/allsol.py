"""wb_allsol: find all solutions (docs/wb_envs.md). Set-valued answers scored by F1 against the gold set.

Families: divisors, quadmod, introots, diophant, primes. Tier 0 / 1 / 2 grow the domain and the gold size.
`meta["chance"]` = Monte Carlo mean F1 of a random same-size subset of the family's natural domain (for an empty
gold set the random answer size is drawn uniformly from 0..6, so chance ~ 1/7 instead of a trivial 1.0).
"""
import random
from fractions import Fraction

from ultraviolet.tasks.score import TASKS
from workbench.envs.common import build_cli, f1_set, mc_chance, parse_list

TASK = "wb_allsol"
def FORMAT(family, meta):  # noqa: N802  format sentence with an example of the answer kind (docs/wb_envs.md)
    if meta.get("kind") == "pair":
        return "List all of them separated by commas, e.g. \\boxed{(1, 2), (3, 4)} (write none if there are none)."
    return "List all of them separated by commas, e.g. \\boxed{1, 2, 4} (write none if there are none)."
FAMILIES = ("divisors", "quadmod", "introots", "diophant", "primes")
MAX_TRIES = 200


# ----------------------------------------------------------------------------------------------- helpers
def divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def is_prime(n):
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def poly_from_roots(roots):
    """Monic coefficients (highest degree first) of prod (x - r)."""
    c = [1]
    for r in roots:
        c = [a - r * b for a, b in zip(c + [0], [0] + c)]
    return c


def fmt_poly(coefs):
    """[1, -6, 11, -6] -> 'x^3 - 6x^2 + 11x - 6'."""
    deg = len(coefs) - 1
    out = []
    for i, a in enumerate(coefs):
        e = deg - i
        if a == 0:
            continue
        mag = abs(a)
        var = "" if e == 0 else ("x" if e == 1 else f"x^{e}")
        term = (str(mag) if (mag != 1 or e == 0) else "") + var
        if not out:
            out.append(("-" if a < 0 else "") + term)
        else:
            out.append(("- " if a < 0 else "+ ") + term)
    return " ".join(out) if out else "0"


def fmt_items(gold, kind):
    if not gold:
        return "none"
    if kind == "pair":
        return ", ".join(f"({x}, {y})" for x, y in gold)
    return ", ".join(str(x) for x in gold)


def _sample_k(rng, k, size, draw):
    """k distinct elements of a domain of `size` elements, drawn by rejection with draw(rng)."""
    k = min(k, size)
    s = set()
    while len(s) < k:
        s.add(draw(rng))
    return list(s)


def chance_of(rng, gold, size, draw):
    crng = random.Random(rng.getrandbits(32))
    g = list(gold)
    if g:
        return mc_chance(crng, lambda r: _sample_k(r, len(g), size, draw), lambda p: f1_set(p, g), n=200)
    return mc_chance(crng, lambda r: _sample_k(r, r.randint(0, 6), size, draw), lambda p: f1_set(p, g), n=200)


# ----------------------------------------------------------------------------------------------- families
def g_divisors(rng, tier):
    lo, hi, dlo, dhi = [(24, 100, 6, 8), (60, 250, 8, 12), (120, 400, 10, 16)][tier]
    for _ in range(MAX_TRIES):
        n = rng.randint(lo, hi)
        d = divisors(n)
        if dlo <= len(d) <= dhi:
            break
    else:
        return None
    body = rng.choice([
        f"Find all positive divisors of ${n}$.",
        f"List every positive integer $d$ such that $d$ divides ${n}$.",
        f"Determine all positive integers $k$ for which $\\frac{{{n}}}{{k}}$ is an integer.",
    ])
    chance = chance_of(rng, d, n, lambda r: r.randint(1, n))
    return body, d, {"n": n}, "int", chance


def g_quadmod(rng, tier):
    mlo, mhi, smax = [(5, 12, 4), (10, 20, 5), (16, 30, 6)][tier]
    want_empty = rng.random() < 0.08
    # plain random (b, c) almost never gives more than 4 roots; ask for a large root set part of the time
    smin = 0 if want_empty else (1 if rng.random() >= (0.0, 0.25, 0.4)[tier] else (1, 3, 4)[tier])
    for _ in range(5 * MAX_TRIES):
        m = rng.randint(mlo, mhi)
        if tier == 0:
            b, c = rng.randrange(m), rng.randrange(m)
        else:
            b, c = rng.randint(-(m - 1), m - 1), rng.randint(-(m - 1), m - 1)
        sols = [x for x in range(m) if (x * x + b * x + c) % m == 0]
        if (not sols) == want_empty and smin <= len(sols) <= smax:
            break
    else:
        return None
    p = fmt_poly([1, b, c])
    body = rng.choice([
        f"Find all integers $x$ with $0 \\le x < {m}$ such that ${p} \\equiv 0 \\pmod{{{m}}}$.",
        f"Determine all residues $x \\in \\{{0, 1, \\dots, {m - 1}\\}}$ satisfying ${p} \\equiv 0 \\pmod{{{m}}}$.",
        f"For which integers $x$ in the range $0 \\le x \\le {m - 1}$ is ${p}$ divisible by ${m}$?",
    ])
    chance = chance_of(rng, sols, m, lambda r: r.randrange(m))
    return body, sols, {"m": m, "b": b, "c": c}, "int", chance


def g_introots(rng, tier):
    if tier == 0:
        roots = rng.sample(range(-5, 6), 3)
    elif tier == 1:
        if rng.random() < 0.5:
            roots = [rng.randint(-9, 9) for _ in range(3)]
        else:
            roots = [rng.randint(-5, 5) for _ in range(4)]
    else:
        roots = [rng.randint(-9, 9) for _ in range(4)]
    coefs = poly_from_roots(roots)
    gold = sorted(set(roots))
    p = fmt_poly(coefs)
    body = rng.choice([
        f"Find all integer roots of ${p} = 0$.",
        f"Determine every integer $x$ that satisfies ${p} = 0$.",
        f"Let $p(x) = {p}$. Find all integers $r$ such that $p(r) = 0$, listing each root once.",
    ])
    chance = chance_of(rng, gold, 19, lambda r: r.randint(-9, 9))
    return body, gold, {"coefs": coefs}, "int", chance


def g_diophant(rng, tier):
    amax, klo, khi = [(9, 1, 2), (12, 2, 3), (15, 3, 5)][tier]
    for _ in range(MAX_TRIES):
        a, b = rng.randint(2, amax), rng.randint(2, amax)
        if gcd(a, b) != 1:
            continue
        k = rng.randint(klo, khi)
        c = rng.randint(max(1, a * b * (k - 1)), a * b * k + a * b // 2)
        sols = [(x, (c - a * x) // b) for x in range(c // a + 1) if (c - a * x) % b == 0]
        if klo <= len(sols) <= khi:
            break
    else:
        return None
    body = rng.choice([
        f"Find all pairs of non-negative integers $(x, y)$ such that ${a}x + {b}y = {c}$.",
        f"Determine all solutions of ${a}x + {b}y = {c}$ in non-negative integers $x, y$.",
        f"How can ${c}$ be written as ${a}x + {b}y$ with $x, y$ non-negative integers? Find all such pairs $(x, y)$.",
    ]) + " Write each solution as an ordered pair $(x, y)$, in increasing order of $x$."
    nx, ny = c // a + 1, c // b + 1
    chance = chance_of(rng, sols, nx * ny, lambda r: (r.randrange(nx), r.randrange(ny)))
    return body, sols, {"a": a, "b": b, "c": c}, "pair", chance


def g_primes(rng, tier):
    slo, shi, alo, gmax = [(20, 30, 100, 4), (30, 45, 2, 10), (45, 60, 2, 16)][tier]
    for _ in range(MAX_TRIES):
        span = rng.randint(slo, shi)
        a = rng.randint(alo, 400 - span)
        b = a + span
        gold = [p for p in range(a, b + 1) if is_prime(p)]
        if 1 <= len(gold) <= gmax:
            break
    else:
        return None
    body = rng.choice([
        f"Find all prime numbers $p$ with ${a} \\le p \\le {b}$.",
        f"List all primes between ${a}$ and ${b}$, inclusive.",
        f"Determine every prime in the interval $[{a}, {b}]$.",
    ])
    chance = chance_of(rng, gold, b - a + 1, lambda r: r.randint(a, b))
    return body, gold, {"a": a, "b": b}, "int", chance


GENS = {"divisors": g_divisors, "quadmod": g_quadmod, "introots": g_introots, "diophant": g_diophant,
        "primes": g_primes}


def gen(rng, tier):
    family = rng.choice(FAMILIES)
    r = GENS[family](rng, tier)
    if r is None:
        return None
    body, gold, params, kind, chance = r
    meta = {"gold": [list(g) for g in gold] if kind == "pair" else list(gold), "kind": kind, "family": family,
            "params": params, "chance": round(chance, 4)}
    return body, fmt_items(gold, kind), meta, family


# ----------------------------------------------------------------------------------------------- scorer
def sc_allsol(pred, item):
    m = item["meta"]
    kind = m["kind"]
    p = parse_list(pred, "pair" if kind == "pair" else "int")
    if p is None:
        return 0.0
    if kind == "pair":
        gold = [(Fraction(x), Fraction(y)) for x, y in m["gold"]]
    else:
        gold = [int(x) for x in m["gold"]]
    return f1_set(p, gold)


TASKS[TASK] = {"kind": "set", "scorer": sc_allsol, "reward": "graded"}

if __name__ == "__main__":
    build_cli(TASK, gen, FORMAT)
