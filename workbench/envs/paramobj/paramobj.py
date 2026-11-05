"""Parameterised-object construction tasks (param-object v0, families po_*). See docs/wb_paramobj.md.

An item asks for a short tuple of integers that parameterises a mathematical object (polynomial coefficients, matrix
entries, a permutation, a lattice point, a pair of divisors, ...) such that the object has stated properties. Every
item states (or implies) a finite answer box, so the valid set is finite and enumerated exactly. Constraints are
chosen so that answers follow from a derivation (linear algebra over the parameters, factor pairs, divisors,
multiplicativity) rather than a scan, and the builder keeps items whose valid count is in [NV_MIN, NV_MAX] and whose
blind-guess rate (valid keys / answer-space size) is small.

Each family: sample(rng, tier) -> params (JSON), statement(params) -> (text, answer format), names(params) (answer
letters, or None for positional tuples), box(params) (inclusive per-coordinate bounds that every valid answer obeys),
ok(params, t) (exact check of the hard condition), canon(params, t) (key; equivalent answers -> one key), solve(params)
-> iterable of valid tuples (exact and complete within the box), space(params) (size of the canonical answer space,
for guess_rate). valid = in box and ok; all_keys = {canon(t) for t in solve}.

Families register in core.FAMILIES (task names wb_po_<name>) and score through core.score / ultraviolet.tasks.score.
Stdlib only.
"""
import functools
import itertools
import json
import math
import random
from fractions import Fraction as F

from workbench.envs.paramobj import core

NV_MIN, NV_MAX = 2, 64          # builder keeps items with this many valid keys
GUESS_MAX = 0.02                # ... and at most this blind-guess rate
ABS_MAX = 10 ** 12              # answers with a larger |component| are rejected before any work
TIERS = (0, 1, 2, 3)
LET = "abcdefghij"
BOX = core.BOX


# ----------------------------------------------------------------------------------------------- helpers
def fstr(v):
    """Exact rational as text: 3, -2, \\frac{7}{6}, -\\frac{7}{6}."""
    v = F(v)
    if v.denominator == 1:
        return str(v.numerator)
    s = "-" if v < 0 else ""
    return f"{s}\\frac{{{abs(v.numerator)}}}{{{v.denominator}}}"


def fenc(v):
    return str(F(v))


def fdec(s):
    return F(s)


def factorize(n):
    n = abs(n); out = {}
    p = 2
    while p * p <= n:
        while n % p == 0:
            out[p] = out.get(p, 0) + 1
            n //= p
        p += 1 if p == 2 else 2
    if n > 1:
        out[n] = out.get(n, 0) + 1
    return out


def divisors(n):
    ds = [1]
    for p, e in factorize(n).items():
        ds = [d * p ** i for d in ds for i in range(e + 1)]
    return sorted(ds)


def sdivisors(n):
    d = divisors(n)
    return d + [-x for x in d]


def names_txt(ns):
    ns = list(ns)
    return ns[0] if len(ns) == 1 else ", ".join(ns[:-1]) + " and " + ns[-1]


def comma(ns):
    return ", ".join(ns)


def pt(p):
    return "(" + ", ".join(str(x) for x in p) + ")"


def lattice(A, b, box):
    """All integer points x in box (list of inclusive (lo, hi)) with A x = b. RREF over Fractions; free variables are
    enumerated over their box ranges, pivots solved and checked for integrality and bounds."""
    k = len(box)
    M = [[F(v) for v in row] + [F(r)] for row, r in zip(A, b)]
    piv, r = [], 0
    for c in range(k):
        pr = next((i for i in range(r, len(M)) if M[i][c] != 0), None)
        if pr is None:
            continue
        M[r], M[pr] = M[pr], M[r]
        inv = M[r][c]
        M[r] = [v / inv for v in M[r]]
        for i in range(len(M)):
            if i != r and M[i][c] != 0:
                f = M[i][c]
                M[i] = [x - f * y for x, y in zip(M[i], M[r])]
        piv.append(c)
        r += 1
    if any(all(v == 0 for v in M[i][:k]) and M[i][k] != 0 for i in range(r, len(M))):
        return []
    free = [c for c in range(k) if c not in piv]
    out = []
    for fv in itertools.product(*[range(box[c][0], box[c][1] + 1) for c in free]):
        x = [None] * k
        for c, v in zip(free, fv):
            x[c] = v
        good = True
        for i, c in enumerate(piv):
            v = M[i][k] - sum(M[i][f] * x[f] for f in free)
            if v.denominator != 1 or not box[c][0] <= v <= box[c][1]:
                good = False
                break
            x[c] = int(v)
        if good:
            out.append(tuple(x))
    return out


def bruteforce(box, pred):
    return [t for t in itertools.product(*[range(lo, hi + 1) for lo, hi in box]) if pred(t)]


def multisets(lo, hi, n):
    return math.comb(hi - lo + n, n)


def nondecr_sum(n, lo, hi, S):
    """Nondecreasing n-tuples with entries in [lo, hi] and sum S."""
    if n == 0:
        if S == 0:
            yield ()
        return
    for x in range(lo, hi + 1):
        rest = S - x
        if rest < x * (n - 1) or rest > hi * (n - 1):
            continue
        for t in nondecr_sum(n - 1, x, hi, rest):
            yield (x,) + t


def distinct_sum(k, lo, S):
    """Increasing k-tuples of integers >= lo with sum S."""
    if k == 0:
        if S == 0:
            yield ()
        return
    for x in range(lo, S + 1):
        if x * k + k * (k - 1) // 2 > S:
            break
        for t in distinct_sum(k - 1, x + 1, S - x):
            yield (x,) + t


# ----------------------------------------------------------------------------------------------- base
class PO(core.Family):
    name = None
    domain = None
    vars = None
    unordered = False           # key = sorted tuple (ok must then be symmetric)

    def names(self, p):
        return self.vars

    def k(self, p):
        return len(self.box(p))

    def parse(self, p, s):
        t = core.parse_ints(s, self.k(p), self.names(p))
        if t is None and s is not None and "=" in s:      # "(a, b, c) = (1, 2, 3)": names on the left, values right
            lhs, _, rhs = str(s).rpartition("=")
            ns = self.names(p)
            lh = [x.strip() for x in lhs.replace("(", " ").replace(")", " ").split(",")]
            if ns and lh == list(ns):
                t = core.parse_ints(rhs, self.k(p), None)
        return t

    def in_box(self, p, t):
        return len(t) == self.k(p) and all(lo <= x <= hi for x, (lo, hi) in zip(t, self.box(p)))

    def valid(self, p, t):
        if t is None or any(abs(x) > ABS_MAX for x in t) or not self.in_box(p, t):
            return False
        return bool(self.ok(p, tuple(t)))

    def canon(self, p, t):
        return tuple(sorted(t)) if self.unordered else tuple(t)

    def key(self, p, t):
        return self.canon(p, tuple(t))

    def all_keys(self, p):
        return {self.canon(p, t) for t in self.solve(p)}

    def solve(self, p):
        return bruteforce(self.box(p), lambda t: self.ok(p, t))

    def fmt(self, p, key):
        return ", ".join(str(x) for x in key)

    def space(self, p):
        if self.unordered:
            (lo, hi), n = self.box(p)[0], self.k(p)
            return multisets(lo, hi, n)
        return math.prod(hi - lo + 1 for lo, hi in self.box(p))

    def guess_rate(self, p, n_keys=None):
        n = len(self.all_keys(p)) if n_keys is None else n_keys
        return n / max(1, self.space(p))

    def answer_fmt(self, p):
        ns = self.names(p)
        return comma(ns) if ns else "a comma-separated list of integers"

    def text(self, p):
        st, f = self.statement(p)
        return st, f, f

    def problem(self, p):
        """Labelled answer form for named parameters (\\boxed{a = ..., b = ...}); the positional form otherwise. The
        midtrained models tend to box a single number under the plain 'Write the answer as a, b' ending."""
        st, f = self.statement(p)
        ns = self.names(p)
        if ns:
            end = ("End with the answer in the form \\boxed{" + ", ".join(f"{n} = ..." for n in ns)
                   + "}, with the dots replaced by the values.")
        else:
            end = f"Write the final answer in \\boxed{{}} as {f}."
        return f"{st} Several answers are possible; any one correct answer is accepted. Give one solution. {end}"


# ----------------------------------------------------------------------------------------------- polynomials
def poly_expr(d):
    terms = ["a", "bx"] + [f"{LET[i]}x^{i}" for i in range(2, d + 1)]
    return " + ".join(terms)


def functional(c, d):
    t = c["t"]
    if t in ("val", "root"):
        x = F(c["x"])
        return [x ** i for i in range(d + 1)]
    if t == "der":
        x, m = F(c["x"]), c["m"]
        return [F(math.perm(i, m)) * x ** (i - m) if i >= m else F(0) for i in range(d + 1)]
    if t == "int":
        lo, hi = F(c["lo"]), F(c["hi"])
        return [(hi ** (i + 1) - lo ** (i + 1)) / (i + 1) for i in range(d + 1)]
    if t == "diff":
        u, v = F(c["x"]), F(c["y"])
        return [u ** i - v ** i for i in range(d + 1)]
    raise ValueError(t)


def con_text(c):
    t = c["t"]
    v = fstr(fdec(c.get("v", "0")))
    if t == "val":
        return f"p({c['x']}) = {v}"
    if t == "root":
        return f"p({c['x']}) = 0"
    if t == "der":
        return f"p{chr(39) * c['m']}({c['x']}) = {v}"
    if t == "int":
        return f"\\int_{{{c['lo']}}}^{{{c['hi']}}} p(x)\\,dx = {v}"
    if t == "diff":
        return f"p({c['x']}) - p({c['y']}) = {v}"
    raise ValueError(t)


INTS = [(0, 1), (0, 2), (-1, 1), (1, 2), (-1, 0)]


def make_con(kind, rng, pts):
    if kind == "val":
        return {"t": "val", "x": rng.choice(pts)}
    if kind in ("der", "der2"):
        return {"t": "der", "x": rng.choice(pts), "m": 1 if kind == "der" else 2}
    if kind == "int":
        lo, hi = rng.choice(INTS)
        return {"t": "int", "lo": lo, "hi": hi}
    if kind == "diff":
        u, v = rng.sample(pts, 2)
        return {"t": "diff", "x": u, "y": v}
    raise ValueError(kind)


class Poly(PO):
    domain = "polynomial"
    SPEC = None                 # tier -> dict(d, B, m, pts, kinds, req)
    nz = False

    def names(self, p):
        return tuple(LET[:p["d"] + 1])

    def box(self, p):
        return [(-p["B"], p["B"])] * (p["d"] + 1)

    def rows(self, p):
        return [functional(c, p["d"]) for c in p["cons"]], [fdec(c.get("v", "0")) for c in p["cons"]]

    def ok(self, p, t):
        A, b = self.rows(p)
        return all(sum(a * x for a, x in zip(r, t)) == v for r, v in zip(A, b)) and (not self.nz or any(t))

    def solve(self, p):
        A, b = self.rows(p)
        return [t for t in lattice(A, b, self.box(p)) if not self.nz or any(t)]

    def cons_text(self, p):
        cs = [con_text(c) for c in p["cons"]]
        return cs[0] if len(cs) == 1 else ", ".join(cs[:-1]) + " and " + cs[-1]

    def statement(self, p):
        ns = self.names(p)
        st = (f"Find integers {comma(ns)} with -{p['B']} <= {comma(ns)} <= {p['B']} such that the polynomial "
              f"p(x) = {poly_expr(p['d'])} satisfies {self.cons_text(p)}.")
        return st, comma(ns)

    def plant(self, rng, s):
        return [rng.randint(-s["B"], s["B"]) for _ in range(s["d"] + 1)]

    def sample(self, rng, tier):
        s = self.SPEC[tier]
        d = s["d"]
        h = self.plant(rng, s)
        if h is None:
            return None
        cons, seen = [], []
        kinds = list(s["kinds"])
        for i in range(60):
            if len(cons) == s["m"]:
                break
            kind = s["req"][len(cons)] if len(cons) < len(s.get("req", ())) else rng.choice(kinds)
            c = make_con(kind, rng, s["pts"])
            fv = functional(c, d)
            if fv in seen or all(v == 0 for v in fv):
                continue
            seen.append(fv)
            c["v"] = fenc(sum(a * x for a, x in zip(fv, h)))
            cons.append(c)
        if len(cons) < s["m"]:
            return None
        return {"d": d, "B": s["B"], "cons": cons}


class PolyValues(Poly):
    name = "po_poly_values"
    SPEC = {0: dict(d=2, B=5, m=2, pts=[0, 1, -1], kinds=["val"]),
            1: dict(d=2, B=9, m=2, pts=[-2, -1, 0, 1, 2], kinds=["val"]),
            2: dict(d=3, B=6, m=3, pts=[-2, -1, 0, 1, 2], kinds=["val"]),
            3: dict(d=4, B=4, m=3, pts=[-2, -1, 0, 1, 2, 3], kinds=["val"])}


class PolySlope(Poly):
    name = "po_poly_slope"
    SPEC = {0: dict(d=2, B=5, m=2, pts=[0, 1], kinds=["val"], req=["der"]),
            1: dict(d=2, B=9, m=2, pts=[-1, 0, 1, 2], kinds=["val", "der"], req=["der"]),
            2: dict(d=3, B=6, m=3, pts=[-1, 0, 1, 2], kinds=["val", "der", "der2"], req=["der"]),
            3: dict(d=3, B=9, m=3, pts=[-2, -1, 0, 1, 2, 3], kinds=["val", "der", "der2"], req=["der2"])}


class PolyIntegral(Poly):
    name = "po_poly_integral"
    SPEC = {0: dict(d=2, B=5, m=2, pts=[0, 1], kinds=["val"], req=["int"]),
            1: dict(d=2, B=9, m=2, pts=[-1, 0, 1, 2], kinds=["val", "int"], req=["int"]),
            2: dict(d=3, B=6, m=3, pts=[-1, 0, 1, 2], kinds=["val", "int"], req=["int"]),
            3: dict(d=3, B=9, m=3, pts=[-1, 0, 1, 2], kinds=["val", "int", "der"], req=["int", "int"])}


class PolyMixed(Poly):
    name = "po_poly_mixed"
    SPEC = {0: dict(d=2, B=6, m=2, pts=[-1, 0, 1, 2], kinds=["val", "der", "int", "diff"]),
            1: dict(d=3, B=6, m=3, pts=[-1, 0, 1, 2], kinds=["val", "der", "der2", "int", "diff"]),
            2: dict(d=3, B=9, m=3, pts=[-2, -1, 0, 1, 2, 3], kinds=["val", "der", "der2", "int", "diff"]),
            3: dict(d=4, B=6, m=4, pts=[-2, -1, 0, 1, 2, 3], kinds=["val", "der", "der2", "int", "diff"])}


class PolyCritical(Poly):
    """p'(r) = 0 (a critical point at r) plus values; the planted polynomial has the critical point."""
    name = "po_poly_critical"
    SPEC = {0: dict(d=2, B=6, m=2, pts=[-1, 0, 1, 2], kinds=["val"], req=["crit"]),
            1: dict(d=3, B=6, m=3, pts=[-1, 0, 1, 2], kinds=["val"], req=["crit"]),
            2: dict(d=3, B=9, m=3, pts=[-2, -1, 0, 1, 2, 3], kinds=["val", "der2"], req=["crit"]),
            3: dict(d=4, B=6, m=4, pts=[-2, -1, 0, 1, 2, 3], kinds=["val", "der2"], req=["crit", "crit"])}
    NEED_VAL = True

    def sample(self, rng, tier):
        s = self.SPEC[tier]
        d, B = s["d"], s["B"]
        ncrit = s["req"].count("crit")
        rs = rng.sample([r for r in s["pts"] if r != 0] + [0], ncrit)
        for _ in range(200):          # planted p with p'(r) = 0 at every r in rs: solve for b (and c when 2 crits)
            h = [rng.randint(-B, B) for _ in range(d + 1)]
            A = [functional({"t": "der", "x": r, "m": 1}, d) for r in rs]
            free_idx = [1, 2][:ncrit]
            rhs = [-sum(a * x for j, (a, x) in enumerate(zip(row, h)) if j not in free_idx) for row in A]
            M = [[row[j] for j in free_idx] for row in A]
            if ncrit == 1:
                if M[0][0] == 0:
                    continue
                sol = [rhs[0] / M[0][0]]
            else:
                det = M[0][0] * M[1][1] - M[0][1] * M[1][0]
                if det == 0:
                    continue
                sol = [(rhs[0] * M[1][1] - rhs[1] * M[0][1]) / det, (M[0][0] * rhs[1] - M[1][0] * rhs[0]) / det]
            if all(v.denominator == 1 and abs(v) <= B for v in sol):
                for j, v in zip(free_idx, sol):
                    h[j] = int(v)
                break
        else:
            return None
        cons = [{"t": "der", "x": r, "m": 1, "v": "0"} for r in rs]
        seen = [functional(c, d) for c in cons]
        for _ in range(60):
            if len(cons) == s["m"]:
                break
            c = make_con("val" if len(cons) == ncrit else rng.choice(s["kinds"]), rng, s["pts"])
            fv = functional(c, d)
            if fv in seen:
                continue
            seen.append(fv)
            c["v"] = fenc(sum(a * x for a, x in zip(fv, h)))
            cons.append(c)
        return {"d": d, "B": B, "cons": cons} if len(cons) == s["m"] else None


class PolyRoots(Poly):
    name = "po_poly_roots"
    nz = True
    SPEC = {0: dict(d=2, B=(4, 9), r=1, rr=5), 1: dict(d=3, B=(4, 8), r=2, rr=4), 2: dict(d=3, B=(6, 12), r=2, rr=5),
            3: dict(d=4, B=(5, 9), r=3, rr=4)}

    def sample(self, rng, tier):
        s = self.SPEC[tier]
        roots = rng.sample([r for r in range(-s["rr"], s["rr"] + 1) if r != 0], s["r"])
        return {"d": s["d"], "B": rng.randint(*s["B"]), "cons": [{"t": "root", "x": r, "v": "0"} for r in sorted(roots)]}

    def statement(self, p):
        ns = self.names(p)
        rs = [c["x"] for c in p["cons"]]
        cond = " = ".join(f"p({r})" for r in rs) + " = 0"
        st = (f"Find integers {comma(ns)} with -{p['B']} <= {comma(ns)} <= {p['B']}, not all zero, such that the "
              f"polynomial p(x) = {poly_expr(p['d'])} satisfies {cond}.")
        return st, comma(ns)


class QuadRootDiff(PO):
    """x^2 + bx + c with two distinct integer roots that differ by D (tiers add sign conditions on the roots)."""
    name = "po_quad_rootdiff"
    domain = "polynomial"
    vars = ("b", "c")
    SPEC = {0: dict(B=(12, 24), D=(1, 5), cond=[None]), 1: dict(B=(20, 40), D=(2, 7), cond=[None, "pos", "neg"]),
            2: dict(B=(30, 60), D=(3, 9), cond=["pos", "neg", "opp"]), 3: dict(B=(50, 90), D=(4, 12), cond=["pos", "neg", "opp"])}

    def box(self, p):
        return [(-p["B"], p["B"])] * 2

    def roots(self, b, c):
        disc = b * b - 4 * c
        if disc <= 0:
            return None
        s = math.isqrt(disc)
        if s * s != disc or (s - b) % 2:
            return None
        return ((-b - s) // 2, (-b + s) // 2)

    def ok(self, p, t):
        r = self.roots(*t)
        if r is None or r[1] - r[0] != p["D"]:
            return False
        return (p["cond"] is None or (p["cond"] == "pos" and r[0] > 0) or (p["cond"] == "neg" and r[1] < 0)
                or (p["cond"] == "opp" and r[0] < 0 < r[1]))

    def solve(self, p):
        out = []
        for r in range(-p["B"] - p["D"], p["B"] + 1):
            b, c = -(2 * r + p["D"]), r * (r + p["D"])
            if abs(b) <= p["B"] and abs(c) <= p["B"] and self.ok(p, (b, c)):
                out.append((b, c))
        return out

    def sample(self, rng, tier):
        s = self.SPEC[tier]
        return {"B": rng.randint(*s["B"]), "D": rng.randint(*s["D"]), "cond": rng.choice(s["cond"])}

    def statement(self, p):
        extra = {None: "", "pos": " Both roots must be positive.", "neg": " Both roots must be negative.",
                 "opp": " One root must be negative and the other positive."}
        return (f"Find integers b, c with -{p['B']} <= b, c <= {p['B']} such that x^2 + bx + c has two integer roots "
                f"that differ by {p['D']}.{extra[p['cond']]}", "b, c")


class Vieta3(PO):
    """(x - a)(x - b)(x - c) with given x^2-coefficient and constant term; key = sorted (a, b, c)."""
    name = "po_vieta3"
    domain = "polynomial"
    vars = ("a", "b", "c")
    unordered = True
    SPEC = {0: 4, 1: 7, 2: 12, 3: 20}

    def box(self, p):
        P = abs(p["P"])
        return [(-P, P)] * 3

    def ok(self, p, t):
        a, b, c = t
        return a + b + c == p["S"] and a * b * c == p["P"]

    def solve(self, p):
        S, P = p["S"], p["P"]
        out = set()
        for a in sdivisors(P):
            for b in sdivisors(P // a):
                c = P // (a * b)
                if a + b + c == S:
                    out.add((a, b, c))
        return sorted(out)

    def space(self, p):
        return multisets(-abs(p["P"]), abs(p["P"]), 3)

    def sample(self, rng, tier):
        B = self.SPEC[tier]
        v = [rng.choice([x for x in range(-B, B + 1) if x]) for _ in range(3)]
        return {"S": sum(v), "P": v[0] * v[1] * v[2]}

    def statement(self, p):
        return (f"Find integers a, b, c such that the cubic (x - a)(x - b)(x - c) has x^2-coefficient {-p['S']} and "
                f"constant term {-p['P']}.", "a, b, c")


# ----------------------------------------------------------------------------------------------- functions / sequences
class LinCommute(PO):
    """f(x) = ax + b, g(x) = cx + d with f(g(x)) = Ax + P and g(f(x)) = Ax + Q."""
    name = "po_lin_commute"
    domain = "function"
    vars = ("a", "b", "c", "d")
    SPEC = {0: 5, 1: 8, 2: 12, 3: 20}

    def box(self, p):
        return [(-p["B"], p["B"])] * 4

    def ok(self, p, t):
        a, b, c, d = t
        return a * c == p["A"] and a * d + b == p["P"] and c * b + d == p["Q"]

    def solve(self, p):
        out = []
        A, P, Q, B = p["A"], p["P"], p["Q"], p["B"]
        for a in sdivisors(A):
            c = A // a
            num = Q - c * P
            if num % (1 - A):
                continue
            d = num // (1 - A)
            b = P - a * d
            t = (a, b, c, d)
            if self.in_box(p, t) and self.ok(p, t):
                out.append(t)
        return out

    def sample(self, rng, tier):
        B = self.SPEC[tier]
        lo = max(3, B // 3)
        while True:
            a, c = rng.randint(-B, B), rng.randint(-B, B)
            if a * c not in (0, 1):
                break
        b, d = rng.randint(-lo, lo), rng.randint(-lo, lo)
        return {"B": B, "A": a * c, "P": a * d + b, "Q": c * b + d}

    def statement(self, p):
        B = p["B"]
        return (f"Find integers a, b, c, d with -{B} <= a, b, c, d <= {B} such that f(x) = ax + b and g(x) = cx + d "
                f"satisfy f(g(x)) = {lin_txt(p['A'], p['P'])} and g(f(x)) = {lin_txt(p['A'], p['Q'])}.", "a, b, c, d")


def lin_txt(m, k):
    s = "x" if m == 1 else "-x" if m == -1 else f"{m}x"
    return s if k == 0 else f"{s} + {k}" if k > 0 else f"{s} - {-k}"


class Mobius(PO):
    """f(x) = (ax + b)/(cx + d), ad - bc != 0, through two points; key = (a, b, c, d) up to scaling."""
    name = "po_mobius"
    domain = "function"
    vars = ("a", "b", "c", "d")
    SPEC = {0: 3, 1: 5, 2: 7, 3: 9}

    def box(self, p):
        return [(-p["B"], p["B"])] * 4

    def ok(self, p, t):
        a, b, c, d = t
        if a * d - b * c == 0:
            return False
        for x, y in p["pts"]:
            x, y = F(x), fdec(y)
            den = c * x + d
            if den == 0 or (a * x + b) / den != y:
                return False
        return True

    def canon(self, p, t):
        g = math.gcd(*t)
        t = tuple(v // g for v in t)
        first = next(v for v in t if v)
        return tuple(-v for v in t) if first < 0 else t

    def solve(self, p):
        A, b = [], []
        for x, y in p["pts"]:
            x, y = F(x), fdec(y)
            A.append([x, F(1), -y * x, -y])
            b.append(F(0))
        return [t for t in lattice(A, b, self.box(p)) if self.ok(p, t)]

    def space(self, p):
        return (2 * p["B"] + 1) ** 4 // 2

    def sample(self, rng, tier):
        B = self.SPEC[tier]
        h = 2 if tier < 2 else 3
        for _ in range(100):
            a, b, c, d = (rng.randint(-h, h) for _ in range(4))
            if a * d - b * c == 0 or c == 0:
                continue
            xs = rng.sample([x for x in range(-3, 4) if c * x + d != 0], 2)
            pts = [[x, fenc(F(a * x + b, c * x + d))] for x in sorted(xs)]
            return {"B": B, "pts": pts}
        return None

    def statement(self, p):
        B = p["B"]
        cond = " and ".join(f"f({x}) = {fstr(fdec(y))}" for x, y in p["pts"])
        return (f"Find integers a, b, c, d with -{B} <= a, b, c, d <= {B} and ad - bc \\ne 0 such that "
                f"f(x) = \\frac{{ax + b}}{{cx + d}} satisfies {cond}.", "a, b, c, d")


def recur(u0, u1, p, q, m):
    a, b = u0, u1
    for _ in range(m):
        a, b = b, p * b + q * a
    return a


class RecurCoef(PO):
    """u_0, u_1 given, u_{n+2} = p u_{n+1} + q u_n; find (p, q) with u_m = T."""
    name = "po_recur_coef"
    domain = "sequence"
    vars = ("p", "q")
    SPEC = {0: (3, 5), 1: (3, 9), 2: (4, 6), 3: (5, 5)}

    def box(self, p):
        return [(-p["B"], p["B"])] * 2

    def ok(self, p, t):
        return recur(p["u0"], p["u1"], t[0], t[1], p["m"]) == p["T"]

    def sample(self, rng, tier):
        m, B = self.SPEC[tier]
        u0, u1 = rng.randint(-3, 3), rng.randint(1, 4)
        pp, qq = rng.randint(-B, B), rng.randint(-B, B)
        return {"u0": u0, "u1": u1, "m": m, "B": B, "T": recur(u0, u1, pp, qq, m)}

    def statement(self, p):
        B = p["B"]
        return (f"Find integers p, q with -{B} <= p, q <= {B} such that the sequence with u_0 = {p['u0']}, "
                f"u_1 = {p['u1']} and u_{{n+2}} = p u_{{n+1}} + q u_n has u_{p['m']} = {p['T']}.", "p, q")


class RecurInit(PO):
    """p, q given; find (u_0, u_1) with u_m = T."""
    name = "po_recur_init"
    domain = "sequence"
    vars = ("x", "y")
    SPEC = {0: (3, 10), 1: (4, 15), 2: (5, 20), 3: (6, 30)}

    def box(self, p):
        return [(-p["B"], p["B"])] * 2

    def ok(self, p, t):
        return recur(t[0], t[1], p["p"], p["q"], p["m"]) == p["T"]

    def solve(self, p):
        c0 = recur(1, 0, p["p"], p["q"], p["m"])
        c1 = recur(0, 1, p["p"], p["q"], p["m"])
        return lattice([[c0, c1]], [p["T"]], self.box(p))

    def sample(self, rng, tier):
        m, B = self.SPEC[tier]
        pp, qq = rng.choice([1, 2, -1, 3, -2]), rng.choice([1, -1, 2, 1])
        u0, u1 = rng.randint(-B, B), rng.randint(-B, B)
        return {"p": pp, "q": qq, "m": m, "B": B, "T": recur(u0, u1, pp, qq, m)}

    def statement(self, p):
        B = p["B"]
        rec = f"u_{{n+2}} = {lin2(p['p'], 'u_{n+1}', p['q'], 'u_n')}"
        return (f"Find integers x, y with -{B} <= x, y <= {B} such that the sequence with u_0 = x, u_1 = y and "
                f"{rec} has u_{p['m']} = {p['T']}.", "x, y")


def lin2(a, s, b, t):
    def term(c, v):
        return v if c == 1 else f"-{v}" if c == -1 else f"{c}{v}"
    first = term(a, s)
    sb = term(abs(b), t)
    return f"{first} + {sb}" if b > 0 else f"{first} - {sb}"


class APSum(PO):
    name = "po_ap_sum"
    domain = "sequence"
    vars = ("a", "d")
    SPEC = {0: ((4, 6), 10), 1: ((5, 9), 20), 2: ((7, 12), 30), 3: ((10, 16), 50)}

    def box(self, p):
        return [(-p["B"], p["B"])] * 2

    def ok(self, p, t):
        n = p["n"]
        return n * t[0] + n * (n - 1) // 2 * t[1] == p["S"]

    def solve(self, p):
        n = p["n"]
        return lattice([[n, n * (n - 1) // 2]], [p["S"]], self.box(p))

    def sample(self, rng, tier):
        nr, B = self.SPEC[tier]
        n = rng.randint(*nr)
        a, d = rng.randint(-B, B), rng.randint(-B, B)
        return {"n": n, "B": B, "S": n * a + n * (n - 1) // 2 * d}

    def statement(self, p):
        B = p["B"]
        return (f"Find integers a, d with -{B} <= a, d <= {B} such that the arithmetic progression a, a + d, a + 2d, "
                f"... has sum of its first {p['n']} terms equal to {p['S']}.", "a, d")


class GPTerm(PO):
    """a r^m = T with r not in {-1, 0, 1}."""
    name = "po_gp_term"
    domain = "sequence"
    vars = ("a", "r")
    SPEC = {0: (2, 3), 1: (3, 3), 2: (2, 6), 3: (4, 4)}

    def rmax(self, p):
        T, m = abs(p["T"]), p["m"]
        r = int(round(T ** (1 / m))) + 1
        while r ** m > T:
            r -= 1
        return max(2, r)

    def box(self, p):
        T = abs(p["T"])
        R = self.rmax(p)
        return [(-T, T), (-R, R)]

    def ok(self, p, t):
        a, r = t
        return abs(r) >= 2 and a * r ** p["m"] == p["T"]

    def solve(self, p):
        out = []
        for r in range(2, self.rmax(p) + 1):
            for s in (r, -r):
                if p["T"] % (s ** p["m"]) == 0:
                    out.append((p["T"] // s ** p["m"], s))
        return out

    def sample(self, rng, tier):
        m, k = self.SPEC[tier]
        primes = [2, 3, 5, 7]
        T = 1
        for _ in range(k):
            T *= rng.choice(primes) ** m
        T *= rng.choice([1, 2, 3, 5, -1, -2, -3])
        return {"m": m, "T": T}

    def statement(self, p):
        return (f"Find integers a, r with r \\ne 0, 1, -1 such that the geometric sequence a, ar, ar^2, ... has "
                f"ar^{p['m']} = {p['T']}.", "a, r")


# ----------------------------------------------------------------------------------------------- number theory
class DiffSquares(PO):
    name = "po_diff_squares"
    domain = "number"
    vars = ("x", "y")
    SPEC = {0: (2, [3, 5, 7, 11, 13, 17, 19, 23]), 1: (3, [3, 5, 7, 11, 13]), 2: (4, [3, 5, 7, 11, 13]),
            3: (4, [2, 3, 5, 7, 11, 13, 17])}

    def box(self, p):
        return [(0, (p["N"] + 1) // 2), (0, (p["N"] + 1) // 2)]

    def ok(self, p, t):
        return t[0] * t[0] - t[1] * t[1] == p["N"]

    def solve(self, p):
        N, out = p["N"], []
        for e in divisors(N):
            f = N // e
            if e <= f and (e + f) % 2 == 0:
                out.append(((e + f) // 2, (f - e) // 2))
        return out

    def space(self, p):
        h = (p["N"] + 1) // 2 + 1
        return h * h

    def sample(self, rng, tier):
        k, pr = self.SPEC[tier]
        N = 1
        for _ in range(k):
            N *= rng.choice(pr)
        return {"N": N}

    def statement(self, p):
        return f"Find nonnegative integers x, y such that x^2 - y^2 = {p['N']}.", "x, y"


class TwoSqComp(PO):
    """x^2 + y^2 = (a^2 + b^2)(c^2 + d^2)...; key = (max, min) of |x|, |y|."""
    name = "po_two_sq_comp"
    domain = "number"
    vars = ("x", "y")
    SPEC = {0: (2, 5), 1: (2, 8), 2: (3, 5), 3: (3, 8)}

    def N(self, p):
        return math.prod(a * a + b * b for a, b in p["fac"])

    def box(self, p):
        r = math.isqrt(self.N(p))
        return [(-r, r), (-r, r)]

    def ok(self, p, t):
        return t[0] ** 2 + t[1] ** 2 == self.N(p)

    def canon(self, p, t):
        return tuple(sorted((abs(t[0]), abs(t[1])), reverse=True))

    def solve(self, p):
        N, out = self.N(p), []
        for x in range(math.isqrt(N) + 1):
            y2 = N - x * x
            y = math.isqrt(y2)
            if y * y == y2:
                for sx in {x, -x}:
                    for sy in {y, -y}:
                        out.append((sx, sy))
        return out

    def space(self, p):
        r = math.isqrt(self.N(p))
        return (r + 1) * (r + 2) // 2

    def sample(self, rng, tier):
        k, h = self.SPEC[tier]
        fac = []
        for _ in range(k):
            a, b = rng.randint(1, h), rng.randint(1, h)
            fac.append(sorted([a, b]))
        return {"fac": fac}

    def statement(self, p):
        prod = "".join(f"({a}^2 + {b}^2)" for a, b in p["fac"])
        return f"Find integers x, y such that x^2 + y^2 = {prod}.", "x, y"


class GcdLcm(PO):
    name = "po_gcd_lcm"
    domain = "number"
    unordered = True
    SPEC = {0: (2, 2), 1: (3, 2), 2: (4, 2), 3: (3, 3)}

    def names(self, p):
        return tuple(LET[:p["k"]])

    def box(self, p):
        return [(1, p["L"])] * p["k"]

    def ok(self, p, t):
        return math.gcd(*t) == p["g"] and math.lcm(*t) == p["L"]

    def solve(self, p):
        ds = [d for d in divisors(p["L"]) if d % p["g"] == 0]
        return [t for t in itertools.combinations_with_replacement(ds, p["k"]) if self.ok(p, t)]

    def space(self, p):
        return multisets(1, p["L"], p["k"])

    def sample(self, rng, tier):
        w, k = self.SPEC[tier]
        g = rng.choice([1, 2, 3, 4, 6, 10])
        ps = rng.sample([2, 3, 5, 7, 11, 13], w)
        L = g * math.prod(q ** rng.randint(1, 2) for q in ps)
        return {"g": g, "L": L, "k": k}

    def statement(self, p):
        ns = self.names(p)
        return (f"Find positive integers {comma(ns)} whose greatest common divisor is {p['g']} and whose least common "
                f"multiple is {p['L']}.", comma(ns))


class Egyptian(PO):
    """1/a + 1/b = u/v, 0 < a <= b."""
    name = "po_egyptian"
    domain = "number"
    vars = ("a", "b")
    unordered = True
    SPEC = {0: (1, 30), 1: (1, 120), 2: (2, 200), 3: (3, 400)}

    def box(self, p):
        u, v = p["u"], p["v"]
        hi = v * (v + 1)
        return [(1, hi), (1, hi)]

    def ok(self, p, t):
        return F(1, t[0]) + F(1, t[1]) == F(p["u"], p["v"])

    def solve(self, p):
        u, v = p["u"], p["v"]
        out = []
        for a in range(v // u + 1, 2 * v // u + 1):
            num, den = v * a, u * a - v
            if den > 0 and num % den == 0 and num // den >= a:
                out.append((a, num // den))
        return out

    def space(self, p):
        return multisets(1, self.box(p)[0][1], 2)

    def sample(self, rng, tier):
        u, vmax = self.SPEC[tier]
        while True:
            v = rng.randint(6, vmax)
            if math.gcd(u, v) == 1:
                return {"u": u, "v": v}

    def statement(self, p):
        return f"Find positive integers a, b such that \\frac{{1}}{{a}} + \\frac{{1}}{{b}} = {fstr(F(p['u'], p['v']))}.", "a, b"


class SFFT(PO):
    """xy + ax + by = c in positive integers."""
    name = "po_sfft"
    domain = "number"
    vars = ("x", "y")
    SPEC = {0: (5, 6), 1: (10, 9), 2: (20, 12), 3: (40, 15)}

    def box(self, p):
        h = abs(p["c"]) + abs(p["a"] * p["b"]) + abs(p["a"]) + abs(p["b"])
        return [(1, h), (1, h)]

    def ok(self, p, t):
        x, y = t
        return x * y + p["a"] * x + p["b"] * y == p["c"]

    def solve(self, p):
        a, b, c = p["a"], p["b"], p["c"]
        M = c + a * b               # (x + b)(y + a) = M; M = 0 is never sampled
        out = []
        if M == 0:
            return bruteforce(self.box(p), lambda t: self.ok(p, t))
        for e in sdivisors(M):
            x, y = e - b, M // e - a
            if x >= 1 and y >= 1 and self.in_box(p, (x, y)):
                out.append((x, y))
        return out

    def sample(self, rng, tier):
        X, A = self.SPEC[tier]
        a, b = rng.randint(-A, A), rng.randint(-A, A)
        x, y = rng.randint(1, X), rng.randint(1, X)
        c = x * y + a * x + b * y
        return None if c + a * b == 0 else {"a": a, "b": b, "c": c}    # (x + b)(y + a) = 0: a whole line

    def statement(self, p):
        lhs = "xy"
        for co, v in ((p["a"], "x"), (p["b"], "y")):
            if co:
                lhs += f" + {v}" if co == 1 else f" - {v}" if co == -1 else f" + {co}{v}" if co > 0 else f" - {-co}{v}"
        return f"Find positive integers x, y such that {lhs} = {p['c']}.", "x, y"


class Lin2Box(PO):
    """ax + by = n with 0 <= x, y <= B (a, b of either sign)."""
    name = "po_lin2_box"
    domain = "number"
    vars = ("x", "y")
    SPEC = {0: (9, 20), 1: (19, 50), 2: (37, 100), 3: (73, 200)}

    def box(self, p):
        return [(0, p["B"]), (0, p["B"])]

    def ok(self, p, t):
        return p["a"] * t[0] + p["b"] * t[1] == p["n"]

    def solve(self, p):
        return lattice([[p["a"], p["b"]]], [p["n"]], self.box(p))

    def sample(self, rng, tier):
        C, B = self.SPEC[tier]
        a = rng.randint(2, C)
        b = rng.choice([1, -1]) * rng.randint(2, C)
        x, y = rng.randint(0, B), rng.randint(0, B)
        return {"a": a, "b": b, "B": B, "n": a * x + b * y}

    def statement(self, p):
        return (f"Find integers x, y with 0 <= x, y <= {p['B']} such that {lin2(p['a'], 'x', p['b'], 'y')} = {p['n']}.",
                "x, y")


class CRTRange(PO):
    name = "po_crt_range"
    domain = "number"
    vars = ("x",)
    SPEC = {0: ([5, 7, 8, 9, 11], 2), 1: ([7, 9, 11, 13, 16], 2), 2: ([4, 5, 7, 9, 11], 3),
            3: ([6, 9, 10, 14, 15, 21], 3)}

    def box(self, p):
        return [(p["L"], p["U"])]

    def ok(self, p, t):
        return all(t[0] % m == r for r, m in p["cong"])

    def solve(self, p):
        return [(x,) for x in range(p["L"], p["U"] + 1) if self.ok(p, (x,))]

    def sample(self, rng, tier):
        mods, k = self.SPEC[tier]
        ms = rng.sample(mods, k)
        x0 = rng.randint(0, 10 ** 4)
        cong = [[x0 % m, m] for m in ms]
        l = math.lcm(*ms)
        L = rng.randint(100, 2000)
        U = L + l * rng.randint(2, 12) + rng.randint(0, l - 1)
        return {"cong": cong, "L": L, "U": U}

    def statement(self, p):
        cs = ", ".join(f"x \\equiv {r} \\pmod{{{m}}}" for r, m in p["cong"])
        return f"Find an integer x with {p['L']} <= x <= {p['U']} such that {cs}.", "x"


@functools.lru_cache(maxsize=None)
def phi_table(n):
    ph = list(range(n + 1))
    for i in range(2, n + 1):
        if ph[i] == i:
            for j in range(i, n + 1, i):
                ph[j] -= ph[j] // i
    return ph


@functools.lru_cache(maxsize=None)
def sigma_table(n):
    s = [0] * (n + 1)
    for i in range(1, n + 1):
        for j in range(i, n + 1, i):
            s[j] += i
    return s


class TotientInv(PO):
    name = "po_totient_inv"
    domain = "number"
    vars = ("n",)
    SPEC = {0: 60, 1: 120, 2: 240, 3: 480}
    TAB = 2 * 480 * 480 + 2

    def box(self, p):
        return [(1, 2 * p["m"] * p["m"] + 1)]

    def ok(self, p, t):
        n = t[0]
        return n >= 1 and (phi_table(self.TAB)[n] if n <= self.TAB else self._phi(n)) == p["m"]

    def _phi(self, n):
        r = n
        for q in factorize(n):
            r -= r // q
        return r

    def solve(self, p):
        ph = phi_table(self.TAB)
        return [(n,) for n in range(1, 2 * p["m"] * p["m"] + 2) if ph[n] == p["m"]]

    def sample(self, rng, tier):
        top = self.SPEC[tier]
        ph = phi_table(self.TAB)
        while True:
            n = rng.randint(3, 5 * top)
            if ph[n] <= top and ph[n] >= top // 4:
                return {"m": ph[n]}

    def statement(self, p):
        return f"Find a positive integer n with \\varphi(n) = {p['m']}, where \\varphi is Euler's totient function.", "n"


class SigmaInv(PO):
    name = "po_sigma_inv"
    domain = "number"
    vars = ("n",)
    SPEC = {0: 400, 1: 1500, 2: 5000, 3: 20000}
    TAB = 20001

    def box(self, p):
        return [(1, p["s"])]

    def ok(self, p, t):
        n = t[0]
        return 1 <= n <= self.TAB - 1 and sigma_table(self.TAB)[n] == p["s"]

    def solve(self, p):
        sg = sigma_table(self.TAB)
        return [(n,) for n in range(1, p["s"] + 1) if sg[n] == p["s"]]

    def sample(self, rng, tier):
        top = self.SPEC[tier]
        sg = sigma_table(self.TAB)
        while True:
            n = rng.randint(2, top)
            if top // 5 <= sg[n] <= top:
                return {"s": sg[n]}

    def statement(self, p):
        return (f"Find a positive integer n whose divisors (including 1 and n) add up to {p['s']}.", "n")


@functools.lru_cache(maxsize=None)
def tau_table(n):
    t = [0] * (n + 1)
    for i in range(1, n + 1):
        for j in range(i, n + 1, i):
            t[j] += 1
    return t


class TauRange(PO):
    name = "po_tau_range"
    domain = "number"
    vars = ("n",)
    SPEC = {0: (20000, 600, (8, 12)), 1: (50000, 1500, (12, 16)), 2: (80000, 3000, (16, 24)), 3: (100000, 6000, (20, 32))}
    TAB = 110000

    def box(self, p):
        return [(p["L"], p["U"])]

    def ok(self, p, t):
        return p["L"] <= t[0] <= p["U"] <= self.TAB and tau_table(self.TAB)[t[0]] == p["k"]

    def solve(self, p):
        tt = tau_table(self.TAB)
        return [(n,) for n in range(p["L"], p["U"] + 1) if tt[n] == p["k"]]

    def sample(self, rng, tier):
        top, w, kr = self.SPEC[tier]
        L = rng.randint(top // 10, top - w)
        return {"L": L, "U": L + rng.randint(w // 2, w), "k": rng.randrange(kr[0], kr[1] + 1, 2)}

    def statement(self, p):
        return f"Find an integer n with {p['L']} <= n <= {p['U']} that has exactly {p['k']} positive divisors.", "n"


class PythLeg(PO):
    name = "po_pyth_leg"
    domain = "number"
    vars = ("b", "c")
    SPEC = {0: (12, 60), 1: (60, 200), 2: (200, 600), 3: (600, 2000)}

    def box(self, p):
        h = p["a"] * p["a"]
        return [(1, h), (1, h)]

    def ok(self, p, t):
        return p["a"] ** 2 + t[0] ** 2 == t[1] ** 2

    def solve(self, p):
        a2, out = p["a"] ** 2, []
        for e in divisors(a2):
            f = a2 // e
            if e < f and (e + f) % 2 == 0:
                out.append(((f - e) // 2, (e + f) // 2))
        return out

    def sample(self, rng, tier):
        return {"a": rng.randint(*self.SPEC[tier])}

    def statement(self, p):
        return f"Find positive integers b, c such that {p['a']}^2 + b^2 = c^2.", "b, c"


class PythPerim(PO):
    name = "po_pyth_perim"
    domain = "geometry"
    vars = ("a", "b", "c")
    SPEC = {0: (240, 600), 1: (600, 1500), 2: (1500, 4000), 3: (4000, 10000)}

    def box(self, p):
        return [(1, p["P"])] * 3

    def ok(self, p, t):
        a, b, c = t
        return 0 < a <= b < c and a + b + c == p["P"] and a * a + b * b == c * c

    def solve(self, p):
        P, out = p["P"], []
        for a in range(1, P // 3 + 1):
            num, den = P * P - 2 * P * a, 2 * (P - a)
            if num % den == 0:
                b = num // den
                c = P - a - b
                if self.ok(p, (a, b, c)):
                    out.append((a, b, c))
        return out

    def space(self, p):
        return max(1, p["P"] ** 2 // 12)

    def sample(self, rng, tier):
        lo, hi = self.SPEC[tier]
        P = rng.randrange(lo, hi, 2)
        return {"P": P}

    def statement(self, p):
        return (f"Find positive integers a <= b < c with a^2 + b^2 = c^2 (the sides of a right triangle) and "
                f"a + b + c = {p['P']}.", "a, b, c")


class GaussFactor(PO):
    """(a + bi)(c + di) = u + vi, both factors non-units; key = factorisation with a + bi rotated into a > 0, b >= 0."""
    name = "po_gauss_factor"
    domain = "number"
    vars = ("a", "b", "c", "d")
    SPEC = {0: 2, 1: 3, 2: 4, 3: 6}

    def box(self, p):
        r = math.isqrt(p["u"] ** 2 + p["v"] ** 2)
        return [(-r, r)] * 4

    def ok(self, p, t):
        a, b, c, d = t
        return (a * c - b * d == p["u"] and a * d + b * c == p["v"] and a * a + b * b > 1 and c * c + d * d > 1)

    def canon(self, p, t):
        a, b, c, d = t
        for _ in range(4):
            if a > 0 and b >= 0:
                break
            a, b, c, d = -b, a, d, -c        # z * i, w * (-i)
        return (a, b, c, d)

    def solve(self, p):
        u, v = p["u"], p["v"]
        N = u * u + v * v
        out = []
        for a in range(1, math.isqrt(N) + 1):
            for b in range(0, math.isqrt(max(0, N - a * a)) + 1):
                nz = a * a + b * b
                if nz <= 1 or nz >= N or N % nz:
                    continue
                cr, ci = u * a + v * b, v * a - u * b
                if cr % nz or ci % nz:
                    continue
                t = (a, b, cr // nz, ci // nz)
                for _ in range(4):
                    out.append(t)
                    a2, b2, c2, d2 = t
                    t = (-b2, a2, d2, -c2)
        return [t for t in set(out) if self.in_box(p, t)]

    def space(self, p):
        r = math.isqrt(p["u"] ** 2 + p["v"] ** 2)
        return (r + 1) ** 2 * (2 * r + 1) ** 2

    def sample(self, rng, tier):
        R = self.SPEC[tier]
        while True:
            a, b, c, d = (rng.randint(-R, R) for _ in range(4))
            if a * a + b * b > 1 and c * c + d * d > 1:
                return {"u": a * c - b * d, "v": a * d + b * c}

    def statement(self, p):
        u, v = p["u"], p["v"]
        z = f"{u} + {v}i" if v >= 0 else f"{u} - {-v}i"
        return (f"Find integers a, b, c, d such that (a + bi)(c + di) = {z}, where a^2 + b^2 > 1 and c^2 + d^2 > 1.",
                "a, b, c, d")


class OrderMod(PO):
    name = "po_order_mod"
    domain = "number"
    vars = ("x",)
    SPEC = {0: (60, 200), 1: (200, 600), 2: (600, 1500), 3: (1500, 4000)}

    def box(self, p):
        return [(1, p["n"] - 1)]

    def order(self, x, n):
        if not 1 <= x < n:
            return None
        for d in divisors(n - 1):         # n is prime
            if pow(x, d, n) == 1:
                return d
        return None

    def ok(self, p, t):
        return self.order(t[0], p["n"]) == p["k"]

    def solve(self, p):
        return [(x,) for x in range(1, p["n"]) if self.order(x, p["n"]) == p["k"]]

    def sample(self, rng, tier):
        lo, hi = self.SPEC[tier]
        while True:
            n = rng.randint(lo, hi)
            if core.is_prime(n):
                break
        ks = [d for d in divisors(n - 1) if 2 < d < n - 1]
        if not ks:
            return None
        return {"n": n, "k": rng.choice(ks)}

    def statement(self, p):
        return (f"Find an integer x with 1 <= x <= {p['n'] - 1} whose multiplicative order modulo {p['n']} is "
                f"{p['k']}, that is, x^{p['k']} \\equiv 1 \\pmod{{{p['n']}}} and x^j \\not\\equiv 1 for 1 <= j < "
                f"{p['k']}.", "x")


class FractionBetween(PO):
    """p1/q1 < a/b < p2/q2 with 1 <= b <= D; key = a/b in lowest terms."""
    name = "po_fraction_between"
    domain = "number"
    vars = ("a", "b")
    SPEC = {0: 25, 1: 40, 2: 70, 3: 120}

    def box(self, p):
        D = p["D"]
        hi = math.floor(F(p["hi"]) * D) + 1
        return [(0, max(1, hi)), (1, D)]

    def ok(self, p, t):
        a, b = t
        return b >= 1 and F(p["lo"]) < F(a, b) < F(p["hi"])

    def canon(self, p, t):
        g = math.gcd(t[0], t[1]) or 1
        return (t[0] // g, t[1] // g)

    def solve(self, p):
        lo, hi, out = F(p["lo"]), F(p["hi"]), []
        for b in range(1, p["D"] + 1):
            for a in range(math.floor(lo * b), math.ceil(hi * b) + 1):
                if lo < F(a, b) < hi and self.in_box(p, (a, b)):
                    out.append((a, b))
        return out

    def space(self, p):
        return max(1, int(0.3 * p["D"] * (self.box(p)[0][1] + 1)))

    def sample(self, rng, tier):
        D = self.SPEC[tier]
        q1, q2 = rng.randint(D // 2 + 1, 3 * D), rng.randint(D // 2 + 1, 3 * D)
        x = F(rng.randint(1, q1 - 1), q1)
        y = x + F(rng.randint(1, 4), q2 * rng.randint(1, 3))
        if y >= 1:
            return None
        return {"lo": fenc(x), "hi": fenc(y), "D": D}

    def statement(self, p):
        return (f"Find integers a, b with 1 <= b <= {p['D']} such that {fstr(F(p['lo']))} < \\frac{{a}}{{b}} < "
                f"{fstr(F(p['hi']))}.", "a, b")


class LinCongruence(PO):
    name = "po_lin_congruence"
    domain = "number"
    vars = ("x",)
    SPEC = {0: ((2, 4), 3), 1: ((3, 8), 4), 2: ((6, 16), 4), 3: ((12, 40), 5)}

    def box(self, p):
        return [(0, p["M"] - 1)]

    def ok(self, p, t):
        return (p["a"] * t[0] - p["b"]) % p["m"] == 0

    def solve(self, p):
        return [(x,) for x in range(p["M"]) if self.ok(p, (x,))]

    def sample(self, rng, tier):
        (glo, ghi), mult = self.SPEC[tier]
        g = rng.randint(glo, ghi)
        m = g * rng.randint(60, 200)
        a = g * rng.randint(2, m // g - 1)
        if math.gcd(a, m) != g:
            return None
        x = rng.randint(0, m - 1)
        M = m * rng.randint(1, mult) if tier >= 2 else m
        return {"a": a, "b": a * x % m, "m": m, "M": M}

    def statement(self, p):
        return (f"Find an integer x with 0 <= x <= {p['M'] - 1} such that {p['a']}x \\equiv {p['b']} "
                f"\\pmod{{{p['m']}}}.", "x")


class SqrtMod(PO):
    name = "po_sqrt_mod"
    domain = "number"
    vars = ("x",)
    SPEC = {0: (2, [11, 13, 17, 19, 23]), 1: (2, [17, 19, 23, 29, 31, 37, 41]), 2: (3, [5, 7, 11, 13, 17, 19]),
            3: (3, [7, 11, 13, 17, 19, 23, 29])}

    def box(self, p):
        return [(0, p["m"] - 1)]

    def ok(self, p, t):
        return (t[0] * t[0] - p["a"]) % p["m"] == 0

    def solve(self, p):
        return [(x,) for x in range(p["m"]) if self.ok(p, (x,))]

    def sample(self, rng, tier):
        k, pr = self.SPEC[tier]
        m = math.prod(rng.sample(pr, k))
        while True:
            x = rng.randint(2, m - 1)
            if math.gcd(x, m) == 1:
                return {"m": m, "a": x * x % m}

    def statement(self, p):
        return f"Find an integer x with 0 <= x <= {p['m'] - 1} such that x^2 \\equiv {p['a']} \\pmod{{{p['m']}}}.", "x"


class ConsecutiveSum(PO):
    name = "po_consecutive_sum"
    domain = "number"
    vars = ("a", "n")
    SPEC = {0: [3, 5, 7], 1: [3, 5, 7, 9, 11], 2: [3, 5, 7, 11, 13], 3: [3, 5, 7, 11, 13, 17, 19]}

    def box(self, p):
        N = p["N"]
        return [(1, N), (2, math.isqrt(2 * N) + 1)]

    def ok(self, p, t):
        a, n = t
        return n >= 2 and a >= 1 and n * a + n * (n - 1) // 2 == p["N"]

    def solve(self, p):
        N, out = p["N"], []
        for n in range(2, math.isqrt(2 * N) + 2):
            r = N - n * (n - 1) // 2
            if r >= n and r % n == 0:
                out.append((r // n, n))
        return out

    def sample(self, rng, tier):
        pr = self.SPEC[tier]
        N = math.prod(rng.choice(pr) for _ in range(tier + 2)) * rng.choice([1, 2, 4, 8])
        return {"N": N}

    def statement(self, p):
        return (f"Find positive integers a, n with n >= 2 such that a + (a + 1) + ... + (a + n - 1) = {p['N']}.",
                "a, n")


# ----------------------------------------------------------------------------------------------- linear algebra
def mat_tex(entries, n):
    rows = [" & ".join(str(e) for e in entries[i * n:(i + 1) * n]) for i in range(n)]
    return "\\begin{pmatrix} " + " \\\\ ".join(rows) + " \\end{pmatrix}"


AM = mat_tex(["a", "b", "c", "d"], 2)


class Mat2TrDet(PO):
    name = "po_mat2_trdet"
    domain = "linalg"
    vars = ("a", "b", "c", "d")
    SPEC = {0: 3, 1: 5, 2: 7, 3: 10}

    def box(self, p):
        return [(-p["B"], p["B"])] * 4

    def ok(self, p, t):
        a, b, c, d = t
        return a + d == p["t"] and a * d - b * c == p["D"]

    def solve(self, p):
        B, out = p["B"], []
        for a in range(-B, B + 1):
            d = p["t"] - a
            if abs(d) > B:
                continue
            prod = a * d - p["D"]          # = bc
            for b in range(-B, B + 1):
                if b == 0:
                    if prod == 0:
                        out += [(a, 0, c, d) for c in range(-B, B + 1)]
                elif prod % b == 0 and abs(prod // b) <= B:
                    out.append((a, b, prod // b, d))
        return out

    def sample(self, rng, tier):
        B = self.SPEC[tier]
        a, b, c, d = (rng.randint(-B, B) for _ in range(4))
        return {"B": B, "t": a + d, "D": a * d - b * c}

    def statement(self, p):
        B = p["B"]
        return (f"Find integers a, b, c, d with -{B} <= a, b, c, d <= {B} such that the matrix A = {AM} has trace "
                f"{p['t']} and determinant {p['D']}.", "a, b, c, d")


class Mat2Eq(PO):
    """A^2 = I (A != +-I), A^2 = 0 (A != 0) or A^2 = A (A != 0, I), plus one linear relation on the entries."""
    domain = "linalg"
    vars = ("a", "b", "c", "d")
    EQ = None
    SPEC = {0: (3, 1), 1: (5, 2), 2: (8, 2), 3: (12, 3)}      # (B, number of nonzero relation coefficients)

    def box(self, p):
        return [(-p["B"], p["B"])] * 4

    def sq_ok(self, t):
        a, b, c, d = t
        s = (a * a + b * c, a * b + b * d, c * a + d * c, c * b + d * d)
        if self.EQ == "inv":
            return s == (1, 0, 0, 1) and t not in ((1, 0, 0, 1), (-1, 0, 0, -1))
        if self.EQ == "nil":
            return s == (0, 0, 0, 0) and any(t)
        return s == t and t not in ((0, 0, 0, 0), (1, 0, 0, 1))

    def rel(self, p, t):
        return sum(c * x for c, x in zip(p["rel"], t)) == p["s"]

    def ok(self, p, t):
        return self.rel(p, t) and self.sq_ok(t)

    def solve(self, p):
        B = p["B"]
        out = []
        for a, b, c in itertools.product(range(-B, B + 1), repeat=3):
            for d in {-a, 1 - a}:
                t = (a, b, c, d)
                if abs(d) <= B and self.rel(p, t) and self.sq_ok(t):
                    out.append(t)
        return sorted(set(out))

    @functools.lru_cache(maxsize=None)
    def _all(self, B):
        return self.solve({"B": B, "rel": [0, 0, 0, 0], "s": 0})

    def sample(self, rng, tier):
        B, nz = self.SPEC[tier]
        idx = rng.sample(range(3), nz)           # a, b, c (d is tied to a by the trace)
        rel = [rng.choice([1, -1, 2, -2, 3]) if i in idx else 0 for i in range(3)] + [0]
        t = rng.choice([u for u in self._all(B) if max(abs(x) for x in u) >= 2])
        sv = sum(c * x for c, x in zip(rel, t))
        if any(sum(c * x for c, x in zip(rel, u)) == sv for u in self._all(B) if max(abs(x) for x in u) <= 1):
            return None                          # a trivial matrix would be valid
        return {"B": B, "rel": rel, "s": sv}

    def cond(self):
        return {"inv": "A^2 = I and A \\ne \\pm I", "nil": "A^2 = 0 and A \\ne 0",
                "idem": "A^2 = A, A \\ne 0 and A \\ne I"}[self.EQ]

    def statement(self, p):
        B = p["B"]
        terms = [(c, v) for c, v in zip(p["rel"], "abcd") if c]
        lhs = ""
        for i, (c, v) in enumerate(terms):
            mag = v if abs(c) == 1 else f"{abs(c)}{v}"
            lhs += (("-" if c < 0 else "") + mag) if i == 0 else ((" - " if c < 0 else " + ") + mag)
        return (f"Find integers a, b, c, d with -{B} <= a, b, c, d <= {B} such that the matrix A = {AM} satisfies "
                f"{self.cond()} and {lhs} = {p['s']}.", "a, b, c, d")


class Mat2Involution(Mat2Eq):
    name = "po_mat2_involution"
    EQ = "inv"


class Mat2Nilpotent(Mat2Eq):
    name = "po_mat2_nilpotent"
    EQ = "nil"


class Mat2Idempotent(Mat2Eq):
    name = "po_mat2_idempotent"
    EQ = "idem"


class Mat2Eigvec(PO):
    """A v = lam v for a given v, and det A = D."""
    name = "po_mat2_eigvec"
    domain = "linalg"
    vars = ("a", "b", "c", "d")
    SPEC = {0: 4, 1: 6, 2: 8, 3: 10}

    def box(self, p):
        return [(-p["B"], p["B"])] * 4

    def ok(self, p, t):
        a, b, c, d = t
        x, y = p["v"]
        lam = p["lam"]
        return a * x + b * y == lam * x and c * x + d * y == lam * y and a * d - b * c == p["D"]

    def solve(self, p):
        x, y = p["v"]
        lam = p["lam"]
        pts = lattice([[x, y, 0, 0], [0, 0, x, y]], [lam * x, lam * y], self.box(p))
        return [t for t in pts if t[0] * t[3] - t[1] * t[2] == p["D"]]

    def sample(self, rng, tier):
        B = self.SPEC[tier]
        for _ in range(100):
            v = [rng.randint(-3, 3), rng.randint(-3, 3)]
            if v == [0, 0]:
                continue
            lam = rng.randint(-3, 3)
            p = {"B": B, "v": v, "lam": lam, "D": 0}
            pts = lattice([[v[0], v[1], 0, 0], [0, 0, v[0], v[1]]], [lam * v[0], lam * v[1]], self.box(p))
            if pts:
                t = rng.choice(pts)
                p["D"] = t[0] * t[3] - t[1] * t[2]
                return p
        return None

    def statement(self, p):
        B = p["B"]
        x, y = p["v"]
        return (f"Find integers a, b, c, d with -{B} <= a, b, c, d <= {B} such that the matrix A = {AM} has "
                f"determinant {p['D']} and A {mat_tex([x, y], 1)} = {p['lam']} {mat_tex([x, y], 1)}.", "a, b, c, d")


class Mat2Commute(PO):
    """A M = M A for a given non-scalar M, and det A = D."""
    name = "po_mat2_commute"
    domain = "linalg"
    vars = ("a", "b", "c", "d")
    SPEC = {0: 4, 1: 6, 2: 8, 3: 10}

    def box(self, p):
        return [(-p["B"], p["B"])] * 4

    def rows(self, p):
        e, f, g, h = p["M"]
        # AM - MA = 0 for A = (a b; c d), M = (e f; g h)
        return [[0, g, -f, 0], [f, h - e, 0, -f], [-g, 0, e - h, g], [0, -g, f, 0]]

    def ok(self, p, t):
        a, b, c, d = t
        return all(sum(r * x for r, x in zip(row, t)) == 0 for row in self.rows(p)) and a * d - b * c == p["D"]

    def solve(self, p):
        pts = lattice(self.rows(p), [0, 0, 0, 0], self.box(p))
        return [t for t in pts if t[0] * t[3] - t[1] * t[2] == p["D"]]

    def sample(self, rng, tier):
        B = self.SPEC[tier]
        while True:
            M = [rng.randint(-3, 3) for _ in range(4)]
            if not (M[1] == 0 and M[2] == 0 and M[0] == M[3]):
                break
        p = {"B": B, "M": M, "D": 0}
        pts = lattice(self.rows(p), [0, 0, 0, 0], self.box(p))
        t = rng.choice(pts)
        p["D"] = t[0] * t[3] - t[1] * t[2]
        return p

    def statement(self, p):
        B = p["B"]
        return (f"Find integers a, b, c, d with -{B} <= a, b, c, d <= {B} such that the matrix A = {AM} has "
                f"determinant {p['D']} and commutes with {mat_tex(p['M'], 2)}.", "a, b, c, d")


class Mat3Det(PO):
    """3x3 matrix with three unknown entries x, y, z (one row for tiers 0-2, scattered for tier 3) and det = D."""
    name = "po_mat3_det"
    domain = "linalg"
    vars = ("x", "y", "z")
    SPEC = {0: (4, 2, "row"), 1: (6, 3, "row"), 2: (9, 4, "row"), 3: (6, 3, "spread")}

    def box(self, p):
        return [(-p["B"], p["B"])] * 3

    def matrix(self, p, t):
        m = list(p["M"])
        for pos, v in zip(p["pos"], t):
            m[pos] = v
        return m

    def det(self, m):
        a, b, c, d, e, f, g, h, i = m
        return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)

    def ok(self, p, t):
        return self.det(self.matrix(p, t)) == p["D"]

    def sample(self, rng, tier):
        B, h, kind = self.SPEC[tier]
        M = [rng.randint(-h, h) for _ in range(9)]
        if kind == "row":
            r = rng.randint(0, 2)
            pos = [3 * r, 3 * r + 1, 3 * r + 2]
        else:
            perm = rng.sample(range(3), 3)
            pos = [3 * i + perm[i] for i in range(3)]
        t = [rng.randint(-B, B) for _ in range(3)]
        p = {"B": B, "M": M, "pos": pos, "D": 0}
        p["D"] = self.det(self.matrix(p, t))
        return p

    def statement(self, p):
        B = p["B"]
        ent = [str(v) for v in p["M"]]
        for pos, n in zip(p["pos"], "xyz"):
            ent[pos] = n
        return (f"Find integers x, y, z with -{B} <= x, y, z <= {B} such that the determinant of "
                f"{mat_tex(ent, 3)} is {p['D']}.", "x, y, z")


class VecDot(PO):
    name = "po_vec_dot"
    domain = "linalg"
    SPEC = {0: (2, 1, 6), 1: (3, 2, 6), 2: (3, 2, 10), 3: (4, 2, 6)}

    def names(self, p):
        return tuple("xyzw"[:p["n"]])

    def box(self, p):
        return [(-p["B"], p["B"])] * p["n"]

    def ok(self, p, t):
        return all(sum(a * x for a, x in zip(u, t)) == s for u, s in zip(p["U"], p["s"]))

    def solve(self, p):
        return lattice(p["U"], p["s"], self.box(p))

    def sample(self, rng, tier):
        n, m, B = self.SPEC[tier]
        U = [[rng.randint(-4, 4) for _ in range(n)] for _ in range(m)]
        if any(not any(u) for u in U):
            return None
        v = [rng.randint(-B, B) for _ in range(n)]
        return {"n": n, "B": B, "U": U, "s": [sum(a * x for a, x in zip(u, v)) for u in U]}

    def statement(self, p):
        ns = self.names(p)
        v = "(" + ", ".join(ns) + ")"
        cs = " and ".join(f"v \\cdot {pt(u)} = {s}" for u, s in zip(p["U"], p["s"]))
        return f"Find integers {comma(ns)} with -{p['B']} <= {comma(ns)} <= {p['B']} such that v = {v} satisfies {cs}.", comma(ns)


class VecCross(PO):
    name = "po_vec_cross"
    domain = "linalg"
    vars = ("x", "y", "z")
    SPEC = {0: (2, 6), 1: (2, 10), 2: (3, 12), 3: (4, 20)}

    def box(self, p):
        return [(-p["B"], p["B"])] * 3

    def rows(self, p):
        a, b, c = p["u"]
        # u x v = (b z - c y, c x - a z, a y - b x)
        return [[0, -c, b], [c, 0, -a], [-b, a, 0]]

    def ok(self, p, t):
        return all(sum(r * x for r, x in zip(row, t)) == w for row, w in zip(self.rows(p), p["w"]))

    def solve(self, p):
        return lattice(self.rows(p), p["w"], self.box(p))

    def sample(self, rng, tier):
        h, B = self.SPEC[tier]
        u = [rng.randint(-h, h) for _ in range(3)]
        if not any(u):
            return None
        v = [rng.randint(-B, B) for _ in range(3)]
        p = {"u": u, "B": B, "w": [0, 0, 0]}
        p["w"] = [sum(r * x for r, x in zip(row, v)) for row in self.rows(p)]
        return p

    def statement(self, p):
        B = p["B"]
        return (f"Find integers x, y, z with -{B} <= x, y, z <= {B} such that {pt(p['u'])} \\times (x, y, z) = "
                f"{pt(p['w'])}.", "x, y, z")


# ----------------------------------------------------------------------------------------------- combinatorics
def stats_of(t):
    s = sorted(t)
    n = len(s)
    from collections import Counter
    cnt = Counter(s)
    top = max(cnt.values())
    modes = [v for v, c in cnt.items() if c == top]
    return {"sum": sum(s), "median": s[n // 2], "mode": modes[0] if len(modes) == 1 and top > 1 else None,
            "range": s[-1] - s[0], "min": s[0], "max": s[-1]}


class StatsMultiset(PO):
    name = "po_stats_multiset"
    domain = "combinatorics"
    unordered = True
    SPEC = {0: (5, 10, ["mean", "median"]), 1: (5, 12, ["mean", "median", "mode"]),
            2: (7, 12, ["mean", "median", "mode"]), 3: (7, 15, ["mean", "median", "mode", "range"])}

    def names(self, p):
        return tuple(LET[:p["n"]])

    def box(self, p):
        return [(0, p["hi"])] * p["n"]

    def ok(self, p, t):
        s = stats_of(t)
        c = p["c"]
        return (s["sum"] == c["sum"] and ("median" not in c or s["median"] == c["median"])
                and ("mode" not in c or s["mode"] == c["mode"]) and ("range" not in c or s["range"] == c["range"]))

    def solve(self, p):
        return [t for t in nondecr_sum(p["n"], 0, p["hi"], p["c"]["sum"]) if self.ok(p, t)]

    def sample(self, rng, tier):
        n, hi, kinds = self.SPEC[tier]
        while True:
            t = sorted(rng.randint(0, hi) for _ in range(n))
            s = stats_of(t)
            if s["sum"] % n == 0 and ("mode" not in kinds or s["mode"] is not None):
                break
        c = {"sum": s["sum"]}
        for k in kinds:
            if k != "mean":
                c[k] = s[k]
        return {"n": n, "hi": hi, "c": c}

    def statement(self, p):
        ns = self.names(p)
        c = p["c"]
        parts = [f"mean is {c['sum'] // p['n']}"]
        if "median" in c:
            parts.append(f"median is {c['median']}")
        if "mode" in c:
            parts.append(f"unique mode is {c['mode']}")
        if "range" in c:
            parts.append(f"range (largest minus smallest) is {c['range']}")
        order = " <= ".join(ns)
        return (f"Find integers {order}, each between 0 and {p['hi']}, whose {names_txt(parts)}.", comma(ns))


def inversions(t):
    return sum(1 for i in range(len(t)) for j in range(i + 1, len(t)) if t[i] > t[j])


def descents(t):
    return [i + 1 for i in range(len(t) - 1) if t[i] > t[i + 1]]


def cycle_type(t):
    n, seen, out = len(t), set(), []
    for i in range(1, n + 1):
        if i in seen:
            continue
        L, j = 0, i
        while j not in seen:
            seen.add(j)
            j = t[j - 1]
            L += 1
        out.append(L)
    return sorted(out, reverse=True)


class Perm(PO):
    domain = "combinatorics"
    SPEC = {0: 6, 1: 7, 2: 7, 3: 8}

    def box(self, p):
        return [(1, p["n"])] * p["n"]

    def is_perm(self, t):
        return sorted(t) == list(range(1, len(t) + 1))

    def solve(self, p):
        return [t for t in itertools.permutations(range(1, p["n"] + 1)) if self.ok(p, t)]

    def space(self, p):
        return math.factorial(p["n"])

    def answer_fmt(self, p):
        return "the values pi(1), ..., pi(n) separated by commas"

    def perm_txt(self, p):
        return f"Find a permutation pi of 1, 2, ..., {p['n']}"


class PermInversions(Perm):
    name = "po_perm_inversions"

    def ok(self, p, t):
        return (self.is_perm(t) and inversions(t) == p["inv"] and t[0] == p["first"]
                and (p.get("last") is None or t[-1] == p["last"]))

    def sample(self, rng, tier):
        n = self.SPEC[tier]
        t = list(range(1, n + 1))
        rng.shuffle(t)
        return {"n": n, "inv": inversions(t), "first": t[0], "last": t[-1] if n >= 8 else None}

    def statement(self, p):
        last = "" if p.get("last") is None else f" and pi({p['n']}) = {p['last']}"
        return (f"{self.perm_txt(p)} with exactly {p['inv']} inversions (pairs i < j with pi(i) > pi(j)), "
                f"pi(1) = {p['first']}{last}.", f"pi(1), pi(2), ..., pi({p['n']})")


class PermDescents(Perm):
    name = "po_perm_descents"

    def ok(self, p, t):
        return self.is_perm(t) and descents(t) == p["D"] and t[-1] == p["last"]

    def sample(self, rng, tier):
        n = self.SPEC[tier]
        t = list(range(1, n + 1))
        rng.shuffle(t)
        return {"n": n, "D": descents(t), "last": t[-1]}

    def statement(self, p):
        D = p["D"]
        ds = "{" + ", ".join(str(d) for d in D) + "}" if D else "empty"
        return (f"{self.perm_txt(p)} whose descent set (the positions i with pi(i) > pi(i+1)) is {ds} and with "
                f"pi({p['n']}) = {p['last']}.", f"pi(1), pi(2), ..., pi({p['n']})")


class PermCycles(Perm):
    name = "po_perm_cycles"

    def ok(self, p, t):
        return (self.is_perm(t) and cycle_type(t) == p["ct"] and t[0] == p["f1"]
                and (p.get("f2") is None or t[1] == p["f2"]))

    def sample(self, rng, tier):
        n = self.SPEC[tier]
        t = list(range(1, n + 1))
        rng.shuffle(t)
        return {"n": n, "ct": cycle_type(t), "f1": t[0], "f2": t[1] if n >= 8 else None}

    def statement(self, p):
        ct = ", ".join(str(c) for c in p["ct"])
        f2 = "" if p.get("f2") is None else f" and pi(2) = {p['f2']}"
        return (f"{self.perm_txt(p)} whose cycle lengths are {ct}, with pi(1) = {p['f1']}{f2}.",
                f"pi(1), pi(2), ..., pi({p['n']})")


class PartitionSq(PO):
    name = "po_partition_sq"
    domain = "combinatorics"
    unordered = True
    SPEC = {0: (3, 28), 1: (3, 40), 2: (4, 40), 3: (4, 60)}

    def names(self, p):
        return tuple(LET[:p["k"]])

    def box(self, p):
        return [(1, p["S"])] * p["k"]

    def ok(self, p, t):
        return len(set(t)) == len(t) and min(t) >= 1 and sum(t) == p["S"] and sum(x * x for x in t) == p["Q"]

    def solve(self, p):
        return [t for t in distinct_sum(p["k"], 1, p["S"]) if sum(x * x for x in t) == p["Q"]]

    def space(self, p):
        return sum(1 for _ in distinct_sum(p["k"], 1, p["S"]))

    def sample(self, rng, tier):
        k, S = self.SPEC[tier]
        t = sorted(rng.sample(range(1, S), k))
        return {"k": k, "S": sum(t), "Q": sum(x * x for x in t)}

    def statement(self, p):
        ns = self.names(p)
        return (f"Find distinct positive integers {comma(ns)} with {' + '.join(ns)} = {p['S']} and "
                f"{' + '.join(n + '^2' for n in ns)} = {p['Q']}.", comma(ns))


class Magic3(PO):
    """3x3 (semi-)magic square with entries in [lo, hi], line sum S and some entries fixed; tiers 2-3 need distinct
    entries."""
    name = "po_magic3"
    domain = "combinatorics"
    SPEC = {0: ("magic", 1, False, 9), 1: ("magic", 1, True, 12), 2: ("semi", 2, True, 8), 3: ("semi", 2, True, 11)}

    def box(self, p):
        return [(p["lo"], p["hi"])] * 9

    def rows(self, p):
        A, b = [], []
        lines = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8]]
        if p["kind"] == "magic":
            lines += [[0, 4, 8], [2, 4, 6]]
        for L in lines:
            A.append([1 if i in L else 0 for i in range(9)])
            b.append(p["S"])
        for i, v in p["fix"]:
            A.append([1 if j == i else 0 for j in range(9)])
            b.append(v)
        return A, b

    def ok(self, p, t):
        A, b = self.rows(p)
        if p["distinct"] and len(set(t)) < 9:
            return False
        return all(sum(a * x for a, x in zip(r, t)) == v for r, v in zip(A, b))

    def solve(self, p):
        A, b = self.rows(p)
        return [t for t in lattice(A, b, self.box(p)) if not p["distinct"] or len(set(t)) == 9]

    def sample(self, rng, tier):
        kind, nfix, distinct, span = self.SPEC[tier]
        c, x, y = rng.randint(-2, 8), rng.randint(-4, 4), rng.randint(-4, 4)
        sq = [c - y, c + x + y, c - x, c - x + y, c, c + x - y, c + x, c - x - y, c + y]
        if kind == "semi":
            sq = [sq[i] for i in rng.choice([[0, 1, 2, 3, 4, 5, 6, 7, 8], [1, 0, 2, 4, 3, 5, 7, 6, 8]])]
        lo = min(sq) - rng.randint(0, 2)
        hi = lo + span + rng.randint(0, 3)
        if max(sq) > hi:
            return None
        cand = [i for i in range(9) if not (kind == "magic" and i == 4)]
        fix = [[i, sq[i]] for i in sorted(rng.sample(cand, nfix))]
        return {"kind": kind, "S": 3 * c, "lo": lo, "hi": hi, "fix": fix, "distinct": distinct}

    def statement(self, p):
        cells = ["\\square"] * 9
        for i, v in p["fix"]:
            cells[i] = str(v)
        lines = "rows, columns and both diagonals" if p["kind"] == "magic" else "rows and columns"
        dist = " distinct" if p["distinct"] else ""
        return (f"Fill the 3x3 grid {mat_tex(cells, 3)} with{dist} integers between {p['lo']} and {p['hi']}, keeping "
                f"the given entries, so that all {lines} have sum {p['S']}.",
                "the nine entries row by row, separated by commas")


def graphical(seq):
    s = sorted(seq, reverse=True)
    if sum(s) % 2:
        return False
    n = len(s)
    for k in range(1, n + 1):
        if sum(s[:k]) > k * (k - 1) + sum(min(d, k) for d in s[k:]):
            return False
    return True


class DegreeSeq(PO):
    """Nonincreasing degree sequence of a simple graph with n vertices and m edges, degrees in [lo, hi]; tier 3: a
    tree with a given number of leaves."""
    name = "po_degree_seq"
    domain = "combinatorics"
    SPEC = {0: (8, "tree"), 1: (10, "tree"), 2: (12, "tree"), 3: (14, "tree")}

    def names(self, p):
        return tuple(LET[:p["n"]])

    def box(self, p):
        return [(p["lo"], p["hi"])] * p["n"]

    def ok(self, p, t):
        if list(t) != sorted(t, reverse=True) or sum(t) != 2 * p["m"]:
            return False
        if p.get("leaves") is not None and sum(1 for d in t if d == 1) != p["leaves"]:
            return False
        return graphical(t)

    def solve(self, p):
        out = []
        for t in itertools.combinations_with_replacement(range(p["hi"], p["lo"] - 1, -1), p["n"]):
            if self.ok(p, t):
                out.append(t)
        return out

    def space(self, p):
        return multisets(p["lo"], p["hi"], p["n"])

    def sample(self, rng, tier):
        n, kind = self.SPEC[tier]
        if kind == "tree":
            seq = [1] * n
            for _ in range(n - 2):          # Pruefer code degrees
                seq[rng.randrange(n)] += 1
            seq.sort(reverse=True)
            return {"n": n, "m": n - 1, "lo": 1, "hi": max(seq) + rng.randint(0, 2), "leaves": seq.count(1)}
        while True:
            seq = sorted((rng.randint(1, n - 1) for _ in range(n)), reverse=True)
            if graphical(seq):
                break
        return {"n": n, "m": sum(seq) // 2, "lo": max(1, min(seq) - rng.randint(0, 1)),
                "hi": min(n - 1, max(seq) + rng.randint(0, 1)), "leaves": None}

    def statement(self, p):
        ns = self.names(p)
        order = " >= ".join(ns)
        if p.get("leaves") is not None:
            return (f"Find integers {order}, each between {p['lo']} and {p['hi']}, that are the vertex degrees of a "
                    f"tree with {p['n']} vertices and exactly {p['leaves']} leaves.", comma(ns))
        return (f"Find integers {order}, each between {p['lo']} and {p['hi']}, that are the vertex degrees of a "
                f"simple graph with {p['n']} vertices and {p['m']} edges.", comma(ns))


class BinaryRuns(PO):
    name = "po_binary_runs"
    domain = "combinatorics"
    SPEC = {0: (8, False), 1: (10, False), 2: (12, True), 3: (14, True)}

    def box(self, p):
        return [(0, 1)] * p["n"]

    def stats(self, t):
        runs = 1 + sum(1 for i in range(1, len(t)) if t[i] != t[i - 1])
        pairs = sum(1 for i in range(1, len(t)) if t[i] == t[i - 1] == 1)
        return sum(t), runs, pairs

    def ok(self, p, t):
        o, r, pr = self.stats(t)
        return o == p["ones"] and r == p["runs"] and (p["pairs"] is None or pr == p["pairs"]) and t[0] == p["first"]

    def sample(self, rng, tier):
        n, use_pairs = self.SPEC[tier]
        t = [rng.randint(0, 1) for _ in range(n)]
        o, r, pr = self.stats(t)
        return {"n": n, "ones": o, "runs": r, "pairs": pr if use_pairs else None, "first": t[0]}

    def statement(self, p):
        extra = "" if p["pairs"] is None else f", exactly {p['pairs']} occurrences of 11 (counting overlaps)"
        return (f"Find a binary string b_1 b_2 ... b_{p['n']} of length {p['n']} with b_1 = {p['first']}, exactly "
                f"{p['ones']} ones, exactly {p['runs']} runs (maximal blocks of equal digits){extra}.",
                f"b_1, b_2, ..., b_{p['n']}")


class MomentWeights(PO):
    """Counts w_0..w_4 of the values 0..4 in a list of W values with given sum and sum of squares."""
    name = "po_moment_weights"
    domain = "combinatorics"
    vars = ("a", "b", "c", "d", "e")
    SPEC = {0: 8, 1: 12, 2: 20, 3: 30}

    def box(self, p):
        return [(0, p["W"])] * 5

    def rows(self, p):
        return [[1] * 5, [0, 1, 2, 3, 4], [0, 1, 4, 9, 16]], [p["W"], p["M1"], p["M2"]]

    def ok(self, p, t):
        A, b = self.rows(p)
        return all(sum(a * x for a, x in zip(r, t)) == v for r, v in zip(A, b))

    def solve(self, p):
        A, b = self.rows(p)
        return lattice(A, b, self.box(p))

    def space(self, p):
        return math.comb(p["W"] + 4, 4)

    def sample(self, rng, tier):
        W = self.SPEC[tier]
        vals = [rng.randint(0, 4) for _ in range(W)]
        return {"W": W, "M1": sum(vals), "M2": sum(v * v for v in vals)}

    def statement(self, p):
        return (f"A list of {p['W']} numbers, each equal to 0, 1, 2, 3 or 4, has sum {p['M1']} and sum of squares "
                f"{p['M2']}. Find how many times each value can occur: nonnegative integers a, b, c, d, e (the "
                f"numbers of 0s, 1s, 2s, 3s and 4s).", "a, b, c, d, e")


# ----------------------------------------------------------------------------------------------- geometry
class LatticeTriArea(PO):
    name = "po_lattice_tri_area"
    domain = "geometry"
    vars = ("x", "y")
    SPEC = {0: (3, 6), 1: (6, 9), 2: (10, 12), 3: (15, 20)}

    def box(self, p):
        return [(-p["R"], p["R"])] * 2

    def cross(self, p, t):
        (ax, ay), (bx, by) = p["A"], p["B"]
        return (bx - ax) * (t[1] - ay) - (by - ay) * (t[0] - ax)

    def ok(self, p, t):
        return abs(self.cross(p, t)) == p["S2"]

    def solve(self, p):
        (ax, ay), (bx, by) = p["A"], p["B"]
        out = []
        for sgn in (1, -1):         # (bx-ax) y - (by-ay) x = s*S2 + (bx-ax) ay - (by-ay) ax
            rhs = sgn * p["S2"] + (bx - ax) * ay - (by - ay) * ax
            out += lattice([[-(by - ay), bx - ax]], [rhs], self.box(p))
        return sorted(set(out))

    def sample(self, rng, tier):
        h, R = self.SPEC[tier]
        A = [rng.randint(-h, h), rng.randint(-h, h)]
        B = [rng.randint(-h, h), rng.randint(-h, h)]
        if A == B:
            return None
        p = {"A": A, "B": B, "R": R, "S2": 0}
        C = [rng.randint(-R, R), rng.randint(-R, R)]
        p["S2"] = abs(self.cross(p, C))
        return p if p["S2"] else None

    def statement(self, p):
        return (f"Find integers x, y with -{p['R']} <= x, y <= {p['R']} such that the triangle with vertices "
                f"A = {pt(p['A'])}, B = {pt(p['B'])} and C = (x, y) has area {fstr(F(p['S2'], 2))}.", "x, y")


class Parallelogram(PO):
    name = "po_parallelogram"
    domain = "geometry"
    SPEC = {0: (2, 5), 1: (2, 12), 2: (3, 6), 3: (3, 15)}

    def names(self, p):
        return tuple("xyz"[:p["dim"]])

    def box(self, p):
        R = 3 * max(abs(v) for P in p["P"] for v in P) + 1
        return [(-R, R)] * p["dim"]

    def ok(self, p, t):
        A, B, C = p["P"]
        cands = [tuple(a + b - c for a, b, c in zip(*tri)) for tri in ((A, B, C), (A, C, B), (B, C, A))]
        return tuple(t) in cands

    def solve(self, p):
        A, B, C = p["P"]
        return sorted({tuple(a + b - c for a, b, c in zip(*tri)) for tri in ((A, B, C), (A, C, B), (B, C, A))})

    def sample(self, rng, tier):
        dim, h = self.SPEC[tier]
        P = [[rng.randint(-h, h) for _ in range(dim)] for _ in range(3)]
        u = [b - a for a, b in zip(P[0], P[1])]
        v = [c - a for a, c in zip(P[0], P[2])]
        cr = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]] if dim == 3 else \
             [u[0] * v[1] - u[1] * v[0]]
        return {"dim": dim, "P": P} if any(cr) else None

    def statement(self, p):
        ns = self.names(p)
        A, B, C = (pt(x) for x in p["P"])
        return (f"The points {A}, {B} and {C} are three vertices of a parallelogram. Find integers {comma(ns)} such "
                f"that ({comma(ns)}) is the fourth vertex.", comma(ns))


class CircleThrough(PO):
    """x^2 + y^2 + Dx + Ey + F = 0 through two lattice points, with a real positive radius."""
    name = "po_circle_through"
    domain = "geometry"
    vars = ("D", "E", "F")
    SPEC = {0: (2, 10), 1: (3, 15), 2: (4, 20), 3: (5, 30)}

    def box(self, p):
        return [(-p["B"], p["B"])] * 3

    def ok(self, p, t):
        D, E, Fv = t
        return D * D + E * E - 4 * Fv > 0 and all(x * x + y * y + D * x + E * y + Fv == 0 for x, y in p["pts"])

    def solve(self, p):
        A = [[x, y, 1] for x, y in p["pts"]]
        b = [-(x * x + y * y) for x, y in p["pts"]]
        return [t for t in lattice(A, b, self.box(p)) if self.ok(p, t)]

    def sample(self, rng, tier):
        h, B = self.SPEC[tier]
        pts = [[rng.randint(-h, h), rng.randint(-h, h)] for _ in range(2)]
        return {"pts": pts, "B": B} if pts[0] != pts[1] else None

    def statement(self, p):
        B = p["B"]
        P1, P2 = (pt(x) for x in p["pts"])
        return (f"Find integers D, E, F with -{B} <= D, E, F <= {B} such that x^2 + y^2 + Dx + Ey + F = 0 is a circle "
                f"(of positive radius) passing through {P1} and {P2}.", "D, E, F")


class IntTriangle(PO):
    name = "po_int_triangle"
    domain = "geometry"
    vars = ("a", "b", "c")
    SPEC = {0: ((60, 100), None), 1: ((100, 160), "obtuse"), 2: ((160, 240), "acute"), 3: ((240, 400), "obtuse")}

    def box(self, p):
        return [(1, p["P"])] * 3

    def kind(self, t):
        a, b, c = sorted(t)
        s = a * a + b * b - c * c
        return "right" if s == 0 else "acute" if s > 0 else "obtuse"

    def ok(self, p, t):
        a, b, c = t
        if not (a <= b <= c and a + b > c and a + b + c == p["P"] and c - a == p["g"]):
            return False
        return p["kind"] is None or self.kind(t) == p["kind"]

    def solve(self, p):
        P, out = p["P"], []
        for a in range(1, P // 3 + 1):
            c = a + p["g"]
            b = P - a - c
            if self.ok(p, (a, b, c)):
                out.append((a, b, c))
        return out

    def space(self, p):
        return max(1, p["P"] ** 2 // 48)

    def sample(self, rng, tier):
        (lo, hi), kind = self.SPEC[tier]
        P = rng.randint(lo, hi)
        g = rng.randint(1, P // 4)
        return {"P": P, "g": g, "kind": kind}

    def statement(self, p):
        k = "" if p["kind"] is None else f" The triangle must be {p['kind']}."
        return (f"Find integers a <= b <= c that are the side lengths of a triangle with perimeter {p['P']} and "
                f"c - a = {p['g']}.{k}", "a, b, c")


# ----------------------------------------------------------------------------------------------- registry / rows
FAMILY_LIST = [
    PolyValues(), PolySlope(), PolyIntegral(), PolyMixed(), PolyCritical(), PolyRoots(), QuadRootDiff(), Vieta3(),
    LinCommute(), Mobius(), RecurCoef(), RecurInit(), APSum(), GPTerm(),
    DiffSquares(), TwoSqComp(), GcdLcm(), Egyptian(), SFFT(), Lin2Box(), CRTRange(), TotientInv(), SigmaInv(),
    TauRange(), PythLeg(), GaussFactor(), OrderMod(), FractionBetween(), LinCongruence(), SqrtMod(), ConsecutiveSum(),
    Mat2TrDet(), Mat2Involution(), Mat2Nilpotent(), Mat2Idempotent(), Mat2Eigvec(), Mat2Commute(), Mat3Det(), VecDot(),
    VecCross(),
    StatsMultiset(), PermInversions(), PermDescents(), PermCycles(), PartitionSq(), Magic3(), BinaryRuns(),
    MomentWeights(),               # DegreeSeq: too few distinct instances (answers depend only on n / leaves / max)
    LatticeTriArea(), Parallelogram(), CircleThrough(), IntTriangle(), PythPerim(),
]
FAMILIES = {f.name: f for f in FAMILY_LIST}
DOMAINS = sorted({f.domain for f in FAMILY_LIST})


def sample_params(family, seed, tier):
    return FAMILIES[family].sample(random.Random(seed), tier)


def item_stats(family, p):
    """(keys sorted, n_raw valid tuples, guess_rate, small = min over keys of max |entry|)."""
    fam = FAMILIES[family]
    raw = list(fam.solve(p))
    keys = sorted({fam.canon(p, t) for t in raw})
    small = min((max(abs(x) for x in k) for k in keys), default=None)
    return keys, len(raw), fam.guess_rate(p, len(keys)), small


def make_row(family, p, tier, split, idx, seed, keys=None, stats=None):
    fam = FAMILIES[family]
    if keys is None:
        keys, n_raw, gr, small = item_stats(family, p)
    else:
        n_raw, gr, small = stats
    problem = fam.problem(p)
    answers = [fam.fmt(p, k) for k in keys]
    return {
        "id": f"po-{family[3:]}-t{tier}-{split}-{idx:06d}", "task": f"{core.TASK_PREFIX}{family}", "source": f"{core.TASK_PREFIX}{family}",
        "split": split, "problem": problem, "prompt": f"Problem:\n{problem}\n\n<think>\n",
        "answer": answers[0] if answers else None, "ref": answers[0] if answers else None,
        "family": family, "domain": fam.domain, "difficulty": tier, "tier": tier,
        "n_valid": len(keys), "bucket": core.bucket_of(len(keys)), "guess_rate": round(gr, 6), "solvable": True,
        "holdout": False,
        "meta": {"family": family, "params": p, "seed": seed, "version": "po-v0", "tier": tier,
                 "n_raw": n_raw, "small": small, "valid_answers": answers[:core.MAX_VALID_LISTED]},
    }


def split_of(family, p, eval_pct=20):
    import zlib
    return "eval" if zlib.crc32(FAMILIES[family].param_key(p).encode()) % 100 < eval_pct else "rl"


def _register():
    core.FAMILIES.update(FAMILIES)
    core.register_task_suite()


_register()
