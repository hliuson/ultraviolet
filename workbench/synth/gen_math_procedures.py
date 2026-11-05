"""math-procedures-v1: deterministic synthetic midtrain slice of short, executable math procedures.

13 families (one sub-generator each); every displayed numeric chain / equation / identity is re-checked by an
independent evaluator (exact sympy arithmetic, numeric identity tests at sample points) and any failing
sample is dropped and counted. Rows are plain text (no <think>), SFT-compatible fields.

  python gen_math_procedures.py --out DIR --tok tokenizer.json --target 10000000 --seed 0 --procs 4
  (the tokenizer only counts tokens to decide where generation stops)
"""
import argparse, collections, hashlib, json, math, os, random, re, sys
from fractions import Fraction as Fr
import sympy as sp

# ----------------------------------------------------------------------------------------------- errors
class VerifyError(Exception):
    pass


class Retry(Exception):  # generation constraint not met (not a verification failure)
    pass


def need(c, msg="verify"):
    if not c:
        raise VerifyError(msg)


def want(c):
    if not c:
        raise Retry()

# --------------------------------------------------------------------------------------------- evaluator
# Parses the displayed plain-text math (implicit multiplication, ^, !, |..|, sqrt/cbrt/ln/sin/cos/tan/floor,
# e, π) into exact sympy values, so every printed step is checked from its text, not from generator state.
_TOK = re.compile(r"\s*(?:(\d+(?:\.\d+)?)|([A-Za-zπ]+)|(\S))")
FUN = {"sqrt": sp.sqrt, "cbrt": lambda a: sp.real_root(a, 3), "ln": sp.log, "exp": sp.exp, "sin": sp.sin,
       "cos": sp.cos, "tan": sp.tan, "floor": sp.floor}
CON = {"e": sp.E, "pi": sp.pi, "π": sp.pi}
SYMMAP = {"·": "*", "×": "*", "÷": "/", "−": "-"}


def _lex(s):
    out = []
    for m in _TOK.finditer(s):
        if m.group(1):
            out.append(("n", m.group(1)))
        elif m.group(2):
            w = m.group(2)
            if w in FUN or w in CON:
                out.append(("i", w))
            else:
                out += [("i", ch) for ch in w]
        elif m.group(3):
            out.append(("o", SYMMAP.get(m.group(3), m.group(3))))
    return out


class _P:
    def __init__(s, toks, env):
        s.t, s.i, s.env, s.ad = toks, 0, env or {}, 0

    def pk(s):
        return s.t[s.i] if s.i < len(s.t) else ("", "")

    def take(s, v=None):
        k, x = s.pk()
        if k == "" or (v is not None and x != v):
            raise VerifyError(f"parse: expected {v} got {x}")
        s.i += 1
        return x

    def isop(s, *ops):
        k, x = s.pk()
        return k == "o" and x in ops

    def expr(s):
        v = s.term()
        while s.isop("+", "-"):
            op = s.take()
            w = s.term()
            v = v + w if op == "+" else v - w
        return v

    def starts(s):
        k, x = s.pk()
        return k in ("n", "i") or (k == "o" and (x == "(" or (x == "|" and s.ad == 0)))

    def term(s):
        v = s.unary()
        while True:
            if s.isop("*", "/"):
                op = s.take()
                w = s.unary()
                if op == "/":
                    need(w != 0, "div0")
                    v = v / w
                else:
                    v = v * w
            elif s.starts():
                v = v * s.power()
            else:
                return v

    def unary(s):
        if s.isop("-"):
            s.take()
            return -s.unary()
        if s.isop("+"):
            s.take()
            return s.unary()
        return s.power()

    def power(s):
        b = s.post()
        if s.isop("^"):
            s.take()
            return b ** s.unary()
        return b

    def post(s):
        v = s.prim()
        while s.isop("!"):
            s.take()
            v = sp.factorial(v)
        return v

    def prim(s):
        k, x = s.pk()
        if k == "n":
            s.take()
            return sp.Rational(x)
        if k == "i":
            s.take()
            if x in FUN:
                return FUN[x](s.prim())
            if x in CON:
                return CON[x]
            if x in s.env:
                return s.env[x]
            raise VerifyError(f"unbound {x}")
        if x == "(":
            s.take()
            save, s.ad = s.ad, 0
            v = s.expr()
            s.take(")")
            s.ad = save
            return v
        if x == "|":
            s.take()
            s.ad += 1
            v = s.expr()
            s.take("|")
            s.ad -= 1
            return sp.Abs(v)
        raise VerifyError(f"parse: bad token {x}")


def E(s, env=None):
    toks = _lex(s)
    p = _P(toks, {k: sp.sympify(v) if not isinstance(v, Fr) else sp.Rational(v.numerator, v.denominator)
                  for k, v in (env or {}).items()})
    v = p.expr()
    need(p.i == len(toks), f"trailing tokens in {s!r}")
    return v


def eq(a, b):
    a, b = sp.sympify(a), sp.sympify(b)
    if a.is_Rational and b.is_Rational:
        return a == b
    d = complex(sp.N(a - b, 30))
    return abs(d) < 1e-10 * max(1.0, abs(complex(sp.N(a, 20))))


def chain(*segs, env=None):
    out = []
    for g in segs:
        if not out or out[-1] != g:
            out.append(g)
    vals = [E(g, env) for g in out]
    for v in vals[1:]:
        need(eq(vals[0], v), f"chain {out}")
    return " = ".join(out)


REL = re.compile(r"(≤|≥|≠|<|>|=)")


def rel_ok(s, env=None):
    parts = REL.split(s)
    vals = [E(p, env) for p in parts[0::2]]
    for a, o, b in zip(vals, parts[1::2], vals[1:]):
        if o == "=":
            ok = eq(a, b)
        elif o == "≠":
            ok = not eq(a, b)
        else:
            d = float(sp.N(a - b, 30))
            ok = {"<": d < -1e-12, ">": d > 1e-12, "≤": d <= 1e-12, "≥": d >= -1e-12}[o]
        if not ok:
            return False
    return True


def holds(s, env=None):
    need(rel_ok(s, env), f"relation fails: {s} @ {env}")
    return s


XS = sp.Symbol("x", real=True)
PTS = [sp.Rational(37, 100), sp.Rational(129, 100), sp.Rational(271, 100)]


def ident(*segs, var="x"):
    """Numerically check that displayed expressions in var agree at sample points."""
    out = []
    for g in segs:
        if not out or out[-1] != g:
            out.append(g)
    for p in PTS:
        vals = [E(g, {var: p}) for g in out]
        for v in vals[1:]:
            need(eq(vals[0], v), f"ident {out} @ {p}")
    return " = ".join(out)


def sym(s, var="x"):
    return E(s, {var: XS})

# ------------------------------------------------------------------------------------------- formatting


def F(q):
    if isinstance(q, sp.Basic):
        need(q.is_Rational, "non-rational")
        return Fr(int(q.p), int(q.q))
    return Fr(q)


def fs(q):
    q = F(q)
    return str(q.numerator) if q.denominator == 1 else f"{q.numerator}/{q.denominator}"


def atom(q):
    q = F(q)
    return fs(q) if (q >= 0 and q.denominator == 1) else f"({fs(q)})"


def join(parts):
    out = ""
    for p in parts:
        if not out:
            out = p
        elif p.startswith("-"):
            out += " - " + p[1:]
        else:
            out += " + " + p
    return out or "0"


def mono(v, e):
    return "" if e == 0 else (v if e == 1 else f"{v}^{e}")


def term(c, m):
    c = F(c)
    if m == "":
        return fs(c)
    a = abs(c)
    body = m if a == 1 else (f"{a.numerator}{m}" if a.denominator == 1 else f"({fs(a)}){m}")
    return ("-" if c < 0 else "") + body


def poly(d, v="x"):
    return join([term(d[e], mono(v, e)) for e in sorted(d, reverse=True) if d[e] != 0])


def P1(d, v="x"):
    return [(d[e], [(v, e)] if e else []) for e in sorted(d, reverse=True) if d[e] != 0]


def lin(*cv):  # lin((3,'x'),(-2,'y'),(5,''))
    return join([term(c, v) for c, v in cv if c != 0])


def sub_segs(terms, env):
    s1, vals = [], []
    for c, fac in terms:
        c = F(c)
        if c == 0:
            continue
        val = c
        for v, e in fac:
            val *= F(env[v]) ** e
        vals.append(val)
        if not fac:
            s1.append(fs(c))
            continue
        a = abs(c)
        multi = a != 1 or len(fac) > 1
        parts = []
        for v, e in fac:
            x = F(env[v])
            base = f"({fs(x)})" if (multi or x < 0 or x.denominator != 1) else fs(x)
            parts.append(base + (f"^{e}" if e > 1 else ""))
        coef = "" if a == 1 else (fs(a) if a.denominator == 1 else f"({fs(a)})")
        s1.append(("-" if c < 0 else "") + coef + "".join(parts))
    tot = sum(vals, Fr(0))
    return [join(s1), join([fs(v) for v in vals]), fs(tot)], tot


def sub(terms, env):
    segs, tot = sub_segs(terms, env)
    return chain(*segs), tot


def peval(d, x):
    return sum((Fr(c) * Fr(x) ** e for e, c in d.items()), Fr(0))


def rint(R, a, b, nonzero=False, exclude=()):
    while True:
        v = R.randint(a, b)
        if (not nonzero or v != 0) and v not in exclude:
            return v


def div_str(num, den):
    return f"{fs(num) if F(num).denominator == 1 else '(' + fs(num) + ')'}/{atom(den) if F(den) < 0 or F(den).denominator != 1 else fs(den)}"


def ex(b, e):
    e = F(e)
    return f"{b}^{e}" if (e.denominator == 1 and e >= 0) else f"{b}^({fs(e)})"


def mul_seg(x, y):
    """chain segments for x·y; a product of two multi-digit numbers is split by place value."""
    x, y = int(x), int(y)
    if min(abs(x), abs(y)) < 10:
        return [f"{x}·{atom(y)}", str(x * y)]
    big, sm = (x, y) if abs(x) >= abs(y) else (y, x)
    sg = -1 if sm < 0 else 1
    sm = abs(sm)
    parts = []
    p, place = sm, 1
    while p:
        dgt = p % 10
        if dgt:
            parts.append(dgt * place)
        p //= 10
        place *= 10
    parts = parts[::-1]
    big = big * sg
    s1 = " + ".join(f"{atom(big)}·{q}" for q in parts)
    s2 = join([str(big * q) for q in parts])
    return [f"{atom(x)}·{atom(y)}", s1, s2, str(x * y)]


def prod_lines(fs_):
    """running product a·b·c·... as one short line per multiplication"""
    out, acc = [], fs_[0]
    for f in fs_[1:]:
        out.append(chain(*mul_seg(acc, f)))
        acc *= f
    return out, acc


def sqfree(n):
    """n = k^2 m with m squarefree."""
    k, m, p = 1, n, 2
    while p * p <= m:
        while m % (p * p) == 0:
            m //= p * p
            k *= p
        p += 1
    return k, m


def rad(k, m):  # k*sqrt(m) display
    if m == 1:
        return str(k)
    return ("" if k == 1 else str(k)) + f"sqrt({m})"

# ---------------------------------------------------------------- sympy -> plain text printer (fam 3, 9-12)


def _is_atomic_base(b):
    return b.is_Symbol or (b.is_Integer and b >= 0) or isinstance(b, (sp.sin, sp.cos, sp.tan, sp.log))


def _wrap(e):
    s = pr(e)
    return f"({s})" if (e.is_Add or (e.is_Mul and len(e.args) > 1) or (e.is_Rational and not e.is_Integer)
                        or (e.is_Number and e < 0)) else s


def pr(e):
    e = sp.sympify(e)
    if e.is_Rational:
        return fs(e)
    if e is sp.E:
        return "e"
    if e is sp.pi:
        return "π"
    if e.is_Symbol:
        return e.name
    if e.is_Add:
        out = ""
        for i, t in enumerate(e.as_ordered_terms()):
            c, _ = t.as_coeff_Mul()
            neg = bool(c.is_negative)
            s = pr(-t) if neg else pr(t)
            out = (("-" if neg else "") + s) if i == 0 else out + (" - " if neg else " + ") + s
        return out
    if e.is_Mul or (e.is_Pow and e.exp.is_negative):
        c, rest = e.as_coeff_Mul()
        if c.is_negative:
            return "-" + pr(-e)
        n, d = sp.fraction(e, exact=True)
        if d != 1:
            ns = f"({pr(n)})" if n.is_Add else pr(n)
            return ns + "/" + _wrap(d) if not d.is_Pow or d.exp == sp.Rational(1, 2) else ns + "/" + pr(d)
        s = "" if c == 1 else fs(c)
        for f in (rest.as_ordered_factors() if rest != 1 else []):
            p = f"({pr(f)})" if f.is_Add else pr(f)
            if s == "":
                s = p
            elif p.startswith("("):
                s += p
            elif s[-1].isdigit() and p[0].isdigit():
                s += "·" + p
            elif re.fullmatch(r"\d+", s):
                s += p
            else:
                s += " " + p
        return s
    if e.is_Pow:
        b, x = e.base, e.exp
        if x == sp.Rational(1, 2):
            return f"sqrt({pr(b)})"
        bs = pr(b) if _is_atomic_base(b) else f"({pr(b)})"
        xs = str(x) if (x.is_Integer and x > 0) else f"({pr(x)})"
        return f"{bs}^{xs}"
    if isinstance(e, sp.exp):
        a = e.args[0]
        return "e^" + (pr(a) if (a.is_Symbol or (a.is_Integer and a > 0)) else f"({pr(a)})")
    if isinstance(e, sp.log):
        a = e.args[0]
        if isinstance(a, sp.Abs):
            return f"ln|{pr(a.args[0])}|"
        return f"ln({pr(a)})"
    if isinstance(e, sp.Abs):
        return f"|{pr(e.args[0])}|"
    if isinstance(e, (sp.sin, sp.cos, sp.tan)):
        return f"{type(e).__name__}({pr(e.args[0])})"
    raise VerifyError(f"pr: {e}")


def mulstr(*xs):
    """display a product of already-printed factors (strings); sums get parenthesised by caller"""
    s = ""
    for p in xs:
        if s == "":
            s = p
        elif p.startswith("("):
            s += p
        elif re.fullmatch(r"-?\d+", s) and not p[0].isdigit():
            s += p
        else:
            s += "·" + p
    return s


def pwb(e):
    e = sp.sympify(e)
    return pr(e) if _is_atomic_base(e) else f"({pr(e)})"


def par(e):
    s = pr(e)
    return f"({s})" if (e.is_Add or s.startswith("-")) else s

# ================================================================================================ families
# Each family function returns (problem, lines, answer). Text = lines + "Answer: ...".

# ---------------------------------------------------------------------------------- 1. substitution checks


def two_sided(L, Rv):
    if L == Rv:
        return f"{fs(L)} = {fs(Rv)} ✓", True
    return f"{fs(L)} ≠ {fs(Rv)}, so not a solution", False


def vs_const(chain_s, val, rhs):
    if val == rhs:
        return chain_s + " ✓", True
    return chain_s + f" ≠ {fs(rhs)}, so not a solution", False


def pick(R, wantsol, trues, slips, is_sol):
    slips = [s for s in slips if s is not None and not is_sol(s)]
    if wantsol or not slips:
        return R.choice(trues)
    return R.choice(slips)


def sol_answer(var, cand, ok):
    return f"{var} = {fs(cand)} is {'a solution' if ok else 'not a solution'}"


def s_lin(R, ws):
    v = R.choice("xxxxtzn")
    a = rint(R, -9, 9, True)
    c = R.choice([0, 0] + [i for i in range(-9, 10) if i not in (0, a)])
    x0 = Fr(rint(R, -10, 10))
    q = R.choice([2, 3])
    if R.random() < 0.2 and (a - c) % q == 0:
        x0 = Fr(rint(R, -10, 10), q)
    b = rint(R, -15, 15)
    d = (a - c) * x0 + b
    want(d.denominator == 1)
    Ld, Rd = {1: a, 0: b}, {1: c, 0: d}
    is_sol = lambda t: peval(Ld, t) == peval(Rd, t)
    slips = [x0 + 1, x0 - 1, -x0, Fr(d + b, a - c) if a != c else None, Fr(d - b, a + c) if a + c else None,
             Fr(b - d, a - c)]
    cand = pick(R, ws, [x0], slips, is_sol)
    env = {v: cand}
    Ls, L = sub(P1(Ld, v), env)
    lines = [f"Substitute {v} = {fs(cand)}."]
    if c == 0:
        s, ok = vs_const("Left side: " + Ls, L, d)
        lines.append(s)
    else:
        Rs, Rv = sub(P1(Rd, v), env)
        lines += [f"Left side: {Ls}", f"Right side: {Rs}"]
        s, ok = two_sided(L, Rv)
        lines.append(s)
    need(ok == is_sol(cand))
    pb = R.choice(["Check whether {c} is a solution of {e}.", "Is {c} a solution of {e}?",
                   "Verify by substitution whether {c} satisfies {e}."])
    return pb.format(c=f"{v} = {fs(cand)}", e=f"{poly(Ld, v)} = {poly(Rd, v)}"), lines, sol_answer(v, cand, ok)


def s_poly(R, ws, deg):
    roots = [rint(R, -8, 8) for _ in range(deg)]
    k = R.choice([1, 1, 1, -1, 2, 3]) if deg == 2 else R.choice([1, 1, -1, 2])
    p = sp.Poly(k * sp.prod([XS - r for r in roots]), XS)
    co = p.all_coeffs()[::-1]
    d = {e: int(co[e]) for e in range(len(co))}
    want(max(abs(x) for x in d.values()) <= 200)
    rhs = 0
    if R.random() < 0.3 and d[0] != 0:
        rhs = -d[0]
        d[0] = 0
    is_sol = lambda t: peval(d, t) == rhs
    slips = [-r for r in roots] + [r + 1 for r in roots] + [r - 1 for r in roots]
    cand = Fr(pick(R, ws, roots, slips, is_sol))
    Ls, L = sub(P1(d, "x"), {"x": cand})
    s, ok = vs_const(Ls, L, rhs)
    need(ok == is_sol(cand))
    lines = [f"Substitute x = {fs(cand)}.", s]
    return (f"Check whether x = {fs(cand)} is a root of {poly(d)} = {rhs}.", lines, sol_answer("x", cand, ok))


VPAIRS = [("x", "y")] * 3 + [("a", "b"), ("u", "v"), ("p", "q"), ("m", "n"), ("s", "t")]


def s_sys(R, ws):
    vx, vy = R.choice(VPAIRS)
    x0, y0 = rint(R, -7, 7), rint(R, -7, 7)
    a1, b1, a2, b2 = (rint(R, -7, 7, True) for _ in range(4))
    want(a1 * b2 - a2 * b1 != 0)
    c1, c2 = a1 * x0 + b1 * y0, a2 * x0 + b2 * y0
    g = math.gcd(a1, b1)
    is_sol = lambda t: a1 * t[0] + b1 * t[1] == c1 and a2 * t[0] + b2 * t[1] == c2
    slips = [(y0, x0), (x0, y0 + 1), (x0 + 1, y0), (-x0, y0), (x0, -y0), (x0 + b1 // g, y0 - a1 // g)]
    slips = [s for s in slips if not is_sol(s)]
    cand = (x0, y0) if (ws or not slips) else R.choice(slips)
    env = {vx: cand[0], vy: cand[1]}
    lines = [f"Substitute {vx} = {cand[0]}, {vy} = {cand[1]}."]
    ok = True
    for i, (a, b, c) in enumerate([(a1, b1, c1), (a2, b2, c2)], 1):
        s, val = sub([(a, [(vx, 1)]), (b, [(vy, 1)])], env)
        s, oki = vs_const(f"Equation {i}: " + s, val, c)
        lines.append(s)
        if not oki:
            ok = False
            break
    need(ok == is_sol(cand))
    e1, e2 = f"{lin((a1, vx), (b1, vy))} = {c1}", f"{lin((a2, vx), (b2, vy))} = {c2}"
    ans = f"({vx}, {vy}) = ({cand[0]}, {cand[1]}) is {'a solution' if ok else 'not a solution'}"
    return f"Check whether ({vx}, {vy}) = ({cand[0]}, {cand[1]}) solves the system {e1}, {e2}.", lines, ans


def s_ineq(R, ws):
    a = rint(R, -6, 6, True)
    b = rint(R, -12, 12)
    xb = rint(R, -8, 8)
    c = a * xb + b
    op = R.choice(["<", "≤", ">", "≥"])
    d = {1: a, 0: b}
    test = lambda t: rel_ok(f"{fs(peval(d, t))} {op} {c}")
    cands = [xb, xb, xb + 1, xb - 1, xb + 2, xb - 2, xb + 3, xb - 3]
    good = [t for t in cands if test(t)]
    bad = [t for t in cands if not test(t)]
    cand = Fr(R.choice(good if (ws and good) or not bad else bad))
    s, val = sub(P1(d), {"x": cand})
    ok = rel_ok(f"{fs(val)} {op} {c}")
    need(ok == test(cand))
    lines = [f"Substitute x = {fs(cand)}.", f"Left side: {s}",
             f"{fs(val)} {op} {c} ✓" if ok else f"{fs(val)} {op} {c} is false, so not a solution"]
    return f"Does x = {fs(cand)} satisfy {poly(d)} {op} {c}?", lines, sol_answer("x", cand, ok)


def s_cineq(R, ws):
    a = rint(R, -5, 5, True)
    b = rint(R, -10, 10)
    lo, hi = sorted([rint(R, -6, 6), rint(R, -6, 6)])
    want(hi - lo >= 1)
    o1, o2 = R.choice(["<", "≤"]), R.choice(["<", "≤"])
    L, H = min(a * lo + b, a * hi + b), max(a * lo + b, a * hi + b)
    d = {1: a, 0: b}
    test = lambda t: rel_ok(f"{L} {o1} {fs(peval(d, t))} {o2} {H}")
    cands = list(range(lo - 2, hi + 3))
    good = [t for t in cands if test(t)]
    bad = [t for t in cands if not test(t)]
    cand = Fr(R.choice(good if (ws and good) or not bad else bad))
    s, val = sub(P1(d), {"x": cand})
    ok = rel_ok(f"{L} {o1} {fs(val)} {o2} {H}")
    need(ok == test(cand))
    rel = f"{L} {o1} {fs(val)} {o2} {H}"
    lines = [f"Substitute x = {fs(cand)}.", f"Middle: {s}", rel + " ✓" if ok else rel + " is false, so not a solution"]
    return f"Does x = {fs(cand)} satisfy {L} {o1} {poly(d)} {o2} {H}?", lines, sol_answer("x", cand, ok)


def s_abs(R, ws):
    a = rint(R, -5, 5, True)
    b = rint(R, -10, 10)
    x0 = rint(R, -8, 8)
    c = abs(a * x0 + b)
    want(c > 0)
    x1 = Fr(-c - b, a) if a * x0 + b > 0 else Fr(c - b, a)
    d = {1: a, 0: b}
    is_sol = lambda t: abs(peval(d, t)) == c
    trues = [Fr(x0)] + ([x1] if x1.denominator == 1 else [])
    cand = Fr(pick(R, ws, trues, [Fr(c + b, a), -Fr(x0), Fr(x0 + 1), Fr(x0 - 1), -x1], is_sol))
    segs, val = sub_segs(P1(d), {"x": cand})
    s = chain(*[f"|{g}|" for g in segs], fs(abs(val)))
    s, ok = vs_const(s, abs(val), c)
    need(ok == is_sol(cand))
    return (f"Check whether x = {fs(cand)} is a solution of |{poly(d)}| = {c}.",
            [f"Substitute x = {fs(cand)}.", f"Left side: {s}"], sol_answer("x", cand, ok))


def s_rad(R, ws):
    t = rint(R, 0, 7)
    x0 = rint(R, -5, 12)
    bb = x0 - t
    aa = t * t - x0
    x1 = 2 * bb + 1 - x0  # other root of x + aa = (x - bb)^2
    is_sol = lambda u: u + aa >= 0 and sp.sqrt(u + aa) == u - bb
    slips = [x1] + [tt * tt - aa for tt in range(0, 9) if tt != t]
    cand = Fr(pick(R, ws, [x0], slips, is_sol))
    want(cand + aa >= 0 and sp.sqrt(cand + aa).is_Integer)
    segs, val = sub_segs(P1({1: 1, 0: aa}), {"x": cand})
    L = sp.sqrt(val)
    Ls = chain(*[f"sqrt({g})" for g in segs], pr(L))
    Rs, Rv = sub(P1({1: 1, 0: -bb}), {"x": cand})
    s, ok = two_sided(F(L), Rv)
    need(ok == is_sol(cand))
    lines = [f"Substitute x = {fs(cand)}.", f"Left side: {Ls}", f"Right side: {Rs}", s]
    return (f"Check whether x = {fs(cand)} is a solution of sqrt({poly({1: 1, 0: aa})}) = {poly({1: 1, 0: -bb})}.",
            lines, sol_answer("x", cand, ok))


def fam_subst(R):
    k = R.choice(["lin"] * 3 + ["quad"] * 2 + ["cubic", "sys", "sys", "ineq", "ineq", "cineq", "abs", "rad"])
    ws = R.random() < 0.5
    if k == "lin":
        return s_lin(R, ws)
    if k == "quad":
        return s_poly(R, ws, 2)
    if k == "cubic":
        return s_poly(R, ws, 3)
    return {"sys": s_sys, "ineq": s_ineq, "cineq": s_cineq, "abs": s_abs, "rad": s_rad}[k](R, ws)

# ------------------------------------------------------------------------------------- 2. equation solving


def lin_solve_lines(a, b, c, d, v):
    """a v + b = c v + d  ->  lines, solution (each line verified at the solution)."""
    want(a != c)
    x0 = Fr(d - b, a - c)
    env = {v: x0}
    lines = []
    if c != 0 or b != 0:
        lhs = join([term(a, v)] + ([term(-c, v)] if c else []))
        rhs = join([fs(d)] + ([fs(-b)] if b else []))
        lines.append(f"{lhs} = {rhs}")
    lines.append(f"{term(a - c, v)} = {fs(d - b)}")
    if a - c == -1:
        lines.append(f"{v} = {fs(x0)}")
    elif a - c != 1:
        lines.append(f"{v} = " + chain(div_str(d - b, a - c), fs(x0)))
    out = []
    for ln in lines:
        if not out or out[-1] != ln:
            out.append(holds(ln, env))
    return out, x0


def e_lin(R):
    v = R.choice("xxxxtzny")
    a = rint(R, -9, 9, True)
    c = R.choice([0] + [i for i in range(-9, 10) if i not in (0, a)])
    form = R.random()
    lines = []
    if form < 0.65:
        b, d = rint(R, -15, 15), rint(R, -15, 15)
        stmt = f"{poly({1: a, 0: b}, v)} = {poly({1: c, 0: d}, v)}"
        Lt, Rt = P1({1: a, 0: b}, v), P1({1: c, 0: d}, v)
    else:
        k, p = rint(R, -6, 6, True, (1,)), rint(R, -9, 9, True)
        d = rint(R, -15, 15)
        a, b = k, k * p
        want(a != c)
        stmt = f"{term(k, '(' + poly({1: 1, 0: p}, v) + ')')} = {poly({1: c, 0: d}, v)}"
        lines.append(f"Expand: {poly({1: a, 0: b}, v)} = {poly({1: c, 0: d}, v)}")
        Lt, Rt = None, P1({1: c, 0: d}, v)
    sl, x0 = lin_solve_lines(a, b, c, d, v)
    if lines:
        holds(lines[0].split(": ", 1)[1], {v: x0})
    holds(stmt, {v: x0})
    lines += sl
    env = {v: x0}
    if Lt is None:
        inner_segs, iv = sub_segs(P1({1: 1, 0: p}, v), env)
        ls = chain(*[f"{fs(k)}({g})" for g in inner_segs], fs(k * iv))
        Lv = k * iv
    else:
        ls, Lv = sub(Lt, env)
    rs, Rv = sub(Rt, env)
    need(Lv == Rv)
    lines.append(f"Check: {ls}; {rs} ✓")
    return f"Solve {stmt}.", lines, f"{v} = {fs(x0)}"


def e_sys(R):
    vx, vy = R.choice(VPAIRS)
    x0, y0 = rint(R, -7, 7), rint(R, -7, 7)
    a1, b1, a2, b2 = (rint(R, -7, 7, True) for _ in range(4))
    det = a1 * b2 - a2 * b1
    want(det != 0)
    c1, c2 = a1 * x0 + b1 * y0, a2 * x0 + b2 * y0
    env = {vx: x0, vy: y0}
    eqs = lambda a, b, c: f"{lin((a, vx), (b, vy))} = {c}"
    L = abs(b1) * abs(b2) // math.gcd(b1, b2)
    m1, m2 = L // abs(b1), L // abs(b2)
    lines = []
    mult = [f"({i}) by {m}" for i, m in ((1, m1), (2, m2)) if m != 1]
    lines.append(f"Eliminate {vy}" + (": multiply " + " and ".join(mult) + "." if mult else "."))
    A1, B1, C1, A2, B2, C2 = a1 * m1, b1 * m1, c1 * m1, a2 * m2, b2 * m2, c2 * m2
    if m1 != 1:
        lines.append(holds(eqs(A1, B1, C1), env))
    if m2 != 1:
        lines.append(holds(eqs(A2, B2, C2), env))
    if B1 == B2:
        A, C, op = A1 - A2, C1 - C2, "Subtract"
    else:
        A, C, op = A1 + A2, C1 + C2, "Add"
    need(A != 0)
    lines.append(f"{op}: " + holds(f"{term(A, vx)} = {C}", env))
    if A == -1:
        lines.append(f"{vx} = {x0}")
    elif A != 1:
        lines.append(f"{vx} = " + chain(div_str(C, A), fs(x0)))
    s1, _ = sub_segs([(a1, [(vx, 1)])], {vx: x0})
    lhs = join([s1[0], term(b1, vy)])
    lines.append(f"Substitute into (1): " + holds(f"{lhs} = {c1}", env))
    lines.append(holds(f"{term(b1, vy)} = " + chain(join([fs(c1), fs(-a1 * x0)]), fs(c1 - a1 * x0)), env))
    if b1 == -1:
        lines.append(f"{vy} = {y0}")
    elif b1 != 1:
        lines.append(f"{vy} = " + chain(div_str(c1 - a1 * x0, b1), fs(y0)))
    s, val = sub([(a2, [(vx, 1)]), (b2, [(vy, 1)])], env)
    need(val == c2)
    lines.append(f"Check (2): {s} ✓")
    return (f"Solve the system (1) {eqs(a1, b1, c1)}, (2) {eqs(a2, b2, c2)}.", lines, f"{vx} = {x0}, {vy} = {y0}")


def fac(r, v="x"):
    return v if r == 0 else f"({poly({1: 1, 0: -r}, v)})"


def e_factor(R):
    r, s = sorted([rint(R, -9, 9), rint(R, -9, 9)])
    k = R.choice([1, 1, 1, 1, 2, 3, -1])
    B, C = -(r + s), r * s
    d = {2: k, 1: k * B, 0: k * C}
    lines = []
    if k != 1:
        lines.append(f"Divide by {atom(k)}: {poly({2: 1, 1: B, 0: C})} = 0")
    if C == 0:
        nz_ = s if r == 0 else r
        lines.append(f"Factor out x: x{fac(nz_)} = 0")
    else:
        lines.append(f"Two numbers with product {C} and sum {B}: {-r} and {-s}.")
        need((-r) * (-s) == C and (-r) + (-s) == B)
        f2 = f"{fac(r)}{fac(s)}" if r != s else f"{fac(r)}^2"
        lines.append(ident(poly({2: 1, 1: B, 0: C}), f2) + " = 0" if False else f"{f2} = 0")
        ident(poly({2: 1, 1: B, 0: C}), f2)
    if C == 0:
        ident(poly({2: 1, 1: B, 0: C}), f"x{fac(s if r == 0 else r)}")
    if r == s:
        lines.append(f"x = {r} (double root)")
        roots = [r]
    else:
        lines.append(f"x = {r} or x = {s}")
        roots = [r, s]
    for t in roots:
        cs, val = sub(P1(d), {"x": t})
        need(val == 0)
        lines.append(f"Check x = {t}: {cs} ✓")
    ans = f"x = {r}" if r == s else f"x = {r} or x = {s}"
    return f"Solve {poly(d)} = 0 by factoring.", lines, ans


def e_formula(R):
    a = rint(R, 1, 5)
    b, c = rint(R, -11, 11), rint(R, -10, 10, True)
    D = b * b - 4 * a * c
    d = {2: a, 1: b, 0: c}
    lines = [f"a = {a}, b = {b}, c = {c}"]
    seg1 = f"{atom(b)}^2 - 4({a})({c})"
    lines.append("D = b^2 - 4ac = " + chain(seg1, join([fs(b * b), fs(-4 * a * c)]), fs(D)))
    if D < 0:
        lines.append("D < 0, so there are no real solutions.")
        return f"Solve {poly(d)} = 0 with the quadratic formula.", lines, "no real solutions"
    sD = math.isqrt(D)
    if D == 0:
        x0 = Fr(-b, 2 * a)
        lines.append("D = 0, so x = -b/(2a) = " + chain(f"{fs(-b)}/{2 * a}", fs(x0)))
        cs, val = sub(P1(d), {"x": x0})
        need(val == 0)
        lines.append(f"Check: {cs} ✓")
        return f"Solve {poly(d)} = 0 with the quadratic formula.", lines, f"x = {fs(x0)}"
    if sD * sD == D:
        lines.append(f"sqrt(D) = {chain(f'sqrt({D})', str(sD))}")
        lines.append(f"x = ({fs(-b)} ± {sD})/{2 * a}")
        r1, r2 = Fr(-b + sD, 2 * a), Fr(-b - sD, 2 * a)
        lines.append("x = " + chain(f"({fs(-b)} + {sD})/{2 * a}", f"{fs(-b + sD)}/{2 * a}", fs(r1)) + " or x = "
                     + chain(f"({fs(-b)} - {sD})/{2 * a}", f"{fs(-b - sD)}/{2 * a}", fs(r2)))
        for t in (r1, r2):
            cs, val = sub(P1(d), {"x": t})
            need(val == 0)
            lines.append(f"Check x = {fs(t)}: {cs} ✓")
        return f"Solve {poly(d)} = 0 with the quadratic formula.", lines, f"x = {fs(r1)} or x = {fs(r2)}"
    k, m = sqfree(D)
    if k > 1:
        lines.append(f"sqrt(D) = " + chain(f"sqrt({D})", f"sqrt({k * k}·{m})", rad(k, m)))
    lines.append(f"x = ({fs(-b)} ± {rad(k, m)})/{2 * a}")
    g = math.gcd(math.gcd(abs(b), k), 2 * a)
    p, q, den = -b // g, k // g, 2 * a // g
    if g > 1:
        lines.append(f"Divide by {g}: x = " + (f"({p} ± {rad(q, m)})/{den}" if den != 1 else f"{p} ± {rad(q, m)}"))
    root = lambda sg: (f"({p} {sg} {rad(q, m)})/{den}" if den != 1 else f"{p} {sg} {rad(q, m)}")
    for sg in "+-":
        need(eq(E(poly(d), {"x": E(root(sg))}), 0), "formula root")
    ssum = chain(f"2({p})/{den}", fs(Fr(2 * p, den)))
    sprod = chain(f"({atom(p)}^2 - {q}^2·{m})/{den}^2", fs(Fr(p * p - q * q * m, den * den)))
    need(Fr(2 * p, den) == Fr(-b, a) and Fr(p * p - q * q * m, den * den) == Fr(c, a))
    lines.append(f"Check: sum of roots = {ssum} = -b/a ✓; product = {sprod} = c/a ✓")
    rs = f"({p} ± {rad(q, m)})/{den}" if den != 1 else f"{p} ± {rad(q, m)}"
    return f"Solve {poly(d)} = 0 with the quadratic formula.", lines, f"x = {rs}"


def e_abs(R):
    a = rint(R, -5, 5, True)
    b = rint(R, -10, 10)
    x0 = rint(R, -9, 9)
    c = abs(a * x0 + b)
    if R.random() < 0.12:
        c = -rint(R, 1, 9)
    k = rint(R, -9, 9) if R.random() < 0.35 else 0
    d = {1: a, 0: b}
    inner = poly(d)
    stmt = f"|{inner}| = {c}" if k == 0 else f"|{inner}| {'+' if k > 0 else '-'} {abs(k)} = {c + k}"
    lines = []
    if k:
        lines.append(f"Isolate: |{inner}| = " + chain(join([fs(c + k), fs(-k)]), fs(c)))
    if c < 0:
        lines.append(f"|{inner}| ≥ 0 can never equal {c}, so there is no solution.")
        return f"Solve {stmt}.", lines, "no solution"
    if c == 0:
        lines.append(f"{inner} = 0")
        sl, r = lin_solve_lines(a, b, 0, 0, "x")
        lines += sl[1:] if len(sl) > 1 else sl
        roots = [r]
    else:
        lines.append(f"{inner} = {c} or {inner} = {-c}")
        ra, rb = Fr(c - b, a), Fr(-c - b, a)
        lines.append(f"{term(a, 'x')} = {fs(c - b)} or {term(a, 'x')} = {fs(-c - b)}")
        holds(f"{term(a, 'x')} = {fs(c - b)}", {"x": ra})
        holds(f"{term(a, 'x')} = {fs(-c - b)}", {"x": rb})
        if a != 1:
            lines.append(f"x = {chain(div_str(c - b, a), fs(ra))} or x = {chain(div_str(-c - b, a), fs(rb))}")
        roots = [ra, rb]
    for t in roots:
        segs, val = sub_segs(P1(d), {"x": t})
        s = chain(*[f"|{g}|" for g in segs], fs(abs(val)))
        need(abs(val) == c)
        lines.append(f"Check x = {fs(t)}: {s} ✓")
    return f"Solve {stmt}.", lines, " or ".join(f"x = {fs(t)}" for t in roots)


def fam_solve(R):
    k = R.choice(["lin"] * 3 + ["sys"] * 3 + ["factor"] * 2 + ["formula"] * 2 + ["abs"] * 2)
    return {"lin": e_lin, "sys": e_sys, "factor": e_factor, "formula": e_formula, "abs": e_abs}[k](R)

# ------------------------------------------------------------------------------------ 3. functions


def rpoly(R, deg, lo=-6, hi=6):
    d = {e: rint(R, lo, hi) for e in range(deg + 1)}
    d[deg] = rint(R, lo, hi, True)
    return d


def f_eval(R):
    deg = R.choice([1, 2, 2, 3])
    d = rpoly(R, deg, -5, 5) if deg < 3 else {3: rint(R, -2, 2, True), 1: rint(R, -5, 5), 0: rint(R, -5, 5)}
    pts = sorted({rint(R, -5, 5) if deg < 3 else rint(R, -3, 3) for _ in range(R.choice([1, 2]))})
    if R.random() < 0.15:
        pts = [Fr(rint(R, -5, 5, True), 2)]
    lines = []
    for t in pts:
        s, _ = sub(P1(d), {"x": t})
        lines.append(f"f({fs(t)}) = {s}")
    ans = ", ".join(f"f({fs(t)}) = {fs(peval(d, t))}" for t in pts)
    return f"Let f(x) = {poly(d)}. Find {' and '.join(f'f({fs(t)})' for t in pts)}.", lines, ans


def f_rat_eval(R):
    a, b, c, d = rint(R, -6, 6, True), rint(R, -9, 9), rint(R, -4, 4, True), rint(R, -9, 9)
    want(a * d - b * c != 0)
    t = rint(R, -6, 6)
    want(c * t + d != 0)
    ns, nv = sub(P1({1: a, 0: b}), {"x": t})
    ds, dv = sub(P1({1: c, 0: d}), {"x": t})
    val = Fr(nv, dv)
    lines = [f"Numerator: {ns}", f"Denominator: {ds}", f"f({t}) = " + chain(div_str(nv, dv), fs(val))]
    return (f"Let f(x) = ({poly({1: a, 0: b})})/({poly({1: c, 0: d})}). Find f({t}).", lines, f"f({t}) = {fs(val)}")


def f_comp(R):
    names = R.choice([("f", "g"), ("f", "g"), ("g", "h"), ("p", "q")])
    fd = rpoly(R, R.choice([1, 2]), -5, 5)
    gd = rpoly(R, R.choice([1, 1, 2]), -5, 5)
    t = rint(R, -4, 4)
    outer_first = R.random() < 0.5
    (o, od), (i, idd) = ((names[0], fd), (names[1], gd)) if outer_first else ((names[1], gd), (names[0], fd))
    s1, v1 = sub(P1(idd), {"x": t})
    want(abs(v1) <= 15)
    s2, v2 = sub(P1(od), {"x": v1})
    want(abs(v2) <= 600)
    lines = [f"Inside first: {i}({t}) = {s1}", f"Then {o}({i}({t})) = {o}({fs(v1)}) = {s2}"]
    return (f"Let {names[0]}(x) = {poly(fd)} and {names[1]}(x) = {poly(gd)}. Find {o}({i}({t})).", lines,
            f"{o}({i}({t})) = {fs(v2)}")


def f_comp_sym(R):
    fd = rpoly(R, R.choice([1, 2]), -5, 5)
    gd = rpoly(R, 1, -5, 5) if 2 in fd else rpoly(R, R.choice([1, 2]), -5, 5)
    gs = poly(gd)
    G = sym(gs)
    parts, ex = [], []
    for e in sorted(fd, reverse=True):
        c = fd[e]
        if c == 0:
            continue
        if e == 0:
            parts.append(fs(c))
            ex.append(fs(c))
        else:
            base = f"({gs})" + (f"^{e}" if e > 1 else "")
            parts.append(("-" if c < 0 else "") + ("" if abs(c) == 1 else str(abs(c))) + base)
            ex.append(pr(sp.expand(c * G ** e)))
    res = pr(sp.expand(sym(poly(fd)).subs(XS, G)))
    ex_s = join(ex)
    lines = ["f(g(x)) = " + ident(join(parts), f"({ex_s})" if False else ex_s, res)]
    return f"Let f(x) = {poly(fd)} and g(x) = {gs}. Find f(g(x)).", lines, f"f(g(x)) = {res}"


def f_inv_lin(R):
    a = rint(R, -7, 7, True, (1,))
    b = rint(R, -12, 12, True)
    n = rint(R, -6, 6)
    k = a * n + b
    lines = [f"y = {poly({1: a, 0: b})}", f"{poly({1: 1, 0: -b}, 'y')} = {term(a, 'x')}"]
    if a > 0:
        inv_y, inv_x = f"({poly({1: 1, 0: -b}, 'y')})/{a}", f"({poly({1: 1, 0: -b})})/{a}"
    else:
        inv_y, inv_x = f"({poly({1: -1, 0: b}, 'y')})/{-a}", f"({poly({1: -1, 0: b})})/{-a}"
    lines.append(f"x = {inv_y}")
    lines.append(f"f^(-1)(x) = {inv_x}")
    for xs in PTS:  # each line holds on the graph y = f(x)
        env = {"x": xs, "y": a * xs + b}
        for ln in lines[:3]:
            holds(ln, env)
        need(eq(E(inv_x, {"x": a * xs + b}), xs))
    num = f"{atom(k)} - {atom(b)}" if a > 0 else f"{atom(b)} - {atom(k)}"
    dd = a if a > 0 else -a
    lines.append(f"f^(-1)({k}) = " + chain(f"({num})/{dd}", div_str((k - b) if a > 0 else (b - k), dd), str(n)))
    cs, cv = sub(P1({1: a, 0: b}), {"x": n})
    need(cv == k)
    lines.append(f"Check: f({n}) = {cs} ✓")
    return (f"Let f(x) = {poly({1: a, 0: b})}. Find f^(-1)(x) and f^(-1)({k}).", lines,
            f"f^(-1)(x) = {inv_x}, f^(-1)({k}) = {n}")


def f_inv_cube(R):
    a = R.choice([1, 1, 2, -1, 3])
    b = rint(R, -10, 10, True)
    n = rint(R, -3, 3)
    k = a * n ** 3 + b
    yb = poly({1: 1, 0: -b}, "y")
    lines = [f"y = {poly({3: a, 0: b})}", f"{yb} = {term(a, 'x^3')}"]
    if a == 1:
        lines.append(f"x = cbrt({yb})")
        inv = f"cbrt({poly({1: 1, 0: -b})})"
    else:
        lines.append(f"x^3 = ({yb})/{atom(a)}")
        lines.append(f"x = cbrt(({yb})/{atom(a)})")
        inv = f"cbrt(({poly({1: 1, 0: -b})})/{atom(a)})"
    lines.append(f"f^(-1)(x) = {inv}")
    for xs in PTS:
        env = {"x": xs, "y": a * xs ** 3 + b}
        for ln in lines[:-1]:
            holds(ln, env)
        need(eq(E(inv, {"x": a * xs ** 3 + b}), xs))
    inner = f"({atom(k)} - {atom(b)})" + ("" if a == 1 else f"/{atom(a)}")
    lines.append(f"f^(-1)({k}) = " + chain(f"cbrt({inner})", f"cbrt({fs(Fr(k - b, a))})", str(n)))
    cs, cv = sub(P1({3: a, 0: b}), {"x": n})
    need(cv == k)
    lines.append(f"Check: f({n}) = {cs} ✓")
    return (f"Let f(x) = {poly({3: a, 0: b})}. Find f^(-1)(x) and f^(-1)({k}).", lines,
            f"f^(-1)(x) = {inv}, f^(-1)({k}) = {n}")


def f_inv_rat(R):
    a, b, c, d = rint(R, -5, 5, True), rint(R, -7, 7), rint(R, -3, 3, True), rint(R, -7, 7)
    want(a * d - b * c != 0)
    n = rint(R, -5, 5)
    want(c * n + d != 0)
    k = Fr(a * n + b, c * n + d)
    want(k.denominator == 1 and c * k - a != 0)
    num, den = poly({1: a, 0: b}), poly({1: c, 0: d})
    lines = [f"y = ({num})/({den})", f"y({den}) = {num}",
             f"{lin((c, 'xy'), (d, 'y'))} = {num}",
             f"{lin((c, 'xy'), (-a, 'x'))} = {lin((-d, 'y'), (b, ''))}",
             f"x({poly({1: c, 0: -a}, 'y')}) = {poly({1: -d, 0: b}, 'y')}",
             f"x = ({poly({1: -d, 0: b}, 'y')})/({poly({1: c, 0: -a}, 'y')})"]
    inv = f"({poly({1: -d, 0: b})})/({poly({1: c, 0: -a})})"
    lines.append(f"f^(-1)(x) = {inv}")
    for xs in PTS:
        env = {"x": xs, "y": (a * xs + b) / (c * xs + d)}
        for ln in lines[:-1]:
            holds(ln, env)
        need(eq(E(inv, {"x": env["y"]}), xs))
    ns, nv = sub(P1({1: -d, 0: b}), {"x": k})
    ds, dv = sub(P1({1: c, 0: -a}), {"x": k})
    need(Fr(nv, dv) == n)
    lines.append(f"f^(-1)({fs(k)}): numerator {ns}, denominator {ds}, so f^(-1)({fs(k)}) = "
                 + chain(div_str(nv, dv), str(n)))
    lines.append(f"Check: f({n}) = " + chain(f"({sub_segs(P1({1: a, 0: b}), {'x': n})[0][0]})/"
                                             f"({sub_segs(P1({1: c, 0: d}), {'x': n})[0][0]})", fs(k)) + " ✓")
    return (f"Let f(x) = ({num})/({den}). Find f^(-1)(x) and f^(-1)({fs(k)}).", lines,
            f"f^(-1)(x) = {inv}, f^(-1)({fs(k)}) = {n}")


def fam_func(R):
    k = R.choice(["eval", "rat", "comp", "comp", "csym", "csym", "inv", "inv", "cube", "irat"])
    return {"eval": f_eval, "rat": f_rat_eval, "comp": f_comp, "csym": f_comp_sym, "inv": f_inv_lin,
            "cube": f_inv_cube, "irat": f_inv_rat}[k](R)

# --------------------------------------------------------------------------------------------- 4. counting
LET = "ABCDEGHKMNPRST"


def c_multiset(R):
    k = R.choice([2, 3, 3, 4])
    cnt = [rint(R, 1, 4) for _ in range(k)]
    want(max(cnt) >= 2 and sum(cnt) <= 9)
    n = sum(cnt)
    if R.random() < 0.6:
        letters = R.sample(LET, k)
        word = "".join(l * c for l, c in zip(letters, cnt))
        items = list(word)
        R.shuffle(items)
        pb = f"Count the distinct arrangements of the letters of {''.join(items)}."
        desc = ", ".join(f"{l}: {c}" for l, c in zip(letters, cnt))
    else:
        letters = [str(x) for x in R.sample(range(1, 10), k)]
        items = [l for l, c in zip(letters, cnt) for _ in range(c)]
        R.shuffle(items)
        pb = f"Count the distinct permutations of the digits {', '.join(items)}."
        desc = ", ".join(f"{l} appears {c}" for l, c in zip(letters, cnt))
    rep = sorted([c for c in cnt if c > 1], reverse=True)
    den = "·".join(f"{c}!" for c in rep)
    dd = f"({den})" if len(rep) > 1 else den
    cm, rest = rep[0], rep[1:]
    dprod = math.prod(math.factorial(c) for c in rest)
    res = math.factorial(n) // math.factorial(cm) // dprod
    need(res == len(set(__import__("itertools").permutations(items))) if n <= 8 else True)
    top = list(range(n, cm, -1))
    lines = [f"n = {n}; {desc}.", f"Arrangements = {n}!/{dd}"]
    lines.append(f"Cancel {cm}!: {n}!/{cm}! = " + "·".join(map(str, top)))
    pl, tv = prod_lines(top) if len(top) > 1 else ([], top[0])
    lines += pl
    if rest:
        rd = "·".join(f"{c}!" for c in rest)
        rv = "·".join(str(math.factorial(c)) for c in rest)
        lines.append(f"Divide by {rd}: " + chain(f"{tv}/({rv})" if len(rest) > 1 else f"{tv}/{rv}",
                                                 f"{tv}/{dprod}", str(res)))
    holds(f"{n}!/{dd} = {res}")
    return pb, lines, str(res)


def c_comb(R):
    kind = R.choice(["C", "C", "P", "CC"])
    if kind == "P":
        n = rint(R, 4, 12)
        k = rint(R, 2, min(4, n))
        seq = "·".join(str(n - i) for i in range(k))
        res = math.perm(n, k)
        lines = [f"P({n}, {k}) = {n}!/{n - k}! = " + chain(f"{n}!/{n - k}!", seq)]
        pl, tv = prod_lines([n - i for i in range(k)])
        need(tv == res)
        lines += pl
        return f"Compute P({n}, {k}).", lines, str(res)
    if kind == "C":
        n = rint(R, 5, 20)
        k = rint(R, 2, n - 2)
        lines = []
        kk = k
        if k > n - k:
            kk = n - k
            lines.append(f"C({n}, {k}) = C({n}, {kk})")
        top = "·".join(str(n - i) for i in range(kk))
        bot = "·".join(str(kk - i) for i in range(kk))
        res = math.comb(n, k)
        want(kk <= 4)
        lines.append(f"C({n}, {kk}) = ({top})/({bot})")
        pl, tv = prod_lines([n - i for i in range(kk)])
        lines += pl
        lines.append(f"{bot} = {math.factorial(kk)}")
        holds(lines[-1])
        lines.append("C = " + chain(f"{tv}/{math.factorial(kk)}", str(res)))
        holds(f"({top})/({bot}) = {res}")
        return f"Compute C({n}, {k}).", lines, str(res)
    n1, n2 = rint(R, 4, 10), rint(R, 4, 10)
    k1, k2 = rint(R, 1, min(4, n1 - 1)), rint(R, 1, min(4, n2 - 1))
    v1, v2 = math.comb(n1, k1), math.comb(n2, k2)
    lines = [f"C({n1}, {k1}) = " + chain(f"({'·'.join(str(n1 - i) for i in range(k1))})/"
                                          f"({'·'.join(str(k1 - i) for i in range(k1))})", str(v1)),
             f"C({n2}, {k2}) = " + chain(f"({'·'.join(str(n2 - i) for i in range(k2))})/"
                                          f"({'·'.join(str(k2 - i) for i in range(k2))})", str(v2)),
             f"Product: " + chain(*mul_seg(v1, v2))]
    return f"Compute C({n1}, {k1})·C({n2}, {k2}).", lines, str(v1 * v2)


def c_incl(R):
    kind = R.choice(["or", "or", "neither", "three", "sets"])
    if kind == "sets":
        A, B = rint(R, 5, 40), rint(R, 5, 40)
        I = rint(R, 0, min(A, B))
        U = A + B - I
        lines = ["|A ∪ B| = |A| + |B| - |A ∩ B| = " + chain(f"{A} + {B} - {I}", str(U))]
        return f"Given |A| = {A}, |B| = {B}, |A ∩ B| = {I}, find |A ∪ B|.", lines, str(U)
    N = R.choice([rint(R, 20, 200), rint(R, 100, 1000)])
    if kind == "three":
        a, b, c = sorted(R.sample([2, 3, 5, 7, 11], 3))
        S = [(f"{a}", a), (f"{b}", b), (f"{c}", c), (f"({a}·{b})", a * b), (f"({a}·{c})", a * c),
             (f"({b}·{c})", b * c), (f"({a}·{b}·{c})", a * b * c)]
        cnt = [N // m for _, m in S]
        lines = [f"floor({N}/{lab}) = " + chain(f"floor({N}/{m})", str(v)) if lab != str(m) else
                 holds(f"floor({N}/{m}) = {v}") for (lab, m), v in zip(S, cnt)]
        tot = cnt[0] + cnt[1] + cnt[2] - cnt[3] - cnt[4] - cnt[5] + cnt[6]
        lines.append("Inclusion-exclusion: " + chain(join([str(x) for x in cnt[:3]] + [f"-{x}" for x in cnt[3:6]]
                                                         + [str(cnt[6])]), str(tot)))
        brute = sum(1 for t in range(1, N + 1) if t % a == 0 or t % b == 0 or t % c == 0)
        need(brute == tot)
        neither = R.random() < 0.4
        if neither:
            lines.append("Not divisible by any: " + chain(f"{N} - {tot}", str(N - tot)))
            return (f"How many integers from 1 to {N} are divisible by none of {a}, {b}, {c}?", lines, str(N - tot))
        return f"How many integers from 1 to {N} are divisible by {a}, {b}, or {c}?", lines, str(tot)
    a, b = R.sample(range(2, 13), 2)
    l = a * b // math.gcd(a, b)
    lines = []
    if math.gcd(a, b) > 1:
        lines.append(f"Divisible by both means divisible by lcm({a}, {b}) = {l}.")
    va, vb, vl = N // a, N // b, N // l
    lines += [f"floor({N}/{a}) = {va}", f"floor({N}/{b}) = {vb}", f"floor({N}/{l}) = {vl}"]
    for x in lines[-3:]:
        holds(x)
    tot = va + vb - vl
    lines.append(f"Divisible by {a} or {b}: " + chain(f"{va} + {vb} - {vl}", str(tot)))
    need(tot == sum(1 for t in range(1, N + 1) if t % a == 0 or t % b == 0))
    if kind == "neither":
        lines.append(f"Neither: " + chain(f"{N} - {tot}", str(N - tot)))
        return f"How many integers from 1 to {N} are divisible by neither {a} nor {b}?", lines, str(N - tot)
    return f"How many integers from 1 to {N} are divisible by {a} or {b}?", lines, str(tot)


def fam_count(R):
    k = R.choice(["ms", "ms", "comb", "comb", "incl", "incl"])
    return {"ms": c_multiset, "comb": c_comb, "incl": c_incl}[k](R)

# ------------------------------------------------------------------------------------------ 5. number theory


def n_gcd(R):
    g = R.choice([1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 14, 15, 18, 21])
    a, b = rint(R, 2, 120) * g, rint(R, 2, 120) * g
    want(a != b)
    a, b = max(a, b), min(a, b)
    want(a <= 3000)
    lines = []
    x, y = a, b
    while y:
        q, r = divmod(x, y)
        lines.append(holds(f"{x} = {q}·{y} + {r}"))
        x, y = y, r
    need(x == math.gcd(a, b))
    lines.append(f"The last nonzero remainder is {x}, so gcd({a}, {b}) = {x}.")
    if R.random() < 0.4 and a // x <= 12:
        l = a * b // x
        lines.append(f"lcm({a}, {b}) = ({a}/{x})·{b} = " + chain(f"{a // x}·{b}", *mul_seg(a // x, b)[1:]))
        holds(f"{a}·{b}/{x} = {l}")
        return f"Find gcd({a}, {b}) and lcm({a}, {b}) using the Euclidean algorithm.", lines, f"gcd = {x}, lcm = {l}"
    return f"Find gcd({a}, {b}) using the Euclidean algorithm.", lines, str(x)


def n_factor(R):
    if R.random() < 0.15:
        N = R.choice([p for p in range(11, 600) if sp.isprime(p)])
        ps = [p for p in range(2, math.isqrt(N) + 1) if sp.isprime(p)]
        nxt = sp.nextprime(ps[-1]) if ps else 2
        need(all(N % p for p in ps) and nxt * nxt > N)
        lines = [f"Test primes up to sqrt({N}): {', '.join(map(str, ps))}.", f"None divides {N}, and "
                 f"{nxt}^2 = {nxt * nxt} > {N}, so {N} is prime."]
        return f"Find the prime factorization of {N}.", lines, f"{N} is prime"
    N = 1
    for p in R.sample([2, 2, 2, 3, 3, 5, 7, 11, 13, 17, 19, 23], R.choice([3, 4, 5])):
        N *= p
    want(4 <= N <= 5000)
    lines, x = [], N
    for p in sp.primerange(2, x + 1):
        while x % p == 0 and x != p:
            lines.append(holds(f"{x} = {p}·{x // p}"))
            x //= p
        if x == p:
            break
    lines.append(f"{x} is prime.")
    fz = sp.factorint(N)
    fstr = "·".join(f"{p}^{e}" if e > 1 else f"{p}" for p, e in sorted(fz.items()))
    holds(f"{N} = {fstr}")
    lines.append(f"{N} = {fstr}")
    return f"Find the prime factorization of {N}.", lines, fstr


DIG = "0123456789ABCDEF"


def tob(n, b):
    s = ""
    while n:
        s = DIG[n % b] + s
        n //= b
    return s or "0"


def n_base(R):
    b = R.choice([2, 2, 2, 3, 4, 5, 6, 7, 8, 8, 16, 16])
    N = rint(R, 5, 300 if b == 2 else 999)
    s = tob(N, b)
    need(int(s, b) == N)
    if R.random() < 0.5:
        lines, x = [], N
        while x:
            q, r = divmod(x, b)
            lines.append(holds(f"{x} = {b}·{q} + {r}") + (f" ({DIG[r]})" if r >= 10 else ""))
            x = q
        lines.append(f"Read the remainders from bottom to top: {s}_{b}")
        return f"Convert {N} to base {b}.", lines, f"{s}_{b}"
    ds = [DIG.index(ch) for ch in s]
    L = len(ds)
    seg1 = " + ".join(f"{d}·{b}^{L - 1 - i}" for i, d in enumerate(ds))
    seg2 = " + ".join(f"{d}·{b ** (L - 1 - i)}" for i, d in enumerate(ds))
    seg3 = " + ".join(str(d * b ** (L - 1 - i)) for i, d in enumerate(ds))
    lines = []
    if b == 16 and any(d >= 10 for d in ds):
        lines.append("Digits: " + ", ".join(f"{ch} = {DIG.index(ch)}" for ch in sorted(set(s)) if DIG.index(ch) >= 10))
    lines.append(f"{s}_{b} = " + chain(seg1, seg2, seg3, str(N)))
    return f"Convert {s}_{b} to base 10.", lines, str(N)


def n_modexp(R):
    m = rint(R, 5, 60)
    a = rint(R, 2, 30)
    want(a % m)
    e = rint(R, 10, 200)
    bits = [1 << i for i in range(e.bit_length()) if e >> i & 1][::-1]
    lines = [holds(f"{e} = " + " + ".join(map(str, bits))) + f" (binary {bin(e)[2:]})"]
    r, p, res = a % m, 1, {}
    if a >= m:
        lines.append(f"{a} ≡ {r} (mod {m})")
    res[1] = r
    lines.append(f"{a}^1 ≡ {r} (mod {m})")
    while p * 2 <= e:
        sq = r * r
        r2 = sq % m
        lines.append(f"{a}^{p * 2} ≡ {r}^2 = {sq} ≡ {r2} (mod {m})")
        need(pow(a, p * 2, m) == r2 and sq == r * r)
        p *= 2
        r = r2
        res[p] = r
    acc = None
    for bt in bits:
        if acc is None:
            acc = res[bt]
            continue
        pr_ = acc * res[bt]
        lines.append(f"{acc}·{res[bt]} = {pr_} ≡ {pr_ % m} (mod {m})" + f"  [{a}^{bt}]")
        acc = pr_ % m
    need(acc == pow(a, e, m))
    if len(bits) > 1:
        lines.insert(len(lines) - len(bits) + 1, f"{a}^{e} = " + "·".join(f"{a}^{bt}" for bt in bits))
    return f"Compute {a}^{e} mod {m} by repeated squaring.", lines, str(acc)


def fam_numth(R):
    k = R.choice(["gcd", "fac", "base", "base", "mod"])
    return {"gcd": n_gcd, "fac": n_factor, "base": n_base, "mod": n_modexp}[k](R)

# ------------------------------------------------------------------------------- 6. exponents and radicals
BASES = {2: [2, 4, 8, 16, 32], 3: [3, 9, 27, 81], 5: [5, 25, 125]}


def x_common(R):
    p = R.choice([2, 2, 3, 5])
    opts = BASES[p]
    A = R.choice(opts)
    m1 = round(math.log(A, p))
    recip = R.random() < 0.2
    a1, b1 = rint(R, 1, 3), rint(R, -4, 4)
    if R.random() < 0.35:  # A^(lin) = C
        e2 = rint(R, -3, 6)
        Cv = Fr(p) ** e2
        Cs = fs(Cv)
        lhs_base = f"(1/{A})" if recip else str(A)
        s1 = -m1 if recip else m1
        stmt = f"{lhs_base}^({poly({1: a1, 0: b1})}) = {Cs}"
        lines = [f"{lhs_base} = {p}^({s1})" if recip or s1 < 0 else f"{A} = {p}^{m1}" if m1 > 1 else f"{A} = {p}^1",
                 f"{Cs} = {p}^({e2})" if e2 < 0 else f"{Cs} = {p}^{e2}"]
        lines.append(f"{p}^({s1}({poly({1: a1, 0: b1})})) = {p}^({e2})")
        L = (s1 * a1, s1 * b1)
        lines.append(f"{poly({1: L[0], 0: L[1]})} = {e2}")
        sl, x0 = lin_solve_lines(L[0], L[1], 0, e2, "x")
        lines += sl[1:]
        ev = Fr(a1) * x0 + b1
        lines.append(f"Check: " + chain(ex(lhs_base, ev), Cs) + " ✓")
    else:
        B = R.choice([o for o in opts if o != A] or opts)
        m2 = round(math.log(B, p))
        a2, b2 = rint(R, 1, 3), rint(R, -4, 4)
        want(m1 * a1 != m2 * a2)
        stmt = f"{A}^({poly({1: a1, 0: b1})}) = {B}^({poly({1: a2, 0: b2})})"
        wr = lambda m, a_, b_: f"{p}^({poly({1: a_, 0: b_})})" if m == 1 else f"{p}^({m}({poly({1: a_, 0: b_})}))"
        lines = [", ".join(f"{X} = {p}^{m}" for X, m in ((A, m1), (B, m2)) if X != p) or f"{A} = {p}",
                 f"{wr(m1, a1, b1)} = {wr(m2, a2, b2)}",
                 f"{poly({1: m1 * a1, 0: m1 * b1})} = {poly({1: m2 * a2, 0: m2 * b2})}"]
        sl, x0 = lin_solve_lines(m1 * a1, m1 * b1, m2 * a2, m2 * b2, "x")
        lines += sl
        e1, e2v = Fr(a1) * x0 + b1, Fr(a2) * x0 + b2
        want(abs(m1 * e1) <= 40)
        c1 = ex(A, e1) if A == p else f"{ex(A, e1)} = {ex(p, m1 * e1)}"
        c2 = ex(B, e2v) if B == p else f"{ex(B, e2v)} = {ex(p, m2 * e2v)}"
        lines.append(f"Check: {c1} and {c2} ✓")
        need(m1 * e1 == m2 * e2v)
        holds(c1)
        holds(c2)
        holds(f"{ex(A, e1)} = {ex(B, e2v)}")
    for ln in lines[2:]:
        if ln.startswith("Check") or "^" in ln:
            continue
        holds(ln.split(": ")[-1], {"x": x0})
    holds(stmt, {"x": x0})
    return f"Solve {stmt}.", lines, f"x = {fs(x0)}"


def x_sqrt(R):
    if R.random() < 0.3:
        m = R.choice([n for n in range(2, 600) if sqfree(n)[0] == 1])
        fz = sp.factorint(m)
        fstr = "·".join(f"{p}" for p in sorted(fz))
        lines = [f"{m} = {fstr}" if len(fz) > 1 else f"{m} is prime.",
                 f"No prime factor appears twice, so sqrt({m}) is already in simplest form."]
        if len(fz) > 1:
            holds(lines[0])
        return f"Simplify sqrt({m}).", lines, f"sqrt({m})"
    k = rint(R, 2, 12)
    m = R.choice([n for n in range(2, 60) if sqfree(n)[0] == 1])
    n = k * k * m
    want(n <= 1500 and (k <= 9 or m <= 9))
    lines = [f"{n} = {k * k}·{m}, and {k * k} = {k}^2",
             f"sqrt({n}) = " + chain(f"sqrt({k * k}·{m})", f"sqrt({k * k})·sqrt({m})", rad(k, m))]
    holds(lines[0].split(", and ")[0])
    return f"Simplify sqrt({n}).", lines, rad(k, m)


def x_cbrt(R):
    k = rint(R, 2, 5)
    m = R.choice([2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15])
    n = k ** 3 * m
    lines = [f"{n} = {k ** 3}·{m}, and {k ** 3} = {k}^3",
             f"cbrt({n}) = " + chain(f"cbrt({k ** 3}·{m})", f"cbrt({k ** 3})·cbrt({m})", f"{k}cbrt({m})")]
    return f"Simplify cbrt({n}).", lines, f"{k}cbrt({m})"


def x_combo(R):
    kind = R.choice(["sum", "prod", "ratl"])
    if kind == "sum":
        m = R.choice([2, 3, 5, 6, 7])
        ks = [rint(R, 1, 6) for _ in range(R.choice([2, 3]))]
        cs = [R.choice([1, 1, 2, 3, -1, -2]) for _ in ks]
        cs[0] = abs(cs[0])
        terms = [(c, k * k * m) for c, k in zip(cs, ks)]
        disp = join([("-" if c < 0 else "") + ("" if abs(c) == 1 else str(abs(c))) + f"sqrt({n})" for c, n in terms])
        simp = join([("-" if c < 0 else "") + rad(abs(c) * k, m) for c, k in zip(cs, ks)])
        tot = sum(c * k for c, k in zip(cs, ks))
        res = "0" if tot == 0 else ("-" if tot < 0 else "") + rad(abs(tot), m)
        lines = [f"Simplify each: " + ", ".join(chain(f"sqrt({n})", rad(k, m)) for (c, n), k in zip(terms, ks)),
                 chain(disp, simp, res)]
        return f"Simplify {disp}.", lines, res
    if kind == "prod":
        a, b = rint(R, 2, 30), rint(R, 2, 30)
        k, m = sqfree(a * b)
        lines = [chain(f"sqrt({a})·sqrt({b})", f"sqrt({a * b})", rad(k, m))]
        if k > 1:
            lines.insert(0, f"{a * b} = {k * k}·{m}")
        return f"Simplify sqrt({a})·sqrt({b}).", lines, rad(k, m)
    m = R.choice([2, 3, 5, 6, 7, 10])
    a = rint(R, 1, 6) * R.choice([1, m])
    g = math.gcd(a, m)
    res = rad(1, m) if a == m else (f"{rad(a // g, m)}/{m // g}" if m // g != 1 else rad(a // g, m))
    lines = [f"Multiply top and bottom by sqrt({m}): " + chain(f"{a}/sqrt({m})", f"{a}sqrt({m})/{m}", res)]
    return f"Rationalize the denominator of {a}/sqrt({m}).", lines, res


def fam_exprad(R):
    k = R.choice(["common"] * 5 + ["sqrt"] * 2 + ["cbrt", "combo", "combo"])
    return {"common": x_common, "sqrt": x_sqrt, "cbrt": x_cbrt, "combo": x_combo}[k](R)

# ------------------------------------------------------------------------------------------------ 7. sequences


def q_arith(R):
    a1 = rint(R, -20, 20)
    d = Fr(rint(R, -9, 9, True)) if R.random() < 0.85 else Fr(rint(R, -9, 9, True), 2)
    i, j = sorted(R.sample(range(1, 12), 2))
    n = rint(R, max(j + 1, 8), 20)
    A = lambda k: a1 + (k - 1) * d
    ai, aj, an = A(i), A(j), A(n)
    want(ai.denominator == 1 and aj.denominator == 1)
    S = Fr(n) * (a1 + an) / 2
    lines = [f"d = (a_{j} - a_{i})/({j} - {i}) = " + chain(f"({fs(aj)} - {atom(ai)})/({j} - {i})",
                                                           f"{fs(aj - ai)}/{j - i}", fs(d))]
    if i > 1:
        lines.append(f"a_1 = a_{i} - {i - 1}d = " + chain(f"{fs(ai)} - {i - 1}({fs(d)})",
                                                          join([fs(ai), fs(-(i - 1) * d)]), fs(a1)))
    lines.append(f"a_{n} = a_1 + {n - 1}d = " + chain(f"{fs(a1)} + {n - 1}({fs(d)})",
                                                      join([fs(a1), fs((n - 1) * d)]), fs(an)))
    sm = a1 + an
    head = f"S_{n} = {n}(a_1 + a_{n})/2 = " + chain(f"{n}({fs(a1)} + {atom(an)})/2", f"{n}({fs(sm)})/2")
    if sm.denominator == 1 and (n % 2 == 0 or sm.numerator % 2 == 0):
        x, y = (n // 2, int(sm)) if n % 2 == 0 else (n, int(sm) // 2)
        lines.append(head + " = " + chain(*mul_seg(x, y)))
        need(Fr(x * y) == S)
    else:
        lines.append(head + " = " + chain(f"{n}({fs(sm)})/2", fs(S)))
    need(S == sum(A(k) for k in range(1, n + 1)))
    return (f"An arithmetic sequence has a_{i} = {fs(ai)} and a_{j} = {fs(aj)}. Find a_{n} and S_{n}.", lines,
            f"a_{n} = {fs(an)}, S_{n} = {fs(S)}")


def q_geo(R):
    r = R.choice([Fr(2), Fr(3), Fr(-2), Fr(-3), Fr(1, 2), Fr(-1, 2), Fr(4), Fr(5), Fr(2, 3), Fr(3, 2)])
    a1 = Fr(rint(R, -6, 6, True))
    i = rint(R, 1, 4)
    gap = R.choice([1, 2, 3])
    j = i + gap
    n = rint(R, max(j + 1, 5), 8)
    A = lambda k: a1 * r ** (k - 1)
    ai, aj, an = A(i), A(j), A(n)
    want(max(abs(x.numerator) for x in (ai, aj, an)) <= 3000 and max(x.denominator for x in (ai, aj, an)) <= 300)
    rg = aj / ai
    pos = gap % 2 == 0
    if pos:
        want(r > 0)
    lines = [f"r^{gap} = a_{j}/a_{i} = " + chain(div_str(aj, ai), fs(rg)) if gap > 1 else
             f"r = a_{j}/a_{i} = " + chain(div_str(aj, ai), fs(rg))]
    if gap > 1:
        lines.append(f"r = {fs(r)}, since {atom(r)}^{gap} = {fs(r ** gap)}" + (" and r > 0" if pos else ""))
        need(r ** gap == rg)
    if i > 1:
        lines.append((f"a_1 = a_{i}/r = " + chain(f"{atom(ai)}/{atom(r)}", fs(a1))) if i == 2 else
                     (f"a_1 = a_{i}/r^{i - 1} = " + chain(f"{atom(ai)}/{atom(r)}^{i - 1}", fs(a1))))
    lines.append(f"a_{n} = a_1·r^{n - 1} = " + chain(f"{atom(a1)}·{atom(r)}^{n - 1}", f"{atom(a1)}·{atom(r ** (n - 1))}",
                                                       fs(an)))
    S = a1 * (r ** n - 1) / (r - 1)
    lines.append(f"S_{n} = a_1(r^{n} - 1)/(r - 1) = " + chain(f"{atom(a1)}({atom(r)}^{n} - 1)/({fs(r)} - 1)",
                                                              f"{atom(a1)}({fs(r ** n - 1)})/({fs(r - 1)})", fs(S)))
    need(S == sum(A(k) for k in range(1, n + 1)))
    return (f"A geometric sequence has a_{i} = {fs(ai)} and a_{j} = {fs(aj)}" + (" with r > 0" if pos else "")
            + f". Find a_{n} and S_{n}.", lines, f"a_{n} = {fs(an)}, S_{n} = {fs(S)}")


def fam_seq(R):
    return q_arith(R) if R.random() < 0.5 else q_geo(R)

# ---------------------------------------------------------------------------------------------- 8. inequalities
FLIP = {"<": ">", ">": "<", "≤": "≥", "≥": "≤"}
GRID = [Fr(k, 4) for k in range(-160, 161)]


def interval(lo, lo_c, hi, hi_c):
    l = "(-∞" if lo is None else ("[" if lo_c else "(") + fs(lo)
    h = "∞)" if hi is None else fs(hi) + ("]" if hi_c else ")")
    return f"{l}, {h}"


def in_iv(t, lo, lo_c, hi, hi_c):
    return (lo is None or t > lo or (lo_c and t == lo)) and (hi is None or t < hi or (hi_c and t == hi))


def same_set(stmt, member):
    for t in GRID:
        need(rel_ok(stmt, {"x": t}) == member(t), f"set mismatch {stmt} @ {t}")


def int_count_lines(lo, lo_c, hi, hi_c):
    m = math.floor(lo) + 1 if (lo.denominator != 1 or not lo_c) else int(lo)
    if lo.denominator != 1:
        m = math.ceil(lo)
    M = math.ceil(hi) - 1 if (hi.denominator != 1 or not hi_c) else int(hi)
    if hi.denominator != 1:
        M = math.floor(hi)
    cnt = max(0, M - m + 1)
    need(cnt == sum(1 for t in range(-200, 201) if in_iv(Fr(t), lo, lo_c, hi, hi_c)))
    if cnt == 0:
        return "There are no integers in this interval.", 0
    if cnt <= 8:
        return f"Integers: {', '.join(str(t) for t in range(m, M + 1))}; count = {cnt}", cnt
    return f"Integers from {m} to {M}: " + chain(f"{M} - {atom(m)} + 1", str(cnt)), cnt


def i_lin(R):
    a = rint(R, -7, 7, True)
    c = R.choice([0, 0] + [i for i in range(-7, 8) if i not in (0, a)])
    b, d = rint(R, -15, 15), rint(R, -15, 15)
    op = R.choice(list(FLIP))
    stmt = f"{poly({1: a, 0: b})} {op} {poly({1: c, 0: d})}"
    A, B = a - c, d - b
    x0 = Fr(B, A)
    lines = []
    if c or b:
        lines.append(f"{join([term(a, 'x')] + ([term(-c, 'x')] if c else []))} {op} "
                     f"{join([fs(d)] + ([fs(-b)] if b else []))}")
    lines.append(f"{term(A, 'x')} {op} {fs(B)}")
    op2 = op if A > 0 else FLIP[op]
    if A != 1:
        lines.append(f"Divide by {A}" + (" and flip the sign" if A < 0 else "") + f": x {op2} " +
                     chain(div_str(B, A), fs(x0)))
    final = f"x {op2} {fs(x0)}"
    up = op2 in ("<", "≤")
    closed = op2 in ("≤", "≥")
    lo, hi = (None, x0) if up else (x0, None)
    member = lambda t: in_iv(t, lo, closed, hi, closed)
    for ln in [stmt] + lines:
        same_set(ln.split(": ")[-1], member)
    same_set(final, member)
    iv = interval(lo, closed, hi, closed)
    lines.append(f"Solution: {iv}")
    if up:
        M = math.floor(x0) if (x0.denominator != 1 or closed) else int(x0) - 1
        need(member(Fr(M)) and not member(Fr(M + 1)))
        lines.append(f"Largest integer solution: {M}")
        return f"Solve {stmt}. Give the interval and the largest integer solution.", lines, f"{iv}; largest integer {M}"
    m = math.ceil(x0) if (x0.denominator != 1 or closed) else int(x0) + 1
    need(member(Fr(m)) and not member(Fr(m - 1)))
    lines.append(f"Smallest integer solution: {m}")
    return f"Solve {stmt}. Give the interval and the smallest integer solution.", lines, f"{iv}; smallest integer {m}"


def compound_lines(a, b, L, o1, o2, H):
    """L o1 a x + b o2 H, o in {<, ≤}: returns lines, (lo, lo_c, hi, hi_c)"""
    lines = [f"{fs(L)} {o1} {poly({1: a, 0: b})} {o2} {fs(H)}"]
    if b:
        lines.append(f"Subtract {fs(b)}: {fs(L - b)} {o1} {term(a, 'x')} {o2} {fs(H - b)}" if b > 0 else
                     f"Add {fs(-b)}: {fs(L - b)} {o1} {term(a, 'x')} {o2} {fs(H - b)}")
    lo, hi = Fr(L - b, a), Fr(H - b, a)
    if a > 0:
        res = (lo, o1 == "≤", hi, o2 == "≤")
        if a != 1:
            lines.append(f"Divide by {a}: {fs(lo)} {o1} x {o2} {fs(hi)}")
    else:
        res = (hi, o2 == "≤", lo, o1 == "≤")
        lines.append(f"Divide by {a} and flip: {fs(lo)} {FLIP[o1]} x {FLIP[o2]} {fs(hi)}")
        lines.append(f"{fs(hi)} {o2} x {o1} {fs(lo)}")
    return lines, res


def i_comp(R):
    a = rint(R, -5, 5, True)
    b = rint(R, -10, 10)
    L = rint(R, -15, 10)
    H = L + rint(R, 2, 20)
    o1, o2 = R.choice(["<", "≤"]), R.choice(["<", "≤"])
    lines, (lo, lc, hi, hc) = compound_lines(a, b, L, o1, o2, H)
    member = lambda t: in_iv(t, lo, lc, hi, hc)
    for ln in lines:
        same_set(ln.split(": ")[-1], member)
    stmt = lines[0]
    lines = lines[1:]
    iv = interval(lo, lc, hi, hc)
    cl, cnt = int_count_lines(lo, lc, hi, hc)
    lines += [f"Solution: {iv}", cl]
    return f"Solve {stmt}. Give the interval and count the integers in it.", lines, f"{iv}; {cnt} integers"


def i_abs(R):
    a = rint(R, -5, 5, True)
    b = rint(R, -10, 10)
    c = rint(R, 1, 12)
    inner = poly({1: a, 0: b})
    op = R.choice(["<", "≤", "<", "≤", ">", "≥"])
    stmt = f"|{inner}| {op} {c}"
    if op in ("<", "≤"):
        lines, (lo, lc, hi, hc) = compound_lines(a, b, -c, op, op, c)
        member = lambda t: in_iv(t, lo, lc, hi, hc)
        same_set(stmt, member)
        for ln in lines:
            same_set(ln.split(": ")[-1], member)
        iv = interval(lo, lc, hi, hc)
        cl, cnt = int_count_lines(lo, lc, hi, hc)
        lines += [f"Solution: {iv}", cl]
        return f"Solve {stmt}. Give the interval and count the integers in it.", lines, f"{iv}; {cnt} integers"
    o2 = "<" if op == ">" else "≤"
    r1, r2 = sorted([Fr(-c - b, a), Fr(c - b, a)])
    closed = op == "≥"
    lines = [f"{inner} {FLIP[op]} {-c} or {inner} {op} {c}"]
    s1, s2 = (f"x {FLIP[op]} {fs(Fr(-c - b, a))}", f"x {op} {fs(Fr(c - b, a))}") if a > 0 else \
        (f"x {op} {fs(Fr(-c - b, a))}", f"x {FLIP[op]} {fs(Fr(c - b, a))}")
    lines.append(f"{s1} or {s2}")
    member = lambda t: (t < r1 or t > r2) or (closed and t in (r1, r2))
    same_set(stmt, member)
    for t in GRID:
        need((rel_ok(s1, {"x": t}) or rel_ok(s2, {"x": t})) == member(t))
    lb, rb = ("]", "[") if closed else (")", "(")
    sol = f"(-∞, {fs(r1)}{lb} ∪ {rb}{fs(r2)}, ∞)"
    lines.append(f"Solution: {sol}")
    lines.append(f"Non-solutions: {interval(r1, not closed, r2, not closed)}")
    cl, cnt = int_count_lines(r1, not closed, r2, not closed)
    lines.append(cl)
    return (f"Solve {stmt}. Give the solution set and count the integers that are not solutions.", lines,
            f"{sol}; {cnt} integers are not solutions")


def fam_ineq(R):
    k = R.choice(["lin", "lin", "comp", "comp", "abs", "abs", "abs"])
    return {"lin": i_lin, "comp": i_comp, "abs": i_abs}[k](R)

# ------------------------------------------------------------------------------------------ 9. coordinates


def pt(R):
    return (rint(R, -9, 9), rint(R, -9, 9))


def g_dist(R):
    (x1, y1), (x2, y2) = pt(R), pt(R)
    want((x1, y1) != (x2, y2))
    dx, dy = x2 - x1, y2 - y1
    s2 = dx * dx + dy * dy
    k, m = sqfree(s2)
    segs = [f"sqrt(({x2} - {atom(x1)})^2 + ({y2} - {atom(y1)})^2)", f"sqrt({atom(dx)}^2 + {atom(dy)}^2)",
            f"sqrt({dx * dx} + {dy * dy})", f"sqrt({s2})"]
    if k > 1:
        segs.append(f"sqrt({k * k}·{m})")
    segs.append(rad(k, m))
    lines = ["d = sqrt((x2 - x1)^2 + (y2 - y1)^2) = " + chain(*segs)]
    return f"Find the distance between ({x1}, {y1}) and ({x2}, {y2}).", lines, f"d = {rad(k, m)}"


def g_mid(R):
    (x1, y1), (x2, y2) = pt(R), pt(R)
    want((x1, y1) != (x2, y2))
    mx, my = Fr(x1 + x2, 2), Fr(y1 + y2, 2)
    cx = chain(f"({x1} + {atom(x2)})/2", f"{x1 + x2}/2", fs(mx))
    cy = chain(f"({y1} + {atom(y2)})/2", f"{y1 + y2}/2", fs(my))
    lines = [f"x: {cx}", f"y: {cy}", f"M = ({fs(mx)}, {fs(my)})"]
    return f"Find the midpoint of ({x1}, {y1}) and ({x2}, {y2}).", lines, f"({fs(mx)}, {fs(my)})"


def g_line(R):
    (x1, y1), (x2, y2) = pt(R), pt(R)
    want((x1, y1) != (x2, y2))
    pb = f"Find the slope and the equation of the line through ({x1}, {y1}) and ({x2}, {y2})."
    if x1 == x2:
        lines = [f"x2 - x1 = {x2} - {atom(x1)} = 0, so the slope is undefined (vertical line).", f"Line: x = {x1}"]
        return pb, lines, f"slope undefined; x = {x1}"
    m = Fr(y2 - y1, x2 - x1)
    lines = ["m = (y2 - y1)/(x2 - x1) = " + chain(f"({y2} - {atom(y1)})/({x2} - {atom(x1)})",
                                                  div_str(y2 - y1, x2 - x1), fs(m))]
    bb = y1 - m * x1
    lines.append(f"b = y1 - m·x1 = " + chain(f"{y1} - {atom(m)}({x1})", join([fs(y1), fs(-m * x1)]), fs(bb)))
    eqn = f"y = {poly({1: m, 0: bb})}"
    s, v = sub(P1({1: m, 0: bb}), {"x": x2})
    need(v == y2)
    lines.append(f"Line: {eqn}")
    lines.append(f"Check ({x2}, {y2}): {s} ✓")
    holds(eqn, {"x": x1, "y": y1})
    return pb, lines, f"m = {fs(m)}; {eqn}"


ANG = [sp.Rational(k, 12) for k in range(0, 24) if k % 2 == 0 or k % 3 == 0]  # multiples of π/6 and π/4


def angs(t):  # t*π display
    t = F(t)
    if t == 0:
        return "0"
    num = "π" if t.numerator == 1 else f"{t.numerator}π"
    return num if t.denominator == 1 else f"{num}/{t.denominator}"


def g_polar(R):
    r = rint(R, 1, 20)
    t = R.choice(ANG)
    th = angs(t)
    cv, sv = sp.cos(t * sp.pi), sp.sin(t * sp.pi)
    x, y = r * cv, r * sv
    lines = [f"x = r·cos(θ) = " + chain(f"{r}cos({th})", f"{r}({pr(cv)})", pr(x)),
             f"y = r·sin(θ) = " + chain(f"{r}sin({th})", f"{r}({pr(sv)})", pr(y))]
    return f"Convert the polar point (r, θ) = ({r}, {th}) to rectangular coordinates.", lines, f"({pr(x)}, {pr(y)})"


def g_rect(R):
    r0 = rint(R, 1, 15)
    t = R.choice(ANG)
    cv, sv = sp.cos(t * sp.pi), sp.sin(t * sp.pi)
    if t.q in (6, 3):  # coordinates involve sqrt(3): scale by 2 to keep it moderate
        r0 = 2 * rint(R, 1, 8)
    x, y = r0 * cv, r0 * sv
    xs, ys = pr(x), pr(y)
    r2 = sp.expand(x * x + y * y)
    need(r2 == r0 * r0)
    lines = [f"r = sqrt(x^2 + y^2) = " + chain(f"sqrt({pwb(x)}^2 + {pwb(y)}^2)", f"sqrt({pr(x * x)} + {pr(y * y)})",
                                               f"sqrt({r0 * r0})", str(r0))]
    th = angs(t)
    if x == 0 or y == 0:
        where = {0: "positive x-axis", sp.Rational(1, 2): "positive y-axis", 1: "negative x-axis",
                 sp.Rational(3, 2): "negative y-axis"}[t]
        lines.append(f"The point is on the {where}, so θ = {th}.")
    else:
        tv = sp.radsimp(y / x)
        lines.append(f"tan(θ) = y/x = " + chain(f"{pwb(y)}/{pwb(x)}", pr(tv)))
        ref = sp.atan(abs(tv)) / sp.pi
        need(ref in (sp.Rational(1, 6), sp.Rational(1, 4), sp.Rational(1, 3)))
        q = {(True, True): "I", (False, True): "II", (False, False): "III", (True, False): "IV"}[(x > 0, y > 0)]
        form = {"I": angs(ref), "II": f"π - {angs(ref)}", "III": f"π + {angs(ref)}", "IV": f"2π - {angs(ref)}"}[q]
        lines.append(f"Reference angle {angs(ref)}; x {'>' if x > 0 else '<'} 0 and y {'>' if y > 0 else '<'} 0, "
                     f"so quadrant {q}: θ = " + chain(form, th))
    need(eq(r0 * sp.cos(t * sp.pi), x) and eq(r0 * sp.sin(t * sp.pi), y))
    holds(f"{r0}cos({th}) = {xs}")
    holds(f"{r0}sin({th}) = {ys}")
    return (f"Convert ({xs}, {ys}) to polar coordinates (r, θ) with r > 0 and 0 ≤ θ < 2π.", lines, f"({r0}, {th})")


def fam_coord(R):
    k = R.choice(["dist", "dist", "mid", "line", "line", "polar", "rect", "rect"])
    return {"dist": g_dist, "mid": g_mid, "line": g_line, "polar": g_polar, "rect": g_rect}[k](R)

# ------------------------------------------------------------------------------------- 10. polynomial division


def pd(p):
    co = sp.Poly(p, XS).all_coeffs()[::-1]
    return {e: F(c) for e, c in enumerate(co)}


def pdisp(p):
    return poly(pd(p)) if p != 0 else "0"


def fam_div(R):
    kind = R.random()
    if kind < 0.5:
        dv = XS - rint(R, -6, 6, True)
    elif kind < 0.7:
        dv = rint(R, 2, 3) * XS + rint(R, -5, 5, True)
    else:
        dv = XS ** 2 + rint(R, -4, 4) * XS + rint(R, -6, 6, True)
    dd = sp.degree(dv, XS)
    qdeg = rint(R, 1, 3 if dd == 1 else 2)
    q = sum(rint(R, -6, 6) * XS ** e for e in range(qdeg)) + rint(R, -5, 5, True) * XS ** qdeg
    r = sum(rint(R, -9, 9) * XS ** e for e in range(dd)) if R.random() < 0.8 else 0
    P = sp.expand(dv * q + r)
    want(max(abs(c) for c in pd(P).values()) <= 120)
    ds = pdisp(dv)
    lines, cur = [], P
    lt_d = sp.LT(dv, XS)
    while cur != 0 and sp.degree(cur, XS) >= dd:
        lt = sp.LT(cur, XS)
        qt = sp.simplify(lt / lt_d)
        prod_ = sp.expand(qt * dv)
        new = sp.expand(cur - prod_)
        lines.append(f"{pr(lt)} ÷ {pr(lt_d)} = {pr(qt)}; {mulstr(par(qt) if qt.is_Add else pr(qt), '(' + ds + ')')} "
                     f"= {pdisp(prod_)}; subtract: {pdisp(new)}")
        ident(pr(lt) + "/(" + pr(lt_d) + ")", pr(qt))
        ident(f"{pr(qt)}({ds})", pdisp(prod_))
        ident(f"({pdisp(cur)}) - ({pdisp(prod_)})", pdisp(new))
        cur = new
    Q, Rm = sp.div(P, dv, XS)
    need(sp.expand(Q - q) == 0 and sp.expand(cur - r) == 0 and sp.expand(Rm - r) == 0)
    qs, rs = pdisp(q), pdisp(r)
    lines.append(f"Quotient: {qs}; remainder: {rs}")
    chk = f"({ds})({qs})" + ("" if r == 0 else (f" + {rs}" if not rs.startswith("-") else f" - {rs[1:]}"
                                               if pd(r).__len__() == 1 else f" + ({rs})"))
    lines.append(f"Check: {ident(chk, pdisp(P))} ✓")
    return f"Divide {pdisp(P)} by {ds}.", lines, f"quotient {qs}, remainder {rs}"

# ---------------------------------------------------------------------------------------- 11. differentiation


def d_check(f_s, segs):
    """each displayed derivative segment equals d/dx of the displayed function."""
    fd = sp.diff(sym(f_s), XS)
    for p in PTS:
        tv = fd.subs(XS, p)
        for g in segs:
            need(eq(E(g, {"x": p}), tv), f"deriv {f_s} -> {g}")


def piece(R, allow_poly=True):
    k = R.choice(["pow", "pow", "exp", "ln", "sin", "cos", "poly", "sqrt", "explin"] if allow_poly else
                 ["pow", "exp", "ln", "sin", "cos", "sqrt"])
    if k == "pow":
        return XS ** rint(R, 1, 5)
    if k == "exp":
        return sp.exp(XS)
    if k == "explin":
        return sp.exp(rint(R, -3, 3, True, (1,)) * XS)
    if k == "ln":
        return sp.log(XS)
    if k == "sin":
        return sp.sin(XS)
    if k == "cos":
        return sp.cos(XS)
    if k == "sqrt":
        return sp.sqrt(XS)
    return rint(R, 1, 4) * XS ** rint(R, 1, 2) + rint(R, -6, 6, True)


def rule_note(t):
    if t.is_Pow and t.base == XS:
        return "power rule"
    if isinstance(t, sp.exp):
        return "exp"
    return None


def dd_sum(R):
    n = rint(R, 2, 4)
    terms = []
    for _ in range(n):
        c = rint(R, -9, 9, True)
        b = R.choice(["pow", "pow", "pow", "pow", "exp", "sin", "cos", "ln", "sqrt", "const"])
        t = {"pow": XS ** rint(R, 1, 7), "exp": sp.exp(XS), "sin": sp.sin(XS), "cos": sp.cos(XS), "ln": sp.log(XS),
             "sqrt": sp.sqrt(XS), "const": sp.Integer(1)}[b]
        terms.append(c * t)
    f = sp.Add(*terms)
    want(len(f.as_ordered_terms()) >= 2)
    fs_ = pr(f)
    lines = []
    for t in f.as_ordered_terms():
        dt = sp.diff(t, XS)
        c, rest = t.as_coeff_Mul()
        if rest.is_Pow and rest.base == XS and rest.exp.is_Integer and rest.exp >= 2:
            nexp = rest.exp
            mid = f"{atom(c)}·{nexp}{pr(XS ** (nexp - 1))}" if c != 1 else f"{nexp}{pr(XS ** (nexp - 1))}"
            segs = [mid, pr(dt)]
        else:
            segs = [pr(dt)]
        d_check(pr(t), segs)
        lines.append(f"d/dx[{pr(t)}] = " + " = ".join(dict.fromkeys(segs)))
    res = pr(sp.diff(f, XS))
    d_check(fs_, [res])
    lines.append(f"f'(x) = {res}")
    return f"Differentiate f(x) = {fs_}.", lines, f"f'(x) = {res}"


def dd_prod(R):
    u, v = piece(R), piece(R)
    want(u != v and not (u.is_polynomial(XS) and v.is_polynomial(XS)))
    c = rint(R, 1, 3)
    u = c * u
    f = u * v
    du, dv = sp.diff(u, XS), sp.diff(v, XS)
    fs_ = mulstr(par(u), par(v))
    mid = join([mulstr(par(du), par(v)), mulstr(par(u), par(dv))])
    res = pr(sp.expand(du * v + u * dv)) if (u * v).is_polynomial(XS) else pr(du * v + u * dv)
    lines = [f"u = {pr(u)}, u' = {pr(du)}", f"v = {pr(v)}, v' = {pr(dv)}",
             f"f'(x) = u'v + uv' = " + " = ".join(dict.fromkeys([mid, res]))]
    d_check(pr(u), [pr(du)])
    d_check(pr(v), [pr(dv)])
    d_check(fs_, [mid, res])
    return f"Differentiate f(x) = {fs_} using the product rule.", lines, f"f'(x) = {res}"


def dd_quot(R):
    u = piece(R)
    v = R.choice([XS + rint(R, -5, 5, True), XS ** 2 + rint(R, 1, 6), sp.exp(XS), XS, rint(R, 1, 3) * XS + rint(R, -4, 4, True)])
    want(sp.simplify(u / v - sp.Integer(1)) != 0 and not (u.has(sp.exp) and v.has(sp.exp)))
    du, dv = sp.diff(u, XS), sp.diff(v, XS)
    fs_ = f"{par(u)}/{par(v) if not v.is_Pow else '(' + pr(v) + ')'}"
    num_mid = join([mulstr(par(du), par(v)), "-" + mulstr(par(u), par(dv)) if not pr(u * dv).startswith("-")
                    else mulstr(par(u), par(dv)).join(["-(", ")"])])
    mid = f"({num_mid})/({pr(v)})^2"
    num = sp.expand(du * v - u * dv)
    res = f"({pr(num)})/({pr(v)})^2" if num.is_Add else f"{par(num)}/({pr(v)})^2"
    lines = [f"u = {pr(u)}, u' = {pr(du)}", f"v = {pr(v)}, v' = {pr(dv)}",
             f"f'(x) = (u'v - uv')/v^2 = " + " = ".join(dict.fromkeys([mid, res]))]
    d_check(pr(u), [pr(du)])
    d_check(pr(v), [pr(dv)])
    d_check(fs_, [mid, res])
    return f"Differentiate f(x) = {fs_} using the quotient rule.", lines, f"f'(x) = {res}"


def dd_chain(R):
    g = R.choice([rint(R, -5, 5, True, (1,)) * XS + rint(R, -6, 6),
                  rint(R, 1, 3) * XS ** 2 + rint(R, -4, 4) * XS + rint(R, 1, 6)])
    U = sp.Symbol("u")
    outs = [U ** rint(R, 2, 6), sp.exp(U), sp.log(U), sp.sin(U), sp.cos(U), sp.sqrt(U)]
    o = R.choice(outs)
    c = R.choice([1, 1, 1, 2, 3, -1])
    o = c * o
    f = o.subs(U, g)
    do = sp.diff(o, U)
    dg = sp.diff(g, XS)
    fs_ = pr(f)
    mid = mulstr(par(do.subs(U, g)), "(" + pr(dg) + ")")
    res = pr(sp.diff(f, XS))
    lines = [f"Inner: u = {pr(g)}, u' = {pr(dg)}", f"Outer: f = {pr(o)}, df/du = {pr(do)}",
             f"f'(x) = (df/du)·u' = " + " = ".join(dict.fromkeys([mid, res]))]
    d_check(fs_, [mid, res])
    d_check(pr(g), [pr(dg)])
    return f"Differentiate f(x) = {fs_} using the chain rule.", lines, f"f'(x) = {res}"


def fam_diff(R):
    k = R.choice(["sum", "sum", "prod", "prod", "quot", "chain", "chain", "chain"])
    return {"sum": dd_sum, "prod": dd_prod, "quot": dd_quot, "chain": dd_chain}[k](R)

# ------------------------------------------------------------------------------------------ 12. integration


def i_check(F_s, integrand_s):
    """d/dx of displayed antiderivative equals displayed integrand at sample points."""
    Fd = sp.diff(sym(F_s), XS)
    for p in PTS:
        need(eq(Fd.subs(XS, p), E(integrand_s, {"x": p})), f"integral {F_s} vs {integrand_s}")


def noabs(e):
    return e.replace(sp.Abs, lambda a: a)


def it_sum(R):
    n = rint(R, 2, 4)
    terms = []
    for _ in range(n):
        c = rint(R, -9, 9, True)
        b = R.choice(["pow", "pow", "pow", "pow", "const", "inv", "exp", "sin", "cos", "sqrt", "inv2"])
        t = {"pow": XS ** rint(R, 1, 6), "const": sp.Integer(1), "inv": 1 / XS, "exp": sp.exp(XS), "sin": sp.sin(XS),
             "cos": sp.cos(XS), "sqrt": sp.sqrt(XS), "inv2": XS ** -2}[b]
        terms.append(c * t)
    f = sp.Add(*terms)
    want(len(f.as_ordered_terms()) >= 2)
    lines, Fs = [], []
    for t in f.as_ordered_terms():
        Ft = sp.integrate(t, XS)
        if t.has(1 / XS) and not t.has(XS ** -2):
            Ft = Ft.replace(sp.log, lambda a: sp.log(sp.Abs(a)))
        c, rest = t.as_coeff_Mul()
        segs = []
        if rest.is_Pow and rest.base == XS and rest.exp != -1:
            k = rest.exp + 1
            segs.append(f"{atom(c)}·x^{k if k.is_Integer and k > 0 else '(' + pr(k) + ')'}/{atom(k)}")
        segs.append(pr(Ft))
        i_check(pr(Ft), pr(t))
        for g in segs:
            ident(g, pr(Ft))
        lines.append(f"∫ {pr(t)} dx = " + " = ".join(dict.fromkeys(segs)))
        Fs.append(Ft)
    Ftot = sp.Add(*Fs)
    i_check(pr(Ftot), pr(f))
    back = pr(sp.diff(noabs(Ftot), XS))
    ident(back, pr(f))
    lines.append(f"F(x) = {pr(Ftot)} + C")
    lines.append(f"Check: d/dx[{pr(Ftot)}] = {back} ✓")
    return f"Find ∫ ({pr(f)}) dx.", lines, f"{pr(Ftot)} + C"


def it_usub(R):
    g = R.choice([rint(R, 2, 5) * XS + rint(R, -6, 6, True),
                  XS ** 2 + rint(R, 1, 6), rint(R, 2, 3) * XS ** 2 + rint(R, 1, 6), XS ** 3 + rint(R, 1, 5)])
    U = sp.Symbol("u")
    h = R.choice([U ** rint(R, 2, 5), sp.exp(U), sp.cos(U), sp.sin(U), 1 / U, sp.sqrt(U)])
    dg = sp.diff(g, XS)
    cont = sp.gcd_terms(dg).as_coeff_Mul()[0]
    base = sp.simplify(dg / cont)
    s = rint(R, 1, 6)
    pref = s * base  # integrand prefactor; pref = (s/cont)·u'
    ratio = sp.Rational(s) / cont
    integrand = pref * h.subs(U, g)
    Hu = sp.integrate(h, U)
    if h == 1 / U:
        Hu = sp.log(sp.Abs(U))
    Fx = (ratio * Hu).subs(U, g)
    integ_s = mulstr(pr(pref) if not pref.is_Add else f"({pr(pref)})", par(h.subs(U, g))) if pref != 1 else pr(h.subs(U, g))
    if h == 1 / U:
        integ_s = f"{par(pref)}/({pr(g)})"
    lines = [f"u = {pr(g)}, du = {pr(dg)} dx"]
    if ratio != 1:
        lines.append(f"{pr(pref)} dx = {atom(ratio)}({pr(dg)} dx) = {atom(ratio)} du")
        ident(pr(pref), f"{atom(ratio)}({pr(dg)})")
    hs = pr(h)
    inner = f"∫ {hs} du" if ratio == 1 else f"{atom(ratio)}∫ {hs} du"
    Hs = pr(ratio * Hu)
    lines.append(f"∫ {integ_s} dx = {inner} = {Hs} + C")
    Fs = pr(Fx)
    lines.append(f"= {Fs} + C")
    i_check(Fs, integ_s)
    back = pr(sp.diff(noabs(Fx), XS))
    ident(back, integ_s)
    lines.append(f"Check: d/dx[{Fs}] = {back} ✓")
    return f"Find ∫ {integ_s} dx.", lines, f"{Fs} + C"


def it_def(R):
    d = rpoly(R, R.choice([1, 2, 3]), -6, 6)
    a, b = sorted(R.sample(range(-3, 5), 2))
    Fd = {e + 1: Fr(c, e + 1) for e, c in d.items() if c}
    sb, vb = sub(P1(Fd), {"x": b})
    sa, va = sub(P1(Fd), {"x": a})
    res = vb - va
    need(sp.integrate(sym(poly(d)), (XS, a, b)) == sp.Rational(res.numerator, res.denominator))
    i_check(poly(Fd), poly(d))
    lines = [f"F(x) = {poly(Fd)}", f"F({b}) = {sb}", f"F({a}) = {sa}",
             f"∫ from {a} to {b} = F({b}) - F({a}) = " + chain(f"{fs(vb)} - {atom(va)}", fs(res))]
    return f"Evaluate the integral of {poly(d)} from x = {a} to x = {b}.", lines, fs(res)


def fam_integ(R):
    k = R.choice(["sum", "sum", "usub", "usub", "usub", "def"])
    return {"sum": it_sum, "usub": it_usub, "def": it_def}[k](R)

# ------------------------------------------------------------------------------------------------- 13. Euler
EULER_F = ["x + y", "x - y", "2x + y", "y - x", "xy", "y", "-2y", "x^2 + y", "1 - y", "x - 2y", "3 - y", "x^2 - y",
           "2y - x", "y + 1"]


def dec(q, nd=4):
    from decimal import Decimal, ROUND_HALF_UP
    q = F(q)
    d = (Decimal(q.numerator) / Decimal(q.denominator)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP)
    s = format(d.normalize(), "f")
    return "0" if s in ("-0", "0") else s


def fam_euler(R):
    fx = R.choice(EULER_F)
    h = R.choice([Fr(1, 10), Fr(1, 10), Fr(1, 5), Fr(1, 4), Fr(1, 2)])
    x0 = Fr(R.choice([0, 0, 0, 1, -1, 2]))
    y0 = Fr(R.choice([1, 1, 2, 0, -1, 3])) if R.random() < 0.8 else Fr(R.choice([1, 3, -1]), 2)
    n = rint(R, 3, 5) if h < Fr(1, 2) else rint(R, 2, 4)
    xt = x0 + n * h
    rows = ["n | x_n | y_n | f(x_n, y_n) | y_{n+1} = y_n + h·f"]
    xs, ys = x0, y0
    rounded = False
    for i in range(n):
        fv = E(fx, {"x": xs, "y": ys})
        fv = F(fv)
        want(Fr(dec(fv, 8)) == fv)
        yn = ys + h * fv
        yd = dec(yn)
        if Fr(yd) != yn:
            rounded = True
        need(abs(Fr(yd) - yn) <= Fr(1, 20000))
        need(eq(E(dec(ys) + " + " + dec(h) + "(" + dec(fv, 8) + ")"), yn))
        rows.append(f"{i} | {dec(xs)} | {dec(ys)} | {dec(fv, 8)} | {yd}")
        xs, ys = xs + h, Fr(yd)
    need(xs == xt)
    lines = [f"f(x, y) = {fx}, h = {dec(h)}, {n} steps from x = {dec(x0)} to x = {dec(xt)}."] + rows
    if rounded:
        lines.append("(values rounded to 4 decimal places)")
    lines.append(f"y({dec(xt)}) ≈ {dec(ys)}")
    return (f"Use Euler's method with h = {dec(h)} to approximate y({dec(xt)}) for y' = {fx}, y({dec(x0)}) = {dec(y0)}.",
            lines, f"y({dec(xt)}) ≈ {dec(ys)}")

# ================================================================================================== driver
FAMILIES = [("subst", fam_subst, 1.5), ("solve", fam_solve, 1.5), ("func", fam_func, 1), ("count", fam_count, 1),
            ("numth", fam_numth, 1), ("exprad", fam_exprad, 1), ("seq", fam_seq, 1), ("ineq", fam_ineq, 1),
            ("coord", fam_coord, 1), ("polydiv", fam_div, 1), ("diff", fam_diff, 1), ("integ", fam_integ, 1),
            ("euler", fam_euler, 1)]


# Statement forms reserved for RL tasks: rows matching any of these are rejected.
RESERVED_FORMS = {k: re.compile(v) for k, v in {
    'sum2sq': 'x\\^\\{?2\\}?\\s*\\+\\s*y\\^\\{?2\\}?\\s*=\\s*\\d+\\b(?!\\s*[a-z(])',
    'x2+2y2': 'x\\^\\{?2\\}?\\s*\\+\\s*2\\s*y\\^\\{?2\\}?\\s*=\\s*\\d+',
    'sum3sq': 'x\\^\\{?2\\}?\\s*\\+\\s*y\\^\\{?2\\}?\\s*\\+\\s*z\\^\\{?2\\}?\\s*=\\s*\\d+',
    'cubes': 'x\\^\\{?3\\}?\\s*\\+\\s*y\\^\\{?3\\}?\\s*\\+\\s*z\\^\\{?3\\}?\\s*=\\s*\\d+',
    'linear_int': '\\b\\d+\\s*x\\s*\\+\\s*\\d+\\s*y\\s*(\\+\\s*\\d+\\s*z\\s*)?=\\s*\\d+',
    'ax2+by2': '\\b\\d+\\s*x\\^\\{?2\\}?\\s*\\+\\s*\\d+\\s*y\\^\\{?2\\}?\\s*=\\s*\\d+',
}.items()}


def render(problem, lines, answer):
    return "\n".join(lines) + f"\nAnswer: {answer}"


def run_family(args):
    name, fn, budget, seed, tokpath, max_rows = args
    R = random.Random(f"{seed}-{name}")
    tk = None
    if tokpath:
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(tokpath)
    st = collections.Counter()
    seen, rows, pend, tot = set(), [], [], 0

    def flush():
        nonlocal tot
        if not pend:
            return
        if tk is not None:
            pe = tk.encode_batch([f"Problem:\n{r['problem']}\n\n" for r in pend])
            re_ = tk.encode_batch([r["trace"] for r in pend])
            lens = [(len(p.ids), len(q.ids)) for p, q in zip(pe, re_)]
        else:
            lens = [(len(r["problem"]) // 2, len(r["trace"]) // 2) for r in pend]
        for r, (lp, lq) in zip(pend, lens):
            r["n_prompt"] = 1 + lp
            r["n_tokens"] = 2 + lp + lq
            tot += r["n_tokens"]
            rows.append(r)
        pend.clear()

    attempts = 0
    while tot < budget and len(rows) + len(pend) < max_rows:
        attempts += 1
        if attempts > 60 * max(1, budget // 50):
            break
        st["attempts"] += 1
        try:
            pb, lines, ans = fn(R)
        except Retry:
            st["retry"] += 1
            continue
        except VerifyError as e:
            st["verify_fail"] += 1
            if st["verify_fail"] <= 5:
                st[f"vf:{str(e)[:80]}"] += 1
            continue
        except (ZeroDivisionError, ValueError, TypeError, KeyError, IndexError, OverflowError, RecursionError) as e:
            st["error"] += 1
            st[f"err:{type(e).__name__}:{str(e)[:60]}"] += 1
            continue
        text = render(pb, lines, ans)
        if pb in seen:
            st["dup"] += 1
            continue
        if any(p.search(pb) or p.search(text) for p in RESERVED_FORMS.values()):
            st["reserved_reject"] += 1
            continue
        seen.add(pb)
        h = hashlib.md5(pb.encode()).hexdigest()[:16]
        pend.append({"problem_id": f"mp1-{name}-{h}", "source": "math-procedures-v1", "slice": "MP",
                     "teacher": "synthetic", "origin": "synth", "family": name, "problem": pb, "trace": text,
                     "thinking": "", "solution": text, "answer": ans, "answer_teacher": ans, "ref_raw": ans,
                     "task": name})
        if len(pend) >= 512:
            flush()
    flush()
    st["kept"] = len(rows)
    return name, rows, dict(st)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tok", default="")
    ap.add_argument("--target", type=float, default=10e6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--families", default="")
    ap.add_argument("--max-rows", type=int, default=10 ** 9)
    a = ap.parse_args()
    sys.setrecursionlimit(10000)
    fams = [f for f in FAMILIES if not a.families or f[0] in a.families.split(",")]
    wsum = sum(w for _, _, w in FAMILIES)
    jobs = [(n, fn, int(a.target * w / wsum), a.seed, a.tok, a.max_rows) for n, fn, w in fams]
    if a.procs > 1:
        import multiprocessing as mp
        with mp.Pool(a.procs) as pool:
            res = pool.map(run_family, jobs, chunksize=1)
    else:
        res = [run_family(j) for j in jobs]
    os.makedirs(a.out, exist_ok=True)
    allrows, stats = [], {"seed": a.seed, "target_tokens": a.target, "tokenizer": a.tok, "families": {}}
    for name, rows, st in res:
        allrows += rows
        ver = st.get("verify_fail", 0) + st.get("error", 0)
        gen_ok = st.get("attempts", 0) - st.get("retry", 0)
        stats["families"][name] = {"rows": len(rows), "tokens": sum(r["n_tokens"] for r in rows),
                                   "resp_tokens": sum(r["n_tokens"] - r["n_prompt"] - 1 for r in rows),
                                   "verify_drops": ver, "verify_drop_rate": round(ver / max(1, gen_ok), 5),
                                   "dups": st.get("dup", 0), "reserved_rejects": st.get("reserved_reject", 0),
                                   "constraint_retries": st.get("retry", 0),
                                   "drop_examples": {k: v for k, v in st.items() if k.startswith(("vf:", "err:"))}}
    random.Random(a.seed).shuffle(allrows)
    path = os.path.join(a.out, "train.jsonl")
    with open(path, "w") as f:
        for r in allrows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    md5 = hashlib.md5(open(path, "rb").read()).hexdigest()
    stats.update(total_rows=len(allrows), total_tokens=sum(r["n_tokens"] for r in allrows), md5=md5)
    json.dump(stats, open(os.path.join(a.out, "stats.json"), "w"), indent=1, ensure_ascii=False)
    with open(os.path.join(a.out, "train.jsonl.md5"), "w") as f:
        f.write(f"{md5}  train.jsonl\n")
    print(json.dumps({k: v for k, v in stats.items() if k != "families"}))
    for n, s in stats["families"].items():
        print(n, s["rows"], s["tokens"], "vdrop", s["verify_drops"], s["verify_drop_rate"], "dup", s["dups"],
              "reserved", s["reserved_rejects"], s["drop_examples"])


if __name__ == "__main__":
    main()
