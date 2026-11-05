"""arith-steps-v1: plain-text midtrain slice of worked multi-digit arithmetic (no <think>, no problem/answer split).

Each document is a run of 5-20 worked problems separated by blank lines. Each problem is `Compute <expr>.`, a few short
steps in one fixed format per operation, then the result in the inline trace form `<expr> = <result>`:
- add / sub: decimal points aligned (scaled to integers), one line per column right to left with the digit sum and
  carry (add) or the borrow (sub); signs are rewritten first (`-45 + 120` -> `120 - 45`).
- mul: points dropped and decimal places counted; a one-digit multiplier gets a per-digit carry chain, a longer one
  gets one partial product per multiplier digit (shifted) and a running sum of the partials.
- div: a decimal divisor is made whole by shifting both points; long division one quotient digit per line
  (`15 into 142: 9 times, 9 * 15 = 135, 142 - 135 = 7`); ends exact, with a remainder, or rounded to 2 places.
- frac: gcd by Euclid, lcm = a * b / gcd, rewrite over the common denominator, combine, simplify via gcd.
- pow: repeated multiplication.  ooo: one reduction per line (parentheses, powers, * and / left to right, then + and -
  left to right).  mod: a mod m by division with remainder; sums, products and powers reduced mod m step by step.
Every intermediate value is asserted against exact Python arithmetic.

Operand lengths for add/sub/mul/div/pow follow an operand-shape histogram of the arithmetic in midtrain reasoning traces
(bundled operand_dist.json; key op|int digits a|dec places a|int digits b|dec places b), tilted
toward longer numbers by weight 1.3^(min(L, 8) - 1), L = the longer operand's digit count; shapes with an operand over
10 digits are dropped.

usage: python gen_arith_steps.py [--dist operand_dist.json] --out DIR [--tok tokenizer.json] [--target-tokens 5e6]
Pure Python (the optional tokenizer is only used to count tokens and stop at the target). Seeded, deterministic.
"""
import argparse, collections, hashlib, json, math, os, random, re
from decimal import Decimal, getcontext
from fractions import Fraction

getcontext().prec = 80
OP_SHARE = {"add": 0.18, "sub": 0.16, "mul": 0.24, "div": 0.14, "pow": 0.07, "frac": 0.08, "ooo": 0.08, "mod": 0.05}
TILT, MAX_LEN = 1.3, 10


def ds(d):
    """Decimal -> plain string, no exponent, no trailing zeros."""
    s = format(d, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def scale_of(d):
    s = ds(d)
    return len(s.split(".")[1]) if "." in s else 0


def rnum(rng, ia, fa):
    """Random non-negative Decimal with ia integer digits (no leading zero unless ia == 1) and exactly fa places."""
    ip = str(rng.randint(1, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(ia - 1)) if ia > 1 else str(
        rng.randint(0 if fa else 1, 9))
    if fa:
        fp = "".join(str(rng.randint(0, 9)) for _ in range(fa - 1)) + str(rng.randint(1, 9))
        return Decimal(ip + "." + fp)
    return Decimal(ip)


def neg_s(d):
    return f"({ds(d)})" if d < 0 else ds(d)


# ---------------------------------------------------------------- column add / sub on non-negative integers
def col_add(x, y):
    xs, ys = str(x), str(y)
    n = max(len(xs), len(ys))
    xs, ys = xs.zfill(n), ys.zfill(n)
    lines, c, out = [], 0, []
    for i in range(n - 1, -1, -1):
        a, b = int(xs[i]), int(ys[i])
        s = a + b + c
        lines.append(f"{a} + {b} + {c} = {s} -> {s % 10}, carry {s // 10}")
        out.append(s % 10); c = s // 10
    if c:
        lines.append(f"final carry {c} -> {c}"); out.append(c)
    digits = "".join(map(str, reversed(out)))
    assert int(digits) == x + y
    return lines, digits


def col_sub(x, y):
    assert x >= y >= 0
    xs = str(x); ys = str(y).zfill(len(xs))
    lines, br, out = [], 0, []
    for i in range(len(xs) - 1, -1, -1):
        a, b = int(xs[i]), int(ys[i])
        if a - br >= b:
            d = a - br - b
            lines.append(f"{a} - {br} - {b} = {d} -> {d}, borrow 0"); nbr = 0
        else:
            d = a + 10 - br - b
            lines.append(f"{a} - {br} - {b} < 0, borrow: {a + 10} - {br} - {b} = {d} -> {d}, borrow 1"); nbr = 1
        out.append(d); br = nbr
    assert br == 0
    digits = "".join(map(str, reversed(out)))
    assert int(digits) == x - y
    return lines, digits


def scaled_pair(a, b):
    k = max(scale_of(a), scale_of(b))
    return int(a * 10 ** k), int(b * 10 ** k), k


def place_point(digits, k, value):
    """'Digits 3265 with 1 decimal place: 326.5' line, asserting the value."""
    d = Decimal(int(digits)) / (10 ** k)
    assert d == value, (digits, k, value)
    if k == 0:
        return (f"Digits {digits} -> {ds(d)}" if digits != ds(d) else None), d
    return f"Digits {digits} with {k} decimal place{'s' if k > 1 else ''}: {ds(d)}", d


def magnitude_addsub(x, y, op):
    """x op y for x, y >= 0 Decimals (sub requires x >= y). -> lines."""
    X, Y, k = scaled_pair(x, y)
    lines = []
    if k:
        lines.append(f"Align the decimal points: {ds(x)} {op} {ds(y)} -> {X} {op} {Y} ({k} decimal place{'s' if k > 1 else ''}).")
    cl, digits = col_add(X, Y) if op == "+" else col_sub(X, Y)
    lines += ["  " + l for l in cl]
    pl, v = place_point(digits, k, x + y if op == "+" else x - y)
    if pl:
        lines.append(pl)
    return lines, v


def addsub_steps(a, b, op):
    """Signed a op b -> (lines, value); the sign rewrite reduces it to magnitudes."""
    val = a + b if op == "+" else a - b
    A, B = abs(a), abs(b)
    pos_b = b >= 0 if op == "+" else b < 0  # effective sign of the second term
    lines = []
    if a >= 0 and pos_b:
        red, neg = (A, B, "+"), False
        if op == "-" or b < 0:
            lines.append(f"Rewrite: {ds(a)} {op} {neg_s(b)} = {ds(A)} + {ds(B)}.")
    elif a < 0 and not pos_b:
        red, neg = (A, B, "+"), True
        lines.append(f"Rewrite: {ds(a)} {op} {neg_s(b)} = -({ds(A)} + {ds(B)}).")
    else:  # opposite signs: larger magnitude minus smaller
        pos_first = a >= 0
        big, small = (A, B) if A >= B else (B, A)
        neg = (A > B and not pos_first) or (B > A and pos_first)
        if a >= 0 and op == "-" and b >= 0:
            if A < B:
                lines.append(f"{ds(B)} > {ds(A)}, so compute {ds(B)} - {ds(A)} and make it negative.")
        else:
            lines.append(f"Rewrite: {ds(a)} {op} {neg_s(b)} = " + (f"-({ds(big)} - {ds(small)})." if neg else f"{ds(big)} - {ds(small)}."))
        red = (big, small, "-")
    ml, v = magnitude_addsub(*red)
    lines += ml
    v = -v if neg else v
    assert v == val
    return lines, val


# ---------------------------------------------------------------- multiplication
def mul_chain(x, d):
    """x * d (d one digit) per-digit carry chain on integers."""
    xs = str(x); lines, c, out = [], 0, []
    for i in range(len(xs) - 1, -1, -1):
        p = int(xs[i]) * d + c
        lines.append(f"{xs[i]} * {d} + {c} = {p} -> {p % 10}, carry {p // 10}")
        out.append(p % 10); c = p // 10
    if c:
        lines.append(f"final carry {c} -> {c}"); out.append(c)
    digits = "".join(map(str, reversed(out))).lstrip("0") or "0"
    assert int(digits) == x * d
    return lines, digits


def mul_steps(a, b):
    val = a * b
    sign = "-" if (a < 0) != (b < 0) and val != 0 else ""
    A, B = abs(a), abs(b)
    ka, kb = scale_of(A), scale_of(B)
    X, Y = int(A * 10 ** ka), int(B * 10 ** kb)
    lines = []
    if sign or a < 0 or b < 0:
        lines.append(f"Signs: the product is {'negative' if sign else 'positive'}; multiply {ds(A)} * {ds(B)}.")
    if ka or kb:
        lines.append(f"Drop the decimal points: {ds(A)} * {ds(B)} -> {X} * {Y} ({ka} + {kb} = {ka + kb} decimal places).")
    if len(str(X)) < len(str(Y)):
        X, Y = Y, X
        lines.append(f"Use the shorter number as the multiplier: {X} * {Y}.")
    if Y < 10:
        cl, digits = mul_chain(X, Y)
        lines += ["  " + l for l in cl]
    else:
        lines.append(f"Partial products of {X} by each digit of {Y}:")
        parts = []
        ys = str(Y)
        for j, ch in enumerate(reversed(ys)):
            p = X * int(ch)
            sh = p * 10 ** j
            if j == 0:
                lines.append(f"  {X} * {ch} = {p}")
            else:
                lines.append(f"  {X} * {ch} = {p}, shift {j}: {sh}")
            parts.append(sh)
        nz = [p for p in parts if p]
        run = nz[0] if nz else 0
        if len(nz) > 1:
            lines.append("Sum the partial products:")
            for p in nz[1:]:
                lines.append(f"  {run} + {p} = {run + p}")
                run += p
        assert run == X * Y
        digits = str(run)
    k = ka + kb
    pl, v = place_point(digits, k, A * B)
    if pl:
        lines.append(pl)
    v = -v if sign else v
    assert v == val
    return lines, val


# ---------------------------------------------------------------- long division
def div_steps(a, b, mode, rng):
    """a / b, b != 0. mode: exact | rem | round. -> lines, value string."""
    lines = []
    A, B = abs(a), abs(b)
    sign = "-" if (a < 0) != (b < 0) and a != 0 else ""
    if a < 0 or b < 0:
        lines.append(f"Signs: the quotient is {'negative' if sign else 'positive'}; divide {ds(A)} / {ds(B)}.")
    kb = scale_of(B)
    if kb:
        A2, B2 = A * 10 ** kb, B * 10 ** kb
        lines.append(f"Shift both points {kb} place{'s' if kb > 1 else ''}: {ds(A)} / {ds(B)} = {ds(A2)} / {ds(B2)}.")
    else:
        A2, B2 = A, B
    ka = scale_of(A2)
    X, Y = int(A2 * 10 ** ka), int(B2)
    # digits of X are processed left to right; the quotient point sits after len(int part of A2) digits
    xs = str(X).zfill(ka + 1)
    n_int = len(xs) - ka
    extra = {"exact": 6, "rem": 0, "round": 3}[mode]
    if mode == "rem":
        assert ka == 0
    q, rem, i = [], 0, 0
    while i < len(xs):
        rem = rem * 10 + int(xs[i]); i += 1
        if rem >= Y or i >= n_int:
            break
    while True:
        qd = rem // Y; r = rem - qd * Y
        assert 0 <= r < Y and qd <= 9
        if qd:
            lines.append(f"  {Y} into {rem}: {qd} times, {qd} * {Y} = {qd * Y}, {rem} - {qd * Y} = {r}")
        else:
            lines.append(f"  {Y} into {rem}: 0 times, remainder {rem}")
        q.append(qd); rem = r
        if i < len(xs) or (rem and extra):
            if i < len(xs):
                d = xs[i]; i += 1
                pt = "write the point, " if i == n_int + 1 else ""
            else:
                d = "0"; extra -= 1; i += 1
                pt = "write the point, " if i == n_int + 1 else ""
            rem = rem * 10 + int(d)
            lines.append(f"  {pt}bring down {d}: {rem}")
            continue
        break
    # quotient: digits q correspond to positions ending at index i-1 of the (extended) digit string
    n_dec = max(0, i - n_int)
    qs = "".join(map(str, q))
    qv = Decimal(int(qs)) / (10 ** n_dec)
    exact_v = Decimal(A2) / Decimal(B2)
    if rem == 0:
        assert qv == exact_v, (a, b, qs, n_dec)
        lines.append(f"Quotient {ds(qv)}, remainder 0.")
        res = sign + ds(qv)
    elif mode == "rem":
        assert int(qs) * Y + rem == X
        lines.append(f"Quotient {ds(qv)}, remainder {rem}.")
        res = f"{sign}{ds(qv)} remainder {rem}" if not sign else None
    else:
        assert n_dec == 3 and abs(qv - exact_v) < Decimal("0.001")
        rd = qv.quantize(Decimal("0.01"), rounding="ROUND_HALF_UP")
        # rounding from 3 truncated places is correct unless the 3rd digit is 5 with nothing after; avoid ambiguity
        true_rd = exact_v.quantize(Decimal("0.01"), rounding="ROUND_HALF_UP")
        if rd != true_rd:
            return None
        lines.append(f"Quotient so far {format(qv, 'f')}; round to 2 decimal places: {format(rd, 'f')}.")
        return lines, f"{ds(a)} / {neg_s(b)} ≈ {sign}{format(rd, 'f')}"
    if res is None:
        return None
    return lines, f"{ds(a)} / {neg_s(b)} = {res}"


# ---------------------------------------------------------------- fractions, gcd
def euclid(a, b, lines):
    """gcd(a, b) with Euclid steps appended to lines (a, b > 0)."""
    x, y = max(a, b), min(a, b)
    steps = []
    while y:
        q, r = divmod(x, y)
        steps.append(f"{x} = {q} * {y} + {r}")
        x, y = y, r
    assert x == math.gcd(a, b)
    lines.append(f"gcd({a}, {b}): " + ", ".join(steps) + f", so gcd = {x}")
    return x


def simplify(n, d, lines):
    """n/d (d > 0) -> simplified Fraction, with steps."""
    g = euclid(abs(n), d, lines) if n else d
    if n == 0:
        lines.append(f"{n}/{d} = 0"); return Fraction(0)
    if g > 1:
        lines.append(f"Divide by {g}: {n} / {g} = {n // g}, {d} / {g} = {d // g}, giving {n // g}/{d // g}")
    return Fraction(n // g, d // g)


def fs(f):
    return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"


def frac_problem(rng):
    op = rng.choice(["+", "+", "-", "-", "*", "/"])
    def rf():
        d = rng.choice([rng.randint(2, 12), rng.randint(2, 12), rng.randint(2, 30), rng.randint(10, 99)])
        return rng.randint(1, max(1, int(d * rng.choice([0.9, 1.5, 3])))), d
    (n1, d1), (n2, d2) = rf(), rf()
    expr = f"{n1}/{d1} {op} {n2}/{d2}"
    val = {"+": Fraction(n1, d1) + Fraction(n2, d2), "-": Fraction(n1, d1) - Fraction(n2, d2),
           "*": Fraction(n1, d1) * Fraction(n2, d2), "/": Fraction(n1, d1) / Fraction(n2, d2)}[op]
    L = []
    if op in "+-":
        g = euclid(d1, d2, L)
        l = d1 * d2 // g
        L.append(f"lcm = {d1} * {d2} / {g} = {l}")
        m1, m2 = l // d1, l // d2
        a1, a2 = n1 * m1, n2 * m2
        L.append(f"{n1}/{d1} = {a1}/{l} (times {m1}), {n2}/{d2} = {a2}/{l} (times {m2})")
        n = a1 + a2 if op == "+" else a1 - a2
        L.append(f"{a1}/{l} {op} {a2}/{l} = {n}/{l}")
        if n < 0:
            L.append(f"The result is negative: -{-n}/{l}")
        f = simplify(abs(n), l, L)
        f = -f if n < 0 else f
    else:
        if op == "/":
            L.append(f"Dividing by {n2}/{d2} is multiplying by {d2}/{n2}.")
            n2, d2 = d2, n2
        n, d = n1 * n2, d1 * d2
        L.append(f"{n1}/{d1} * {n2}/{d2} = ({n1} * {n2})/({d1} * {d2}) = {n}/{d}")
        f = simplify(n, d, L)
    assert f == val
    return expr, L, fs(f), "frac"


# ---------------------------------------------------------------- order of operations
def ooo_problem(rng):
    for _ in range(200):
        n_ops = rng.randint(2, 5)
        def num():
            return rng.choice([rng.randint(1, 9), rng.randint(2, 20), rng.randint(10, 99), rng.randint(2, 12),
                               rng.randint(100, 999)])
        toks = [num()]
        for _ in range(n_ops):
            o = rng.choices(["+", "-", "*", "/", "^"], [3, 3, 3, 2, 0 if toks[-2:-1] == ["^"] else 1])[0]
            if o == "^":
                toks += ["^", rng.choice([2, 2, 2, 3])]
            else:
                toks += [o, num()]
        # optional parentheses around a + / - sub-run
        if rng.random() < 0.7:
            idx = [j for j in range(0, len(toks) - 2, 2) if toks[j + 1] in "+-" and isinstance(toks[j + 2], int)]
            if idx:
                j = rng.choice(idx)
                toks = toks[:j] + ["("] + toks[j:j + 3] + [")"] + toks[j + 3:]
        res = ooo_eval(toks)
        if res is not None:
            return res
    return None


def tstr(toks):
    s = ""
    for t in toks:
        if t == "(":
            s += " ("
        elif t == ")":
            s += ")"
        elif t == "^":
            s += "^"
        elif isinstance(t, int):
            s += (str(t) if s.endswith(("(", "^")) or not s else " " + str(t))
        else:
            s += " " + t
    return s.strip()


def py_eval(toks):
    s = "".join(f"Fraction({t})" if isinstance(t, int) else ("**" if t == "^" else t) for t in toks)
    return eval(s, {"Fraction": Fraction})


def ooo_eval(toks):
    expr = tstr(toks)
    try:
        target = py_eval(toks)
    except (ZeroDivisionError, OverflowError):
        return None
    if target.denominator != 1 or target < 0 or abs(target) > 10 ** 7:
        return None
    L, t = [], list(toks)
    while len(t) > 1:
        # innermost parenthesised group
        if "(" in t:
            i = max(j for j, x in enumerate(t) if x == "(")
            j = t.index(")", i)
            lo, hi, kind = i + 1, j, "Parentheses"
        else:
            lo, hi, kind = 0, len(t), None
        seg = t[lo:hi]
        if "^" in seg:
            k = lo + max(m for m, x in enumerate(seg) if x == "^"); k0 = "Power"
        elif any(x in ("*", "/") for x in seg):
            k = lo + min(m for m, x in enumerate(seg) if x in ("*", "/")); k0 = "Multiply/divide"
        else:
            k = lo + min(m for m, x in enumerate(seg) if x in ("+", "-")); k0 = "Add/subtract"
        a, o, b = t[k - 1], t[k], t[k + 1]
        if not (isinstance(a, int) and isinstance(b, int)):
            return None
        if o == "+":
            c = a + b
        elif o == "-":
            c = a - b
        elif o == "*":
            c = a * b
        elif o == "/":
            if b == 0 or a % b:
                return None
            c = a // b
        else:
            if a > 20:
                return None
            c = a ** b
        if o == "*" and min(a, b) >= 100:
            return None
        if c < 0 or c > 10 ** 7:
            return None
        t = t[:k - 1] + [c] + t[k + 2:]
        # drop parentheses around a single number
        changed = True
        while changed:
            changed = False
            for m in range(len(t) - 2):
                if t[m] == "(" and isinstance(t[m + 1], int) and t[m + 2] == ")":
                    t = t[:m] + [t[m + 1]] + t[m + 3:]; changed = True; break
        line = f"{kind or k0}: {a} {o} {b} = {c}" if o != "^" else f"{kind or k0}: {a}^{b} = " + \
            (f"{a} * {a} = {c}" if b == 2 else f"{a} * {a} * {a} = {c}")
        L.append(line + (f", giving {tstr(t)}" if len(t) > 1 else ""))
    assert t[0] == target, (expr, t, target)
    if len(L) < 2:
        return None
    return expr, L, str(t[0]), "ooo"


# ---------------------------------------------------------------- powers and mod
def pow_problem(rng, shape):
    ia, fa, ib, _ = shape
    for _ in range(100):
        base = rnum(rng, min(ia, 3), min(fa, 1))
        e = rng.choice([2, 2, 2, 3, 3, 4, 5]) if ib == 1 else rng.randint(4, 12)
        if base != int(base):
            e = min(e, 3)
        if base <= 1 or e < 2:
            continue
        v = base ** e
        if len(ds(v).replace(".", "")) > 12:
            continue
        L, run = [], base
        for _ in range(e - 1):
            nxt = run * base
            L.append(f"{ds(run)} * {ds(base)} = {ds(nxt)}")
            run = nxt
        assert run == v
        return f"{ds(base)}^{e}", L, ds(v), "pow"
    return None


def mod_problem(rng):
    m = rng.choice([rng.randint(2, 12), rng.randint(2, 20), rng.randint(7, 50)])
    kind = rng.choice(["plain", "plain", "sum", "prod", "pow"])
    L = []

    def red(x):
        q, r = divmod(x, m)
        if x < m:
            L.append(f"{x} mod {m} = {x}")
        elif q >= 10:  # multi-digit quotient: long division
            dl, _ = div_steps(Decimal(x), Decimal(m), "rem", rng)
            L.append(f"{x} mod {m}: divide {x} by {m}")
            L.extend(dl)
            L.append(f"so {x} mod {m} = {r}")
        else:
            L.append(f"{x} mod {m}: {m} into {x}: {q} times, {q} * {m} = {q * m}, {x} - {q * m} = {r}, so {x} mod {m} = {r}")
        return r
    if kind == "plain":
        x = rng.randint(m + 1, 10 ** rng.randint(2, 5))
        r = red(x); expr, val = f"{x} mod {m}", x % m
    elif kind in ("sum", "prod"):
        x, y = rng.randint(m, 10 ** rng.randint(2, 4)), rng.randint(m, 10 ** rng.randint(2, 4))
        o = "+" if kind == "sum" else "*"
        rx, ry = red(x), red(y)
        s = rx + ry if o == "+" else rx * ry
        L.append(f"{rx} {o} {ry} = {s}")
        r = red(s); expr, val = f"({x} {o} {y}) mod {m}", (x + y if o == "+" else x * y) % m
    else:
        b, e = rng.randint(2, 30), rng.randint(3, 9)
        rb = red(b) if b >= m else b
        run = rb
        for _ in range(e - 1):
            p = run * rb
            L.append(f"{run} * {rb} = {p}")
            run = red(p)
        r = run; expr, val = f"{b}^{e} mod {m}", pow(b, e, m)
    assert r == val
    return expr, L, str(r), "mod"


# ---------------------------------------------------------------- shapes, problems, documents
def load_shapes(path):
    h = json.load(open(path))["hist"]
    sh = collections.defaultdict(list)
    for k, v in h.items():
        o, ia, fa, ib, fb = k.split("|"); ia, fa, ib, fb = int(ia), int(fa), int(ib), int(fb)
        if ia + fa > MAX_LEN or ib + fb > MAX_LEN:
            continue
        L = max(ia + fa, ib + fb)
        sh[o].append(((ia, fa, ib, fb), v * TILT ** (min(L, 8) - 1)))
    return {o: ([s for s, _ in v], [w for _, w in v]) for o, v in sh.items()}


def make_problem(rng, op, shapes):
    if op == "frac":
        return frac_problem(rng)
    if op == "ooo":
        return ooo_problem(rng)
    if op == "mod":
        return mod_problem(rng)
    key = {"add": "+", "sub": "-", "mul": "*", "div": "/", "pow": "^"}[op]
    S, W = shapes[key]
    shape = rng.choices(S, W)[0]
    if op == "pow":
        return pow_problem(rng, shape)
    ia, fa, ib, fb = shape
    if op in ("add", "sub", "mul"):
        if op == "mul" and (ia + fa) + (ib + fb) > 14:
            return None
        a, b = rnum(rng, ia, fa), rnum(rng, ib, fb)
        if rng.random() < 0.08:
            a = -a
        if rng.random() < 0.08:
            b = -b
        o = {"add": "+", "sub": "-", "mul": "*"}[op]
        L, v = addsub_steps(a, b, o) if op != "mul" else mul_steps(a, b)
        return f"{ds(a)} {o} {neg_s(b)}", L, ds(v), op
    # div: shape (ia, fa) / (ib, fb) -> divisor from b, dividend built from a quotient
    mode = rng.choices(["exact", "rem", "round"], [0.6, 0.25, 0.15])[0]
    ib2 = min(ib, 4)
    b = rnum(rng, ib2, min(fb, 2))
    if b == 1 or b == 0:
        return None
    if mode == "exact":
        # quotient with enough digits that a has about ia + fa digits, decimal quotient if the divisor allows it
        qd = max(1, ia - ib2 + rng.choice([0, 1, 1, 2]))
        q = rnum(rng, qd, rng.choice([0, 0, 1, 2]) if fa or rng.random() < 0.3 else 0)
        a = q * b
        if scale_of(a) > 4 or a == 0:
            return None
    else:
        a = rnum(rng, max(ia, ib2 + 1), 0)
        if mode == "rem":
            b = Decimal(int(b)) if int(b) > 1 else Decimal(rng.randint(2, 9))
            if int(a) % int(b) == 0:
                return None
    if a <= b and mode == "rem":
        return None
    if rng.random() < 0.05 and mode != "rem":
        a = -a
    r = div_steps(a, b, mode, rng)
    if r is None:
        return None
    L, res = r
    lhs, _, rhs = res.partition(" = ") if " = " in res else res.partition(" ≈ ")
    return lhs, L, rhs, "div" if " = " in res else "div~"


def render_problem(expr, L, res, kind):
    rel = "≈" if kind == "div~" else "="
    head = f"Evaluate {expr}." if kind == "ooo" else f"Compute {expr}."
    return "\n".join([head] + L + [f"{expr} {rel} {res}"])


def gen_doc(rng, shapes):
    n = rng.randint(5, 20)
    if rng.random() < 0.4:  # themed document: mostly one operation
        main = rng.choices(list(OP_SHARE), list(OP_SHARE.values()))[0]
        ops = [main if rng.random() < 0.8 else rng.choices(list(OP_SHARE), list(OP_SHARE.values()))[0] for _ in range(n)]
    else:
        ops = rng.choices(list(OP_SHARE), list(OP_SHARE.values()), k=n)
    probs, counts = [], collections.Counter()
    for op in ops:
        p = None
        while p is None:
            p = make_problem(rng, op, shapes)
        probs.append(render_problem(*p)); counts[op] += 1
    return "\n\n".join(probs), counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "operand_dist.json")); ap.add_argument("--out", required=True)
    ap.add_argument("--tok"); ap.add_argument("--target-tokens", type=float, default=5e6)
    ap.add_argument("--n-docs", type=int, default=0, help="fixed doc count (no tokenizer needed)")
    ap.add_argument("--seed", type=int, default=20261024); ap.add_argument("--name", default="arith-steps-v1")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    shapes = load_shapes(a.dist)
    tk = None
    if a.tok:
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(a.tok)
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, a.name + ".jsonl")
    tot_tok = rows = 0
    op_rows, op_tok = collections.Counter(), collections.Counter()
    hist = collections.Counter()
    with open(path, "w") as f:
        while (a.n_docs and rows < a.n_docs) or (not a.n_docs and tot_tok < a.target_tokens):
            text, counts = gen_doc(rng, shapes)
            ntok = len(tk.encode(text).ids) + 2 if tk else None  # + <eos> at both ends, as in sft.py
            # per-op token shares and operand-length histogram from the problems' result lines
            for blk in text.split("\n\n"):
                op = blk_op(blk)
                if tk:
                    op_tok[op] += len(tk.encode(blk).ids)
                for nlen in blk_operand_lens(blk):
                    hist[f"{op}|{nlen}"] += 1
            op_rows.update(counts)
            r = {"problem_id": f"{a.name}-{rows:06d}", "source": a.name, "slice": "arith", "teacher": "synthetic",
                 "origin": "synth", "problem": None, "trace": None, "thinking": None, "solution": None,
                 "answer": None, "answer_teacher": None, "ref_raw": None, "n_tokens": ntok, "task": "arith-steps",
                 "format": "plain", "text": text, "ops": dict(counts)}
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            rows += 1; tot_tok += ntok or 0
    md5 = hashlib.md5(open(path, "rb").read()).hexdigest()
    open(path + ".md5", "w").write(f"{md5}  {a.name}.jsonl\n")
    P = sum(op_rows.values()); T = sum(op_tok.values()) or 1
    st = {"name": a.name, "seed": a.seed, "rows": rows, "tokens": tot_tok, "problems": P, "md5": md5,
          "tokenizer": a.tok, "problems_per_op": dict(op_rows), "problem_share": {k: round(v / P, 4) for k, v in op_rows.items()},
          "tokens_per_op": dict(op_tok), "token_share": {k: round(v / T, 4) for k, v in op_tok.items()},
          "operand_len_hist": dict(sorted(hist.items(), key=lambda kv: (kv[0].split("|")[0], int(kv[0].split("|")[1])))),
          "operand_len_hist_note": "op|digit count (int + decimal digits) of each operand in the problem's result line "
                                   "(frac/ooo/mod: every number in the expression)",
          "op_share_target": OP_SHARE, "tilt": TILT, "max_operand_len": MAX_LEN}
    json.dump(st, open(os.path.join(a.out, "stats.json"), "w"), indent=1)
    print(json.dumps({k: st[k] for k in ("rows", "tokens", "problems", "md5", "problem_share", "token_share")}, indent=1))


def blk_op(blk):
    head = blk.split("\n", 1)[0]
    last = blk.rsplit("\n", 1)[1]
    if head.startswith("Evaluate"):
        return "ooo"
    if " mod " in head:
        return "mod"
    e = head[len("Compute "):-1]
    if "^" in e:
        return "pow"
    toks = e.split(" ")
    if len(toks) == 3 and "/" in toks[0]:
        return "frac"
    return {"+": "add", "-": "sub", "*": "mul", "/": "div"}[toks[1]] if len(toks) >= 3 else "pow"


def blk_operand_lens(blk):
    head = blk.split("\n", 1)[0]
    e = head.split(" ", 1)[1][:-1]
    return [len(x.replace(".", "").lstrip("-")) for x in re.findall(r"\d+(?:\.\d+)?", e)]


if __name__ == "__main__":
    main()
