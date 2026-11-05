"""wb_sysword: multi-unknown word problems (docs/wb_envs.md). 2-3 integer unknowns, answer = the vector of unknowns in
the order the question states. Reward = fraction of components correct (wrong length = 0).

Families / tiers (tier 0 = 2 unknowns, stated numbers <= 50; tier 1 = 2-3 unknowns, <= 200; tier 2 = 3 unknowns,
<= 1000):
    ages     2-3 people; sums, differences, "in k years / k years ago A is m times as old as B"   tiers 0, 1, 2
    coins    counts of 2-3 coin / bill / ticket types from total count + total value (+ a relation) tiers 0, 1, 2
    shares   a total split among 2-3 people with "m times", "half", "d more / fewer" relations     tiers 0, 1, 2
    digits   two-digit (tiers 0-1) or three-digit (tier 2) number from digit relations              tiers 0, 1, 2
    rect     rectangle sides from perimeter + a linear relation (length first, then width)          tiers 0, 1
    pure     plain linear systems: 2x2 (tier 0), 2x2 or 3x3 (tier 1), 3x3 (tier 2)                 tiers 0, 1, 2
Generation samples the unknowns first and derives the relations from them. Each relation is an integer row
[a1, a2, (a3,) rhs] in asked order (meta["eqs"]); the square system is solved exactly (Fractions) and the item is kept
only if the solution is unique and equals the sampled unknowns. chance = mean over components of P(uniform int in
[1, 100] hits it) (analytic, ~0.01).
Stdlib + common.py only.
"""
import re
from fractions import Fraction

from ultraviolet.tasks.score import TASKS
from workbench.envs.common import build_cli, clean, parse_list, vec_credit

TASK = "wb_sysword"
FORMAT = "Give the values in the order asked, separated by commas, e.g. \\boxed{12, 7}."
CAP = (50, 200, 1000)
NAMES = ("Maya", "Leo", "Ana", "Omar", "Priya", "Ben", "Sofia", "Kenji", "Lena", "Diego", "Zara", "Felix", "Nora",
         "Ravi", "Ivy", "Hugo", "Mei", "Sam", "Tara", "Jonah")
TIMES = {2: "twice", 3: "three times", 4: "four times", 5: "five times", 6: "six times"}
FAMILIES = {0: ("ages", "coins", "shares", "digits", "rect", "pure"),
            1: ("ages", "coins", "shares", "digits", "rect", "pure"),
            2: ("ages", "coins", "shares", "digits", "pure")}
RETRIES = 60


# ----------------------------------------------------------------------------------------------- exact algebra
def solve(rows):
    """Square integer system rows [a1..an, rhs] -> unique solution as Fractions, or None if singular / not square."""
    n = len(rows)
    if n == 0 or any(len(r) != n + 1 for r in rows):
        return None
    m = [[Fraction(x) for x in r] for r in rows]
    for c in range(n):
        p = next((r for r in range(c, n) if m[r][c] != 0), None)
        if p is None:
            return None
        m[c], m[p] = m[p], m[c]
        for r in range(n):
            if r != c and m[r][c] != 0:
                f = m[r][c] / m[c][c]
                m[r] = [a - f * b for a, b in zip(m[r], m[c])]
    return [m[i][n] / m[i][i] for i in range(n)]


def _ok(rows, gold, nums, cap):
    sol = solve(rows)
    if sol is None or sol != [Fraction(g) for g in gold]:
        return False
    return all(abs(x) <= cap for x in list(nums) + list(gold))


def _row(n, coefs, rhs):
    r = [0] * n
    for i, a in coefs.items():
        r[i] += a
    return r + [rhs]


# ----------------------------------------------------------------------------------------------- text helpers
def _join(xs):
    xs = list(xs)
    return xs[0] if len(xs) == 1 else (" and ".join(xs) if len(xs) == 2 else ", ".join(xs[:-1]) + " and " + xs[-1])


def _order(labels):
    """'Give A and then B.' / 'Give A, then B, then C.'"""
    if len(labels) == 2:
        return f"Give {labels[0]} and then {labels[1]}."
    return "Give " + ", then ".join(labels) + "."


def _yrs(k):
    return f"{k} year" if k == 1 else f"{k} years"


def _times(m):
    return TIMES.get(m, f"{m} times")


def _money(cents):
    return f"\\${cents // 100}.{cents % 100:02d}"


# ----------------------------------------------------------------------------------------------- ages
def _ages(rng, tier):
    cap = CAP[tier]
    n = 2 if tier == 0 else (rng.choice((2, 3)) if tier == 1 else 3)
    hi = (40, 80, 95)[tier]
    for _ in range(RETRIES):
        ppl = rng.sample(NAMES, n)                       # asked order
        age = [rng.randint(2, hi) for _ in range(n)]
        if len(set(age)) < n:
            continue
        cands = []                                       # (row, text, nums)
        cands.append((_row(n, {i: 1 for i in range(n)}, sum(age)),
                      rng.choice((f"Today the ages of {_join(ppl)} add up to {sum(age)}.",
                                  f"Together, {_join(ppl)} are {sum(age)} years old.")), [sum(age)]))
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                a, b = age[i], age[j]
                if a > b:
                    d = a - b
                    cands.append((_row(n, {i: 1, j: -1}, d), f"{ppl[i]} is {_yrs(d)} older than {ppl[j]}.", [d]))
                for m in (2, 3, 4):
                    num = a - m * b                      # a + k = m (b + k)  ->  k = (a - m b) / (m - 1)
                    if num % (m - 1):
                        continue
                    k = num // (m - 1)
                    if b + k < 1 or abs(k) > (10, 30, 60)[tier]:
                        continue
                    row = _row(n, {i: 1, j: -m}, (m - 1) * k)
                    if k > 0:
                        t = f"In {_yrs(k)}, {ppl[i]} will be {_times(m)} as old as {ppl[j]}."
                    elif k < 0:
                        t = f"{_yrs(-k)} ago, {ppl[i]} was {_times(m)} as old as {ppl[j]}."
                    else:
                        t = f"{ppl[i]} is {_times(m)} as old as {ppl[j]}."
                    cands.append((row, t, [abs(k), m]))
        if n == 3:
            i, j = rng.sample(range(3), 2)
            k = rng.randint(1, 15)
            s = age[i] + age[j] + 2 * k
            cands.append((_row(n, {i: 1, j: 1}, s - 2 * k),
                          f"In {_yrs(k)}, the ages of {ppl[i]} and {ppl[j]} will add up to {s}.", [k, s]))
        mult = [c for c in cands if len(c[2]) == 2 and c[2][-1] in (2, 3, 4) and "add up" not in c[1]]
        if not mult:
            continue
        pick = [rng.choice(mult)]
        rest = [c for c in cands if c is not pick[0]]
        pick += rng.sample(rest, min(n - 1, len(rest)))
        if len(pick) < n:
            continue
        rng.shuffle(pick)
        rows = [c[0] for c in pick]
        if not _ok(rows, age, [x for c in pick for x in c[2]], cap):
            continue
        labels = [f"{p}'s age" for p in ppl]
        q = rng.choice((f"How old is each of them now? {_order(labels)}",
                        f"Find their current ages. {_order(labels)}",
                        f"What are their ages today? {_order(labels)}"))
        intro = rng.choice(("", f"{_join(rng.sample(ppl, n))} are relatives. ", "Consider the following. "))
        body = intro + " ".join(c[1] for c in pick) + " " + q
        return body, age, rows, list(ppl)
    return None


# ----------------------------------------------------------------------------------------------- coins
COIN_SETS = {   # (unit, [(singular, plural, value)]); unit 'c' = cents, '$' = dollars
    "coins": ("c", [("penny", "pennies", 1), ("nickel", "nickels", 5), ("dime", "dimes", 10),
                    ("quarter", "quarters", 25), ("half-dollar", "half-dollars", 50)]),
    "bills": ("$", [("one-dollar bill", "one-dollar bills", 1), ("two-dollar bill", "two-dollar bills", 2),
                    ("five-dollar bill", "five-dollar bills", 5), ("ten-dollar bill", "ten-dollar bills", 10),
                    ("twenty-dollar bill", "twenty-dollar bills", 20)]),
    "tickets": ("$", [("child ticket", "child tickets", 4), ("student ticket", "student tickets", 6),
                      ("adult ticket", "adult tickets", 9), ("senior ticket", "senior tickets", 5),
                      ("VIP ticket", "VIP tickets", 15)]),
}


def _coins(rng, tier):
    cap = CAP[tier]
    n = 2 if tier == 0 else (rng.choice((2, 3)) if tier == 1 else 3)
    for _ in range(RETRIES):
        kind = rng.choice(("coins", "bills", "tickets"))
        unit, table = COIN_SETS[kind]
        types = sorted(rng.sample(table, n), key=lambda t: t[2])
        if len({t[2] for t in types}) < n:
            continue
        hi = (12, 30, 60)[tier]
        cnt = [rng.randint(1, hi) for _ in range(n)]
        vals = [t[2] for t in types]
        tot_n, tot_v = sum(cnt), sum(c * v for c, v in zip(cnt, vals))
        rows = [[1] * n + [tot_n], vals + [tot_v]]
        nums = [tot_n, tot_v] + vals
        extra = ""
        if n == 3:
            i, j = rng.sample(range(3), 2)
            if cnt[i] % cnt[j] == 0 and cnt[i] // cnt[j] in (2, 3, 4):
                m = cnt[i] // cnt[j]
                rows.append(_row(3, {i: 1, j: -m}, 0))
                extra = f"There are {_times(m)} as many {types[i][1]} as {types[j][1]}."
                nums.append(m)
            elif cnt[i] > cnt[j]:
                d = cnt[i] - cnt[j]
                rows.append(_row(3, {i: 1, j: -1}, d))
                extra = f"There are {d} more {types[i][1]} than {types[j][1]}."
                nums.append(d)
            else:
                continue
        if not _ok(rows, cnt, nums, cap):
            continue
        order = rng.sample(range(n), n)                   # asked order
        gold = [cnt[i] for i in order]
        rows = [[r[i] for i in order] + [r[-1]] for r in rows]
        plur = [types[i][1] for i in order]
        names = list(plur)
        labels = [f"the number of {p}" for p in plur]
        lst = _join([t[1] for t in types])
        if kind == "coins":
            vtxt = rng.choice((_money(tot_v), f"{tot_v} cents"))
            price = ""
            lead = rng.choice((f"A jar holds only {lst}. There are {tot_n} coins worth {vtxt} in total.",
                               f"{rng.choice(NAMES)} has {tot_n} coins, all {lst}, with a total value of {vtxt}."))
        elif kind == "bills":
            lead = rng.choice((f"A wallet holds {tot_n} bills, all {lst}, worth \\${tot_v} altogether.",
                               f"{rng.choice(NAMES)} has \\${tot_v} in {lst}, {tot_n} bills in all."))
            price = ""
        else:
            price = " ".join(f"{'An' if t[0][0] in 'aeiou' else 'A'} {t[0]} costs \\${t[2]}." for t in types)
            lead = rng.choice((f"A theater sold {tot_n} tickets for a total of \\${tot_v}.",
                               f"{tot_n} tickets were sold for a show, bringing in \\${tot_v}."))
        q = rng.choice(("How many of each were there?", "Find how many of each kind there are."))
        body = " ".join(x for x in (lead, price, extra, q, _order(labels)) if x)
        return body, gold, rows, names
    return None


# ----------------------------------------------------------------------------------------------- shares
SHARE_OBJ = ("marbles", "stickers", "baseball cards", "stamps", "shells", "dollars", "books", "beads")


def _shares(rng, tier):
    cap = CAP[tier]
    n = 2 if tier == 0 else (rng.choice((2, 3)) if tier == 1 else 3)
    obj = rng.choice(SHARE_OBJ)
    hi = (15, 50, 250)[tier]
    for _ in range(RETRIES):
        ppl = rng.sample(NAMES, n)                       # asked order
        intro = rng.sample(range(n), n)                  # order relations are introduced in
        v = [0] * n
        v[intro[0]] = rng.randint(1, hi)
        rows, texts, nums = [], [], []
        good = True
        for pos in range(1, n):
            i, j = intro[pos], intro[rng.randrange(pos)]  # i relates to an earlier j
            r = rng.random()
            if r < 0.4:
                m = rng.choice((2, 3, 4) if tier else (2, 3))
                v[i] = m * v[j]
                rows.append(_row(n, {i: 1, j: -m}, 0))
                texts.append(f"{ppl[i]} has {_times(m)} as many as {ppl[j]}.")
                nums.append(m)
            elif r < 0.5 and v[j] % 2 == 0:
                v[i] = v[j] // 2
                rows.append(_row(n, {i: 2, j: -1}, 0))
                texts.append(f"{ppl[i]} has half as many as {ppl[j]}.")
            else:
                d = rng.randint(1, (10, 40, 150)[tier]) * rng.choice((1, -1))
                v[i] = v[j] + d
                rows.append(_row(n, {i: 1, j: -1}, d))
                texts.append(f"{ppl[i]} has {abs(d)} {'more' if d > 0 else 'fewer'} than {ppl[j]}.")
                nums.append(abs(d))
            if v[i] < 1:
                good = False
                break
        if not good or len(set(v)) < n:
            continue
        tot = sum(v)
        rows.insert(0, _row(n, {i: 1 for i in range(n)}, tot))
        nums.append(tot)
        if not _ok(rows, v, nums, cap):
            continue
        lead = rng.choice((f"{_join(rng.sample(ppl, n))} share {tot} {obj}.",
                           f"{tot} {obj} are split among {_join(rng.sample(ppl, n))}.",
                           f"Together, {_join(rng.sample(ppl, n))} have {tot} {obj}."))
        labels = [f"{p}'s number of {obj}" if obj != "dollars" else f"{p}'s amount" for p in ppl]
        q = rng.choice((f"How many {obj} does each person have?", "How many does each have?"))
        body = " ".join([lead] + texts + [q, _order(labels)])
        return body, v, rows, list(ppl)
    return None


# ----------------------------------------------------------------------------------------------- digits
def _digits(rng, tier):
    three = tier == 2
    n = 3 if three else 2
    pos = ("hundreds", "tens", "units") if three else ("tens", "units")
    for _ in range(RETRIES):
        d = [rng.randint(1, 9)] + [rng.randint(0, 9) for _ in range(n - 2)] + [rng.randint(1, 9)]
        N = int("".join(map(str, d)))
        R = int("".join(map(str, reversed(d))))
        cands = [(_row(n, {i: 1 for i in range(n)}, sum(d)), f"The sum of its digits is {sum(d)}.", [sum(d)])]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if d[i] > d[j]:
                    k = d[i] - d[j]
                    cands.append((_row(n, {i: 1, j: -1}, k),
                                  f"The {pos[i]} digit is {k} more than the {pos[j]} digit.", [k]))
                if d[j] and d[i] % d[j] == 0 and d[i] // d[j] in (2, 3, 4):
                    m = d[i] // d[j]
                    cands.append((_row(n, {i: 1, j: -m}, 0),
                                  f"The {pos[i]} digit is {_times(m)} the {pos[j]} digit.", [m]))
        if R != N:
            k = R - N                                     # 9 (u - t)  or  99 (u - h)
            co = 9 if not three else 99
            row = _row(n, {0: -co, n - 1: co}, k)
            cands.append((row, f"Reversing its digits gives a number that is {abs(k)} "
                               f"{'more' if k > 0 else 'less'} than the original number.", [abs(k)]))
        s = sum(d)
        if N % s == 0 and not three:
            m = N // s                                    # 10 t + u = m (t + u)
            cands.append((_row(2, {0: 10 - m, 1: 1 - m}, 0), f"The number is {m} times the sum of its digits.", [m]))
        if three and d[1] == d[0] + d[2]:
            cands.append((_row(3, {1: 1, 0: -1, 2: -1}, 0),
                          "Its tens digit equals the sum of its hundreds and units digits.", []))
        if three:
            i, j = rng.sample(range(3), 2)
            cands.append((_row(3, {i: 1, j: 1}, d[i] + d[j]),
                          f"Its {pos[i]} and {pos[j]} digits add up to {d[i] + d[j]}.", [d[i] + d[j]]))
        if len(cands) < n:
            continue
        pick = rng.sample(cands, n)
        rows = [c[0] for c in pick]
        if not _ok(rows, d, [x for c in pick for x in c[2]], CAP[tier]):
            continue
        kind = "three-digit" if three else "two-digit"
        lead = rng.choice((f"A {kind} number has no zero units digit.",
                           f"Think of a {kind} number whose units digit is not zero.",
                           f"A {kind} number does not end in 0."))
        labels = [f"its {p} digit" for p in pos]
        q = rng.choice(("Find the number.", "What is the number?"))
        body = " ".join([lead] + [c[1] for c in pick] + [q, _order(labels)])
        return body, d, rows, [f"{p} digit" for p in pos]
    return None


# ----------------------------------------------------------------------------------------------- rect
RECT_OBJ = (("rectangle", "cm"), ("rectangular garden", "m"), ("rectangular poster", "inches"),
            ("rectangular field", "m"), ("rectangular rug", "feet"), ("picture frame", "cm"))


def _rect(rng, tier):
    cap = CAP[tier]
    for _ in range(RETRIES):
        W = rng.randint(1, (10, 40, 0)[tier])
        a = rng.choice((1, 2, 3) if tier else (1, 2))
        b = rng.randint(-W + 1 if a > 1 else 1, (8, 30, 0)[tier])
        L = a * W + b
        if L <= W:
            continue
        P = 2 * (L + W)
        obj, unit = rng.choice(RECT_OBJ)
        if a == 1:
            rel = f"Its length is {b} {unit} more than its width."
        elif b == 0:
            rel = f"Its length is {_times(a)} its width."
        else:
            rel = f"Its length is {abs(b)} {unit} {'more' if b > 0 else 'less'} than {_times(a)} its width."
        rows = [[2, 2, P], [1, -a, b]]
        if not _ok(rows, [L, W], [P, a, abs(b)], cap):
            continue
        lead = rng.choice((f"A {obj} has a perimeter of {P} {unit}.", f"The perimeter of a {obj} is {P} {unit}."))
        q = rng.choice(("Find its length and width.", "What are its dimensions?"))
        body = " ".join([lead, rel, q, "Give the length and then the width."])
        return body, [L, W], rows, ["length", "width"]
    return None


# ----------------------------------------------------------------------------------------------- pure
VARSETS = (("x", "y", "z"), ("x", "y", "z"), ("a", "b", "c"), ("p", "q", "r"))


def _lin(coefs, vs):
    out = ""
    for c, v in zip(coefs, vs):
        if c == 0:
            continue
        mag = "" if abs(c) == 1 else str(abs(c))
        if not out:
            out = ("-" if c < 0 else "") + mag + v
        else:
            out += (" - " if c < 0 else " + ") + mag + v
    return out


def _pure(rng, tier):
    cap = CAP[tier]
    n = 2 if tier == 0 else (rng.choice((2, 3)) if tier == 1 else 3)
    cmax, smax = (5, 7, 9)[tier], (9, 15, 30)[tier]
    vs = rng.choice(VARSETS)[:n]
    for _ in range(RETRIES):
        sol = [rng.randint(-smax, smax) for _ in range(n)]
        rows = []
        for _ in range(n):
            co = [rng.randint(-cmax, cmax) for _ in range(n)]
            if sum(1 for c in co if c) < 2:
                continue
            rows.append(co + [sum(c * s for c, s in zip(co, sol))])
        if len(rows) < n or not _ok(rows, sol, [x for r in rows for x in r], cap):
            continue
        eqs = [f"{_lin(r[:-1], vs)} = {r[-1]}" for r in rows]
        sys_txt = ", ".join(f"${e}$" for e in eqs[:-1]) + f" and ${eqs[-1]}$"
        order = _order(list(vs))
        lead = rng.choice((f"Solve the system of equations {sys_txt}.",
                           f"Find the integers {_join(vs)} satisfying {sys_txt}.",
                           f"The numbers {_join(vs)} satisfy {sys_txt}. Find them."))
        return lead + " " + order, sol, rows, list(vs)
    return None


GENS = {"ages": _ages, "coins": _coins, "shares": _shares, "digits": _digits, "rect": _rect, "pure": _pure}


# ----------------------------------------------------------------------------------------------- gen / score
def gen(rng, tier):
    fam = rng.choice(FAMILIES[tier])
    r = GENS[fam](rng, tier)
    if r is None:
        return None
    body, gold, rows, names = r
    if solve(rows) != [Fraction(g) for g in gold]:      # final uniqueness / consistency guard
        return None
    chance = sum(0.01 for g in gold if 1 <= g <= 100) / len(gold)
    meta = {"gold": list(gold), "names": list(names), "eqs": [list(r) for r in rows], "n_unknowns": len(gold),
            "chance": chance}
    return body, ", ".join(map(str, gold)), meta, fam


_TUPLE_EQ = re.compile(r"^\(([^()]*)\)\s*=\s*\(([^()]*)\)$")


def _unwrap(pred):
    """Accept '(3, 4)' and '(x, y) = (3, 4)' as well as the forms common.parse_list handles."""
    s = clean(pred)
    m = _TUPLE_EQ.match(s)
    if m:
        return m.group(2)
    if s.startswith("(") and s.endswith(")") and "(" not in s[1:-1] and ")" not in s[1:-1]:
        return s[1:-1]
    return s


def sc_sysword(pred, item):
    gold = item.get("meta", {}).get("gold")
    if gold is None:
        gold = parse_list(item["answer"], "int")
    if pred is None:
        return 0.0
    return vec_credit(parse_list(_unwrap(pred), "int"), gold)


TASKS[TASK] = {"kind": "list", "scorer": sc_sysword, "reward": "graded"}

if __name__ == "__main__":
    build_cli(TASK, gen, FORMAT)
