"""workbench/envs/allsol.py (wb_allsol): F1 partial credit, gold self-check, brute-force gold, chance, determinism."""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ultraviolet.tasks.score import TASKS, score  # noqa: E402
from workbench.envs import allsol as A  # noqa: E402
from workbench.envs import common as C  # noqa: E402

FAMS = set(A.FAMILIES)


def item(gold, kind="int"):
    return {"task": "wb_allsol", "answer": "", "meta": {"gold": gold, "kind": kind}}


def sc(box, it):
    return score("wb_allsol", "</think>\n\n\\boxed{" + box + "}", it)


def test_registered():
    assert TASKS["wb_allsol"]["kind"] == "set" and TASKS["wb_allsol"]["reward"] == "graded"


def test_f1_partial_credit_ints():
    it = item([1, 2, 3, 4, 6, 12])
    assert sc("1, 2, 3, 4, 6, 12", it) == 1.0
    assert sc("12, 6, 4, 3, 2, 1", it) == 1.0                                 # order free
    assert sc("1, 1, 2, 3, 4, 6, 12, 12", it) == 1.0                          # duplicates dropped
    assert abs(sc("1, 2, 3, 4, 6", it) - 2 * 1 * (5 / 6) / (1 + 5 / 6)) < 1e-9  # missing one divisor
    assert abs(sc("1, 2, 3, 4, 6, 12, 5", it) - 2 * (6 / 7) / (6 / 7 + 1)) < 1e-9  # one extra wrong element
    assert abs(sc("\\{1, 2, 3\\}", it) - 2 * 0.5 / 1.5) < 1e-9
    assert sc("5, 7", it) == 0.0
    assert sc("none", it) == 0.0
    assert sc("1, a", it) == 0.0                                              # unparsable -> 0


def test_none_vs_empty_gold():
    it = item([])
    assert sc("none", it) == 1.0
    assert sc("\\text{none}", it) == 1.0
    assert sc("\\emptyset", it) == 1.0
    assert sc("0", it) == 0.0
    assert sc("1, 2", it) == 0.0


def test_pairs():
    it = item([[1, 2], [3, 4]], "pair")
    assert sc("(1, 2), (3, 4)", it) == 1.0
    assert sc("\\{(1,2),(3,4)\\}", it) == 1.0
    assert sc("(3, 4), (1, 2)", it) == 1.0
    assert abs(sc("(1, 2)", it) - 2 / 3) < 1e-9
    assert abs(sc("(1, 2), (2, 1)", it) - 0.5) < 1e-9
    assert sc("(2, 1), (4, 3)", it) == 0.0
    assert sc("1, 2, 3, 4", it) == 0.0


def test_fmt_poly():
    assert A.fmt_poly(A.poly_from_roots([1, 2, 3])) == "x^3 - 6x^2 + 11x - 6"
    assert A.fmt_poly(A.poly_from_roots([0, 1, -1])) == "x^3 - x"
    assert A.fmt_poly(A.poly_from_roots([-1, -1, -1, -1])) == "x^4 + 4x^3 + 6x^2 + 4x + 1"
    assert A.fmt_poly([1, 0, -5]) == "x^2 - 5"


def _is_prime(n):
    return n > 1 and all(n % d for d in range(2, n))


def brute(meta):
    f, p = meta["family"], meta["params"]
    if f == "divisors":
        return [d for d in range(1, p["n"] + 1) if p["n"] % d == 0]
    if f == "quadmod":
        m = p["m"]
        return [x for x in range(m) if (x * x + p["b"] * x + p["c"]) % m == 0]
    if f == "introots":
        cs = p["coefs"]
        bound = 1 + max(abs(c) for c in cs[1:])                 # Cauchy bound for a monic polynomial
        ev = lambda x: sum(c * x ** (len(cs) - 1 - i) for i, c in enumerate(cs))  # noqa: E731
        return [x for x in range(-bound, bound + 1) if ev(x) == 0]
    if f == "diophant":
        a, b, c = p["a"], p["b"], p["c"]
        return [[x, y] for x in range(c + 1) for y in range(c + 1) if a * x + b * y == c]
    if f == "primes":
        return [q for q in range(p["a"], p["b"] + 1) if _is_prime(q)]
    raise AssertionError(f)


def gen_items(tier, n=60, base=0):
    out = []
    for i in range(n * 3):
        r = A.gen(random.Random(base + 1000 * tier + i), tier)
        if r is not None:
            out.append(r)
        if len(out) >= n:
            break
    assert len(out) == n
    return out


def test_generated_gold_scores_one_and_matches_brute_force():
    for tier in range(3):
        fams = set()
        for body, ans, meta, fam in gen_items(tier):
            fams.add(fam)
            it = C.make_item("wb_allsol", fam, tier, "eval", 0, body, A.FORMAT, ans, meta)
            assert sc(ans, it) == 1.0, (ans, meta)
            assert brute(meta) == meta["gold"], (body, meta)
            assert ans == ("none" if not meta["gold"] else ans)
            assert 0.0 <= meta["chance"] <= 1.0
            assert meta["kind"] == ("pair" if fam == "diophant" else "int")
            assert "$" in body
        assert fams == FAMS, (tier, fams)


def test_tier_shapes():
    for tier in range(3):
        for body, ans, meta, fam in gen_items(tier, base=77):
            g, p = meta["gold"], meta["params"]
            if fam == "divisors":
                assert 24 <= p["n"] <= 400 and 6 <= len(g) <= 16
                assert tier != 0 or p["n"] <= 100
            elif fam == "quadmod":
                assert 5 <= p["m"] <= 30 and len(g) <= 6
                assert tier != 0 or (p["m"] <= 12 and len(g) <= 4)
            elif fam == "introots":
                assert len(p["coefs"]) in (4, 5) and all(-9 <= x <= 9 for x in g)
                assert tier != 0 or (len(p["coefs"]) == 4 and len(g) == 3)
            elif fam == "diophant":
                assert 1 <= len(g) <= 5 and 2 <= p["a"] <= 15 and 2 <= p["b"] <= 15
                assert tier != 0 or len(g) <= 2
                assert g == sorted(g)
            elif fam == "primes":
                span = p["b"] - p["a"]
                assert 20 <= span <= 60 and 2 <= p["a"] and p["b"] <= 400 and len(g) >= 1
                assert tier != 0 or (span <= 30 and len(g) <= 4)


def test_quadmod_empty_rate():
    n_q, n_empty = 0, 0
    for i in range(400):
        r = A.g_quadmod(random.Random(5000 + i), i % 3)
        if r is None:
            continue
        n_q += 1
        n_empty += not r[1]
    assert n_q > 350 and 0 < n_empty <= 0.12 * n_q, (n_q, n_empty)


def test_chance():
    ch = [meta["chance"] for t in range(3) for _, _, meta, _ in gen_items(t, base=11)]
    assert all(0.0 <= c <= 1.0 for c in ch)
    assert sum(c < 0.5 for c in ch) >= 0.9 * len(ch)
    assert sum(ch) / len(ch) < 0.3


def test_determinism():
    for tier in range(3):
        for s in range(20):
            assert A.gen(random.Random(s), tier) == A.gen(random.Random(s), tier)


def test_build_items():
    items, stats = C.build_items("wb_allsol", A.gen, A.FORMAT, {"eval": 30, "rl": 60}, seed=3)
    assert len(items["eval"]) == 30 and len(items["rl"]) == 60
    allit = items["eval"] + items["rl"]
    assert {it["family"] for it in allit} == FAMS
    assert {it["tier"] for it in items["eval"]} == {0, 1, 2}
    assert not {C.qkey(it["problem"]) for it in items["eval"]} & {C.qkey(it["problem"]) for it in items["rl"]}
    for it in allit:
        assert it["problem"].endswith(A.FORMAT(it["family"], it["meta"]) + " " + C.INSTR)
