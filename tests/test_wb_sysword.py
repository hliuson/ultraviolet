"""workbench/envs/sysword.py: wb_sysword parsing / graded scoring, gold self-check, relation re-derivation from
meta["eqs"], determinism, build_items coverage (docs/wb_envs.md)."""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fractions import Fraction  # noqa: E402

from ultraviolet.tasks.score import TASKS, score, score_pred  # noqa: E402
from workbench.envs import common as C  # noqa: E402
from workbench.envs import sysword as S  # noqa: E402

TASK = "wb_sysword"


def item(gold):
    return {"task": TASK, "answer": ", ".join(map(str, gold)), "meta": {"gold": list(gold)}}


def items(tier, n=60):
    out, i = [], 0
    while len(out) < n and i < 20 * n:
        r = S.gen(random.Random(1000 * tier + i), tier)
        i += 1
        if r is not None:
            out.append(r)
    assert len(out) == n
    return out


def det(rows):
    m = [[Fraction(x) for x in r[:-1]] for r in rows]
    n, d = len(m), Fraction(1)
    for c in range(n):
        p = next((r for r in range(c, n) if m[r][c] != 0), None)
        if p is None:
            return 0
        if p != c:
            m[c], m[p], d = m[p], m[c], -d
        d *= m[c][c]
        for r in range(c + 1, n):
            f = m[r][c] / m[c][c]
            m[r] = [a - f * b for a, b in zip(m[r], m[c])]
    return d


def test_registered():
    assert TASKS[TASK]["kind"] == "list" and TASKS[TASK]["reward"] == "graded"
    assert TASKS[TASK]["scorer"] is S.sc_sysword


def test_partial_credit_and_length():
    it2, it3 = item([3, 4]), item([5, 6, 7])
    assert S.sc_sysword("3, 4", it2) == 1.0
    assert S.sc_sysword("3, 9", it2) == 0.5
    assert S.sc_sysword("4, 3", it2) == 0.0                   # order matters
    assert abs(S.sc_sysword("5, 0, 0", it3) - 1 / 3) < 1e-12
    assert abs(S.sc_sysword("5, 6, 0", it3) - 2 / 3) < 1e-12
    assert S.sc_sysword("3", it2) == 0.5                      # prefix credit: first unknown right
    assert S.sc_sysword("4", it2) == 0.0
    assert abs(S.sc_sysword("3, 4, 5", it2) - 2 / 3) < 1e-12  # extra component dilutes
    assert S.sc_sysword("3, a", it2) == 0.0
    assert S.sc_sysword("3.5, 4", it2) == 0.0
    assert S.sc_sysword(None, it2) == 0.0
    assert score_pred(TASK, "3, 9", it2) == 0.5


def test_answer_forms_parse():
    it2 = item([3, 4])
    for p in ("x = 3, y = 4", "3, 4", "(3, 4)", "(x, y) = (3, 4)", "\\left(3, 4\\right)", "3 and 4", "$3, 4$",
              "Maya = 3, Leo = 4", "[3, 4]", "3; 4"):
        assert S.sc_sysword(p, it2) == 1.0, p
    assert S.sc_sysword("(3, 4, 5)", item([3, 4, 5])) == 1.0
    assert S.sc_sysword("(3, 4), (5, 6)", it2) == 0.0


def test_solver():
    assert S.solve([[2, 3, 12], [1, -1, 1]]) == [3, 2]
    assert S.solve([[1, 1, 2], [2, 2, 4]]) is None             # singular
    assert S.solve([[1, 1, 1, 6], [0, 1, 1, 5], [0, 0, 1, 3]]) == [1, 2, 3]
    assert S.solve([[1, 1, 2]]) is None                          # not square


def test_gold_scores_one_and_relations_hold():
    for tier in range(3):
        for body, ans, meta, fam in items(tier):
            gold, eqs = meta["gold"], meta["eqs"]
            it = C.make_item(TASK, fam, tier, "eval", 0, body, S.FORMAT, ans, meta)
            assert score(TASK, "</think>\n\n\\boxed{" + ans + "}", it) == 1.0, (tier, fam, ans)
            assert all(isinstance(g, int) for g in gold) and ans == ", ".join(map(str, gold))
            assert len(meta["names"]) == len(gold) and len(eqs) == len(gold)
            assert all(len(r) == len(gold) + 1 and all(isinstance(x, int) for x in r) for r in eqs)
            for r in eqs:
                assert sum(a * g for a, g in zip(r[:-1], gold)) == r[-1], (fam, r, gold)
            assert det(eqs) != 0, (fam, eqs)
            assert S.solve(eqs) == gold
            assert 0.0 <= meta["chance"] <= 0.05
            n = len(gold)
            assert n == (2 if tier == 0 else 3 if tier == 2 else n) and n in (2, 3)
            if fam != "pure":
                assert all(g >= (0 if fam == "digits" else 1) for g in gold), (fam, gold)
                assert max(abs(g) for g in gold) <= S.CAP[tier]
            if fam in ("ages", "shares"):
                assert len(set(meta["names"])) == n
                for nm in meta["names"]:
                    assert nm in body
            if fam == "digits":
                assert all(0 <= g <= 9 for g in gold) and gold[0] >= 1 and n == (3 if tier == 2 else 2)
            if fam == "rect":
                assert gold[0] > gold[1] and tier < 2
            assert " then " in body                              # order is stated explicitly


def test_determinism():
    for tier in range(3):
        for i in range(30):
            a, b = S.gen(random.Random(i), tier), S.gen(random.Random(i), tier)
            assert a == b


def test_build_items_covers_families():
    out, stats = C.build_items(TASK, S.gen, S.FORMAT, {"eval": 30, "rl": 60})
    assert len(out["eval"]) == 30 and len(out["rl"]) == 60
    allit = out["eval"] + out["rl"]
    assert {it["family"] for it in allit} == set(S.GENS)
    assert {it["tier"] for it in allit} == {0, 1, 2}
    ev = {C.qkey(it["problem"]) for it in out["eval"]}
    assert not ev & {C.qkey(it["problem"]) for it in out["rl"]}
    for it in allit:
        assert it["problem"].endswith(S.FORMAT + " " + C.INSTR)
        assert it["family"] in S.FAMILIES[it["tier"]]
