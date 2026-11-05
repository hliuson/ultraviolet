"""workbench/envs/resid.py: wb_resid residue vectors (docs/wb_envs.md)."""
import os
import random
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ultraviolet.tasks.score import TASKS, score  # noqa: E402
from workbench.envs import common as C  # noqa: E402
from workbench.envs import resid as R  # noqa: E402

ITEM = {"task": "wb_resid", "answer": "3, 5, 7", "meta": {"gold": [3, 5, 7], "moduli": [4, 9, 11]}}


def _gen(seed, tier):
    for i in range(100):
        r = R.gen(random.Random(seed * 1000 + i), tier)
        if r is not None:
            return r
    raise AssertionError("gen kept returning None")


def _value(expr):
    """Evaluate the rendered LaTeX expression independently: a^{e}, \\cdot, +, -, plain integers."""
    py = expr.replace("\\cdot", "*")
    py = re.sub(r"\^\{(\d+)\}", r"**\1", py)
    assert re.fullmatch(r"[\d\s*+\-]+", py), expr
    return eval(py)  # noqa: S307  digits and operators only


def test_registered():
    assert TASKS["wb_resid"]["reward"] == "graded" and TASKS["wb_resid"]["scorer"] is R.sc_resid


def test_scorer_formats():
    for p in ("3, 5, 7", "3,5,7", "\\{3, 5, 7\\}", "3 5 7", "3; 5; 7", "$3, 5, 7$", "[3, 5, 7]"):
        assert R.sc_resid(p, ITEM) == 1.0, p
    assert R.sc_resid("N \\equiv 3, 5, 7", ITEM) in (0.0, 1.0)  # fine either way
    assert score("wb_resid", "</think>\n\n\\boxed{3, 5, 7}", ITEM) == 1.0


def test_scorer_partial_and_wrong():
    assert abs(R.sc_resid("3, 5, 8", ITEM) - 2 / 3) < 1e-12
    assert abs(R.sc_resid("0, 5, 7", ITEM) - 2 / 3) < 1e-12
    assert R.sc_resid("0, 0, 0", ITEM) == 0.0
    assert abs(R.sc_resid("3, 5", ITEM) - 2 / 3) < 1e-12        # prefix credit: 2 aligned matches / 3
    assert abs(R.sc_resid("3", ITEM) - 1 / 3) < 1e-12
    assert abs(R.sc_resid("3, 5, 7, 1", ITEM) - 3 / 4) < 1e-12  # extra component dilutes
    assert R.sc_resid("5", ITEM) == 0.0                         # single number that is not the first residue
    assert abs(R.sc_resid("7, 5, 7", ITEM) - 2 / 3) < 1e-12   # 7 == 3 mod 4 but out of range: wrong
    assert R.sc_resid("a, b, c", ITEM) == 0.0
    assert R.sc_resid(None, ITEM) == 0.0
    assert abs(score("wb_resid", "</think>\n\n\\boxed{3, 1, 7}", ITEM) - 2 / 3) < 1e-12
    no_meta = {"task": "wb_resid", "answer": "3, 5, 7", "meta": {}}
    assert R.sc_resid("3, 5, 7", no_meta) == 1.0


def test_gold_scores_one_and_is_correct():
    fams = {0: set(), 1: set(), 2: set()}
    for tier in range(3):
        n = 0
        for i in range(400):
            r = R.gen(random.Random(10_000 * tier + i), tier)
            if r is None:
                continue
            body, ans, meta, fam = r
            fams[tier].add(fam)
            item = {"task": "wb_resid", "answer": ans, "meta": meta}
            assert score("wb_resid", "</think>\n\n\\boxed{" + ans + "}", item) == 1.0, (body, ans)
            mods, gold = meta["moduli"], meta["gold"]
            v = _value(meta["expr"])
            assert v > 0 and gold == [v % m for m in mods]
            assert all(0 <= g < m for g, m in zip(gold, mods))
            assert len(set(mods)) == len(mods) and set(mods) <= set(R.ALL_MODS)
            assert ans == ", ".join(map(str, gold))
            assert 0.0 <= meta["chance"] <= 1.0
            assert abs(meta["chance"] - sum(1 / m for m in mods) / len(mods)) < 1e-12
            assert meta["expr"] in body and all(f"{m}" in body for m in mods)
            if tier == 0:
                assert fam == "explicit" and len(mods) == 2 and set(mods) <= set(R.DIGIT_MODS) and v < 10 ** 6
            elif tier == 1:
                assert 2 <= len(mods) <= 3 and max(mods) <= 13
                if fam == "power":
                    a, e = map(int, re.fullmatch(r"(\d+)\^\{(\d+)\}", meta["expr"]).groups())
                    assert 2 <= a <= 12 and 5 <= e <= 30 and pow(a, e) == v
                    assert gold == [pow(a, e, m) for m in mods]
                else:
                    assert 10 ** 6 <= v < 10 ** 12
            else:
                assert fam == "expr" and len(mods) == 3
                assert all(int(x) <= 100 for x in re.findall(r"\^\{(\d+)\}", meta["expr"]))
            assert sum(1 for g in gold if g == 0) <= 1
            n += 1
            if n >= 100:
                break
        assert n == 100, tier
    assert fams == {0: {"explicit"}, 1: {"explicit", "power"}, 2: {"expr"}}


def test_determinism():
    for tier in range(3):
        assert _gen(7, tier) == _gen(7, tier)
    assert _gen(1, 2) != _gen(2, 2)


def test_build_items():
    items, stats = C.build_items("wb_resid", R.gen, R.FORMAT, {"eval": 30, "rl": 60}, seed=0)
    assert len(items["eval"]) == 30 and len(items["rl"]) == 60
    for split, its in items.items():
        tiers = [it["tier"] for it in its]
        assert {t: tiers.count(t) for t in range(3)} == {t: len(its) // 3 for t in range(3)}, split
        for it in its:
            assert it["problem"].endswith(R.FORMAT + " " + C.INSTR)
            assert score("wb_resid", "</think>\n\n\\boxed{" + it["answer"] + "}", it) == 1.0
    ev = {C.qkey(it["problem"]) for it in items["eval"]}
    assert not ev & {C.qkey(it["problem"]) for it in items["rl"]}
    again, _ = C.build_items("wb_resid", R.gen, R.FORMAT, {"eval": 30, "rl": 60}, seed=0)
    assert again == items
