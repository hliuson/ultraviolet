"""workbench/envs/common.py: answer parsing + graded rewards, and the score_pred float seam (docs/wb_envs.md)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from fractions import Fraction  # noqa: E402

from ultraviolet.tasks.score import TASKS, score, score_pred  # noqa: E402
from workbench.envs import common as C  # noqa: E402


def test_parse_list_ints():
    assert C.parse_list("3, 5, 7") == [3, 5, 7]
    assert C.parse_list("3; 5; 7") == [3, 5, 7]
    assert C.parse_list("3 5 7") == [3, 5, 7]
    assert C.parse_list("\\{1, 2, 3\\}") == [1, 2, 3]
    assert C.parse_list("[1, 2, 3]") == [1, 2, 3]
    assert C.parse_list("1, 2 \\text{ and } 3") == [1, 2, 3]
    assert C.parse_list("1, 2 and 3") == [1, 2, 3]
    assert C.parse_list("x = 2, y = -3") == [2, -3]
    assert C.parse_list("$2, 3$") == [2, 3]
    assert C.parse_list("1, 2, 3.") == [1, 2, 3]
    assert C.parse_list("4{,} 9{,}2") == [4, 9, 2]        # LaTeX thin-space commas (seen in baseline rollouts)
    assert C.parse_list("2.5, 3") is None          # not integers
    assert C.parse_list("1, a") is None
    assert C.parse_list("") is None
    assert C.parse_list(None) is None


def test_parse_list_none_and_nums():
    assert C.parse_list("none") == []
    assert C.parse_list("\\text{none}") == []
    assert C.parse_list("\\emptyset") == []
    assert C.parse_list("No solution") == []
    assert C.parse_list("1/2, \\frac{3}{4}, 2", kind="num") == [Fraction(1, 2), Fraction(3, 4), Fraction(2)]


def test_parse_pairs():
    assert C.parse_list("(1, 2), (3, 4)", kind="pair") == [(1, 2), (3, 4)]
    assert C.parse_list("(x, y) = (1, 2)", kind="pair") == [(1, 2)]   # variable-name left side dropped
    assert C.parse_list("(1, 2)", kind="pair") == [(1, 2)]
    assert C.parse_list("(3, 4)", kind="int") == [3, 4]
    assert C.parse_list("(1,2);(3,4)", kind="pair") == [(1, 2), (3, 4)]
    assert C.parse_list("\\{(0, 5), (3, 1)\\}", kind="pair") == [(0, 5), (3, 1)]
    assert C.parse_list("(1, 2, 3)", kind="pair") is None
    assert C.parse_list("none", kind="pair") == []


def test_frac_correct():
    assert C.frac_correct([1, 2, 3], [1, 2, 3]) == 1.0
    assert C.frac_correct([1, 0, 3], [1, 2, 3]) == 2 / 3
    assert C.frac_correct([1, 2], [1, 2, 3]) == 0.0
    assert C.frac_correct(None, [1]) == 0.0


def test_f1_set():
    assert C.f1_set([1, 2, 3], [1, 2, 3]) == 1.0
    assert C.f1_set([3, 2, 1, 1], [1, 2, 3]) == 1.0            # order / duplicates free
    assert abs(C.f1_set([1, 2], [1, 2, 3]) - 0.8) < 1e-12       # P=1, R=2/3
    assert abs(C.f1_set([1, 2, 3, 4, 5, 6], [1, 2, 3]) - 2 / 3) < 1e-12
    assert C.f1_set([4, 5], [1, 2, 3]) == 0.0
    assert C.f1_set([], [1]) == 0.0
    assert C.f1_set([], []) == 1.0
    assert C.f1_set([1], []) == 0.0
    assert C.f1_set(None, [1]) == 0.0


def test_frac_cells():
    gold = [1, 2, 3, 4]
    assert C.frac_cells([1, 2, 3, 4], gold, [1, 3]) == 1.0
    assert C.frac_cells([1, 2, 9, 4], gold, [1, 2]) == 0.5
    assert C.frac_cells([1, 9, 9, 9], gold, [1, 3]) == 0.0
    assert C.frac_cells([1, 2, 3], gold, [1, 3]) == 0.0


def test_score_pred_float_seam():
    def graded(pred, item):
        return float(pred)

    def binary(pred, item):
        return pred == "1"

    TASKS["wb__test_graded"] = {"kind": "list", "scorer": graded, "reward": "graded"}
    TASKS["wb__test_binary"] = {"kind": "number", "scorer": binary}
    try:
        it = {"task": "wb__test_graded", "answer": "x", "meta": {}}
        assert score_pred("wb__test_graded", "0.5", it) == 0.5
        assert score_pred("wb__test_graded", "7", it) == 1.0           # clipped
        assert score_pred("wb__test_graded", "-1", it) == 0.0
        assert score_pred("wb__test_graded", "nan-ish", it) == 0.0     # scorer raised -> 0
        assert score("wb__test_graded", "think</think>\n\\boxed{0.25}", it) == 0.25
        assert score("wb__test_graded", "no close \\boxed{1}", it) == 0.0
        itb = {"task": "wb__test_binary", "answer": "1", "meta": {}}
        assert score_pred("wb__test_binary", "1", itb) == 1.0
        assert score_pred("wb__test_binary", "2", itb) == 0.0
    finally:
        TASKS.pop("wb__test_graded")
        TASKS.pop("wb__test_binary")


def test_make_item_format():
    it = C.make_item("wb_x", "fam", 1, "eval", 3, "Find stuff.", "Give them in order, separated by commas.", "1, 2",
                     {"chance": 0.1})
    assert it["problem"] == "Find stuff.\nGive them in order, separated by commas. Put the final answer in \\boxed{}."
    assert it["prompt"] == "Problem:\n" + it["problem"] + "\n\n<think>\n"
    assert it["id"] == "wb_x-eval-000003" and it["tier"] == 1 and it["family"] == "fam" and it["ref"] == "1, 2"
