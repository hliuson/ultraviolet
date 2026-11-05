"""Scorer tests for ultraviolet.tasks (stdlib only)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ultraviolet.tasks.score import (answer_tail, extract_answer, last_boxed, parse_fraction_pair,  # noqa: E402
                                       parse_number, score, score_pred)


def item(task, answer, **meta):
    return {"task": task, "answer": answer, "ref": answer, "meta": meta}


def wrap(box, think="some reasoning"):
    return f"{think}\n</think>\n\nSo the answer is \\boxed{{{box}}}."


def S(task, it, text, **kw):
    return score(task, text, it, **kw)


# ------------------------------------------------------------------------------------------ extraction
def test_last_boxed_nested_and_last():
    assert last_boxed("\\boxed{1} then \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
    assert last_boxed("\\boxed {7}") == "7"
    assert last_boxed("no box") is None
    assert last_boxed("\\boxed{unclosed") is None


def test_requires_think_close():
    it = item("chain_sum", "42")
    assert S("chain_sum", it, "thinking \\boxed{42}") == 0.0              # no </think>
    assert S("chain_sum", it, "thinking \\boxed{42}", require_close=False) == 1.0
    assert S("chain_sum", it, "<think>x</think>\n\n\\boxed{42}") == 1.0
    # box inside the think block only does not count
    assert S("chain_sum", it, "\\boxed{42}</think>\n\nThe answer is 42.") == 0.0
    # last </think> wins
    assert S("chain_sum", it, "a</think>b</think>\\boxed{42}") == 1.0
    assert answer_tail("x") is None


def test_no_bare_number_fallback():
    assert S("chain_sum", item("chain_sum", "42"), "x</think>\n\nThe answer is 42") == 0.0


# ------------------------------------------------------------------------------------------ numbers
@pytest.mark.parametrize("box", ["42", "42.", " 42 ", "42.0", "$42$", "\\text{42}", "x = 42", "42 apples",
                                 "+42", "\\$42", "84/2", "\\frac{84}{2}"])
def test_number_variants_right(box):
    assert S("chain_sum", item("chain_sum", "42"), wrap(box)) == 1.0


@pytest.mark.parametrize("box", ["43", "-42", "4.2", "420", "", "forty-two", "42 or 43", "4 2"])
def test_number_variants_wrong(box):
    assert S("chain_sum", item("chain_sum", "42"), wrap(box)) == 0.0


def test_number_commas_and_negative():
    it = item("gsm8k", "1234")
    assert S("gsm8k", it, wrap("1,234")) == 1.0
    assert S("gsm8k", it, wrap("1,2345")) == 0.0
    it = item("basic_arithmetic", "-56")
    assert S("basic_arithmetic", it, wrap("-56")) == 1.0
    assert S("basic_arithmetic", it, wrap("\u221256")) == 1.0
    assert S("basic_arithmetic", it, wrap("56")) == 0.0


def test_int_strict_rejects_units_and_text():
    it = item("letter_counting", "3")
    assert S("letter_counting", it, wrap("3")) == 1.0
    assert S("letter_counting", it, wrap("3 times")) == 0.0
    assert S("gcd", item("gcd", "6"), wrap("6")) == 1.0
    assert S("gcd", item("gcd", "6"), wrap("12")) == 0.0


def test_parse_number():
    from fractions import Fraction
    assert parse_number("-\\frac{7}{2}") == Fraction(-7, 2)
    assert parse_number("3.50") == Fraction(7, 2)
    assert parse_number("12%") == 12
    assert parse_number("abc") is None


# ------------------------------------------------------------------------------------------ fractions
def test_fraction_lowest_terms():
    it = item("fraction_simplification", "23/27")
    for b in ("23/27", "\\frac{23}{27}", "\\dfrac{23}{27}", "$23/27$", " 23 / 27 "):
        assert S("fraction_simplification", it, wrap(b)) == 1.0, b
    for b in ("46/54", "0.85185", "27/23", "-23/27", "23"):
        assert S("fraction_simplification", it, wrap(b)) == 0.0, b
    assert parse_fraction_pair("-\\frac{1}{3}") == (-1, 3)


# ------------------------------------------------------------------------------------------ strings
def test_base_conversion():
    it = item("base_conversion", "1d5")
    for b in ("1d5", "1D5", "0x1d5", "1d5_{16}", "01d5"):
        assert S("base_conversion", it, wrap(b)) == 1.0, b
    for b in ("1d6", "469", "1 d 5x"):
        assert S("base_conversion", it, wrap(b)) == 0.0, b
    assert S("base_conversion", item("base_conversion", "0"), wrap("0")) == 1.0


def test_time():
    it = item("time_intervals", "06:09")
    for b in ("06:09", "6:09", "\\text{06:09}"):
        assert S("time_intervals", it, wrap(b)) == 1.0, b
    for b in ("06:10", "06:09:00", "6h 9m", "369"):
        assert S("time_intervals", it, wrap(b)) == 0.0, b
    it = item("time_intervals", "00:34:28")
    assert S("time_intervals", it, wrap("00:34:28")) == 1.0
    assert S("time_intervals", it, wrap("00:34")) == 0.0


def test_calendar():
    it = item("calendar_arithmetic", "Friday")
    for b in ("Friday", "friday", "\\text{Friday}", "Friday."):
        assert S("calendar_arithmetic", it, wrap(b)) == 1.0, b
    for b in ("Fri", "Thursday", "Friday or Saturday"):
        assert S("calendar_arithmetic", it, wrap(b)) == 0.0, b
    assert S("calendar_arithmetic", item("calendar_arithmetic", "12"), wrap("12")) == 1.0
    assert S("calendar_arithmetic", item("calendar_arithmetic", "12"), wrap("13")) == 0.0


def test_spell_backward():
    it = item("spell_backward", "ainomma")
    for b in ("ainomma", "AINOMMA", "\\text{ainomma}", "a i n o m m a", "\"ainomma\""):
        assert S("spell_backward", it, wrap(b)) == 1.0, b
    for b in ("ainomm", "ammonia", "ainommaa"):
        assert S("spell_backward", it, wrap(b)) == 0.0, b


# ------------------------------------------------------------------------------------------ countdown
def test_countdown():
    it = item("countdown", "5*(65 + 35)", numbers=[65, 5, 35], target=500)
    for b in ("5*(65 + 35)", "(35+65)\\times 5", "(65+35)*5 = 500", "5 \\cdot (35 + 65)"):
        assert S("countdown", it, wrap(b)) == 1.0, b
    for b in ("500", "5*65+35", "5*(65+35)+0", "5*100", "(65+35)*5*1", "5**2"):
        assert S("countdown", it, wrap(b)) == 0.0, b
    it = item("countdown", "8/(3-8/3)", numbers=[8, 3, 8, 3], target=24)
    assert S("countdown", it, wrap("8/(3-8/3)")) == 1.0  # exact rational arithmetic


# ------------------------------------------------------------------------------------------ sudoku
def test_sudoku():
    puz = [[2, 0, 0, 3], [0, 0, 1, 0], [1, 0, 3, 0], [3, 0, 2, 1]]
    it = item("mini_sudoku", "2143 4312 1234 3421", puzzle=puz)
    for b in ("2143 4312 1234 3421", "2 1 4 3\n4 3 1 2\n1 2 3 4\n3 4 2 1", "2143\\\\4312\\\\1234\\\\3421",
              "2,1,4,3,4,3,1,2,1,2,3,4,3,4,2,1"):
        assert S("mini_sudoku", it, wrap(b)) == 1.0, b
    for b in ("2143 4312 1234 3412", "1243 4312 1234 3421", "2143 4312 1234", "2143 4312 1234 3421 1"):
        assert S("mini_sudoku", it, wrap(b)) == 0.0, b


# ------------------------------------------------------------------------------------------ knights
def test_knights():
    it = item("knights_knaves", "Amelia is a hero, Elizabeth is a villain", names=["Amelia", "Elizabeth"],
              solution=[True, False], knight="hero", knave="villain")
    for b in ("Amelia is a hero, Elizabeth is a villain", "amelia: hero, elizabeth: villain",
              "Elizabeth is a villain and Amelia is a hero", "\\text{Amelia is a hero, Elizabeth is a villain}"):
        assert S("knights_knaves", it, wrap(b)) == 1.0, b
    for b in ("Amelia is a villain, Elizabeth is a villain", "Amelia is a hero",
              "Amelia is a hero/villain, Elizabeth is a hero/villain", "Amelia is a hero, Elizabeth is a hero",
              "Amelia is a hero, Elizabeth is a villain, Amelia is a villain"):
        assert S("knights_knaves", it, wrap(b)) == 0.0, b
    it2 = dict(it, meta=dict(it["meta"], knight="altruist", knave="egoist"))
    assert S("knights_knaves", it2, wrap("Amelia is an altruist, Elizabeth is an egoist")) == 1.0


# ------------------------------------------------------------------------------------------ math
def test_math_basic_equivalences():
    it = item("math", "\\frac{1}{2}")
    for b in ("\\frac{1}{2}", "\\dfrac{1}{2}", "1/2", "0.5", "\\frac12"):
        assert S("math", it, wrap(b)) == 1.0, b
    assert S("math", it, wrap("\\frac{1}{3}")) == 0.0
    it = item("math", "10^\\circ")
    assert S("math", it, wrap("10")) == 1.0
    it = item("math", "2\\sqrt{3}")
    assert S("math", it, wrap("2\\sqrt3")) == 1.0
    assert S("math", it, wrap("3\\sqrt{2}")) == 0.0


def test_math_verify_if_available():
    pytest.importorskip("math_verify")
    it = item("math", "\\frac{\\sqrt{3}}{2}")
    assert S("math", it, wrap("\\sqrt{3}/2")) == 1.0
    it = item("math", "(1,2)")
    assert S("math", it, wrap("(1, 2)")) == 1.0
    assert S("math", it, wrap("(2, 1)")) == 0.0
    it = item("math", "x^2+2x+1")
    assert S("math", it, wrap("(x+1)^2")) == 1.0


def test_gsm8k():
    it = item("gsm8k", "18")
    for b in ("18", "\\$18", "18 dollars", "18.00"):
        assert S("gsm8k", it, wrap(b)) == 1.0, b
    assert S("gsm8k", it, wrap("17")) == 0.0


def test_score_pred_unknown_task_is_zero():
    assert score_pred("chain_sum", None, item("chain_sum", "1")) == 0.0


def test_knights_name_substring():
    it = item("knights_knaves", "Ella is an altruist, Isabella is an egoist", names=["Ella", "Isabella"],
              solution=[True, False], knight="altruist", knave="egoist")
    assert S("knights_knaves", it, wrap("Ella is an altruist, Isabella is an egoist")) == 1.0
    assert S("knights_knaves", it, wrap("Isabella is an egoist, Ella is an altruist")) == 1.0
    assert S("knights_knaves", it, wrap("Ella is an egoist, Isabella is an egoist")) == 0.0

def test_heldout_scorers():
    from ultraviolet.tasks import score
    cases = [("ho_prime_factorization", "2 \\times 2 \\times 3", "2 × 2 × 3", 1),
             ("ho_prime_factorization", "2^2 \\cdot 3", "2 × 2 × 3", 1),
             ("ho_prime_factorization", "2 x 3", "2 × 2 × 3", 0),
             ("ho_number_sorting", "52.2, 30, 14, -49", "['52.2', '30.0', '14.0', '-49.0']", 1),
             ("ho_number_sorting", "30, 52.2", "['52.2', '30.0']", 0),
             ("ho_bitwise_arithmetic", "0x5b993", "0x5b993", 1), ("ho_bitwise_arithmetic", "375187", "0x5b993", 1),
             ("ho_dice", "0.5", "1/2", 1), ("ho_dice", "1/3", "1/2", 0),
             ("ho_syllogism", "yes", "Yes", 1), ("ho_family_relationships", "Mother-in-law", "mother-in-law", 1),
             ("ho_complex_arithmetic", "-15 + 3i", "-15.0 + 3.0i", 1),
             ("ho_complex_arithmetic", "-15 - 3i", "-15.0 + 3.0i", 0),
             ("ho_decimal_arithmetic", "6.320988", "6.32098837209", 1),
             ("ho_decimal_arithmetic", "6.33", "6.32098837209", 0),
             ("ho_rotate_matrix", "1 6\\\\ 0 0", "1 6\n0 0", 1), ("ho_svamp", "27", "27", 1)]
    for task, pred, gold, exp in cases:
        assert score(task, "</think>\\boxed{" + pred + "}", {"task": task, "answer": gold}) == exp, (task, pred)
