"""grader v2 (scripts/grader_v2.py).

Unit cases from the MATH-500 grader audit, gold self-match and model-form match on the eval sets, and adversarial
negatives. The eval-set tests read math500.jsonl, gsm8k_test.jsonl, gr_big2.jsonl and twin500.jsonl from the
directory in GRADER_DATA and are skipped when it is missing. `python tests/test_grader_v2.py` prints the report
(per set: v1 / v2 failures of the model-form check, remaining v2 failures)."""
import json
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import grader_v2  # noqa: E402
from grader_v2 import correct as v2  # noqa: E402


def v1(pred, ref):
    """eval_math.correct (copied: eval_math imports torch)."""
    def norm(a):
        a = str(a).strip()
        a = re.sub(r"\\text\{([^}]*)\}", r"\1", a)
        a = re.sub(r"\\(mathrm|mathbf)\{([^}]*)\}", r"\2", a)
        for t in ("\\!", "\\,", "\\;", "\\ ", "$", "\\left", "\\right", "^\\circ", "^{\\circ}", "\\%", "%", "dfrac",
                  "tfrac"):
            a = a.replace(t, "frac" if t in ("dfrac", "tfrac") else "")
        return a.replace(" ", "").rstrip(".").lower()

    def to_num(a):
        a = norm(a).replace(",", "")
        m = re.fullmatch(r"-?\\frac\{(-?[\d.]+)\}\{(-?[\d.]+)\}", a)
        try:
            if m:
                v = float(m.group(1)) / float(m.group(2))
                return -v if a.startswith("-") else v
            if re.fullmatch(r"-?[\d.]+/[\d.]+", a):
                p, q = a.split("/")
                return float(p) / float(q)
            return float(a)
        except (ValueError, ZeroDivisionError):
            return None
    if pred is None:
        return False
    if norm(pred) == norm(ref):
        return True
    a, b = to_num(pred), to_num(ref)
    return a is not None and b is not None and abs(a - b) <= 1e-6 * max(1.0, abs(b))


# ---------------------------------------------------------------------------------------------- audit cases
# (gold, prediction the model would write): v1 rejects all of these, v2 must accept them
AUDIT_POS = [
    ("\\frac43", "\\frac{4}{3}"), ("\\frac43", "4/3"), ("\\frac 59", "\\frac{5}{9}"), ("\\frac 59", "\\dfrac{5}{9}"),
    ("\\frac14", "\\frac{1}{4}"), ("\\frac65", "\\frac{6}{5}"), ("\\frac9{5}", "\\frac{9}{5}"),
    ("\\frac{270}7\\text{ degrees}", "\\frac{270}{7}"), ("\\frac{270}7\\text{ degrees}", "\\frac{270}{7}^\\circ"),
    ("864 \\mbox{ inches}^2", "864"), ("15\\mbox{ cm}^2", "15"), ("15\\mbox{ cm}^2", "15 \\text{ cm}^2"),
    ("\\$36", "36"), ("\\$32,\\!348", "32348"), ("\\$32,\\!348", "32,348"), ("\\$18.90", "18.90"),
    ("\\$18.90", "18.9"), ("\\$18.90", "\\$18.90"), ("5.4 \\text{ cents}", "5.4"), ("\\$40", "40"),
    ("99\\pi \\text{ square inches}", "99\\pi"), ("135\\text{ square feet}", "135"), ("1599 \\text{ dollars}", "1599"),
    ("\\text{(C)}", "C"), ("\\text{(C)}", "(C)"), ("\\text{(C)}", "\\textbf{(C)}"), ("\\text{(E)}", "E"),
    ("\\text{(A)}", "\\text{A}"),
    ("11\\sqrt2", "11\\sqrt{2}"), ("\\sqrt2", "\\sqrt{2}"), ("\\sqrt{2}", "\\sqrt 2"), ("\\frac{\\sqrt3}{2}",
                                                                                          "\\frac{\\sqrt{3}}{2}"),
    ("\\frac{\\sqrt{3}}{3}", "\\frac{\\sqrt3}3"), ("2\\sqrt[3]{2}", "2\\sqrt[3]2"),
    ("5", "x = 5"), ("5", "x=5"), ("-3", "a = -3"), ("12", "n_1 = 12"), ("x=5", "5"), ("x=5", "x = 5"),
    ("5", "5 \\text{ units}"), ("5", "5 units"), ("5", "5 \\text{ cm}"), ("24", "24 \\text{ square units}"),
    ("10,\\!080", "10080"), ("10,\\!080", "10{,}080"), ("11,\\! 111,\\! 111,\\! 100", "11111111100"),
    ("1\\frac{4}{5}", "\\frac{9}{5}"), ("1\\frac{4}{5}", "1.8"), ("(3,4]", "(3, 4]"),
    ("\\left( 3, \\frac{\\pi}{2} \\right)", "(3,\\frac\\pi2)"), ("\\left( \\frac{3}{2}, -13 \\right)", "(1.5,-13)"),
    ("3, 5, 7", "7, 5, 3"), ("1,-2", "-2, 1"), ("[2,5)", "x \\in [2,5)"), ("\\{1,2\\}", "\\left\\{1, 2\\right\\}"),
    ("\\begin{pmatrix} -1/3 \\\\ 2/3 \\\\ 5/3 \\end{pmatrix}", "\\begin{pmatrix}-1/3\\\\2/3\\\\5/3\\end{pmatrix}"),
    ("90^\\circ", "90"), ("25\\%", "25"), ("\\text{Evelyn}", "Evelyn"), ("x^{2}+1", "x^2+1"),
    ("\\dfrac{17}{50}", "0.34"), ("-\\frac{35}{9}", "\\frac{-35}{9}"), ("90", "90^\\circ,"), ("4", "\\$4.00"),
    ("22", "a_8 = 22"),
]


@pytest.mark.parametrize("gold,pred", AUDIT_POS)
def test_audit_positive(gold, pred):
    assert v2(pred, gold), (gold, pred)


def test_audit_cases_were_v1_failures():
    """Most audit cases are rejected by v1 (the motivation); v2 accepts every case v1 accepts."""
    fails = [c for c in AUDIT_POS if not v1(c[1], c[0])]
    assert len(fails) >= 35


# (gold, prediction) that must be rejected by v2
NEG = [
    ("\\frac{1}{2}", "\\frac{2}{1}"), ("\\frac12", "2"), ("\\frac{1}{2}", "2/1"), ("\\frac43", "\\frac34"),
    ("5", "6"), ("5", "-5"), ("-3", "3"), ("\\frac{1}{3}", "0.3333"), ("\\frac{1}{3}", "0.33"),
    ("(2,5)", "(5,2)"), ("(2,5]", "(2,5)"), ("[2,5)", "(2,5)"), ("(3,4]", "[3,4]"), ("(1,2)", "1,2,3"),
    ("(-\\infty,3]", "(-\\infty,3)"), ("\\left( 3, \\frac{\\pi}{2} \\right)", "(\\frac{\\pi}{2}, 3)"),
    ("11\\sqrt2", "11\\sqrt{3}"), ("11\\sqrt2", "\\sqrt{22}"), ("\\sqrt{2}", "2"), ("2\\sqrt{3}", "\\sqrt{23}"),
    ("\\sqrt23", "\\sqrt{23}"), ("\\text{(C)}", "D"), ("\\text{(C)}", "(B)"), ("\\text{(C)}", "3"),
    ("\\$36", "37"), ("\\$18.90", "18.09"), ("15\\mbox{ cm}^2", "16"), ("5.4 \\text{ cents}", "54"),
    ("12", "1,2"), ("1000", "1,00"), ("10,\\!080", "1008"), ("3, 5, 7", "3, 5"), ("3, 5, 7", "3, 5, 7, 9"),
    ("1,-2", "1,2"), ("x=5", "y=5"), ("x=5", "6"), ("5", "x = 6"), ("5", "x = 5 = y"),
    ("1\\frac{4}{5}", "\\frac{4}{5}"), ("1\\frac{4}{5}", "1"), ("2\\pi", "2"), ("2\\pi", "\\pi"),
    ("3+4i", "3+4"), ("3+4i", "3-4i"), ("1011_2", "1011"), ("1011_2", "11"), ("x^2+1", "x^3+1"),
    ("\\frac{\\sqrt{3}}{2}", "\\frac{\\sqrt{2}}{3}"), ("-\\frac{1}{2}", "\\frac{1}{2}"), ("0", "o"),
    ("\\text{east}", "\\text{west}"), ("\\begin{pmatrix} 1 \\\\ 2 \\end{pmatrix}", "\\begin{pmatrix}2\\\\1\\end{pmatrix}"),
    ("\\frac{270}7\\text{ degrees}", "270"), ("5", ""), ("5", None), ("y = 2x+3", "2x+3"), ("90^\\circ", "180"),
    ("2", "2\\pi"), ("4", "4\\sqrt{2}"), ("2", "2 x"), ("5", "5i"), ("10", "10^2"),
    ("10000", "1,0000"), ("1,-2", "1,"),
]


@pytest.mark.parametrize("gold,pred", NEG)
def test_negative(gold, pred):
    assert not v2(pred, gold), (gold, pred)


def test_brace_args():
    b = grader_v2.brace_args
    assert b("\\frac43") == "\\frac{4}{3}"
    assert b("\\frac 59") == "\\frac{5}{9}"
    assert b("\\frac\\pi2") == "\\frac{\\pi}{2}"
    assert b("\\frac{\\sqrt3}2") == "\\frac{\\sqrt{3}}{2}"
    assert b("11\\sqrt2") == "11\\sqrt{2}"
    assert b("\\sqrt[3]2") == "\\sqrt[3]{2}"
    assert b("\\sqrt23") == "\\sqrt{2}3"
    assert b("\\frac{1}{2}") == "\\frac{1}{2}"


# ---------------------------------------------------------------------------------------------- eval sets
DATA = os.environ.get("GRADER_DATA", "")
SETS = ("math500", "gsm8k_test", "gr_big2", "twin500")


def golds(name):
    p = os.path.join(DATA, name + ".jsonl")
    if not os.path.exists(p):
        pytest.skip(f"no {p}")
    return [str(json.loads(l)["answer"]) for l in open(p)]


def model_form(g):
    """The gold as a model writes it under the prompt's \\boxed{} convention (independent of grader_v2.canon):
    \\$ / \\! dropped, a trailing \\text / \\mbox unit dropped, \\text{(C)} -> C, brace-less \\frac / \\sqrt braced,
    thousands separators dropped."""
    s = g.replace("\\$", "").replace("\\!", "")
    s = re.sub(r"\s*\\(?:text|mbox)\{\s*[a-z][a-z ]+\}(\^\d)?\s*$", "", s).strip() or g
    s = re.sub(r"^\\text\{\((\w)\)\}$", r"\1", s)
    for _ in range(2):
        s = re.sub(r"\\frac\s*(\d)\s*(\d)", r"\\frac{\1}{\2}", s)
        s = re.sub(r"\\frac\s*(\d)\{", r"\\frac{\1}{", s)
        s = re.sub(r"\\frac(\{[^{}]*\})(\d)", r"\\frac\1{\2}", s)
        s = re.sub(r"\\sqrt\s*(\d)", r"\\sqrt{\1}", s)
    if re.fullmatch(r"\d{1,3}(,\s*\d{3})+", s):
        s = s.replace(",", "").replace(" ", "")
    return s


def variant_set(g):
    """Model-form variants that must all match (audit gold-variant check, minus the deliberately rejected ones)."""
    m = model_form(g)
    out = {"as_is": g, "dollars": f"${g}$", "no_spaces": g.replace(" ", ""), "model_form": m,
           "model_form_boxed_dollars": f"${m}$"}
    if "\\frac" in m:
        out["dfrac"] = m.replace("\\frac", "\\dfrac")
    f = re.fullmatch(r"(-?)\\frac\{(\d+)\}\{(\d+)\}", m)
    if f:
        out["slash"] = f"{f.group(1)}{f.group(2)}/{f.group(3)}"
    if re.fullmatch(r"-?\d+", m):
        out["x_eq"] = f"x = {m}"
        out["with_units"] = f"{m} \\text{{ units}}"
        out["trailing_dot0"] = f"{m}.0"
    if "\\left" in m:
        out["no_left_right"] = m.replace("\\left", "").replace("\\right", "")
    if re.search(r"\\sqrt\{\d\}", m):
        out["sqrt_braceless"] = re.sub(r"\\sqrt\{(\d)\}", r"\\sqrt\1", m)
    if "^\\circ" in m:
        out["no_degree"] = m.replace("^\\circ", "")
    elif re.fullmatch(r"\d+", m):
        out["degree"] = m + "^\\circ"
    return out


def check_set(name):
    G = golds(name)
    rep = {"n": len(G), "self_v1_fail": [], "self_v2_fail": [], "model_v1_fail": [], "model_v2_fail": [],
           "variant_v2_fail": []}
    for g in G:
        if not v1(g, g):
            rep["self_v1_fail"].append(g)
        if not v2(g, g):
            rep["self_v2_fail"].append(g)
        m = model_form(g)
        if not v1(m, g):
            rep["model_v1_fail"].append(g)
        if not v2(m, g):
            rep["model_v2_fail"].append(g)
        for k, v in variant_set(g).items():
            if not v2(v, g):
                rep["variant_v2_fail"].append((g, k, v))
    return rep


@pytest.mark.parametrize("name", SETS)
def test_eval_set_golds(name):
    r = check_set(name)
    assert not r["self_v2_fail"], r["self_v2_fail"]
    assert not r["model_v2_fail"], r["model_v2_fail"]
    assert not r["variant_v2_fail"], r["variant_v2_fail"][:20]


@pytest.mark.parametrize("name", SETS)
def test_v2_superset_of_v1_on_variants(name):
    """Every gold variant v1 accepts, v2 accepts (v2 only differs from v1 by accepting more canonical forms, except
    v1's comma-stripping, e.g. `1,2` read as 12, which v2 rejects)."""
    bad = []
    for g in golds(name):
        for k, v in variant_set(g).items():
            if v1(v, g) and not v2(v, g):
                bad.append((g, k, v))
    assert not bad, bad[:20]


def test_cross_gold_rejection():
    """Different MATH-500 golds must not match each other unless they are the same number (sanity check against an
    over-permissive canon)."""
    G = sorted(set(golds("math500")))
    hits = []
    for i, a in enumerate(G):
        for b in G[i + 1:]:
            if v2(a, b) or v2(b, a):
                hits.append((a, b))
    # equal values written differently (or the same unordered list) are the only allowed collisions
    for a, b in hits:
        na, nb = grader_v2.to_num(grader_v2.canon(a)), grader_v2.to_num(grader_v2.canon(b))
        same_list = sorted(grader_v2._split_top(grader_v2.canon(a))) == sorted(grader_v2._split_top(grader_v2.canon(b)))
        same_rhs = re.sub(r"^[a-z]=", "", grader_v2.canon(a)) == re.sub(r"^[a-z]=", "", grader_v2.canon(b))  # 5, x=5
        assert (na is not None and nb is not None and abs(na - nb) < 1e-9) or \
            grader_v2.canon(a) == grader_v2.canon(b) or same_list or same_rhs, (a, b)  # "1,-2" vs "-2,1": unordered root lists


if __name__ == "__main__":
    for n in SETS:
        r = check_set(n)
        print(n, json.dumps({k: (v if k == "n" else len(v)) for k, v in r.items()}))
        for k in ("model_v1_fail", "model_v2_fail", "self_v2_fail", "variant_v2_fail"):
            if r[k]:
                print("  ", k, json.dumps(r[k][:40], ensure_ascii=False))
