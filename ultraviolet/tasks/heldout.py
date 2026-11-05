"""Held-out eval tasks (heldout_v1): reasoning-gym tasks outside our 22 + SVAMP / ASDiv / MAWPS. Never used for SFT,
midtrain or RL. Scorers are stdlib-only and binary; importing this module registers the tasks in score.TASKS (task
names prefixed "ho_"), so `ultraviolet.tasks.score(item["task"], completion, item)` works unchanged.
"""
import re
from fractions import Fraction

from ultraviolet.tasks.score import TASKS, norm_word, parse_number, sc_number, strip_latex

_NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _nums(s):
    if s is None:
        return None
    s = strip_latex(s).replace("−", "-")
    try:
        return [Fraction(x) for x in re.findall(_NUM, s)]
    except (ValueError, ZeroDivisionError):
        return None


def sc_num_seq(pred, item):
    """Ordered list of numbers (sorting, filtering, spiral / rotated matrix): same values in the same order;
    separators, brackets and quotes free."""
    a, b = _nums(pred), _nums(item["answer"])
    return a is not None and b is not None and len(b) > 0 and a == b


def sc_factors(pred, item):
    """Prime factorisation: multiset of factors (2 x 2 x 3, 2^2 * 3, 2 \\times 2 \\times 3) equals the gold."""
    if pred is None:
        return False
    s = strip_latex(pred).replace("\\times", "*").replace("\\cdot", "*").replace("×", "*").replace("·", "*")
    out = []
    for part in re.split(r"[*x]", s.replace(" ", "")):
        m = re.fullmatch(r"(\d+)(?:\^\{?(\d+)\}?)?", part)
        if not m:
            return False
        out += [int(m.group(1))] * int(m.group(2) or 1)
    gold = [int(x) for x in re.findall(r"\d+", str(item["answer"]))]
    return sorted(out) == sorted(gold)


def sc_rel(pred, item, tol=1e-4):
    """Decimal arithmetic: relative error <= tol (gold is a 12-significant-digit rounding)."""
    a, b = parse_number(pred, allow_units=False), parse_number(item["answer"])
    if a is None or b is None:
        return False
    return abs(a - b) <= tol * max(abs(b), Fraction(1, 10**6))


def _int_any(s):
    if s is None:
        return None
    s = strip_latex(s).replace(" ", "").rstrip(".").lower()
    m = re.fullmatch(r"([-+]?)0x([0-9a-f]+)", s)
    if m:
        v = int(m.group(2), 16)
        return -v if m.group(1) == "-" else v
    m = re.fullmatch(r"[-+]?\d+", s)
    return int(s) if m else None


def sc_hex(pred, item):
    """Bitwise arithmetic: same integer value, hex (0x..) or decimal."""
    a, b = _int_any(pred), _int_any(str(item["answer"]))
    return a is not None and a == b


def sc_value_fraction(pred, item):
    """Probability: same rational value (a/b, \\frac, decimal exact)."""
    return sc_number(pred, item)


def sc_choice(pred, item):
    """Yes/No, True/False, a relation word or a name: normalised word equality."""
    return pred is not None and norm_word(pred) == norm_word(str(item["answer"]))


def _cplx(s):
    if s is None:
        return None
    s = strip_latex(s).replace(" ", "").replace("j", "i").replace("−", "-")
    try:
        if "i" not in s:
            return complex(Fraction(s)), None
        s = re.sub(r"(^|[-+])i", r"\g<1>1i", s)
        return complex(s.replace("i", "j")), None
    except (ValueError, ZeroDivisionError):
        return None


def sc_complex(pred, item):
    a, b = _cplx(pred), _cplx(str(item["answer"]))
    return a is not None and b is not None and abs(a[0] - b[0]) < 1e-6


HELDOUT = {
    "ho_decimal_arithmetic": {"kind": "number", "scorer": sc_rel},
    "ho_prime_factorization": {"kind": "string", "scorer": sc_factors},
    "ho_number_sorting": {"kind": "list", "scorer": sc_num_seq},
    "ho_number_filtering": {"kind": "list", "scorer": sc_num_seq},
    "ho_bitwise_arithmetic": {"kind": "number", "scorer": sc_hex},
    "ho_dice": {"kind": "fraction", "scorer": sc_value_fraction},
    "ho_syllogism": {"kind": "string", "scorer": sc_choice},
    "ho_family_relationships": {"kind": "string", "scorer": sc_choice},
    "ho_number_sequence": {"kind": "number", "scorer": sc_number},
    "ho_count_primes": {"kind": "number", "scorer": sc_number},
    "ho_aiw": {"kind": "number", "scorer": sc_number},
    "ho_simple_geometry": {"kind": "number", "scorer": sc_number},
    "ho_course_schedule": {"kind": "string", "scorer": sc_choice},
    "ho_zebra_puzzles": {"kind": "string", "scorer": sc_choice},
    "ho_rotate_matrix": {"kind": "list", "scorer": sc_num_seq},
    "ho_spiral_matrix": {"kind": "list", "scorer": sc_num_seq},
    "ho_complex_arithmetic": {"kind": "string", "scorer": sc_complex},
    "ho_svamp": {"kind": "number", "scorer": sc_number},
    "ho_asdiv": {"kind": "number", "scorer": sc_number},
    "ho_mawps": {"kind": "number", "scorer": sc_number},
}
TASKS.update(HELDOUT)
