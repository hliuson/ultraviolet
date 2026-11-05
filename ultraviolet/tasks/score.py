"""Per-task scorers for the verifiable task suite (docs/tasks.md).

Entry point: score(task, completion_text, item) -> reward in [0, 1] ([0, inf) for ratio tasks).
  Every task declares its reward kind in TASKS[task]["reward"] (absent = "binary"; see reward_kind):
    binary: the scorer returns a bool (or exactly 0 / 1); score() returns 1.0 / 0.0. Every core and ho_ task.
    graded: the scorer returns a float in [0, 1] (partial credit, e.g. the workbench envs, docs/wb_envs.md); "solved"
            means reward == 1.0. Trainer methods that assume 0/1 rewards refuse graded pools (rl_grpo.binary_only).
    ratio:  the scorer returns a float >= 0, unbounded above: an improvement ratio over a shown input, so
            resubmitting the input scores 1. reward >= 1 then means "at least as good as the input", not solved.
            Treated as graded by the trainer guard.
  A binary task whose scorer returns anything else is a bug and raises TypeError (it is never silently graded).
  completion_text: the model's continuation of the prompt "Problem:\\n{problem}\\n\\n<think>\\n" (it may also start
  with "<think>" when it comes from a chat teacher). The answer is read from the text after the LAST "</think>";
  with no "</think>" the score is 0 (require_close=True, the RL/eval convention) or the whole text is used
  (require_close=False, for no-think / answer-only probes that already prepend "</think>").
  Answer = the last \\boxed{...} (balanced braces) of that tail.
  No bare-number fallback: an unboxed answer scores 0.
  item: the pool row (needs "task", "answer"; some tasks read "meta").

Dependencies: stdlib only, plus optional math_verify for the MATH tasks (falls back to string / numeric equivalence
when it is not importable; set TASKS_PYDEPS to a directory holding math_verify + latex2sympy2_extended + antlr4 to
make it importable from a venv that lacks it).
"""
import os
import re
import sys
from fractions import Fraction

# ----------------------------------------------------------------------------------------------- extraction
THINK_CLOSE = "</think>"


def last_boxed(s):
    """Content of the last \\boxed{...} / \\fbox{...} with balanced braces (None if absent or unbalanced)."""
    if s is None:
        return None
    i = max(s.rfind("\\boxed"), s.rfind("\\fbox"))
    if i < 0:
        return None
    j = i + (6 if s.startswith("\\boxed", i) else 5)
    while j < len(s) and s[j] == " ":
        j += 1
    if j >= len(s) or s[j] != "{":
        return None
    depth = 0
    for k in range(j, len(s)):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                return s[j + 1:k]
    return None


def answer_tail(text, require_close=True):
    """Text after the last </think> (None if there is none and require_close)."""
    if text is None:
        return None
    if THINK_CLOSE in text:
        return text.rsplit(THINK_CLOSE, 1)[1]
    return None if require_close else text


def extract_answer(task, text, require_close=True):
    _ensure(task)
    tail = answer_tail(text, require_close)
    if tail is None:
        return None
    return last_boxed(tail)


# ----------------------------------------------------------------------------------------------- normalisers
def strip_latex(s):
    s = str(s).strip()
    for _ in range(3):
        s = re.sub(r"\\(?:text|textbf|mathrm|mathbf|mbox|textit|operatorname)\s*\{([^{}]*)\}", r"\1", s)
    for t in ("\\left", "\\right", "\\!", "\\,", "\\;", "\\:", "\\ ", "\\quad", "\\qquad", "~"):
        s = s.replace(t, " ")
    s = s.replace("\\$", "$").replace("$", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    return s.strip()


_NUM_RE = re.compile(r"[-+]?(?:\d+(?:,\d{3})*|\d*)(?:\.\d+)?")


def parse_number(s, allow_units=True):
    """Exact rational value of a boxed number: 1,234 / -12 / 3.50 / 7/2 / \\frac{7}{2} / -\\frac{7}{2} / 12% / $18 /
    "x = 5" / "18 apples" (allow_units). Returns Fraction or None."""
    if s is None:
        return None
    s = strip_latex(s)
    s = re.sub(r"^[A-Za-z]\w*\s*=\s*", "", s)           # x = 5
    s = s.replace("\u2212", "-").replace("^\\circ", "").replace("^{\\circ}", "").replace("\\%", "%")
    s = s.strip().rstrip(".").strip()
    m = re.fullmatch(r"([-+]?)\s*\\frac\s*\{\s*([-+]?\d+)\s*\}\s*\{\s*([-+]?\d+)\s*\}", s)
    if m:
        try:
            v = Fraction(int(m.group(2)), int(m.group(3)))
        except ZeroDivisionError:
            return None
        return -v if m.group(1) == "-" else v
    m = re.fullmatch(r"([-+]?\d+)\s*/\s*([-+]?\d+)", s)
    if m:
        try:
            return Fraction(int(m.group(1)), int(m.group(2)))
        except ZeroDivisionError:
            return None
    m = re.fullmatch(r"([-+]?)\s*\$?\s*([-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|[-+]?\.\d+)\s*(%?)\s*([A-Za-z][A-Za-z .]*)?", s)
    if not m or (m.group(4) and not allow_units):
        return None
    try:
        v = Fraction(m.group(2).replace(",", ""))
    except (ValueError, ZeroDivisionError):
        return None
    return -v if m.group(1) == "-" else v


def parse_fraction_pair(s):
    """(num, den) as written (sign on the numerator), for 'lowest terms' checks. Integers give (n, 1)."""
    if s is None:
        return None
    s = strip_latex(s).replace("\u2212", "-").replace(" ", "").rstrip(".")
    m = (re.fullmatch(r"([-+]?)\\frac\{([-+]?\d+)\}\{([-+]?\d+)\}", s)
         or re.fullmatch(r"([-+]?)\(?([-+]?\d+)\)?/\(?([-+]?\d+)\)?", s))
    if m:
        n, d = int(m.group(2)), int(m.group(3))
        if m.group(1) == "-":
            n = -n
        if d < 0:
            n, d = -n, -d
        return (n, d) if d else None
    m = re.fullmatch(r"[-+]?\d+", s)
    return (int(s), 1) if m else None


def norm_word(s):
    if s is None:
        return None
    s = strip_latex(s).strip().strip("\"'`").rstrip(".").strip()
    return re.sub(r"\s+", "", s).lower()


# ----------------------------------------------------------------------------------------------- task scorers
def sc_number(pred, item):
    a, b = parse_number(pred), parse_number(item["answer"])
    return a is not None and b is not None and a == b


def sc_int_strict(pred, item):
    """Integer answer, units not allowed (letter/bit counts)."""
    a, b = parse_number(pred, allow_units=False), parse_number(item["answer"])
    return a is not None and b is not None and a == b


def sc_fraction_lowest(pred, item):
    p, g = parse_fraction_pair(pred), parse_fraction_pair(item["answer"])
    if p is None or g is None:
        return False
    return p == g  # gold is in lowest terms with positive denominator: equality == same value AND lowest terms


def sc_base(pred, item):
    if pred is None:
        return False
    s = strip_latex(pred).replace(" ", "").lower().rstrip(".")
    s = re.sub(r"_\{?\d+\}?$", "", s)                          # 1010_2, 1010_{2}
    s = re.sub(r"^(0x|0b|0o)", "", s)
    s = s.lstrip("0") or "0"
    g = str(item["answer"]).lower().lstrip("0") or "0"
    return bool(re.fullmatch(r"[0-9a-z]+", s)) and s == g


def _time_parts(s):
    s = strip_latex(s).replace(" ", "").rstrip(".")
    m = re.fullmatch(r"(-?\d+):(\d{1,2})(?::(\d{1,2})(?:\.(\d{1,3}))?)?", s)
    if not m:
        return None
    return tuple(None if x is None else int(x) for x in m.groups())


def sc_time(pred, item):
    if pred is None:
        return False
    a, b = _time_parts(pred), _time_parts(str(item["answer"]))
    return a is not None and b is not None and a == b


_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")


def sc_calendar(pred, item):
    g = str(item["answer"]).strip()
    if g.lower() in _WEEKDAYS:
        return norm_word(pred) == g.lower()
    return sc_int_strict(pred, item)


def sc_word(pred, item):
    return pred is not None and norm_word(pred) == norm_word(item["answer"])


def _countdown_eval(expr, numbers):
    """Exact value of an arithmetic expression over ints with + - * / and parens; checks the multiset of literals
    equals `numbers` (each used exactly once). Returns Fraction or None."""
    import ast
    expr = strip_latex(expr).replace("\\times", "*").replace("\\cdot", "*").replace("\u00d7", "*")
    expr = expr.replace("\\div", "/").replace("\u00f7", "/").replace("\u2212", "-").replace("x", "*")
    expr = expr.replace("[", "(").replace("]", ")").replace("{", "(").replace("}", ")")
    if "=" in expr:
        expr = expr.split("=")[0]
    if not re.fullmatch(r"[\d\s+\-*/().]+", expr) or "**" in expr or "//" in expr or len(expr) > 200:
        return None
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError:
        return None
    used = []

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and type(n.value) is int:
            used.append(n.value)
            return Fraction(n.value)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -ev(n.operand)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.UAdd):
            return ev(n.operand)
        if isinstance(n, ast.BinOp) and type(n.op) in (ast.Add, ast.Sub, ast.Mult, ast.Div):
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Add):
                return a + b
            if isinstance(n.op, ast.Sub):
                return a - b
            if isinstance(n.op, ast.Mult):
                return a * b
            if b == 0:
                raise ZeroDivisionError
            return a / b
        raise ValueError("bad node")

    try:
        v = ev(tree)
    except (ValueError, ZeroDivisionError, RecursionError):
        return None
    if sorted(used) != sorted(numbers):
        return None
    return v


def sc_countdown(pred, item):
    if pred is None:
        return False
    m = item["meta"]
    v = _countdown_eval(pred, list(m["numbers"]))
    return v is not None and v == m["target"]


def sc_sudoku(pred, item):
    """Accept any valid completed 4x4 grid consistent with the puzzle (digits read in order; separators free)."""
    if pred is None:
        return False
    s = strip_latex(pred)
    s = re.sub(r"\\begin\{[a-z]*\}|\\end\{[a-z]*\}|\\\\|&", " ", s)
    if re.search(r"[05-9]", s) or re.search(r"[A-Za-z]", s):
        return False
    d = [int(c) for c in re.findall(r"[1-4]", s)]
    if len(d) != 16:
        return False
    g = [d[i * 4:(i + 1) * 4] for i in range(4)]
    puz = item["meta"]["puzzle"]
    full = {1, 2, 3, 4}
    for i in range(4):
        if set(g[i]) != full or {g[r][i] for r in range(4)} != full:
            return False
        for j in range(4):
            if puz[i][j] and puz[i][j] != g[i][j]:
                return False
    for bi in (0, 2):
        for bj in (0, 2):
            if {g[bi + a][bj + b] for a in (0, 1) for b in (0, 1)} != full:
                return False
    return True


def sc_knights(pred, item):
    """Box like 'Amelia is a hero, Elizabeth is a villain' (or 'Amelia: hero, ...'). Each name must be followed by a
    role word; all names must be assigned exactly as in the solution."""
    if pred is None:
        return False
    m = item["meta"]
    s = strip_latex(pred).lower()
    good, bad = m["knight"].lower(), m["knave"].lower()
    for name, truth in zip(m["names"], m["solution"]):
        hits = re.findall(r"(?<![a-z])" + re.escape(name.lower()) + r"\s*(?:is\s+(?:an?\s+)?|:\s*|=\s*|-\s*|\(\s*)([a-z]+)(?![a-z]|\s*/)", s)
        roles = {h for h in hits if h in (good, bad)}
        if len(roles) != 1:
            return False
        if (roles.pop() == good) != bool(truth):
            return False
    return True


# --- math (GSM8K / MATH)
_MV = None


def _math_verify():
    global _MV
    if _MV is None:
        try:
            from math_verify import parse, verify  # noqa: F401
            _MV = (parse, verify)
        except Exception:
            extra = os.environ.get("TASKS_PYDEPS", "")
            if os.path.isdir(extra) and extra not in sys.path:
                sys.path.append(extra)
                try:
                    from math_verify import parse, verify  # noqa: F401
                    _MV = (parse, verify)
                except Exception:
                    _MV = False
            else:
                _MV = False
    return _MV or None


def _norm_math(a):
    a = strip_latex(a)
    a = re.sub(r"\^\{?\\circ\}?", "", a)
    a = a.replace("\\%", "").replace("%", "")
    a = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", a)
    a = re.sub(r"\\frac\{([^{}]*)\}(\d)", r"\\frac{\1}{\2}", a)
    a = re.sub(r"\\sqrt(\d)", r"\\sqrt{\1}", a)
    a = re.sub(r"^[a-zA-Z]\s*=\s*", "", a)
    a = a.replace(" ", "").rstrip(".").lower()
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", a):
        a = a.replace(",", "")
    return a


def math_equiv(pred, gold):
    """MATH-style equivalence. Fast path: normalised string / exact number; then math_verify if available."""
    if pred is None:
        return False
    if _norm_math(pred) == _norm_math(gold):
        return True
    a, b = parse_number(pred, allow_units=False), parse_number(gold, allow_units=False)
    if a is not None and b is not None:
        return a == b
    mv = _math_verify()
    if mv is None:
        return False
    import threading
    parse, verify = mv
    main = threading.current_thread() is threading.main_thread()
    kw_p = {} if main else {"parsing_timeout": None}
    kw_v = {} if main else {"timeout_seconds": None}
    try:
        g = parse("$" + gold + "$", **kw_p)
        p = parse("$" + pred + "$", **kw_p)
        return bool(g) and bool(p) and bool(verify(g, p, **kw_v))
    except Exception:
        return False


def sc_math(pred, item):
    return math_equiv(pred, str(item["answer"]))


def sc_gsm8k(pred, item):
    return sc_number(pred, item)


# ----------------------------------------------------------------------------------------------- registry
# Excluded from datasets, pools and evals: tokenizer-bound tasks that need the character content of multi-character
# tokens, which RL cannot create. Scorers stay registered so older pools that contain them can still be scored;
# builders skip these unless explicitly asked.
DROPPED_TASKS = frozenset({"letter_counting", "spell_backward"})
# kind: answer form (used by the writer and the answer-only probe); scorer: fn(pred, item) -> bool;
# reward (optional): "binary" (default) | "graded" (scorer returns a float in [0, 1]) | "ratio" (float >= 0); see reward_kind
REWARD_KINDS = ("binary", "graded", "ratio")
TASKS = {
    "chain_sum": {"kind": "number", "scorer": sc_number},
    "products": {"kind": "number", "scorer": sc_number},
    "basic_arithmetic": {"kind": "number", "scorer": sc_number},
    "simple_equations": {"kind": "number", "scorer": sc_number},
    "gcd": {"kind": "number", "scorer": sc_int_strict},
    "lcm": {"kind": "number", "scorer": sc_int_strict},
    "fraction_simplification": {"kind": "fraction", "scorer": sc_fraction_lowest},
    "base_conversion": {"kind": "string", "scorer": sc_base},
    "count_bits": {"kind": "number", "scorer": sc_int_strict},
    "time_intervals": {"kind": "time", "scorer": sc_time},
    "calendar_arithmetic": {"kind": "string", "scorer": sc_calendar},
    "leg_counting": {"kind": "number", "scorer": sc_number},
    "gsm_symbolic": {"kind": "number", "scorer": sc_number},
    "countdown": {"kind": "expression", "scorer": sc_countdown},
    "letter_counting": {"kind": "number", "scorer": sc_int_strict},
    "spell_backward": {"kind": "string", "scorer": sc_word},
    "mini_sudoku": {"kind": "grid", "scorer": sc_sudoku},
    "knights_knaves": {"kind": "string", "scorer": sc_knights},
    "gsm8k": {"kind": "number", "scorer": sc_gsm8k},
    "math": {"kind": "math", "scorer": sc_math},
}


def _ensure(task):
    if task not in TASKS and task.startswith("ho_"):
        import ultraviolet.tasks.heldout  # noqa: F401  registers the held-out tasks
    elif task not in TASKS and task.startswith("wb_"):
        import workbench.envs  # noqa: F401  registers the workbench RL envs (the one core -> workbench seam)


def reward_kind(task):
    """"binary" or "graded" for a registered task (registers ho_ / wb_ tasks on first use). Unknown tasks are
    binary (they score 0)."""
    _ensure(task)
    k = TASKS.get(task, {}).get("reward", "binary")
    if k not in REWARD_KINDS:
        raise ValueError(f"task {task!r}: reward must be one of {REWARD_KINDS}, got {k!r}")
    return k


def score_pred(task, pred, item):
    """Score an already-extracted answer (box content). Binary tasks: 1.0 / 0.0 (the scorer
    must return a bool, None or exactly 0 / 1, else TypeError). Graded tasks: the scorer's float, clipped to [0, 1]
    (ratio tasks: to [0, inf)) (NaN -> 0). A scorer that raises scores 0 either way."""
    kind = reward_kind(task)
    try:
        v = TASKS[task]["scorer"](pred, item)
    except Exception:
        return 0.0
    if kind == "binary":
        if isinstance(v, bool) or v is None:
            return 1.0 if v else 0.0
        if isinstance(v, (int, float)) and v in (0, 1):
            return float(v)
        raise TypeError(f"binary task {task!r}: scorer returned {v!r}; register it with \"reward\": \"graded\"")
    v = float(v)
    if v != v:
        return 0.0
    return max(0.0, v) if kind == "ratio" else min(1.0, max(0.0, v))


def score(task, completion_text, item, require_close=True):
    """Reward in [0, 1] for the completion's final answer on `item`: 1.0 / 0.0 for binary tasks, partial credit for
    graded ones ("solved" = 1.0); see the module docstring."""
    task = task or item.get("task")
    return score_pred(task, extract_answer(task, completion_text, require_close), item)


def score_item(completion_text, item, require_close=True):
    return score(item["task"], completion_text, item, require_close)
