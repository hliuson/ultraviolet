"""Strict answer grader `grader-v2`. Pure Python, no dependencies.

grader v1 = eval_math.correct (norm string match, or to_num within 1e-6). v1 cannot match some MATH-500 golds in their
canonical form (brace-less \\frac43 / \\sqrt2, units, \\$, \\text{(C)}) and rejects `x = 5` and `5 \\text{ cm}`.
v2 canonicalises both sides (`canon`), then compares as strings, as exact numbers, element-wise inside matching
brackets, or as an unordered unbracketed comma list. Changes against v1:
  - \\frac / \\dfrac / \\tfrac / \\sqrt arguments are braced (\\frac43 -> \\frac{4}{3}, \\sqrt2 -> \\sqrt{2}); ^{c} -> ^c;
  - \\$, $, \\%, %, ^\\circ, \\left / \\right, \\displaystyle and \\! \\, \\; \\: spacing are dropped (v1 already dropped most);
  - a trailing unit is dropped: \\text{ words} / \\mbox{ words} (letters only, >= 2 letters, optional ^n), or a
    known unit word after a space (cm, inches, units, ...), only when something non-empty remains;
  - a multiple-choice letter \\text{(C)} / \\textbf{(C)} / (C) / C (A-E) -> c;
  - in the prediction only, and only when the gold has no `=` / `\\in`: a leading single-variable `x =` / `a_1 =` /
    `x \\in` is dropped (also on each element of a comma list);
  - commas are removed only as thousands separators (1,000 / 10{,}000), not between arbitrary digits (v1 read `1,2`
    as 12);
  - lists: same brackets on both sides -> element-wise, in order (intervals and tuples stay ordered); no brackets on
    the gold -> order-insensitive multiset of elements.
  - gold-only readings: a mixed number n\\frac{p}{q} (p < q) also matches its value; a gold `x = <number>` also
    matches the bare number;
  - trailing `.`, `,`, `;` are dropped.
Exact numbers keep v1's 1e-6 relative tolerance; a rounded decimal for a fraction (0.3333 for 1/3) still fails."""
import re

VERSION = "v2"
UNIT_WORDS = (r"(?:cm|mm|km|m|in|inch|inches|ft|feet|foot|yd|yards?|miles?|mi|units?|sq|square|cubic|degrees?|"
              r"cents?|dollars?|hours?|hrs?|minutes?|mins?|seconds?|secs?|days?|weeks?|years?|mph|pounds?|lbs?|"
              r"ounces?|oz|grams?|g|kg|liters?|litres?|ml|meters?|metres?|centimeters?|kilometers?|people|students)")
_SPACING = ("\\displaystyle", "\\!", "\\,", "\\;", "\\:", "\\ ", "~")
_LETTER = re.compile(r"\(?([a-e])\)?")


def _group(s, i):
    """s[i] == '{': return (content, index after the matching '}') or (None, i) if unbalanced."""
    depth = 0
    for k in range(i, len(s)):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                return s[i + 1:k], k + 1
    return None, i


def _arg(s, i):
    """One TeX macro argument starting at s[i] (spaces skipped): returns (braced-normalised text, next index)."""
    while i < len(s) and s[i] == " ":
        i += 1
    if i >= len(s):
        return None, i
    if s[i] == "{":
        c, j = _group(s, i)
        if c is None:
            return None, i
        return "{" + brace_args(c) + "}", j
    if s[i] == "\\":
        m = re.match(r"\\[A-Za-z]+", s[i:])
        if m:
            return "{" + m.group(0) + "}", i + len(m.group(0))
        return None, i
    if s[i] in "}^_":
        return None, i
    return "{" + s[i] + "}", i + 1


def brace_args(s):
    """Brace the arguments of \\frac (two) and \\sqrt (one, after an optional [n]): \\frac43 -> \\frac{4}{3},
    \\frac\\pi2 -> \\frac{\\pi}{2}, \\sqrt 2 -> \\sqrt{2}, \\sqrt[3]2 -> \\sqrt[3]{2}. Recurses into groups."""
    out, i = [], 0
    while i < len(s):
        m = re.match(r"\\(frac|sqrt)(?![A-Za-z])", s[i:])
        if not m:
            if s[i] == "{":
                c, j = _group(s, i)
                if c is not None:
                    out.append("{" + brace_args(c) + "}")
                    i = j
                    continue
            out.append(s[i])
            i += 1
            continue
        cmd = m.group(0)
        j = i + len(cmd)
        if m.group(1) == "frac":
            a1, j1 = _arg(s, j)
            a2, j2 = _arg(s, j1) if a1 is not None else (None, j1)
            if a1 is None or a2 is None:
                out.append(cmd)
                i = j
                continue
            out.append(cmd + a1 + a2)
            i = j2
        else:
            opt = ""
            while j < len(s) and s[j] == " ":
                j += 1
            if j < len(s) and s[j] == "[":
                k = s.find("]", j)
                if k < 0:
                    out.append(cmd)
                    i = j
                    continue
                opt, j = s[j:k + 1], k + 1
            a1, j1 = _arg(s, j)
            if a1 is None:
                out.append(cmd + opt)
                i = j
                continue
            out.append(cmd + opt + a1)
            i = j1
    return "".join(out)


def _unwrap_text(a):
    return re.sub(r"\\(?:text|textbf|textrm|textnormal|mbox|mathrm|mathbf|operatorname)\s*\{([^{}]*)\}", r"\1", a)


def canon(a, is_pred=False, gold_raw=None):
    """Canonical form of one answer string (lowercase, no whitespace)."""
    a = str(a).strip()
    if a.startswith("$") and a.endswith("$") and len(a) > 1:
        a = a.strip("$")
    a = a.replace("\\$", "").replace("$", "")
    a = a.replace("\\\\", "\x00")  # protect the TeX row separator \\ from the "\ " spacing rule
    for t in _SPACING:
        a = a.replace(t, " " if t in ("\\ ", "~") else "")
    a = re.sub(r"\\(?:left|right|big|Big|bigg|Bigg)(?![A-Za-z])", "", a)
    a = re.sub(r"\\[dtc]frac(?![A-Za-z])", r"\\frac", a)
    a = re.sub(r"\^\s*\{?\s*\\circ\s*\}?|°|\\textdegree|\\degree", "", a)
    a = a.replace("\\%", "").replace("%", "")
    a = a.strip()
    # trailing units: \text{ cm}^2, \mbox{ inches}, or a bare unit word after a space
    for _ in range(2):
        b = re.sub(r"\s*\\(?:text|mbox|textrm|textnormal|mathrm)\s*\{\s*[A-Za-z]{2,}(?:[ .]+[A-Za-z]+)*\.?\s*\}"
                   r"(?:\s*\^\s*\{?\d\}?)?\s*$", "", a)
        b = re.sub(r"(?<=[\d}a-zA-Z])\s+" + UNIT_WORDS + r"(?:\s+" + UNIT_WORDS + r")*(?:\s*\^\s*\{?\d\}?)?\s*\.?$",
                   "", b) if re.search(r"\d", b) else b
        if b.strip():
            a = b.strip()
    a = _unwrap_text(a)
    if is_pred and gold_raw is not None and "=" not in gold_raw and "\\in" not in gold_raw:
        a = _strip_assign(a)
    a = brace_args(a)
    a = re.sub(r"([\^_])\{(\w)\}", r"\1\2", a)
    a = re.sub(r"\s+", "", a).rstrip(".,;").lower()  # trailing punctuation (v1 stripped "." only, read "," via to_num)
    a = re.sub(r"\{,\}", ",", a)
    if re.fullmatch(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?", a):
        a = a.replace(",", "")
    m = _LETTER.fullmatch(a)
    if m:
        a = m.group(1)
    return a.replace("\x00", "\\\\")


_ASSIGN = re.compile(r"^\s*[A-Za-z](?:_\{?\w+\}?)?\s*(?:=|\\in(?![A-Za-z]))\s*")


def _strip_assign(a):
    parts = _split_top(a)
    if all(_ASSIGN.match(p) and p.count("=") <= 1 for p in parts):
        return ",".join(_ASSIGN.sub("", p, count=1) for p in parts)
    return a


def _split_top(s):
    """Split on commas at bracket depth 0."""
    out, depth, cur = [], 0, []
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return out


def to_num(c):
    """Exact number from a canonical string: integer / decimal, p/q, \\frac{p}{q} (signed). None otherwise."""
    m = re.fullmatch(r"(-?)\\frac\{(-?\d+(?:\.\d+)?)\}\{(-?\d+(?:\.\d+)?)\}", c)
    try:
        if m:
            v = float(m.group(2)) / float(m.group(3))
            return -v if m.group(1) else v
        m = re.fullmatch(r"(-?\d+(?:\.\d+)?)/(-?\d+(?:\.\d+)?)", c)
        if m:
            return float(m.group(1)) / float(m.group(2))
        if re.fullmatch(r"[-+]?(?:\d+\.?\d*|\.\d+)", c):
            return float(c)
    except ZeroDivisionError:
        return None
    return None


def _eq(p, g):
    if p == g:
        return True
    a, b = to_num(p), to_num(g)
    return a is not None and b is not None and abs(a - b) <= 1e-6 * max(1.0, abs(b))


_BR = {"(": ")", "[": "]", "\\{": "\\}"}


def _brackets(c):
    for o, e in _BR.items():
        if c.startswith(o) and c.endswith(e) and len(c) >= len(o) + len(e):
            inner = c[len(o):len(c) - len(e)]
            depth = 0
            for ch in inner:  # the outer pair must enclose the whole string, e.g. not "(1,2)\cup(3,4)"
                depth += ch in "([{"
                depth -= ch in ")]}"
                if depth < 0:
                    break
            else:
                if depth == 0:
                    return c[:len(o)], inner, c[len(c) - len(e):]
    if c.endswith(("]", ")")) and c[:1] in "([":
        inner = c[1:-1]
        if "," in inner and not re.search(r"[()\[\]]", inner):  # half-open interval (2,5] / [2,5)
            return c[0], inner, c[-1]
    return None


def equal_canon(p, g, _depth=0):
    if _eq(p, g):
        return True
    if _depth > 3:
        return False
    bg, bp = _brackets(g), _brackets(p)
    if bg and bp and bg[0] == bp[0] and bg[2] == bp[2]:
        eg, ep = _split_top(bg[1]), _split_top(bp[1])
        if len(eg) > 1 and len(eg) == len(ep):
            return all(equal_canon(x, y, _depth + 1) for x, y in zip(ep, eg))
        return False
    if bg is None and bp is None:
        eg, ep = _split_top(g), _split_top(p)
        if len(eg) > 1 and len(eg) == len(ep):
            left = list(eg)
            for x in ep:
                for k, y in enumerate(left):
                    if equal_canon(x, y, _depth + 1):
                        del left[k]
                        break
                else:
                    return False
            return True
    return False


_MIXED = re.compile(r"(\d+)\\frac\{(\d+)\}\{(\d+)\}")
_GOLD_ASSIGN = re.compile(r"[a-z](?:_\w+)?=([^=]+)")


def correct(pred, ref):
    """grader v2: True iff pred is the gold answer up to the canonicalisation above. Two gold-only readings: a mixed
    number n\\frac{p}{q} (p < q, MATH convention) also matches its value, and a gold `x = <number>` also matches the
    bare number."""
    if pred is None or ref is None:
        return False
    p, g = canon(pred, True, str(ref)), canon(ref)
    if not p:
        return False
    if equal_canon(p, g):
        return True
    m = _MIXED.fullmatch(g)
    if m and int(m.group(2)) < int(m.group(3)) and int(m.group(3)) > 0:
        v, pv = int(m.group(1)) + int(m.group(2)) / int(m.group(3)), to_num(p)
        return pv is not None and abs(pv - v) <= 1e-6 * max(1.0, abs(v))
    m = _GOLD_ASSIGN.fullmatch(g)
    if m and not re.search(r"[a-z]", re.sub(r"\\[a-z]+", "", m.group(1))):
        return equal_canon(p, m.group(1))
    return False
