"""Answer-type survey of NuminaMath-1.5 / DeepMath reference answers (classify() is also used by eval_v3). Usage: python -I anstype.py numina15_dir deepmath_dir out.json"""
import collections, glob, json, os, random, re, sys
try:
    import pyarrow.parquet as pq  # only the parquet survey needs it; eval_v3 imports classify() alone
except ImportError:  # pragma: no cover
    pq = None

NUMFRAC = re.compile(r"^-?\d{1,7}(\.\d{1,6})?$|^-?\\[dt]?frac\{\d{1,6}\}\{\d{1,6}\}$|^-?\d{1,6}/\d{1,6}$")
NOSOL = re.compile(r"no (real |integer |such )?(solution|value|number)s?|does ?n[o']t exist|not exist|\\(empty|varnothing)|\bnone\b|∅|impossible|no such", re.I)
YESNO = re.compile(r"^(\\text\{)?\s*(yes|no|true|false|correct|incorrect)\b", re.I)
MCQ = re.compile(r"^(\\text\{)?\(?\s*[A-E]\s*\)?(\})?$|^\\(text|mathrm|textbf)\{\(?[A-E]\)?\}$")
CMDS = re.compile(r"\\[a-zA-Z]+")


def norm(a):
    a = (a or "").strip().strip("$").strip()
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", a):
        a = a.replace(",", "")
    return a


def top_comma(a):
    d = 0
    for ch in a:
        if ch in "([{":
            d += 1
        elif ch in ")]}":
            d -= 1
        elif ch == "," and d == 0:
            return True
    return False


def classify(a):
    a = norm(a)
    if not a:
        return "empty"
    if NUMFRAC.match(a):
        return "scalar_numfrac"
    if MCQ.match(a):
        return "mcq_letter"
    if NOSOL.search(a):
        return "no_solution"
    if YESNO.match(a):
        return "yes_no"
    core = CMDS.sub("", a)
    words = re.findall(r"[A-Za-z]{3,}", core)
    if "\\infty" in a or "\\cup" in a or re.match(r"^[\(\[].*,.*[\)\]]$", a) and (a[0] == "[" or a[-1] == "]"):
        return "interval"
    if re.match(r"^\\\{.*\\\}$", a) or re.match(r"^\{.*\}$", a):
        return "set"
    if re.match(r"^\(.*,.*\)$", a):
        return "tuple_or_open_interval"
    if top_comma(a) or re.search(r"\bor\b|\\text\{ or \}|\\text\{and\}|\band\b", a):
        return "list"
    if words and len(" ".join(words)) > 0 and not re.search(r"[=<>]", a) and not re.search(r"\d", a):
        return "text_word"
    if re.search(r"[<>]|\\[lg]eq?|\\le\b|\\ge\b", a):
        return "inequality"
    if "=" in a:
        return "equation"
    if re.search(r"(?<![a-zA-Z\\])[a-zA-Z](?![a-zA-Z])", core.replace("pi", "")):
        return "expression"
    return "scalar_other"   # sqrt, pi, units, percent, sci notation, ...


def main():
    n15, dm, out = sys.argv[1:4]
    res = {}
    ex = collections.defaultdict(list)
    rng = random.Random(0)
    c = collections.defaultdict(collections.Counter)
    for f in sorted(glob.glob(os.path.join(n15, "*.parquet"))):
        for r in pq.read_table(f, columns=["problem", "answer", "source", "question_type", "problem_is_valid",
                                           "solution_is_valid"]).to_pylist():
            src = r["source"]
            c["qtype/" + src][str(r.get("question_type"))] += 1
            p = (r.get("problem") or "").strip()
            if not (r.get("question_type") in ("math-word-problem", None, "NaN") and r.get("problem_is_valid") != "No"
                    and r.get("solution_is_valid") != "No" and 20 <= len(p) <= 1500 and "[asy]" not in p
                    and "figure" not in p.lower()):
                c["filtered_out"][src] += 1
                continue
            t = classify(r.get("answer"))
            c["atype/" + src][t] += 1
            if src in ("orca_math", "synthetic_math", "cn_k12") and len(ex[src + "/" + t]) < 6 and rng.random() < 0.05:
                ex[src + "/" + t].append([p[:200], norm(r.get("answer"))[:80]])
            # MCQ rows (answer type of MCQ questions)
        for r in pq.read_table(f, columns=["answer", "source", "question_type"]).to_pylist():
            if r.get("question_type") == "MCQ":
                c["mcq_answers/" + r["source"]][classify(r.get("answer"))] += 1
    # DeepMath
    for f in sorted(glob.glob(os.path.join(dm, "**", "*.parquet"), recursive=True)):
        cols = pq.read_schema(f).names
        for r in pq.read_table(f, columns=[x for x in ("final_answer", "difficulty", "topic") if x in cols]).to_pylist():
            d = r.get("difficulty")
            c["deepmath_diff"][str(d)] += 1
            if d is not None and float(d) <= 4.5:
                c["deepmath_atype_le4.5"][classify(r.get("final_answer"))] += 1
            c["deepmath_atype_all"][classify(r.get("final_answer"))] += 1
    res = {k: dict(v.most_common()) for k, v in c.items()}
    res["examples"] = ex
    json.dump(res, open(out, "w"), indent=1, ensure_ascii=False)


main()
