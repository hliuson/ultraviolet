"""workbench/envs/grid.py: wb_grid missing-numbers scorer (prefix-aligned partial credit), generated gold, magic / line-sum invariants, uniqueness."""
import itertools
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from ultraviolet.tasks.score import TASKS, score  # noqa: E402
from workbench.envs import common as C  # noqa: E402
from workbench.envs import grid as G  # noqa: E402

MAGIC = [8, 1, 6, 3, 5, 7, 4, 9, 2]


def _item(gold, blank, shape=(3, 3), with_gb=True):
    given = [None if i in blank else v for i, v in enumerate(gold)]
    gb = [gold[i] for i in blank]
    meta = {"gold": gold, "blank": blank, "shape": list(shape), "given": given}
    if with_gb:
        meta["gold_blank"] = gb
    return {"task": "wb_grid", "answer": ", ".join(map(str, gb)), "meta": meta}


def _items(tier, n=60, seed=0):
    out = []
    for i in range(2000):
        r = G.gen(random.Random(seed * 100_000 + 31 * i + tier), tier)
        if r is not None:
            out.append(r)
        if len(out) >= n:
            break
    assert len(out) == n
    return out


def test_registered():
    assert TASKS["wb_grid"]["scorer"] is G.sc_grid and TASKS["wb_grid"]["reward"] == "graded"
    assert TASKS["wb_grid"]["kind"] == "grid"


def test_partial_credit_and_separators():
    it = _item(MAGIC, [0, 2, 6, 8])                                 # missing: 8, 6, 4, 2
    assert it["meta"]["gold_blank"] == [8, 6, 4, 2]
    assert G.sc_grid("8, 6, 4, 2", it) == 1.0
    assert G.sc_grid("8, 6, 4, 3", it) == 0.75                      # one of 4 blanks wrong
    assert G.sc_grid("8", it) == 0.25                               # single correct first value
    assert G.sc_grid("6", it) == 0.0                                # single value, misaligned
    assert G.sc_grid("8, 6, 4, 2, 1", it) == 4 / 5                  # one extra trailing value
    assert G.sc_grid("1, 2, 3, 5", it) == 0.0                       # wrong values
    assert G.sc_grid("8, 1, 6, 3, 5, 7, 4, 9, 2", it) == 1 / 9      # full grid: only position 0 aligns
    assert G.sc_grid("8 6\n4 2", it) == 1.0
    assert G.sc_grid("8; 6; 4; 2", it) == 1.0
    assert G.sc_grid("8 6 4 2", it) == 1.0
    assert G.sc_grid("8, 6; 4, 2", it) == 1.0
    assert G.sc_grid("\\begin{array}{cc} 8 & 6 \\\\ 4 & 2 \\end{array}", it) == 1.0
    assert G.sc_grid("8, x, 4, 2", it) == 0.0
    assert G.sc_grid(None, it) == 0.0
    assert G.sc_grid("8, 6, 4, 2", _item(MAGIC, [0, 2, 6, 8], with_gb=False)) == 1.0   # answer fallback
    assert score("wb_grid", "</think>\n\n\\boxed{8, 6, 4, 1}", it) == 0.75
    assert score("wb_grid", "</think>\n\n\\boxed{8}", it) == 0.25


def test_rank_helper():
    assert G.rank([[1, 0], [0, 1], [1, 1]]) == 2
    assert G.rank([[1, 2, 3], [2, 4, 6]]) == 1
    assert G.rank([MAGIC_ROW for MAGIC_ROW in G.MAGIC_COEF]) == 3


def _det3(m):
    (a, b, c), (d, e, f), (g, h, i) = m
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def _forest(R, Cc, blank):
    """Independent uniqueness check: blanks = edges row_i -- col_j; unique iff acyclic (union-find)."""
    parent = list(range(R + Cc))

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x
    for b in blank:
        u, v = find(b // Cc), find(R + b % Cc)
        if u == v:
            return False
        parent[u] = v
    return True


def _brute_unique(it_meta):
    """Tier-0 brute force: number of positive completions with entries in [1, max] matching all line sums."""
    R, Cc = it_meta["shape"]
    blank, given = it_meta["blank"], it_meta["given"]
    rs, cs, vmax = it_meta["row_sums"], it_meta["col_sums"], it_meta["max"]
    n = 0
    for vals in itertools.product(range(1, vmax + 1), repeat=len(blank) - 1):
        g = list(given)
        for b, v in zip(blank, vals):
            g[b] = v
        last = blank[-1]
        r = last // Cc
        g[last] = rs[r] - sum(g[r * Cc + j] for j in range(Cc) if r * Cc + j != last)
        if g[last] < 1:
            continue
        if all(sum(g[i * Cc:(i + 1) * Cc]) == rs[i] for i in range(R)) and \
                all(sum(g[j::Cc]) == cs[j] for j in range(Cc)):
            n += 1
    return n


def test_generated_gold_and_invariants():
    fams = set()
    for tier in range(3):
        for body, ans, meta, fam in _items(tier):
            fams.add(fam)
            it = C.make_item("wb_grid", fam, tier, "eval", 0, body, G.FORMAT, ans, meta)
            assert score("wb_grid", "</think>\n\n\\boxed{" + ans + "}", it) == 1.0
            assert score("wb_grid", "</think>\n\n\\boxed{" + ", ".join(map(str, meta["gold"])) + "}", it) < 1.0
            gold, blank, (R, Cc), given = meta["gold"], meta["blank"], meta["shape"], meta["given"]
            gb = meta["gold_blank"]
            assert len(gold) == R * Cc == len(given) and C.parse_list(ans) == gb
            assert blank == sorted(blank) and gb == [gold[i] for i in blank]
            assert f"There are {len(blank)} missing numbers." in body
            assert all((given[i] is None) == (i in blank) for i in range(R * Cc))
            assert all(given[i] == gold[i] for i in range(R * Cc) if given[i] is not None)
            assert body.count("_") >= len(blank)
            assert 0.0 <= meta["chance"] <= 1.0 and meta["chance"] < 0.2
            if fam == "magic3":
                S = meta["S"]
                assert f"Magic sum {S}." in body and (R, Cc) == (3, 3) and len(blank) == 6
                assert all(sum(gold[i] for i in ln) == S for ln in G.MAGIC_LINES)
                assert len(set(gold)) == 9 and min(gold) >= 1
                cells = [i for i in range(9) if i not in blank]
                assert not any(set(cells) == set(ln) for ln in G.MAGIC_LINES)
                assert _det3([G.MAGIC_COEF[i] for i in cells]) != 0     # (c, a, b) pinned
                cmax = (12, 20, 40)[tier]
                assert S // 3 <= cmax
                assert (4 in cells) == (tier == 0)
                if tier == 2:
                    assert sorted(i // 3 for i in cells) == [0, 1, 2]
            else:
                assert (R, Cc) == ((3, 3), (3, 4), (4, 4))[tier]
                assert len(blank) in ((3, 4), (5, 6), (7,))[tier]
                assert meta["row_sums"] == [sum(gold[i * Cc:(i + 1) * Cc]) for i in range(R)]
                assert meta["col_sums"] == [sum(gold[j::Cc]) for j in range(Cc)]
                assert "Row sums: " + ", ".join(map(str, meta["row_sums"])) + "." in body
                assert "Column sums: " + ", ".join(map(str, meta["col_sums"])) + "." in body
                assert 1 <= min(gold) and max(gold) <= (20 if tier == 0 else 50)
                assert _forest(R, Cc, blank)
                rounds, ok = G.cascade_rounds(R, Cc, blank)
                assert ok and rounds >= (1, 2, 3)[tier]
    assert fams == {"magic3", "sumgrid"}


def test_sumgrid_tier0_brute_force_unique():
    k = 0
    for body, ans, meta, fam in _items(0, n=40, seed=3):
        if fam != "sumgrid":
            continue
        assert _brute_unique(meta) == 1
        k += 1
    assert k >= 5


def test_non_unique_blanks_rejected_by_rank():
    # a 2x2 rectangle of blanks (cycle) is not unique
    assert G.rank(G.line_matrix(3, 3, [0, 1, 3, 4])) == 3
    assert not _forest(3, 3, [0, 1, 3, 4])
    assert G.rank(G.line_matrix(3, 3, [0, 1, 5])) == 3 and _forest(3, 3, [0, 1, 5])


def test_determinism():
    for tier in range(3):
        a = G.gen(random.Random(12345 + tier), tier)
        b = G.gen(random.Random(12345 + tier), tier)
        assert a == b


def test_build_items_covers_families():
    items, stats = C.build_items("wb_grid", G.gen, G.FORMAT, {"eval": 30, "rl": 60}, seed=0)
    assert len(items["eval"]) == 30 and len(items["rl"]) == 60
    allit = items["eval"] + items["rl"]
    assert {it["family"] for it in allit} == {"magic3", "sumgrid"}
    assert {it["tier"] for it in allit} == {0, 1, 2}
    assert not ({C.qkey(it["problem"]) for it in items["eval"]} & {C.qkey(it["problem"]) for it in items["rl"]})
    for it in allit:
        assert it["problem"].endswith(G.FORMAT + " " + C.INSTR)
