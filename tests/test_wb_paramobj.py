"""wb_po_* (workbench/envs/paramobj, docs/wb_paramobj.md): registration, deterministic regeneration, scoring, builder."""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultraviolet.tasks.score import TASKS, reward_kind, score  # noqa: E402
from workbench.envs.paramobj import build, core, paramobj as po  # noqa: E402

SEED, PARAMS, VALID = 141063620200000000, {"n": 6, "B": 10, "S": 81}, ["-9, 9", "-4, 7", "1, 5", "6, 3"]


def test_registered():
    names = [t for t in TASKS if t.startswith("wb_po_")]
    assert len(names) == len(po.FAMILIES) == 53
    assert all(reward_kind(t) == "binary" for t in names)


def test_row_regenerates_from_seed():
    assert build.seed_of("po_ap_sum", 0, 0) == SEED
    p = po.sample_params("po_ap_sum", SEED, 0)
    assert p == PARAMS
    r = po.make_row("po_ap_sum", p, 0, "rl", 0, SEED)
    assert r["id"] == "po-ap_sum-t0-rl-000000" and r["task"] == "wb_po_ap_sum"
    assert r["meta"]["valid_answers"] == VALID and r["n_valid"] == 4
    assert {core.canon(r, a) for a in VALID} == core.enumerate_valid(r)


def test_scoring():
    r = po.make_row("po_ap_sum", PARAMS, 0, "rl", 0, SEED)
    for a in VALID + ["a = 1, d = 5"]:
        assert score(r["task"], f"x</think>\n\\boxed{{{a}}}", r) == 1.0, a
    assert score(r["task"], "x</think>\n\\boxed{1, 6}", r) == 0.0
    assert score(r["task"], "\\boxed{1, 5}", r) == 0.0          # no </think>
    s = core.score(r, "x</think>\n\\boxed{6, 3}")
    assert s["correct_strict"] and s["key"] == core.canon(r, "6, 3")


def test_build(tmp_path):
    build.main(["po", "--out", str(tmp_path), "--families", "po_ap_sum", "--tiers", "0", "--per-cell", "3",
                "--max-draws", "200"])
    rows = [json.loads(x) for x in (tmp_path / "rl_train.jsonl").read_text().splitlines()]
    assert len(rows) == 3 and rows[0]["meta"]["seed"] == SEED
    for r in rows:
        assert 2 <= r["n_valid"] <= 64 and r["guess_rate"] <= 0.02
        assert all(core.verify(r, a) for a in r["meta"]["valid_answers"])
