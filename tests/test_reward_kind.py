"""Reward kinds (ultraviolet.tasks.score: TASKS[task]["reward"] = "binary" | "graded") and the trainer guard that keeps
binary-only methods off graded pools (rl_grpo.binary_only / graded_tasks)."""
import argparse
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ultraviolet.tasks.score import TASKS, reward_kind, score, score_pred  # noqa: E402


@pytest.fixture
def tmp_tasks():
    added = []

    def add(name, spec):
        TASKS[name] = spec
        added.append(name)

    yield add
    for n in added:
        TASKS.pop(n, None)


def test_core_and_heldout_tasks_are_binary():
    import ultraviolet.tasks.heldout  # noqa: F401
    for t in TASKS:
        if not t.startswith("wb_"):
            assert reward_kind(t) == "binary", t


def test_workbench_envs_declare_graded():
    import workbench.envs  # noqa: F401
    wb = [t for t in TASKS if t.startswith("wb_") and not t.startswith("wb__")]
    want = {t: "binary" for t in TASKS if t.startswith("wb_po_")}   # any one valid answer
    wb = [t for t in wb if not TASKS[t]["scorer"].__module__.startswith("workbench.envs.local.")]  # local (gitignored) envs
    assert wb and all(reward_kind(t) == want.get(t, "graded") for t in wb), wb


def test_unknown_task_is_binary_and_scores_zero():
    assert reward_kind("wb_does_not_exist") == "binary"
    assert score("wb_does_not_exist", "x</think>\\boxed{1}", {"answer": "1"}) == 0.0


def test_invalid_reward_kind_rejected(tmp_tasks):
    tmp_tasks("wb__bad_kind", {"kind": "number", "scorer": lambda p, it: True, "reward": "continuous"})
    with pytest.raises(ValueError):
        reward_kind("wb__bad_kind")


def test_binary_scorer_returning_fraction_raises(tmp_tasks):
    tmp_tasks("wb__half", {"kind": "number", "scorer": lambda p, it: 0.5})        # forgot "reward": "graded"
    with pytest.raises(TypeError):
        score_pred("wb__half", "1", {"answer": "1"})


def test_binary_scorer_accepts_bool_none_and_exact_01(tmp_tasks):
    for name, v, want in (("wb__t", True, 1.0), ("wb__f", False, 0.0), ("wb__n", None, 0.0),
                          ("wb__one", 1, 1.0), ("wb__zero", 0.0, 0.0)):
        tmp_tasks(name, {"kind": "number", "scorer": lambda p, it, v=v: v})
        assert score_pred(name, "x", {"answer": "x"}) == want, name


def test_graded_scorer_clips_and_handles_nan(tmp_tasks):
    tmp_tasks("wb__g", {"kind": "list", "scorer": lambda p, it: float(p), "reward": "graded"})
    it = {"answer": "x"}
    assert score_pred("wb__g", "0.25", it) == 0.25
    assert score_pred("wb__g", "3", it) == 1.0
    assert score_pred("wb__g", "-2", it) == 0.0
    assert score_pred("wb__g", "nan", it) == 0.0
    assert score_pred("wb__g", "oops", it) == 0.0                                # scorer raised


def test_ratio_scorer_is_unbounded_above(tmp_tasks):
    tmp_tasks("wb__r", {"kind": "list", "scorer": lambda p, it: float(p), "reward": "ratio"})
    it = {"answer": "x"}
    assert score_pred("wb__r", "1", it) == 1.0
    assert score_pred("wb__r", "37.5", it) == 37.5
    assert score_pred("wb__r", "-2", it) == 0.0
    assert score_pred("wb__r", "nan", it) == 0.0
    import rl_grpo as m
    assert m.graded_tasks([{"task": "wb__r"}]) == ["wb__r"]                     # the binary-only guard covers it


def test_scorer_exception_scores_zero_for_both_kinds(tmp_tasks):
    def boom(p, it):
        raise RuntimeError

    tmp_tasks("wb__boom_b", {"kind": "number", "scorer": boom})
    tmp_tasks("wb__boom_g", {"kind": "list", "scorer": boom, "reward": "graded"})
    assert score_pred("wb__boom_b", "1", {}) == 0.0 and score_pred("wb__boom_g", "1", {}) == 0.0


# ------------------------------------------------------------------------------------------ trainer guard
rl_grpo = None


def _rl():
    global rl_grpo
    if rl_grpo is None:
        pytest.importorskip("torch")
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
        import rl_grpo as m
        rl_grpo = m
    return rl_grpo


def ns(**kw):
    d = dict(adv="grpo", resample_zero=0, sampler="uniform")
    d.update(kw)
    return argparse.Namespace(**d)


def test_binary_only_lists_each_binary_method():
    m = _rl()
    assert m.binary_only(ns()) == []
    assert m.binary_only(ns(sampler="pool")) == []
    for adv in ("maxrl", "psr", "nsr", "wreinforce"):
        assert m.binary_only(ns(adv=adv)) == [f"--adv {adv}"]
    assert m.binary_only(ns(resample_zero=4)) == ["--resample-zero"]


def test_graded_tasks_from_rows(tmp_tasks):
    m = _rl()
    tmp_tasks("wb__g2", {"kind": "list", "scorer": lambda p, it: 1.0, "reward": "graded"})
    rows = [{"task": "gsm8k"}, {"task": "wb__g2"}, {"task": "wb__g2"}, {"problem": "no task field"}]
    assert m.graded_tasks(rows) == ["wb__g2"]
    assert m.graded_tasks([{"task": "gsm8k"}, {"problem": "p"}]) == []
