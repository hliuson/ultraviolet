"""Elastic --resume onto a different world size (rl_grpo.resume_elastic, rl_sampler.rewidth_sampler_state):
counter continuity, sampler state migration, rng re-derivation, and the unchanged same-world path."""
import collections
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import rl_grpo  # noqa: E402
from rl_sampler import DifficultyPoolSampler, UniformSampler, jumped_state, rewidth_sampler_state  # noqa: E402

TOT = {"rollouts": 1001, "gen_tokens": 7_000_003, "groups_started": 17, "ds_filtered_groups": 5,
       "ds_kept_zero_var_groups": 3, "stale_groups": 0, "orphan_rollouts": 2}


def test_rank_share_sums():
    for n in (0, 1, 15, 16, 1001):
        for W in (1, 2, 3, 4):
            sh = [rl_grpo.rank_share(n, W, r) for r in range(W)]
            assert sum(sh) == n and max(sh) - min(sh) <= 1
    assert [rl_grpo.rank_share(16, 2, r) for r in range(2)] == [8, 8]


def test_global_totals_and_split_continuity():
    # 1 rank (old format: no per-rank totals): exact
    R1 = {"world": 1, "totals": dict(TOT), "ranks": [{}]}
    g, exact = rl_grpo.global_totals(R1)
    assert exact and g == TOT
    for W in (2, 3):
        parts = [rl_grpo.split_totals(g, W, r) for r in range(W)]
        merged = rl_grpo.merge_recs([{f"total_{k}": v for k, v in p.items()} for p in parts])
        assert merged == {f"total_{k}": v for k, v in TOT.items()}  # logged (summed) totals continue exactly
        for p in parts:  # each rank's share ~ the global count / world
            assert abs(p["rollouts"] * W - TOT["rollouts"]) < W
    # 2 ranks with per-rank totals: the sum (not master x 2)
    R2 = {"world": 2, "totals": {"rollouts": 10}, "ranks": [{"totals": {"rollouts": 10}}, {"totals": {"rollouts": 13}}]}
    assert rl_grpo.global_totals(R2) == ({"rollouts": 23}, True)
    # 2 ranks, old format: master x world, flagged approximate
    R2o = {"world": 2, "totals": {"rollouts": 10}, "ranks": [{}, {}]}
    assert rl_grpo.global_totals(R2o) == ({"rollouts": 20}, False)
    # 1 -> 1 is the same-world path and never splits
    assert rl_grpo.split_totals({"x": 2.5}, 2, 0) == {"x": 1.25}


def pool(rank, world, seed=0, n=40):
    """--sampler pool as main() builds it: per-rank draw rng (seed + 1000 r), pool state identical on every rank."""
    return DifficultyPoolSampler(n, np.random.default_rng(seed + 1000 * rank))


def test_pool_state_migration_1_to_2():
    old = pool(0, 1)
    rng = np.random.default_rng(5)
    for k in range(5):
        idx = old.sample(4)
        old.update(idx, (rng.random(len(idx)) < .5).astype(float).tolist())
    saved = old.state_dict()
    step = 5
    news = []
    for r in range(2):
        s = pool(r, 2)
        fresh = s.state_dict()
        st = rewidth_sampler_state(saved, fresh, step)
        assert st["rng"] == jumped_state(fresh["rng"], step) != saved["rng"]
        s.load_state_dict(st)
        np.testing.assert_array_equal(s.pool, old.pool)  # shared pool state carried over
        news.append(s)
    assert news[0].rng.bit_generator.state != news[1].rng.bit_generator.state
    assert news[0].sample(8) != news[1].sample(8)  # per-rank draws differ after the migration


def test_uniform_shared_stream_migration():
    # uniform: the shared permutation stream is carried over whole; rank r of 2 takes positions r, r + 2, ...
    u = UniformSampler(10, np.random.default_rng(0))
    u.sample(7)
    sv = u.state_dict()
    ref = UniformSampler(10, np.random.default_rng(0))
    ref.load_state_dict(sv)
    nxt = ref.sample(6)
    for r in range(2):
        s = UniformSampler(10, np.random.default_rng(0), shard=(r, 2))
        s.load_state_dict(rewidth_sampler_state(sv, s.state_dict(), 3))
        assert s.sample(3) == nxt[r::2]
    with pytest.raises(AssertionError):  # sampler kind must match
        rewidth_sampler_state(sv, pool(0, 1).state_dict(), 1)


class FakeTr:
    def __init__(self):
        self.loaded = None

    def load_state_dict(self, d):
        self.loaded = d


def _resume_state(world):
    ranks = []
    for r in range(world):
        ranks.append({"sampler": pool(r, world).state_dict(), "rrng": np.random.default_rng(99 + r).bit_generator.state,
                      "torch": torch.get_rng_state(), "cuda": None, "totals": {"rollouts": 100 + r}})
    return {"step": 7, "world": world, "trainer": {"t": 1}, "ranks": ranks, "totals": {"rollouts": 100},
            "train_s": 3600.0, "hours": 1.5}


def test_apply_resume_same_world_unchanged_and_elastic():
    rl_grpo.RS.update(world_change=None)
    R = _resume_state(1)
    tr = FakeTr()
    s = pool(0, 1)
    rl_grpo.apply_resume(R, None, tr, s, 0, 1, seed=0)
    assert tr.loaded == {"t": 1} and rl_grpo.RS["start"] == 7 and rl_grpo.RS["world_change"] is None
    assert rl_grpo.RS["totals"] == collections.Counter({"rollouts": 100})
    assert rl_grpo.RRNG.bit_generator.state == R["ranks"][0]["rrng"]  # saved rngs restored exactly
    # elastic 1 -> 2: counters split, rngs re-derived (distinct per rank, not the step-0 streams), clock carried
    outs = []
    for r in range(2):
        rl_grpo.RS.update(world_change=None, totals=collections.Counter())
        s = pool(r, 2)
        rl_grpo.apply_resume(R, None, FakeTr(), s, r, 2, seed=0)
        assert rl_grpo.RS["world_change"] == (1, 2) and rl_grpo.RS["start"] == 7 and rl_grpo.RS["train_s"] == 3600.0
        assert rl_grpo.RRNG.bit_generator.state != np.random.default_rng(11 + 1000 * r).bit_generator.state
        outs.append((rl_grpo.RS["totals"]["rollouts"], rl_grpo.RRNG.random(), torch.rand(1).item()))
    assert outs[0][0] + outs[1][0] == 100
    assert outs[0][1] != outs[1][1] and outs[0][2] != outs[1][2]
    rl_grpo.RS.update(world_change=None, start=0, totals=collections.Counter(), train_s=0.0, hours=0.0)
