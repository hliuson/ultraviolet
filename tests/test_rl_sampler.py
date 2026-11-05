import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from rl_sampler import POOLS, DifficultyPoolSampler, UniformSampler, make_sampler  # noqa: E402


def test_pool_boundaries():
    s = DifficultyPoolSampler(1, np.random.default_rng(0))
    G = 8
    got = {r: POOLS[s.pool_of(r / G)] for r in range(G + 1)}
    assert got[0] == "zero"
    assert got[1] == "low" and got[2] == "low"  # 0.125, 0.25 -> (0, 0.25]
    assert got[3] == "mid" and got[6] == "mid"  # 0.375 .. 0.75 -> (0.25, 0.75]
    assert got[7] == "high"  # 0.875 -> (0.75, 1)
    assert got[8] == "solved"


def test_unseen_uniform_and_with_replacement():
    s = DifficultyPoolSampler(3, np.random.default_rng(0))
    assert np.allclose(s.probs(), 1 / 3)
    idx = s.sample(10)  # more than n: must be with replacement
    assert len(idx) == 10 and set(idx) <= {0, 1, 2}
    assert s.stats()["pool_drawn_unseen"] == 1.0
    assert s.stats()["pool_n_unseen"] == 3


def test_weights_proportional():
    s = DifficultyPoolSampler(6, np.random.default_rng(0))
    s.update([1, 2, 3, 4, 5], [0.0, 0.125, 0.5, 0.875, 1.0])  # 0 stays unseen
    w = np.array([1.0, 0.1, 0.5, 1.0, 0.3, 0.05])
    assert np.allclose(s.probs(), w / w.sum())
    st = s.stats()
    assert [st[f"pool_n_{p}"] for p in POOLS] == [1, 1, 1, 1, 1, 1]
    rng_counts = np.bincount(s.sample(200000), minlength=6) / 200000
    assert np.allclose(rng_counts, w / w.sum(), atol=0.01)


def test_latest_reward_wins_and_reclassify():
    s = DifficultyPoolSampler(2, np.random.default_rng(0))
    s.update([0, 0], [0.0, 1.0])
    assert POOLS[s.pool[0]] == "solved"
    s.update([0], [0.5])
    assert POOLS[s.pool[0]] == "mid"


def test_custom_thresholds_weights_and_zero_weights():
    s = DifficultyPoolSampler(2, np.random.default_rng(0), lo=0.1, hi=0.9, weights=(1, 0, 0, 0, 0, 0))
    s.update([0, 1], [0.0, 1.0])
    assert np.allclose(s.probs(), 0.5)  # all weights zero -> uniform fallback
    s2 = DifficultyPoolSampler(1, np.random.default_rng(0), lo=0.1, hi=0.9)
    assert POOLS[s2.pool_of(0.2)] == "mid" and POOLS[s2.pool_of(0.95)] == "high"
    with pytest.raises(AssertionError):
        DifficultyPoolSampler(1, np.random.default_rng(0), lo=0.8, hi=0.5)


def test_uniform_epochs():
    u = make_sampler("uniform", 5, np.random.default_rng(0))
    assert isinstance(u, UniformSampler)
    a = u.sample(5)
    assert sorted(a) == list(range(5))
    assert sorted(u.sample(5)) == list(range(5))


def test_uniform_shard_covers_epoch():
    import numpy as np
    from rl_sampler import UniformSampler
    n, W = 20, 2
    ss = [UniformSampler(n, np.random.default_rng(5), (r, W)) for r in range(W)]
    got = ss[0].sample(10) + ss[1].sample(10)
    assert sorted(got) == list(range(n))  # the two ranks split one epoch exactly
    single = UniformSampler(n, np.random.default_rng(5)).sample(n)
    assert single[0::2] == got[:10] and single[1::2] == got[10:]
