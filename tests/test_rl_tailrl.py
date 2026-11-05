"""rl_grpo --adv tailrl: TailRL tail-likelihood weights and advantages (arXiv 2609.02987 Eqs. 26-27, 30, 107, Cor. 14)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import rl_grpo as m  # noqa: E402


def test_weights_closed_form():
    # N=2, rewards 0.2, 0.8: w = 0.2/2 = 0.1 and 0.1 + 0.6/1 = 0.7
    assert np.allclose(m.tail_weights([0.8, 0.2]), [0.7, 0.1])
    # sum of weights = max reward (Eq. 107); ties share the same weight; zero reward -> weight 0
    r = np.array([0.0, 3.0, 1.0, 1.0, 42.0, 0.0, 1.0, 2.5])
    w = m.tail_weights(r)
    assert abs(w.sum() - r.max()) < 1e-12
    assert w[2] == w[3] == w[6] and w[0] == w[5] == 0.0
    assert np.all(np.diff(w[np.argsort(r)]) >= -1e-15)  # monotone in the reward


def test_weights_match_integral():
    rng = np.random.default_rng(0)
    r = rng.exponential(size=8)
    taus = np.linspace(0, r.max(), 200001)[:-1] + r.max() / 400000
    cnt = (r[None, :] > taus[:, None]).sum(1)
    want = [np.sum((taus < ri) / cnt) * (r.max() / 200000) for ri in r]
    assert np.allclose(m.tail_weights(r), want, atol=1e-4)


def test_advantage_centered_and_binary_equals_maxrl():
    a = m.group_adv([0.8, 0.2], "tailrl")
    assert np.allclose(a, [0.6, -0.6])  # N (w - mean w) = 2 * (+-0.3)
    for r in ([0, 0, 1, 1], [1, 0, 0, 0, 0, 0, 0, 0], [1] * 8):
        assert np.allclose(m.group_adv(r, "tailrl"), m.group_adv(r, "maxrl")), r
    assert np.all(m.group_adv([0] * 8, "tailrl") == 0)
    assert abs(m.group_adv([0, 1, 1, 40, 0, 1, 2, 1], "tailrl").sum()) < 1e-9


def test_tailrl_is_graded_safe():
    class A:
        adv, resample_zero, sampler = "tailrl", 0, "uniform"
    assert m.binary_only(A()) == []
