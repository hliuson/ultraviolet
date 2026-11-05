"""Replication flags in rl_grpo.py: GRPO std / length normalisation, PSR / NSR /
W-REINFORCE advantages, entropy bonus, Clip-Cov, replication eval keys, repl_eval pass@k / maj@k. CPU only."""
import os
import sys
import types
from math import comb

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import rl_grpo  # noqa: E402
import repl_eval  # noqa: E402


def test_group_adv_defaults_unchanged():
    r = [1, 0, 0, 1, 0, 0, 0, 0]
    np.testing.assert_allclose(rl_grpo.group_adv(r, "grpo"), np.array(r) - 0.25)
    np.testing.assert_allclose(rl_grpo.group_adv(r, "grpo", **rl_grpo.adv_args(types.SimpleNamespace())),
                               np.array(r) - 0.25)
    np.testing.assert_allclose(rl_grpo.group_adv(r, "maxrl"), (np.array(r) - 0.25) / 0.25)


def test_group_adv_grpo_std():
    r = np.array([1, 0, 0, 1, 0, 0, 0, 0], float)
    a = rl_grpo.group_adv(r, "grpo", std_norm=True)
    np.testing.assert_allclose(a, (r - r.mean()) / (r.std(ddof=1) + 1e-6))
    assert np.all(rl_grpo.group_adv(np.zeros(8), "grpo", std_norm=True) == 0)
    assert np.all(rl_grpo.group_adv(np.ones(8), "grpo", std_norm=True) == 0)


def test_group_adv_psr_nsr_wreinforce():
    r = [1, 0, 1, 0]
    np.testing.assert_array_equal(rl_grpo.group_adv(r, "psr"), [1, 0, 1, 0])
    np.testing.assert_array_equal(rl_grpo.group_adv(r, "nsr"), [0, -1, 0, -1])
    np.testing.assert_allclose(rl_grpo.group_adv(r, "wreinforce", lam=0.1), [0.1, -1, 0.1, -1])
    # no baseline: all-wrong groups still get NSR signal, all-correct groups PSR signal
    np.testing.assert_array_equal(rl_grpo.group_adv([0, 0], "nsr"), [-1, -1])
    np.testing.assert_array_equal(rl_grpo.group_adv([1, 1], "psr"), [1, 1])


def _pg_inputs(T=7, d=8, V=11, seed=0):
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(1, T, d, generator=g)
    w = torch.randn(V, d, generator=g)
    y = torch.randint(0, V, (1, T), generator=g)
    y[0, :2] = -1
    adv = torch.full((1, T), 0.5)
    return h, w, y, adv


def test_pg_loss_entropy_term():
    sc = (23.0, 5.0, 7.5)
    h, w, y, adv = _pg_inputs()
    l0, _ = rl_grpo.pg_loss(h, w, y, adv, None, sc, 2.0)
    l1, _ = rl_grpo.pg_loss(h, w, y, adv, None, sc, 2.0, ent_coef=0.0)
    assert torch.equal(l0, l1)  # off -> identical
    l2, _ = rl_grpo.pg_loss(h, w, y, adv, None, sc, 2.0, ent_coef=0.01)
    z = rl_grpo.softcap_logits(h.reshape(-1, h.size(-1)), w, sc)
    lsm = torch.log_softmax(z, -1)
    H = (-(lsm.exp() * lsm).sum(-1) * (y.reshape(-1) != -1).float()).sum()
    torch.testing.assert_close(l2, l0 - 0.01 * H)


class _Raw:
    def __init__(self, w):
        self.embed = types.SimpleNamespace(weight=w)
        self.cfg = types.SimpleNamespace(softcap=(23.0, 5.0, 7.5))


def test_clip_cov_detaches_selected_tokens():
    torch.manual_seed(0)
    T, d, V = 64, 8, 13
    w = torch.randn(V, d) * 3
    x = torch.randn(2, T, d)
    y = torch.randint(0, V, (2, T))
    y[:, :3] = -1
    A = torch.zeros(2, T)
    A[0], A[1] = 0.9, -0.9
    fake = types.SimpleNamespace(raw=_Raw(w), trunk=lambda x_: x_, _cc_calls=0,
                                 a=types.SimpleNamespace(temperature=1.0, clip_cov=0.05, cov_lb=0.0, cov_ub=1e9,
                                                         seed=0))
    A0 = A.clone()
    st = rl_grpo.Trainer.clip_cov(fake, [(x, y, A)])
    N = int((y != -1).sum())
    assert st["clipcov_tokens"] == N
    assert st["clipcov_detached"] == min(int(0.05 * N), st["clipcov_candidates"]) > 0
    changed = (A != A0)
    assert int(changed.sum()) == st["clipcov_detached"]
    assert torch.all(A[changed] == 0)
    # the detached tokens have positive covariance (cov_lb = 0)
    lp = rl_grpo.token_logp(x, w, y, (23.0, 5.0, 7.5)).reshape(y.shape)
    m = y != -1
    cov = (lp - lp[m].mean()) * (A0 - A0[m].mean())
    assert torch.all(cov[changed] > 0)
    # ratio 0 -> untouched
    A2 = A0.clone()
    fake.a.clip_cov = 1e-9
    st2 = rl_grpo.Trainer.clip_cov(fake, [(x, y, A2)])
    assert st2["clipcov_detached"] == 0 and torch.equal(A2, A0)


def test_pass_at_k_unbiased():
    n, c = np.array([16, 16, 16, 16]), np.array([0, 1, 5, 16])
    for k in (1, 2, 8, 16):
        np.testing.assert_allclose(repl_eval.pass_at_k(n, c, k), [1 - comb(16 - ci, k) / comb(16, k) for ci in c])


def test_maj_at_k_and_summary():
    ok = np.array([[1, 0, 1, 0], [0, 0, 0, 0]], bool)
    pred = np.array([[5, 7, 5, 7], [3, 3, 0, 0]], np.int64)
    np.testing.assert_allclose(repl_eval.maj_at_k(ok, pred, 2), [0.5 * (1 + 1), 0.0])  # ties -> first seen
    np.testing.assert_allclose(repl_eval.maj_at_k(ok, pred, 4), [1.0, 0.0])
    s = repl_eval.summary(ok, np.array([[10, 20, 30, 40], [5, 5, 5, 5]]), np.ones_like(ok), pred)
    assert s["pass@1"] == 0.25 and s["pass@4"] == 0.5 and s["maj@4"] == 0.5
    assert s["mean_len_correct"] == 20 and s["frac_unsolved"] == 0.5


def test_repl_keys_and_parse():
    assert rl_grpo.parse_repl_evals("a:/x.jsonl:25:4,b:/y.jsonl:100:16") == [("a", "/x.jsonl", 25, 4),
                                                                          ("b", "/y.jsonl", 100, 16)]
    k = rl_grpo.repl_keys({"math500": {"pass@1": 0.1, "pass@2": 0.15, "pass@4": 0.2, "maj@4": 0.12, "P": 500}})
    assert k["sum/math500_pass1"] == 0.1 and k["sum/math500_pass4"] == 0.2 and k["sum/math500_maj4"] == 0.12
    assert k["diag/repl/math500/pass4"] == 0.2


def test_sample_top_p():
    from ultraviolet.model import sample
    torch.manual_seed(0)
    p = torch.tensor([0.5, 0.3, 0.15, 0.05])
    logits = p.log().repeat(20000, 1)
    full = torch.bincount(sample(logits, 1.0), minlength=4).float() / 20000
    assert torch.allclose(full, p, atol=0.02)  # top_p = 1 -> plain sampling (unchanged)
    s = sample(logits, 1.0, top_p=0.7)  # 0.5 + 0.3 = 0.8 >= 0.7 -> tokens {0, 1}
    c = torch.bincount(s, minlength=4).float() / 20000
    assert c[2] == 0 and c[3] == 0 and abs(c[0] - 0.625) < 0.02
    assert torch.all(sample(logits, 1.0, top_p=0.4) == 0)  # first token always kept
    s = sample(logits, 0.6, top_p=0.95)  # temperature first, then the nucleus of the tempered distribution
    pt = torch.softmax(p.log() / 0.6, -1)
    keep = (pt.cumsum(0) - pt) < 0.95
    c = torch.bincount(s, minlength=4).float() / 20000
    assert torch.allclose(c, torch.where(keep, pt, 0.0) / pt[keep].sum(), atol=0.02)
