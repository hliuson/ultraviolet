"""--ppo-schedule (multiple optimizer steps per batch with the PPO clip): schedule parsing, group-aligned minibatch
cuts, the clipped-surrogate gradient of pg_loss (CPU), and the Trainer path against the one-step path (CUDA)."""
import copy
import os
import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import rl_grpo  # noqa: E402
from test_rl_stream import _models, _tf_logp, cuda  # noqa: E402


def test_parse_ppo_schedule_and_rung():
    s = rl_grpo.parse_ppo_schedule("2x4:50, 2x2:150,2x1")
    assert s == [(2, 4, 50), (2, 2, 150), (2, 1, None)]
    assert [rl_grpo.ppo_rung(s, k) for k in (0, 49, 50, 149, 150, 999)] == [(2, 4)] * 2 + [(2, 2)] * 2 + [(2, 1)] * 2
    assert rl_grpo.parse_ppo_schedule("") == []
    assert rl_grpo.ppo_rung(rl_grpo.parse_ppo_schedule("4x1:10"), 20) == (4, 1)  # past the last bound: last rung
    for bad in ("2x1:50,2x1:40", "2x1,1x1", "0x1"):
        with pytest.raises(AssertionError):
            rl_grpo.parse_ppo_schedule(bad)


def _seqs(sizes):
    out = []
    for g, n in enumerate(sizes):
        out += [(np.array([100 + g, 7, 8, 9, 10], dtype=np.int64), 3)] * n
    return out


def test_ppo_cuts_whole_groups():
    assert rl_grpo.ppo_cuts(_seqs([16] * 4), 2) == [0, 32, 64]
    assert rl_grpo.ppo_cuts(_seqs([16] * 4), 4) == [0, 16, 32, 48, 64]
    assert rl_grpo.ppo_cuts(_seqs([8, 24, 16]), 2) == [0, 32, 48]  # nearest group start to 24 is 32
    assert rl_grpo.ppo_cuts(_seqs([16]), 2) == [0, 8, 16]  # fewer groups than minibatches: plain split
    assert rl_grpo.ppo_cuts([], 2) == [0, 0, 0]
    assert len(rl_grpo.ppo_cuts(_seqs([40, 4, 4]), 3)) == 4  # always M minibatches (may be empty)


def test_pg_loss_ppo_gradient_matches_clipped_surrogate():
    torch.manual_seed(0)
    N, D, V, sc, cap, lo, hi = 64, 16, 32, (30.0, 0.0, 10.0), 2.0, 0.2, 0.28
    h0, w = torch.randn(N, D), torch.randn(V, D)
    y = torch.randint(0, V, (N,))
    y[-5:] = -1
    adv = torch.randn(N)
    with torch.no_grad():
        lp = -torch.nn.functional.cross_entropy(rl_grpo.softcap_logits(h0, w, sc), y.clamp_min(0), reduction="none")
    pc = lp + 0.4 * torch.randn(N)  # pi_old away from pi: some tokens clip on each side
    oc = pc + 0.5 * torch.randn(N)  # behaviour log-probs: some TIS weights hit the cap
    h = h0.clone().requires_grad_(True)
    loss, st = rl_grpo.pg_loss(h, w, y, adv, oc, sc, cap, ppo_lp=pc, clip=(lo, hi), chunk=16)
    loss.backward()
    g = h.grad.clone()
    h2 = h0.clone().requires_grad_(True)
    lp2 = -torch.nn.functional.cross_entropy(rl_grpo.softcap_logits(h2, w, sc), y.clamp_min(0), reduction="none")
    r = torch.exp(lp2 - pc)
    tis = torch.clamp(torch.exp(pc - oc), max=cap)
    m = (y != -1).float()
    ref = -(torch.minimum(r * adv, torch.clamp(r, 1 - lo, 1 + hi) * adv) * tis * m).sum()
    ref.backward()
    torch.testing.assert_close(g, h2.grad, rtol=1e-4, atol=1e-5)
    rd = r.detach()
    assert st[2] == m.sum() and st[0] == pytest.approx(float((tis * m).sum()), rel=1e-5)
    assert st[3] == float((((adv < 0) & (rd < 1 - lo)).float() * m).sum()) > 0
    assert st[4] == float((((adv > 0) & (rd > 1 + hi)).float() * m).sum()) > 0
    assert st[1] == float(((tis >= cap).float() * m).sum()) > 0
    assert float(st[5]) == pytest.approx(float(((pc - lp) * m).sum()), rel=1e-4, abs=1e-4)


def _args(**kw):
    d = dict(mode="1gpu", optim="adam", adam_lr=1e-4, muon_lr=9e-4, warmup=0, max_new=16, tok_budget=512, pad_to=16,
             temperature=1.0, tis_cap=2.0, grad_clip=1.0, ppo_clip_low=0.2, ppo_clip_high=0.28, ppo_schedule="",
             grad_probe_every=0)
    d.update(kw)
    return types.SimpleNamespace(**d)


def _batch(raw, n_groups=4, G=4):
    rng = np.random.default_rng(1)
    seqs, advs, blps = [], [], []
    for g in range(n_groups):
        prompt = [5 + g, 6, 7, 8]
        for j in range(G):
            row = rng.integers(1, 512, size=6).tolist()
            seqs.append((np.array(prompt + row, dtype=np.int64), len(prompt)))
            blps.append(_tf_logp(raw, prompt, row, 1.0))
            advs.append(float(rng.normal()) if j else 0.0)
    return seqs, advs, blps


def _spy(tr):
    """Record the unscaled flat gradient that reaches each optimizer step (Adam's first step is ~lr * sign(g), so
    comparing parameter deltas would amplify fp16 noise on near-zero gradients)."""
    out, orig = [], tr._apply

    def f(k, st):
        out.append(torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten().float()
                              for p in tr.params]) / tr.scaler.get_scale())
        return orig(k, st)
    tr._apply = f
    return out


def _rel(a, b):
    return float((a - b).norm() / b.norm())


@cuda
def test_ppo_1x1_equals_one_step_path():
    raw, _ = _models()
    seqs, advs, blps = _batch(raw)
    t0 = rl_grpo.Trainer(copy.deepcopy(raw), torch.device("cuda"), _args())
    t1 = rl_grpo.Trainer(copy.deepcopy(raw), torch.device("cuda"), _args(ppo_schedule="1x1"))
    g0, g1 = _spy(t0), _spy(t1)
    st0, st1 = t0.step(0, seqs, advs, blps), t1.step(0, seqs, advs, blps)
    assert len(g0) == len(g1) == 1 and g0[0].norm() > 0 and _rel(g1[0], g0[0]) < 1e-3
    assert st1["ppo_steps"] == 1 and st1["ppo_clip_low_frac"] == 0 and st1["ppo_clip_high_frac"] == 0
    assert st1["train_seqs"] == st0["train_seqs"] and st1["tis_mean"] == pytest.approx(st0["tis_mean"], abs=1e-3)


@cuda
def test_ppo_2x1_first_minibatch_is_a_one_step_update_on_its_half():
    """2 x 1: the first step's gradient equals a one-step update's on the first two groups alone (same per-minibatch
    normaliser); the second step's is taken after the first update (differs from the one-step gradient on its half)."""
    raw, _ = _models()
    seqs, advs, blps = _batch(raw)
    half = 8  # 2 groups of 4
    ta = rl_grpo.Trainer(copy.deepcopy(raw), torch.device("cuda"), _args())
    tc = rl_grpo.Trainer(copy.deepcopy(raw), torch.device("cuda"), _args())
    tb = rl_grpo.Trainer(copy.deepcopy(raw), torch.device("cuda"), _args(ppo_schedule="2x1"))
    ga, gc, gb = _spy(ta), _spy(tc), _spy(tb)
    ta.step(0, seqs[:half], advs[:half], blps[:half])
    tc.step(0, seqs[half:], advs[half:], blps[half:])
    st = tb.step(0, seqs, advs, blps)
    assert st["ppo_steps"] == 2 and st["ppo_minibatches"] == 2 and st["ppo_epochs"] == 1
    assert _rel(gb[0], ga[0]) < 1e-3
    assert 1e-3 < _rel(gb[1], gc[0]) < 0.5  # moved by one lr=1e-4 step, still close


@cuda
def test_ppo_epochs_take_m_times_e_steps_and_move_off_policy():
    raw, _ = _models()
    seqs, advs, blps = _batch(raw)
    tr = rl_grpo.Trainer(copy.deepcopy(raw), torch.device("cuda"), _args(ppo_schedule="2x3:1,1x1"))
    st = tr.step(0, seqs, advs, blps)
    assert st["ppo_steps"] == 6 and st["ppo_epochs"] == 3 and st["ppo_kl_last"] != 0
    assert np.isfinite(st["grad_norm"]) and st["nonfinite_skip"] == 0
    st2 = tr.step(1, seqs, advs, None)  # rung 2 (1 x 1), no behaviour log-probs: tis = 1
    assert st2["ppo_steps"] == 1 and st2["tis_mean"] == pytest.approx(1.0)


def test_probe_stats_estimates_true_gradient_and_noise():
    """probe_stats on simulated per-group gradients g_i = G + noise (cov Sigma): averaged over draws, g2 -> |G|^2 and
    trs -> tr(Sigma); cos0 of a gradient with itself is 1."""
    rng = np.random.default_rng(0)
    D, na, nb, reps = 50, 16, 16, 4000
    G = torch.tensor(rng.normal(size=D), dtype=torch.float64) * 0.3
    sd = torch.tensor(rng.uniform(0.5, 2.0, size=D), dtype=torch.float64)
    g2, trs = [], []
    for _ in range(reps):
        ga = G + (torch.randn(na, D, dtype=torch.float64) * sd).mean(0)
        gb = G + (torch.randn(nb, D, dtype=torch.float64) * sd).mean(0)
        st, g = rl_grpo.probe_stats(ga, gb, na, nb)
        g2.append(st["g2"]); trs.append(st["trs"])
    assert np.mean(g2) == pytest.approx(float(G @ G), rel=0.1)
    assert np.mean(trs) == pytest.approx(float((sd ** 2).sum()), rel=0.05)
    st, _ = rl_grpo.probe_stats(ga, gb, na, nb, g0=ga + gb)
    assert st["cos0"] == pytest.approx(1.0) and -1 <= st["cos"] <= 1


@cuda
@pytest.mark.parametrize("sched,E", [("1x1", 1), ("2x1", 1), ("2x3", 3)])
def test_grad_probe_logs_and_leaves_the_update_unchanged(sched, E):
    """--grad-probe-every: E + 1 probes (before each pass and after the last); the update's own step gradients are the
    same as without the probe; probe 0 is at the update's start (cos0 = 1) over all 4 groups."""
    raw, _ = _models()
    seqs, advs, blps = _batch(raw)
    t0 = rl_grpo.Trainer(copy.deepcopy(raw), torch.device("cuda"), _args(ppo_schedule=sched))
    t1 = rl_grpo.Trainer(copy.deepcopy(raw), torch.device("cuda"), _args(ppo_schedule=sched, grad_probe_every=5))
    g0, g1 = _spy(t0), _spy(t1)
    st0, st1 = t0.step(0, seqs, advs, blps), t1.step(0, seqs, advs, blps)
    assert len(g0) == len(g1) and all(_rel(b, a) < 1e-3 for a, b in zip(g0, g1))
    assert not any(k.startswith("gp_") for k in st0)
    assert all(f"gp_cos_{i}" in st1 and f"gp_trs_{i}" in st1 for i in range(E + 1)) and f"gp_cos_{E + 1}" not in st1
    assert st1["gp_cos0_0"] == pytest.approx(1.0, abs=1e-4) and st1["gp_groups"] == 4
    assert all(np.isfinite(st1[f"gp_g2_{i}"]) and -1 <= st1[f"gp_cos_{i}"] <= 1 for i in range(E + 1))
    assert not any(k.startswith("gp_") for k in t1.step(1, seqs, advs, blps))  # off-probe update (1 % 5 != 0)
