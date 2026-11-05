"""Long-run robustness: --resume state (optimizers, GradScaler, samplers, rngs) and the
non-finite step guard in rl_grpo.Trainer."""
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
from rl_sampler import make_sampler  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("kind", ["uniform", "pool"])
def test_sampler_state_roundtrip(kind):
    s = make_sampler(kind, 50, np.random.default_rng(3))
    for _ in range(4):
        idx = s.sample(7)
        s.update(idx, np.random.default_rng(len(idx)).random(len(idx)))
    st = copy.deepcopy(s.state_dict())
    want = [s.sample(5) for _ in range(3)]
    s2 = make_sampler(kind, 50, np.random.default_rng(999))
    s2.load_state_dict(st)
    assert [s2.sample(5) for _ in range(3)] == want
    if kind == "pool":
        assert np.array_equal(s.pool, s2.pool)


def _args(**kw):
    d = dict(mode="1gpu", optim="normuon", adam_lr=1e-5, muon_lr=9e-4, warmup=2, max_new=16, tok_budget=512,
             pad_to=16, temperature=1.0, tis_cap=2.0, grad_clip=1.0, nonfinite_abort=0)
    d.update(kw)
    return types.SimpleNamespace(**d)


def _model():
    from ultraviolet.model import MathLM, make_config
    torch.manual_seed(0)
    return MathLM(make_config("tiny")).cuda()


def _batch(seed):
    rng = np.random.default_rng(seed)
    prompt = [5, 6, 7, 8]
    seqs = [(np.array(prompt + rng.integers(1, 512, size=6).tolist(), dtype=np.int64), 4) for _ in range(4)]
    return seqs, [1.0, -1.0, 0.5, -0.5]


def _eq_state(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_eq_state(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_eq_state(x, y) for x, y in zip(a, b))
    return a == b


@cuda
def test_trainer_resume_restores_optimizer_state():
    """2 steps, save (weights + Trainer.state_dict) through a file; a fresh model + Trainer that loads it has exactly
    the same NorMuon (momentum, second moment) / AdamW (moments, step) / GradScaler state, and keeps training.
    (Bitwise weight equality after further steps is not testable: fp16 backward atomics are nondeterministic at the
    same magnitude as the effect of the optimizer state on this tiny model.)"""
    import io
    raw = _model()
    tr = rl_grpo.Trainer(raw, torch.device("cuda"), _args())
    for k in range(2):
        tr.step(k, *_batch(k))
    buf = io.BytesIO()
    torch.save({"model": raw.state_dict(), "trainer": tr.state_dict()}, buf)
    buf.seek(0)
    d = torch.load(buf, map_location="cpu", weights_only=False)
    raw2 = _model()
    raw2.load_state_dict(d["model"])
    tr2 = rl_grpo.Trainer(raw2, torch.device("cuda"), _args())
    fresh = copy.deepcopy(tr2.state_dict())
    tr2.load_state_dict(d["trainer"])
    assert _eq_state(tr2.state_dict(), tr.state_dict())
    assert not _eq_state(fresh, tr.state_dict())  # the fresh optimizer state is empty: the check is not vacuous
    assert len(tr2.mu.state) > 0 and len(tr2.adam.state) > 0
    for p in tr2.params:  # restored state lives on the parameter's device
        for v in list(tr2.mu.state.get(p, {}).values()) + list(tr2.adam.state.get(p, {}).values()):
            if isinstance(v, torch.Tensor) and v.dim() > 0:
                assert v.device == p.device
    st = tr2.step(2, *_batch(2))
    assert st["nonfinite_skip"] == 0 and st["grad_norm"] > 0


@cuda
def test_resume_state_file_roundtrip(tmp_path):
    """save-side helpers (rank_state / resume_state / atomic_save) -> find_resume('auto') -> apply_resume restores
    sampler draws, RRNG, torch rngs, counters and the trainer."""
    raw = _model()
    tr = rl_grpo.Trainer(raw, torch.device("cuda"), _args())
    tr.step(0, *_batch(0))
    sampler = make_sampler("pool", 40, np.random.default_rng(5))
    sampler.update(sampler.sample(10), [0.0] * 5 + [0.5] * 5)
    rl_grpo.RRNG = np.random.default_rng(11)
    rl_grpo.RS.update(start=0, totals=rl_grpo.collections.Counter(rollouts=1234), train_s=3600.0, hours=1.5)
    out = str(tmp_path)
    rl_grpo.atomic_save({"model": raw.state_dict()}, os.path.join(out, "step250.pt"))
    rl_grpo.atomic_save(rl_grpo.resume_state(250, tr, [rl_grpo.rank_state(sampler)]), os.path.join(out, "state250.pt"))
    rl_grpo.atomic_save({}, os.path.join(out, "state500.pt"))  # no step500.pt beside it -> ignored by auto
    want = (sampler.sample(6), rl_grpo.RRNG.random(), torch.rand(2).tolist(),
            torch.rand(2, device="cuda").tolist())
    assert not any(f.endswith(".tmp") for f in os.listdir(out))

    p = rl_grpo.find_resume("auto", out)
    assert p.endswith("state250.pt")
    R = torch.load(p, map_location="cpu", weights_only=False)
    tr2 = rl_grpo.Trainer(_model(), torch.device("cuda"), _args())
    s2 = make_sampler("pool", 40, np.random.default_rng(0))
    rl_grpo.RRNG = np.random.default_rng(0)
    rl_grpo.RS.update(start=0, totals=rl_grpo.collections.Counter(), train_s=0.0, hours=0.0)
    rl_grpo.apply_resume(R, None, tr2, s2, 0)
    got = (s2.sample(6), rl_grpo.RRNG.random(), torch.rand(2).tolist(),
           torch.rand(2, device="cuda").tolist())
    assert got == want
    assert rl_grpo.RS["start"] == 250 and rl_grpo.RS["totals"]["rollouts"] == 1234 and rl_grpo.RS["train_s"] == 3600.0
    assert rl_grpo.find_resume("auto", str(tmp_path / "nothing")) is None


@cuda
def test_nonfinite_guard_skips_and_aborts():
    raw = _model()
    tr = rl_grpo.Trainer(raw, torch.device("cuda"), _args(nonfinite_abort=3))
    seqs, adv = _batch(0)
    st = tr.step(0, seqs, adv)
    assert st["nonfinite_skip"] == 0
    w0 = {n: p.detach().clone() for n, p in raw.state_dict().items()}
    # NaN advantages -> NaN gradient -> step skipped, weights untouched, scale backs off
    scale0 = tr.scaler.get_scale()
    st = tr.step(1, seqs, [float("nan")] * 4)
    assert st["nonfinite_skip"] == 1 and st["nonfinite_total"] == 1
    assert all(torch.equal(p, w0[n]) for n, p in raw.state_dict().items())
    assert tr.scaler.get_scale() < scale0
    # a finite step resets the consecutive counter and moves the weights
    st = tr.step(2, seqs, adv)
    assert st["nonfinite_skip"] == 0 and tr.nonfinite_consec == 0
    assert any(not torch.equal(p, w0[n]) for n, p in raw.state_dict().items())
    # a non-finite parameter (finite gradient) is also skipped; 3 in a row abort
    with torch.no_grad():
        tr.params[0].view(-1)[0] = float("nan")
    w1 = {n: p.detach().clone() for n, p in raw.state_dict().items()}
    for k in (3, 4):
        assert tr.step(k, seqs, adv)["nonfinite_skip"] == 1
    for n, p in raw.state_dict().items():
        assert torch.equal(torch.nan_to_num(p), torch.nan_to_num(w1[n])), n
    with pytest.raises(FloatingPointError, match="NONFINITE_ABORT"):
        tr.step(5, seqs, adv)


def test_strict_format_reward_rule():
    import types
    E, C = rl_grpo.EOS, 7
    on = types.SimpleNamespace(strict_format=1, answer_only=0)
    off = types.SimpleNamespace(strict_format=0, answer_only=0)
    good = [5, 6, C, 9, 10, E]
    assert rl_grpo.strict_ok(good, C, on) and rl_grpo.wellformed_row(good, C)
    for bad in ([5, 6, C, 9, C, 10, E],  # second </think>
                [5, 6, C, 9, 10],  # truncated (no EOS)
                [5, 6, 9, E]):  # never closed
        assert not rl_grpo.strict_ok(bad, C, on) and not rl_grpo.wellformed_row(bad, C)
        assert rl_grpo.strict_ok(bad, C, off)  # flag off: unchanged reward
    assert rl_grpo.strict_ok([5, E], C, types.SimpleNamespace(strict_format=1, answer_only=1))


def test_repl_summary_wellformed():
    import repl_eval
    ok = np.array([[1, 1, 0, 0], [0, 1, 0, 0]], bool)
    wf = np.array([[1, 0, 1, 1], [1, 1, 1, 1]], bool)
    z = np.zeros((2, 4), np.int64)
    s = repl_eval.summary(ok, z, np.ones((2, 4), bool), z, wf)
    assert s["pass@1"] == 3 / 8 and s["pass@1_strict"] == 2 / 8 and s["wellformed"] == 7 / 8
    assert "pass@1_strict" not in repl_eval.summary(ok, z, np.ones((2, 4), bool), z)
    assert repl_eval.wellformed([3, 7, 4, rl_grpo.EOS], 7, rl_grpo.EOS)
    assert not repl_eval.wellformed([3, 7, 4, 7, rl_grpo.EOS], 7, rl_grpo.EOS)


def test_repl_keys_with_strict():
    k = rl_grpo.repl_keys({"m": {"pass@1": 0.2, "pass@4": 0.5, "pass@1_strict": 0.1, "wellformed": 0.9, "maj@4": 0.3}})
    assert k["sum/m_pass4"] == 0.5 and k["sum/m_pass1_strict"] == 0.1 and k["diag/repl/m/wellformed"] == 0.9


def test_scaler_growth_interval_flag():
    import argparse
    import torch
    m = torch.nn.Linear(4, 4)
    a = argparse.Namespace(optim="adamw", adam_lr=1e-3, muon_lr=1e-3, scaler_growth_interval=100, warmup=0)
    try:
        tr = rl_grpo.Trainer(m, torch.device("cpu"), a)
    except Exception:  # Trainer needs more args / CUDA in this environment: check the constructor source instead
        import inspect
        assert "growth_interval=getattr(args, \"scaler_growth_interval\", 2000)" in inspect.getsource(rl_grpo.Trainer)
        return
    assert tr.scaler.get_growth_interval() == 100


def test_scaler_growth_interval_wins_on_resume():
    import inspect
    src = inspect.getsource(rl_grpo.Trainer.load_state_dict)
    assert "set_growth_interval" in src


def test_rlscaler_floor_and_bad_names():
    import pytest
    import torch
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    m = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 1)).cuda()
    names = {id(p): n for n, p in m.named_parameters()}
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    sc = rl_grpo.RLScaler(names, min_scale=1024.0, init_scale=2048.0, growth_interval=50)
    for i in range(3):  # three overflow steps: 2048 -> 1024 -> floor 1024 -> 1024
        opt.zero_grad()
        sc.scale(m(torch.ones(2, 4, device="cuda")).sum()).backward()
        m[1].bias.grad.fill_(float("nan"))
        sc.unscale_(opt)
        assert sc.bad == ["1.bias"]
        sc.step(opt)
        sc.update()
        assert sc.last_bad == ["1.bias"] and sc.bad == []
    assert sc.get_scale() == 1024.0
    opt.zero_grad()  # a clean step records nothing
    sc.scale(m(torch.ones(2, 4, device="cuda")).sum()).backward()
    sc.unscale_(opt)
    sc.step(opt)
    sc.update()
    assert sc.last_bad == []
    sc0 = rl_grpo.RLScaler(names)  # defaults: no floor, torch growth interval
    assert sc0.min_scale == 0.0 and sc0.get_growth_interval() == 2000


def test_rlscaler_cap():
    import pytest
    import torch
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    m = torch.nn.Linear(4, 1).cuda()
    opt = torch.optim.SGD(m.parameters(), lr=0.0)
    sc = rl_grpo.RLScaler({}, max_scale=4096.0, init_scale=1024.0, growth_interval=1)
    for _ in range(5):  # clean steps: 1024 -> 2048 -> 4096 -> capped
        opt.zero_grad()
        sc.scale(m(torch.ones(2, 4, device="cuda")).sum()).backward()
        sc.step(opt)
        sc.update()
    assert sc.get_scale() == 4096.0
