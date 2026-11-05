import copy
import os
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import rl_grpo  # noqa: E402


def test_group_adv_grpo_and_maxrl():
    r = np.array([1, 0, 0, 1, 0, 0, 0, 0], float)
    np.testing.assert_allclose(rl_grpo.group_adv(r, "grpo"), r - 0.25)
    a = rl_grpo.group_adv(r, "maxrl")
    np.testing.assert_allclose(a, (r - 0.25) / 0.25)
    N, K = len(r), r.sum()
    np.testing.assert_allclose(a / N, r / K - 1 / N)  # paper eq. 10 weights (r_j/K - 1/N)
    assert not rl_grpo.group_adv(np.zeros(4), "maxrl").any()
    assert not rl_grpo.group_adv(np.ones(4), "maxrl").any()
    with pytest.raises(ValueError):
        rl_grpo.group_adv(r, "nope")


def test_parse_pool_G():
    g = rl_grpo.parse_pool_G("low=32,zero=16", 8)
    assert g == {"unseen": 8, "zero": 16, "low": 32, "mid": 8, "high": 8, "solved": 8}
    assert rl_grpo.parse_pool_G("", 8)["low"] == 8
    with pytest.raises(AssertionError):
        rl_grpo.parse_pool_G("bogus=3", 8)


def test_group_stats():
    st = rl_grpo.group_stats([np.array([0, 1.0]), np.array([0, 0.0]), np.array([1, 1.0])])
    assert st["reward"] == pytest.approx(0.5)
    assert st["frac_groups_nonzero_adv"] == pytest.approx(1 / 3)
    assert st["frac_groups_all_correct"] == pytest.approx(1 / 3)


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _models():
    from ultraviolet.model import MathLM, make_config
    torch.manual_seed(0)
    raw = MathLM(make_config("tiny")).cuda()
    with torch.no_grad():
        for n, p in raw.named_parameters():
            if n.endswith(("wo", "w_down", "ve_gate", "attn_gate", "xsa_alpha")):
                p.normal_(0, 0.3)
    gen = copy.deepcopy(raw).half()
    gen.fuse_for_inference(torch.float16)
    return raw, gen


def _tf_logp(m, prompt, row, T):
    x = torch.tensor([prompt + row], device="cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        h, _ = m.trunk(x)
        lg = m.logits(h)[0].float() / T
    lp = torch.log_softmax(lg, -1)
    s = len(prompt)
    idx = torch.arange(s - 1, s - 1 + len(row), device="cuda")
    return lp[idx, torch.tensor(row, device="cuda")].cpu().numpy()


def _run(gen, prompts, n_slots=4, max_new=24, T=1.0, eos=0):
    from rl_stream import StreamGen
    sg = StreamGen(gen, n_slots, max(len(p) for p in prompts), max_new, T, eos, check_every=4, compile=False)
    for i, p in enumerate(prompts):
        sg.add(p, i)
    out = {}
    while sg.n_outstanding():
        for meta, row, lp, v0 in sg.chunk():
            assert meta not in out
            out[meta] = (row, lp, v0)
    return sg, out


@cuda
def test_stream_all_requests_and_behaviour_logp():
    raw, gen = _models()
    rng = np.random.default_rng(0)
    prompts = [rng.integers(1, 512, size=rng.integers(3, 9)).tolist() for _ in range(11)]
    sg, out = _run(gen, prompts)
    assert sorted(out) == list(range(11))
    for i, (row, lp, v0) in out.items():
        assert 1 <= len(row) <= 24 and len(lp) == len(row) and v0 == 0
        if 0 in row:
            assert row.index(0) == len(row) - 1
        ref = _tf_logp(gen, prompts[i], row, 1.0)
        assert np.abs(ref - lp).mean() < 0.05, (i, np.abs(ref - lp).max())


@cuda
@pytest.mark.parametrize("use_graph", [True, False])
def test_stream_logp_across_graph_buckets(use_graph):
    """Rows cross several global-context buckets (bucket=8): the recorded log-probs must come from the graph that
    was replayed (regression: DecodeRunner.out_logits points at the last *captured* graph's buffer)."""
    from rl_stream import StreamGen
    raw, gen = _models()
    rng = np.random.default_rng(3)
    prompts = [rng.integers(1, 512, size=rng.integers(3, 7)).tolist() for _ in range(9)]
    sg = StreamGen(gen, 3, 6, 40, 1.0, -5, check_every=4, compile=False, use_graph=use_graph, bucket=8)
    for i, p in enumerate(prompts):
        sg.add(p, i)
    out = {}
    while sg.n_outstanding():
        for meta, row, lp, v0 in sg.chunk():
            out[meta] = (row, lp)
    assert sorted(out) == list(range(9))
    for i, (row, lp) in out.items():
        assert len(row) == 40
        ref = _tf_logp(gen, prompts[i], row, 1.0)
        assert np.abs(ref - lp).max() < 0.1, (i, np.abs(ref - lp).max())


@cuda
def test_sync_weights_inplace_updates_graph_weights():
    from rl_stream import StreamGen, sync_weights_inplace
    raw, gen = _models()
    prompts = [[5, 6, 7, 8]] * 4
    sg = StreamGen(gen, 4, 4, 16, 1.0, 0, check_every=4, compile=False, use_graph=True)
    for i, p in enumerate(prompts):
        sg.add(p, ("a", i))
    while sg.n_outstanding():
        sg.chunk()
    fused_ids = [id(t) for m in gen.modules() if getattr(m, "_fused", None) for t in m._fused.values()]
    with torch.no_grad():
        for p in raw.parameters():
            p.add_(0.2 * torch.randn_like(p))
    sync_weights_inplace(gen, raw)
    assert fused_ids == [id(t) for m in gen.modules() if getattr(m, "_fused", None) for t in m._fused.values()]
    sg.version += 1
    for i, p in enumerate(prompts):
        sg.add(p, ("b", i))
    got = []
    while sg.n_outstanding():
        got += sg.chunk()
    ref_m = copy.deepcopy(raw).half()
    ref_m.fuse_for_inference(torch.float16)
    for meta, row, lp, v0 in got:
        assert v0 == 1
        ref = _tf_logp(ref_m, prompts[0], row, 1.0)
        assert np.abs(ref - lp).mean() < 0.05


@cuda
def test_trainer_step_with_behaviour_logps():
    """blps equal to the current policy's log-probs -> TIS weights ~1, nothing clipped."""
    import types
    raw, gen = _models()
    a = types.SimpleNamespace(mode="1gpu", optim="adam", adam_lr=1e-5, muon_lr=9e-4, warmup=0, max_new=16,
                              tok_budget=512, pad_to=16, temperature=1.0, tis_cap=2.0, grad_clip=1.0)
    tr = rl_grpo.Trainer(raw, torch.device("cuda"), a)
    prompt = [5, 6, 7, 8]
    seqs, blps = [], []
    rng = np.random.default_rng(1)
    for _ in range(4):
        row = rng.integers(1, 512, size=6).tolist()
        seqs.append((np.array(prompt + row, dtype=np.int64), len(prompt)))
        blps.append(_tf_logp(raw, prompt, row, 1.0))
    st = tr.step(0, seqs, [1.0, -1.0, 0.5, -0.5], blps)
    assert st["train_seqs"] == 4
    assert abs(st["tis_mean"] - 1.0) < 0.02 and st["tis_clip_frac"] == 0.0
    with pytest.raises(AssertionError):
        tr.step(1, seqs, [1.0, -1.0, 0.5, -0.5], [b[:-1] for b in blps])


def test_overlong_soft_penalty_and_advantages():
    pen = rl_grpo.overlong_pen
    assert pen(1500, True, 2048, 0) == 0.0  # off
    assert pen(1536, True, 2048, 1536) == 0.0  # at L_start: no penalty
    assert pen(1000, True, 2048, 1536) == 0.0
    assert abs(pen(1792, True, 2048, 1536) + 0.5) < 1e-12  # halfway: -0.5
    assert pen(2048, True, 2048, 1536) == -1.0  # at the cap: -1
    assert pen(5000, True, 2048, 1536) == -1.0  # clipped
    assert pen(900, False, 2048, 1536) == -1.0  # unclosed / truncated: full -1
    # applied before the group baseline: two correct answers, the long one gets the lower advantage; an all-wrong
    # group gets signal only from length (shorter wrong > longer / truncated wrong)
    r = np.array([1.0, 1.0, 0.0, 0.0])
    p = np.array([pen(L, c, 2048, 1536) for L, c in [(800, True), (1792, True), (900, True), (2048, False)]])
    a = rl_grpo.group_adv(r + p, "grpo")
    assert abs(a.sum()) < 1e-12 and a[0] > a[1] > a[2] > a[3]
    np.testing.assert_allclose(a, (r + p) - (r + p).mean())
    a0 = rl_grpo.group_adv(np.zeros(3) + np.array([0.0, -0.5, -1.0]), "grpo")
    np.testing.assert_allclose(a0, [0.5, 0.0, -0.5])


def test_overlong_mixed_only():
    pen = np.array([0.0, -0.5, -1.0])
    wrong = np.zeros(3)
    np.testing.assert_allclose(rl_grpo.shaped_rewards(wrong, pen, True), wrong)  # all-wrong: untouched
    assert not rl_grpo.group_adv(rl_grpo.shaped_rewards(wrong, pen, True)).any()
    np.testing.assert_allclose(rl_grpo.shaped_rewards(wrong, pen, False), pen)  # mixed_only off: length-only signal
    assert rl_grpo.len_only(wrong, rl_grpo.shaped_rewards(wrong, pen, False)) == 1
    assert rl_grpo.len_only(wrong, rl_grpo.shaped_rewards(wrong, pen, True)) == 0
    mixed = np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(rl_grpo.shaped_rewards(mixed, pen, True), mixed + pen)  # mixed: shaped
    assert rl_grpo.len_only(mixed, mixed + pen) == 0
    allc = np.ones(3)  # all correct: shaped (prefers shorter correct), counted as length-only signal
    assert rl_grpo.len_only(allc, rl_grpo.shaped_rewards(allc, pen, True)) == 1


def test_reward_modes():
    import types
    vocab = {1: "think...", 2: "</think>", 3: "\n\n\\boxed{42}", 4: "\n\n\\boxed{}", 5: "\n\n42"}
    tk = types.SimpleNamespace(decode=lambda o, skip_special_tokens=False: "".join(vocab[t] for t in o))
    fo = rl_grpo.format_ok
    assert fo(tk, 2, [1, 2, 3], True) == 1.0  # closed + non-empty boxed after </think>
    assert fo(tk, 2, [1, 3], True) == 0.0  # never closed
    assert fo(tk, 2, [1, 2, 4], True) == 0.0  # empty box
    assert fo(tk, 2, [1, 2, 5], True) == 0.0  # no box
    a = types.SimpleNamespace(reward_mode="format", answer_direct=1, random_p=0.5)
    rng = np.random.default_rng(0)
    assert rl_grpo.train_reward(0.0, True, [1, 2, 3], tk, 2, a, rng) == 1.0  # wrong but well-formed -> 1
    a.reward_mode = "correct"
    assert rl_grpo.train_reward(0.0, True, [1, 2, 3], tk, 2, a, rng) == 0.0
    a.reward_mode, a.random_p = "random", 0.3
    r = [rl_grpo.train_reward(1.0, True, [1, 2, 3], tk, 2, a, rng) for _ in range(4000)]
    assert set(r) <= {0.0, 1.0} and abs(np.mean(r) - 0.3) < 0.03  # independent of the content


def _ds_group(rew, pen=None):
    return {"rew": list(map(float, rew)), "pen": list(map(float, pen)) if pen is not None else [0.0] * len(rew)}


def test_ds_drop_reward_mode_unchanged():
    import types
    a = types.SimpleNamespace(dyn_sampling=1, adv="grpo")
    for rew, drop in [([1, 1, 1, 1], True), ([0, 0, 0, 0], True), ([1, 0, 0, 1], False)]:
        g = _ds_group(rew, [0, -0.5, 0, 0])
        assert rl_grpo.ds_drop(g, a) == drop and "adv" not in g  # default filter: raw reward range only
    a.dyn_sampling_on = "reward"
    assert rl_grpo.ds_drop(_ds_group([1, 1, 1, 1]), a)


def test_ds_drop_adv_mode():
    import collections
    import types
    tot = collections.Counter()
    a = types.SimpleNamespace(dyn_sampling=1, dyn_sampling_on="adv", adv="grpo", overlong_mixed_only=1)
    for rew in ([1] * 4, [0] * 4, [1, 0, 1, 1]):  # no shaping: adv filter == reward filter
        g = _ds_group(rew)
        assert rl_grpo.ds_drop(g, a, tot) == (np.ptp(rew) == 0)
    assert tot["ds_kept_zero_var_groups"] == 0
    pen = [0, -0.5, 0, 0]
    g = _ds_group([1] * 4, pen)  # all correct, one overlong: shaped rewards differ -> nonzero advantages -> kept
    assert not rl_grpo.ds_drop(g, a, tot)
    assert np.abs(g["adv"]).max() > 1e-12 and tot["ds_kept_zero_var_groups"] == 1
    np.testing.assert_allclose(g["adv"], rl_grpo.group_adv(np.array([1, 0.5, 1, 1]), "grpo"))
    assert rl_grpo.ds_drop(_ds_group([0] * 4, pen), a, tot)  # all wrong, mixed-only shaping: no signal -> drop
    a.overlong_mixed_only = 0
    assert not rl_grpo.ds_drop(_ds_group([0] * 4, pen), a, tot)  # all-wrong groups shaped too -> length-only signal
    assert tot["ds_kept_zero_var_groups"] == 2
    a.adv_norm = "batch"  # batch scaling comes later (scale_batch_adv), not at filter time
    g = _ds_group([1] * 4, pen)
    assert not rl_grpo.ds_drop(g, a)
    np.testing.assert_allclose(g["adv"], rl_grpo.group_adv(np.array([1, 0.5, 1, 1]), "grpo"))
