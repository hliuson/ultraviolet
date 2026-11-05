import math
import numpy as np

import pytest
import torch

from ultraviolet.model import MathLM, generate, make_config
from ultraviolet.optim import apply_schedule, build_optimizers, polar_express


def tiny(**kw):
    torch.manual_seed(0)
    cfg = make_config("tiny", **kw)
    m = MathLM(cfg)
    # perturb zero-init outputs / gates so the test exercises every path
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith(("wo", "w_down", "ve_gate", "attn_gate", "xsa_alpha", "dense2")):
                p.normal_(0, 0.3)
    return m


@pytest.mark.parametrize("xsa,mudd", [(False, False), (True, True)])
def test_shapes_and_loss(xsa, mudd):
    m = tiny(xsa=xsa, mudd=mudd)
    idx = torch.randint(0, 512, (2, 24))
    logits = m(idx)
    assert logits.shape == (2, 24, 512)
    a, b, c = m.cfg.softcap
    assert logits.min() >= 0 and logits.max() <= a
    loss = m(idx, idx)
    loss2 = m(idx, idx, loss_chunk=7)
    assert torch.allclose(loss, loss2, atol=1e-5)
    loss.backward()
    assert math.isfinite(loss.item())


@pytest.mark.parametrize("xsa,mudd", [(False, False), (True, True)])
def test_causal_and_window(xsa, mudd):
    m = tiny(xsa=xsa, mudd=mudd).eval()
    T = 20
    idx = torch.randint(0, 512, (1, T))
    base = m(idx)
    # changing a future token never changes earlier logits
    idx2 = idx.clone()
    idx2[0, 12] = (idx2[0, 12] + 1) % 512
    out2 = m(idx2)
    assert torch.allclose(base[0, :12], out2[0, :12], atol=1e-6)
    assert not torch.allclose(base[0, 12:], out2[0, 12:], atol=1e-6)


def test_sliding_window_mask():
    m = tiny(global_every=0)  # all local, window 8
    msk = m._local_mask(12, "cpu")
    assert msk[10, 3].item() is True and msk[10, 2].item() is False and msk[10, 11].item() is False
    assert int(msk[10].sum()) == 8
    # single local layer: output at position i ignores tokens <= i - window ... only for one layer deep;
    # with all-local depth L the receptive field is L*(window-1). Check a 1-layer model exactly.
    torch.manual_seed(0)
    cfg = make_config("tiny", n_layers=1, global_every=0, ve_every=0)
    m1 = MathLM(cfg).eval()
    with torch.no_grad():
        m1.blocks[0].attn.wo.normal_(0, 0.3)
    idx = torch.randint(0, 512, (1, 20))
    a = m1(idx)
    idx2 = idx.clone()
    idx2[0, 2] = (idx2[0, 2] + 1) % 512
    b = m1(idx2)
    assert torch.allclose(a[0, 10:], b[0, 10:], atol=1e-6)  # position 10 sees keys 3..10
    assert not torch.allclose(a[0, 9], b[0, 9], atol=1e-6)  # position 9 sees key 2


@pytest.mark.parametrize("xsa,mudd", [(False, False), (True, True)])
def test_cache_decode_matches_full(xsa, mudd):
    m = tiny(xsa=xsa, mudd=mudd).eval().double()
    T, t0 = 30, 5  # window 8: ring buffer wraps several times
    idx = torch.randint(1, 512, (3, T))
    full = m(idx)
    st = m.new_state(3, T, dtype=torch.float64, device="cpu")
    starts = torch.zeros(3, dtype=torch.long)
    lg = m.prefill(idx[:, :t0], starts, st)
    assert torch.allclose(lg, full[:, t0 - 1], atol=1e-8)
    for t in range(t0, T):
        lg = m.decode_step(idx[:, t], st, g_len=T)
        assert torch.allclose(lg, full[:, t], atol=1e-8), t


def test_left_padded_prefill_matches_unpadded():
    m = tiny().eval().double()
    seqs = [torch.randint(1, 512, (n,)) for n in (14, 9, 4)]
    T, extra = 14, 12
    idx = torch.zeros(3, T, dtype=torch.long)
    starts = torch.tensor([T - len(s) for s in seqs])
    for b, s in enumerate(seqs):
        idx[b, T - len(s):] = s
    cont = torch.randint(1, 512, (3, extra))
    st = m.new_state(3, T + extra, dtype=torch.float64, device="cpu")
    outs = [m.prefill(idx, starts, st)]
    for t in range(extra):
        outs.append(m.decode_step(cont[:, t], st, g_len=T + extra))
    for b, s in enumerate(seqs):
        ref = m(torch.cat([s, cont[b]])[None])[0]
        n = len(s)
        assert torch.allclose(outs[0][b], ref[n - 1], atol=1e-8)
        for t in range(extra):
            assert torch.allclose(outs[t + 1][b], ref[n + t], atol=1e-8), (b, t)


def test_generate_greedy_matches_full_argmax():
    m = tiny().eval()
    m.fuse_for_inference(torch.float32)
    prompts = [[5, 6, 7, 8], [9, 10]]
    out = generate(m, prompts, max_new=12, temperature=0.0, use_graph=False, dtype=torch.float32, bucket=4)
    for p, o in zip(prompts, out):
        seq = list(p)
        for _ in range(12):
            nxt = int(m(torch.tensor([seq]))[0, -1].argmax())
            seq.append(nxt)
        assert seq[len(p):] == o


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda")
def test_cuda_graph_decode_matches_eager():
    m = tiny(xsa=True, mudd=True).cuda().eval()
    prompts = [[5, 6, 7, 8, 3, 2], [9, 10]]
    m.half().fuse_for_inference(torch.float16)
    a = generate(m, prompts, max_new=20, temperature=0.0, use_graph=False, bucket=8)
    b = generate(m, prompts, max_new=20, temperature=0.0, use_graph=True, bucket=8)
    assert a == b



@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda")
def test_compiled_graph_decode_logits_match_eager():
    from ultraviolet.model import DecodeRunner
    m = tiny(xsa=True, mudd=True).cuda().eval().half()
    m.fuse_for_inference(torch.float16)
    seq = torch.randint(1, 512, (3, 24), device="cuda")
    outs = []
    for compile_, graph in ((False, False), (True, True)):
        st = m.new_state(3, 24, torch.float16, "cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            m.prefill(seq[:, :4], torch.zeros(3, dtype=torch.long, device="cuda"), st)
        r = DecodeRunner(m, st, bucket=8, use_graph=graph, temperature=0.0, compile=compile_)
        lg = []
        for t in range(4, 24):
            r.step(seq[:, t])
            lg.append(r.out_logits.clone())
        outs.append(torch.stack(lg))
    assert torch.allclose(outs[0], outs[1], atol=3e-2), (outs[0] - outs[1]).abs().max()


def test_polar_express_orthogonalizes():
    torch.manual_seed(0)
    for shape in [(64, 32), (32, 96)]:
        U = polar_express(torch.randn(*shape))
        s = torch.linalg.svdvals(U)
        assert s.min() > 0.6 and s.max() < 1.25, (shape, s.min(), s.max())


@pytest.mark.parametrize("xsa,mudd", [(False, False), (True, True)])
def test_optimizer_step(xsa, mudd):
    m = tiny(xsa=xsa, mudd=mudd)
    mu, adam = build_optimizers(m)
    n_mu = sum(p.numel() for g in mu.param_groups for p in g["params"])
    n_ad = sum(p.numel() for g in adam.param_groups for p in g["params"])
    assert n_mu + n_ad == sum(p.numel() for p in m.parameters())
    idx = torch.randint(0, 512, (4, 16))
    losses = []
    for step in range(30):
        apply_schedule(mu, adam, step, 30, warmup=3)
        loss = m(idx[:, :-1], idx[:, 1:])
        loss.backward()
        mu.step()
        adam.step()
        mu.zero_grad(set_to_none=True)
        adam.zero_grad(set_to_none=True)
        losses.append(loss.item())
    assert losses[-1] < losses[0] - 1.0, losses


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda")
def test_flex_matches_sdpa():
    torch.manual_seed(0)
    cfg = make_config("tiny", window=48, xsa=True, mudd=True)
    m = MathLM(cfg).cuda()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith(("wo", "w_down", "ve_gate", "attn_gate", "xsa_alpha", "dense2")):
                p.normal_(0, 0.3)
    idx = torch.randint(0, 512, (2, 256), device="cuda")
    ref = m(idx[:, :-1], idx[:, 1:])
    m.attn_impl = "flex"
    out = m(idx[:, :-1], idx[:, 1:])
    assert torch.allclose(ref, out, atol=1e-4), (ref.item(), out.item())


def test_continuous_batching_matches_static_greedy():
    from ultraviolet.model import generate_continuous
    m = tiny().eval()
    m.fuse_for_inference(torch.float32)
    torch.manual_seed(1)
    prompts = [torch.randint(1, 512, (n,)).tolist() for n in (5, 3, 9, 4, 6)]
    lens = [7, 12, 3, 9, 5]
    out = generate_continuous(m, prompts, max_new=12, batch=2, temperature=0.0, use_graph=False, compile=False,
                              dtype=torch.float32, target_lens=lens, check_every=4, bucket=8)
    for p, n, o in zip(prompts, lens, out):
        ref = generate(m, [p], max_new=12, temperature=0.0, use_graph=False, compile=False, dtype=torch.float32,
                       bucket=8)[0]
        assert o == ref[:n], (p, n, o, ref)


def _perturbed(**kw):
    m = tiny(**kw)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith(("wo", "w_a", "w_b")):
                p.normal_(0, 0.3)
    return m


def test_conv_mixer_cache_decode_and_padding():
    m = _perturbed(conv_layers=[0, 2]).eval().double()
    assert m.blocks[0].attn.kind == "conv" and m.blocks[1].attn.kind == "attn"
    T, t0 = 20, 5
    idx = torch.randint(1, 512, (2, T))
    full = m(idx)
    st = m.new_state(2, T, dtype=torch.float64, device="cpu")
    lg = m.prefill(idx[:, :t0], torch.zeros(2, dtype=torch.long), st)
    assert torch.allclose(lg, full[:, t0 - 1], atol=1e-8)
    for t in range(t0, T):
        assert torch.allclose(m.decode_step(idx[:, t], st, g_len=T), full[:, t], atol=1e-8), t
    # left padding: row 1 has a 3-token prompt padded to 6
    st = m.new_state(2, T, dtype=torch.float64, device="cpu")
    pad = torch.zeros(2, 6, dtype=torch.long)
    pad[0] = idx[0, :6]
    pad[1, 3:] = idx[1, :3]
    lg = m.prefill(pad, torch.tensor([0, 3]), st)
    ref1 = m(idx[1:2, :3])[0, -1]
    assert torch.allclose(lg[1], ref1, atol=1e-8)
    lg2 = m.decode_step(idx[:, 6], st, g_len=T)
    ref1b = m(torch.cat([idx[1, :3], idx[1, 6:7]])[None])[0, -1]
    assert torch.allclose(lg2[1], ref1b, atol=1e-8)


def _has_fla():
    try:
        import fla.ops.gated_delta_rule  # noqa: F401
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(not _has_fla(), reason="needs fla + cuda")
def test_gdn_mixer_cache_decode_matches_chunk():
    torch.manual_seed(0)
    m = _perturbed(local_mixer="gdn", gdn_dk=16, gdn_dv=16).cuda().eval()
    assert m.blocks[0].attn.kind == "gdn" and m.blocks[1].attn.kind == "attn"
    T, t0 = 40, 7
    idx = torch.randint(1, 512, (2, T), device="cuda")
    full = m(idx)
    st = m.new_state(2, T, dtype=torch.float32, device="cuda")
    lg = m.prefill(idx[:, :t0], torch.zeros(2, dtype=torch.long, device="cuda"), st)
    assert torch.allclose(lg, full[:, t0 - 1], atol=2e-3), (lg - full[:, t0 - 1]).abs().max()
    for t in range(t0, T):
        out = m.decode_step(idx[:, t], st, g_len=T)
        assert torch.allclose(out, full[:, t], atol=2e-3), (t, (out - full[:, t]).abs().max())
    # left padded prefill
    st = m.new_state(2, T, dtype=torch.float32, device="cuda")
    pad = torch.zeros(2, 9, dtype=torch.long, device="cuda")
    pad[0] = idx[0, :9]
    pad[1, 4:] = idx[1, :5]
    lg = m.prefill(pad, torch.tensor([0, 4], device="cuda"), st)
    assert torch.allclose(lg[1], m(idx[1:2, :5])[0, -1], atol=2e-3)
    out = m.decode_step(idx[:, 9], st, g_len=T)
    ref = m(torch.cat([idx[1, :5], idx[1, 9:10]])[None])[0, -1]
    assert torch.allclose(out[1], ref, atol=2e-3)


def test_mixer_params_classified():
    m = tiny(conv_layers=[0], local_mixer="gdn")
    mu, adam = build_optimizers(m)
    n_mu = sum(p.numel() for g in mu.param_groups for p in g["params"])
    n_ad = sum(p.numel() for g in adam.param_groups for p in g["params"])
    assert n_mu + n_ad == sum(p.numel() for p in m.parameters())


def test_prefill_chunking_matches_unchunked():
    m = _perturbed(conv_layers=[0]).eval().double()
    idx = torch.randint(1, 512, (5, 12))
    starts = torch.tensor([0, 3, 0, 5, 1])
    st1 = m.new_state(5, 20, dtype=torch.float64, device="cpu")
    ref = m.prefill(idx, starts, st1)
    m.prefill_chunk_tokens = 24  # 2 rows per chunk
    st2 = m.new_state(5, 20, dtype=torch.float64, device="cpu")
    out = m.prefill(idx, starts, st2)
    del m.prefill_chunk_tokens
    assert torch.allclose(out, ref, atol=1e-10)
    tok = torch.randint(1, 512, (5,))
    assert torch.allclose(m.decode_step(tok, st1, g_len=20), m.decode_step(tok, st2, g_len=20), atol=1e-10)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs cuda")
@pytest.mark.parametrize("impl", ["kernel-fp16", "kernel-int8"])
def test_kernel_decode_impl_matches_sdpa(impl):
    torch.manual_seed(0)
    m = _perturbed(head_dim=64).cuda().half().eval()
    assert m.resolve_decode_impl(torch.float16, "cuda") == "kernel-int8"   # the default on CUDA fp16
    assert m.resolve_decode_impl(torch.float32, "cuda") == "sdpa"
    T, t0 = 48, 9
    idx = torch.randint(1, 512, (3, T), device="cuda")
    outs = {}
    for mode in ("sdpa", impl):
        m.decode_impl = mode
        st = m.new_state(3, T, dtype=torch.float16, device="cuda")
        assert (st.kv_dtype is None) == (mode == "sdpa")
        lg = [m.prefill(idx[:, :t0], torch.tensor([0, 2, 4], device="cuda"), st)]
        for t in range(t0, T):
            lg.append(m.decode_step(idx[:, t], st, g_len=T))
        outs[mode] = torch.stack(lg).float()
    m.decode_impl = "auto"
    tol = 2e-2 if impl == "kernel-fp16" else 0.15
    assert (outs["sdpa"] - outs[impl]).abs().max() < tol, (outs["sdpa"] - outs[impl]).abs().max()


def test_trace_stream_epochs_and_rows(tmp_path):
    from ultraviolet.data import TraceStream, trace_rows
    docs = [[5, 6, 0], [7, 0], [8, 9, 10, 0], [11, 0]]
    tok = np.array([t for d in docs for t in d], dtype=np.uint16)
    off = np.cumsum([0] + [len(d) for d in docs])
    np.save(tmp_path / "tokens.npy", tok)
    np.save(tmp_path / "offsets.npy", off)
    ts = TraceStream(str(tmp_path), seed=0)
    L = ts.L
    T = 3
    stream = np.concatenate([ts.window(j, T)[:T] for j in range(4 * L // T)])
    for e in range(3):  # every epoch is a permutation of the docs: each doc exactly once
        ep = stream[e * L:(e + 1) * L]
        assert sorted(ep.tolist()) == sorted(tok.tolist())
    assert ts.window(0, T)[-1] == ts.window(1, T)[0]  # windows overlap by the next-token target
    G, frac = 64, 0.2
    tot = sum(b - a for a, b in (trace_rows(s, frac, G) for s in range(1000)))
    assert tot == int(frac * G * 1000)
