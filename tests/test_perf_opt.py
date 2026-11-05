"""Equivalence tests for the opt-in performance paths (docs/perf_opt.md). CUDA only (sm75 kernels).

Tolerances are fp16-level: each new path is compared with the current path and, where it matters, both with an
fp32 reference (the new path must not be further from fp32 than the current one, up to a small slack).
"""
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))

from ultraviolet.model import MathLM, make_config  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
DEV = "cuda"


def small_cfg(**kw):
    # uv-50m-alpha geometry (MQA 8 x 64, window 256, XSA, VE, 2 NoPE globals), fewer layers / smaller vocab
    return make_config("uv-50m-alpha", vocab_size=2048, n_layers=5, d_model=256, d_ff=640, **kw)


def model(seed=0, **kw):
    torch.manual_seed(seed)
    m = MathLM(small_cfg(**kw)).to(DEV)
    with torch.no_grad():  # non-trivial weights (zero-init output projections would hide bugs)
        for p in m.parameters():
            if p.dim() == 2:
                p.add_(torch.randn_like(p) * 0.02)
    return m


# ------------------------------------------------------------------------------------------------ varlen attention
@cuda
@pytest.mark.parametrize("window", [None, 256])
def test_varlen_attention_matches_reference(window):
    from ultraviolet.kernels.varlen_attn import varlen_attention, varlen_attention_ref
    torch.manual_seed(0)
    lens = [300, 1000, 57, 700, 1]
    N = sum(lens)
    cu = torch.tensor([0] + list(np.cumsum(lens)), dtype=torch.int32, device=DEV)
    q = torch.randn(1, N, 8, 64, device=DEV, dtype=torch.half, requires_grad=True)
    k = torch.randn(1, N, 1, 64, device=DEV, dtype=torch.half, requires_grad=True)
    v = torch.randn(1, N, 1, 64, device=DEV, dtype=torch.half, requires_grad=True)
    sc = 1.13 / 8
    o = varlen_attention(q, k, v, cu, max(lens), sc, window)
    go = torch.randn_like(o)
    o.backward(go)
    g = [x.grad.clone() for x in (q, k, v)]
    qf, kf, vf = (x.detach().float().requires_grad_() for x in (q, k, v))
    r = varlen_attention_ref(qf, kf, vf, cu, sc, window)
    r.backward(go.float())
    assert (o.float() - r).abs().max() < 2e-3
    for a, b in zip(g, (qf.grad, kf.grad, vf.grad)):
        assert ((a.float() - b).norm() / b.norm()) < 5e-3


@cuda
def test_trunk_varlen_matches_sdpa_padded_and_packed():
    """Uniform (B, T) batch: train_attn varlen == sdpa (dense local mask); packed sequences == each sequence alone."""
    m = model()
    torch.manual_seed(1)
    x = torch.randint(0, 2048, (2, 600), device=DEV)
    with torch.autocast("cuda", dtype=torch.float16):
        h0 = m.trunk(x)[0]
        m.train_attn = "varlen"
        h1 = m.trunk(x)[0]
        m.train_attn = "sdpa"
    with torch.no_grad():
        h32 = m.trunk(x)[0]
    e0, e1 = (h0.float() - h32).abs().mean(), (h1.float() - h32).abs().mean()
    assert (h0.float() - h1.float()).abs().max() < 2e-2
    assert e1 < 1.1 * e0 + 1e-4, (e0, e1)
    # packed: three sequences back to back, positions restart per sequence
    lens = [600, 333, 91]
    seqs = [torch.randint(0, 2048, (L,), device=DEV) for L in lens]
    xp = torch.cat(seqs)[None]
    pos = torch.cat([torch.arange(L, device=DEV) for L in lens])
    cu = torch.tensor([0] + list(np.cumsum(lens)), dtype=torch.int32, device=DEV)
    with torch.autocast("cuda", dtype=torch.float16):
        hp = m.trunk(xp, pos=pos, cu_seqlens=cu, max_seqlen=max(lens))[0][0]
        ref = torch.cat([m.trunk(s[None])[0][0] for s in seqs])
    assert (hp.float() - ref.float()).abs().max() < 2e-2


# ------------------------------------------------------------------------------------------------ packing
def test_build_packed_layout():
    from ultraviolet.perf import build_packed
    rng = np.random.default_rng(0)
    items = []
    for i in range(20):
        L, s = int(rng.integers(5, 300)), int(rng.integers(2, 5))
        t = rng.integers(1, 100, size=L)
        items.append((t, s, float(i + 1), rng.normal(size=L - s).astype(np.float32)))
    mbs = build_packed(items, 1024, 0, seg_max=512)
    seen = {}
    for mb in mbs:
        cu = mb["cu"]
        assert cu[0] == 0 and cu[-1] == 1024 and np.all(np.diff(cu) <= max(mb["max_seqlen"], 1))
        for a, b in zip(cu[:-1], cu[1:]):
            assert np.all(mb["pos"][a:b] == np.arange(b - a))
            Av = mb["A"][0, a:b]
            if Av[0] > 0:
                i = int(Av[0]) - 1
                t, s, _, bl = items[i]
                assert b - a == len(t) - 1
                assert np.all(mb["x"][0, a:b] == t[:-1])
                assert np.all(mb["y"][0, a + s - 1:b] == t[s:]) and np.all(mb["y"][0, a:a + s - 1] == -1)
                assert np.allclose(mb["OL"][0, a + s - 1:b], bl)
                seen[i] = True
        assert np.all(mb["tgt"] == np.nonzero(mb["y"][0] != -1)[0])
    assert len(seen) == len(items)


# ------------------------------------------------------------------------------------------------ head losses
@cuda
@pytest.mark.parametrize("variant", ["head", "head16"])
def test_pg_loss_targets_match_pg_loss(variant):
    import rl_grpo
    from ultraviolet.perf import pg_loss_targets, pg_loss_targets16
    torch.manual_seed(0)
    V, d, N = 2048, 256, 3000
    w = (torch.randn(V, d, device=DEV) * 0.05).requires_grad_()
    h = F.rms_norm(torch.randn(1, N, d, device=DEV), (d,)).requires_grad_()
    y = torch.randint(0, V, (1, N), device=DEV)
    y[0, :500] = -1
    y[0, 1500:1700] = -1
    A = torch.randn(1, N, device=DEV)
    OL = -torch.rand(1, N, device=DEV) * 8
    sc = (23.0, 5.0, 7.5)
    l0, s0 = rl_grpo.pg_loss(h, w, y, A, OL, sc, 2.0, 1.0)
    g0 = torch.autograd.grad(l0, (h, w))
    tgt = (y.reshape(-1) != -1).nonzero().squeeze(1)
    if variant == "head":
        l1, s1 = pg_loss_targets(h, w, y, A, OL, tgt, sc, 2.0, 1.0, chunk=1024)
    else:
        l1, s1 = pg_loss_targets16(h, w, y, A, OL, tgt, sc, 2.0, 1.0, chunk=1024)
    g1 = torch.autograd.grad(l1, (h, w))
    tol = 1e-5 if variant == "head" else 3e-3
    assert abs(float(l1 - l0)) / abs(float(l0)) < tol
    assert torch.allclose(s0, s1, rtol=1e-3, atol=1.0)
    for a, b in zip(g1, g0):
        assert float((a - b).norm() / b.norm()) < (1e-4 if variant == "head" else 3e-3)


# ------------------------------------------------------------------------------------------------ prefill / decode
@cuda
@pytest.mark.parametrize("compiled", [False, True])
def test_prefill_packed_matches_padded(compiled):
    from ultraviolet.kernels.kv_quant import dequantize_kv
    from ultraviolet.model import _left_pad
    m = model().half().eval()
    m.fuse_for_inference(torch.float16)
    torch.manual_seed(3)
    prompts = [torch.randint(1, 2048, (L,)).tolist() for L in (40, 300, 7, 280)]
    idx, starts = _left_pad(prompts, 0, torch.device(DEV))
    outs = []
    m.prefill_compile = compiled
    for packed in (False, True):
        m.prefill_packed = packed
        st = m.new_state(6, 700, torch.float16, torch.device(DEV))
        rows = torch.tensor([5, 0, 2, 3], device=DEV)
        with torch.autocast("cuda", dtype=torch.float16):
            lg = m.prefill(idx, starts, st, rows=rows)
            tok = torch.zeros(6, dtype=torch.long, device=DEV)
            tok[rows] = lg.argmax(-1)
            dl = [m.decode_step(tok, st, 1024) for _ in range(3)]
        outs.append((lg.float(), st, [x.float() for x in dl]))
    m.prefill_packed = False
    (l0, st0, d0), (l1, st1, d1) = outs
    assert (l0 - l1).abs().max() < 5e-2
    assert torch.equal(st0.pos, st1.pos) and torch.equal(st0.starts, st1.starts)
    rows = [5, 0, 2, 3]
    T = idx.size(1)
    for l in range(m.cfg.n_layers):
        S = st0.k[l].size(2)
        for j, r in enumerate(rows):
            lo = int(starts[j]) if m.cfg.is_global(l) else max(int(starts[j]), T - S)
            sl = torch.arange(lo, T + 3, device=DEV) % S if not m.cfg.is_global(l) else torch.arange(lo, T + 3)
            a = dequantize_kv(st0.k[l][r, :, sl], st0.k_scale[l][r, :, sl]).float()
            b = dequantize_kv(st1.k[l][r, :, sl], st1.k_scale[l][r, :, sl]).float()
            assert (a - b).abs().max() < 0.05, (l, r)
    for a, b in zip(d0, d1):
        assert (a[rows] - b[rows]).abs().max() < 5e-2


@cuda
def test_runner_logp_in_graph():
    """DecodeRunner(want_logp) records log softmax(logits / T)[sampled] like rl_stream's eager computation, across a
    graph bucket boundary; sampled tokens are identical (same RNG consumption)."""
    from ultraviolet.model import DecodeRunner
    m = model().half().eval()
    m.fuse_for_inference(torch.float16)
    res = []
    for want in (False, True):
        st = m.new_state(8, 2100, torch.float16, torch.device(DEV))
        st.set_pos(1000)
        torch.manual_seed(0)
        run = DecodeRunner(m, st, bucket=1024, temperature=0.9, compile=True)
        run.want_logp = want
        tok = torch.ones(8, dtype=torch.long, device=DEV)
        toks, lps, refs = [], [], []
        logits_by_bucket = {}
        for i in range(40):
            gl = run.g_len(int(st.host_pos.max()))
            tok = run.step(tok).clone()
            lg = logits_by_bucket.setdefault(gl, run.out_logits)
            refs.append(F.log_softmax(lg.float() / 0.9, -1).gather(1, tok[:, None])[:, 0].clone())
            lps.append(run.out_logp.clone())
            toks.append(tok)
        res.append((torch.stack(toks), torch.stack(lps), torch.stack(refs)))
    (t0, _, r0), (t1, l1, r1) = res
    assert torch.equal(t0, t1)
    assert (l1 - r1).abs().max() < 1e-4


@cuda
@pytest.mark.parametrize("flags", [("hv",), ("lp", "hv")])
def test_streamgen_fast_chunk_same_outputs(flags):
    """StreamGen with the hv / lp variants returns the same rollouts (tokens, order, metadata) as rl_stream.StreamGen
    for the same seed; behaviour log-probs agree to 1e-3."""
    import rl_stream
    from ultraviolet.perf import make_streamgen
    m = model().half().eval()
    m.fuse_for_inference(torch.float16)
    outs = []
    for cls in (rl_stream.StreamGen, make_streamgen(rl_stream.StreamGen, set(flags))):
        torch.manual_seed(0)
        g = cls(m, 16, 40, 48, 1.0, 1, check_every=8)
        rng = np.random.default_rng(0)
        for i in range(40):
            g.add(rng.integers(2, 2048, size=int(rng.integers(5, 40))).tolist(), i)
        res = []
        for _ in range(30):
            res += g.chunk()
        outs.append(res)
    a, b = outs
    assert len(a) == len(b) > 0
    for (m0, r0, l0, v0), (m1, r1, l1, v1) in zip(a, b):
        assert m0 == m1 and r0 == r1 and v0 == v1
        assert np.abs(l0 - l1).max() < 1e-3  # fp16 summation-order noise
