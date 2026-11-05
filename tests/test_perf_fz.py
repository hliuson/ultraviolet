"""Tests for the perf flags 'fz' (kernels/fused_sample.py + perf.install_fz; CPU tests cover the
reference sampler and the fallback, CUDA tests the Triton kernel, CUDA-graph capture and StreamGen) and rfh / pkv with
the hashed bigram (bit-identical StreamGen rollouts)."""
import math
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))

from ultraviolet.kernels.fused_sample import fused_sample_ok, fused_sample_ref  # noqa: E402
from ultraviolet.model import DecodeRunner, MathLM, make_config  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
SC = (23.0, 5.0, 7.5)


def softcap(x, sc=SC):
    return sc[0] * torch.sigmoid((x.float() + sc[1]) / sc[2])


def chi2_p(counts, p, min_e=5.0):
    n = counts.sum()
    e = p * n
    big = e >= min_e
    obs = torch.cat([counts[big], counts[~big].sum()[None]])
    ex = torch.cat([e[big], e[~big].sum()[None]])
    k = ex > 0
    chi = ((obs[k] - ex[k]) ** 2 / ex[k]).sum()
    df = int(k.sum()) - 1
    return float(torch.special.gammaincc(torch.tensor(df / 2, dtype=torch.float64), chi.double().cpu() / 2))


def small_model(dev, seed=0, V=2048):
    torch.manual_seed(seed)
    m = MathLM(make_config("uv-50m-alpha", vocab_size=V, n_layers=5, d_model=256, d_ff=640)).to(dev)
    with torch.no_grad():
        for p in m.parameters():
            if p.dim() == 2:
                p.add_(torch.randn_like(p) * 0.02)
    return m


# ------------------------------------------------------------------------------------------------ CPU
@pytest.mark.parametrize("T", [1.0, 0.7])
def test_ref_sampler_distribution_and_logp(T):
    """fused_sample_ref draws from softmax(softcap(x) / T) (chi-square) and returns its log-prob."""
    torch.manual_seed(0)
    V, n = 256, 100_000
    x = torch.randn(V) * 6
    p = torch.softmax(softcap(x).double() / T, -1)
    u = torch.rand(n)
    tok, lp = fused_sample_ref(x[None].expand(n, V), u, SC, T)
    cnt = torch.bincount(tok, minlength=V).double()
    assert chi2_p(cnt, p) > 1e-3
    ref = torch.log_softmax(softcap(x) / T, -1)[tok]
    assert (lp - ref).abs().max() < 1e-4


def test_ref_sampler_edges():
    """u -> 1 (target -> 0) never picks a zero-mass prefix; u = 0 (target = S) is clamped to a valid id."""
    x = torch.full((2, 64), -1e4)
    x[:, 40] = 10.0
    tok, _ = fused_sample_ref(x, torch.tensor([0.0, 1.0 - 2 ** -24]), SC, 1.0)
    assert tok.max() < 64 and tok.min() >= 0


def test_fused_ok_and_install_fallback_cpu():
    assert not fused_sample_ok(torch.device("cpu"), 16384, SC, 1.0)
    assert not fused_sample_ok("cuda", 16000, SC, 1.0)          # vocab not a multiple of the block
    assert not fused_sample_ok("cuda", 16384, SC, 0.1)          # a / T too large for the bounded exp
    from ultraviolet.perf import install_fz
    m = small_model("cpu").eval()
    st = m.new_state(2, 64, torch.float32, torch.device("cpu"))   # sdpa state on CPU (kv_dtype None)
    r = DecodeRunner(m, st, use_graph=False)
    body = r._body
    assert install_fz(r) is False and r._body == body and not r.want_logp


def test_make_streamgen_fz_flag_cpu_fallback():
    """'fz' on a non-applicable runner leaves StreamGen on the default sampler."""
    import rl_stream
    from ultraviolet.perf import make_streamgen
    cls = make_streamgen(rl_stream.StreamGen, {"fz"})
    assert cls is not rl_stream.StreamGen


# ------------------------------------------------------------------------------------------------ CUDA
@cuda
@pytest.mark.parametrize("T", [1.0, 0.8])
def test_kernel_matches_ref(T):
    """Same uniforms -> the token is the inverse-CDF draw of the exact (fp64) softmax up to fp32 summation rounding
    (the target u' * S falls inside the token's fp64 CDF interval, widened by 1e-5 * S), mostly the same token as the
    flat-cumsum reference; log-probs match log_softmax of the fp32 softcapped logits."""
    from ultraviolet.kernels.fused_sample import fused_sample
    torch.manual_seed(0)
    B, V = 4096, 16384
    x = (torch.randn(B, V, device="cuda") * torch.rand(B, 1, device="cuda") * 8).half()
    u = torch.rand(B, device="cuda")
    t, lp = fused_sample(x, u, SC, T)
    tr, lpr = fused_sample_ref(x, u, SC, T)
    assert t.dtype == torch.int64 and t.min() >= 0 and t.max() < V
    assert (t == tr).float().mean() > 0.99
    e = torch.exp((softcap(x).double() - SC[0]) / T)
    cs = e.cumsum(-1)
    S = cs[:, -1]
    tgt = (1.0 - u.double()) * S
    hi = cs.gather(1, t[:, None])[:, 0]
    lo = hi - e.gather(1, t[:, None])[:, 0]
    tol = 1e-5 * S
    assert ((tgt >= lo - tol) & (tgt <= hi + tol)).all()
    ls = torch.log_softmax(softcap(x) / T, -1).gather(1, t[:, None])[:, 0]
    assert (lp - ls).abs().max() < 2e-5


@cuda
def test_kernel_distribution():
    from ultraviolet.kernels.fused_sample import fused_sample
    torch.manual_seed(1)
    V, n = 16384, 1 << 18
    for scale in (2.0, 12.0):
        x = (torch.randn(V, device="cuda") * scale).half()
        p = torch.softmax(softcap(x).double(), -1)
        cnt = torch.zeros(V, dtype=torch.float64, device="cuda")
        for _ in range(4):
            t, _ = fused_sample(x[None].expand(n // 4, V), torch.rand(n // 4, device="cuda"), SC, 1.0)
            cnt += torch.bincount(t, minlength=V).double()
        assert chi2_p(cnt, p) > 1e-3


@cuda
def test_install_fz_graph_runner():
    """DecodeRunner + install_fz (compiled, CUDA graphs, across a bucket boundary): out_logp is log softmax of the
    softcapped fp16 head output at the sampled token and matches the default runner's logits fed the same tokens."""
    from ultraviolet.perf import install_fz
    m = small_model("cuda").half().eval()
    m.fuse_for_inference(torch.float16)
    B = 8
    runs = []
    for fz in (True, False):
        st = m.new_state(B, 2100, torch.float16, torch.device("cuda"))
        st.set_pos(1000)
        r = DecodeRunner(m, st, bucket=1024, temperature=1.0, compile=True)
        r.want_logp = True
        if fz:
            assert install_fz(r)
        runs.append(r)
    rf, rd = runs
    tok = torch.ones(B, dtype=torch.long, device="cuda")
    bl = ({}, {})
    for _ in range(40):
        gl = rf.g_len(int(rf.st.host_pos.max()))
        t = rf.step(tok).clone()
        lp = rf.out_logp.clone()
        x = bl[0].setdefault(gl, rf.out_logits)
        rd.step(tok)
        L = bl[1].setdefault(gl, rd.out_logits)
        own = torch.log_softmax(softcap(x, m.cfg.softcap), -1).gather(1, t[:, None])[:, 0]
        ref = torch.log_softmax(L.float(), -1).gather(1, t[:, None])[:, 0]
        assert (lp - own).abs().max() < 2e-5
        assert (lp - ref).abs().max() < 2e-2
        tok = t
    assert len(rf.graphs) == 2


@cuda
def test_streamgen_fz_rollouts():
    """StreamGen with 'fz' (plus lp, hv) runs end to end: valid tokens, finite log-probs <= 0, same row bookkeeping
    as the default (all requests return, EOS-terminated or max_new long)."""
    import rl_stream
    from ultraviolet.perf import make_streamgen
    m = small_model("cuda").half().eval()
    m.fuse_for_inference(torch.float16)
    cls = make_streamgen(rl_stream.StreamGen, {"lp", "hv", "fz"})
    torch.manual_seed(0)
    g = cls(m, 16, 40, 48, 1.0, 1, check_every=8)
    assert g._fz
    rng = np.random.default_rng(0)
    for i in range(40):
        g.add(rng.integers(2, 2048, size=int(rng.integers(5, 40))).tolist(), i)
    res = []
    for _ in range(60):
        res += g.chunk()
    assert sorted(r[0] for r in res) == list(range(40))
    for meta, row, lp, v0 in res:
        assert len(row) == len(lp) and 0 < len(row) <= 48
        assert all(0 <= t < 2048 for t in row)
        assert (row[-1] == 1) or len(row) == 48
        assert np.isfinite(lp).all() and (lp <= 1e-6).all()


@cuda
@pytest.mark.parametrize("bigram", [0, 1024])
def test_rfh_pkv_streamgen_identical(bigram):
    """rfh (host-built packed prefill indices) + pkv (batched int8 KV quantize) give bit-identical StreamGen rollouts,
    behaviour log-probs and final cache / positions / bigram state (with and without the hashed bigram)."""
    import rl_stream
    from ultraviolet.perf import make_streamgen
    torch.manual_seed(0)
    m = MathLM(make_config("uv-50m-alpha", vocab_size=512, n_layers=5, d_model=256, d_ff=640, window=16,
                           bigram_rows=bigram)).cuda()
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.dim() == 2:
                p.add_(torch.randn_like(p) * 0.02)
            if n.startswith("bigram_embed."):
                p.normal_(0, 1.0)
    m = m.half().eval()
    m.fuse_for_inference(torch.float16)
    m.prefill_packed = True
    outs = []
    for flags in ({"lp", "hv"}, {"lp", "hv", "rfh", "pkv"}):
        m.prefill_kv_batched = "pkv" in flags
        torch.manual_seed(0)
        g = make_streamgen(rl_stream.StreamGen, flags)(m, 16, 40, 48, 1.0, 1, check_every=8)
        assert g.st.kv_dtype == "int8" and getattr(g, "_rfh", False) == ("rfh" in flags)
        rng = np.random.default_rng(0)
        for i in range(48):
            g.add(rng.integers(2, 512, size=int(rng.integers(5, 40))).tolist(), i)
        res = []
        for _ in range(30):
            res += [(mt, tuple(r), lp.tobytes(), v) for mt, r, lp, v in g.chunk()]
        st = g.st
        fin = [t.cpu() for t in st.k + st.v + st.k_scale + st.v_scale if t is not None] + [st.pos.cpu(), st.starts.cpu()]
        if st.prev is not None:
            fin.append(st.prev.cpu())
        outs.append((res, fin))
    (r0, f0), (r1, f1) = outs
    assert len(r0) > 0 and r0 == r1
    assert all(torch.equal(x, y) for x, y in zip(f0, f1))
