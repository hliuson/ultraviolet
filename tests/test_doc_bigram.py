"""Opt-in pretraining features of ultraviolet.model: document masking (doc_layout / doc_inputs /
TokenShards.sample_docs), the hashed bigram embedding (MathConfig.bigram_rows) and the VE block pattern
(MathConfig.ve_block_pos). CPU tests run everywhere; CUDA tests cover the varlen kernel, CUDA graphs, the kernel
decode state, the packed prefill, and a train_pt.py smoke run."""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from ultraviolet.model import (DecodeRunner, MathConfig, MathLM, _left_pad, doc_layout,  # noqa: E402
                                      make_config)
from ultraviolet.optim import build_optimizers  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
EOS = 0


def tiny(**kw):
    torch.manual_seed(0)
    m = MathLM(make_config("tiny", **kw))
    with torch.no_grad():  # perturb zero-init outputs / gates / tables so every path is exercised
        for n, p in m.named_parameters():
            if n.endswith(("wo", "w_down", "ve_gate", "attn_gate", "xsa_alpha", "dense2")):
                p.normal_(0, 0.3)
            if n.startswith("bigram_embed."):
                p.normal_(0, 1.0)
            if n == "bigram_lambdas":
                p.uniform_(0.2, 0.6)
    return m


# ------------------------------------------------------------------------------------------------ VE pattern
def test_ve_block_pos_layers():
    c = make_config("uv-50m-alpha", ve_block_pos=[1, 4])
    assert [l for l in range(c.n_layers) if c.has_ve(l)] == [1, 4, 6, 9]
    assert [l for l in range(c.n_layers) if c.is_global(l)] == [4, 9]
    d = make_config("uv-50m-alpha")
    assert [l for l in range(d.n_layers) if d.has_ve(l)] == [1, 4, 7]  # default rule unchanged
    assert c.param_counts() == d.param_counts()  # one shared table either way
    m = MathLM(make_config("tiny", ve_block_pos=[0]))  # period 2 -> layers 0, 2
    assert [l for l, b in enumerate(m.blocks) if b.attn.has_ve] == [0, 2]
    assert sorted(k for k in m.state_dict() if "ve_gate" in k) == ["blocks.0.attn.ve_gate", "blocks.2.attn.ve_gate"]
    m2 = MathLM(MathConfig(**m.cfg.to_dict()))  # checkpoint round trip
    m2.load_state_dict(m.state_dict())


def test_ve_block_pos_decode_matches_full():
    m = tiny(ve_block_pos=[0, 1]).eval().double()
    T, t0 = 24, 5
    idx = torch.randint(1, 512, (2, T))
    full = m(idx)
    st = m.new_state(2, T, dtype=torch.float64, device="cpu")
    assert torch.allclose(m.prefill(idx[:, :t0], torch.zeros(2, dtype=torch.long), st), full[:, t0 - 1], atol=1e-8)
    for t in range(t0, T):
        assert torch.allclose(m.decode_step(idx[:, t], st, g_len=T), full[:, t], atol=1e-8)


# ------------------------------------------------------------------------------------------------ doc masking
def test_doc_layout():
    x = torch.tensor([[7, 3, 0, 4, 5, 6, 8, 9, 0, 1]])
    st, pos = doc_layout(x, EOS, 0)
    assert st[0].tolist() == [1, 0, 1, 0, 0, 0, 0, 0, 1, 0]
    assert pos[0].tolist() == [0, 1, 0, 1, 2, 3, 4, 5, 0, 1]
    st, pos = doc_layout(x, EOS, 3)  # the 6-token doc [0 4 5 6 8 9] splits into 3 + 3
    assert st[0].tolist() == [1, 0, 1, 0, 0, 1, 0, 0, 1, 0]
    assert pos[0].tolist() == [0, 1, 0, 1, 2, 0, 1, 2, 0, 1]


def _packed_row(lens, seed=0):
    g = torch.Generator().manual_seed(seed)
    docs = [torch.cat([torch.zeros(1, dtype=torch.long), torch.randint(1, 512, (n,), generator=g)]) for n in lens]
    return docs, torch.cat(docs)[None]


@pytest.mark.parametrize("bigram", [0, 300])
@pytest.mark.parametrize("xsa_mudd", [False, True])
def test_doc_mask_doc2_equals_alone(bigram, xsa_mudd):
    """Logits of doc 2 inside a packed row == doc 2 run alone (fp32), incl. sliding-window local layers (window 8 <
    doc length) and positions restarting at the doc's BOS (EOS)."""
    m = tiny(bigram_rows=bigram, xsa=xsa_mudd, mudd=xsa_mudd).eval()
    docs, row = _packed_row([10, 19, 7])
    out = m(row, docs=m.doc_inputs(row, EOS, 0, packed=False))
    a = len(docs[0])
    alone = m(docs[1][None])
    assert torch.allclose(out[0, a:a + len(docs[1])], alone[0], atol=1e-5), \
        (out[0, a:a + len(docs[1])] - alone[0]).abs().max()
    # without doc masking doc 2 sees doc 1
    plain = m(row)
    assert not torch.allclose(plain[0, a:a + len(docs[1])], alone[0], atol=1e-4)


def test_doc_mask_split_chunk_is_fresh_segment():
    m = tiny(bigram_rows=300).eval()
    docs, row = _packed_row([5, 20])
    L = 8
    out = m(row, docs=m.doc_inputs(row, EOS, L, packed=False))
    a = len(docs[0])
    # doc 2 = [EOS, 20 tokens] -> chunks of 8, 8, 5; the second chunk alone (no BOS, bigram reset)
    ch = docs[1][L:2 * L]
    assert torch.allclose(out[0, a + L:a + 2 * L], m(ch[None])[0], atol=1e-5)


def test_sample_docs(tmp_path):
    from ultraviolet.data import TokenShards
    rng = np.random.default_rng(0)
    toks = []
    for _ in range(300):
        toks += rng.integers(1, 50, int(rng.integers(1, 40))).tolist() + [EOS]
    np.asarray(toks, dtype=np.uint16).tofile(tmp_path / "train_0000.bin")
    json.dump({"vocab": 64, "dtype": "uint16", "eos_id": EOS}, open(tmp_path / "meta.json", "w"))
    ts = TokenShards(str(tmp_path), "train")
    arr = np.asarray(toks)
    for L in (0, 16):
        st = ts.segment_starts(0, L)
        assert st[0] == 0 and (np.diff(st) > 0).all()
        eos = set(np.flatnonzero(arr == EOS).tolist())
        assert eos <= set(st.tolist())
        _, pos = doc_layout(torch.from_numpy(arr[None]).long(), EOS, L)
        assert set(st.tolist()) == set(np.flatnonzero(pos[0].numpy() == 0).tolist())
        b = ts.sample_docs(64, 32, np.random.default_rng(1), L)
        assert b.shape == (64, 33)
        bs = set(st.tolist())
        rows = arr.tolist()
        for r in b:  # every window starts at a segment start
            r = r.tolist()
            o = [i for i in bs if rows[i:i + 33] == r]
            assert o, "window does not start at a segment start"


# ------------------------------------------------------------------------------------------------ bigram
def test_bigram_params_and_groups():
    V = 32768
    for preset in ("uv-50m-alpha", "v-deep-mqa"):
        for mult in (2, 5):
            for bd in (0, 128):
                c = make_config(preset, vocab_size=V, bigram_rows=mult * V, bigram_dim=bd)
                bdim = bd or c.d_model
                exp = mult * V * bdim + (c.d_model * bdim if bdim < c.d_model else 0) + c.n_layers
                assert c.param_counts()["bigram"] == exp
                m = MathLM(make_config("tiny", bigram_rows=mult * 512, bigram_dim=bd and 32))
                assert sum(p.numel() for n, p in m.named_parameters() if n.startswith("bigram")) == \
                    m.cfg.param_counts()["bigram"]
    assert "bigram" not in make_config("tiny").param_counts()
    m = MathLM(make_config("tiny", bigram_rows=1024, bigram_dim=32))
    mu, ad = build_optimizers(m)
    names = {id(p): n for n, p in m.named_parameters()}
    grp = {g["name"]: [names[id(p)] for p in g["params"]] for g in ad.param_groups}
    assert grp["bigram"] == ["bigram_embed.weight"] and "bigram_lambdas" in grp["scalars"]
    assert any(names[id(p)] == "bigram_proj" for g in mu.param_groups for p in g["params"])
    assert (m.bigram_embed.weight == 0).all()


def test_bigram_index_and_reset():
    from ultraviolet.model import bigram_index
    t, p = torch.tensor([5, 7]), torch.tensor([3, 3])
    h = bigram_index(t, p, torch.tensor([False, True]), 100)
    assert h[0] == ((36313 * 5) ^ (27191 * 3)) % 99 and h[1] == 99
    m = tiny(bigram_rows=300).eval()
    x = torch.randint(1, 512, (1, 12))
    a = m(x)
    x2 = x.clone()
    x2[0, 5] = (x2[0, 5] + 1) % 512
    b = m(x2)
    assert torch.allclose(a[0, :5], b[0, :5])


def test_old_checkpoint_loads_with_bigram_off():
    m = tiny()
    d = m.cfg.to_dict()
    for k in ("bigram_rows", "bigram_dim", "bigram_lambda_init", "ve_block_pos"):
        d.pop(k)
    m2 = MathLM(MathConfig(**d))
    m2.load_state_dict(m.state_dict())
    assert m2.cfg.bigram_rows == 0 and not hasattr(m2, "bigram_embed")
    mb = tiny(bigram_rows=300, bigram_dim=16)
    mb2 = MathLM(MathConfig(**mb.cfg.to_dict()))
    mb2.load_state_dict(mb.state_dict())
    x = torch.randint(0, 512, (2, 9))
    assert torch.equal(mb.eval()(x), mb2.eval()(x))


def _bigram_decode_check(m, dev, dtype, tol, use_graph=False, compile_=False, prompt_lens=(11, 4, 1, 7), extra=14):
    g = torch.Generator().manual_seed(5)
    prompts = [torch.randint(1, 512, (n,), generator=g).tolist() for n in prompt_lens]
    cont = torch.randint(1, 512, (len(prompts), extra), generator=g).to(dev)
    idx, starts = _left_pad(prompts, 0, torch.device(dev))
    T = idx.size(1)
    st = m.new_state(len(prompts), T + extra + 1, dtype, torch.device(dev))
    ac = torch.autocast("cuda", dtype=dtype) if dev == "cuda" else torch.autocast("cpu", enabled=False)
    with torch.no_grad():  # as generate(): prefill under autocast, decode steps without
        with ac:
            outs = [m.prefill(idx, starts, st).float()]
        r = DecodeRunner(m, st, bucket=8, use_graph=use_graph, temperature=0.0, compile=compile_)
        for t in range(extra):
            r.step(cont[:, t])
            outs.append(r.out_logits.float().clone())
    errs = []
    for b, p in enumerate(prompts):
        with torch.no_grad(), ac:
            ref = m(torch.cat([torch.tensor(p, device=dev), cont[b]])[None])[0].float()
        n = len(p)
        errs.append((outs[0][b] - ref[n - 1]).abs().max())
        for t in range(extra):
            errs.append((outs[t + 1][b] - ref[n + t]).abs().max())
    err = max(float(e) for e in errs)
    assert err < tol, err
    return err


@pytest.mark.parametrize("bd", [0, 16])
@pytest.mark.parametrize("xsa_mudd", [False, True])
def test_bigram_prefill_decode_matches_forward_cpu(bd, xsa_mudd):
    m = tiny(bigram_rows=300, bigram_dim=bd, xsa=xsa_mudd, mudd=xsa_mudd).eval().double()
    _bigram_decode_check(m, "cpu", torch.float64, 1e-5)  # fp32 logits: ulp-level (same with bigram off)


def test_bigram_changes_outputs():
    m = tiny(bigram_rows=300).eval()
    x = torch.randint(1, 512, (1, 10))
    a = m(x)
    with torch.no_grad():
        m.bigram_lambdas.zero_()
    assert not torch.allclose(a, m(x))


# ------------------------------------------------------------------------------------------------ CUDA
def _cuda_model(**kw):
    torch.manual_seed(0)
    m = MathLM(make_config("uv-50m-alpha", vocab_size=512, n_layers=5, d_model=256, d_ff=640, window=16, **kw))
    with torch.no_grad():
        for n, p in m.named_parameters():
            if p.dim() == 2:
                p.add_(torch.randn_like(p) * 0.02)
            if n.startswith("bigram_embed."):
                p.normal_(0, 1.0)
    return m.cuda()


@cuda
@pytest.mark.parametrize("bigram", [0, 1024])
def test_doc_mask_varlen_cuda_matches_alone(bigram):
    m = _cuda_model(bigram_rows=bigram).eval()  # fp32 through the varlen (CUTLASS) kernel
    docs, row = _packed_row([30, 41, 12, 9])
    row = row.cuda()
    d = m.doc_inputs(row, EOS, 0)
    assert d["cu"] is not None
    with torch.no_grad():
        out = m(row, docs=d)
        dense = m(row, docs=m.doc_inputs(row, EOS, 0, packed=False))
        a = len(docs[0])
        alone = m(docs[1][None].cuda())
    assert (out[0, a:a + len(docs[1])] - alone[0]).abs().max() < 1e-4
    assert (out - dense).abs().max() < 1e-4
    # fp16 autocast training-style forward with loss + backward
    m.train()
    x = torch.cat([row, row], 0)
    with torch.autocast("cuda", dtype=torch.float16):
        loss = m(x[:, :-1], x[:, 1:], docs=m.doc_inputs(x[:, :-1], EOS, 16))
    loss.backward()
    assert torch.isfinite(loss)


@cuda
@pytest.mark.parametrize("impl", ["sdpa", "kernel-fp16", "kernel-int8"])
@pytest.mark.parametrize("graph", [False, True])
def test_bigram_decode_cuda(impl, graph):
    m = _cuda_model(bigram_rows=1024, bigram_dim=64).half().eval()
    m.fuse_for_inference(torch.float16)
    m.decode_impl = impl
    tol = {"sdpa": 5e-2, "kernel-fp16": 5e-2, "kernel-int8": 0.3}[impl]
    _bigram_decode_check(m, "cuda", torch.float16, tol, use_graph=graph, compile_=graph and impl == "sdpa")


@cuda
def test_bigram_prefill_packed_matches_padded():
    m = _cuda_model(bigram_rows=1024).half().eval()
    m.fuse_for_inference(torch.float16)
    m.decode_impl = "kernel-fp16"
    m.prefill_packed = True
    _bigram_decode_check(m, "cuda", torch.float16, 5e-2, use_graph=True)


@cuda
def test_bigram_packed_trunk_matches_padded():
    """perf 'pack' path: segments packed with cu_seqlens reset the bigram at each start."""
    m = _cuda_model(bigram_rows=1024).eval()
    g = torch.Generator().manual_seed(2)
    seqs = [torch.randint(1, 512, (n,), generator=g) for n in (17, 33, 5)]
    x = torch.cat(seqs)[None].cuda()
    cu = torch.tensor([0, 17, 50, 55], dtype=torch.int32, device="cuda")
    pos = torch.cat([torch.arange(n) for n in (17, 33, 5)]).cuda()
    with torch.no_grad():
        h = m.trunk(x, pos=pos, cu_seqlens=cu, max_seqlen=33)[0]
        o = 0
        for s in seqs:
            ref = m.trunk(s[None].cuda())[0]
            assert (h[0, o:o + len(s)] - ref[0]).abs().max() < 1e-4
            o += len(s)


@cuda
def test_train_pt_smoke(tmp_path):
    rng = np.random.default_rng(0)
    for split, n in (("train", 3000), ("val", 300)):
        toks = []
        for _ in range(n):
            toks += rng.integers(1, 512, int(rng.integers(3, 150))).tolist() + [EOS]
        np.asarray(toks, dtype=np.uint16).tofile(tmp_path / f"{split}_0000.bin")
    json.dump({"vocab": 512, "dtype": "uint16", "eos_id": EOS, "train": {"bytes_per_token": 3.0},
               "val": {"bytes_per_token": 3.0}}, open(tmp_path / "meta.json", "w"))
    out = tmp_path / "run"
    cmd = [sys.executable, os.path.join(HERE, "..", "scripts", "train_pt.py"), "--data", str(tmp_path),
           "--out", str(out), "--name", "smoke", "--preset", "tiny", "--seq-len", "256", "--micro-bs", "4",
           "--batch-tokens", "2048", "--tokens", str(2048 * 6), "--val-every", "3", "--val-tokens", "4096",
           "--warmup", "2", "--wandb-mode", "disabled", "--doc-mask", "1", "--max-doc-len", "64",
           "--bigram-rows", "1024", "--bigram-dim", "32", "--ve-block-pos", "1"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    res = json.load(open(out / "result.json"))
    assert np.isfinite(res["final_val_loss"])
    s = torch.load(out / "final.pt", map_location="cpu", weights_only=False)
    assert s["cfg"]["bigram_rows"] == 1024 and s["cfg"]["ve_block_pos"] == [1]
    m = MathLM(MathConfig(**s["cfg"]))
    m.load_state_dict(s["model"])
    assert m.bigram_embed.weight.abs().sum() > 0  # the table trained
    print(r.stdout[-1500:])
