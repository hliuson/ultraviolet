"""Triton decode attention + int8 KV cache vs references (CUDA only)."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

if torch.cuda.is_available():
    from ultraviolet.kernels import (decode_attention, decode_attention_ref, dequantize_kv, kv_write,
                                            quantize_kv)
    from ultraviolet.kernels.fast_decode import patch_model, unpatch_model
    from ultraviolet.model import DecodeRunner, MathLM, generate, generate_continuous, make_config

DEV = "cuda"


def _case(B, KH, G, hd, S, int8, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    q = torch.randn(B, KH * G, hd, device=DEV, generator=g).half()
    k = torch.randn(B, KH, S, hd, device=DEV, generator=g).half()
    v = torch.randn(B, KH, S, hd, device=DEV, generator=g).half()
    if not int8:
        return q, k, v, None, None
    kq, ks = quantize_kv(k)
    vq, vs = quantize_kv(v)
    return q, kq, vq, ks, vs


BACKENDS = ["triton", "cuda"]


def _skip(backend, G, hd):
    if backend == "cuda" and not (hd in (64, 128) and G <= 8):
        pytest.skip("cuda kernel: hd 64/128, G <= 8")


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("int8", [False, True])
@pytest.mark.parametrize("B,KH,G,hd", [(1, 1, 1, 128), (3, 1, 4, 128), (17, 1, 5, 128), (5, 2, 4, 64),
                                       (9, 1, 8, 64), (4, 2, 2, 16), (6, 1, 7, 128)])
@pytest.mark.parametrize("splits", [1, 3, None])
def test_global_ragged(backend, int8, B, KH, G, hd, splits):
    _skip(backend, G, hd)
    S = 700
    q, k, v, ks, vs = _case(B, KH, G, hd, S, int8)
    pos = torch.randint(0, S, (B,), device=DEV)
    pos[0] = S - 1
    starts = (torch.rand(B, device=DEV) * (pos + 1).float()).long()  # left pad: start <= pos
    starts[0] = 0
    scale = 1.13 / hd ** 0.5
    out = decode_attention(q, k, v, pos, starts, scale, k_scale=ks, v_scale=vs, num_splits=splits, backend=backend)
    ref = decode_attention_ref(q, k, v, pos, starts, scale, k_scale=ks, v_scale=vs)
    torch.testing.assert_close(out.float(), ref.float(), atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("int8", [False, True])
@pytest.mark.parametrize("W,hd,G", [(128, 128, 5), (256, 64, 8), (1024, 128, 4), (100, 128, 4)])
@pytest.mark.parametrize("splits", [1, 4])
def test_ring_wraparound(backend, int8, W, hd, G, splits):
    B = 12
    q, k, v, ks, vs = _case(B, 1, G, hd, W, int8, seed=1)
    # positions before, at and well past the wrap point; some rows left-padded past the window start
    pos = torch.tensor([0, 5, W - 2, W - 1, W, W + 1, 2 * W + 7, 3 * W - 1, 5 * W + 3, W + 10, 4 * W, 7],
                       device=DEV)
    starts = torch.tensor([0, 2, 0, 3, 0, 4, 0, 2 * W + 5, 0, 30, 4 * W - 3, 7], device=DEV)
    scale = 0.1
    out = decode_attention(q, k, v, pos, starts, scale, window=W, k_scale=ks, v_scale=vs, num_splits=splits,
                           backend=backend)
    ref = decode_attention_ref(q, k, v, pos, starts, scale, window=W, k_scale=ks, v_scale=vs)
    torch.testing.assert_close(out.float(), ref.float(), atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("backend", BACKENDS)
def test_window_smaller_than_cache(backend):
    """Sliding window on a non-ring (full-length) cache."""
    B, S, W = 6, 500, 64
    q, k, v, _, _ = _case(B, 1, 4, 128, S, False, seed=2)
    pos = torch.tensor([10, 63, 64, 200, 499, 300], device=DEV)
    starts = torch.tensor([0, 0, 1, 180, 0, 290], device=DEV)
    out = decode_attention(q, k, v, pos, starts, 0.1, window=W, ring=False, num_splits=2, backend=backend)
    ref = decode_attention_ref(q, k, v, pos, starts, 0.1, window=W, ring=False)
    torch.testing.assert_close(out.float(), ref.float(), atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("backend", BACKENDS)
def test_matches_sdpa_fp16(backend):
    """fp16 kernel vs the model's SDPA formulation (additive mask, G query heads as rows)."""
    B, G, hd, S = 32, 5, 128, 2048
    q, k, v, _, _ = _case(B, 1, G, hd, S, False, seed=3)
    pos = torch.randint(S // 2, S, (B,), device=DEV)
    starts = torch.randint(0, 50, (B,), device=DEV)
    t = torch.arange(S, device=DEV)[None]
    bias = torch.zeros(B, S, device=DEV, dtype=torch.half).masked_fill(~((t <= pos[:, None]) & (t >= starts[:, None])),
                                                                      float("-inf"))
    bias = bias[:, None, None].expand(B, 1, G, S).contiguous()
    z = torch.nn.functional.scaled_dot_product_attention(q.view(B, 1, G, hd), k, v, attn_mask=bias, scale=0.1)
    out = decode_attention(q, k, v, pos, starts, 0.1, backend=backend)
    torch.testing.assert_close(out.view_as(z), z, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("backend", BACKENDS)
def test_non_contiguous_global_slice(backend):
    """The model passes cache[:, :, :g_len]-style views; strides must be honoured (and int8 + GQA K=2)."""
    B, KH, G, hd, S = 5, 2, 4, 128, 512
    q, k, v, ks, vs = _case(B, KH, G, hd, S, True, seed=4)
    g_len = 256
    pos = torch.tensor([0, 100, 255, 17, 200], device=DEV)
    starts = torch.tensor([0, 3, 0, 17, 150], device=DEV)
    kk, vv, kss, vss = k[:, :, :g_len], v[:, :, :g_len], ks[:, :, :g_len], vs[:, :, :g_len]
    out = decode_attention(q, kk, vv, pos, starts, 0.1, k_scale=kss, v_scale=vss, num_splits=2, backend=backend)
    ref = decode_attention_ref(q, k, v, pos, starts, 0.1, k_scale=ks, v_scale=vs)
    torch.testing.assert_close(out.float(), ref.float(), atol=2e-3, rtol=2e-3)


def test_quantize_roundtrip_and_write():
    x = torch.randn(64, 3, 128, device=DEV).half() * 3
    x[0, 0] = 0
    q, s = quantize_kv(x)
    err = (dequantize_kv(q, s).float() - x.float()).abs()
    assert err.max() <= (s.float().unsqueeze(-1) * 0.5 + 1e-3).max()
    assert (q[0, 0] == 0).all()
    cache = torch.zeros(4, 1, 16, 128, dtype=torch.int8, device=DEV)
    sc = torch.zeros(4, 1, 16, dtype=torch.half, device=DEV)
    idx = torch.tensor([0 * 16 + 3, 1 * 16 + 15, 2 * 16 + 0, 3 * 16 + 7], device=DEV)
    xv = torch.randn(4, 128, device=DEV).half()
    kv_write(cache, sc, idx, xv)
    back = dequantize_kv(cache.view(-1, 128)[idx], sc.view(-1)[idx])
    torch.testing.assert_close(back.float(), xv.float(), atol=0.03, rtol=0.03)


# ------------------------------------------------------------------ model-level
def _model(**kw):
    torch.manual_seed(0)
    cfg = make_config("tiny", **kw)
    m = MathLM(cfg)
    with torch.no_grad():
        for n, p in m.named_parameters():
            if n.endswith(("wo", "w_down", "ve_gate", "attn_gate", "xsa_alpha", "dense2")):
                p.normal_(0, 0.3)
    return m.to(DEV).half().fuse_for_inference(torch.float16)


def _teacher_forced_logits(m, prompts_idx, starts, cont, max_len):
    st = m.new_state(prompts_idx.size(0), max_len, torch.float16, DEV)
    with torch.autocast("cuda", dtype=torch.float16):
        outs = [m.prefill(prompts_idx, starts, st)]
    for t in range(cont.size(1)):
        outs.append(m.decode_step(cont[:, t], st, max_len))
    return torch.stack(outs, 1).float()


@pytest.mark.parametrize("xsa,hd,conv", [(False, 16, False), (True, 16, False), (True, 64, False), (False, 128, False),
                                         (True, 64, True)])
def test_patched_decode_matches_model(xsa, hd, conv):
    """Patched fp16 path == original SDPA path through ring wrap (tiny window 8) and left padding; int8 close.
    hd 16 runs the Triton kernel, hd 64/128 the CUDA one (backend auto). conv: layer 0 is a conv mixer (no KV)."""
    kw = {"conv_layers": [0]} if conv else {}
    m = _model(xsa=xsa, head_dim=hd, n_kv_heads=1, **kw)
    V = m.cfg.vocab_size
    B, P, N = 6, 5, 30
    idx = torch.randint(1, V, (B, P), device=DEV)
    starts = torch.tensor([0, 1, 2, 3, 4, 0], device=DEV)
    cont = torch.randint(1, V, (B, N), device=DEV)
    ref = _teacher_forced_logits(m, idx, starts, cont, P + N + 1)
    patch_model(m, "fp16")
    out = _teacher_forced_logits(m, idx, starts, cont, P + N + 1)
    torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)
    unpatch_model(m)
    patch_model(m, "int8")
    o8 = _teacher_forced_logits(m, idx, starts, cont, P + N + 1)
    assert (o8 - ref).abs().max() < 0.25
    assert (o8.argmax(-1) == ref.argmax(-1)).float().mean() > 0.9
    unpatch_model(m)


@pytest.mark.parametrize("hd", [16, 64])
def test_patched_graph_compiled_generate_matches_eager(hd):
    m = _model(head_dim=hd, n_kv_heads=1)
    prompts = [[1, 2, 3, 4, 5, 6], [7, 8, 9], [10, 11, 12, 13]]
    ref = generate(m, prompts, 25, temperature=0.0, use_graph=False, compile=False, bucket=8)
    patch_model(m, "fp16")
    out = generate(m, prompts, 25, temperature=0.0, use_graph=True, compile=True, bucket=8)
    assert out == ref
    cont = generate_continuous(m, prompts * 3, 20, batch=4, temperature=0.0, use_graph=True, compile=False,
                               bucket=8)
    unpatch_model(m)
    cref = generate_continuous(m, prompts * 3, 20, batch=4, temperature=0.0, use_graph=False, compile=False,
                               bucket=8)
    assert cont == cref
