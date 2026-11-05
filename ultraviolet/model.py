"""Small math LM based on Limite-1B-Violetto, tuned for batched-decode throughput.

Kept from Violetto (cheap): weightless pre-norm RMSNorm, QK-norm, GQA/MQA, SwiGLU, k:1 sliding-window (partial RoPE,
base 1024) : global (NoPE) layers, one shared gated value-embedding table on every 3rd layer, per-head sigmoid
attention output gate, per-layer residual/post lambdas, qkv/o scalar scales, sigmoid logit softcap
23*sigmoid((z+5)/7.5), tied embeddings. Optional: XSA (exclusive self-attention) and simplified MUDD residual mixing.

Window convention: a local layer's query at position i sees keys j with i - window < j <= i (exactly `window` keys,
incl. itself), so the local KV cache is a ring buffer of exactly `window` slots.

Inference: `DecodeState` holds a static per-layer KV cache (full length for global layers, a `window`-slot ring for
local layers); `generate()` does left-padded batched prefill + CUDA-graph decode (graphs bucketed by context length).
RoPE is relative, so a shared absolute position counter with left padding (pad keys masked) is exact.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# --------------------------------------------------------------------------------------------- config
@dataclass
class MathConfig:
    vocab_size: int = 32768
    d_model: int = 512
    n_layers: int = 16
    n_heads: int = 4
    n_kv_heads: int = 1
    head_dim: int = 128
    d_ff: int = 0                  # 0 -> 2.6 * d_model rounded to a multiple of 64
    window: int = 1024             # local layers: keys (i - window, i]
    global_every: int = 4          # layer l is global iff l % global_every == global_every - 1 (1: all global, 0: none)
    global_layers: list | None = None  # explicit override of global_every
    rope_frac: float = 0.5         # partial RoPE on local layers (fraction of head_dim rotated)
    rope_base: float = 1024.0
    global_rope: bool = False      # Violetto: globals are NoPE
    attn_scale: float = 0.0        # 0 -> 1.13 / sqrt(head_dim)  (= 0.1 at head_dim 128, Violetto's value)
    ve_every: int = 3              # value embeddings on layers l % ve_every == 1 (0: off)
    ve_gate_channels: int = 12
    attn_gate_channels: int = 128  # per-head output gate reads x[:, :C] (0: off)
    xsa: bool = False
    xsa_eps: float = 1e-4
    mudd: bool = False             # simplified MUDD at layers {L//2, L-1}, taps {0, l//2, l}
    mudd_inter: int = 32
    # token mixers for local layers: "attn" (sliding window) or "gdn" (Gated DeltaNet via fla); conv_layers lists
    # local layers whose attention is replaced by an attention-free gated short-conv mixer
    local_mixer: str = "attn"
    conv_layers: list | None = None
    conv_kernel: int = 4
    gdn_heads: int = 0             # 0 -> n_heads
    gdn_dk: int = 64
    gdn_dv: int = 128
    softcap: tuple = (23.0, 5.0, 7.5)
    init_std_embed: float = 0.02   # tied embedding init
    init_std_ve: float = 0.1
    # value-embedding layer pattern repeated in every global_every-sized block: layer l has a VE iff
    # l % global_every in ve_block_pos (None: the l % ve_every == 1 rule). uv-50m-alpha: [1, 4] -> layers 1, 4, 6, 9
    ve_block_pos: list | None = None
    # hashed bigram embedding (modded-nanogpt PR #201): row (36313 t ^ 27191 t_prev) % (bigram_rows - 1), row
    # bigram_rows - 1 reserved for "no previous token" (sequence / document start, left pad). Zero-init table added
    # to the residual before every layer with per-layer learned lambdas (init bigram_lambda_init). 0: off.
    bigram_rows: int = 0
    bigram_dim: int = 0            # 0 -> d_model; < d_model: table of this width + a (d_model, bigram_dim) projection
    bigram_lambda_init: float = 0.1

    def __post_init__(self):
        if not self.d_ff:
            self.d_ff = int(round(2.6 * self.d_model / 64) * 64)
        assert self.n_heads % self.n_kv_heads == 0
        if not self.attn_scale:
            self.attn_scale = 1.13 / math.sqrt(self.head_dim)
        self.softcap = tuple(self.softcap)
        if self.global_layers is not None:
            self.global_layers = sorted(int(x) for x in self.global_layers)
        if self.ve_block_pos is not None:
            assert self.global_every > 0, "ve_block_pos needs global_every > 0 (the block period)"
            self.ve_block_pos = sorted(int(x) for x in self.ve_block_pos)
        if self.bigram_rows:
            assert self.bigram_rows >= 2
            if not self.bigram_dim:
                self.bigram_dim = self.d_model
            assert self.bigram_dim <= self.d_model

    def is_global(self, l: int) -> bool:
        if self.global_layers is not None:
            return l in self.global_layers
        if self.global_every <= 0:
            return False
        return l % self.global_every == self.global_every - 1

    def mixer(self, l: int) -> str:
        if self.is_global(l):
            return "attn"
        if self.conv_layers and l in self.conv_layers:
            return "conv"
        return self.local_mixer

    def has_ve(self, l: int) -> bool:
        if self.ve_block_pos is not None:
            return l % self.global_every in self.ve_block_pos
        return self.ve_every > 0 and l % self.ve_every == 1 % self.ve_every

    def bigram_params(self) -> int:
        if not self.bigram_rows:
            return 0
        bd = self.bigram_dim
        return self.bigram_rows * bd + (self.d_model * bd if bd < self.d_model else 0) + self.n_layers

    @property
    def mudd_sites(self) -> list[int]:
        return sorted({self.n_layers // 2, self.n_layers - 1}) if self.mudd else []

    def mudd_taps(self, l: int) -> list[int]:
        return [0, l // 2, l]

    def to_dict(self) -> dict:
        return asdict(self)

    # ---- analytic sizes
    def param_counts(self) -> dict:
        d, hd = self.d_model, self.head_dim
        attn = d * hd * (2 * self.n_heads + 2 * self.n_kv_heads)
        mlp = 3 * d * self.d_ff
        H = self.gdn_heads or self.n_heads
        mix = {"attn": attn, "conv": 3 * d * d, "gdn": d * H * (2 * self.gdn_dk + 3 * self.gdn_dv)}
        nonemb = sum(mix[self.mixer(l)] + mlp for l in range(self.n_layers))
        embed = self.vocab_size * d
        ve = self.vocab_size * self.n_kv_heads * hd if any(self.has_ve(l) for l in range(self.n_layers)) else 0
        out = {"non_embedding": nonemb, "embedding_tied": embed, "value_embedding": ve,
               "total_with_tied_embed": nonemb + embed, "total_incl_ve": nonemb + embed + ve}
        if self.bigram_rows:  # table + projection + lambdas (keys only present when on)
            bg = self.bigram_params()
            out.update(bigram=bg, total_incl_bigram=nonemb + embed + ve + bg)
        return out

    def kv_bytes_per_seq(self, ctx: int, dtype_bytes: int = 2) -> int:
        """KV-cache bytes for one sequence at context `ctx` (local layers capped at window)."""
        per_pos = 2 * self.n_kv_heads * self.head_dim * dtype_bytes
        tot = 0
        for l in range(self.n_layers):
            mx = self.mixer(l)
            if mx == "attn":
                tot += per_pos * (ctx if self.is_global(l) else min(ctx, self.window))
            elif mx == "conv":
                tot += (self.conv_kernel - 1) * self.d_model * dtype_bytes
            else:  # gdn: fp32 state + conv tails
                H = self.gdn_heads or self.n_heads
                tot += H * self.gdn_dk * self.gdn_dv * 4 + (self.conv_kernel - 1) * H * (2 * self.gdn_dk +
                                                                                         self.gdn_dv) * dtype_bytes
        return tot

    def kv_bytes_per_token_global(self, dtype_bytes: int = 2) -> int:
        """Marginal KV bytes per extra context token beyond the window (only global layers grow)."""
        ng = sum(self.is_global(l) for l in range(self.n_layers))
        return 2 * self.n_kv_heads * self.head_dim * dtype_bytes * ng


# Candidate ~50M configs (vocab 32k). Non-embedding ~40-44M; tied embedding ~16-21M.
PRESETS: dict[str, dict] = {
    # Violetto shape scaled down: deep/narrow, GQA 4q/2kv, hd 128, 3 local : 1 global.
    "v-deep-gqa": dict(d_model=512, n_layers=16, n_heads=4, n_kv_heads=2, head_dim=128, d_ff=1280, global_every=4),
    # same but MQA (1 kv head) -> half the KV cache.
    "v-deep-mqa": dict(d_model=512, n_layers=16, n_heads=4, n_kv_heads=1, head_dim=128, d_ff=1344, global_every=4),
    # same as deep-mqa with only 2 global layers (7:1).
    "v-deep-mqa-g8": dict(d_model=512, n_layers=16, n_heads=4, n_kv_heads=1, head_dim=128, d_ff=1344, global_every=8),
    # shallower / wider, MQA, 4 local : 1 global (2 globals).
    "wide-mqa": dict(d_model=640, n_layers=10, n_heads=5, n_kv_heads=1, head_dim=128, d_ff=1664, global_every=5),
    # main config: wide-mqa with 8 x 64 query heads (G = 8 is the decode kernel's limit), window 256, XSA
    "uv-50m-alpha": dict(d_model=640, n_layers=10, n_heads=8, n_kv_heads=1, head_dim=64, d_ff=1664, global_every=5,
                      window=256, xsa=True),
    # tiny config for tests
    "tiny": dict(vocab_size=512, d_model=64, n_layers=4, n_heads=4, n_kv_heads=2, head_dim=16, d_ff=128,
                 window=8, global_every=2, attn_gate_channels=16, ve_gate_channels=4),
}


def make_config(name: str, **over) -> MathConfig:
    kw = dict(PRESETS[name])
    kw.update({k: v for k, v in over.items() if v is not None})
    return MathConfig(**kw)


# --------------------------------------------------------------------------------------------- helpers
def rms_norm(x: Tensor) -> Tensor:
    return F.rms_norm(x, (x.size(-1),))


def _scalar(v: float) -> nn.Parameter:
    return nn.Parameter(torch.tensor(float(v)))


class Rotary(nn.Module):
    """Partial RoPE: rotates the first 2*n_pairs dims (half-split pairing); frequencies (1/base)^linspace(0,1)."""

    def __init__(self, head_dim: int, frac: float, base: float):
        super().__init__()
        self.n_pairs = int(head_dim * frac) // 2
        inv = (1.0 / base) ** torch.linspace(0, 1, self.n_pairs, dtype=torch.float32)
        self.register_buffer("inv_freq", inv, persistent=False)
        self._base = base

    def _apply(self, fn, recurse=True):  # keep frequencies fp32 under .half()/.to(dtype)
        super()._apply(fn, recurse)
        self.inv_freq = ((1.0 / self._base) ** torch.linspace(0, 1, self.n_pairs, dtype=torch.float32)).to(
            self.inv_freq.device)
        return self

    def cos_sin(self, pos: Tensor) -> tuple[Tensor, Tensor]:
        th = pos.float()[..., None] * self.inv_freq  # (T, n_pairs)
        return th.cos(), th.sin()


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor, n_pairs: int) -> Tensor:
    """x: (B, T, H, hd); cos/sin: (T, n_pairs), (B, 1, n_pairs) (decode) or (B, T, n_pairs) (per-row positions)."""
    if n_pairs == 0:
        return x
    c = cos.unsqueeze(-2).to(x.dtype)  # == cos[:, None, :] for 2-D cos and for (B, 1, n_pairs)
    s = sin.unsqueeze(-2).to(x.dtype)
    x1, x2, rest = x[..., :n_pairs], x[..., n_pairs:2 * n_pairs], x[..., 2 * n_pairs:]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c, rest], dim=-1)


BIGRAM_MULS = (36313, 27191)  # x[t], x[t-1] (modded-nanogpt)


def bigram_index(tok: Tensor, prev: Tensor, no_prev: Tensor, rows: int) -> Tensor:
    """Hashed bigram row of (prev, tok): (36313 tok ^ 27191 prev) % (rows - 1); rows - 1 where no_prev. int64
    arithmetic (exact; equals the speedrun's int32 hash for vocab < 59139)."""
    h = torch.bitwise_xor(BIGRAM_MULS[0] * tok, BIGRAM_MULS[1] * prev) % (rows - 1)
    return torch.where(no_prev, torch.full_like(h, rows - 1), h)


def doc_layout(idx: Tensor, eos_id: int = 0, max_doc_len: int = 0) -> tuple[Tensor, Tensor]:
    """Document segmentation of (B, T) token rows for document-masked training. A segment starts at t = 0, at
    every EOS (the EOS separating two docs is the BOS of the next one, matching inference prompts [EOS] + prompt),
    and every max_doc_len tokens inside a long doc (split chunks start fresh segments). Returns (starts (B, T) bool,
    pos (B, T) long = position within the segment)."""
    B, T = idx.shape
    t = torch.arange(T, device=idx.device)
    st = idx == eos_id
    st[:, 0] = True
    last = torch.cummax(torch.where(st, t, torch.zeros_like(t)), dim=1).values
    pd = t - last
    if max_doc_len and max_doc_len < T:
        st = st | (pd % max_doc_len == 0)
        pd = pd % max_doc_len
    return st, pd


# --------------------------------------------------------------------------------------------- layers
class Attention(nn.Module):
    def __init__(self, cfg: MathConfig, l: int):
        super().__init__()
        self.cfg, self.l = cfg, l
        d, hd, H, K = cfg.d_model, cfg.head_dim, cfg.n_heads, cfg.n_kv_heads
        self.H, self.K, self.G, self.hd = H, K, H // K, hd
        self.is_global = cfg.is_global(l)
        self.rope = (not self.is_global) or cfg.global_rope
        self.has_ve = cfg.has_ve(l)
        self.wq = nn.Parameter(torch.empty(H * hd, d))
        self.wk = nn.Parameter(torch.empty(K * hd, d))
        self.wv = nn.Parameter(torch.empty(K * hd, d))
        self.wo = nn.Parameter(torch.zeros(d, H * hd))
        self.wv._normuon_red_dim = -1  # Violetto: W_v always normalises per output row
        self.qkv_scale = _scalar(1.0)
        self.o_scale = _scalar(1.0)
        if self.has_ve:
            self.ve_gate = nn.Parameter(torch.zeros(K, cfg.ve_gate_channels))
        self.gate_ch = min(cfg.attn_gate_channels, d)
        if self.gate_ch:
            self.attn_gate = nn.Parameter(torch.zeros(H, self.gate_ch))
        if cfg.xsa:
            self.xsa_alpha = nn.Parameter(torch.zeros(H))
        self._fused: dict | None = None

    def reset_parameters(self):
        d = self.cfg.d_model
        bound = math.sqrt(3) * 0.5 / math.sqrt(d)
        for w in (self.wq, self.wk, self.wv):
            nn.init.uniform_(w, -bound, bound)
        nn.init.zeros_(self.wo)

    def fuse(self, dtype):
        """Cache scaled/fused weights for inference (call after training; invalidated by .train())."""
        with torch.no_grad():
            f = {"wqkv": (torch.cat([self.wq, self.wk, self.wv]) * self.qkv_scale).to(dtype),
                 "wo": (self.wo * self.o_scale).to(dtype)}
            if self.has_ve:
                f["ve_gate"] = self.ve_gate.to(dtype)
            if self.gate_ch:
                f["attn_gate"] = self.attn_gate.to(dtype)
            if self.cfg.xsa:
                f["xsa"] = torch.tanh(self.xsa_alpha.float())
        self._fused = f

    def qkv(self, xn: Tensor, ve: Tensor | None, cos: Tensor, sin: Tensor, rope_pairs: int):
        B, T, _ = xn.shape
        H, K, hd = self.H, self.K, self.hd
        if self._fused is not None:
            w = self._fused["wqkv"]
        else:
            w = torch.cat([self.wq, self.wk, self.wv]) * self.qkv_scale
        qkv = F.linear(xn, w.to(xn.dtype))
        q, k, v = qkv.split([H * hd, K * hd, K * hd], dim=-1)
        q, k, v = q.view(B, T, H, hd), k.view(B, T, K, hd), v.view(B, T, K, hd)
        if self.has_ve and ve is not None:
            gw = self._fused["ve_gate"] if self._fused is not None else self.ve_gate
            g = 2.0 * torch.sigmoid(F.linear(xn[..., :gw.size(-1)], gw.to(xn.dtype)))  # (B, T, K)
            v = v + g.unsqueeze(-1) * ve.to(v.dtype)
        q, k = rms_norm(q), rms_norm(k)
        if self.rope:
            q, k = apply_rope(q, cos, sin, rope_pairs), apply_rope(k, cos, sin, rope_pairs)
        return q, k, v

    def post(self, z: Tensor, xn: Tensor, v: Tensor) -> Tensor:
        """z: (B, T, H, hd) attention output; v: (B, T, K, hd) current-token values."""
        B, T = z.shape[:2]
        if self.cfg.xsa:
            a = self._fused["xsa"] if self._fused is not None else torch.tanh(self.xsa_alpha.float())
            zg = z.view(B, T, self.K, self.G, self.hd).float()
            vd = F.normalize(v.float(), dim=-1, eps=self.cfg.xsa_eps).unsqueeze(3)
            proj = (zg * vd).sum(-1, keepdim=True)
            z = (zg - a.view(1, 1, self.K, self.G, 1) * proj * vd).to(z.dtype).view(B, T, self.H, self.hd)
        if self.gate_ch:
            gw = self._fused["attn_gate"] if self._fused is not None else self.attn_gate
            g = 2.0 * torch.sigmoid(F.linear(xn[..., :self.gate_ch], gw.to(xn.dtype)))
            z = z * g.unsqueeze(-1).to(z.dtype)
        wo = self._fused["wo"] if self._fused is not None else self.wo * self.o_scale
        return F.linear(z.reshape(B, T, -1), wo.to(z.dtype))

    kind = "attn"

    def forward(self, xn, ve, cos, sin, rope_pairs, mask, valid=None):
        """Full-sequence attention (training / prefill). mask: None (causal) or bool (T,T)/(B,1,T,T). Returns
        output and (k, v) for caching."""
        q, k, v = self.qkv(xn, ve, cos, sin, rope_pairs)
        if isinstance(mask, tuple) and mask[0] == "varlen":  # ("varlen", cu_seqlens, max_seqlen, window|None)
            from .kernels.varlen_attn import varlen_attention
            B, T = q.shape[:2]
            # SDPA is autocast-listed (casts q/k/v to fp16); the raw kernel is not, and rms_norm returns fp32 under
            # autocast, so cast explicitly
            dt = torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else xn.dtype
            z = varlen_attention(q.reshape(1, B * T, self.H, self.hd).to(dt), k.reshape(1, B * T, self.K, self.hd).to(dt),
                                 v.reshape(1, B * T, self.K, self.hd).to(dt), mask[1], mask[2],
                                 self.cfg.attn_scale, mask[3])
            return self.post(z.view(B, T, self.H, self.hd), xn, v), (k, v)
        qt, kt, vt = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if isinstance(mask, tuple):  # ("flex", BlockMask): block-sparse causal / sliding-window, native GQA
            z = _flex_attention()(qt, kt.to(qt.dtype), vt.to(qt.dtype), block_mask=mask[1],
                                  scale=self.cfg.attn_scale, enable_gqa=self.G > 1)
            return self.post(z.transpose(1, 2), xn, v), (k, v)
        if self.G > 1:
            kt, vt = kt.repeat_interleave(self.G, dim=1), vt.repeat_interleave(self.G, dim=1)
        if mask is None:
            z = F.scaled_dot_product_attention(qt, kt, vt, is_causal=True, scale=self.cfg.attn_scale)
        else:
            z = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=mask, scale=self.cfg.attn_scale)
        z = z.transpose(1, 2)
        return self.post(z, xn, v), (k, v)


class MLP(nn.Module):
    def __init__(self, cfg: MathConfig):
        super().__init__()
        d, f = cfg.d_model, cfg.d_ff
        self.w_gate = nn.Parameter(torch.empty(f, d))
        self.w_up = nn.Parameter(torch.empty(f, d))
        self.w_down = nn.Parameter(torch.zeros(d, f))
        self._fused = None

    def reset_parameters(self):
        d = self.w_gate.size(1)
        bound = math.sqrt(3) * 0.5 / math.sqrt(d)
        nn.init.uniform_(self.w_gate, -bound, bound)
        nn.init.uniform_(self.w_up, -bound, bound)
        nn.init.zeros_(self.w_down)

    def fuse(self, dtype):
        with torch.no_grad():
            self._fused = {"gu": torch.cat([self.w_gate, self.w_up]).to(dtype), "down": self.w_down.to(dtype)}

    def forward(self, x):
        if self._fused is not None:
            g, u = F.linear(x, self._fused["gu"]).chunk(2, dim=-1)
            return F.linear(F.silu(g) * u, self._fused["down"])
        return F.linear(F.silu(F.linear(x, self.w_gate.to(x.dtype))) * F.linear(x, self.w_up.to(x.dtype)),
                        self.w_down.to(x.dtype))


# --------------------------------------------------------------------------------------------- non-attention mixers
class ShortConv(nn.Module):
    """Depthwise causal conv over time (kernel K). Decode state = the last K-1 inputs per row."""

    def __init__(self, C: int, K: int = 4):
        super().__init__()
        self.K = K
        self.w = nn.Parameter(torch.zeros(C, K))

    def reset_parameters(self):
        nn.init.normal_(self.w, std=1.0 / math.sqrt(self.K))
        with torch.no_grad():
            self.w[:, -1] += 1.0  # start near identity

    def forward(self, u: Tensor) -> tuple[Tensor, Tensor]:
        B, T, C = u.shape
        x = F.pad(u.transpose(1, 2), (self.K - 1, 0))
        y = F.conv1d(x, self.w[:, None, :].to(u.dtype), groups=C).transpose(1, 2)
        tail = x[:, :, -(self.K - 1):].transpose(1, 2)  # last K-1 inputs (zero-padded if T < K-1)
        return y, tail

    def step(self, u: Tensor, state: Tensor) -> Tensor:
        """u (B, 1, C); state (B, K-1, C) updated in place."""
        win = torch.cat([state, u.to(state.dtype)], 1)
        y = (win * self.w.t()[None].to(win.dtype)).sum(1, keepdim=True)
        state.copy_(win[:, 1:])
        return y.to(u.dtype)


class ConvMixer(nn.Module):
    """Attention-free token mixer: y = W_o(silu(dwconv_K(W_in x)) * silu(W_gate x)). ~3 d^2 params."""
    kind = "conv"

    def __init__(self, cfg: MathConfig, l: int):
        super().__init__()
        d = cfg.d_model
        self.is_global, self.has_ve = False, False
        self.w_in = nn.Parameter(torch.empty(d, d))
        self.wgate = nn.Parameter(torch.empty(d, d))
        self.wo = nn.Parameter(torch.zeros(d, d))
        self.conv = ShortConv(d, cfg.conv_kernel)

    def reset_parameters(self):
        bound = math.sqrt(3) * 0.5 / math.sqrt(self.w_in.size(1))
        nn.init.uniform_(self.w_in, -bound, bound)
        nn.init.uniform_(self.wgate, -bound, bound)
        nn.init.zeros_(self.wo)
        self.conv.reset_parameters()

    def forward(self, xn, ve, cos, sin, rope_pairs, mask, valid=None):
        u = F.linear(xn, self.w_in.to(xn.dtype))
        if valid is not None:
            u = u * valid[..., None].to(u.dtype)
        y, tail = self.conv(u)
        y = F.silu(y) * F.silu(F.linear(xn, self.wgate.to(xn.dtype)))
        return F.linear(y, self.wo.to(y.dtype)), {"conv": tail}

    def new_state(self, B, dtype, device):
        return {"conv": torch.zeros(B, self.conv.K - 1, self.w_in.size(0), dtype=dtype, device=device)}

    def step(self, xn, state):
        u = F.linear(xn, self.w_in.to(xn.dtype))
        y = F.silu(self.conv.step(u, state["conv"])) * F.silu(F.linear(xn, self.wgate.to(xn.dtype)))
        return F.linear(y, self.wo.to(y.dtype))


class GDNMixer(nn.Module):
    """Gated DeltaNet token mixer (Yang et al. 2024; fla layout): short-conv'd q/k/v (SiLU), L2-normed q/k,
    beta = sigmoid(W_b x), log-decay g = -exp(A_log) * softplus(W_a x + dt_bias), gated RMS output norm.
    Training/prefill use fla's chunk kernel (Triton; runs on sm75 fp16); decode is a torch recurrent step with a
    constant (B, H, dk, dv) fp32 state + conv tails."""
    kind = "gdn"

    def __init__(self, cfg: MathConfig, l: int):
        super().__init__()
        d = cfg.d_model
        H, dk, dv = cfg.gdn_heads or cfg.n_heads, cfg.gdn_dk, cfg.gdn_dv
        self.H, self.dk, self.dv = H, dk, dv
        self.is_global, self.has_ve = False, False
        self.wq = nn.Parameter(torch.empty(H * dk, d))
        self.wk = nn.Parameter(torch.empty(H * dk, d))
        self.wv = nn.Parameter(torch.empty(H * dv, d))
        self.wgate = nn.Parameter(torch.empty(H * dv, d))
        self.wo = nn.Parameter(torch.zeros(d, H * dv))
        self.w_a = nn.Parameter(torch.zeros(H, d))
        self.w_b = nn.Parameter(torch.zeros(H, d))
        self.A_log = nn.Parameter(torch.zeros(H))
        self.dt_bias = nn.Parameter(torch.zeros(H))
        self.conv_q, self.conv_k, self.conv_v = (ShortConv(H * dk, cfg.conv_kernel), ShortConv(H * dk, cfg.conv_kernel),
                                                 ShortConv(H * dv, cfg.conv_kernel))

    def reset_parameters(self):
        d = self.wq.size(1)
        bound = math.sqrt(3) * 0.5 / math.sqrt(d)
        for w in (self.wq, self.wk, self.wv, self.wgate):
            nn.init.uniform_(w, -bound, bound)
        nn.init.zeros_(self.wo)
        nn.init.normal_(self.w_a, std=0.02)
        nn.init.normal_(self.w_b, std=0.02)
        with torch.no_grad():
            self.A_log.copy_(torch.empty_like(self.A_log).uniform_(1, 16).log())
            dt = torch.exp(torch.empty_like(self.dt_bias).uniform_(math.log(1e-3), math.log(1e-1))).clamp_min(1e-4)
            self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus
        for c in (self.conv_q, self.conv_k, self.conv_v):
            c.reset_parameters()

    def _gates(self, xn):
        beta = torch.sigmoid(F.linear(xn.float(), self.w_b.float()))
        g = -self.A_log.float().exp() * F.softplus(F.linear(xn.float(), self.w_a.float()) + self.dt_bias.float())
        return g, beta

    def forward(self, xn, ve, cos, sin, rope_pairs, mask, valid=None):
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        B, T, _ = xn.shape
        H, dk, dv = self.H, self.dk, self.dv
        dt = xn.dtype if xn.dtype in (torch.float16, torch.bfloat16) else torch.float32
        uq, uk, uv = (F.linear(xn, w.to(xn.dtype)) for w in (self.wq, self.wk, self.wv))
        g, beta = self._gates(xn)
        if valid is not None:  # left padding: pads contribute nothing and do not decay the state
            m = valid[..., None].to(uq.dtype)
            uq, uk, uv = uq * m, uk * m, uv * m
            g, beta = g * valid[..., None], beta * valid[..., None]
        (q, tq), (k, tk), (v, tv) = self.conv_q(uq), self.conv_k(uk), self.conv_v(uv)
        q = F.normalize(F.silu(q).view(B, T, H, dk).float(), dim=-1).to(dt)
        k = F.normalize(F.silu(k).view(B, T, H, dk).float(), dim=-1).to(dt)
        v = F.silu(v).view(B, T, H, dv).to(dt)
        if valid is not None:
            k = k * valid[..., None, None].to(k.dtype)
        o, S = chunk_gated_delta_rule(q, k, v, g, beta.to(dt), scale=dk ** -0.5, output_final_state=True)
        o = rms_norm(o.float()).to(xn.dtype if xn.dtype != torch.float32 else torch.float32)
        o = o.reshape(B, T, H * dv) * F.silu(F.linear(xn, self.wgate.to(xn.dtype))).to(o.dtype)
        return F.linear(o, self.wo.to(o.dtype)), {"q": tq, "k": tk, "v": tv, "S": S.float()}

    def new_state(self, B, dtype, device):
        H, dk, dv, K = self.H, self.dk, self.dv, self.conv_q.K
        return {"q": torch.zeros(B, K - 1, H * dk, dtype=dtype, device=device),
                "k": torch.zeros(B, K - 1, H * dk, dtype=dtype, device=device),
                "v": torch.zeros(B, K - 1, H * dv, dtype=dtype, device=device),
                "S": torch.zeros(B, H, dk, dv, dtype=torch.float32, device=device)}

    def step(self, xn, state):
        B = xn.size(0)
        H, dk, dv = self.H, self.dk, self.dv
        q = self.conv_q.step(F.linear(xn, self.wq.to(xn.dtype)), state["q"])
        k = self.conv_k.step(F.linear(xn, self.wk.to(xn.dtype)), state["k"])
        v = self.conv_v.step(F.linear(xn, self.wv.to(xn.dtype)), state["v"])
        q = F.normalize(F.silu(q).view(B, H, dk).float(), dim=-1) * dk ** -0.5
        k = F.normalize(F.silu(k).view(B, H, dk).float(), dim=-1)
        v = F.silu(v).view(B, H, dv).float()
        g, beta = self._gates(xn)
        g, beta = g.view(B, H), beta.view(B, H)
        S = state["S"] * g.exp()[..., None, None]
        pred = torch.einsum("bhk,bhkv->bhv", k, S)
        S = S + beta[..., None, None] * k[..., :, None] * (v - pred)[..., None, :]
        o = torch.einsum("bhk,bhkv->bhv", q, S)
        state["S"].copy_(S)
        o = rms_norm(o).to(xn.dtype).reshape(B, 1, H * dv) * F.silu(F.linear(xn, self.wgate.to(xn.dtype)))
        return F.linear(o, self.wo.to(o.dtype))


class Block(nn.Module):
    def __init__(self, cfg: MathConfig, l: int):
        super().__init__()
        mx = cfg.mixer(l)
        self.attn = {"attn": Attention, "conv": ConvMixer, "gdn": GDNMixer}[mx](cfg, l)
        self.mlp = MLP(cfg)
        self.resid_attn = _scalar(1.0)
        self.post_attn = _scalar(1.0)
        self.resid_mlp = _scalar(1.0)
        self.post_mlp = _scalar(1.0)

    def finish(self, resid_base: Tensor, a: Tensor) -> Tensor:
        dt = a.dtype
        h = self.resid_attn.to(dt) * resid_base + self.post_attn.to(dt) * a
        m = self.mlp(rms_norm(h))
        return self.resid_mlp.to(dt) * h + self.post_mlp.to(dt) * m


class Mudd(nn.Module):
    """Simplified MUDD (Violetto sec. 2.4): at site l, attention input and residual base become token-dependent
    mixes of taps {0, l//2, l}; identity at init (W2 = 0, bias selects the current state)."""

    def __init__(self, cfg: MathConfig):
        super().__init__()
        self.sites = cfg.mudd_sites
        n = len(self.sites)
        self.dense1 = nn.Parameter(torch.empty(cfg.mudd_inter, cfg.d_model))
        self.dense2 = nn.Parameter(torch.zeros(n, 2, 3, cfg.mudd_inter))  # [site, {attn,res}, tap, inter]
        b = torch.zeros(n, 2, 3)
        b[..., 2] = 1.0
        self.bias = nn.Parameter(b)

    def reset_parameters(self):
        nn.init.normal_(self.dense1, std=1.0 / math.sqrt(self.dense1.size(1)))
        nn.init.zeros_(self.dense2)
        with torch.no_grad():
            self.bias.zero_()
            self.bias[..., 2] = 1.0

    def forward(self, taps: list[Tensor], cur: Tensor, site: int) -> tuple[Tensor, Tensor]:
        i = self.sites.index(site)
        dt = cur.dtype
        h = F.gelu(F.linear(rms_norm(cur), self.dense1.to(dt)))
        out = []
        for r in range(2):
            w = F.linear(h, self.dense2[i, r].to(dt)) + self.bias[i, r].to(dt)  # (B, T, 3)
            mix = w[..., 0:1] * taps[0]
            for j in range(1, len(taps)):
                mix = mix + w[..., j:j + 1] * taps[j]
            out.append(mix)
        return out[0], out[1]


# --------------------------------------------------------------------------------------------- model
class MathLM(nn.Module):
    def __init__(self, cfg: MathConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.use_ve = any(cfg.has_ve(l) for l in range(cfg.n_layers))
        if self.use_ve:
            self.value_embed = nn.Embedding(cfg.vocab_size, cfg.n_kv_heads * cfg.head_dim)
        self.blocks = nn.ModuleList([Block(cfg, l) for l in range(cfg.n_layers)])
        self.rotary = Rotary(cfg.head_dim, cfg.rope_frac, cfg.rope_base)
        self.mudd = Mudd(cfg) if cfg.mudd else None
        if cfg.bigram_rows:
            self.bigram_embed = nn.Embedding(cfg.bigram_rows, cfg.bigram_dim)
            self.bigram_lambdas = nn.Parameter(torch.full((cfg.n_layers,), float(cfg.bigram_lambda_init)))
            if cfg.bigram_dim < cfg.d_model:
                self.bigram_proj = nn.Parameter(torch.empty(cfg.d_model, cfg.bigram_dim))
        self.retained = sorted({t for s in cfg.mudd_sites for t in cfg.mudd_taps(s) if t != s})
        self._mask_cache: dict = {}
        self.attn_impl = "sdpa"  # "flex": FlexAttention block-sparse masks for training (CUDA; set by trainer)
        # perf flag (docs/perf_opt.md): "varlen" = training/full-sequence attention through the CUTLASS
        # memory-efficient kernel with cu_seqlens + window_size (local layers read only their window; packed
        # sequences via trunk(..., cu_seqlens=)). Default "sdpa" keeps the original path.
        self.train_attn = "sdpa"
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embed.weight, std=self.cfg.init_std_embed)
        if self.use_ve:
            nn.init.normal_(self.value_embed.weight, std=self.cfg.init_std_ve)
        for b in self.blocks:
            b.attn.reset_parameters()
            b.mlp.reset_parameters()
        if self.mudd is not None:
            self.mudd.reset_parameters()
        if self.cfg.bigram_rows:
            nn.init.zeros_(self.bigram_embed.weight)
            with torch.no_grad():
                self.bigram_lambdas.fill_(float(self.cfg.bigram_lambda_init))
            if self.cfg.bigram_dim < self.cfg.d_model:
                bound = math.sqrt(3) * 0.5 / math.sqrt(self.cfg.bigram_dim)
                nn.init.uniform_(self.bigram_proj, -bound, bound)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            for b in self.blocks:
                if b.attn.kind == "attn":
                    b.attn._fused = None
                b.mlp._fused = None
        return self

    def fuse_for_inference(self, dtype=torch.float16):
        self.eval()
        for b in self.blocks:
            if b.attn.kind == "attn":
                b.attn.fuse(dtype)
            b.mlp.fuse(dtype)
        return self

    def num_params(self) -> dict:
        return self.cfg.param_counts()

    # ---- masks
    def _local_mask(self, T: int, device) -> Tensor:
        key = ("local", T, str(device))
        m = self._mask_cache.get(key)
        if m is None:
            i = torch.arange(T, device=device)
            d = i[:, None] - i[None, :]
            m = (d >= 0) & (d < self.cfg.window)
            self._mask_cache[key] = m
        return m

    def _block_mask(self, T: int, window: int | None, device):
        key = ("flex", T, window, str(device))
        m = self._mask_cache.get(key)
        if m is None:
            from torch.nn.attention.flex_attention import create_block_mask
            if window is None:
                def mod(b, h, q, k):
                    return q >= k
            else:
                def mod(b, h, q, k):
                    return (q >= k) & (q - k < window)
            m = create_block_mask(mod, None, None, T, T, device=device)
            self._mask_cache[key] = m
        return m

    def _value_embeds(self, idx: Tensor) -> Tensor | None:
        if not self.use_ve:
            return None
        return self.value_embed(idx).view(*idx.shape, self.cfg.n_kv_heads, self.cfg.head_dim)

    def _bigram_vec(self, bid: Tensor) -> Tensor:
        e = self.bigram_embed(bid)
        if self.cfg.bigram_dim < self.cfg.d_model:
            e = F.linear(e, self.bigram_proj.to(e.dtype))
        return e

    def _bigram_embeds(self, idx: Tensor, no_prev: Tensor) -> Tensor | None:
        """idx (B, T); no_prev (B, T) bool marks tokens without a previous token in their sequence / document."""
        if not self.cfg.bigram_rows:
            return None
        prev = torch.cat([idx[:, :1], idx[:, :-1]], 1)
        return self._bigram_vec(bigram_index(idx, prev, no_prev, self.cfg.bigram_rows))

    def _bigram_step(self, tok: Tensor, st: "DecodeState") -> Tensor | None:
        """Decode: bigram of (st.prev, tok) for tok (B,), then st.prev <- tok (in place; graph-safe)."""
        if not self.cfg.bigram_rows:
            return None
        bid = bigram_index(tok, st.prev.clamp_min(0), st.prev < 0, self.cfg.bigram_rows)
        st.prev.copy_(tok)
        return self._bigram_vec(bid).unsqueeze(1)

    def _inject(self, x: Tensor, xb: Tensor | None, l: int) -> Tensor:
        if xb is None:
            return x
        return x + self.bigram_lambdas[l].to(x.dtype) * xb.to(x.dtype)

    def doc_inputs(self, idx: Tensor, eos_id: int = 0, max_doc_len: int = 0, packed: bool | None = None) -> dict:
        """Document-masking inputs for forward(..., docs=). packed (default: on CUDA) runs the rows as one (1, B*T)
        varlen batch with cu_seqlens at the segment starts (needs CUDA); otherwise dense (B, 1, T, T) masks."""
        B, T = idx.shape
        starts, pos = doc_layout(idx, eos_id, max_doc_len)
        d = {"starts": starts, "pos": pos, "cu": None,
             "max_seqlen": min(T, max_doc_len) if max_doc_len else T}
        if packed if packed is not None else idx.is_cuda:
            nz = starts.reshape(-1).nonzero().squeeze(1)
            d["cu"] = torch.cat([nz, nz.new_full((1,), B * T)]).to(torch.int32)
        return d

    # ---- trunk
    def trunk(self, idx: Tensor, pos: Tensor | None = None, key_valid: Tensor | None = None,
              collect_kv: bool = False, cu_seqlens: Tensor | None = None, max_seqlen: int | None = None,
              seg_starts: Tensor | None = None):
        """idx (B, T). key_valid (B, T) bool marks real (non-pad) tokens for left-padded prefill. Returns final
        normed hidden (B, T, d) and optionally per-layer (k, v).
        Packed sequences (perf flag, docs/perf_opt.md): idx (1, N) holds several sequences back to back,
        cu_seqlens (S+1,) int32 their boundaries (every segment <= max_seqlen), pos (N,) each token's position
        within its own sequence; attention is causal within each segment (varlen kernel).
        Document masking without the varlen kernel (CPU / reference): seg_starts (B, T) bool marks segment starts
        (doc_layout); attention is causal within each segment (dense masks) and pos defaults to within-segment."""
        B, T = idx.shape
        dev = idx.device
        if seg_starts is not None:
            assert cu_seqlens is None and key_valid is None
            assert all(b.attn.kind == "attn" for b in self.blocks), "document masking needs attention-only mixers"
            if pos is None:
                t = torch.arange(T, device=dev)
                pos = t - torch.cummax(torch.where(seg_starts, t, torch.zeros_like(t)), dim=1).values
        if pos is None:
            pos = torch.arange(T, device=dev)
        cos, sin = self.rotary.cos_sin(pos)
        rp = self.rotary.n_pairs
        ve = self._value_embeds(idx)
        xb = None
        if self.cfg.bigram_rows:
            if cu_seqlens is not None:  # packed: every segment start has no previous token
                nop = torch.zeros(B * T + 1, dtype=torch.bool, device=dev)
                nop[cu_seqlens[:-1].long()] = True
                nop = nop[:B * T].view(B, T)
            elif seg_starts is not None:
                nop = seg_starts
            elif key_valid is not None:  # left pad: first real token (and pads) get the reserved row
                nop = ~(key_valid & torch.cat([torch.zeros_like(key_valid[:, :1]), key_valid[:, :-1]], 1))
            else:
                nop = torch.zeros(B, T, dtype=torch.bool, device=dev)
                nop[:, 0] = True
            xb = self._bigram_embeds(idx, nop)
        x = rms_norm(self.embed(idx))
        x0 = x
        if cu_seqlens is None and key_valid is None and seg_starts is None and self.train_attn == "varlen" \
                and idx.is_cuda and self.attn_impl != "flex":
            cu_seqlens = torch.arange(0, B * T + 1, T, device=dev, dtype=torch.int32)
            max_seqlen = T
        if cu_seqlens is not None:
            assert key_valid is None and idx.is_cuda
            masks = {True: ("varlen", cu_seqlens, max_seqlen, None),
                     False: ("varlen", cu_seqlens, max_seqlen, self.cfg.window)}
        elif seg_starts is not None:
            seg = torch.cumsum(seg_starts.long(), 1)
            same = (seg[:, :, None] == seg[:, None, :])[:, None]          # (B, 1, T, T)
            causal = torch.ones(T, T, dtype=torch.bool, device=dev).tril()
            masks = {True: same & causal, False: same & self._local_mask(T, dev)}
        elif key_valid is None and self.attn_impl == "flex" and idx.is_cuda:
            masks = {True: ("flex", self._block_mask(T, None, dev)),
                     False: ("flex", self._block_mask(T, self.cfg.window, dev))}
        elif key_valid is None and T <= self.cfg.window:
            masks = {True: None, False: None}
        elif key_valid is None:
            masks = {True: None, False: self._local_mask(T, dev)}
        else:
            causal = torch.ones(T, T, dtype=torch.bool, device=dev).tril()
            eye = torch.eye(T, dtype=torch.bool, device=dev)
            kvm = key_valid[:, None, None, :]
            masks = {True: (causal & kvm) | eye, False: (self._local_mask(T, dev) & kvm) | eye}
        hist = {0: x0} if 0 in self.retained else {}
        kvs = []
        for l, blk in enumerate(self.blocks):
            x = self._inject(x, xb, l)
            if self.mudd is not None and l in self.cfg.mudd_sites:
                taps = [hist[t] if t != l else x for t in self.cfg.mudd_taps(l)]
                a_in, base = self.mudd(taps, x, l)
                xn = rms_norm(a_in)
            else:
                xn, base = rms_norm(x), x
            a, kv = blk.attn(xn, ve, cos, sin, rp, masks[blk.attn.is_global], key_valid)
            x = blk.finish(base, a)
            if collect_kv:
                kvs.append(kv)
            if l + 1 in self.retained:
                hist[l + 1] = x
        return rms_norm(x), kvs

    def logits(self, h: Tensor) -> Tensor:
        a, b, c = self.cfg.softcap
        z = F.linear(h, self.embed.weight.to(h.dtype)).float()
        return a * torch.sigmoid((z + b) / c)

    def forward(self, idx: Tensor, targets: Tensor | None = None, loss_chunk: int = 2048, docs: dict | None = None):
        """docs (doc_inputs): document-masked attention, within-document positions and bigram resets."""
        if docs is None:
            h, _ = self.trunk(idx)
        elif docs.get("cu") is not None:
            B, T = idx.shape
            h, _ = self.trunk(idx.reshape(1, B * T), pos=docs["pos"].reshape(-1), cu_seqlens=docs["cu"],
                              max_seqlen=docs["max_seqlen"])
            h = h.view(B, T, -1)
        else:
            h, _ = self.trunk(idx, pos=docs["pos"], seg_starts=docs["starts"])
        if targets is None:
            return self.logits(h)
        return chunked_softcap_ce(h.reshape(-1, h.size(-1)), self.embed.weight, targets.reshape(-1),
                                  self.cfg.softcap, loss_chunk)

    # ---- inference
    # Decode implementation. "auto" = the sm75 tensor-core decode kernel with an int8 KV cache
    # (ultraviolet/kernels, docs/math_arch.md) whenever the cache is fp16 on CUDA and head_dim is 64/128;
    # otherwise the SDPA path below. Override per model: "sdpa" | "kernel-fp16" | "kernel-int8".
    decode_impl = "auto"
    decode_backend = "auto"        # kernel backend: "auto" | "cuda" | "triton"
    decode_num_splits = None       # optional {"global": n, "local": n}

    def resolve_decode_impl(self, dtype, device) -> str:
        impl = self.decode_impl
        if impl == "auto":
            ok = (torch.device(device).type == "cuda" and dtype == torch.float16 and self.cfg.head_dim in (64, 128)
                  and self.cfg.n_heads // self.cfg.n_kv_heads <= 8)
            impl = "kernel-int8" if ok else "sdpa"
        return impl

    def new_state(self, batch: int, max_len: int, dtype=torch.float16, device=None) -> "DecodeState":
        dev = device or self.embed.weight.device
        impl = self.resolve_decode_impl(dtype, dev)
        if impl == "sdpa":
            st = DecodeState(self.cfg, batch, max_len, dtype, dev)
        else:
            from .kernels.fast_decode import KernelDecodeState
            st = KernelDecodeState(self.cfg, batch, max_len, dtype, dev, kv_dtype=impl.split("-")[1])
        for l, b in enumerate(self.blocks):
            if b.attn.kind != "attn":
                st.mix[l] = b.attn.new_state(batch, dtype, dev)
        return st

    prefill_chunk_tokens = 65536
    # perf flag (docs/perf_opt.md): kernel-state prefill packs the rows' real tokens (no left padding) through the
    # varlen attention kernel; same positions / cache slots as the padded prefill
    prefill_packed = False
    prefill_compile = False        # with prefill_packed: torch.compile(dynamic=True) the packed prefill trunk

    @torch.no_grad()
    def prefill(self, idx: Tensor, starts: Tensor, st: "DecodeState", rows: Tensor | None = None) -> Tensor:
        """idx (n, T) left-padded (row i's real tokens at positions >= starts[i]). Writes the KV of cache rows `rows`
        (default: all B rows, n == B) from scratch, sets those rows' position to T, and returns the last-position
        logits (n, V). Stale cache contents beyond T are masked by position, so no reset is needed."""
        if st.kv_dtype is not None:  # kernel state: same prefill, quantizing cache writes
            from .kernels.fast_decode import prefill_fast
            return prefill_fast(self, idx, starts, st, rows)
        n, T = idx.shape
        assert T <= st.max_len
        chunk = max(1, self.prefill_chunk_tokens // T)
        if n > chunk:  # bound activation memory (a 512 x 300 prefill otherwise OOMs an 11 GB card)
            if rows is None:
                assert n == st.B
                rows = torch.arange(n, device=idx.device)
            return torch.cat([self.prefill(idx[i:i + chunk], starts[i:i + chunk], st, rows[i:i + chunk])
                              for i in range(0, n, chunk)])
        key_valid = torch.arange(T, device=idx.device)[None, :] >= starts[:, None]
        h, kvs = self.trunk(idx, key_valid=key_valid, collect_kv=True)
        full = rows is None
        if full:
            assert n == st.B
            rows = torch.arange(n, device=idx.device)
        for l, kv in enumerate(kvs):
            if isinstance(kv, dict):  # recurrent / conv mixer state
                for name, val in kv.items():
                    buf = st.mix[l][name]
                    if full:
                        buf.copy_(val.to(buf.dtype))
                    else:
                        buf[rows] = val.to(buf.dtype)
                continue
            k, v = kv
            kt, vt = k.transpose(1, 2).to(st.dtype), v.transpose(1, 2).to(st.dtype)  # (n, K, T, hd)
            for cache, new in ((st.k[l], kt), (st.v[l], vt)):
                S = cache.size(2)
                if self.cfg.is_global(l):
                    if full:
                        cache[:, :, :T] = new
                    else:
                        cache[rows, :, :T] = new
                else:
                    lo = max(0, T - S)
                    slots = torch.arange(lo, T, device=idx.device) % S
                    if full:
                        cache[:, :, slots] = new[:, :, lo:]
                    else:
                        blk = cache[rows]
                        blk[:, :, slots] = new[:, :, lo:]
                        cache[rows] = blk
        st.starts[rows] = starts
        st.pos[rows] = T
        st.host_pos[rows.cpu().numpy()] = T
        if st.prev is not None:
            st.prev[rows] = idx[:, -1]
        return self.logits(h[:, -1])

    def decode_step(self, tok: Tensor, st: "DecodeState", g_len: int, skip_attn: bool = False) -> Tensor:
        """One decode step for all B rows, row b at its own position st.pos[b] (device tensors; graph-capturable).
        Global layers attend over cache[:, :, :g_len] (g_len > max position; static per graph). Increments st.pos.
        skip_attn (profiling only) replaces cache write + attention with a pass-through."""
        if st.kv_dtype is not None:  # kernel state (int8 / fp16 cache): no masks, rows read only their valid keys
            from .kernels.fast_decode import decode_step_fast
            return decode_step_fast(self, tok, st, g_len, skip_attn, self.decode_num_splits, self.decode_backend)
        cfg = self.cfg
        B = tok.size(0)
        p = st.pos  # (B,) long
        cos, sin = self.rotary.cos_sin(p)  # (B, n_pairs)
        cos, sin = cos[:, None, :], sin[:, None, :]  # broadcast as (B, 1, 1, n_pairs) inside apply_rope
        rp = self.rotary.n_pairs
        idx = tok.view(B, 1)
        ve = self._value_embeds(idx)
        xb = self._bigram_step(tok, st)
        x = rms_norm(self.embed(idx)).to(st.dtype)
        hist = {0: x} if 0 in self.retained else {}
        dev, dt = tok.device, st.dtype
        K, G, hd = cfg.n_kv_heads, cfg.n_heads // cfg.n_kv_heads, cfg.head_dim
        pc, sc = p[:, None], st.starts[:, None]
        gpos = torch.arange(g_len, device=dev)[None, :]
        g_valid = (gpos <= pc) & (gpos >= sc)
        W = st.local_len
        spos = pc - torch.remainder(pc - torch.arange(W, device=dev)[None, :], W)
        l_valid = (spos >= 0) & (spos >= sc)
        # dense (B, K, G, S) additive masks, built once per step (stride-0 masks can trip the efficient-attention
        # kernel under torch.compile: "cutlassF: no kernel found")
        g_bias = torch.zeros(g_valid.shape, device=dev, dtype=dt).masked_fill(~g_valid, float("-inf"))
        l_bias = torch.zeros(l_valid.shape, device=dev, dtype=dt).masked_fill(~l_valid, float("-inf"))
        g_bias = g_bias[:, None, None, :].expand(B, K, G, g_len).contiguous()
        l_bias = l_bias[:, None, None, :].expand(B, K, G, W).contiguous()
        # cache writes as index_copy_ into a flat (B*K*S, hd) view: inductor keeps this in place (a scatter_ on the
        # 4-D cache gets functionalised into a full-cache copy per layer)
        bk = (torch.arange(B, device=dev)[:, None] * K + torch.arange(K, device=dev)[None, :])  # (B, K)
        gidx = (bk * st.max_len + pc).reshape(-1)
        lidx = (bk * W + torch.remainder(pc, W)).reshape(-1)
        for l, blk in enumerate(self.blocks):
            at = blk.attn
            x = self._inject(x, xb, l)
            if self.mudd is not None and l in cfg.mudd_sites:
                taps = [hist[t] if t != l else x for t in cfg.mudd_taps(l)]
                a_in, base = self.mudd(taps, x, l)
                xn = rms_norm(a_in)
            else:
                xn, base = rms_norm(x), x
            if at.kind != "attn":
                x = blk.finish(base, at.step(xn, st.mix[l]))
                if l + 1 in self.retained:
                    hist[l + 1] = x
                continue
            q, k, v = at.qkv(xn, ve, cos, sin, rp)  # (B,1,H,hd), (B,1,K,hd)
            if skip_attn:
                z = q.reshape(B, 1, at.H, at.hd)
            else:
                ck, cv = st.k[l], st.v[l]
                ix = gidx if at.is_global else lidx
                ck.view(-1, hd).index_copy_(0, ix, k.reshape(B * K, hd))
                cv.view(-1, hd).index_copy_(0, ix, v.reshape(B * K, hd))
                if at.is_global:
                    kk, vv, bias = ck[:, :, :g_len], cv[:, :, :g_len], g_bias
                else:
                    kk, vv, bias = ck, cv, l_bias
                qg = q.reshape(B, at.K, at.G, at.hd).contiguous()  # G query heads of a kv group = G query rows
                z = F.scaled_dot_product_attention(qg, kk, vv, attn_mask=bias, scale=cfg.attn_scale)
                z = z.reshape(B, 1, at.H, at.hd)
            a = at.post(z, xn, v)
            x = blk.finish(base, a)
            if l + 1 in self.retained:
                hist[l + 1] = x
        st.pos.add_(1)
        return self.logits(rms_norm(x)[:, 0])

_FLEX = None


def _flex_attention():
    global _FLEX
    if _FLEX is None:
        from torch.nn.attention.flex_attention import flex_attention
        _FLEX = torch.compile(flex_attention, dynamic=False)
    return _FLEX


def chunked_softcap_ce(h: Tensor, w: Tensor, t: Tensor, softcap, chunk: int) -> Tensor:
    """Mean CE of softcapped tied-head logits, computed in token chunks with recomputation so the (N, V) fp32
    logits are never all resident."""
    a, b, c = softcap

    def f(hc, w, tc):
        z = F.linear(hc, w.to(hc.dtype)).float()
        z = a * torch.sigmoid((z + b) / c)
        return F.cross_entropy(z, tc, reduction="sum", ignore_index=-1)

    N = h.size(0)
    if N <= chunk:
        tot = f(h, w, t)
    else:
        tot = h.new_zeros((), dtype=torch.float32)
        for i in range(0, N, chunk):
            tot = tot + torch.utils.checkpoint.checkpoint(f, h[i:i + chunk], w, t[i:i + chunk], use_reentrant=False)
    return tot / (t != -1).sum().clamp_min(1)


# --------------------------------------------------------------------------------------------- decode state
class DecodeState:
    """Static KV cache: global layers (B, K, max_len, hd); local layers ring buffer (B, K, window, hd).
    Every row has its own position (pos, host_pos) and left-pad start, so rows can be refilled independently."""
    kv_dtype = None  # set by kernels.fast_decode.KernelDecodeState ("fp16" | "int8")

    def __init__(self, cfg: MathConfig, B: int, max_len: int, dtype, device):
        self.cfg, self.B, self.max_len, self.dtype = cfg, B, max_len, dtype
        self.local_len = min(cfg.window, max_len)
        self.k, self.v = [], []
        self.mix: dict[int, dict] = {}  # non-attention layers: conv tails / recurrent state (filled by new_state)
        for l in range(cfg.n_layers):
            if cfg.mixer(l) != "attn":
                self.k.append(None)
                self.v.append(None)
                continue
            S = max_len if cfg.is_global(l) else self.local_len
            self.k.append(torch.zeros(B, cfg.n_kv_heads, S, cfg.head_dim, dtype=dtype, device=device))
            self.v.append(torch.zeros(B, cfg.n_kv_heads, S, cfg.head_dim, dtype=dtype, device=device))
        self.pos = torch.zeros(B, dtype=torch.long, device=device)
        self.starts = torch.zeros(B, dtype=torch.long, device=device)
        self.host_pos = np.zeros(B, dtype=np.int64)
        # hashed bigram: previous token per row (-1: none -> reserved row); set by prefill, advanced by decode_step
        self.prev = torch.full((B,), -1, dtype=torch.long, device=device) if cfg.bigram_rows else None

    def set_pos(self, p: int):
        self.pos.fill_(p)
        self.host_pos[:] = p

    def nbytes(self) -> int:
        n = sum(t.numel() * t.element_size() for t in self.k + self.v if t is not None)
        return n + sum(t.numel() * t.element_size() for d in self.mix.values() for t in d.values())


class DecodeRunner:
    """Runs decode steps, optionally torch.compile'd and replayed from CUDA graphs captured per global-context
    bucket (g_len = max row position rounded up to `bucket`)."""

    def __init__(self, model: MathLM, st: DecodeState, bucket: int = 1024, use_graph: bool = True,
                 temperature: float = 1.0, top_k: int = 0, compile: bool = False, skip_attn: bool = False,
                 top_p: float = 1.0):
        self.m, self.st, self.bucket = model, st, bucket
        self.use_graph = use_graph and st.pos.is_cuda
        self.temperature, self.top_k, self.top_p = temperature, top_k, top_p
        self.tok = torch.zeros(st.B, dtype=torch.long, device=st.pos.device)
        self.out_tok = torch.zeros_like(self.tok)
        self.out_logits = None
        # perf flag (docs/perf_opt.md): want_logp = True also computes log softmax(logits / T)[sampled token] inside
        # the compiled / captured step into the static buffer out_logp (B,) fp32 (set before the first step)
        self.want_logp = False
        self.out_logp = torch.zeros(st.B, dtype=torch.float32, device=st.pos.device)
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self.pool = None
        self.skip_attn = skip_attn
        # torch.compile fuses the norm/rope/gate/residual/softcap/sampling elementwise work; g_len becomes a
        # dynamic int after the first recompile, so all buckets share ~2 compiled graphs
        self.compile = compile and st.pos.is_cuda

    def g_len(self, p: int) -> int:
        return min(self.st.max_len, ((p + 1 + self.bucket - 1) // self.bucket) * self.bucket)

    def _body(self, g_len: int):
        if self.want_logp:
            fn = _decode_and_sample_logp_compiled() if self.compile else _decode_and_sample_logp
            logits, nxt, lp = fn(self.m, self.st, self.tok, g_len, self.temperature, self.top_k, self.skip_attn,
                                 self.top_p)
            self.out_logp.copy_(lp)
            self.out_tok.copy_(nxt)
            return logits
        fn = _decode_and_sample_compiled() if self.compile else _decode_and_sample
        logits, nxt = fn(self.m, self.st, self.tok, g_len, self.temperature, self.top_k, self.skip_attn, self.top_p)
        self.out_tok.copy_(nxt)
        return logits

    def step(self, tok: Tensor) -> Tensor:
        """Feed tok (B,) at each row's current position; returns the next sampled tokens (B,) (a static buffer)."""
        p = int(self.st.host_pos.max())
        assert p < self.st.max_len, "cache full"
        gl = self.g_len(p)
        self.tok.copy_(tok)
        if not self.use_graph:
            self.out_logits = self._body(gl)
        else:
            g = self.graphs.get(gl)
            if g is None:
                # warm up on a side stream, then capture; pos is restored after each (the cache slot written at
                # the current positions is rewritten by the real replay with identical inputs)
                s = torch.cuda.Stream()
                s.wait_stream(torch.cuda.current_stream())
                pos0 = self.st.pos.clone()
                prev = getattr(self.st, "prev", None)  # bigram prev-token buffer: also advanced by the step
                prev0 = prev.clone() if prev is not None else None
                with torch.cuda.stream(s):
                    for _ in range(2):
                        self._body(gl)
                        self.st.pos.copy_(pos0)
                        if prev0 is not None:
                            prev.copy_(prev0)
                torch.cuda.current_stream().wait_stream(s)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=self.pool, capture_error_mode="thread_local"):
                    self.out_logits = self._body(gl)
                self.pool = g.pool()
                self.st.pos.copy_(pos0)
                if prev0 is not None:
                    prev.copy_(prev0)
                self.graphs[gl] = g
            g.replay()
        self.st.host_pos += 1
        return self.out_tok


@torch.no_grad()
def _decode_and_sample(m: MathLM, st: DecodeState, tok: Tensor, g_len: int, temperature: float, top_k: int,
                       skip_attn: bool = False, top_p: float = 1.0):
    logits = m.decode_step(tok, st, g_len, skip_attn=skip_attn)
    return logits, sample(logits, temperature, top_k, top_p)


@torch.no_grad()
def _decode_and_sample_logp(m: MathLM, st: DecodeState, tok: Tensor, g_len: int, temperature: float, top_k: int,
                            skip_attn: bool = False, top_p: float = 1.0):
    """_decode_and_sample plus the behaviour log-prob log softmax(logits / T)[next] (fp32), as rl_stream records it
    (that is the untruncated softmax: top-k / top-p are not applied to it, matching StreamGen)."""
    logits = m.decode_step(tok, st, g_len, skip_attn=skip_attn)
    nxt = sample(logits, temperature, top_k, top_p)
    z = logits.float() / temperature
    lp = z.gather(1, nxt.view(-1, 1))[:, 0] - torch.logsumexp(z, -1)
    return logits, nxt, lp


_COMPILED = None
_COMPILED_LP = None


def _decode_and_sample_logp_compiled():
    global _COMPILED_LP
    if _COMPILED_LP is None:
        import torch._dynamo
        torch._dynamo.config.cache_size_limit = max(64, torch._dynamo.config.cache_size_limit)
        _COMPILED_LP = torch.compile(_decode_and_sample_logp)
    return _COMPILED_LP


def _decode_and_sample_compiled():
    global _COMPILED
    if _COMPILED is None:
        import torch._dynamo
        torch._dynamo.config.cache_size_limit = max(64, torch._dynamo.config.cache_size_limit)
        _COMPILED = torch.compile(_decode_and_sample)
    return _COMPILED


def sample(logits: Tensor, temperature: float, top_k: int = 0, top_p: float = 1.0) -> Tensor:
    """Graph-capturable sampling (Gumbel-max); temperature 0 -> greedy. top_p < 1: nucleus sampling on
    softmax(logits / temperature) -- keep the smallest prefix of tokens sorted by probability whose mass reaches
    top_p (the first token always kept), as in vLLM / HF."""
    if temperature <= 0:
        return logits.argmax(-1)
    z = logits.float() / temperature
    if top_k:
        kth = torch.topk(z, top_k, dim=-1).values[..., -1:]
        z = z.masked_fill(z < kth, float("-inf"))
    if top_p < 1.0:
        zs = torch.sort(z, dim=-1, descending=True).values
        ps = torch.softmax(zs, -1)
        keep = (ps.cumsum(-1) - ps) < top_p
        cut = zs.masked_fill(~keep, float("inf")).amin(-1, keepdim=True)
        z = z.masked_fill(z < cut, float("-inf"))
    u = torch.rand_like(z).clamp_(1e-10, 1.0)
    return (z - torch.log(-torch.log(u))).argmax(-1)


def _left_pad(prompts, pad_id, dev):
    Tp = max(len(p) for p in prompts)
    idx = torch.full((len(prompts), Tp), pad_id, dtype=torch.long)
    starts = torch.zeros(len(prompts), dtype=torch.long)
    for b, p in enumerate(prompts):
        idx[b, Tp - len(p):] = torch.as_tensor(p, dtype=torch.long)
        starts[b] = Tp - len(p)
    return idx.to(dev), starts.to(dev)


def _autocast(dev, dtype):
    return torch.autocast("cuda", dtype=dtype) if dev.type == "cuda" else torch.autocast("cpu", enabled=False)


@torch.no_grad()
def generate(model: MathLM, prompts: list[list[int]], max_new: int, temperature: float = 1.0, top_k: int = 0,
             eos_id: int | None = None, pad_id: int = 0, max_len: int | None = None, use_graph: bool = True,
             bucket: int = 1024, dtype=torch.float16, compile: bool = True) -> list[list[int]]:
    """Static-batch sampling: all prompts prefilled together, decoded in lockstep. Model in eval mode with
    fuse_for_inference(dtype) applied. For many prompts of varied output length use `generate_continuous`."""
    dev = model.embed.weight.device
    B = len(prompts)
    idx, starts = _left_pad(prompts, pad_id, dev)
    max_len = max_len or (idx.size(1) + max_new)
    st = model.new_state(B, max_len, dtype, dev)
    with _autocast(dev, dtype):
        logits = model.prefill(idx, starts, st)
    tok = sample(logits, temperature, top_k)
    runner = DecodeRunner(model, st, bucket=bucket, use_graph=use_graph, temperature=temperature, top_k=top_k,
                          compile=compile)
    out = [tok.clone()]
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    if eos_id is not None:
        done |= tok == eos_id
    for i in range(max_new - 1):
        if st.host_pos.max() >= max_len:
            break
        tok = runner.step(tok).clone()
        out.append(tok)
        if eos_id is not None:
            done |= tok == eos_id
            if i % 64 == 63 and bool(done.all()):
                break
    seq = torch.stack(out, 1).tolist()
    res = []
    for row in seq:
        if eos_id is not None and eos_id in row:
            row = row[:row.index(eos_id) + 1]
        res.append(row)
    return res


@torch.no_grad()
def generate_continuous(model: MathLM, prompts: list[list[int]], max_new: int, batch: int,
                        temperature: float = 1.0, top_k: int = 0, eos_id: int | None = None, pad_id: int = 0,
                        check_every: int = 16, use_graph: bool = True, bucket: int = 1024, dtype=torch.float16,
                        compile: bool = True, target_lens: list[int] | None = None, stats: dict | None = None,
                        top_p: float = 1.0):
    """Continuous batching: `batch` cache slots; every `check_every` steps finished rows (EOS or max_new tokens,
    or `target_lens[i]` tokens -- for benchmarking with untrained models) are harvested and refilled with the next
    prompts (batched left-padded prefill into just those rows). Returns completions in prompt order.
    Cost of a check: one small device->host copy; tokens decoded after EOS inside a window are discarded."""
    dev = model.embed.weight.device
    n = len(prompts)
    B = min(batch, n)
    max_len = max(len(p) for p in prompts) + max_new + check_every
    st = model.new_state(B, max_len, dtype, dev)
    runner = DecodeRunner(model, st, bucket=bucket, use_graph=use_graph, temperature=temperature, top_k=top_k,
                          compile=compile, top_p=top_p)
    buf = torch.zeros(B, max_new + check_every, dtype=torch.long, device=dev)
    cnt = torch.zeros(B, dtype=torch.long, device=dev)
    slot_item = [-1] * B
    limit = np.zeros(B, dtype=np.int64)
    host_cnt = np.zeros(B, dtype=np.int64)
    results: list = [None] * n
    nxt = 0
    tok = torch.zeros(B, dtype=torch.long, device=dev)
    n_prefill = n_steps = 0

    def refill(rows):
        nonlocal nxt, n_prefill
        rows = [r for r in rows if nxt < n] if nxt < n else []
        rows = rows[:n - nxt]
        if not rows:
            return
        items = list(range(nxt, nxt + len(rows)))
        nxt += len(rows)
        idx, starts = _left_pad([prompts[i] for i in items], pad_id, dev)
        rt = torch.tensor(rows, device=dev)
        buf[rt] = -1
        with _autocast(dev, dtype):
            logits = model.prefill(idx, starts, st, rows=rt)
        first = sample(logits, temperature, top_k, top_p)
        tok[rt] = first
        buf[rt, 0] = first
        cnt[rt] = 1
        for r, i in zip(rows, items):
            slot_item[r] = i
            limit[r] = min(max_new, target_lens[i]) if target_lens is not None else max_new
            host_cnt[r] = 1
        n_prefill += len(rows)

    refill(list(range(B)))
    active = np.array([s >= 0 for s in slot_item])
    while active.any():
        for _ in range(check_every):
            tok = runner.step(tok)
            buf.scatter_(1, cnt.view(B, 1), tok.view(B, 1))
            cnt.add_(1).clamp_(max=buf.size(1) - 1)
            n_steps += 1
        host_cnt += check_every
        done_rows = []
        cnt_h = None
        if eos_id is not None:
            hit = (buf == eos_id).any(1).cpu().numpy()
        for r in range(B):
            if slot_item[r] < 0:
                continue
            fin = host_cnt[r] >= limit[r] or (eos_id is not None and hit[r])
            if fin:
                if cnt_h is None:
                    rows_buf = buf.cpu().numpy()
                    cnt_h = True
                row = rows_buf[r, :min(host_cnt[r], limit[r])].tolist()
                if eos_id is not None and eos_id in row:
                    row = row[:row.index(eos_id) + 1]
                results[slot_item[r]] = row
                slot_item[r] = -1
                done_rows.append(r)
        if done_rows:
            refill(done_rows)
            idle = [r for r in done_rows if slot_item[r] < 0]
            if idle:  # no prompts left: park the row at position 0 so it never overruns the cache
                it = torch.tensor(idle, device=dev)
                st.pos[it] = 0
                st.starts[it] = 0
                st.host_pos[idle] = 0
        active = np.array([s >= 0 for s in slot_item])
    if stats is not None:
        stats.update(steps=n_steps, prefilled=n_prefill, slots=B)
    return results
