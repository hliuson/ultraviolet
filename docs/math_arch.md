# UV-50M architecture (`uv-50m-alpha`)

Code: `ultraviolet/model.py` (`MathConfig`, `PRESETS`, `make_config`, `MathLM`), `ultraviolet/optim.py`.
Figure: `docs/figures/arch_uv_50m_alpha.tex` (`pdflatex arch_uv_50m_alpha.tex`).

```python
from ultraviolet.model import MathLM, make_config
cfg = make_config("uv-50m-alpha", vocab_size=16384)   # train_pt.py takes vocab_size from the data's meta.json
model = MathLM(cfg)
```

## Spec

| | |
|---|---|
| layers | 10, pre-norm (weightless RMSNorm) |
| d_model | 640 |
| attention | multi-query: 8 query heads x head_dim 64, 1 KV head (G = 8) |
| MLP | SwiGLU, d_ff 1664 |
| layer mix | layer l is global iff `l % 5 == 4`: globals L4, L9; the other 8 are local |
| local layers | sliding window 256 (query i sees keys `(i-256, i]`), partial RoPE on 32 of 64 dims, base 1024 |
| global layers | full causal attention, no positional encoding (NoPE) |
| value embeddings | one shared table (16384 x 64) added to v through a gate `2*sigmoid(W x[:, :12])`, on layers `l % 3 == 1` (L1, L4, L7); `--ve-block-pos` in train_pt.py overrides the pattern |
| QK | RMS QK-norm before RoPE; scale `1.13 / sqrt(head_dim)` = 0.141 |
| XSA | on: per head, `z -= tanh(alpha_h) * (z . v_hat) * v_hat` removes the component along the token's own value |
| output gate | per head, `z *= 2*sigmoid(W x[:, :128])` |
| residual | `h = lambda_r * x + lambda_p * attn`, `x' = mu_r * h + mu_p * mlp` (learned per-layer scalars); W_QKV and W_O carry learned scalar scales |
| head | tied embeddings; logit softcap `23 * sigmoid((z + 5) / 7.5)` |
| vocab | 16384, byte-level BPE (`scripts/train_tokenizer.py`) |

Parameters: 39.32M non-embedding + 10.49M tied embedding + 1.05M value embedding = 50.86M (plus ~10k scalars and gate
weights). `cfg.param_counts()` reports the breakdown.

Context: there is no fixed position limit. Local layers see 256 keys and global layers are NoPE. The training
sequence length is `train_pt.py --seq-len`.

Tokenizer special tokens (same ids at every vocab size): `<|endoftext|>`=0 (EOS, document separator), `<|im_start|>`=1,
`<|im_end|>`=2, `<think>`=3, `</think>`=4, `<|pad|>`=5, `<|reserved_0..63|>`=6..69. Pretokenization is the Qwen2 regex
(single digits) with LaTeX commands (`\frac`, `\alpha`) kept as whole pretokens.

Other `MathConfig` options (MUDD, conv / Gated DeltaNet local mixers, hashed bigram embedding) default off and are not
used by `uv-50m-alpha`.

## Optimizer and schedule (`ultraviolet/optim.py`)

- `build_optimizers(model)` returns `(NorMuon, AdamW)`. NorMuon (lr 0.018) updates hidden 2D matrices: Nesterov
  momentum warmed 0.85 -> 0.95, Polar Express orthogonalisation (5 steps, fp32), per-neuron second moment
  (beta2 0.9), cautious weight decay scaled by lr^2. Embeddings, value embeddings, scalars and gates use AdamW groups
  with per-group LRs (`ADAM_GROUPS`).
- `apply_schedule`: linear warmup, flat, then linear decay to 0 over the last 20% of steps.
- DDP: `distributed=True` splits the orthogonalisation across ranks; results are identical.

## Precision

`train_pt.py --dtype fp16` (default; autocast + GradScaler, for Turing) or `bf16`. Training attention is `--attn`
`sdpa` (default), `flex` (FlexAttention; needs sm80+, falls back to SDPA on Turing) or `varlen` (CUTLASS window
kernel, see docs/perf_opt.md).

## Inference

- `model.new_state(B, max_len, dtype)` builds a static per-layer KV cache: global layers hold `(B, K, max_len, hd)`,
  local layers a `window`-slot ring buffer (position p in slot `p % window`). Each row has its own position
  (`st.pos`) and left-pad start (`st.starts`), so rows can be refilled independently. QK-norm and RoPE are applied
  before the cache write.
- `MathLM.decode_impl` (default `"auto"`) selects `"kernel-int8"` (int8 KV cache + tensor-core decode kernel,
  below) when the cache dtype is fp16 on CUDA, head_dim is 64 or 128 and G <= 8, else `"sdpa"`. It can
  be set to `"sdpa"`, `"kernel-fp16"` or `"kernel-int8"`.
- `DecodeRunner` runs the decode step plus sampling (temperature / top-k / top-p), optionally `torch.compile`d, with
  one CUDA graph per global-context bucket (default 1024).
- `generate(model, prompts, max_new, ...)`: static batch, left-padded prefill, lockstep decode.
- `generate_continuous(model, prompts, max_new, batch, ...)`: continuous batching; every `check_every` (16) steps
  finished rows (EOS or `max_new`) are harvested and refilled with a batched prefill into just those rows. Prefill is
  chunked at `model.prefill_chunk_tokens` (65536).
- KV cache size (int8): 132 B per token per layer. A sequence at 4k context takes 1.35 MB (8 local rings of 256 slots
  plus 2 global layers); each extra context token adds 264 B.
- Throughput: on one RTX 2080 Ti (int8 kernel, CUDA graphs + compile, 4k context) decode runs at about 104k tok/s at
  batch 256.

## Decode kernels

`ultraviolet/kernels/` (tests: `tests/test_math_kernels.py`). `decode_attention` is a split-K decode attention over an
fp16 or int8 KV cache, registered as a `torch.library.custom_op` (opaque to `torch.compile`, CUDA-graph safe). The
`cuda` backend is compiled at runtime with NVRTC (no build step) and uses `mma.sync` tensor cores on sm75+ for head_dim
64 / 128 and G <= 8; `triton` covers other shapes. `fast_decode.patch_model(model, "int8" | "fp16")` binds the kernel
path onto a model (`unpatch_model` undoes it).

The int8 cache is symmetric per token and head (scale = absmax / 127, stored in fp16): `2 * K * (hd + 2)` bytes per
token per layer, half the fp16 cache. On a trained checkpoint it changes teacher-forced NLL by +0.002% (99.7% top-1
agreement with fp16 logits).
