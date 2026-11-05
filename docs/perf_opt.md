# Performance flags

Opt-in speed paths for RL (`scripts/rl_grpo.py --perf a,b,...`) and pretraining (`scripts/train_pt.py --attn varlen`).
With no flags the default code paths are unchanged. Code: `ultraviolet/perf.py`, `ultraviolet/kernels/varlen_attn.py`.
Tests: `tests/test_perf_opt.py`.

**Recommended RL flags:** `--perf pack,head16,pfc,lp,hv` (1.7-2.2x faster RL steps on an RTX 2080 Ti, 64 prompts x 8,
512 slots, max-new 2048). Use `head` instead of `head16` for log-probs bitwise identical to the default path.

## Flags

| flag | applies to | what it does | exact? |
|---|---|---|---|
| `varlen` | training | attention through PyTorch's CUTLASS kernel with `cu_seqlens` and `window_size`; local layers read only their window instead of a dense T x T mask | fp tolerance |
| `pack` | RL trainer | packs sequences with nonzero advantage back to back into fixed `--tok-budget` (16384) token micro-batches. Implies `varlen` | fp tolerance |
| `head` | RL trainer | policy-gradient head (softcap / CE / TIS) only on target rows, in compiled chunks; fp32 GEMM | bitwise-identical log-probs |
| `head16` | RL trainer | as `head`, with fp16 GEMMs (fp32 accumulate and output). Falls back to `head` when `--entropy-coef > 0` | no: slightly changed arithmetic |
| `h16f` | with `head16` | head gradient computed in the forward pass | identical to `head16` |
| `pf` | generation | refill prefill packs the refilled rows' prompt tokens (no left padding) through the varlen trunk | fp tolerance |
| `pfc` | generation | `pf` plus `torch.compile(dynamic=True)` of the packed prefill | fp tolerance |
| `pkv` | with `pf`/`pfc` | one int8 quantize call for all layers' prefill K/V | bit-identical |
| `rfh` | with `pf`/`pfc`, int8 KV | packed prefill indices built on the host (no device syncs) | bit-identical |
| `lp` | generation | behaviour log-prob computed inside the CUDA-graphed decode step instead of an eager (B, V) `log_softmax` | same tokens; log-probs within 1e-4 |
| `hv` | generation | harvest copies only finished rows to the host | same rollouts |
| `fz` | generation (int8/fp16 kernel KV, T > 0, no top-k / top-p) | fused Triton sampler (softcap + draw + log-prob). Implies `lp` | same distribution, different random stream |
| `acc16` | generation | fp16-accumulate GEMMs in prefill and decode only | no: changes sampler numerics |

The trainer flags (`pack`, `varlen`, `head`, `head16`) cannot be combined with `--ppo-schedule`; with `--clip-cov` or
`--mode 2gpu` the default trainer is used.

## Pretraining

`train_pt.py --attn varlen`: about 1.7x fwd+bwd tok/s over `sdpa` at T=4096 on a 2080 Ti, same semantics (no masking
at document boundaries). `--doc-mask 1` instead masks attention at document (EOS) boundaries (segments capped by
`--max-doc-len`, default 2048); it is faster but changes the objective.

## Benchmark

`PYTHONPATH=.:scripts python scripts/bench_perf.py MODE --out DIR --ckpt CK.pt --tok tokenizer.json --pool POOL.jsonl`
(always pass `--ckpt`, `--tok` and `--pool`; the built-in defaults are not portable).

| mode | what |
|---|---|
| `rl` | runs `rl_grpo.main()` with logging, evals and saves stubbed out and reports per-step generation / training time; `--perf` sets flags; `--capture PATH` saves one step's training batch |
| `train` | replays a captured batch through `;`-separated flag sets (`--variants "base;fp32;pack,head16"`; `fp32` = full-fp32 reference): time, memory, grad and log-prob error |
| `decode` | steady-state generation profile |
| `prefill` | refill prefill time for `--prefill-rows 4,16,64` |
| `pt` | pretrain step (`--seq-len 4096 --micro-bs 4`); `--variants "base;varlen"` |

## fp16 loss scaling in RL

RL uses fp16 autocast with a GradScaler. Isolated non-finite gradient steps are skipped, but with torch's default
`growth_interval=2000` each skip halves the scale for the rest of a short run. Use `--scaler-growth-interval 100`, and
optionally `--scaler-min-scale` / `--scaler-max-scale` (0 = none). Skipped steps record which parameters had
non-finite gradients.
