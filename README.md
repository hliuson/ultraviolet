# ultraviolet

Ultraviolet 50M (UV-50M) is a 50.8 million parameter decoder-only transformer model trained primarily on math for
rapid RL experimentation on language models. It is intended to be a small model organism for
reinforcement learning which can be effectively trained on any consumer GPU.

UV-50M's name and architecture are derived from the
[Limite 1B Violetto](https://huggingface.co/paradigma-inc/limite-1b-violetto) model produced by Paradigma. UV-50M is
optimized for inference speed rather than maximal performance per parameter. This allows reinforcement learning
experiments to run at high speed on consumer GPUs: at short context lengths (under 1k tokens) it exceeds 200k
tokens per second of throughput on a single 2080 Ti.

This repo holds the training, evaluation and RL code. Weights and data are on Hugging Face:

| repo | what |
|---|---|
| [harribo11/ultraviolet-50m-base-alpha](https://huggingface.co/harribo11/ultraviolet-50m-base-alpha) | pretrained base model |
| [harribo11/ultraviolet-50m-mid-alpha](https://huggingface.co/harribo11/ultraviolet-50m-mid-alpha) | reasoning model (thinks, then answers in `\boxed{}`) |
| [harribo11/violetto-math-traces](https://huggingface.co/datasets/harribo11/violetto-math-traces) | the reasoning traces mid-alpha was trained on |

## Use the model

```python
import torch
from huggingface_hub import hf_hub_download
from ultraviolet.model import MathLM, MathConfig

s = torch.load(hf_hub_download("harribo11/ultraviolet-50m-mid-alpha", "model.pt"), map_location="cpu", weights_only=False)
model = MathLM(MathConfig(**s["cfg"]))
model.load_state_dict(s["model"])
tok = hf_hub_download("harribo11/ultraviolet-50m-mid-alpha", "tokenizer/tokenizer.json")
```

Prompt: `<|endoftext|>Problem:\n{problem}\n\n<think>\n`. The model writes its reasoning, `</think>`, then a solution
ending in `\boxed{answer}`.

## Layout

```
ultraviolet/   model, optimizers, data loaders, decode kernels, verifiable task scorers
scripts/       tokenizer, pretraining, SFT, RL (rl_grpo.py), evals (eval_v3.py), benchmarks; data/ builds the corpus
workbench/     synthetic data generators (synth/) and RL environments (envs/)
tests/         pytest suite
docs/          architecture, kernels, corpus, tasks, environments
```

Run everything from the repo root with `PYTHONPATH=.:scripts`.

```bash
# pretrain (fp16 on Turing, bf16 on Ampere+); DDP via torchrun
python scripts/train_pt.py --data DIR --preset uv-50m-alpha --tokens 2e8 --out RUN --name NAME --dtype fp16

# RL from a checkpoint on a task pool
torchrun --standalone --nproc_per_node 4 scripts/rl_grpo.py --ckpt CKPT.pt --tok tokenizer.json \
  --pool POOL.jsonl --eval-problems EVAL.jsonl --filters FILTERS.json --out RUN --name NAME --mode 1gpu --stream 1 \
  --n-prompts 256 --group 8 --optim normuon --strict-format 1 --max-new 2048

# eval: pass@1 / pass@k at T=1.0
python scripts/eval_v3.py --help

# tests
python -m pytest -q tests
```

Runs log to wandb through `scripts/tr_log.py`: project `$WANDB_PROJECT` (default `ultraviolet`), entity
`$WANDB_ENTITY`. Set `WANDB_DIR` to a scratch directory.

## License

Apache-2.0.
