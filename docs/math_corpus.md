# Math pretraining corpus

Scripts: `scripts/data/download.py`, `scripts/data/download_v2.py`, `scripts/data/build.py`, `scripts/data/spec_v2.json`.
Run with `python -I`; they need `pyarrow` and `huggingface_hub`.

## Build

```bash
python -I scripts/data/download.py RAW              # v1 sources + eval sets (ev_*), revisions pinned in download_manifest.json
python -I scripts/data/download_v2.py RAW2          # Nemotron-CC-Math and Big-Math (gated: needs HF_TOKEN)
python -I scripts/data/build.py RAW OUT [--workers 48] [--pool 20000]                       # v1 mix
python -I scripts/data/build.py RAW OUT --spec scripts/data/spec_v2.json --bigmath BIGMATH.parquet  # v2 mix
```

`build.py` always reads the eval sets from `RAW/ev_*`. A spec maps `source -> {raw, max_files, split, kind, license}`;
set the `raw` paths in `spec_v2.json` (`RAW`, `RAW2`) to your download directories. `--pool-only` writes only the held-out pools.

Output: `OUT/{train,val}/<source>-NNNN.jsonl`, one `{"text", "source"}` per line; validation is 0.5% of docs by text
hash. Encode for training with `python -m ultraviolet.data encode --tok tokenizer.json --inputs 'OUT/train/*.jsonl'
--val-inputs 'OUT/val/*.jsonl' --out DATA` (uint16 shards, EOS after every doc, `meta.json`).

## Document formats

- Web and textbook: raw text.
- Short QA: `Problem:\n{problem}\n\nSolution:\n{solution}`.
- Reasoning traces: `Problem:\n{problem}\n\n<think>...</think>\n{solution}`. OpenR1 keeps only the first
  math-verify-correct, complete generation per problem.

## Decontamination

A document is dropped if it shares any word 13-gram with a protected problem (text lowercased, split on
non-alphanumerics; statements of 8-12 words use their full length as the n-gram; statements under 8 words are
skipped). Protected sets:
- eval: MATH-500, GSM8K test, AIME 2024/2025/2026, AMC23, HMMT Feb 2025/2026;
- the held-out pools below (they are excluded from pretraining).

## Held-out problem pools

- `heldout_pool.jsonl` (`--pool`, 20,000): NuminaMath-1.5 math word problems with integer answers, 40-800 characters,
  no figure, not marked invalid. Rows `{id, problem, answer, source, problem_type, synthetic}`.
- `heldout_pool_bigmath.jsonl` (`--bigmath`, `--pool2` 30,000): Big-Math-RL-Verified problems with numeric answers
  (integer, decimal, `a/b`, `\frac{a}{b}`), 40-800 characters, no figure. Same fields plus `llama8b_solve_rate`.

Both are drawn round-robin over sources with seed 0, after dropping problems that hit the eval 13-grams.

## Mixes (estimated tokens = UTF-8 bytes / 4)

| source (HF) | kind | v1 | v2 |
|---|---|---|---|
| finemath-4plus (15 files) | web | 1.95B | - |
| Nemotron-CC-Math-v1 4plus (2 files) | web | - | 1.70B |
| Nemotron-CC-Math-v1 4plus_MIND (1 file) | textbook dialogue | - | 0.74B |
| cosmopedia auto_math_text (v1 11 files, v2 5) | textbook | 0.83B | 0.37B |
| cosmopedia khanacademy | textbook | 0.02B | 0.02B |
| NuminaMath-CoT | short QA | 0.28B | 0.25B |
| NuminaMath-1.5 | short QA | 0.23B | 0.21B |
| OpenMathInstruct-2 train_1M | short QA | 0.31B | 0.30B |
| MetaMathQA-395K | short QA | 0.05B | 0.03B |
| OpenR1-Math-220k | trace | 0.22B | 0.20B |
| OpenMathReasoning cot (9 files) | trace | 1.04B | 1.00B |
| total | | 4.93B | 4.83B |

v2 also excludes the Big-Math pool. Licences: finemath ODC-BY; cosmopedia, NuminaMath, OpenR1 Apache-2.0;
OpenMathInstruct-2 and OpenMathReasoning CC-BY-4.0; MetaMathQA MIT; Big-Math Apache-2.0. Nemotron-CC-Math is under the
NVIDIA Data Agreement for Model Training, which does not allow redistributing the data.
