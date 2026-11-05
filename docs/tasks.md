# Verifiable task suite

Code: `ultraviolet/tasks/` (`score.py` scorers, `build.py` item builder, `heldout.py` held-out `ho_` tasks). Tests: `tests/test_tasks.py`.

## Row format

`{id, task, source, split, problem, prompt, answer, ref, meta}`
- `problem` ends with one answer instruction: `Put the final answer in \boxed{}.`
- `prompt` = `Problem:\n{problem}\n\n<think>\n`.
- `answer` == `ref` is the normalised gold; `meta` holds what the scorer needs (countdown numbers, sudoku puzzle,
  ...).

## Scoring

```python
from ultraviolet.tasks import score
r = score(row["task"], completion_text, row)   # completion_text = everything generated after the prompt
```

- The answer is read only from the text after the last `</think>`; no `</think>` scores 0
  (`require_close=False` lifts this for probes that prepend `</think>` themselves).
- The answer is the last balanced `\boxed{...}`; there is no bare-number fallback.
- Reward kind is set per task in `TASKS[task]["reward"]`: `binary` (default; 1.0 / 0.0; a scorer returning anything
  else raises TypeError) or `graded` (float in [0, 1]; solved = 1.0). All tasks below and the `ho_` tasks are binary;
  the workbench envs (`wb_*`) are graded. Evals count pass@k / maj@k on reward == 1.0.
- `rl_grpo.py` refuses methods that assume 0/1 rewards (`--adv maxrl|psr|nsr|wreinforce`, `--resample-zero`) on
  pools with graded tasks unless `--allow-binary-methods 1`.
- Dependencies: stdlib only, plus optional `math_verify` for `math` (otherwise normalised-string and exact-rational
  comparison). `TASKS_PYDEPS` may point to a directory holding `math_verify`, `latex2sympy2_extended` and `antlr4`.
- reasoning-gym's partial-credit `score_answer` is never used. Every built item's own boxed gold scores 1.

## Tasks

| task | source / settings | check |
|---|---|---|
| chain_sum | reasoning-gym (RG), 3-5 terms, 2-3 digits | exact rational (units, `x =`, `$`, commas tolerated) |
| products | own generator, 2-3 digit factors | exact |
| basic_arithmetic | RG, 3-4 terms, 1-2 digits, integer answers | exact |
| simple_equations | RG, 2-3 terms, values <= 50 | exact |
| gcd | RG, 2 numbers in [2, 300]; gcd = 1 items subsampled | integer |
| lcm | RG, 2 numbers in [2, 100] | integer |
| fraction_simplification | RG, value <= 60, factor 2-20 | must equal the reduced fraction (`a/b`, `\frac`, `\dfrac`) |
| base_conversion | RG, bases 2-16, value <= 300 | lowercase string; `0x`, subscripts, leading zeros stripped |
| count_bits | RG, n <= 65535 | integer |
| time_intervals | RG `time`, `time_seconds` | same field count, numeric equality |
| calendar_arithmetic | RG weekday_offset, count_days, offsets <= 30 | weekday name (case-insensitive) or integer |
| leg_counting | RG, 2-4 animals, <= 10 each; leg counts stated | exact |
| gsm_symbolic | RG, difficulty 1.0 | exact |
| countdown | RG, 3-4 numbers <= 50, target 10-100 | expression with + - * / and parens using exactly the given numbers, value == target |
| mini_sudoku | RG 4x4, 4-8 blanks | any valid completion consistent with the puzzle |
| knights_knaves | RG, 2 people | each name gets exactly one role |
| gsm8k | openai/gsm8k (test -> eval) | exact rational |
| math | hendrycks_math Level 1-2 (test -> eval) | normalised string, exact number, then math_verify |

`letter_counting` and `spell_backward` remain registered but are excluded from built datasets (`DROPPED_TASKS`):
they depend on the characters inside multi-character tokens.

## Splits

`build.py` writes `OUT/<task>/{eval,rl,sft}.jsonl` and `stats.json`. Splits are built in the order eval, rl, sft; an
item whose normalised question already appears in an earlier split is dropped, so the splits are disjoint by question
text. reasoning-gym item i of split s uses `Random(seed_s + i)` with seeds eval 30,000,000, rl 20,000,000,
sft 10,000,000. HF datasets split by their own train/test split, then by a crc32 hash of the row id.
