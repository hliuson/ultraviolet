# Workbench RL environments (graded)

Code: `workbench/envs/` (`resid.py`, `allsol.py`, `grid.py`, `sysword.py`, shared helpers in `common.py`). Tests:
`tests/test_wb_*.py`. Tasks register in `ultraviolet.tasks.score.TASKS` and are scored like every other task
(docs/tasks.md). Rewards are graded in [0, 1]; solved means reward == 1.0.

## Build

```bash
python -m workbench.envs.<env> build --out DIR [--n-rl 5000] [--n-eval 300] [--n-sft 0] [--seed 0] \
    [--tier-weights 1,1,1] [--gr-chance 0.1]
python -m workbench.envs.<env> show [--n 2]        # print sample problems per tier
```

`<env>` is `resid`, `allsol`, `grid` or `sysword`. Writes `rl.jsonl`, `eval.jsonl` (and `sft.jsonl` with `--n-sft`),
`filters.json` (`eval_guess_resistant_ids`: eval items with `meta.chance <= --gr-chance`) and `summary.json`. Items are
tier-stratified, disjoint across splits by normalised problem text, and every gold answer is checked to score 1.0.

Use with RL: `rl_grpo.py --pool DIR/rl.jsonl --eval-problems DIR/eval.jsonl --filters DIR/filters.json`. Graded
rewards work with `--adv grpo` and the uniform / pool samplers.

## Row format

The docs/tasks.md row plus top-level `family` and `tier` (0-2, easy to hard), and `meta.chance`: the expected reward
of a well-formed random answer (analytic or seeded Monte Carlo). Answers go in `\boxed{a, b, c}`; pairs as `(x, y)`;
fractions as `a/b` or `\frac{a}{b}`; the empty set as `\boxed{none}`. A format sentence with an example precedes the
standard `Put the final answer in \boxed{}.`

## Rewards

- Vector answers (wb_resid, wb_sysword, wb_grid): `vec_credit` = matches at aligned positions / max(len pred,
  len gold). A correct first component alone earns 1/k; extra components dilute the score.
- Set answers (wb_allsol): F1 between the deduplicated predicted set and the gold set; an empty gold set scores 1
  only for `none`.

## wb_resid: residue vectors

"Find the remainders when N is divided by 4, 9 and 11, in that order." Answer: the residues.
- `explicit`: tier 0, N with 5-6 digits and 2 moduli with digit rules {3, 4, 5, 8, 9, 10, 11}; tier 1 (25%), N with
  7-12 digits and 2-3 moduli <= 13 including one of {7, 12, 13}.
- `power`: tier 1 (75%), a^e with 2 <= a <= 12, 5 <= e <= 30, 2-3 moduli <= 13.
- `expr`: tier 2, two-term sums / differences / products of powers (e <= 100), 3 moduli from
  {3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 16, 25}.
- No modulus divides another within an item; at most one residue is trivial. chance = mean(1 / m_i).

## wb_allsol: find all solutions

Set-valued; tiers grow the domain and the number of solutions.
- `divisors`: all positive divisors of n (24 <= n <= 400, 6-16 divisors).
- `quadmod`: all x in [0, m) with x^2 + bx + c = 0 (mod m), 5 <= m <= 30, 0-6 solutions.
- `introots`: all integer roots of an expanded monic cubic or quartic with roots in [-9, 9] (repeated roots listed
  once).
- `diophant`: all non-negative integer solutions (x, y) of ax + by = c, 1-5 solutions; answer as pairs.
- `primes`: all primes in [a, b], span 20-60.
- chance = Monte Carlo mean F1 of a random subset of the family's domain.

## wb_grid: line-sum grid completion

Fill the blanks of a grid of positive integers; the answer is the missing numbers in reading order (the problem
states how many). Completions are unique (checked by exact rank).
- `magic3`: 3x3 magic square, magic sum stated, 3 given cells that pin it. Tier 0: centre given, c <= 12; tier 1:
  centre hidden, c <= 20; tier 2: centre hidden, c <= 40.
- `sumgrid`: all row and column sums given. Tier 0: 3x3, 3-4 blanks, entries <= 20; tier 1: 3x4, 5-6 blanks, entries
  <= 50, "fill lines with one blank" cascade of >= 2 rounds; tier 2: 4x4, 7 blanks, cascade of >= 3 rounds.
- chance: Monte Carlo over uniform blank values.

## wb_sysword: multi-unknown word problems

2-3 integer unknowns, answered in the order asked. Families `ages`, `coins`, `shares`, `digits`, `rect` and `pure`
(plain linear systems). Tier 0: 2 unknowns, numbers <= 50; tier 1: 2-3 unknowns, <= 200; tier 2: 3 unknowns, <= 1000.
Unknowns are sampled first and the relations derived from them; the system is solved exactly and kept only if the
solution is unique. chance ~ 0.01.
