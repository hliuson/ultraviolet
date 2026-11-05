# wb_po_*: param-object construction tasks

Code: `workbench/envs/paramobj/` (`paramobj.py` families, `core.py` parsing / verification / scoring, `build.py` pool
builder). Tests: `tests/test_wb_paramobj.py`. Stdlib only.

Each item asks for a short integer tuple that parameterises a mathematical object (polynomial coefficients, matrix
entries, a lattice point, a permutation, ...) with stated properties. Every item states a finite answer box, so the
full set of valid answers is enumerated exactly; **any one valid answer is correct** (binary reward). Example:

> Find integers a, d with -10 <= a, d <= 10 such that the arithmetic progression a, a + d, a + 2d, ... has sum of its
> first 6 terms equal to 81. Several answers are possible; any one correct answer is accepted. Give one solution. End
> with the answer in the form \boxed{a = ..., d = ...}, with the dots replaced by the values.

Valid answers: `-9, 9`, `-4, 7`, `1, 5`, `6, 3`.

## Families (53, tiers 0-3)

Tier 0 uses small numbers and short derivations; tiers 1-3 grow the parameters and the answer box. Task name =
`wb_` + family (e.g. `wb_po_ap_sum`).

| domain | families (`po_` + name) |
|---|---|
| combinatorics | binary_runs, magic3, moment_weights, partition_sq, perm_cycles, perm_descents, perm_inversions, stats_multiset |
| function | lin_commute, mobius |
| geometry | circle_through, int_triangle, lattice_tri_area, parallelogram, pyth_perim |
| linalg | mat2_commute, mat2_eigvec, mat2_idempotent, mat2_involution, mat2_nilpotent, mat2_trdet, mat3_det, vec_cross, vec_dot |
| number | consecutive_sum, crt_range, diff_squares, egyptian, fraction_between, gauss_factor, gcd_lcm, lin2_box, lin_congruence, order_mod, pyth_leg, sfft, sigma_inv, sqrt_mod, tau_range, totient_inv, two_sq_comp |
| polynomial | poly_critical, poly_integral, poly_mixed, poly_roots, poly_slope, poly_values, quad_rootdiff, vieta3 |
| sequence | ap_sum, gp_term, recur_coef, recur_init |

## Building a pool

```bash
python -m workbench.envs.paramobj.build po --out DIR [--families po_ap_sum,...] [--tiers 0,1,2,3] \
    [--per-cell 4000] [--eval-per-cell 16] [--eval-pct 20] [--max-draws 100000]
```

Writes `rl_train.jsonl`, `eval_witness.jsonl` and `stats.json`. Items go to eval or rl by a crc32 hash of their
canonical params (`--eval-pct` to eval), so the splits are disjoint. Items are kept if they have 2-64 valid answers
and a blind-guess rate (valid answers / answer-space size) of at most 0.02. Generation is deterministic: candidate j of (family, tier) uses seed
`(crc32(family) + 1) * 10^8 + tier * 10^6 + j`, stored in `meta.seed`.

## Row format and scoring

Rows follow docs/tasks.md (`id, task, problem, prompt, answer, ...`) plus `family`, `domain`, `tier`, `n_valid`,
`guess_rate` and `meta` (`params`, `seed`, `valid_answers`). `ultraviolet.tasks.score` and `scripts/rl_grpo.py` score
them like any other task: the last `\boxed{}` after the single `</think>` must verify.

```python
from workbench.envs.paramobj import core
core.score(row, completion)   # {"correct", "key", "wellformed", "correct_strict", "pred"}
core.verify(row, "1, 5")      # True
core.enumerate_valid(row)     # canonical keys of every valid answer
```
