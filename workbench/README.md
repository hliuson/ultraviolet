# workbench: synthetic data and RL environment work

Synthetic training data generators and RL environments for UV-50M. They build on the core (`ultraviolet/`,
`scripts/`) but are fenced off from it, and the two halves are fenced off from each other.
`tests/test_workbench_fence.py` enforces the rules below.

```
workbench/
  synth/   synthetic plain-text data generators (midtrain / SFT data)
  envs/    RL environments: task items + verifiers, scored through ultraviolet.tasks
```

## Fence rules

1. **Core never imports workbench.** The one exception is the lazy hook in `ultraviolet/tasks/score.py`: a task whose
   name starts with `wb_` imports `workbench.envs`, which registers its scorers. Nothing else in `ultraviolet/` or
   `scripts/` may reference `workbench`.
2. **synth stays generic.** Generators produce upstream-style math text only. They never import `workbench.envs`,
   `ultraviolet.tasks` or the RL trainer. They never emit RL task families or answer
   formats: midtrain data stays generic.
3. **envs never import synth.** An env defines problems and checks answers. It does not borrow a data generator's
   renderer or distribution.
4. **Env task names start with `wb_`.** Registering any other name, or overwriting an existing `TASKS` entry, is an
   error.

## synth/

Plain Python CLIs, seeded and deterministic. Each output row is checked against exact arithmetic before it is kept.

- `gen_arith_steps.py`: arith-steps-v1, worked multi-digit arithmetic (add/sub/mul/div, fractions, powers, order of
  operations, mod). Operand lengths follow the bundled `operand_dist.json`, measured on midtrain reasoning traces.
- `gen_math_procedures.py`: math-procedures-v1, 13 families of short executable procedures, checked with sympy. Rows
  that match a statement form reserved for RL tasks are rejected.

arith-steps-v1 and math-procedures-v1 are in ultraviolet-50m-mid-alpha's SFT data.
`--tok` only counts tokens to stop at the target; the released slices counted with an earlier 16k tokenizer.

## envs/

An env module registers its tasks in `ultraviolet.tasks.score.TASKS`:

```python
from ultraviolet.tasks.score import TASKS
TASKS["wb_<name>"] = {"kind": "math", "scorer": sc_<name>}   # scorer(pred, item) -> bool; pred = last \boxed{} content
# graded (partial-credit) env: scorer returns a float in [0, 1] and the entry must say so:
TASKS["wb_<name>"] = {"kind": "list", "scorer": sc_<name>, "reward": "graded"}
```

Add the import to `envs/__init__.py`. Pool rows follow the item format in docs/tasks.md
(`{id, task, problem, prompt, answer, ...}`, with `problem` ending in `Put the final answer in \boxed{}.`). With
that, `rl_grpo.py` and the evals score `wb_` tasks unchanged. The released envs are documented in docs/wb_envs.md and
docs/wb_paramobj.md.
