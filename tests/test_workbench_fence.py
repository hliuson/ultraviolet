"""Workbench fence (workbench/README.md): static import checks over the source tree, plus the wb_ registration rule."""
import ast
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SEAM = ROOT / "ultraviolet" / "tasks" / "score.py"
SYNTH_BANNED = ("workbench.envs", "ultraviolet.tasks",
                "rl_grpo", "rl_stream", "rl_sampler")


def py(d):
    return sorted((ROOT / d).rglob("*.py"))


def imports(path):
    out = []
    for n in ast.walk(ast.parse(path.read_text())):
        if isinstance(n, ast.Import):
            out += [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            out += [n.module] + [f"{n.module}.{a.name}" for a in n.names]
    return out


def hits(mods, banned):
    return [m for m in mods if any(m == b or m.startswith(b + ".") for b in banned)]


def test_core_does_not_import_workbench():
    for f in py("ultraviolet") + py("scripts"):
        bad = hits(imports(f), ("workbench",))
        assert bad == (["workbench.envs"] if f == SEAM else []), (str(f), bad)


def test_synth_stays_generic():
    for f in py("workbench/synth"):
        assert not hits(imports(f), SYNTH_BANNED), (str(f), hits(imports(f), SYNTH_BANNED))


def test_envs_do_not_import_synth():
    synth_mods = ("workbench.synth",) + tuple(p.stem for p in py("workbench/synth"))
    for f in py("workbench/envs"):
        assert not hits(imports(f), synth_mods), (str(f), hits(imports(f), synth_mods))


def test_env_tasks_are_wb_prefixed():
    from ultraviolet.tasks.score import TASKS
    import ultraviolet.tasks.heldout  # noqa: F401  register the ho_ tasks first so they are not counted as env tasks
    before = dict(TASKS)
    import workbench.envs  # noqa: F401
    new = set(TASKS) - set(before)
    assert all(t.startswith("wb_") for t in new), sorted(new)
    assert all(TASKS[t] is v for t, v in before.items()), "a workbench env overwrote a core task"


def test_wb_task_triggers_lazy_import():
    from ultraviolet.tasks.score import score
    sys.modules.pop("workbench.envs", None)
    score("wb_does_not_exist", "x</think>\n\\boxed{1}", {"answer": "1"})  # unknown task scores 0, but imports envs
    assert "workbench.envs" in sys.modules
