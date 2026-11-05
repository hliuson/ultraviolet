"""Workbench RL environments. Importing this package registers every env's `wb_*` tasks in
ultraviolet.tasks.score.TASKS (score.py imports it lazily on the first `wb_` task). See workbench/README.md."""
# One import per env module, e.g.:
# from workbench.envs import my_env  # noqa: F401
from workbench.envs import resid  # noqa: E402,F401  wb_resid: residue vectors (docs/wb_envs.md)
from workbench.envs import allsol  # noqa: E402,F401  wb_allsol: find all solutions (F1)
from workbench.envs import grid  # noqa: E402,F401  wb_grid: line-sum grid completion
from workbench.envs import sysword  # noqa: E402,F401  wb_sysword: multi-unknown word problems
from workbench.envs import paramobj  # noqa: E402,F401  wb_po_*: param-object construction tasks (docs/wb_paramobj.md)

# Optional local envs: each package in workbench/envs/local/ (gitignored) registers its wb_ tasks the same way.
import importlib as _il, os as _os, pkgutil as _pk  # noqa: E402
_LOCAL = _os.path.join(_os.path.dirname(__file__), "local")
if _os.path.isfile(_os.path.join(_LOCAL, "__init__.py")):
    for _m in _pk.iter_modules([_LOCAL]):
        _il.import_module(f"workbench.envs.local.{_m.name}")
