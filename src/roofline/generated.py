"""Tier 3: count work from a case's generated inputs (paged / ragged attention and the like).

The inputs are rebuilt the way the evaluator builds them: the framework seeds of the case, the
built-in distributions drawn in declaration order from the evaluator's own helpers, then the task's
``generator``. The evaluator draws on the GPU, and torch's CPU and GPU generators produce different
streams from the same seed, so only a GPU run with the scoring image's torch reproduces the timed
inputs; a CPU run gives only a statistically similar sample, so it emits no data-model floor
(``floor.task_floors`` keeps the shape-only floor or tier ``none``). The evaluator times both seeds of a case, so the smaller work is used.

Data models only read tensors through ``.tolist()`` and ``.shape``, so they are unit-tested with
plain stand-ins.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any, Callable

from ..task_contract import TaskContract
from .interface import CaseShapes
from .operators import Estimate

DataModel = Callable[[CaseShapes, dict[str, Any]], Estimate]
DATA_MODELS: dict[str, tuple[str, DataModel]] = {}


def register_data_model(operator: str, label: str, model: DataModel) -> None:
    if operator in DATA_MODELS:
        raise ValueError(f"operator {operator!r} already has a data model")
    DATA_MODELS[operator] = (label, model)


def _task_function(task_dir: Path, task_id: str, entry: dict) -> Callable:
    path = (task_dir / entry["file"]).resolve(strict=True)
    if not path.is_relative_to(task_dir.resolve()):
        raise ValueError(f"{task_id}: generator escapes the task")
    name = "_aka_roofline_generator_" + "".join(c if c.isalnum() else "_" for c in task_id)
    module = sys.modules.get(name)
    if module is None:
        module = _load_task_module(name, path, task_dir.resolve())
    return getattr(module, entry["symbol"])


def _load_task_module(name: str, path: Path, root: Path):
    """Load a task file the way the evaluator does: its directory on ``sys.path`` for sibling imports.

    The path and the bytecode setting are restored afterwards, no ``__pycache__`` is written into the
    task package, and sibling modules are dropped from ``sys.modules`` so another task's file of the
    same name is not shadowed by this one.
    """
    before, saved_path, saved_bytecode = set(sys.modules), list(sys.path), sys.dont_write_bytecode
    sys.path.insert(0, str(path.parent))
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    finally:
        sys.path[:] = saved_path
        sys.dont_write_bytecode = saved_bytecode
        for added in set(sys.modules) - before:
            origin = getattr(sys.modules[added], "__file__", None)
            if origin and Path(origin).resolve().is_relative_to(root):
                del sys.modules[added]
    return module


def default_device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def generated_inputs(task_dir: Path, contract: TaskContract, case_id: str, seed_index: int,
                     device: str = "cuda") -> dict[str, Any]:
    """The logical inputs the evaluator generates for one case and seed (before any layout transform)."""
    import torch  # noqa: F401  (fail early with a clear ImportError)
    from ..tools.arena_eval import values
    from ..tools.arena_eval.contract import Contract

    resolved = Contract(contract.resolved_contract(contract.baseline_language))
    case = resolved.case(case_id)
    symbols, scalars = resolved.symbols(case), resolved.scalars(case)
    target = torch.device(device)
    generator = values.device_generator(case.seeds[seed_index], target)
    logical: dict[str, Any] = {}
    owned = []
    for name, declaration in resolved.inputs.items():
        if declaration["kind"] == "scalar":
            logical[name] = scalars[name]
        elif declaration["distribution"]["kind"] == "generator":
            owned.append(name)
        else:
            logical[name] = values._builtin(declaration, resolved.shape(declaration, symbols), generator, target)
    if owned:
        function = _task_function(task_dir, contract.task_id, contract.interface["generator"])
        produced = function({"id": case.case_id, "axes": dict(symbols), "scalars": dict(scalars)},
                            dict(logical), generator, target)
        logical.update(produced)
    return logical


from .operators import attention_data, partial_data, routing_data  # noqa: E402,F401  (register data models)
