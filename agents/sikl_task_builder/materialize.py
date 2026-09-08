"""Deterministic task emission; only the input adapter is builder-editable."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import yaml

from . import BUILDER_VERSION
from .bundle import ImportProblem, TaskSpec, fingerprint, source_files
from .config import Config

TEMPLATES = Path(__file__).with_name("templates")
EDITABLE = "scripts/task_inputs.py"


def task_files(task: TaskSpec, config: Config) -> dict[str, str]:
    from src.preprocessing import _resolve_gfx_arch
    arch = _resolve_gfx_arch(config.target_gpu_model)
    if not arch:
        raise ImportProblem("unsupported_platform", config.target_gpu_model)
    aliases = {arch.lower(), config.target_gpu_model.lower()}
    for role, solution in (("baseline", task.baseline), ("reference", task.reference)):
        targets = solution["spec"].get("target_hardware", [])
        if not targets or not aliases.intersection(str(t).lower() for t in targets):
            raise ImportProblem("platform_deferred", f"{role} hardware {targets} does not match {config.target_gpu_model}")
    contract = {**task.contract(), "policy": config.policy}
    files = {"scripts/__init__.py": "", "source/__init__.py": "",
             # Tuple outputs follow the definition's insertion order.
             "scripts/workload.json": json.dumps(contract, indent=2) + "\n"}
    for name in ("task_api.py", "task_runner.py", "task_inputs.py"):
        files[f"scripts/{name}"] = (TEMPLATES / name).read_text()
    for role, solution in (("baseline", task.baseline), ("reference", task.reference)):
        for path, content in source_files(solution).items():
            files[f"scripts/{role}/{path}"] = content
    for path, content in source_files(task.baseline).items():
        files[f"source/implementation/{path}"] = content
    entry = repr(task.baseline["spec"]["entry_point"])
    files["source/kernel.py"] = (
        '"""Initial production baseline. Replace run with your Triton implementation."""\n'
        "from pathlib import Path\nfrom scripts.task_api import load_solution\n\n"
        f"_initial = load_solution(Path(__file__).parent / 'implementation', {entry})\n\n"
        "def run(**kwargs):\n    return _initial(**kwargs)\n"
    )
    editable_sources = sorted(p for p in files if p.startswith("source/") and p.endswith(".py") and not p.endswith("__init__.py"))
    cfg = {
        "task_type": "instruction2triton", "source_file_path": editable_sources,
        "target_kernel_functions": ["run"],
        "compile_command": ["python3 scripts/task_runner.py --mode compile"],
        "correctness_command": ["python3 scripts/task_runner.py --mode correctness"],
        "performance_command": ["python3 scripts/task_runner.py --mode performance"],
        "compile_timeout": config.command_timeout, "correctness_timeout": config.command_timeout,
        "performance_timeout": config.command_timeout,
        "platform_support": {"required_arch": arch, "status": "active"},
        "prompt": {"instructions": (
            f"Implement {task.task_id} in Triton with the source/kernel.py run(**kwargs) interface. "
            "The initial source is the production baseline, not a completed rewrite. "
            "Read scripts/workload.json and the protected reference for the full contract. "
            "Implement your own GPU computation; do not delegate it to the protected baseline "
            "or a library product. Keep all workload cases, input/output dtypes and semantics. "
            "Inputs are functional and may not be mutated. Edit only declared source files."
        )},
    }
    files["config.yaml"] = yaml.safe_dump(cfg, sort_keys=False)
    files["scripts/provenance.json"] = json.dumps({
        "builder_version": BUILDER_VERSION, "source_digest": task.digest,
        "definition": task.task_id, "origins": task.origins,
        "source_solutions": {
            role: {k: v for k, v in solution.items() if k != "sources"}
            for role, solution in (("baseline", task.baseline), ("reference", task.reference))
        },
        "policy": config.policy, "synthesized_inputs": True,
        "reference_modified": False,
    }, indent=2, sort_keys=True) + "\n"
    return files


def materialize_task(task: TaskSpec, config: Config, destination: Path) -> dict:
    files = task_files(task, config)
    if destination.exists():
        raise ImportProblem("destination_exists", f"Refusing to replace existing draft: {destination}")
    destination.mkdir(parents=True)
    for relative, content in files.items():
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return {"task_id": task.task_id, "files": sorted(files), "editable": [EDITABLE]}


def task_tree(root: Path) -> dict[str, str]:
    files = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.is_symlink():
            raise ImportProblem("symlink", f"Symlink in task: {relative}")
        if "__pycache__" in relative.parts or relative.parts[0] == "build":
            continue
        if path.is_file():
            import hashlib
            files[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return files


def task_digest(root: Path) -> str:
    return fingerprint(task_tree(root))


def check_contract(task: TaskSpec, config: Config, draft: Path) -> dict:
    expected = task_files(task, config)
    actual = task_tree(draft)
    diagnostics = []
    for relative, content in expected.items():
        path = draft / relative
        if not path.is_file():
            diagnostics.append({"code": "missing_file", "path": relative})
        elif relative != EDITABLE and path.read_text() != content:
            diagnostics.append({"code": "contract_changed", "path": relative})
    for relative in actual:
        if relative not in expected:
            diagnostics.append({"code": "undeclared_file", "path": relative})
    for path in draft.rglob("*.py"):
        if "__pycache__" not in path.parts:
            try:
                ast.parse(path.read_text(), filename=str(path))
            except SyntaxError as exc:
                diagnostics.append({"code": "syntax_error", "path": str(path.relative_to(draft)), "message": str(exc)})
    return {"ok": not diagnostics, "task_id": task.task_id,
            "task_digest": task_digest(draft), "diagnostics": diagnostics}
