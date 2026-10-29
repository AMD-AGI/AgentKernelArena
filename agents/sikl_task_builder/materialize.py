"""Deterministic task emission; only the input adapter is builder-editable."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import yaml

from . import BUILDER_VERSION
from .bundle import ImportProblem, TaskSpec, case_manifest, fingerprint, source_files, solution_targets
from .checkout import SchemaTask
from .config import Config

TEMPLATES = Path(__file__).with_name("templates")
EDITABLE = "scripts/task_inputs.py"
RUNTIME_ROOT = "/opt/sikl"
SCHEMA_DESCRIPTION = f"""\
Make the SIKL v5 kernel in source/ faster on every workload of start.json. start.json is the committed
schema: its kernel ABI, workloads, reference and error bounds are fixed. Edit only the files under source/;
do not add or remove files, and keep the entry point a top-level def. The kernel must be correct for every
scalar value inside its declared range.

Each evaluation rebuilds the schema from source/ and runs `python -m sikl validate --stages audit,bench`
at the pinned SIKL commit, as a SIKL pull request that adds this kernel would be judged.

A committed kernel is exempt from SIKL's source rules SA0-SA9 only while its bytes are unchanged. Once
edited it is audited as a new kernel, and committed kernels rarely pass that audit, so the first change
must make the whole file comply: no `global`/`nonlocal` (SA1), no module-level mutable state or caches
(SA2), no functools caching except the single `@functools.cache` compile function (SA3), imports limited to
a few standard modules, torch, flydsl and triton (SA4), no forbidden names (SA5), no mutation of non-local
objects or inputs (SA6, SA7), no allocation or other side effect in launch (SA8). The rules are in
{RUNTIME_ROOT}/sikl/ops/README.md; the authority is
{RUNTIME_ROOT}/sikl/core/validation/stages/_audit/_rules.py (AKA_SIKL_ROOT replaces {RUNTIME_ROOT}).
A correctness FAIL names every rejected line or failing workload.
"""


def authoring_editable(task: TaskSpec) -> list[str]:
    # A supplied initializer owns the distribution and cannot be repaired by
    # choosing an easier input population after a numerical failure.
    return [] if task.definition.get("initialize") else [EDITABLE]


def task_files(task: TaskSpec, config: Config) -> dict[str, str]:
    from src.preprocessing import _resolve_gfx_arch
    arch = _resolve_gfx_arch(config.target_gpu_model)
    if not arch:
        raise ImportProblem("unsupported_platform", config.target_gpu_model)
    aliases = {arch.lower(), config.target_gpu_model.lower()}
    for role, solution in (("baseline", task.baseline), ("reference", task.reference)):
        targets = solution_targets(solution)
        if not any(t["arch"] == arch if "arch" in t else t["hardware_id"].lower() in aliases for t in targets):
            raise ImportProblem("platform_deferred", f"{role} hardware {targets} does not match {config.target_gpu_model}")
    contract = {**task.contract(), "policy": config.policy,
                "cases": case_manifest(task.definition, task.rows)}
    files = {"scripts/__init__.py": "", "source/__init__.py": "",
             # Tuple outputs follow the definition's insertion order.
             "scripts/workload.json": json.dumps(contract, indent=2) + "\n"}
    for name in ("task_api.py", "task_runner.py", "task_inputs.py"):
        files[f"scripts/{name}"] = (TEMPLATES / name).read_text()
    for name in ("initialize", "compare"):
        if task.definition.get(name):
            files[f"scripts/{name}/main.py"] = task.definition[name]
    for role, solution in (("baseline", task.baseline), ("reference", task.reference)):
        for path, content in source_files(solution).items():
            files[f"scripts/{role}/{path}"] = content
    for path, content in source_files(task.baseline).items():
        files[f"source/implementation/{path}"] = content
    entry = repr(task.baseline["spec"]["entry_point"])
    files["source/kernel.py"] = (
        f'"""Initial production baseline. Replace run with your {config.target_language} implementation."""\n'
        "from pathlib import Path\nfrom scripts.task_api import load_solution\n\n"
        f"_initial = load_solution(Path(__file__).parent / 'implementation', {entry})\n\n"
        "def run(**kwargs):\n    return _initial(**kwargs)\n"
    )
    editable_sources = sorted(p for p in files if p.startswith("source/") and p.endswith(".py") and not p.endswith("__init__.py"))
    cfg = {
        "schema_version": 2,
        "candidate": {"language": config.target_language, "initial_state": "implemented",
                      "initial_language": task.baseline["spec"]["language"],
                      "editable": editable_sources,
                      "entrypoints": [{"file": "source/kernel.py", "kind": "function", "symbol": "run"}]},
        "baseline": {"kind": "provided", "language": task.baseline["spec"]["language"],
                     "source_files": sorted(p for p in files if p.startswith("scripts/baseline/"))},
        "evaluation": {"runner": ["python3", "scripts/task_runner.py"],
                       "workloads": "scripts/workload.json", "timeout_s": config.command_timeout},
        "platform_support": {"required_arch": arch, "status": "active"},
        "description": (
            f"Implement {task.task_id} in {config.target_language} with the source/kernel.py run(**kwargs) interface. "
            "The initial source is the production baseline, not a completed rewrite. "
            "Read scripts/workload.json and the protected reference for the full contract. "
            "The supported axis and scalar combinations are exactly those enumerated by "
            "the supplied workload rows; tensor values vary according to the task initializer "
            "and replay-refill checks. The source definition documents the upstream operator "
            "and does not extend this task to unlisted axis or scalar combinations. "
            "Implement your own GPU computation; do not delegate it to the protected baseline "
            "or a library product. Keep all workload cases, input/output dtypes and semantics. "
            "Inputs are functional and may not be mutated. Edit only declared source files."
        ),
    }
    from src.task_spec import TaskSpec as ArenaTaskSpec
    ArenaTaskSpec.from_mapping(cfg, task_id=config.arena_task_id(task.task_id))
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
        "callback_sources": {name: f"scripts/{name}/main.py" for name in ("reference", "initialize", "compare")
                             if task.definition.get(name)},
    }, indent=2, sort_keys=True) + "\n"
    return files


def materialize_task(task: TaskSpec, config: Config, destination: Path) -> dict:
    files = task_files(task, config)
    write_files(destination, files)
    return {"task_id": task.task_id, "files": sorted(files), "editable": authoring_editable(task)}


def write_files(destination: Path, files: dict[str, str]) -> None:
    if destination.exists():
        raise ImportProblem("destination_exists", f"Refusing to replace existing draft: {destination}")
    destination.mkdir(parents=True)
    for relative, content in files.items():
        path = destination / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def schema_task_files(task: SchemaTask, config, commit: str) -> dict[str, str]:
    """A v5 task judges source/ with `sikl validate`, so it carries the committed schema, not semantics."""
    from src.preprocessing import _resolve_gfx_arch
    arch = _resolve_gfx_arch(config.target_gpu_model)
    kernel = task.data["kernel"]
    entry_file, _, symbol = kernel["entry_point"].partition("::")
    cfg = {
        "schema_version": 2,
        "description": SCHEMA_DESCRIPTION,
        "kernel_identity": {"logical_operator": task.name},
        "candidate": {"language": task.language, "initial_state": "implemented",
                      "editable": [{"path": f"source/{s['path']}", "scope": "file"} for s in kernel["sources"]],
                      "entrypoints": [{"file": f"source/{entry_file}", "kind": "function", "symbol": symbol}]},
        "baseline": {"kind": "provided", "language": task.language, "source_files": ["start.json"]},
        "evaluation": {"runner": ["python3", "-I", "-B", "scripts/task_runner.py"], "timeout_s": config.command_timeout},
        "platform_support": {"required_arch": arch, "status": "active"},
        "exports": [{"format": "sikl-schema", "output": "export.json",
                     "command": ["python3", "-I", "-B", "scripts/export_schema.py"], "timeout_s": 60}],
    }
    from src.task_spec import TaskSpec as ArenaTaskSpec
    ArenaTaskSpec.from_mapping(cfg, task_id=config.arena_task_id(task.task_id))
    files = {"config.yaml": yaml.safe_dump(cfg, sort_keys=False), "start.json": task.text}
    for source in kernel["sources"]:
        files[f"source/{source['path']}"] = source["content"]
    for name in ("task_runner.py", "export_schema.py"):
        files[f"scripts/{name}"] = (TEMPLATES / "sikl_v5" / name).read_text()
    files["scripts/provenance.json"] = json.dumps({
        "builder_version": BUILDER_VERSION, "sikl_commit": commit, "schema_path": task.path,
        "oid": task.oid, "name": task.name, "uuid": task.uuid, "arch": arch,
    }, indent=2, sort_keys=True) + "\n"
    return files


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
        elif relative not in authoring_editable(task) and path.read_text() != content:
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
