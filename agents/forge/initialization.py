"""Initialize declared HIP/Triton candidates with KernelForge's implementer.

This is a correctness-only Forge phase, not another Arena agent. The subsequent
search is the unmodified forge-loop algorithm with an independent timing anchor.
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
from pathlib import Path
import stat
import subprocess
import time

from agents.forge import bridge
from agents.forge.bundles import candidate_files, protected_paths
from agents.forge.task_context import TaskContext
from src.task_spec import resolve_task_path


def materialize_targets(spec, root: Path, anchor: str) -> None:
    """Make missing *declared* files visible to Forge/Git; preserve existing code.

    Empty files are not accepted implementations: the task's compiler and full
    correctness check must pass after the actual implementer session.
    """
    paths = {anchor, *(scope.path for scope in spec.candidate.editable if scope.scope != "tree")}
    for relative in sorted(paths):
        if relative in protected_paths(spec) and not any(
            scope.scope == "symbols" and scope.path == relative for scope in spec.candidate.editable
        ):
            raise ValueError(f"Initialization target overlaps a protected path: {relative}")
        path = resolve_task_path(root, relative)
        if path.exists():
            if not path.is_file():
                raise ValueError(f"Initialization target is not a file: {relative}")
            continue
        if any(scope.path == relative and scope.scope == "symbols" for scope in spec.candidate.editable):
            raise ValueError(f"Cannot reconstruct a missing symbol-scoped harness: {relative}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    candidate_files(spec, root)  # Reject links and unresolved declarations now.


def _git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def protected_inventory(spec, root: Path) -> dict:
    editable = candidate_files(spec, root, required=False)
    names = _git(root, "ls-files", "-z").split("\0")
    inventory = {}
    for relative in filter(None, names):
        if relative in editable:
            continue
        path = root / relative
        info = path.lstat()
        payload = os.readlink(path).encode() if stat.S_ISLNK(info.st_mode) else path.read_bytes()
        inventory[relative] = (info.st_mode, hashlib.sha256(payload).hexdigest())
    return inventory


def _bundle_hashes(spec, root):
    return {name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in candidate_files(spec, root).items()}


def _save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _new_path_patterns(spec, root: Path) -> list[str]:
    """Authorize new files only in declared trees and existing directories."""
    patterns = []
    for scope in spec.candidate.editable:
        if scope.scope != "tree":
            continue
        tree = resolve_task_path(root, scope.path, must_exist=True)
        for directory in [tree, *sorted(path for path in tree.rglob("*") if path.is_dir())]:
            relative = directory.relative_to(root).as_posix()
            resolve_task_path(root, relative, must_exist=True)
            patterns.append(relative + "/*")
    return patterns


def _make_initialization_agent(config, *, new_path_patterns=(), **options):
    """Bind either published implementer API without dropping gate settings."""
    from kernelforge.orchestrator import agent

    if "setup" not in inspect.signature(agent.make_agent_fn).parameters:
        return agent.make_agent_fn(config=config, **options)
    from kernelforge.loop.gate_settings import GateMeasurement

    task_keys = ("program_md", "kernel_backend_name", "profiling_enabled", "task_type",
                 "source_files", "target_functions", "extra_protected_paths")
    session_keys = ("session_timeout_sec", "permission_mode", "insession_gate")
    task = agent.ImplementerTask(**{key: options.pop(key) for key in task_keys},
                                 commit_new_paths=list(new_path_patterns))
    session = agent.SessionOptions(**{key: options.pop(key) for key in session_keys})
    timeout = options.pop("validation_timeout_sec")
    measurement = GateMeasurement(
        driver_script=options.pop("driver_script"),
        interposed_driver_path=options.pop("interposed_driver_path"),
        correctness_only=options.pop("correctness_only"),
        stage_timeout_sec=timeout, bench_timeout_sec=timeout,
    )
    usage = options.pop("usage")
    if options:
        raise ValueError(f"Unmapped Forge implementer settings: {sorted(options)}")
    return agent.make_agent_fn(config=config,
                              setup=agent.ImplementerSetup(task=task, measurement=measurement, session=session),
                              usage=usage)


async def initialize(plan: dict) -> dict:
    # Caller checks the pinned upstream before entering this phase.
    from kernelforge.config import Config
    from kernelforge.tracker.usage import UsageAccumulator
    from agents.forge.program import program_text

    context = TaskContext.load(plan["context"])
    spec, root = context.spec, Path(plan["engine_root"])
    if spec.candidate.language not in ("hip", "triton"):
        raise ValueError("The generic Forge initialization phase supports HIP and Triton")
    settings = plan["agent_config"]
    report = {"status": "RUNNING", "target_language": spec.candidate.language,
              "mechanism": "kernelforge.orchestrator.agent.make_agent_fn",
              "attempts": [], "arena_verdict": "pending"}
    report_path = plan["initialization_result"]
    def checkpoint(totals=None):
        report["llm_usage"] = totals if totals is not None else usage.totals()
        _save(report_path, report)
    usage = UsageAccumulator(on_update=checkpoint)
    checkpoint()
    remaining = plan["deadline_unix"] - time.time()
    phase_deadline = time.time() + remaining * settings["initialization_budget_fraction"]
    plan["phase_deadline_unix"] = phase_deadline
    _save(os.environ["ARENA_FORGE_PLAN"], plan)
    starting_head = _git(root, "rev-parse", "HEAD")
    protected = protected_inventory(spec, root)
    history = "No valid target implementation exists yet. Implement the declared task interface."
    try:
        for attempt in range(1, settings["initialization_max_attempts"] + 1):
            remaining = phase_deadline - time.time()
            if remaining < 1:
                raise TimeoutError("Forge initialization exhausted its reserved phase budget")
            timeout = max(1, int(min(settings["session_timeout_seconds"], remaining)))
            kwargs = dict(workspace=str(root), gpu_target=plan["gpu_arch"], gpu_type=plan["gpu_type"],
                          agent_backend=settings["agent_backend"], agent_timeout_sec=timeout)
            if settings.get("model"):
                kwargs["agent_model"] = settings["model"]
            config = Config.from_env(**kwargs)
            files = candidate_files(spec, root)
            agent = _make_initialization_agent(config=config, new_path_patterns=_new_path_patterns(spec, root),
                program_md=program_text(plan, initialize=True),
                kernel_backend_name=spec.candidate.language, usage=usage,
                insession_gate=True, correctness_only=True,
                driver_script=str(root / "arena_forge_driver.py"),
                interposed_driver_path=str(root / "arena_forge_driver.py"),
                session_timeout_sec=timeout, validation_timeout_sec=timeout,
                permission_mode=settings["permission_mode"], profiling_enabled=False,
                task_type="repository", source_files=list(map(str, files.values())),
                target_functions=[entry.symbol for entry in spec.candidate.entrypoints if entry.symbol],
                extra_protected_paths=[str(root / name) for name in protected])
            row = {"attempt": attempt, "status": "RUNNING", "backend": agent.backend_name,
                   "model": agent.backend_model, "requested_backend": agent.requested_backend}
            report["attempts"].append(row)
            checkpoint()
            sink = {}
            try:
                rationale = await asyncio.wait_for(agent(str(root / plan["anchor"]), history,
                            session_sink=sink), timeout=min(timeout, phase_deadline - time.time()))
            finally:
                for key in ("session_id", "end_reason", "turns", "gate_passed", "integrity_violation",
                            "integrity_reason", "workspace_contention", "findings"):
                    if key in sink:
                        row[key] = sink[key]
                checkpoint()
            if sink.get("workspace_contention") or sink.get("integrity_violation"):
                raise RuntimeError("Forge initialization session failed workspace integrity: " +
                                   str(sink.get("integrity_reason") or "workspace contention"))
            if _git(root, "rev-parse", "HEAD") != starting_head or protected_inventory(spec, root) != protected:
                raise RuntimeError("Forge initialization changed protected files or Git history")
            try:
                before = _bundle_hashes(spec, root)
                checked = bridge.execute(plan, root, role="candidate", action="correctness")
                if before != _bundle_hashes(spec, root):
                    raise RuntimeError("Candidate changed during initialization validation")
            except Exception as exc:
                row.update(status="REJECTED", error=f"{type(exc).__name__}: {exc}")
                details = "\n".join(command.stdout + command.stderr for command in getattr(exc, "commands", ()))
                history = "\n".join([history, f"Attempt {attempt}: {rationale}", row["error"], details[-6000:]])
                checkpoint()
                continue
            paths = [scope.path for scope in spec.candidate.editable]
            _git(root, "add", "-A", "--", *paths)
            _git(root, "commit", "--quiet", "--allow-empty", "--only", "-m",
                 "Arena Forge: initial correct target implementation", "--", *paths)
            commit = _git(root, "rev-parse", "HEAD")
            row.update(status="PASS", candidate_sha256=before, correctness=checked.to_mapping())
            report.update(status="PASS", commit=commit)
            checkpoint()
            return report
        raise RuntimeError("Forge initialization did not produce a candidate passing the complete task checks")
    except BaseException as exc:
        report.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        if report["attempts"] and report["attempts"][-1]["status"] == "RUNNING":
            report["attempts"][-1].update(status="FAILED", error=report["error"])
        checkpoint()
        raise
    finally:
        plan.pop("phase_deadline_unix", None)
        _save(os.environ["ARENA_FORGE_PLAN"], plan)


def prepare_loop(plan: dict, argv: list[str]) -> list[str]:
    """Refresh the full source bundle after initialization, then run one loop."""
    from kernelforge import cli
    from agents.forge.adapter import engine_budget_hours
    from agents.forge.program import program_text
    if not argv or argv[0] != "forge-loop":
        raise ValueError("Initialization must be followed by forge-loop")
    root = Path(plan["engine_root"])
    context = TaskContext.load(plan["context"])
    files = candidate_files(context.spec, root)
    argv = list(argv)
    argv[argv.index("--source-files") + 1] = ",".join(map(str, files.values()))
    # Initialization has already spent part of this campaign's budget.
    argv[argv.index("--max-hours") + 1] = str(engine_budget_hours(plan))
    if any(parameter.name == "deadline_unix" for parameter in cli.main.commands["forge-loop"].params):
        if "--deadline-unix" in argv:
            argv[argv.index("--deadline-unix") + 1] = str(plan["deadline_unix"])
        else:
            argv += ["--deadline-unix", str(plan["deadline_unix"])]
    # Keep the program in the already tracked root. Only the adapter writes it.
    Path(plan["program"]).write_text(program_text(plan))
    _git(root, "add", "--", Path(plan["program"]).relative_to(root).as_posix())
    _git(root, "commit", "--quiet", "--allow-empty", "-m", "Arena Forge: enter optimization phase")
    return argv
