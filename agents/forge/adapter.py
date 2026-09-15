"""Unified Forge launcher: task facts select an internal engine workflow."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

import yaml

from agents.forge import bridge
from agents.forge.common import _read_forge_result, _resolve_gpu_arch, _resolve_gpu_type, run_forge_subprocess
from agents.forge.task_context import TaskContext
from agents.forge.bundles import allow_candidate_paths, candidate_files, committed_candidate, copy_workspace, install_candidate
from src.task_spec import resolve_task_path


class ForgeRunError(RuntimeError):
    """A campaign failed; its diagnostic directory and candidate are retained."""


def _write(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _digest(spec, root: Path):
    return {name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in candidate_files(spec, root, required=False).items()}


def choose_workflow(context: TaskContext, *, target_verified: bool, requested: str = "auto") -> str:
    if requested not in ("auto", "optimize", "rewrite"):
        raise ValueError("Forge workflow must be auto, optimize, or rewrite")
    requires_initialization = not target_verified
    if requested == "optimize" and requires_initialization:
        raise ValueError("Forge optimize requires a verified current implementation in the target language")
    if requested == "rewrite" and target_verified:
        raise ValueError("Candidate already satisfies the target contract; use auto or optimize")
    if requires_initialization and context.spec.candidate.language not in ("flydsl", "hip", "triton"):
        raise ValueError(f"Forge initialization to {context.spec.candidate.language} is unsupported; "
                         "reviewed initialization backends are FlyDSL, HIP and Triton")
    if not requires_initialization:
        return "optimize"
    return "rewrite" if context.spec.candidate.language == "flydsl" else "initialize"


def _config(eval_config: dict) -> dict:
    config = yaml.safe_load(Path(__file__).with_name("agent_config.yaml").read_text())
    overrides = eval_config.get("agent") or {}
    if not isinstance(overrides, dict):
        raise ValueError("agent run configuration must be a mapping")
    allowed = {"workflow", "model", "timeout_seconds", "permission_mode", "agent_backend",
               "python", "max_port_attempts", "supervisor_backend", "session_timeout_seconds",
               "initialization_max_attempts", "initialization_budget_fraction", "codex_auth_mode"}
    config.update({key: value for key, value in overrides.items() if key in allowed})
    if config["codex_auth_mode"] not in ("gateway", "cli"):
        raise ValueError("Forge codex_auth_mode must be gateway or cli")
    if overrides.get("agent_backend") not in (None, "claude") and "model" not in overrides:
        # A Claude default is not a model ID for another provider. Let upstream
        # resolve that provider's configured model unless explicitly overridden.
        config["model"] = None
    timeout = config["timeout_seconds"]
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Forge timeout_seconds must be finite and positive")
    for field in ("max_port_attempts", "session_timeout_seconds", "initialization_max_attempts"):
        if type(config[field]) is not int or config[field] < 1:
            raise ValueError(f"Forge {field} must be a positive integer")
    fraction = config["initialization_budget_fraction"]
    if type(fraction) not in (int, float) or not 0 < fraction < 1:
        raise ValueError("Forge initialization_budget_fraction must be between zero and one")
    return config


def _initialize_git(root: Path) -> None:
    with (root / ".gitignore").open("a") as stream:
        stream.write("\n__pycache__/\n*.pyc\nbuild/\n.forge_rewrite/\nforge_experiments/\n")
    for command in (["init", "--quiet"], ["config", "user.email", "arena-forge@local"],
                    ["config", "user.name", "Arena Forge"], ["add", "-A"],
                    ["commit", "--quiet", "--allow-empty", "-m", "Arena task snapshot"]):
        subprocess.run(["git", *command], cwd=root, check=True, capture_output=True, text=True)


def _anchor(context: TaskContext) -> str:
    if context.spec.candidate.entrypoints:
        return context.spec.candidate.entrypoints[0].file
    for scope in context.spec.candidate.editable:
        if scope.scope != "tree":
            return scope.path
    files = candidate_files(context.spec, context.workspace, required=False)
    if not files:
        raise ValueError("Forge requires an entrypoint file when the candidate is an empty tree")
    return next(iter(files))


def require_supported_backend(spec, capabilities: dict) -> str:
    """Preserve the task language; an unrelated backend is not a substitute."""
    language = spec.candidate.language
    if language not in capabilities["backends"]:
        raise ForgeRunError(f"KernelForge has no {language} backend")
    return language


def build_command(plan: dict, context: TaskContext, config: dict, *, gpu_arch: str, gpu_type: str) -> list[str]:
    root = Path(plan["engine_root"])
    command = [config.get("python") or sys.executable, str(Path(__file__).with_name("upstream.py")),
               "forge-rewrite-by-flydsl" if plan["workflow"] == "rewrite" else "forge-loop",
               "--workspace", str(root), "--driver", str(root / "arena_forge_driver.py"),
               "--experiments-dir", str(root / "forge_experiments"), "--result-json", plan["result"],
               "--max-hours", str(max(1.0, (plan["deadline_unix"] - time.time()) / 3600)),
               "--deadline-unix", str(plan["deadline_unix"]),
               "--gpu-target", gpu_arch, "--gpu-type", gpu_type,
               "--permission-mode", config["permission_mode"]]
    if config.get("model"):
        command.extend(["--model", config["model"]])
    symbols = [entry.symbol for entry in context.spec.candidate.entrypoints if entry.symbol]
    identity = context.spec.to_mapping().get("kernel_identity", {})
    operator = identity.get("logical_operator") or context.spec.task_id
    if plan["workflow"] != "rewrite":
        files = candidate_files(context.spec, root)
        command += ["--kernel", str(root / plan["anchor"]),
                    "--kernel-backend", context.spec.candidate.language,
                    "--source-files", ",".join(map(str, files.values())),
                    "--target-functions", ",".join(symbols), "--operator-name", operator,
                    "--framework", identity.get("source_owner", "standalone"),
                    "--task-type", "image_kernel",  # upstream multi-file switch only
                    "--baseline-json", plan["baseline"], "--program-md-file", plan["program"],
                    "--no-prepare-task", "--no-profiling", "--no-specialist-probe",
                    "--no-experience-kb", "--lanes", "1", "--agent-backend", config["agent_backend"],
                    "--session-timeout-sec", str(config["session_timeout_seconds"])]
    else:
        command += ["--source-kernel", plan["source"], "--flydsl-kernel-name", plan["anchor"],
                    "--logical-op-name", operator, "--no-prepare-driver", "--no-rewrite-kb",
                    "--max-port-attempts", str(config["max_port_attempts"]),
                    "--supervisor-backend", config["supervisor_backend"] or config["agent_backend"]]
        if identity.get("source_owner") in ("aiter", "vllm", "sglang"):
            command += ["--framework", identity["source_owner"]]
        if context.spec.candidate.initial_language in ("triton", "hip", "cuda", "cpp"):
            command += ["--source-language", context.spec.candidate.initial_language]
    if plan["workflow"] == "initialize":
        command.insert(2, "--arena-initialize")
    return command


def launch(eval_config: dict, task_config_dir: str, workspace: str) -> str:
    """Consume only the framework context; config_path is the standard ABI."""
    del task_config_dir
    started = time.time()
    config = _config(eval_config)
    context = TaskContext.load(workspace=Path(workspace))
    deadline = started + config["timeout_seconds"]
    logger = logging.getLogger(__name__)
    # A fresh sibling keeps provider scratch and reports outside the editable
    # task. Never delete or reuse artifacts from a previous invocation.
    artifact_root = Path(tempfile.mkdtemp(prefix=context.workspace.name + "-forge-", dir=context.workspace.parent))
    status_path = artifact_root / "arena_forge_status.json"
    status = {"status": "RUNNING", "task_id": context.spec.task_id, "artifacts": [],
              "arena_verdict": "pending", "artifact_directory": str(artifact_root)}
    _write(status_path, status)
    logger.info("Forge artifacts: %s", artifact_root)
    output = ""
    try:
        engine = artifact_root / "engine"
        template = artifact_root / "template"
        copy_workspace(context.workspace, template)
        copy_workspace(context.workspace, engine)
        plan = {"version": 1, "context": str(context.path), "workflow": "optimize",
                "engine_root": str(engine), "template": str(template), "deadline_unix": deadline,
                "anchor": _anchor(context), "result": str(artifact_root / "engine_result.json"),
                "baseline": str(artifact_root / "baseline.json"), "program": str(engine / "arena_program.md"),
                "initialization_result": str(artifact_root / "initialization.json"),
                "agent_config": config, "gpu_arch": _resolve_gpu_arch(eval_config),
                "gpu_type": _resolve_gpu_type(eval_config)}
        plan_path = artifact_root / "bridge_plan.json"
        _write(plan_path, plan)
        env = os.environ.copy()
        env.update({"ARENA_FORGE_PLAN": str(plan_path), "PYTHONUNBUFFERED": "1", "IS_SANDBOX": "1",
                    "FORGE_AGENT_BACKEND": config["agent_backend"],
                    "FORGE_AGENT_TIMEOUT_SEC": str(config["session_timeout_seconds"])})
        arena_root = Path(__file__).resolve().parents[2]
        env["PYTHONPATH"] = os.pathsep.join([str(arena_root), *filter(None, [env.get("PYTHONPATH", "")])])
        # Probe in the exact interpreter used by both outer and nested CLIs.
        probe = subprocess.run([config.get("python") or sys.executable,
                                str(Path(__file__).with_name("upstream.py")), "--arena-probe"],
                               env=env, capture_output=True, text=True,
                               timeout=max(1, min(60, deadline - time.time())))
        if probe.returncode:
            raise ForgeRunError("KernelForge runtime preflight failed: " + probe.stderr[-3000:])
        capabilities = json.loads(probe.stdout)
        status["engine"] = capabilities
        require_supported_backend(context.spec, capabilities)
        candidate = context.spec.candidate
        initially_target = candidate.initial_state == "implemented" and candidate.initial_language == candidate.language
        changed = _digest(context.spec, context.workspace) != _digest(context.spec, context.baseline_workspace)
        target_verified = False
        if initially_target or changed:
            # A broken current implementation or environment is an error, not
            # permission to reinterpret a task as an unimplemented candidate.
            bridge.execute(plan, engine, role="candidate", action="correctness")
            target_verified = True
        plan["workflow"] = choose_workflow(context, target_verified=target_verified,
                                           requested=config["workflow"])
        status["workflow"] = plan["workflow"]
        if plan["workflow"] == "initialize":
            from agents.forge.initialization import materialize_targets
            materialize_targets(context.spec, engine, plan["anchor"])
        if plan["workflow"] == "rewrite":
            if Path(plan["anchor"]).suffix != ".py":
                raise ForgeRunError("FlyDSL rewrite requires a Python candidate entry file")
            sources = context.spec.baseline.source_files
            source = resolve_task_path(template, sources[0], must_exist=True) if sources else template / plan["anchor"]
            if not source.is_file():
                # A spec-only task is valid. The source hint is the task contract;
                # the independently provided baseline still executes normally.
                source = template / "README.md"
            hint = engine / ("arena_source_hint" + (source.suffix if source.is_file() else ".txt"))
            hint.write_text(source.read_text() if source.is_file() else context.spec.to_mapping().get("description", ""))
            plan["source"] = str(hint)
            # Remove candidate shadows at the engine root. The complete candidate
            # bundle will live under the producer's current attempt directory.
            for path in candidate_files(context.spec, engine, required=False).values():
                path.unlink()
        _write(plan_path, plan)
        from agents.forge.upstream import program_text
        Path(plan["program"]).write_text(program_text(plan, initialize=plan["workflow"] == "initialize"))
        (engine / "arena_forge_driver.py").write_text(bridge.render_driver(plan_path, arena_root))
        if plan["workflow"] != "rewrite":
            allow_candidate_paths(engine, context.spec)
        _initialize_git(engine)
        if plan["workflow"] != "rewrite":
            baseline = bridge.execute(plan, engine, role="baseline", action="performance")
            values = bridge.timings(baseline)
            _write(Path(plan["baseline"]), {"wall_ms": statistics.fmean(values.values()), "case_times": values})
        command = build_command(plan, context, config, gpu_arch=plan["gpu_arch"], gpu_type=plan["gpu_type"])
        status["command"] = command
        _write(status_path, status)
        remaining = deadline - time.time()
        if remaining <= 0:
            raise TimeoutError("Forge budget exhausted before engine launch")
        process, stdout, stderr, timed_out = run_forge_subprocess(command, workspace=str(engine), env=env,
                                            timeout_seconds=remaining, logger=logger)
        output = "\n".join([*stdout, *stderr])
        (artifact_root / "engine.log").write_text(output)
        result = _read_forge_result(Path(plan["result"]), "\n".join(stdout))
        status.update({"exit_code": process.returncode, "timed_out": timed_out, "engine_result": result})
        if Path(plan["initialization_result"]).is_file():
            status["initialization"] = json.loads(Path(plan["initialization_result"]).read_text())
        if timed_out:
            raise TimeoutError("Forge campaign exceeded its shared deadline; partial artifacts retained")
        if process.returncode != 0 or not isinstance(result, dict):
            raise ForgeRunError("Forge engine failed or omitted its structured result")
        prefix = ""
        selected_commit = result.get("best_commit")
        if plan["workflow"] == "rewrite":
            if result.get("port_ok") is not True:
                raise ForgeRunError("Forge PORT did not deliver an implementation")
            attempts = result.get("temporary_paths")
            if not isinstance(attempts, list) or len(attempts) != 1:
                raise ForgeRunError("Forge result must identify exactly one current attempt")
            if not str(attempts[0]).startswith(".forge_rewrite/"):
                raise ForgeRunError("Forge result candidate is outside the attempt directory")
            resolve_task_path(engine, attempts[0], must_exist=True)
            prefix = attempts[0]
            selected_commit = result.get("flydsl_best_commit") or result.get("best_commit")
        elif plan["workflow"] == "initialize" and not selected_commit:
            # Native loop results can omit best_commit when no iteration earns
            # KEEP. Preserve the independently validated first implementation;
            # do not fabricate a loop win or rewrite its reported measurements.
            initial = status.get("initialization", {})
            if initial.get("status") == "PASS":
                selected_commit = initial.get("commit")
                status["delivery_selection"] = "initial_correct_implementation"
        source = committed_candidate(context.spec, engine, selected_commit,
                                     artifact_root / "delivery", prefix=prefix)
        status["selected_commit"] = selected_commit
        # No newest-file search, no git checkout that discards an uncommitted
        # candidate, no task-family-specific exporter. Arena evaluates delivery.
        status["artifacts"] = install_candidate(context.spec, source, context.workspace)
        status["status"] = "DELIVERED"
        _write(status_path, status)
        return output + "\nARENA_FORGE_STATUS=" + json.dumps(status)
    except BaseException as exc:
        status.update({"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"})
        _write(status_path, status)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise ForgeRunError(f"{exc}; Forge diagnostics: {status_path}") from exc
