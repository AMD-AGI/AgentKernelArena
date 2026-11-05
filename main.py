# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import argparse
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from src.tasks import get_task_config
from src.preprocessing import (
    get_task_workspace_path,
    is_task_complete,
    setup_rocm_env,
)
from src.module_registration import AgentType, load_agent_launcher, load_post_processing_handler
from src.runtime_env import apply_subprocess_python_path
from src.task_spec import required_gpu_arches
from src.tools.agent_cost import AgentCostCollector
from src.tools.agent_time_budget import validate_run_budget_mode


QUEUE_DIR_NAME = ".parallel"
QUEUE_STATES = ("pending", "running", "done", "failed")


parser = argparse.ArgumentParser(description="arguments for AgentKernelArena")
parser.add_argument(
    "--config_name",
    type=str,
    default="example_configs/quickstart_claude_mi300.yaml",
    help=(
        "run configuration for AgentKernelArena (default: "
        "example_configs/quickstart_claude_mi300.yaml for the generic MI300-series target). "
        "Select a model-specific config for exact hardware tuning context."
    ),
)
parser.add_argument(
    "--run-suffix",
    type=str,
    default=None,
    help="Suffix appended to the run directory name, e.g. --run-suffix composer2_hip -> run_20260416_120000_composer2_hip",
)
parser.add_argument(
    "--resume-run",
    type=str,
    default=None,
    help="Resume an existing run by specifying the run directory name (e.g., run_20250115_143022)",
)
parser.add_argument(
    "--resume-latest",
    action="store_true",
    help="Resume the most recent run in the workspace",
)
parser.add_argument(
    "--run-name",
    type=str,
    default=None,
    help="Internal: explicit run directory name for parallel workers/post-processing",
)
parser.add_argument(
    "--parallel-init",
    action="store_true",
    help="Internal: initialize a shared parallel task queue for --run-name",
)
parser.add_argument(
    "--parallel-worker",
    action="store_true",
    help="Internal: run tasks claimed from the shared parallel queue",
)
parser.add_argument(
    "--worker-id",
    type=str,
    default=None,
    help="Internal: worker identifier used by --parallel-worker",
)
parser.add_argument(
    "--postprocess-only",
    action="store_true",
    help="Internal: run only final post-processing for --run-name",
)


def _extract_timestamp(run_directory_name: str) -> str | None:
    m = re.match(r"^run_(\d{8}_\d{6})", run_directory_name)
    return m.group(1) if m else None


def _run_suffix_from_name(run_directory_name: str) -> str:
    m = re.match(r"^run_\d{8}_\d{6}(_[A-Za-z0-9._-]+)?$", run_directory_name)
    return m.group(1) if m and m.group(1) else ""


def _validate_run_suffix(run_suffix: str | None) -> bool:
    return run_suffix is None or bool(re.fullmatch(r"[A-Za-z0-9._-]+", run_suffix))


def _load_config(config_name: str) -> dict[str, Any]:
    with open(config_name, "r") as f:
        return yaml.safe_load(f) or {}


def _resolve_agent(agent_string: str) -> AgentType | None:
    try:
        return AgentType.from_string(agent_string)
    except ValueError as e:
        print(f"Error: {e}")
        return None


def _resolve_run(
    args: argparse.Namespace,
    workspace_directory: Path,
) -> tuple[Path, str, str, bool] | None:
    """Return (run_directory, run_directory_name, timestamp, resume_mode)."""
    if args.run_name:
        run_directory_name = args.run_name
        timestamp = _extract_timestamp(run_directory_name)
        if not timestamp:
            print(
                f"Error: Invalid run directory name format: {run_directory_name}. "
                "Expected format: run_YYYYMMDD_HHMMSS[_suffix]"
            )
            return None
        run_directory = workspace_directory / run_directory_name
        resume_mode = run_directory.exists()
        run_directory.mkdir(parents=True, exist_ok=True)
        return run_directory, run_directory_name, timestamp, resume_mode

    if args.resume_run:
        run_directory_name = args.resume_run
        run_directory = workspace_directory / run_directory_name
        if not run_directory.exists():
            print(f"Error: Run directory does not exist: {run_directory}")
            return None
        timestamp = _extract_timestamp(run_directory_name)
        if not timestamp:
            print(
                f"Error: Invalid run directory name format: {run_directory_name}. "
                "Expected format: run_YYYYMMDD_HHMMSS[_suffix]"
            )
            return None
        return run_directory, run_directory_name, timestamp, True

    if args.resume_latest:
        run_dirs = sorted(
            [
                d
                for d in workspace_directory.iterdir()
                if d.is_dir() and d.name.startswith("run_") and not d.name.endswith("_heldout")
            ],
            key=lambda x: x.name,
            reverse=True,
        )
        if not run_dirs:
            print(f"Error: No run directories found in {workspace_directory}")
            return None
        run_directory = run_dirs[0]
        run_directory_name = run_directory.name
        timestamp = _extract_timestamp(run_directory_name) or datetime.now().strftime("%Y%m%d_%H%M%S")
        return run_directory, run_directory_name, timestamp, True

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.run_suffix}" if args.run_suffix else ""
    run_directory_name = f"run_{timestamp}{suffix}"
    run_directory = workspace_directory / run_directory_name
    run_directory.mkdir(parents=True, exist_ok=True)
    return run_directory, run_directory_name, timestamp, False


def _configure_logging(
    config: dict[str, Any],
    agent: AgentType,
    timestamp: str,
    run_directory_name: str,
    args: argparse.Namespace,
    role: str | None = None,
) -> logging.Logger:
    log_dir = Path(config["log_directory"])
    log_dir.mkdir(parents=True, exist_ok=True)

    log_suffix = f"_{args.run_suffix}" if args.run_suffix else _run_suffix_from_name(run_directory_name)
    role_suffix = f"_{role}" if role else ""
    log_filename = f"{config['target_gpu_model']}_{agent.value}_{timestamp}{log_suffix}{role_suffix}.log"
    log_path = log_dir / log_filename

    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("AgentKernelArena Framework Started")
    logger.info("=" * 80)
    logger.info(f"Log file: {log_path}")
    return logger


def _discover_tasks(tasks: list, overrides: dict[str, str] | None = None) -> dict[str, str]:
    """Resolve selectors; ``{path, target_language}`` entries record per-task languages.

    A path prefix may select several tasks; each selected task runs once, so a
    task cannot be selected again with a different target language.
    """
    if not tasks:
        raise ValueError("No task selectors were configured")
    from src.task_contract import TARGET_LANGUAGES

    task_config_dict: dict[str, str] = {}
    for selector in tasks:
        language = None
        if isinstance(selector, dict):
            if set(selector) - {"path", "target_language"} or "path" not in selector:
                raise ValueError("A task entry mapping takes path and optional target_language")
            language = selector.get("target_language")
            selector = selector["path"]
            if language is not None and language not in TARGET_LANGUAGES:
                raise ValueError(f"target_language must be one of {TARGET_LANGUAGES}")
        if not isinstance(selector, str) or not selector:
            raise ValueError("Task selectors are nonempty path prefixes or 'all'")
        discovered = get_task_config() if selector == "all" else get_task_config(category=selector)
        if not discovered:
            raise ValueError(f"Configured task selector {selector!r} matched no task configs")
        for task_name, path in discovered.items():
            if overrides is not None and language is not None:
                if overrides.get(task_name, language) != language:
                    raise ValueError(f"Task {task_name} is selected with two target languages")
                overrides[task_name] = language
            task_config_dict[task_name] = path
    return task_config_dict


def should_run_task_for_platform(
    task_name: str,
    task_config: dict[str, Any],
    current_gfx_arch: str | None,
    logger: logging.Logger,
) -> bool:
    """Return whether a task's optional platform metadata includes this run."""
    platform_support = task_config.get("platform_support")
    if platform_support is None:
        return True
    if not isinstance(platform_support, dict):
        logger.warning(
            "Task %s has non-dict platform_support=%r; treating task as runnable",
            task_name,
            platform_support,
        )
        return True

    raw_status = platform_support.get("status", "active")
    status = str(raw_status).strip().lower() if raw_status is not None else "active"
    if status == "skip":
        skip_reason = str(platform_support.get("skip_reason") or "").strip()
        suffix = f": {skip_reason}" if skip_reason else ""
        logger.warning(
            "Skipping task %s before workspace setup: platform_support.status=skip%s",
            task_name,
            suffix,
        )
        return False
    if status and status != "active":
        logger.warning(
            "Task %s has unsupported platform_support.status=%r; treating task as runnable",
            task_name,
            raw_status,
        )

    required_arch = required_gpu_arches(platform_support)
    if not required_arch:
        return True
    if not current_gfx_arch:
        logger.warning(
            "Skipping task %s before workspace setup: platform_support.required_arch=%s, "
            "but current GPU arch could not be resolved",
            task_name,
            required_arch,
        )
        return False
    if current_gfx_arch not in required_arch:
        logger.warning(
            "Skipping task %s before workspace setup: platform_support.required_arch=%s "
            "does not match current GPU arch %s",
            task_name,
            required_arch,
            current_gfx_arch,
        )
        return False

    return True


def filter_tasks_by_platform(
    task_config_dict: dict[str, str],
    current_gfx_arch: str | None,
    logger: logging.Logger,
) -> dict[str, str]:
    """Filter task configs using their optional platform_support metadata."""
    runnable_tasks: dict[str, str] = {}
    skipped_tasks: list[str] = []

    for task_name, task_config_dir in task_config_dict.items():
        with open(task_config_dir, "r") as f:
            task_config = yaml.safe_load(f) or {}
        if should_run_task_for_platform(task_name, task_config, current_gfx_arch, logger):
            runnable_tasks[task_name] = task_config_dir
        else:
            skipped_tasks.append(task_name)

    if skipped_tasks:
        logger.warning(
            "Platform support preflight skipped %d task(s): %s",
            len(skipped_tasks),
            skipped_tasks,
        )
    return runnable_tasks


def _build_context(
    args: argparse.Namespace,
    *,
    need_agent_launcher: bool,
    role: str | None = None,
) -> dict[str, Any] | None:
    if not _validate_run_suffix(args.run_suffix):
        print("Error: --run-suffix may only contain letters, numbers, dot, underscore, and dash")
        return None

    config = _load_config(args.config_name)
    tasks = config["tasks"]
    agent = _resolve_agent(config["agent"]["template"])
    if agent is None:
        return None
    try:
        validate_run_budget_mode(agent.value, config["agent"])
    except ValueError as error:
        print(f"Error: {error}")
        return None

    project_root = Path(__file__).resolve().parent
    workspace_directory_name = (
        f"{config['workspace_directory_prefix']}_{config['target_gpu_model']}_{agent.value}"
    )
    workspace_directory = (project_root / workspace_directory_name).resolve()
    resolved_run = _resolve_run(args, workspace_directory)
    if resolved_run is None:
        return None
    run_directory, run_directory_name, timestamp, resume_mode = resolved_run

    logger = _configure_logging(config, agent, timestamp, run_directory_name, args, role=role)
    logger.info(f"Agent: {agent.value}")
    logger.info(f"Target Architecture: {config['target_gpu_model']}")
    logger.info(f"Workspace Directory: {workspace_directory}")
    logger.info(f"Run Directory: {run_directory}")
    logger.info(f"{'RESUME' if resume_mode else 'NEW'} RUN: {run_directory_name}")
    if args.worker_id is not None:
        logger.info(f"Parallel Worker ID: {args.worker_id}")
    for env_name in (
        "AGENT_KERNEL_ARENA_HOST_GPU_ID",
        "ROCR_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
    ):
        if os.environ.get(env_name):
            logger.info(f"{env_name}={os.environ[env_name]}")

    python_path = apply_subprocess_python_path()
    logger.info(f"Subprocess Python environment: {python_path}")
    setup_rocm_env(config["target_gpu_model"], logger)
    current_gfx_arch = os.environ.get("PYTORCH_ROCM_ARCH")

    agent_launcher = None
    if need_agent_launcher:
        try:
            agent_launcher = load_agent_launcher(agent, logger)
        except Exception as e:
            logger.error(f"Failed to load agent launcher: {e}")
            return None

    try:
        from src.task_contract import TARGET_LANGUAGES

        if config.get("target_language") is not None and config["target_language"] not in TARGET_LANGUAGES:
            raise ValueError(f"target_language must be one of {TARGET_LANGUAGES}")
        overrides: dict[str, str] = {}
        configured_tasks = _discover_tasks(tasks, overrides)
        config["_task_target_languages"] = overrides
        if agent.value != "qa_auditor" and config.get("target_language") is None and any(
                Path(path).name == "task.yaml" and name not in overrides
                for name, path in configured_tasks.items()):
            raise ValueError("Schema-v3 tasks need target_language in the run config "
                             f"(one of {TARGET_LANGUAGES}) or per task")
    except ValueError as error:
        logger.error("Task discovery failed: %s", error)
        return None
    logger.info(f"Found {len(configured_tasks)} configured task(s)")
    task_config_dict = filter_tasks_by_platform(configured_tasks, current_gfx_arch, logger)
    logger.info(f"Found {len(task_config_dict)} runnable task(s) after platform preflight")
    logger.info(f"Tasks: {list(task_config_dict.keys())}")

    return {
        "args": args,
        "config": config,
        "agent": agent,
        "agent_launcher": agent_launcher,
        "workspace_directory": workspace_directory,
        "run_directory": run_directory,
        "run_directory_name": run_directory_name,
        "timestamp": timestamp,
        "resume_mode": resume_mode,
        "logger": logger,
        "task_config_dict": task_config_dict,
    }


def _filter_completed_tasks(
    task_config_dict: dict[str, str],
    run_directory: Path,
    timestamp: str,
    agent: AgentType,
    logger: logging.Logger,
) -> dict[str, str]:
    tasks_to_run: dict[str, str] = {}
    skipped_tasks = []

    for task_name, task_config_dir in task_config_dict.items():
        if is_task_complete(run_directory, task_name, timestamp, agent.value):
            skipped_tasks.append(task_name)
            logger.info(f"Skipping completed task: {task_name}")
        else:
            tasks_to_run[task_name] = task_config_dir

    logger.info(
        f"Resume mode: {len(skipped_tasks)} task(s) already completed, "
        f"{len(tasks_to_run)} task(s) remaining"
    )
    if skipped_tasks:
        logger.info(f"Skipped tasks: {skipped_tasks}")
    return tasks_to_run


def run_task(
    *,
    eval_config: dict[str, Any],
    agent: AgentType,
    agent_launcher: Any,
    task_name: str,
    task_config_dir: str,
    run_directory: Path,
    timestamp: str,
    logger: logging.Logger,
    task_index: int,
    total_tasks: int,
) -> tuple[bool, Path | None]:
    """Run one task and record the agent's cost for it.

    The launcher is wrapped so that whatever usage the provider reports is
    collected; after the task, one record per launched workspace is written to
    ``<run_dir>/agent_costs/<workspace>.json`` (null cost when nothing was
    reported). Writing the record never changes the task outcome.
    """
    agent_settings = eval_config.get("agent") if isinstance(eval_config, dict) else None
    collector = AgentCostCollector(agent=agent.value, task_name=task_name,
                                   run_settings=agent_settings)
    try:
        return _run_task(
            eval_config=eval_config,
            agent=agent,
            agent_launcher=collector.wrap(agent_launcher) if agent_launcher is not None else None,
            task_name=task_name,
            task_config_dir=task_config_dir,
            run_directory=run_directory,
            timestamp=timestamp,
            logger=logger,
            task_index=task_index,
            total_tasks=total_tasks,
        )
    finally:
        collector.write(run_directory, logger)


def _run_task(
    *,
    eval_config: dict[str, Any],
    agent: AgentType,
    agent_launcher: Any,
    task_name: str,
    task_config_dir: str,
    run_directory: Path,
    timestamp: str,
    logger: logging.Logger,
    task_index: int,
    total_tasks: int,
) -> tuple[bool, Path | None]:
    """Run one schema-v3 kernel task or schema-v2 serving task through the shared lifecycle."""
    workspace_path: Path | None = None
    logger.info("=" * 80)
    logger.info(f"Task {task_index}/{total_tasks}: {task_name}")
    logger.info("=" * 80)

    try:
        with open(task_config_dir, "r") as handle:
            declaration = yaml.safe_load(handle) or {}
        expected = 3 if Path(task_config_dir).name == "task.yaml" else 2
        if type(declaration.get("schema_version")) is not int or declaration["schema_version"] != expected:
            raise ValueError(f"Unsupported task schema version: {declaration.get('schema_version')!r}")
        from src.task_run import run_task_v2

        return run_task_v2(
            eval_config=eval_config, agent=agent, agent_launcher=agent_launcher,
            task_name=task_name, task_config_dir=task_config_dir,
            run_directory=run_directory, timestamp=timestamp, logger=logger,
        )
    except Exception as e:
        logger.error(f"Task {task_name} failed with error: {e}", exc_info=True)
        if agent == AgentType.QA_AUDITOR:
            # Setup can fail after creating the workspace. Persist that failure
            # as a complete audit report so a resumed run does not retry it
            # forever and post-processing can report it as a blocking finding.
            report_workspace = workspace_path or get_task_workspace_path(run_directory, task_name, timestamp)
            if report_workspace.is_dir():
                try:
                    from agents.qa_auditor.report import build_report, write_report

                    write_report(report_workspace, build_report(
                        task_id=task_name, findings=[], task_package_sha256=None, target_language=None,
                        request_id=None, framework_errors=[
                            f"task setup/execution failed before the audit: {type(e).__name__}: {e}"]))
                    return True, report_workspace
                except Exception:
                    logger.error("Could not write the audit failure report for %s", task_name, exc_info=True)
        return False, workspace_path


def run_post_processing(agent: AgentType, workspace_paths: list[str], logger: logging.Logger) -> bool:
    logger.info("=" * 80)
    logger.info("Running Post-Processing")
    logger.info("=" * 80)

    try:
        post_processing_handler = load_post_processing_handler(agent, logger)
        result = post_processing_handler(workspace_paths, logger)
        return result is not False
    except NotImplementedError as e:
        logger.warning(f"Post-processing skipped: {e}")
        return True
    except Exception as e:
        logger.error(f"Post-processing failed: {e}", exc_info=True)
        return False


def _queue_root(run_directory: Path) -> Path:
    return run_directory / QUEUE_DIR_NAME


def _queue_state_dir(run_directory: Path, state: str) -> Path:
    return _queue_root(run_directory) / state


def _descriptor_name(index: int, task_name: str) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", task_name).strip("_")
    return f"{index:06d}_{safe_name or 'task'}.yaml"


def _write_descriptor(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with tmp_path.open("w") as f:
        yaml.safe_dump(payload, f, default_flow_style=False, sort_keys=False)
    tmp_path.replace(path)


def _read_descriptor(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def initialize_parallel_queue(context: dict[str, Any]) -> None:
    run_directory: Path = context["run_directory"]
    task_config_dict: dict[str, str] = context["task_config_dict"]
    timestamp: str = context["timestamp"]
    agent: AgentType = context["agent"]
    logger: logging.Logger = context["logger"]

    for state in QUEUE_STATES:
        _queue_state_dir(run_directory, state).mkdir(parents=True, exist_ok=True)

    for state in QUEUE_STATES:
        for descriptor in _queue_state_dir(run_directory, state).glob("*.yaml"):
            descriptor.unlink()

    total_tasks = len(task_config_dict)
    queued = 0
    completed = 0
    for index, (task_name, task_config_dir) in enumerate(task_config_dict.items(), 1):
        workspace_path = get_task_workspace_path(run_directory, task_name, timestamp)
        payload = {
            "index": index,
            "total_tasks": total_tasks,
            "task_name": task_name,
            "task_config_dir": task_config_dir,
            "workspace_path": str(workspace_path),
        }
        if is_task_complete(run_directory, task_name, timestamp, agent.value):
            payload["status"] = "already_complete"
            state = "done"
            completed += 1
        else:
            payload["status"] = "pending"
            state = "pending"
            queued += 1
        _write_descriptor(_queue_state_dir(run_directory, state) / _descriptor_name(index, task_name), payload)

    logger.info(
        f"Parallel queue initialized: queued={queued}, already_complete={completed}, "
        f"total={total_tasks}"
    )


def claim_next_descriptor(run_directory: Path, worker_id: str, logger: logging.Logger) -> Path | None:
    pending_dir = _queue_state_dir(run_directory, "pending")
    running_dir = _queue_state_dir(run_directory, "running")
    running_dir.mkdir(parents=True, exist_ok=True)

    for descriptor in sorted(pending_dir.glob("*.yaml")):
        claimed = running_dir / f"worker_{worker_id}__{descriptor.name}"
        try:
            descriptor.rename(claimed)
            logger.info(f"Claimed task descriptor: {claimed.name}")
            return claimed
        except FileNotFoundError:
            continue
    return None


def finish_descriptor(
    descriptor: Path,
    state: str,
    *,
    workspace_path: Path | None,
    worker_id: str,
) -> None:
    payload = _read_descriptor(descriptor)
    payload["status"] = state
    payload["worker_id"] = worker_id
    if workspace_path is not None:
        payload["workspace_path"] = str(workspace_path)
    _write_descriptor(descriptor, payload)
    final_dir = descriptor.parent.parent / state
    final_dir.mkdir(parents=True, exist_ok=True)
    descriptor.rename(final_dir / descriptor.name)


def collect_existing_workspace_paths(
    run_directory: Path,
    task_config_dict: dict[str, str],
    timestamp: str,
) -> list[str]:
    workspace_paths = []
    for task_name in task_config_dict:
        workspace_path = get_task_workspace_path(run_directory, task_name, timestamp)
        if workspace_path.exists():
            workspace_paths.append(str(workspace_path))
    return workspace_paths


def run_serial(args: argparse.Namespace) -> int:
    context = _build_context(args, need_agent_launcher=True)
    if context is None:
        return 1

    all_task_config_dict = context["task_config_dict"]
    task_config_dict = all_task_config_dict
    if context["resume_mode"]:
        task_config_dict = _filter_completed_tasks(
            task_config_dict,
            context["run_directory"],
            context["timestamp"],
            context["agent"],
            context["logger"],
        )

    if not task_config_dict:
        context["logger"].info("All tasks are already completed. Nothing to run.")
        if context["agent"] == AgentType.QA_AUDITOR:
            workspace_paths = collect_existing_workspace_paths(
                context["run_directory"],
                all_task_config_dict,
                context["timestamp"],
            )
            return 0 if run_post_processing(
                context["agent"], workspace_paths, context["logger"]
            ) else 1
        return 0

    workspace_paths: list[str] = []
    execution_failed = False
    total_tasks = len(task_config_dict)
    for index, (task_name, task_config_dir) in enumerate(task_config_dict.items(), 1):
        completed, workspace_path = run_task(
            eval_config=context["config"],
            agent=context["agent"],
            agent_launcher=context["agent_launcher"],
            task_name=task_name,
            task_config_dir=task_config_dir,
            run_directory=context["run_directory"],
            timestamp=context["timestamp"],
            logger=context["logger"],
            task_index=index,
            total_tasks=total_tasks,
        )
        if workspace_path is not None:
            workspace_paths.append(str(workspace_path))
        if not completed:
            execution_failed = True

    if context["agent"] == AgentType.QA_AUDITOR and context["resume_mode"]:
        # Audit summaries are gates, so a resumed run must include reports
        # from both prior and newly completed workspaces.
        workspace_paths = collect_existing_workspace_paths(
            context["run_directory"],
            all_task_config_dict,
            context["timestamp"],
        )

    post_processing_passed = run_post_processing(
        context["agent"], workspace_paths, context["logger"]
    )
    context["logger"].info("=" * 80)
    context["logger"].info("AgentKernelArena Framework Completed")
    context["logger"].info("=" * 80)
    return 1 if execution_failed or not post_processing_passed else 0


def run_parallel_init(args: argparse.Namespace) -> int:
    context = _build_context(args, need_agent_launcher=False, role="parallel_init")
    if context is None:
        return 1
    initialize_parallel_queue(context)
    context["logger"].info(f"Parallel run name: {context['run_directory_name']}")
    context["logger"].info("Parallel queue initialization completed")
    return 0


def run_parallel_worker(args: argparse.Namespace) -> int:
    worker_id = args.worker_id or "0"
    context = _build_context(
        args,
        need_agent_launcher=True,
        role=f"worker{worker_id}",
    )
    if context is None:
        return 1

    failures = 0
    processed = 0
    while True:
        descriptor = claim_next_descriptor(context["run_directory"], worker_id, context["logger"])
        if descriptor is None:
            break

        payload = _read_descriptor(descriptor)
        success, workspace_path = run_task(
            eval_config=context["config"],
            agent=context["agent"],
            agent_launcher=context["agent_launcher"],
            task_name=payload["task_name"],
            task_config_dir=payload["task_config_dir"],
            run_directory=context["run_directory"],
            timestamp=context["timestamp"],
            logger=context["logger"],
            task_index=int(payload.get("index", processed + 1)),
            total_tasks=int(payload.get("total_tasks", len(context["task_config_dict"]))),
        )
        processed += 1
        if success:
            finish_descriptor(descriptor, "done", workspace_path=workspace_path, worker_id=worker_id)
        else:
            failures += 1
            finish_descriptor(descriptor, "failed", workspace_path=workspace_path, worker_id=worker_id)

    context["logger"].info(
        f"Parallel worker {worker_id} completed: processed={processed}, failures={failures}"
    )
    return 1 if failures else 0


def run_postprocess_only(args: argparse.Namespace) -> int:
    context = _build_context(args, need_agent_launcher=False, role="postprocess")
    if context is None:
        return 1

    workspace_paths = collect_existing_workspace_paths(
        context["run_directory"],
        context["task_config_dict"],
        context["timestamp"],
    )
    context["logger"].info(f"Post-processing {len(workspace_paths)} workspace(s)")
    post_processing_passed = run_post_processing(
        context["agent"], workspace_paths, context["logger"]
    )

    pending_descriptors = list(_queue_state_dir(context["run_directory"], "pending").glob("*.yaml"))
    running_descriptors = list(_queue_state_dir(context["run_directory"], "running").glob("*.yaml"))
    failed_descriptors = list(_queue_state_dir(context["run_directory"], "failed").glob("*.yaml"))
    if pending_descriptors or running_descriptors:
        context["logger"].error(
            "Parallel run has unfinished task descriptor(s): "
            f"pending={len(pending_descriptors)}, running={len(running_descriptors)}"
        )
        return 1
    if failed_descriptors:
        context["logger"].error(f"Parallel run has {len(failed_descriptors)} failed task(s)")
        return 1
    if not post_processing_passed:
        context["logger"].error("Post-processing gate failed")
        return 1

    context["logger"].info("=" * 80)
    context["logger"].info("AgentKernelArena Framework Completed")
    context["logger"].info("=" * 80)
    return 0


def main() -> None:
    args = parser.parse_args()
    mode_count = sum([args.parallel_init, args.parallel_worker, args.postprocess_only])
    if mode_count > 1:
        print("Error: choose only one of --parallel-init, --parallel-worker, --postprocess-only")
        raise SystemExit(1)

    if args.parallel_init:
        raise SystemExit(run_parallel_init(args))
    if args.parallel_worker:
        raise SystemExit(run_parallel_worker(args))
    if args.postprocess_only:
        raise SystemExit(run_postprocess_only(args))
    raise SystemExit(run_serial(args))


if __name__ == "__main__":
    main()
