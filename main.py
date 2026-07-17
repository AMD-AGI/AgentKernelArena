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
    setup_workspace,
)
from src.module_registration import AgentType, load_agent_launcher, load_post_processing_handler
from src.evaluator import (
    evaluate_compilation,
    evaluate_kernel,
    measure_baseline,
    write_task_result,
)
from src.runtime_env import apply_subprocess_python_path
from src.perf_helper_materialization import materialize_perf_helpers_in_workspace


QUEUE_DIR_NAME = ".parallel"
QUEUE_STATES = ("pending", "running", "done", "failed")


parser = argparse.ArgumentParser(description="arguments for AgentKernelArena")
parser.add_argument(
    "--config_name",
    type=str,
    default="example_configs/quickstart_claude_mi300.yaml",
    help=(
        "run configuration for AgentKernelArena (default: "
        "example_configs/quickstart_claude_mi300.yaml for MI300/MI300X). "
        "Select a matching config explicitly when using another GPU."
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


def _discover_tasks(tasks: list[str]) -> dict[str, str]:
    if "all" in tasks:
        return get_task_config()

    task_config_dict: dict[str, str] = {}
    for category in tasks:
        task_config_dict.update(get_task_config(category=category))
    return task_config_dict


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

    agent_launcher = None
    if need_agent_launcher:
        try:
            agent_launcher = load_agent_launcher(agent, logger)
        except Exception as e:
            logger.error(f"Failed to load agent launcher: {e}")
            return None

    task_config_dict = _discover_tasks(tasks)
    logger.info(f"Found {len(task_config_dict)} configured task(s)")
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
    workspace_path: Path | None = None
    logger.info("=" * 80)
    logger.info(f"Task {task_index}/{total_tasks}: {task_name}")
    logger.info("=" * 80)

    try:
        workspace_path = setup_workspace(
            task_config_dir,
            run_directory,
            timestamp,
            logger,
            task_name=task_name,
        )

        with open(task_config_dir, "r") as f:
            task_config = yaml.safe_load(f) or {}

        task_type = task_config.get("task_type", "")
        is_validator = agent == AgentType.TASK_VALIDATOR

        baseline_cases = []
        if is_validator:
            logger.info("task_validator run: skipping baseline/evaluation/perf-plot benchmark pipeline")
        elif task_type == "torch2hip":
            logger.info("torch2hip task: skipping baseline compilation, measuring PyTorch baseline directly...")
            baseline_cases = measure_baseline(workspace_path, task_config, logger)
        else:
            logger.info("Compiling original kernel for baseline measurement...")
            pass_compilation, comp_error = evaluate_compilation(workspace_path, task_config, logger)
            if not pass_compilation:
                logger.warning(f"Baseline compilation failed: {comp_error}")
                logger.warning("Baseline measurement will be skipped")
                baseline_cases = []
            else:
                logger.info("Measuring baseline performance...")
                baseline_cases = measure_baseline(workspace_path, task_config, logger)

        logger.info(f"Launching agent: {agent.value}")
        agent_launcher(
            eval_config=eval_config,
            task_config_dir=task_config_dir,
            workspace=str(workspace_path),
        )
        logger.info("Agent execution completed")

        if not is_validator:
            materialize_perf_helpers_in_workspace(workspace_path, logger=logger)
            logger.info("Running centralized evaluation...")
            evaluation_results = evaluate_kernel(
                workspace_path,
                task_config,
                baseline_cases,
                logger,
            )
            write_task_result(
                workspace_path,
                evaluation_results,
                baseline_cases,
                task_name,
                agent.value,
                logger,
            )

        if not is_task_complete(run_directory, task_name, timestamp, agent.value):
            expected_report = "validation_report.yaml" if is_validator else "task_result.yaml"
            logger.error(f"Task {task_name} did not produce expected completion report: {expected_report}")
            return False, workspace_path

        logger.info(f"Task {task_name} completed successfully")
        return True, workspace_path
    except Exception as e:
        logger.error(f"Task {task_name} failed with error: {e}", exc_info=True)
        return False, workspace_path


def run_post_processing(agent: AgentType, workspace_paths: list[str], logger: logging.Logger) -> None:
    logger.info("=" * 80)
    logger.info("Running Post-Processing")
    logger.info("=" * 80)

    try:
        post_processing_handler = load_post_processing_handler(agent, logger)
        post_processing_handler(workspace_paths, logger)
    except NotImplementedError as e:
        logger.warning(f"Post-processing skipped: {e}")
    except Exception as e:
        logger.error(f"Post-processing failed: {e}", exc_info=True)


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

    task_config_dict = context["task_config_dict"]
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
        return 0

    workspace_paths: list[str] = []
    total_tasks = len(task_config_dict)
    for index, (task_name, task_config_dir) in enumerate(task_config_dict.items(), 1):
        _, workspace_path = run_task(
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

    run_post_processing(context["agent"], workspace_paths, context["logger"])
    context["logger"].info("=" * 80)
    context["logger"].info("AgentKernelArena Framework Completed")
    context["logger"].info("=" * 80)
    return 0


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
    run_post_processing(context["agent"], workspace_paths, context["logger"])

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
