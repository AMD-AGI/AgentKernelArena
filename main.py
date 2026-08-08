# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import argparse
import hashlib
import json
import logging
import os
import re
import stat
import tempfile
import time
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
from src.harness_guard import snapshot_workspace_harness, verify_workspace_harness
from src.campaign import (
    CampaignError,
    FORMAL_LIVE_EXECUTION_SHA256,
    campaign_task_path_component,
    ensure_campaign_manifest,
    parse_campaign_policy,
    resolve_session_receipt_schema,
    run_matched_task_campaign,
    deterministic_task_gpu_mapping,
    ordered_gpu_pool,
    _campaign_failure_reasons,
    validate_formal_task_binding,
)


QUEUE_DIR_NAME = ".parallel"
QUEUE_STATES = ("pending", "running", "done", "failed")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_FORMAL_PRIMARY_FAILURE_REASONS = frozenset(
    {"descriptor_gpu_affinity_mismatch", "formal_task_not_canonical"}
)


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
    if not tasks:
        raise ValueError("No task selectors were configured")

    if "all" in tasks:
        discovered = get_task_config()
        if not discovered:
            raise ValueError("Task selector 'all' matched no task configs")
        return discovered

    task_config_dict: dict[str, str] = {}
    for category in tasks:
        discovered = get_task_config(category=category)
        if not discovered:
            raise ValueError(
                f"Configured task selector {category!r} matched no task configs"
            )
        task_config_dict.update(discovered)
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

    required_arch = platform_support.get("required_arch")
    if not required_arch:
        return True
    if not isinstance(required_arch, str):
        logger.warning(
            "Task %s has non-string platform_support.required_arch=%r; treating task as runnable",
            task_name,
            required_arch,
        )
        return True

    required_arch = required_arch.strip()
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
    if required_arch != current_gfx_arch:
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
        configured_tasks = _discover_tasks(tasks)
    except ValueError as error:
        logger.error("Task discovery failed: %s", error)
        return None
    logger.info(f"Found {len(configured_tasks)} configured task(s)")
    task_config_dict = filter_tasks_by_platform(configured_tasks, current_gfx_arch, logger)
    logger.info(f"Found {len(task_config_dict)} runnable task(s) after platform preflight")
    logger.info(f"Tasks: {list(task_config_dict.keys())}")

    context = {
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
    try:
        campaign_manifest = ensure_campaign_manifest(
            run_directory=run_directory,
            eval_config=config,
            run_config_path=Path(args.config_name),
            task_config_paths=task_config_dict,
            agent_name=agent.value,
        )
    except CampaignError as error:
        logger.error("Campaign preflight failed: %s", error)
        return None
    if campaign_manifest is not None:
        logger.info("Pinned matched-campaign manifest: %s", campaign_manifest)
    return context


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


def _run_single_task(
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
    campaign_attempt = eval_config.get("campaign_attempt")
    deadline = (
        float(campaign_attempt["task_deadline_monotonic"])
        if isinstance(campaign_attempt, dict)
        and campaign_attempt.get("task_deadline_monotonic") is not None
        else None
    )
    evaluation_elapsed = 0.0
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

        # Task packages may include a previously committed validator report.
        # It is evidence about an older source snapshot, not completion evidence
        # for this run. Remove the copied report before launching the validator
        # so an agent/backend failure cannot be mistaken for a successful run.
        if is_validator:
            stale_report = workspace_path / "validation_report.yaml"
            if stale_report.exists():
                stale_report.unlink()
                logger.info("Removed copied stale validation_report.yaml before validation")

        baseline_cases = []
        evaluation_started = time.monotonic()
        if is_validator:
            logger.info("task_validator run: skipping baseline/evaluation/perf-plot benchmark pipeline")
        elif task_type == "torch2hip":
            logger.info("torch2hip task: skipping baseline compilation, measuring PyTorch baseline directly...")
            baseline_cases = measure_baseline(
                workspace_path, task_config, logger, deadline_monotonic=deadline
            )
        else:
            logger.info("Compiling original kernel for baseline measurement...")
            pass_compilation, comp_error = evaluate_compilation(
                workspace_path, task_config, logger, deadline_monotonic=deadline
            )
            if not pass_compilation:
                logger.warning(f"Baseline compilation failed: {comp_error}")
                logger.warning("Baseline measurement will be skipped")
                baseline_cases = []
            else:
                logger.info("Measuring baseline performance...")
                baseline_cases = measure_baseline(
                    workspace_path, task_config, logger, deadline_monotonic=deadline
                )
        evaluation_elapsed += time.monotonic() - evaluation_started

        harness_snapshot = snapshot_workspace_harness(workspace_path)

        logger.info(f"Launching agent: {agent.value}")
        agent_error: Exception | None = None
        try:
            agent_launcher(
                eval_config=eval_config,
                task_config_dir=task_config_dir,
                workspace=str(workspace_path),
            )
            logger.info("Agent execution completed")
        except Exception as error:
            if not isinstance(campaign_attempt, dict):
                raise
            agent_error = error
            logger.error(
                "Campaign agent session failed; running diagnostic central evaluation: %s",
                error,
                exc_info=True,
            )

        if not is_validator:
            # Agents work inside the task workspace and could accidentally modify
            # protected harness/test files or generated perf helpers. Verify the
            # harness is untouched, then re-materialize perf helpers from
            # src/tools/perf/ so benchmark methodology stays canonical.
            verify_workspace_harness(harness_snapshot, logger=logger)
            materialize_perf_helpers_in_workspace(workspace_path, logger=logger)
            logger.info("Running centralized evaluation...")
            evaluation_started = time.monotonic()
            evaluation_results = evaluate_kernel(
                workspace_path,
                task_config,
                baseline_cases,
                logger,
                deadline_monotonic=deadline,
            )
            write_task_result(
                workspace_path,
                evaluation_results,
                baseline_cases,
                task_name,
                agent.value,
                logger,
            )
            evaluation_elapsed += time.monotonic() - evaluation_started
            if isinstance(campaign_attempt, dict):
                result_path = workspace_path / "task_result.yaml"
                result = yaml.safe_load(result_path.read_text(encoding="utf-8")) or {}
                result.update(
                    _campaign_evaluation_metadata(
                        agent=agent,
                        campaign_attempt=campaign_attempt,
                        agent_error=agent_error,
                    )
                )
                result_path.write_text(
                    yaml.safe_dump(result, default_flow_style=False, sort_keys=False),
                    encoding="utf-8",
                )

        if isinstance(campaign_attempt, dict):
            campaign_attempt["evaluation_elapsed_seconds"] = evaluation_elapsed

        if not is_task_complete(run_directory, task_name, timestamp, agent.value):
            expected_report = "validation_report.yaml" if is_validator else "task_result.yaml"
            logger.error(f"Task {task_name} did not produce expected completion report: {expected_report}")
            return False, workspace_path

        if agent_error is not None:
            logger.error("Task %s has diagnostic evidence but its agent session failed", task_name)
            return False, workspace_path
        logger.info(f"Task {task_name} completed successfully")
        return True, workspace_path
    except Exception as e:
        if isinstance(campaign_attempt, dict):
            campaign_attempt["evaluation_elapsed_seconds"] = evaluation_elapsed
        logger.error(f"Task {task_name} failed with error: {e}", exc_info=True)
        return False, workspace_path


def _campaign_evaluation_metadata(
    *,
    agent: AgentType,
    campaign_attempt: dict[str, Any],
    agent_error: Exception | None,
) -> dict[str, Any]:
    """Classify evaluation from the sealed treatment and its immutable receipt."""

    session_succeeded = agent_error is None
    metadata: dict[str, Any] = {
        "evaluation_mode": (
            "candidate_scoring_v1"
            if session_succeeded
            else "diagnostic_baseline_replay_v1"
        ),
        "agent_session_score_eligible": session_succeeded,
        "agent_session_succeeded": session_succeeded,
        "agent_session_error_type": (
            type(agent_error).__name__ if agent_error is not None else None
        ),
        "agent_session_terminal_status": None,
    }
    if agent not in {AgentType.APEX, AgentType.CODEX}:
        return metadata
    if agent_error is not None:
        return metadata

    try:
        manifest_path = Path(campaign_attempt["campaign_manifest_path"])
        manifest_metadata = manifest_path.lstat()
        expected_manifest_sha256 = campaign_attempt["campaign_manifest_sha256"]
        if (
            not manifest_path.is_absolute()
            or not stat.S_ISREG(manifest_metadata.st_mode)
            or manifest_path.is_symlink()
            or manifest_metadata.st_nlink != 1
            or manifest_metadata.st_mode & 0o222
            or not isinstance(expected_manifest_sha256, str)
            or not _SHA256.fullmatch(expected_manifest_sha256)
            or hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            != expected_manifest_sha256
        ):
            raise ValueError("unsafe or changed campaign manifest")
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        sealed_agent = manifest.get("agent") if isinstance(manifest, dict) else None
        sealed_comparison = (
            manifest.get("comparison_contract") if isinstance(manifest, dict) else None
        )
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema") != "aka.matched-campaign/v1"
            or manifest.get("formal_execution_sha256")
            != FORMAL_LIVE_EXECUTION_SHA256
            or not isinstance(sealed_comparison, dict)
            or sealed_comparison.get("formal_execution_sha256")
            != FORMAL_LIVE_EXECUTION_SHA256
            or not isinstance(sealed_agent, dict)
            or sealed_agent.get("template") != agent.value
        ):
            raise ValueError("campaign agent template mismatch")
        expected_receipt_schema = resolve_session_receipt_schema(
            agent.value, sealed_agent.get("session_receipt_schema")
        )
        if expected_receipt_schema is None:
            raise ValueError("unsupported sealed receipt schema")
    except (KeyError, OSError, TypeError, ValueError, yaml.YAMLError, UnicodeDecodeError):
        metadata["evaluation_mode"] = "diagnostic_unbound_session_replay_v1"
        metadata["agent_session_score_eligible"] = False
        metadata["agent_session_succeeded"] = False
        if agent_error is None:
            metadata["agent_session_error_type"] = "CampaignManifestMetadataError"
        return metadata

    raw_receipt = campaign_attempt.get("receipt_path")
    try:
        receipt_path = Path(raw_receipt)
        receipt_metadata = receipt_path.lstat()
        if (
            not receipt_path.is_absolute()
            or not stat.S_ISREG(receipt_metadata.st_mode)
            or receipt_path.is_symlink()
            or receipt_metadata.st_nlink != 1
            or receipt_metadata.st_mode & 0o222
        ):
            raise ValueError("unsafe Apex session receipt")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if not isinstance(receipt, dict):
            raise ValueError("Apex session receipt is not an object")
    except (OSError, TypeError, ValueError, json.JSONDecodeError, UnicodeDecodeError):
        metadata["evaluation_mode"] = "diagnostic_unbound_session_replay_v1"
        metadata["agent_session_score_eligible"] = False
        metadata["agent_session_succeeded"] = False
        if agent_error is None:
            metadata["agent_session_error_type"] = "ApexReceiptMetadataError"
        return metadata

    terminal_status = receipt.get("terminal_status")
    metadata["agent_session_terminal_status"] = (
        terminal_status if isinstance(terminal_status, str) else None
    )
    if agent_error is not None:
        return metadata
    if (
        receipt.get("schema") != expected_receipt_schema
        or receipt.get("session_succeeded") is not True
    ):
        metadata["evaluation_mode"] = "diagnostic_unbound_session_replay_v1"
        metadata["agent_session_score_eligible"] = False
        metadata["agent_session_succeeded"] = False
        metadata["agent_session_error_type"] = "ApexReceiptMetadataError"
    elif agent is AgentType.APEX and terminal_status == "candidate_ready":
        metadata["evaluation_mode"] = "candidate_scoring_v1"
        metadata["agent_session_score_eligible"] = True
    elif agent is AgentType.APEX and terminal_status == "no_gain":
        metadata["evaluation_mode"] = "no_candidate_baseline_replay_v1"
        metadata["agent_session_score_eligible"] = False
    elif agent is AgentType.CODEX:
        integrity = receipt.get("workspace_integrity")
        final_changes = (
            integrity.get("final_changes") if isinstance(integrity, dict) else None
        )
        changed_files = (
            final_changes.get("changed_files")
            if isinstance(final_changes, dict)
            else None
        )
        if not isinstance(changed_files, list) or any(
            not isinstance(path, str) or not path for path in changed_files
        ):
            metadata["evaluation_mode"] = "diagnostic_unbound_session_replay_v1"
            metadata["agent_session_score_eligible"] = False
            metadata["agent_session_succeeded"] = False
            metadata["agent_session_error_type"] = "CodexReceiptMetadataError"
        elif changed_files:
            metadata["evaluation_mode"] = "candidate_scoring_v1"
            metadata["agent_session_score_eligible"] = True
            metadata["agent_session_terminal_status"] = "candidate_ready"
        else:
            metadata["evaluation_mode"] = "no_candidate_baseline_replay_v1"
            metadata["agent_session_score_eligible"] = False
            metadata["agent_session_terminal_status"] = "no_gain"
    else:
        metadata["evaluation_mode"] = "diagnostic_unbound_session_replay_v1"
        metadata["agent_session_score_eligible"] = False
    return metadata


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
    """Run one ordinary task or a three-session matched campaign task."""
    if parse_campaign_policy(eval_config) is None:
        return _run_single_task(
            eval_config=eval_config,
            agent=agent,
            agent_launcher=agent_launcher,
            task_name=task_name,
            task_config_dir=task_config_dir,
            run_directory=run_directory,
            timestamp=timestamp,
            logger=logger,
            task_index=task_index,
            total_tasks=total_tasks,
        )
    try:
        return run_matched_task_campaign(
            eval_config=eval_config,
            agent=agent,
            agent_launcher=agent_launcher,
            task_name=task_name,
            task_config_dir=task_config_dir,
            run_directory=run_directory,
            timestamp=timestamp,
            logger=logger,
            task_index=task_index,
            total_tasks=total_tasks,
            single_attempt=_run_single_task,
        )
    except CampaignError as error:
        logger.error("Matched campaign task failed: %s", error, exc_info=True)
        return False, None


def run_post_processing(
    agent: AgentType,
    workspace_paths: list[str],
    logger: logging.Logger,
    *,
    run_directory: Path | None = None,
) -> None:
    logger.info("=" * 80)
    logger.info("Running Post-Processing")
    logger.info("=" * 80)

    formal = bool(
        run_directory is not None
        and (run_directory / "campaign_manifest.yaml").exists()
    )
    try:
        post_processing_handler = load_post_processing_handler(agent, logger)
        if agent == AgentType.TASK_VALIDATOR:
            post_processing_handler(workspace_paths, logger)
        else:
            post_processing_handler(
                workspace_paths, logger, run_directory=run_directory
            )
    except NotImplementedError as e:
        logger.warning(f"Post-processing skipped: {e}")
        if formal:
            raise
    except Exception as e:
        logger.error(f"Post-processing failed: {e}", exc_info=True)
        if formal:
            raise


def _queue_root(run_directory: Path) -> Path:
    return run_directory / QUEUE_DIR_NAME


def _queue_state_dir(run_directory: Path, state: str) -> Path:
    return _queue_root(run_directory) / state


def _descriptor_name(index: int, task_name: str) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", task_name).strip("_")
    return f"{index:06d}_{safe_name or 'task'}.yaml"


def _write_descriptor(
    path: Path, payload: dict[str, Any], *, no_clobber: bool = False
) -> None:
    encoded = yaml.safe_dump(
        payload, default_flow_style=False, sort_keys=False
    ).encode("utf-8")
    if no_clobber:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags, 0o600)
        except FileExistsError as error:
            raise CampaignError(f"descriptor already exists: {path}") from error
        try:
            written = 0
            while written < len(encoded):
                written += os.write(descriptor, encoded[written:])
            os.fsync(descriptor)
        except Exception:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            raise
        finally:
            os.close(descriptor)
        return

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        written = 0
        while written < len(encoded):
            written += os.write(descriptor, encoded[written:])
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _read_descriptor(path: Path) -> dict[str, Any]:
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def _formal_descriptor_snapshot(
    path: Path,
) -> tuple[tuple[int, int, int, str], dict[str, Any]]:
    """Read one regular descriptor without following links and return its identity."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        # Another worker may atomically claim a path returned by our glob before
        # we open it. Preserve ENOENT so claim_next_descriptor can treat that as
        # the expected loser side of the race and continue scanning.
        raise
    except OSError as error:
        raise CampaignError(f"cannot safely open formal descriptor: {path}") from error
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_mode & 0o222
        ):
            raise CampaignError(f"unsafe formal descriptor: {path}")
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > 1024 * 1024:
                raise CampaignError(f"formal descriptor exceeds size limit: {path}")
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise CampaignError(f"formal descriptor changed while being read: {path}")
    encoded = b"".join(chunks)
    try:
        payload = yaml.safe_load(encoded) or {}
    except (UnicodeError, yaml.YAMLError) as error:
        raise CampaignError(f"formal descriptor is unreadable: {path}") from error
    if not isinstance(payload, dict):
        raise CampaignError(f"formal descriptor must contain a mapping: {path}")
    identity = (
        before.st_dev,
        before.st_ino,
        len(encoded),
        hashlib.sha256(encoded).hexdigest(),
    )
    return identity, payload


def _validate_formal_descriptor_payload(
    run_directory: Path,
    descriptor: Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    index = payload.get("index")
    total_tasks = payload.get("total_tasks")
    task_name = payload.get("task_name")
    task_config_dir = payload.get("task_config_dir")
    assigned_gpu = payload.get("assigned_host_gpu_id")
    if (
        type(index) is not int
        or type(total_tasks) is not int
        or not isinstance(task_name, str)
        or not task_name
        or not isinstance(task_config_dir, str)
        or not isinstance(assigned_gpu, str)
        or payload.get("status") != "pending"
    ):
        raise CampaignError("formal descriptor fields are malformed")
    expected_name = _descriptor_name(index, task_name)
    if descriptor.parent.name == "pending":
        name_matches = descriptor.name == expected_name
    elif descriptor.parent.name == "running":
        name_matches = descriptor.name.endswith(f"__{expected_name}")
    else:
        name_matches = False
    if not name_matches:
        raise CampaignError("formal descriptor filename differs from task identity")
    binding = validate_formal_task_binding(
        run_directory=run_directory,
        task_name=task_name,
        task_index=index,
        total_tasks=total_tasks,
        task_config_path=task_config_dir,
        assigned_host_gpu_id=assigned_gpu,
    )
    if binding.get("formal_execution_sha256") != FORMAL_LIVE_EXECUTION_SHA256:
        raise CampaignError("formal descriptor is not bound to live v5 execution")
    return binding


def _require_safe_queue_directory(path: Path) -> None:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise CampaignError(f"formal queue directory is unavailable: {path}") from error
    if not stat.S_ISDIR(metadata.st_mode) or path.is_symlink():
        raise CampaignError(f"unsafe formal queue directory: {path}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _formal_failure_binding(
    run_directory: Path,
    task_name: str,
    explicit_reason: str | None,
) -> dict[str, Any]:
    if explicit_reason not in _FORMAL_PRIMARY_FAILURE_REASONS:
        raise CampaignError(f"unsupported formal failure reason: {explicit_reason!r}")
    manifest_path = run_directory / "campaign_manifest.yaml"
    try:
        manifest_metadata = manifest_path.lstat()
        manifest_safe = (
            manifest_path.is_file()
            and not manifest_path.is_symlink()
            and manifest_metadata.st_nlink == 1
            and not manifest_metadata.st_mode & 0o222
        )
    except OSError:
        manifest_safe = False
    if not manifest_safe:
        raise CampaignError("formal failure marker requires immutable campaign manifest")
    manifest = _read_descriptor(manifest_path)
    comparison = manifest.get("comparison_contract")
    comparison_sha256 = manifest.get("comparison_contract_sha256")
    campaign_manifest_sha256 = _sha256_file(manifest_path)
    tasks = manifest.get("configuration", {}).get("tasks")
    if (
        manifest.get("schema") != "aka.matched-campaign/v1"
        or manifest.get("formal_execution_sha256")
        != FORMAL_LIVE_EXECUTION_SHA256
        or not isinstance(comparison, dict)
        or comparison.get("formal_execution_sha256")
        != FORMAL_LIVE_EXECUTION_SHA256
        or not isinstance(comparison_sha256, str)
        or not _SHA256.fullmatch(comparison_sha256)
        or hashlib.sha256(
            json.dumps(comparison, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        != comparison_sha256
        or not isinstance(tasks, list)
        or sum(
            isinstance(task, dict) and task.get("task_name") == task_name
            for task in tasks
        )
        != 1
    ):
        raise CampaignError("formal failure marker campaign binding is invalid")

    evidence_path = (
        run_directory
        / ".campaign_attempts"
        / campaign_task_path_component(task_name)
        / "task_campaign.yaml"
    )
    binding: dict[str, Any] = {
        "schema": "aka.formal-task-failure/v1",
        "task_name": task_name,
        "primary_reason": None,
        "campaign_manifest_sha256": campaign_manifest_sha256,
        "comparison_contract_sha256": comparison_sha256,
        "campaign_evidence_path": None,
        "campaign_evidence_sha256": None,
        "reason_codes": ["immutable_task_campaign_evidence_missing"],
    }
    try:
        metadata = evidence_path.lstat()
        safe = (
            evidence_path.is_file()
            and not evidence_path.is_symlink()
            and metadata.st_nlink == 1
            and not metadata.st_mode & 0o222
        )
    except OSError:
        safe = False
    if not safe:
        return binding

    try:
        evidence = _read_descriptor(evidence_path)
    except (OSError, UnicodeError, yaml.YAMLError):
        binding["reason_codes"] = ["task_campaign_evidence_unreadable"]
        return binding
    if not isinstance(evidence, dict):
        binding["reason_codes"] = ["task_campaign_evidence_unreadable"]
        return binding
    evidence_reasons = _campaign_failure_reasons(evidence)
    if (
        evidence.get("schema") != "aka.matched-task-attempts/v1"
        or evidence.get("task_name") != task_name
        or evidence.get("campaign_manifest_sha256") != campaign_manifest_sha256
        or evidence.get("comparison_contract_sha256") != comparison_sha256
        or evidence.get("failure_reasons") != evidence_reasons
        or not evidence_reasons
    ):
        binding["reason_codes"] = ["task_campaign_evidence_contract_invalid"]
        return binding
    reason_codes = sorted(set(evidence_reasons + [explicit_reason]))
    binding.update(
        {
            "primary_reason": explicit_reason,
            "campaign_evidence_path": str(evidence_path.relative_to(run_directory)),
            "campaign_evidence_sha256": _sha256_file(evidence_path),
            "reason_codes": reason_codes,
        }
    )
    return binding


def _validated_formal_task_bindings(
    run_directory: Path,
    task_config_dict: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """Resolve every task through the immutable campaign GPU binding."""

    assignments = deterministic_task_gpu_mapping(list(task_config_dict))
    bindings: dict[str, dict[str, Any]] = {}
    for index, (task_name, task_config_dir) in enumerate(
        task_config_dict.items(), 1
    ):
        bindings[task_name] = validate_formal_task_binding(
            run_directory=run_directory,
            task_name=task_name,
            task_index=index,
            total_tasks=len(task_config_dict),
            task_config_path=task_config_dir,
            assigned_host_gpu_id=assignments[index - 1][
                "assigned_host_gpu_id"
            ],
        )
    return bindings


def initialize_parallel_queue(context: dict[str, Any]) -> None:
    run_directory: Path = context["run_directory"]
    task_config_dict: dict[str, str] = context["task_config_dict"]
    timestamp: str = context["timestamp"]
    agent: AgentType = context["agent"]
    logger: logging.Logger = context["logger"]

    formal = parse_campaign_policy(context["config"]) is not None
    formal_bindings = (
        _validated_formal_task_bindings(run_directory, task_config_dict)
        if formal
        else {}
    )

    queue_root = _queue_root(run_directory)
    if formal:
        if queue_root.exists() or queue_root.is_symlink():
            raise CampaignError(
                f"formal parallel queue already exists; use a fresh run: {queue_root}"
            )
        queue_root.mkdir()
        for state in QUEUE_STATES:
            _queue_state_dir(run_directory, state).mkdir()
    else:
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
        binding = formal_bindings.get(task_name)
        payload = {
            "index": index,
            "total_tasks": total_tasks,
            "task_name": task_name,
            "task_config_dir": (
                binding["config_path"] if binding is not None else task_config_dir
            ),
            "workspace_path": str(workspace_path),
        }
        if binding is not None:
            payload["assigned_host_gpu_id"] = binding["assigned_host_gpu_id"]
        if is_task_complete(run_directory, task_name, timestamp, agent.value):
            payload["status"] = "already_complete"
            state = "done"
            completed += 1
        else:
            payload["status"] = "pending"
            state = "pending"
            queued += 1
        descriptor_path = (
            _queue_state_dir(run_directory, state)
            / _descriptor_name(index, task_name)
        )
        _write_descriptor(
            descriptor_path,
            payload,
            no_clobber=formal,
        )
        if formal:
            descriptor_path.chmod(0o444)

    logger.info(
        f"Parallel queue initialized: queued={queued}, already_complete={completed}, "
        f"total={total_tasks}"
    )


def claim_next_descriptor(
    run_directory: Path,
    worker_id: str,
    logger: logging.Logger,
    host_gpu_id: str | None = None,
) -> Path | None:
    pending_dir = _queue_state_dir(run_directory, "pending")
    running_dir = _queue_state_dir(run_directory, "running")
    running_dir.mkdir(parents=True, exist_ok=True)
    formal = (run_directory / "campaign_manifest.yaml").exists()
    if formal:
        _require_safe_queue_directory(pending_dir)
        _require_safe_queue_directory(running_dir)

    for descriptor in sorted(pending_dir.glob("*.yaml")):
        if formal:
            try:
                identity, payload = _formal_descriptor_snapshot(descriptor)
            except FileNotFoundError:
                # A competing worker won the atomic rename after this worker's
                # directory scan but before it could open the descriptor.
                continue
            _validate_formal_descriptor_payload(run_directory, descriptor, payload)
        else:
            payload = _read_descriptor(descriptor)
        assigned = payload.get("assigned_host_gpu_id")
        if assigned is not None and assigned != host_gpu_id:
            continue
        if formal:
            if not re.fullmatch(r"[A-Za-z0-9._-]+", worker_id):
                raise CampaignError("formal worker ID is not filename-safe")
            claimed = running_dir / f"worker_{worker_id}__{descriptor.name}"
            if claimed.exists() or claimed.is_symlink():
                raise CampaignError(f"formal claim path already exists: {claimed}")
        else:
            worker_component = re.sub(
                r"[^A-Za-z0-9._-]+", "_", worker_id
            ).strip("_") or "worker"
            claimed = running_dir / f"worker_{worker_component}__{descriptor.name}"
        if formal:
            try:
                os.rename(descriptor, claimed)
            except FileNotFoundError:
                # Another worker won the claim between our validated snapshot
                # and the atomic rename.
                continue
            try:
                claimed_identity, claimed_payload = _formal_descriptor_snapshot(claimed)
            except FileNotFoundError as error:
                raise CampaignError(
                    "formal claimed descriptor disappeared before verification"
                ) from error
            if claimed_identity != identity or claimed_payload != payload:
                raise CampaignError("formal descriptor identity changed during claim")
            _validate_formal_descriptor_payload(
                run_directory, claimed, claimed_payload
            )
        else:
            try:
                descriptor.rename(claimed)
            except FileNotFoundError:
                continue
        logger.info(f"Claimed task descriptor: {claimed.name}")
        return claimed
    return None


def finish_descriptor(
    descriptor: Path,
    state: str,
    *,
    workspace_path: Path | None,
    worker_id: str,
    failure_reason: str | None = None,
) -> None:
    run_directory = descriptor.parent.parent.parent
    formal = (run_directory / "campaign_manifest.yaml").exists()
    if formal:
        _, payload = _formal_descriptor_snapshot(descriptor)
    else:
        payload = _read_descriptor(descriptor)
    payload["status"] = state
    payload["worker_id"] = worker_id
    if workspace_path is not None:
        payload["workspace_path"] = str(workspace_path)
    if state == "failed" and formal:
        payload["failure"] = _formal_failure_binding(
            run_directory,
            str(payload.get("task_name") or ""),
            failure_reason,
        )
    _write_descriptor(descriptor, payload)
    final_dir = descriptor.parent.parent / state
    final_dir.mkdir(parents=True, exist_ok=True)
    final_path = final_dir / descriptor.name
    if formal:
        _require_safe_queue_directory(descriptor.parent)
        _require_safe_queue_directory(final_dir)
        if final_path.exists() or final_path.is_symlink():
            raise CampaignError(
                f"formal terminal descriptor already exists: {final_path}"
            )
        descriptor.chmod(0o444)
        identity, payload_after_write = _formal_descriptor_snapshot(descriptor)
        os.rename(descriptor, final_path)
        final_identity, final_payload = _formal_descriptor_snapshot(final_path)
        if final_identity != identity or final_payload != payload_after_write:
            raise CampaignError("formal descriptor identity changed during finalization")
    else:
        descriptor.rename(final_path)


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
    try:
        formal_bindings = (
            _validated_formal_task_bindings(
                context["run_directory"], task_config_dict
            )
            if parse_campaign_policy(context["config"]) is not None
            else {}
        )
    except CampaignError as error:
        context["logger"].error(
            "Formal serial task binding failed: %s", error
        )
        return 1

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
        binding = formal_bindings.get(task_name)
        task_eval_config = context["config"]
        task_index = index
        task_total = total_tasks
        if binding is not None:
            task_eval_config = dict(context["config"])
            task_eval_config["assigned_host_gpu_id"] = binding[
                "assigned_host_gpu_id"
            ]
            task_config_dir = binding["config_path"]
            task_index = binding["task_index"]
            task_total = binding["total_tasks"]
        _, workspace_path = run_task(
            eval_config=task_eval_config,
            agent=context["agent"],
            agent_launcher=context["agent_launcher"],
            task_name=task_name,
            task_config_dir=task_config_dir,
            run_directory=context["run_directory"],
            timestamp=context["timestamp"],
            logger=context["logger"],
            task_index=task_index,
            total_tasks=task_total,
        )
        if workspace_path is not None:
            workspace_paths.append(str(workspace_path))

    try:
        run_post_processing(
            context["agent"],
            workspace_paths,
            context["logger"],
            run_directory=context["run_directory"],
        )
    except Exception:
        return 1
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
    host_gpu_id = os.environ.get("AGENT_KERNEL_ARENA_HOST_GPU_ID")
    formal_campaign = parse_campaign_policy(context["config"]) is not None
    if formal_campaign:
        pool = ordered_gpu_pool()
        if host_gpu_id not in pool:
            context["logger"].error(
                "Worker host GPU %r is not in deterministic pool %s", host_gpu_id, pool
            )
            return 1
    while True:
        descriptor = claim_next_descriptor(
            context["run_directory"], worker_id, context["logger"], host_gpu_id
        )
        if descriptor is None:
            break

        if formal_campaign:
            _, payload = _formal_descriptor_snapshot(descriptor)
            _validate_formal_descriptor_payload(
                context["run_directory"], descriptor, payload
            )
        else:
            payload = _read_descriptor(descriptor)
        assigned_host_gpu_id = payload.get("assigned_host_gpu_id")
        if assigned_host_gpu_id is not None and assigned_host_gpu_id != host_gpu_id:
            context["logger"].error(
                "Descriptor GPU affinity mismatch: assigned=%s worker=%s",
                assigned_host_gpu_id,
                host_gpu_id,
            )
            finish_descriptor(
                descriptor,
                "failed",
                workspace_path=None,
                worker_id=worker_id,
                failure_reason="descriptor_gpu_affinity_mismatch",
            )
            failures += 1
            continue
        task_eval_config = dict(context["config"])
        if assigned_host_gpu_id is not None:
            task_eval_config["assigned_host_gpu_id"] = assigned_host_gpu_id
        success, workspace_path = run_task(
            eval_config=task_eval_config,
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
            finish_descriptor(
                descriptor,
                "failed",
                workspace_path=workspace_path,
                worker_id=worker_id,
                failure_reason="formal_task_not_canonical",
            )

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
    try:
        run_post_processing(
            context["agent"],
            workspace_paths,
            context["logger"],
            run_directory=context["run_directory"],
        )
    except Exception:
        return 1

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
