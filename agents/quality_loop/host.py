# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Host-side GitHub preflight and publication for credential isolation."""
from __future__ import annotations

import argparse
import dataclasses
import logging
from datetime import datetime, timezone
from pathlib import Path

from .config import QualityLoopConfig, load_config
from .github import GitHubPublisher
from .orchestrator import QualityLoop
from .state import AuditState, resolve_worktree, stable_fingerprint, validate_run_id


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("agent_config.yaml")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="quality_loop host publication boundary")
    parser.add_argument("action", choices=("start", "check", "paths", "finalize"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--no-publish", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument("--run-id")
    return parser


def _effective_config(args: argparse.Namespace) -> QualityLoopConfig:
    path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    config = load_config(path)
    if args.tasks:
        config = dataclasses.replace(config, tasks=tuple(args.tasks))
    if args.no_publish:
        config = dataclasses.replace(
            config,
            github=dataclasses.replace(config.github, publish=False),
        )
    return config


def _logger() -> logging.Logger:
    logger = logging.getLogger("quality_loop.host")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    return logger


def _paths(config: QualityLoopConfig, run_id: str) -> tuple[Path, Path, Path]:
    artifact = (REPO_ROOT / config.artifact_root / run_id).resolve()
    worktree = (REPO_ROOT / config.worktree_root / run_id).resolve()
    return artifact, worktree, artifact / "state.yaml"


def start(config: QualityLoopConfig, run_id: str, logger: logging.Logger) -> str:
    validate_run_id(run_id)
    publisher = GitHubPublisher(REPO_ROOT, config.github, logger)
    preflight = publisher.preflight()
    artifact, worktree, state_path = _paths(config, run_id)
    if artifact.exists() or worktree.exists():
        raise RuntimeError(f"quality_loop run already exists: {run_id}")
    branch = f"{config.github.branch_prefix}/{run_id}"
    publisher.create_worktree(
        path=worktree,
        branch=branch,
        base_branch=preflight.default_branch,
    )
    AuditState.create(
        state_path,
        run_id=run_id,
        config_fingerprint=stable_fingerprint(config),
        repo_slug=preflight.repo_slug,
        base_sha=preflight.base_sha,
        base_branch=preflight.default_branch,
        branch=branch,
        worktree=worktree.relative_to(REPO_ROOT),
    )
    return run_id


def check(config: QualityLoopConfig, run_id: str, logger: logging.Logger) -> str:
    validate_run_id(run_id)
    publisher = GitHubPublisher(REPO_ROOT, config.github, logger)
    preflight = publisher.preflight()
    _, worktree, state_path = _paths(config, run_id)
    state = AuditState.load(state_path)
    if state.data.get("config_fingerprint") != stable_fingerprint(config):
        raise RuntimeError("resume config does not match the original quality_loop run")
    if state.data.get("repo_slug") != preflight.repo_slug:
        raise RuntimeError("resume repository does not match the original quality_loop run")
    if (
        resolve_worktree(REPO_ROOT, str(state.data.get("worktree"))) != worktree
        or not worktree.is_dir()
    ):
        raise RuntimeError(f"resume worktree is missing or changed: {worktree}")
    return run_id


def finalize(config: QualityLoopConfig, run_id: str, logger: logging.Logger) -> str:
    validate_run_id(run_id)
    publisher = GitHubPublisher(REPO_ROOT, config.github, logger)
    preflight = publisher.preflight()
    artifact, worktree, state_path = _paths(config, run_id)
    state = AuditState.load(state_path)
    if state.data.get("status") != "awaiting_publication":
        raise RuntimeError(
            f"quality_loop run is not ready for publication: {state.data.get('status')}"
        )
    if state.data.get("config_fingerprint") != stable_fingerprint(config):
        raise RuntimeError("publication config does not match the original quality_loop run")
    if state.data.get("repo_slug") != preflight.repo_slug:
        raise RuntimeError("publication repository does not match the original quality_loop run")
    if (
        resolve_worktree(REPO_ROOT, str(state.data.get("worktree"))) != worktree
        or not worktree.is_dir()
    ):
        raise RuntimeError(f"publication worktree is missing or changed: {worktree}")

    expected_paths = {
        f"tasks/{task_id}/{relative}"
        for task_id, record in state.data.get("tasks", {}).items()
        if record.get("state") == "completed" and record.get("commit_pending")
        for relative in record.get("changes", [])
    }
    publisher.verify_pending_changes(
        worktree=worktree,
        branch=str(state.data["branch"]),
        base_sha=str(state.data["base_sha"]),
        expected_paths=expected_paths,
    )

    for task_id, record in state.data.get("tasks", {}).items():
        if record.get("state") != "completed" or not record.get("commit_pending"):
            continue
        commit = publisher.commit_task(worktree, task_id)
        if not commit:
            raise RuntimeError(f"accepted changes disappeared before commit: {task_id}")
        record["commit"] = commit
        record["commit_pending"] = False
        state.save()

    workflow = QualityLoop(REPO_ROOT, config, logger=logger, publisher=publisher)
    workflow.state = state
    workflow.artifact_dir = artifact
    workflow.worktree = worktree
    workflow.preflight = preflight
    report_path = workflow._write_report()
    pr_url = None
    if config.github.publish:
        pr_url = publisher.publish_draft_pr(
            worktree=worktree,
            repo_slug=str(state.data["repo_slug"]),
            branch=str(state.data["branch"]),
            base_branch=str(state.data["base_branch"]),
            title="audit(tasks): quality_loop task quality pass",
            body=workflow._pull_request_body(report_path),
            artifact_dir=artifact,
        )
    state.finish("completed", pull_request_url=pr_url)
    return pr_url or "no pull request (no accepted changes or publication disabled)"


def main(argv: list[str] | None = None) -> int:
    args, unknown = _parser().parse_known_args(argv)
    if unknown:
        raise ValueError(f"unknown quality_loop host arguments: {unknown}")
    config = _effective_config(args)
    run_id = args.run_id or args.resume
    if args.action == "start":
        run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        print(start(config, run_id, _logger()))
    elif args.action == "check":
        if not run_id:
            raise ValueError("check requires --resume or --run-id")
        print(check(config, run_id, _logger()))
    elif args.action == "paths":
        if not run_id:
            raise ValueError("paths requires --resume or --run-id")
        validate_run_id(run_id)
        artifact, worktree, _ = _paths(config, run_id)
        print(artifact.relative_to(REPO_ROOT))
        print(worktree.relative_to(REPO_ROOT))
    else:
        if not run_id:
            raise ValueError("finalize requires --run-id")
        print(finalize(config, run_id, _logger()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
