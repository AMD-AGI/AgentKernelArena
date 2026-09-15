# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
from __future__ import annotations

import logging
import hashlib
import math
import os
import shutil
import statistics
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from agents.quality_loop.backend import AgentBackend, CodexBackend
from agents.quality_loop.config import QualityLoopConfig
from agents.quality_loop.filesystem import (
    TreeChanges,
    apply_changes,
    diff_trees,
    is_case_path,
    is_generated_path,
    restore_committed_perf_stubs,
    snapshot_tree,
)
from agents.quality_loop.github import GitHubPublisher, PreflightResult
from agents.quality_loop.prompts import (
    case_enhancement_prompt,
    optimizer_prompt,
    repair_prompt,
    reviewer_prompt,
)
from agents.quality_loop.state import (
    AuditState,
    resolve_worktree,
    stable_fingerprint,
    validate_run_id,
)
from agents.quality_loop.runtime import InitialValidationRejected, create_session
from agents.task_validator.launch_agent import launch_agent as launch_validator
from agents.task_validator.report_schema import (
    COMPLETION_MARKER_FILENAME, validation_report_is_complete,
)
from src.harness_guard import snapshot_workspace_harness, verify_workspace_harness
from src.perf_helper_materialization import materialize_perf_helpers_in_workspace
from src.preprocessing import _resolve_gfx_arch, setup_workspace
from src.prompt_builder import prompt_builder
from src.task_spec import TaskSpec, load_task_spec, required_gpu_arches, resolve_task_path


def _task_slug(task_id: str) -> str:
    return task_id.replace("/", "__") + "-" + hashlib.sha256(task_id.encode()).hexdigest()[:12]


def _source_paths(spec: TaskSpec, root: Path) -> tuple[str, ...]:
    """Expand declared edit scopes; directory names and suffixes imply no language."""
    paths = set()
    for scope in spec.candidate.editable:
        source = resolve_task_path(root, scope.path)
        if scope.scope == "tree":
            if source.exists():
                for path in source.rglob("*"):
                    relative = path.relative_to(root).as_posix()
                    resolve_task_path(root, relative)
                    if path.is_file() and not is_generated_path(relative):
                        paths.add(relative)
        else:
            paths.add(scope.path)
    return tuple(sorted(paths))


def _materialized_destinations(spec: TaskSpec) -> tuple[str, ...]:
    return tuple(item["destination"] for item in
                 spec.to_mapping().get("workspace", {}).get("sources", []))


def _filtered_changes(
    before: dict[str, str],
    after: dict[str, str],
    *,
    materialized: tuple[str, ...] = (),
) -> TreeChanges:
    changes = diff_trees(before, after)
    return TreeChanges(
        added=tuple(p for p in changes.added if not is_generated_path(p, materialized=materialized)),
        modified=tuple(
            p for p in changes.modified if not is_generated_path(p, materialized=materialized)
        ),
        deleted=tuple(
            p for p in changes.deleted if not is_generated_path(p, materialized=materialized)
        ),
    )


def _validation_warnings(report: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    for name, check in (report.get("checks") or {}).items():
        if isinstance(check, dict) and str(check.get("status", "")).upper() == "WARN":
            warnings.append(f"{name}: {check.get('details') or check.get('analysis') or 'warning'}")
    return warnings


def _review_is_valid(review: Any) -> bool:
    if not isinstance(review, dict):
        return False
    for key in (
        "accepted",
        "logic_equivalent",
        "evidence_sufficient",
        "case_enhancement_needed",
    ):
        if not isinstance(review.get(key), bool):
            return False
    return isinstance(review.get("summary"), str) and isinstance(
        review.get("case_rationale"), str
    )


def difficulty_is_easy(
    *,
    speedups: list[float],
    result: dict[str, Any],
    review: dict[str, Any],
    config: QualityLoopConfig,
) -> bool:
    """Return true only for a reproducible, review-approved first-iteration 5x gain."""
    return bool(
        len(speedups) == config.easy_confirmation_runs
        and all(math.isfinite(value) and value > 0 for value in speedups)
        and statistics.median(speedups) >= config.easy_speedup_threshold
        and result.get("pass_compilation") is True
        and result.get("pass_correctness") is True
        and result.get("pass_tool_gate", not config.evaluation_tools) is True
        and result.get("tool_policy_satisfied", not config.evaluation_tools) is True
        and result.get("benchmark_method_consistent") is True
        and int(result.get("valid_baseline_cases", 0)) > 0
        and result.get("valid_baseline_cases") == result.get("valid_optimized_cases")
        and review.get("accepted") is True
        and review.get("logic_equivalent") is True
        and review.get("evidence_sufficient") is True
    )


class QualityLoop:
    def __init__(
        self,
        repo_root: Path,
        config: QualityLoopConfig,
        *,
        logger: logging.Logger,
        backend: AgentBackend | None = None,
        reviewer_backend: AgentBackend | None = None,
        publisher: GitHubPublisher | None = None,
        defer_github: bool = False,
        validator_launcher=None,
        session_factory=None,
    ):
        self.repo_root = repo_root.resolve()
        self.config = config
        self.logger = logger
        self.backend = backend or CodexBackend(config.backend, logger)
        self.reviewer_backend = reviewer_backend or CodexBackend(config.reviewer, logger)
        self.publisher = publisher or GitHubPublisher(self.repo_root, config.github, logger)
        self.defer_github = defer_github
        self.validator_launcher = validator_launcher or launch_validator
        self.session_factory = session_factory or create_session
        self._sessions: dict[Path, Any] = {}
        self._session_task_ids: dict[Path, str] = {}
        self._validation_evidence: list[dict[str, Any]] = []
        self.state: AuditState | None = None
        self.artifact_dir: Path | None = None
        self.worktree: Path | None = None
        self.preflight: PreflightResult | None = None

    def discover_tasks(self, root: Path | None = None) -> dict[str, Path]:
        tasks_root = (root or self.repo_root) / "tasks"
        discovered = {
            str(path.parent.relative_to(tasks_root)): path
            for path in tasks_root.rglob("config.yaml")
        }
        # Do not discover acquisition copies or nested harness fixtures as tasks.
        discovered = {
            task_id: path for task_id, path in discovered.items()
            if not any((parent / "config.yaml").is_file()
                       for parent in path.parent.parents
                       if parent != tasks_root and parent.is_relative_to(tasks_root))
        }
        if "all" in self.config.tasks:
            return dict(sorted(discovered.items()))
        selected: dict[str, Path] = {}
        missing: list[str] = []
        for selector in self.config.tasks:
            matches = {
                task_id: path
                for task_id, path in discovered.items()
                if task_id == selector or task_id.startswith(selector.rstrip("/") + "/")
            }
            if not matches:
                missing.append(selector)
            selected.update(matches)
        if missing:
            raise ValueError(f"task selector(s) matched nothing: {missing}")
        return dict(sorted(selected.items()))

    def plan(self) -> dict[str, Any]:
        tasks = self.discover_tasks()
        gfx_arch = _resolve_gfx_arch(self.config.target_gpu_model)
        runnable: list[str] = []
        deferred: list[str] = []
        for task_id, config_path in tasks.items():
            task_config = load_task_spec(config_path, task_id=task_id).to_mapping()
            if self._platform_matches(task_config, gfx_arch):
                runnable.append(task_id)
            else:
                deferred.append(task_id)
        return {
            "total": len(tasks),
            "runnable": runnable,
            "platform_deferred": deferred,
            "target_gpu_model": self.config.target_gpu_model,
            "gfx_arch": gfx_arch,
            "backend": self.config.backend.name,
            "optimization_iterations": self.config.optimization_iterations,
        }

    @staticmethod
    def _platform_matches(task_config: dict[str, Any], gfx_arch: str | None) -> bool:
        platform = task_config.get("platform_support")
        if not isinstance(platform, dict):
            return True
        if str(platform.get("status", "active")).strip().lower() == "skip":
            return False
        required = required_gpu_arches(platform)
        return not required or gfx_arch in required

    def run(
        self,
        *,
        resume_run_id: str | None = None,
        skip_preflight: bool = False,
    ) -> Path:
        run_id = resume_run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        validate_run_id(run_id)
        self.artifact_dir = (self.repo_root / self.config.artifact_root / run_id).resolve()
        state_path = self.artifact_dir / "state.yaml"
        fingerprint = stable_fingerprint(self.config)

        if resume_run_id:
            self.state = AuditState.load(state_path)
            if self.state.data.get("config_fingerprint") != fingerprint:
                raise RuntimeError("resume config does not match the original quality_loop run")
            self.worktree = resolve_worktree(
                self.repo_root, str(self.state.data["worktree"])
            )
            if not self.worktree.is_dir():
                raise RuntimeError(f"resume worktree is missing: {self.worktree}")
            if skip_preflight:
                self.preflight = PreflightResult(
                    repo_slug=str(self.state.data["repo_slug"]),
                    default_branch=str(self.state.data["base_branch"]),
                    base_sha=str(self.state.data["base_sha"]),
                    viewer_permission="WRITE",
                )
            else:
                self.preflight = self.publisher.preflight()
        else:
            if skip_preflight:
                raise ValueError("skip_preflight is only valid for a host-initialized resume run")
            self.preflight = self.publisher.preflight()
            branch = f"{self.config.github.branch_prefix}/{run_id}"
            self.worktree = (self.repo_root / self.config.worktree_root / run_id).resolve()
            self.publisher.create_worktree(
                path=self.worktree,
                branch=branch,
                base_branch=self.preflight.default_branch,
            )
            self.state = AuditState.create(
                state_path,
                run_id=run_id,
                config_fingerprint=fingerprint,
                repo_slug=self.preflight.repo_slug,
                base_sha=self.preflight.base_sha,
                base_branch=self.preflight.default_branch,
                branch=branch,
                worktree=self.worktree.relative_to(self.repo_root),
            )

        assert self.state is not None and self.worktree is not None
        tasks = self.discover_tasks(self.worktree)
        gfx_arch = _resolve_gfx_arch(self.config.target_gpu_model)
        for index, (task_id, config_path) in enumerate(tasks.items(), 1):
            if self.state.is_terminal(task_id) and self._terminal_evidence_current(task_id, config_path.parent):
                self.logger.info("Resume: skipping terminal task %s", task_id)
                continue
            self.logger.info("quality_loop task %d/%d: %s", index, len(tasks), task_id)
            try:
                task_config = load_task_spec(config_path, task_id=task_id).to_mapping()
                if not self._platform_matches(task_config, gfx_arch):
                    self.state.transition(
                        task_id, "platform_deferred",
                        reason=f"requires a different platform than {gfx_arch or 'unknown'}",
                        accepted_fingerprint=stable_fingerprint(snapshot_tree(config_path.parent)),
                    )
                    continue
                self._process_task(task_id, config_path.parent)
            except Exception as exc:
                self.logger.exception("quality_loop task failed: %s", task_id)
                # Tooling/agent/runtime failures are not evidence that the task is
                # defective. Keep them resumable without publishing task changes.
                self.state.transition(task_id, "infrastructure_failed", error=str(exc))

        report_path = self._write_report()
        infrastructure_failures = [
            task_id
            for task_id, record in self.state.data.get("tasks", {}).items()
            if record.get("state") == "infrastructure_failed"
        ]
        if infrastructure_failures:
            self.state.finish("incomplete")
            raise RuntimeError(
                "quality_loop cannot publish until infrastructure failures are resumed: "
                + ", ".join(infrastructure_failures)
            )
        if self.defer_github:
            self.state.finish("awaiting_publication")
            return report_path
        pr_url = None
        if self.config.github.publish and not self.defer_github:
            pr_url = self.publisher.publish_draft_pr(
                worktree=self.worktree,
                repo_slug=str(self.state.data["repo_slug"]),
                branch=str(self.state.data["branch"]),
                base_branch=str(self.state.data["base_branch"]),
                title="audit(tasks): quality_loop task quality pass",
                body=self._pull_request_body(report_path),
                artifact_dir=self.artifact_dir,
            )
        self.state.finish("completed", pull_request_url=pr_url)
        return report_path

    def _process_task(self, task_id: str, canonical_task: Path) -> None:
        assert self.state is not None
        assert self.artifact_dir is not None
        assert self.worktree is not None
        load_task_spec(canonical_task / "config.yaml", task_id=task_id)
        task_artifacts = self.artifact_dir / "tasks" / _task_slug(task_id) / uuid.uuid4().hex
        original_task = task_artifacts / "original_task"
        candidate_task = task_artifacts / "candidate_task"
        task_artifacts.mkdir(parents=True)
        self._validation_evidence = []
        self._copy_task(canonical_task, original_task)
        self._copy_task(canonical_task, candidate_task)
        original_tree = snapshot_tree(original_task)
        original_validation_status = "FAIL"

        self.state.transition(task_id, "validating",
                              attempt_artifacts=str(task_artifacts.relative_to(self.repo_root)),
                              accepted_fingerprint=stable_fingerprint(snapshot_tree(canonical_task)))
        validation_workspace, validation = self._validate(
            task_id, candidate_task, task_artifacts / "validation_initial"
        )
        original_validation_status = str(validation.get("overall_status", "FAIL")).upper()
        warnings = _validation_warnings(validation)

        if original_validation_status == "FAIL":
            self.state.transition(task_id, "repairing", warnings=warnings)
            spec = load_task_spec(candidate_task / "config.yaml", task_id=task_id)
            before = snapshot_tree(validation_workspace)
            self.backend.run(
                repair_prompt(validation, task_id), validation_workspace, role="repair"
            )
            after = snapshot_tree(validation_workspace)
            changes = _filtered_changes(
                before, after, materialized=_materialized_destinations(spec)
            )
            if changes.empty:
                self._handle_unrepairable(task_id, validation)
                return
            apply_changes(validation_workspace, candidate_task, changes)
            load_task_spec(candidate_task / "config.yaml", task_id=task_id)
            restore_committed_perf_stubs(candidate_task)
            _, validation = self._validate(
                task_id, candidate_task, task_artifacts / "validation_repaired"
            )
            warnings = _validation_warnings(validation)
            if validation.get("overall_status") != "PASS":
                self._handle_unrepairable(task_id, validation)
                return

        self.state.transition(task_id, "optimizing", warnings=warnings)
        optimization_workspace, baseline_cases, result = self._optimize_once(
            task_id, candidate_task, task_artifacts / "optimization"
        )
        review = self._review(task_id, optimization_workspace, result)
        speedups = [float(result.get("speedup_ratio") or 0.0)]
        if speedups[0] >= self.config.easy_speedup_threshold and review.get("accepted"):
            for _ in range(1, self.config.easy_confirmation_runs):
                eval_result = self._evaluate_session(optimization_workspace)
                repeated_valid = bool(
                    eval_result.get("pass_compilation") is True
                    and eval_result.get("pass_correctness") is True
                    and eval_result.get("benchmark_method_consistent") is True
                    and eval_result.get("pass_tool_gate", not self.config.evaluation_tools) is True
                    and eval_result.get("tool_policy_satisfied", not self.config.evaluation_tools) is True
                    and int(eval_result.get("valid_baseline_cases", 0)) > 0
                    and eval_result.get("valid_baseline_cases")
                    == eval_result.get("valid_optimized_cases")
                )
                speedups.append(
                    float(eval_result.get("speedup_ratio") or 0.0)
                    if repeated_valid
                    else 0.0
                )

        hardened = False
        hardening_reason = "candidate did not meet the configured easy-task gate"
        spec = load_task_spec(candidate_task / "config.yaml", task_id=task_id)
        if difficulty_is_easy(
            speedups=speedups,
            result=result,
            review=review,
            config=self.config,
        ):
            hardened, hardening_reason = self._promote_baseline(
                task_id,
                original_task,
                candidate_task,
                optimization_workspace,
                task_artifacts,
            )

        cases_enhanced = False
        if (
            self.config.case_enhancement
            and review.get("accepted") is True
            and review.get("case_enhancement_needed") is True
            and original_validation_status in {"PASS", "WARN"}
        ):
            cases_enhanced = self._enhance_cases(
                task_id,
                original_task,
                candidate_task,
                str(review.get("case_rationale", "")),
                task_artifacts,
                optimized_workspace=optimization_workspace,
            )

        restore_committed_perf_stubs(candidate_task)
        candidate_tree = snapshot_tree(candidate_task)
        final_changes = _filtered_changes(
            original_tree,
            candidate_tree,
            materialized=_materialized_destinations(spec),
        )
        commit = None
        commit_pending = False
        if not final_changes.empty:
            _, final_validation = self._validate(
                task_id, candidate_task, task_artifacts / "validation_final"
            )
            if final_validation.get("overall_status") != "PASS":
                self._handle_unrepairable(task_id, final_validation)
                return
            apply_changes(candidate_task, canonical_task, final_changes)
            restore_committed_perf_stubs(canonical_task)
            if self.defer_github:
                # A linked worktree's .git file contains a host-absolute path,
                # which is intentionally unavailable inside the GPU container.
                # The credential-bearing host finalizer verifies and commits it.
                commit_pending = True
            else:
                commit = self.publisher.commit_task(self.worktree, task_id)
        self.state.transition(
            task_id,
            "completed",
            warnings=warnings,
            changes=list(final_changes.paths),
            commit=commit,
            commit_pending=commit_pending,
            speedups=speedups,
            reviewer=review,
            baseline_hardened=hardened,
            baseline_hardening_reason=hardening_reason,
            cases_enhanced=cases_enhanced,
            validation_evidence=list(self._validation_evidence),
            accepted_fingerprint=stable_fingerprint(snapshot_tree(canonical_task)),
        )

    def _validate(
        self, task_id: str, task_dir: Path, stage_dir: Path
    ) -> tuple[Path, dict[str, Any]]:
        spec = load_task_spec(task_dir / "config.yaml", task_id=task_id)
        # Preserve the stable discovery ID even for a repaired scratch package.
        package = stage_dir / "definitions" / "tasks" / task_id
        package.parent.mkdir(parents=True, exist_ok=True)
        self._copy_task(task_dir, package)
        workspace = self._make_workspace(task_id, package, stage_dir / "execution")
        report_path = workspace / "validation_report.yaml"
        marker_path = workspace / COMPLETION_MARKER_FILENAME
        if report_path.exists() or marker_path.exists():
            raise RuntimeError("fresh validator workspace contains pre-existing evidence")
        before = snapshot_tree(workspace)
        started = time.time_ns()
        settings = self._eval_config(task_id=task_id, validator=True)
        from src.task_session import TaskSession
        from src.task_run import validate_task_session

        session = TaskSession.create(spec, workspace, stage_dir / "validation-session", self.logger)
        # The shared runner passes the captured manifest, initial-state and
        # baseline evidence before the semantic reviewer starts.
        def launch(*, eval_config, task_config_dir, workspace):
            return self.validator_launcher(eval_config, task_config_dir, workspace)
        validate_task_session(session, eval_config=settings, task_config_dir=str(package / "config.yaml"),
                              agent_launcher=launch)
        if not validation_report_is_complete(workspace):
            raise RuntimeError(f"validator did not finalize a complete report for {task_id}")
        if report_path.is_symlink() or marker_path.is_symlink():
            raise RuntimeError("validator evidence must not be a symlink")
        report = yaml.safe_load(report_path.read_text(encoding="utf-8"))
        if report.get("task_name") != task_id:
            raise RuntimeError("validator report names a different task")
        timestamp = datetime.fromisoformat(str(report.get("validation_timestamp", "")))
        if timestamp.tzinfo is None:
            raise RuntimeError("validator report timestamp must identify its timezone")
        if (report_path.stat().st_mtime_ns < started
                or timestamp.timestamp() < started / 1e9 - 1
                or timestamp.timestamp() > time.time() + 5):
            raise RuntimeError("validator returned stale or future-dated evidence")
        changes = _filtered_changes(before, snapshot_tree(workspace),
                                    materialized=_materialized_destinations(spec))
        if not changes.empty:
            raise RuntimeError(f"validator modified task files: {changes.paths}")
        if report.get("framework_status") != "PASS":
            raise RuntimeError("validator framework failed; this is not a repairable task judgment")
        self._validation_evidence.append({
            "workspace": str(workspace.relative_to(self.repo_root)),
            "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
            "task_fingerprint": stable_fingerprint(snapshot_tree(task_dir)),
            "spec": spec.to_mapping(),
        })
        return workspace, report

    def _new_session(self, task_id: str, workspace: Path, spec: TaskSpec, stage_dir: Path):
        session = self.session_factory(
            spec=spec, workspace=workspace, artifact_dir=stage_dir,
            eval_config=self._eval_config(task_id=task_id), logger=self.logger,
        )
        self._session_task_ids[workspace] = task_id
        return session

    def _evaluate_session(self, workspace: Path) -> dict[str, Any]:
        session = self._sessions[workspace]
        result_path = workspace / "task_result.yaml"
        # Prior measurements remain reviewable; none can be mistaken for this call.
        self._reset_path(result_path)
        result = session.evaluate_candidate()
        if not isinstance(result, dict) or not result_path.is_file() or result_path.is_symlink():
            raise RuntimeError("shared TaskSession did not produce a fresh task_result.yaml")
        if yaml.safe_load(result_path.read_text()) != result:
            raise RuntimeError("TaskSession result disagrees with its finalized report")
        if result.get("task_name") != self._session_task_ids.get(workspace):
            raise RuntimeError("TaskSession reported a different task identity")
        return result

    def _optimize_once(
        self, task_id: str, task_dir: Path, stage_dir: Path
    ) -> tuple[Path, list[Any], dict[str, Any]]:
        spec = load_task_spec(task_dir / "config.yaml", task_id=task_id)
        workspace = self._make_workspace(task_id, task_dir, stage_dir)
        session = self._new_session(task_id, workspace, spec, stage_dir / "evaluation")
        # The framework owns initial-state validation and the independent baseline.
        # Empty candidates and cross-language tasks follow their declarations.
        session.prepare()
        self._sessions[workspace] = session
        original_sources = stage_dir / "original_sources"
        original_sources.mkdir()
        source_manifest = {}
        for relative in _source_paths(spec, workspace):
            source = resolve_task_path(workspace, relative)
            if not source.is_file():
                source_manifest[relative] = "missing"
                continue
            destination = original_sources / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            source_manifest[relative] = "copied"
        (original_sources / "manifest.yaml").write_text(
            yaml.safe_dump(source_manifest, sort_keys=True), encoding="utf-8"
        )
        original_source_tree = snapshot_tree(original_sources)
        harness = snapshot_workspace_harness(workspace, task_root=task_dir)
        base_prompt = prompt_builder(
            str(task_dir / "config.yaml"), str(workspace),
            self._eval_config(task_id=task_id), self.logger,
        )
        base_prompt += (
            f"\n\nRead-only framework task context: `{session.agent_context_path}`. "
            "Use its frozen baseline workspace for baseline checks; never derive "
            "a new baseline from your modified candidate workspace."
        )
        self.backend.run(optimizer_prompt(base_prompt, task_id, spec), workspace, role="optimizer")
        verify_workspace_harness(harness, logger=self.logger)
        if snapshot_tree(original_sources) != original_source_tree:
            raise RuntimeError("optimizer modified the protected original-source snapshot")
        materialize_perf_helpers_in_workspace(workspace, logger=self.logger)
        result = self._evaluate_session(workspace)
        if result.get("task_name") != task_id:
            raise RuntimeError("TaskSession reported a different task identity")
        verify_workspace_harness(harness, logger=self.logger)
        shutil.copytree(original_sources, workspace / ".quality_loop_original_sources")
        return workspace, session.baseline_cases, result

    def _review(
        self, task_id: str, workspace: Path, result: dict[str, Any]
    ) -> dict[str, Any]:
        output_name = "quality_loop_review.yaml"
        if (workspace / output_name).exists() or (workspace / output_name).is_symlink():
            raise RuntimeError("review workspace contains an old reviewer decision")
        result_path = workspace / "task_result.yaml"
        if (not result_path.is_file() or result_path.is_symlink()
                or yaml.safe_load(result_path.read_text()) != result):
            raise RuntimeError("review requires the current framework evaluation evidence")
        before = snapshot_tree(workspace)
        evidence_names = (
            "task_result.yaml",
            "baseline_perf.yaml",
            "optimized_perf.yaml",
        )
        evidence_before = {
            name: (workspace / name).read_bytes()
            for name in evidence_names
            if (workspace / name).is_file()
        }
        original_sources = workspace / ".quality_loop_original_sources"
        original_before = snapshot_tree(original_sources)
        self.reviewer_backend.run(
            reviewer_prompt(task_id, workspace / "task_result.yaml", output_name),
            workspace,
            role="reviewer",
        )
        after = snapshot_tree(workspace)
        evidence_after = {
            name: (workspace / name).read_bytes()
            for name in evidence_names
            if (workspace / name).is_file()
        }
        if (
            evidence_after != evidence_before
            or snapshot_tree(original_sources) != original_before
        ):
            raise RuntimeError("reviewer modified protected evaluation evidence")
        changes = diff_trees(before, after)
        unexpected = [path for path in changes.paths if path != output_name]
        if unexpected:
            raise RuntimeError(f"reviewer modified non-review files: {unexpected}")
        review_path = workspace / output_name
        if review_path.is_symlink():
            raise RuntimeError("reviewer decision must not be a symlink")
        review = (
            yaml.safe_load(review_path.read_text(encoding="utf-8"))
            if review_path.exists()
            else None
        )
        if not _review_is_valid(review):
            raise RuntimeError(f"reviewer returned an invalid decision for {task_id}")
        if not (
            result.get("pass_compilation") is True
            and result.get("pass_correctness") is True
            and result.get("benchmark_method_consistent") is True
            and result.get("pass_tool_gate", not self.config.evaluation_tools) is True
            and result.get("tool_policy_satisfied", not self.config.evaluation_tools) is True
        ):
            review["accepted"] = False
            review["evidence_sufficient"] = False
            review["summary"] = (
                "Deterministic evaluator gate rejected the candidate. "
                + review["summary"]
            )
        return review

    def _promote_baseline(
        self, task_id: str, original_task: Path, candidate_task: Path,
        optimized_workspace: Path, task_artifacts: Path,
    ) -> tuple[bool, str | None]:
        spec = load_task_spec(candidate_task / "config.yaml", task_id=task_id)
        if spec.baseline.kind != "initial_candidate":
            return False, "provided baseline is independent; copying a candidate does not replace it"
        if spec.candidate.initial_state != "implemented":
            return False, "there is no implemented starting candidate to promote"
        sources = _source_paths(spec, candidate_task)
        destinations = _materialized_destinations(spec)
        if not sources or any(
            not resolve_task_path(candidate_task, path).is_file()
            or any(path == dest or path.startswith(dest + "/") for dest in destinations)
            for path in sources
        ):
            return False, "task has no promotable committed source baseline"
        backup = task_artifacts / "baseline_before_promotion"
        self._copy_task(candidate_task, backup)
        accepted = False
        try:
            self._install_candidate(spec, optimized_workspace, candidate_task)
            promoted = spec.to_mapping()
            promoted["candidate"]["initial_language"] = spec.candidate.language
            promoted["baseline"]["language"] = spec.candidate.language
            promoted = TaskSpec.from_mapping(promoted, task_id=task_id).to_mapping()
            (candidate_task / "config.yaml").write_text(yaml.safe_dump(promoted, sort_keys=False))
            restore_committed_perf_stubs(candidate_task)
            if not self._dual_correctness_gate(
                task_id, original_task, candidate_task, task_artifacts / "hardening_gate"
            ):
                return False, "promoted baseline failed the dual correctness gate"
            _, validation = self._validate(
                task_id, candidate_task, task_artifacts / "validation_hardened"
            )
            if validation.get("overall_status") != "PASS":
                return False, "promoted baseline needs a clean fresh task validation PASS"
            accepted = True
            return True, None
        finally:
            if not accepted:
                self._replace_directory(candidate_task, backup)

    def _enhance_cases(
        self, task_id: str, original_task: Path, candidate_task: Path,
        rationale: str, task_artifacts: Path, *, optimized_workspace: Path,
    ) -> bool:
        case_workspace = self._make_workspace(
            task_id, candidate_task, task_artifacts / "case_candidate"
        )
        spec = load_task_spec(candidate_task / "config.yaml", task_id=task_id)
        before = snapshot_tree(case_workspace)
        self.backend.run(case_enhancement_prompt(task_id, rationale), case_workspace,
                         role="case_enhancer")
        changes = _filtered_changes(before, snapshot_tree(case_workspace),
                                    materialized=_materialized_destinations(spec))
        if changes.empty:
            return False
        workloads = spec.to_mapping()["evaluation"].get("workloads")
        if any(
            (not is_case_path(path) and path != workloads)
            or any(scope.contains(path) for scope in spec.candidate.editable)
            or Path(path).name == "performance_utils_pytest.py"
            for path in changes.paths
        ):
            self.logger.warning("Rejecting non-case changes from case enhancer: %s", changes.paths)
            return False
        backup = task_artifacts / "candidate_before_cases"
        self._copy_task(candidate_task, backup)
        accepted = False
        try:
            apply_changes(case_workspace, candidate_task, changes)
            restore_committed_perf_stubs(candidate_task)
            if not self._dual_correctness_gate(
                task_id, original_task, candidate_task, task_artifacts / "case_gate",
                optimized_workspace=optimized_workspace,
            ):
                return False
            _, validation = self._validate(
                task_id, candidate_task, task_artifacts / "validation_cases"
            )
            accepted = validation.get("overall_status") == "PASS"
            return accepted
        finally:
            if not accepted:
                self._replace_directory(candidate_task, backup)

    @staticmethod
    def _install_candidate(spec: TaskSpec, source: Path, destination: Path) -> None:
        """Install all declared files/helpers, preserving nested paths and removals."""
        paths = set(_source_paths(spec, source)) | set(_source_paths(spec, destination))
        for relative in sorted(paths):
            src = resolve_task_path(source, relative)
            dst = resolve_task_path(destination, relative)
            if src.is_file():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            elif any(scope.scope == "tree" and scope.contains(relative)
                     for scope in spec.candidate.editable):
                QualityLoop._reset_path(dst)
            else:
                raise RuntimeError(f"candidate omitted declared file {relative}")

    def _dual_correctness_gate(
        self, task_id: str, original_task: Path, candidate_task: Path, stage_dir: Path,
        *, optimized_workspace: Path | None = None,
    ) -> bool:
        original_spec = load_task_spec(original_task / "config.yaml", task_id=task_id)
        proposed_spec = load_task_spec(candidate_task / "config.yaml", task_id=task_id)
        original_with_cases = stage_dir / "original_with_cases"
        original_with_cases.parent.mkdir(parents=True, exist_ok=True)
        self._copy_task(candidate_task, original_with_cases)
        # Retain proposed cases, but prepare the ORIGINAL role/state/language.
        # Preparation must never snapshot the optimized candidate as its baseline.
        config = proposed_spec.to_mapping()
        config["candidate"] = original_spec.to_mapping()["candidate"]
        config["baseline"] = original_spec.to_mapping()["baseline"]
        check_spec = TaskSpec.from_mapping(config, task_id=task_id)
        (original_with_cases / "config.yaml").write_text(yaml.safe_dump(check_spec.to_mapping()))
        self._restore_initial_candidate(original_spec, original_task, original_with_cases)
        workspace = self._make_workspace(task_id, original_with_cases, stage_dir / "execution")
        session = self._new_session(task_id, workspace, check_spec, stage_dir / "evaluation")
        try:
            session.prepare()
        except InitialValidationRejected:
            self.logger.warning("Original baseline did not pass proposed cases for %s", task_id)
            return False
        # A generation task's empty original is never called as a candidate.
        # The actual optimized implementation is checked against the new cases.
        self._install_candidate(proposed_spec, optimized_workspace or candidate_task, workspace)
        return session.check_candidate() is True

    @staticmethod
    def _restore_initial_candidate(spec: TaskSpec, source: Path, destination: Path) -> None:
        for relative in set(_source_paths(spec, source)) | set(_source_paths(spec, destination)):
            src = resolve_task_path(source, relative)
            dst = resolve_task_path(destination, relative)
            if src.is_file():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            elif dst.exists() or dst.is_symlink():
                QualityLoop._reset_path(dst)

    def _handle_unrepairable(self, task_id: str, report: dict[str, Any]) -> None:
        assert self.state is not None
        self.state.transition(
            task_id,
            "reported_failure",
            warnings=_validation_warnings(report),
            validation_report=report,
            validation_evidence=list(self._validation_evidence),
        )

    def _make_workspace(self, task_id: str, task_dir: Path, stage_dir: Path) -> Path:
        self._reset_path(stage_dir)
        stage_dir.mkdir(parents=True, exist_ok=True)
        return setup_workspace(
            str(task_dir / "config.yaml"),
            stage_dir,
            "qualityloop",
            self.logger,
            task_name=task_id,
        )

    def _eval_config(self, *, task_id: str | None = None, validator: bool = False) -> dict[str, Any]:
        backend = self.config.validator if validator else self.config.backend
        result = {
            "target_gpu_model": self.config.target_gpu_model,
            "agent": {
                "template": "task_validator" if validator else "codex",
                "backend": backend.name,
                "model": backend.model,
                "effort": backend.effort,
                "timeout_seconds": backend.timeout_seconds,
                "python_path": os.environ.get("AGENT_KERNEL_ARENA_PYTHON"),
                "max_iterations": 1,
            },
        }
        if task_id is not None:
            result["task_id"] = task_id
            result["_task_id"] = task_id  # Shared prompt builder identity for scratch packages.
        if self.config.evaluation_tools:
            result["evaluation_tools"] = dict(self.config.evaluation_tools)
        return result

    def _terminal_evidence_current(self, task_id: str, canonical_task: Path) -> bool:
        assert self.state is not None
        record = self.state.task(task_id)
        if record.get("accepted_fingerprint") != stable_fingerprint(snapshot_tree(canonical_task)):
            return False
        if record.get("state") == "platform_deferred":
            return True
        evidence = record.get("validation_evidence")
        if not isinstance(evidence, list) or not evidence:
            return False
        try:
            for item in evidence:
                relative = Path(item["workspace"])
                if relative.is_absolute() or ".." in relative.parts:
                    return False
                workspace = (self.repo_root / relative).resolve(strict=True)
                if not workspace.is_relative_to(self.repo_root):
                    return False
                path = workspace / "validation_report.yaml"
                if path.is_symlink() or (workspace / COMPLETION_MARKER_FILENAME).is_symlink():
                    return False
                if not validation_report_is_complete(workspace):
                    return False
                if hashlib.sha256(path.read_bytes()).hexdigest() != item.get("report_sha256"):
                    return False
                report = yaml.safe_load(path.read_text())
                if report.get("task_name") != task_id or report.get("framework_status") != "PASS":
                    return False
        except (KeyError, TypeError, ValueError, OSError, yaml.YAMLError):
            return False
        return True

    @staticmethod
    def _copy_task(source: Path, destination: Path) -> None:
        # Copy declarations/sources, never prior evaluation results. Leave the
        # original experiment artifacts exactly where their owner stored them.
        def ignore(directory, names):
            return [name for name in names if is_generated_path(name)]
        shutil.copytree(source, destination, symlinks=True, ignore=ignore)

    @staticmethod
    def _reset_path(path: Path) -> None:
        """Archive a previous attempt rather than deleting user-owned evidence."""
        if not path.exists() and not path.is_symlink():
            return
        history = path.parent / ".quality_loop_history"
        history.mkdir(exist_ok=True)
        path.rename(history / (path.name + "-" + uuid.uuid4().hex))

    @classmethod
    def _replace_directory(cls, destination: Path, source: Path) -> None:
        cls._reset_path(destination)
        cls._copy_task(source, destination)

    def _write_report(self) -> Path:
        assert self.state is not None and self.artifact_dir is not None
        records = self.state.data.get("tasks", {})
        counts: dict[str, int] = {}
        warning_count = 0
        for record in records.values():
            status = str(record.get("state", "unknown"))
            counts[status] = counts.get(status, 0) + 1
            warning_count += len(record.get("warnings") or [])
        report = {
            "run_id": self.state.data["run_id"],
            "repo": self.state.data["repo_slug"],
            "base_sha": self.state.data["base_sha"],
            "target_gpu_model": self.config.target_gpu_model,
            "backend": self.config.backend.name,
            "optimization_iterations": 1,
            "counts": counts,
            "warning_count": warning_count,
            "tasks": records,
        }
        path = self.artifact_dir / "audit_report.yaml"
        path.write_text(
            yaml.safe_dump(report, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        return path

    def _pull_request_body(self, report_path: Path) -> str:
        assert self.state is not None
        records = self.state.data.get("tasks", {})
        completed = [task for task, value in records.items() if value.get("state") == "completed"]
        changed = [task for task in completed if records[task].get("changes")]
        unresolved = [
            task for task, value in records.items() if value.get("state") == "reported_failure"
        ]
        warnings = sum(len(value.get("warnings") or []) for value in records.values())
        report_relative = report_path.relative_to(self.repo_root)
        return f"""## Summary

- Audited tasks: {len(records)}
- Accepted task changes: {len(changed)}
- Validator warnings recorded: {warnings}
- Unresolved validation failures: {len(unresolved)}
- Optimizer: Codex, exactly one iteration per task
- Easy-task threshold: reproducible {self.config.easy_speedup_threshold:.1f}x

## Changed tasks

{chr(10).join(f'- `{task}`' for task in changed) or '- None'}

## Unresolved validation failures

{chr(10).join(f'- `{task}`' for task in unresolved) or '- None'}

The full machine-readable report is stored in the local run artifact
`{report_relative}`. Every accepted task change passed a fresh framework-finalized
task validation. Promotions and case changes also checked the original declared
baseline and the actual optimized candidate through the shared task lifecycle.
"""
