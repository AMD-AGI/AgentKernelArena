# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Audit, repair, optimize and harden schema-v3 kernel task packages.

A task package is ``tasks/<id>/task.yaml`` with its workloads, reference,
vendored baseline, optional inputs.py and README.md (docs/how-to/add-task.md).
Every task judgment is a framework-finalized QA audit (agents/qa_auditor); every
accepted change to a package needs a fresh audit PASS of the changed package.
"""
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

from agents.qa_auditor.report import (
    COMPLETION_MARKER_FILENAME, REPORT_FILENAME, load_report, report_is_current,
)
from agents.qa_auditor.rubric import WARNING
from agents.quality_loop.backend import AgentBackend, CodexBackend
from agents.quality_loop.config import QualityLoopConfig
from agents.quality_loop.filesystem import (
    TreeChanges,
    apply_changes,
    diff_trees,
    is_generated_path,
    snapshot_tree,
)
from agents.quality_loop.github import GitHubPublisher, PreflightResult
from agents.quality_loop.prompts import (
    case_enhancement_prompt,
    optimizer_prompt,
    repair_prompt,
    repair_review_prompt,
    reviewer_prompt,
)
from agents.quality_loop.state import (
    AuditState,
    resolve_worktree,
    stable_fingerprint,
    validate_run_id,
)
from agents.quality_loop.runtime import InitialValidationRejected, create_session
from agents.quality_loop.repair_policy import RepairPolicy, RepairRejected, blocking_findings
from src.harness_guard import snapshot_workspace_harness, verify_workspace_harness
from src.preprocessing import _resolve_gfx_arch, setup_workspace
from src.prompt_builder import prompt_builder
from src.task_contract import (
    BASELINE_DIR, SOLUTION_DIR, TASK_FILE, audit_target_language, load_task_contract,
)
from src.task_spec import TaskConfigError, TaskSpec, load_task_spec, required_gpu_arches, resolve_task_path

# Case enhancement may extend the workloads and their generator, nothing else.
CASE_FILES = ("workloads.json", "inputs.py")


def _launch_auditor(eval_config, task_config_dir, workspace):
    from agents.qa_auditor.launch_agent import launch_agent

    return launch_agent(eval_config, task_config_dir, workspace)


def _task_slug(task_id: str) -> str:
    return task_id.replace("/", "__") + "-" + hashlib.sha256(task_id.encode()).hexdigest()[:12]


def _solution_paths(root: Path) -> tuple[str, ...]:
    """The candidate tree's files; directory names and suffixes imply no language."""
    paths = set()
    source = resolve_task_path(root, SOLUTION_DIR)
    if source.exists():
        for path in source.rglob("*"):
            relative = path.relative_to(root).as_posix()
            resolve_task_path(root, relative)
            if path.is_file() and not is_generated_path(path.relative_to(source).as_posix(), root=source):
                paths.add(relative)
    return tuple(sorted(paths))


def _package_digest(root: Path) -> str:
    """The materialization record's task_package_sha256 for this directory."""
    from src.task_materialization import _STALE_OUTPUTS, _Deadline, _digest, _tree_manifest

    return _digest(_tree_manifest(Path(root), _Deadline(3600), excludes=tuple(_STALE_OUTPUTS)))


def _filtered_changes(
    before: dict[str, str],
    after: dict[str, str],
    *,
    source_root: Path | None = None,
) -> TreeChanges:
    """Task-package changes between two snapshots; framework/runtime files excluded."""
    changes = diff_trees(before, after)

    def keep(path: str) -> bool:
        return not is_generated_path(path, root=source_root)

    return TreeChanges(
        added=tuple(p for p in changes.added if keep(p)),
        modified=tuple(p for p in changes.modified if keep(p)),
        deleted=tuple(p for p in changes.deleted if keep(p)),
    )


def _audit_warnings(report: dict[str, Any]) -> list[str]:
    return [f"{finding.get('id')}: {finding.get('detail') or finding.get('title') or 'warning'}"
            for finding in report.get("findings", [])
            if isinstance(finding, dict) and finding.get("severity") == WARNING]


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
        and result.get("pass_verification", True) is True
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
        auditor_launcher=None,
        session_factory=None,
    ):
        self.repo_root = repo_root.resolve()
        self.config = config
        self.logger = logger
        self.backend = backend or CodexBackend(config.backend, logger)
        self.reviewer_backend = reviewer_backend or CodexBackend(config.reviewer, logger)
        self.publisher = publisher or GitHubPublisher(self.repo_root, config.github, logger)
        self.defer_github = defer_github
        self.auditor_launcher = auditor_launcher or _launch_auditor
        self.session_factory = session_factory or create_session
        self._sessions: dict[Path, Any] = {}
        self._session_task_ids: dict[Path, str] = {}
        self._audit_evidence: list[dict[str, Any]] = []
        self.state: AuditState | None = None
        self.artifact_dir: Path | None = None
        self.worktree: Path | None = None
        self.preflight: PreflightResult | None = None

    def discover_tasks(self, root: Path | None = None) -> dict[str, Path]:
        """Schema-v3 packages by stable ID; nested task.yaml files are not tasks."""
        tasks_root = (root or self.repo_root) / "tasks"
        discovered = {
            str(path.parent.relative_to(tasks_root)): path
            for path in tasks_root.rglob(TASK_FILE)
        }
        discovered = {
            task_id: path for task_id, path in discovered.items()
            if not any((parent / TASK_FILE).is_file()
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
            raise ValueError(f"task selector(s) matched no schema-v3 task: {missing}")
        return dict(sorted(selected.items()))

    def plan(self) -> dict[str, Any]:
        tasks = self.discover_tasks()
        gfx_arch = _resolve_gfx_arch(self.config.target_gpu_model)
        runnable: list[str] = []
        deferred: list[str] = []
        for task_id, task_file in tasks.items():
            contract = load_task_contract(task_file, task_id=task_id)
            if self._platform_matches(contract.raw, gfx_arch):
                runnable.append(task_id)
            else:
                deferred.append(task_id)
        return {
            "total": len(tasks),
            "runnable": runnable,
            "platform_deferred": deferred,
            "target_gpu_model": self.config.target_gpu_model,
            "gfx_arch": gfx_arch,
            "target_language": self.config.target_language or "audit language",
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
        for index, (task_id, task_file) in enumerate(tasks.items(), 1):
            if self.state.is_terminal(task_id) and self._terminal_evidence_current(task_id, task_file.parent):
                self.logger.info("Resume: skipping terminal task %s", task_id)
                continue
            self.logger.info("quality_loop task %d/%d: %s", index, len(tasks), task_id)
            try:
                contract = load_task_contract(task_file, task_id=task_id)
                if not self._platform_matches(contract.raw, gfx_arch):
                    self.state.transition(
                        task_id, "platform_deferred",
                        reason=f"requires a different platform than {gfx_arch or 'unknown'}",
                        accepted_fingerprint=stable_fingerprint(snapshot_tree(task_file.parent)),
                    )
                    continue
                self._process_task(task_id, task_file.parent)
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

    def _optimization_language(self, task_id: str, task_dir: Path) -> str:
        return self.config.target_language or audit_target_language(task_dir / TASK_FILE, task_id=task_id)

    def _process_task(self, task_id: str, canonical_task: Path) -> None:
        assert self.state is not None
        assert self.artifact_dir is not None
        assert self.worktree is not None
        load_task_contract(canonical_task / TASK_FILE, task_id=task_id)
        task_artifacts = self.artifact_dir / "tasks" / _task_slug(task_id) / uuid.uuid4().hex
        original_task = task_artifacts / "original_task"
        candidate_task = task_artifacts / "candidate_task"
        task_artifacts.mkdir(parents=True)
        self._audit_evidence = []
        self._copy_task(canonical_task, original_task)
        self._copy_task(canonical_task, candidate_task)
        original_tree = snapshot_tree(original_task)

        self.state.transition(task_id, "auditing",
                              attempt_artifacts=str(task_artifacts.relative_to(self.repo_root)),
                              accepted_fingerprint=stable_fingerprint(snapshot_tree(canonical_task)))
        audit_workspace, report = self._audit(task_id, candidate_task, task_artifacts / "audit_initial")
        original_status = str(report.get("status", "FAIL")).upper()
        warnings = _audit_warnings(report)

        repair_review = None
        if original_status == "FAIL" or blocking_findings(report, self.config.repair_warnings):
            self.state.transition(task_id, "repairing", warnings=warnings)
            try:
                repaired = self._repair_task(
                    task_id, original_task, candidate_task, task_artifacts, audit_workspace, report,
                )
            except RepairRejected as exc:
                self.state.transition(task_id, "repair_rejected", repair_rejection=str(exc))
                self._handle_unrepairable(task_id, report)
                return
            if repaired is None:
                return
            report, repair_review = repaired
            warnings = _audit_warnings(report)

        if self.config.mode == "repair_only":
            changes = _filtered_changes(original_tree, snapshot_tree(candidate_task),
                                        source_root=candidate_task)
            if not changes.empty:
                if report.get("status") != "PASS" or repair_review is None:
                    self._handle_unrepairable(task_id, report)
                    return
                apply_changes(candidate_task, canonical_task, changes)
            commit = (self.publisher.commit_task(self.worktree, task_id)
                      if not changes.empty and not self.defer_github else None)
            self.state.transition(
                task_id, "completed", mode="repair_only", warnings=warnings,
                changes=list(changes.paths), commit=commit,
                commit_pending=not changes.empty and self.defer_github,
                speedups=[], reviewer=repair_review, baseline_hardened=False,
                cases_enhanced=False, audit_evidence=list(self._audit_evidence),
                accepted_fingerprint=stable_fingerprint(snapshot_tree(canonical_task)),
            )
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
                    and eval_result.get("pass_verification", True) is True
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
            and original_status == "PASS"
        ):
            cases_enhanced = self._enhance_cases(
                task_id,
                original_task,
                candidate_task,
                str(review.get("case_rationale", "")),
                task_artifacts,
                optimized_workspace=optimization_workspace,
            )

        final_changes = _filtered_changes(original_tree, snapshot_tree(candidate_task),
                                          source_root=candidate_task)
        commit = None
        commit_pending = False
        if not final_changes.empty:
            _, final_report = self._audit(
                task_id, candidate_task, task_artifacts / "audit_final"
            )
            if final_report.get("status") != "PASS":
                self._handle_unrepairable(task_id, final_report)
                return
            apply_changes(candidate_task, canonical_task, final_changes)
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
            target_language=self._optimization_language(task_id, original_task),
            reviewer=review,
            baseline_hardened=hardened,
            baseline_hardening_reason=hardening_reason,
            cases_enhanced=cases_enhanced,
            audit_evidence=list(self._audit_evidence),
            accepted_fingerprint=stable_fingerprint(snapshot_tree(canonical_task)),
        )

    def _audit(
        self, task_id: str, task_dir: Path, stage_dir: Path
    ) -> tuple[Path, dict[str, Any]]:
        """One fresh QA audit of ``task_dir``; operational failures raise.

        The returned report is the framework-finalized qa_report.yaml for this
        exact package, this request and the current rubric. Its status (PASS or
        FAIL) is the task judgment; warnings never block.
        """
        # Preserve the stable discovery ID even for a repaired scratch package.
        package = stage_dir / "definitions" / "tasks" / task_id
        package.parent.mkdir(parents=True, exist_ok=True)
        self._copy_task(task_dir, package)
        task_file = package / TASK_FILE
        language = audit_target_language(task_file, task_id=task_id)
        spec = load_task_spec(task_file, task_id=task_id, target_language=language)
        workspace = self._make_workspace(task_id, package, stage_dir / "execution",
                                         target_language=language)
        report_path = workspace / REPORT_FILENAME
        marker_path = workspace / COMPLETION_MARKER_FILENAME
        if os.path.lexists(report_path) or os.path.lexists(marker_path):
            raise RuntimeError("fresh audit workspace contains pre-existing QA evidence")
        before, package_before = snapshot_tree(workspace), snapshot_tree(package)
        started = time.time()
        settings = self._eval_config(task_id=task_id, auditor=True)
        from src.task_session import TaskSession
        from src.task_run import audit_task_session

        session = TaskSession.create(spec, workspace, stage_dir / "audit-session", self.logger)
        request_id = uuid.uuid4().hex

        def launch(*, eval_config, task_config_dir, workspace):
            return self.auditor_launcher(eval_config, task_config_dir, workspace)

        audit_task_session(session, eval_config=settings, task_config_dir=str(task_file),
                           agent_launcher=launch, audit_request_id=request_id)
        report = load_report(workspace)
        if report is None:
            raise RuntimeError(f"QA auditor did not finalize a complete report for {task_id}")
        if report.get("task_id") != task_id or report.get("audit_request_id") != request_id:
            raise RuntimeError("QA report belongs to another task or audit request")
        try:
            generated = datetime.fromisoformat(str(report.get("generated_at", "")))
        except ValueError as exc:
            raise RuntimeError("QA report has no valid generation time") from exc
        if (generated.tzinfo is None or generated.timestamp() < started - 1
                or generated.timestamp() > time.time() + 5):
            raise RuntimeError("QA auditor returned stale or future-dated evidence")
        if snapshot_tree(package) != package_before or not report_is_current(report, _package_digest(package)):
            raise RuntimeError("QA report does not qualify this exact task package and rubric")
        changes = _filtered_changes(before, snapshot_tree(workspace), source_root=workspace)
        if not changes.empty:
            raise RuntimeError(f"QA audit modified task files: {changes.paths}")
        framework = [f for f in report.get("findings", []) if f.get("id") == "QA-AUDIT"]
        if framework:
            # The audit itself failed; that is not a repairable task judgment.
            raise RuntimeError("QA audit did not run to completion: "
                               + "; ".join(str(f.get("detail"))[:500] for f in framework))
        self._audit_evidence.append({
            "workspace": str(workspace.relative_to(self.repo_root)),
            "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
            "task_package_sha256": report["task_package_sha256"],
            "task_fingerprint": stable_fingerprint(snapshot_tree(task_dir)),
            "target_language": language,
            "status": report["status"],
        })
        return workspace, report

    def _repair_task(self, task_id, original_task, candidate_task, artifacts, workspace, report):
        policy = RepairPolicy.capture(candidate_task, task_id)
        retained = blocking_findings(report, self.config.repair_warnings)
        attempted = {stable_fingerprint(snapshot_tree(candidate_task))}
        language = audit_target_language(candidate_task / TASK_FILE, task_id=task_id)
        for attempt in range(1, self.config.max_repair_attempts + 1):
            stage = artifacts / f"repair_{attempt:02d}"
            editor = self._make_workspace(task_id, candidate_task, stage / "editor",
                                          target_language=language)
            before = snapshot_tree(editor)
            self.backend.run(repair_prompt(report, task_id, findings=retained,
                                           warning_checks=self.config.repair_warnings),
                             editor, role="repair")
            changes = _filtered_changes(before, snapshot_tree(editor), source_root=editor)
            if changes.empty:
                break
            proposal = stage / "proposal"
            self._copy_task(candidate_task, proposal)
            apply_changes(editor, proposal, changes)
            policy.verify(proposal)
            fingerprint = stable_fingerprint(snapshot_tree(proposal))
            if fingerprint in attempted:
                self.logger.warning("Stopping repeated repair proposal for %s", task_id)
                break
            attempted.add(fingerprint)
            audited, next_report = self._audit(task_id, proposal, stage / "audit")
            retained.update(blocking_findings(next_report, self.config.repair_warnings))
            report = next_report
            if report.get("status") == "PASS" and not blocking_findings(report, self.config.repair_warnings):
                decision = self._review_repair(task_id, original_task, audited, retained)
                if decision.get("accepted") is True:
                    self._replace_directory(candidate_task, proposal)
                    return report, decision
                self.state.transition(task_id, "repair_review_rejected", repair_review=decision)
                break
            # Keep each proposal/evidence tree; the next editor starts from this
            # changed package, never from an unchanged audit rerun.
            self._replace_directory(candidate_task, proposal)
        self._handle_unrepairable(task_id, report)
        return None

    def _review_repair(self, task_id, original, workspace, findings):
        output_name = "quality_loop_repair_review.yaml"
        output = workspace / output_name
        if output.exists() or output.is_symlink():
            raise RuntimeError("Repair review requires a fresh decision")
        before, original_before = snapshot_tree(workspace), snapshot_tree(original)
        self.reviewer_backend.run(repair_review_prompt(task_id, original, findings, output_name),
                                  workspace, role="repair_reviewer")
        changes = diff_trees(before, snapshot_tree(workspace))
        if (snapshot_tree(original) != original_before
                or set(changes.paths) != {output_name} or changes.deleted
                or not output.is_file() or output.is_symlink()):
            raise RuntimeError("Repair reviewer changed protected files or omitted its decision")
        result = yaml.safe_load(output.read_text())
        if (not isinstance(result, dict) or not isinstance(result.get("summary"), str)
                or any(type(result.get(k)) is not bool for k in
                       ("accepted", "contract_preserved", "scoring_preserved"))
                or not isinstance(result.get("resolved_findings"), list)
                or any(not isinstance(k, str) for k in result["resolved_findings"])
                or len(set(result["resolved_findings"])) != len(result["resolved_findings"])):
            raise RuntimeError("Invalid independent repair review")
        if (result["contract_preserved"] is not True or result["scoring_preserved"] is not True
                or set(result["resolved_findings"]) != set(findings)):
            result["accepted"] = False
        return result

    def _new_session(self, task_id: str, workspace: Path, spec: TaskSpec, stage_dir: Path):
        session = self.session_factory(
            spec=spec, workspace=workspace, artifact_dir=stage_dir,
            eval_config=self._eval_config(task_id=task_id, language=spec.target_language),
            logger=self.logger,
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
        language = self._optimization_language(task_id, task_dir)
        spec = load_task_spec(task_dir / TASK_FILE, task_id=task_id, target_language=language)
        workspace = self._make_workspace(task_id, task_dir, stage_dir, target_language=language)
        session = self._new_session(task_id, workspace, spec, stage_dir / "evaluation")
        # The framework owns initial-state validation and the independent baseline.
        # An unimplemented template is legal only before the optimizer runs.
        session.prepare()
        self._sessions[workspace] = session
        original_sources = stage_dir / "original_sources"
        original_sources.mkdir()
        for relative in _solution_paths(workspace):
            destination = original_sources / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(resolve_task_path(workspace, relative), destination)
        original_source_tree = snapshot_tree(original_sources)
        harness = snapshot_workspace_harness(workspace, task_root=task_dir, task_spec=spec)
        settings = self._eval_config(task_id=task_id, language=language)
        base_prompt = prompt_builder(str(task_dir / TASK_FILE), str(workspace), settings, self.logger)
        base_prompt += (
            f"\n\nRead-only framework task context: `{session.agent_context_path}`. "
            "Use its frozen baseline workspace for baseline checks; never derive "
            "a new baseline from your modified candidate workspace."
        )
        self.backend.run(optimizer_prompt(base_prompt, task_id, spec), workspace, role="optimizer")
        verify_workspace_harness(harness, logger=self.logger)
        if snapshot_tree(original_sources) != original_source_tree:
            raise RuntimeError("optimizer modified the protected original-source snapshot")
        result = self._evaluate_session(workspace)
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
        evidence_before = result_path.read_bytes()
        original_sources = workspace / ".quality_loop_original_sources"
        original_before = snapshot_tree(original_sources)
        session = self._sessions.get(workspace)
        if session is None:
            raise RuntimeError("Review requires protected evaluation evidence from the controller session")
        review_evidence = session.review_evidence(result)
        self.reviewer_backend.run(
            reviewer_prompt(task_id, workspace / "task_result.yaml", output_name,
                            evidence_path=review_evidence.path, evidence_sha256=review_evidence.sha256),
            workspace,
            role="reviewer",
        )
        session.verify_review_evidence(review_evidence)
        after = snapshot_tree(workspace)
        if (
            not result_path.is_file() or result_path.read_bytes() != evidence_before
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
            and result.get("pass_verification", True) is True
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
        """Replace the vendored baseline with the reviewed easy-task candidate.

        Only a same-language candidate is promoted: it started as a copy of the
        baseline, so it is a derivative of the vendored code and keeps its
        upstream license. Provenance records the promotion. A fresh QA PASS of
        the promoted package and the dual correctness gate are both required.
        """
        session = self._sessions.get(optimized_workspace)
        spec = session.session.spec if session is not None else None
        if spec is None or spec.contract is None:
            return False, "promotion requires the controller's schema-v3 optimization session"
        if spec.candidate.initial_state != "implemented":
            return False, (f"a {spec.candidate.language} candidate cannot replace the "
                           f"{spec.contract['baseline_language']} baseline; only same-language "
                           "candidates are promoted")
        sources = _solution_paths(optimized_workspace)
        entry = spec.candidate.entrypoints[0].file
        if entry not in sources:
            return False, "optimized candidate has no entry file to promote"
        backup = task_artifacts / "baseline_before_promotion"
        self._copy_task(candidate_task, backup)
        accepted = False
        try:
            baseline = candidate_task / BASELINE_DIR
            for path in sorted(baseline.rglob("*"), reverse=True):
                if path.is_file() and not path.name.startswith("LICENSE") and path.suffix != ".md":
                    path.unlink()
                elif path.is_dir() and not any(path.iterdir()):
                    path.rmdir()
            digest = hashlib.sha256()
            for relative in sources:
                data = resolve_task_path(optimized_workspace, relative, must_exist=True).read_bytes()
                digest.update(relative.encode() + b"\0" + data)
                target = baseline / Path(relative).relative_to(SOLUTION_DIR)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            raw = yaml.safe_load((candidate_task / TASK_FILE).read_text(encoding="utf-8"))
            previous = raw["baseline"]["provenance"]
            run_id = self.state.data["run_id"] if self.state is not None else "unknown"
            raw["baseline"]["provenance"] = {
                **previous,
                "source": f"AgentKernelArena quality_loop run {run_id}",
                "revision": digest.hexdigest(),
                "modifications": (
                    f"Reviewed {spec.candidate.language} candidate promoted by quality_loop run {run_id} "
                    f"(median {self.config.easy_speedup_threshold:g}x or more over the previous baseline). "
                    f"It replaces the baseline vendored from {previous['source']} at "
                    f"{previous['revision']}, from which it was derived; the upstream license applies. "
                    f"Earlier modifications: {previous.get('modifications', 'none')}"),
            }
            (candidate_task / TASK_FILE).write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
            try:
                load_task_contract(candidate_task / TASK_FILE, task_id=task_id)
            except (TaskConfigError, ValueError) as exc:
                return False, f"promoted package is not a valid task: {exc}"
            if not self._dual_correctness_gate(
                task_id, original_task, candidate_task, task_artifacts / "hardening_gate",
                optimized_workspace=optimized_workspace,
            ):
                return False, "promoted baseline failed the dual correctness gate"
            _, report = self._audit(task_id, candidate_task, task_artifacts / "audit_hardened")
            if report.get("status") != "PASS":
                return False, "promoted baseline needs a fresh QA audit PASS"
            accepted = True
            return True, None
        finally:
            if not accepted:
                self._replace_directory(candidate_task, backup)

    def _enhance_cases(
        self, task_id: str, original_task: Path, candidate_task: Path,
        rationale: str, task_artifacts: Path, *, optimized_workspace: Path,
    ) -> bool:
        language = audit_target_language(candidate_task / TASK_FILE, task_id=task_id)
        case_workspace = self._make_workspace(
            task_id, candidate_task, task_artifacts / "case_candidate", target_language=language
        )
        before = snapshot_tree(case_workspace)
        self.backend.run(case_enhancement_prompt(task_id, rationale), case_workspace,
                         role="case_enhancer")
        changes = _filtered_changes(before, snapshot_tree(case_workspace), source_root=case_workspace)
        if changes.empty:
            return False
        if any(path not in CASE_FILES for path in changes.paths) or changes.deleted:
            self.logger.warning("Rejecting non-case changes from case enhancer: %s", changes.paths)
            return False
        proposal = task_artifacts / "case_proposal"
        self._copy_task(candidate_task, proposal)
        apply_changes(case_workspace, proposal, changes)
        try:
            # Cases may only be added, and only as correctness checks.
            RepairPolicy.capture(candidate_task, task_id).verify(proposal)
        except RepairRejected as exc:
            self.logger.warning("Rejecting case enhancement for %s: %s", task_id, exc)
            return False
        backup = task_artifacts / "candidate_before_cases"
        self._copy_task(candidate_task, backup)
        accepted = False
        try:
            self._replace_directory(candidate_task, proposal)
            if not self._dual_correctness_gate(
                task_id, original_task, candidate_task, task_artifacts / "case_gate",
                optimized_workspace=optimized_workspace,
            ):
                return False
            _, report = self._audit(task_id, candidate_task, task_artifacts / "audit_cases")
            accepted = report.get("status") == "PASS"
            return accepted
        finally:
            if not accepted:
                self._replace_directory(candidate_task, backup)

    @staticmethod
    def _install_candidate(source: Path, destination: Path) -> None:
        """Install the complete solution/ tree, preserving nested helpers and removals."""
        wanted = set(_solution_paths(source))
        for relative in _solution_paths(destination):
            if relative not in wanted:
                resolve_task_path(destination, relative).unlink()
        for relative in sorted(wanted):
            src = resolve_task_path(source, relative, must_exist=True)
            dst = resolve_task_path(destination, relative)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    def _dual_correctness_gate(
        self, task_id: str, original_task: Path, proposed_task: Path, stage_dir: Path,
        *, optimized_workspace: Path,
    ) -> bool:
        """The original baseline and the optimized candidate both pass the proposed cases.

        The proposed package keeps its workloads and inputs but takes the
        ORIGINAL baseline and declaration, so preparation never treats an
        optimized candidate as its own baseline. The workspace is materialized
        for the optimized candidate's language, whose solution/ is then
        checked with compile and full correctness.
        """
        session = self._sessions.get(optimized_workspace)
        language = (session.session.spec.target_language if session is not None
                    else self._optimization_language(task_id, original_task))
        original_with_cases = stage_dir / "original_with_cases"
        original_with_cases.parent.mkdir(parents=True, exist_ok=True)
        self._copy_task(proposed_task, original_with_cases, exclude_root=(TASK_FILE, BASELINE_DIR))
        shutil.copy2(original_task / TASK_FILE, original_with_cases / TASK_FILE)
        self._copy_task(original_task / BASELINE_DIR, original_with_cases / BASELINE_DIR)
        check_spec = load_task_spec(original_with_cases / TASK_FILE, task_id=task_id, target_language=language)
        workspace = self._make_workspace(task_id, original_with_cases, stage_dir / "execution",
                                         target_language=language)
        evaluation = self._new_session(task_id, workspace, check_spec, stage_dir / "evaluation")
        try:
            evaluation.prepare()
        except InitialValidationRejected:
            self.logger.warning("Original baseline did not pass proposed cases for %s", task_id)
            return False
        # An unimplemented template is never called as a candidate. The actual
        # optimized implementation is checked against the new cases.
        self._install_candidate(optimized_workspace, workspace)
        return evaluation.check_candidate() is True

    def _handle_unrepairable(self, task_id: str, report: dict[str, Any]) -> None:
        assert self.state is not None
        self.state.transition(
            task_id,
            "reported_failure",
            warnings=_audit_warnings(report),
            qa_report=report,
            audit_evidence=list(self._audit_evidence),
        )

    def _make_workspace(self, task_id: str, task_dir: Path, stage_dir: Path, *,
                        target_language: str) -> Path:
        self._reset_path(stage_dir)
        stage_dir.mkdir(parents=True, exist_ok=True)
        return setup_workspace(
            str(task_dir / TASK_FILE),
            stage_dir,
            "qualityloop",
            self.logger,
            task_name=task_id,
            target_language=target_language,
        )

    def _eval_config(self, *, task_id: str | None = None, auditor: bool = False,
                     language: str | None = None) -> dict[str, Any]:
        backend = self.config.auditor if auditor else self.config.backend
        agent = {
            "template": "qa_auditor" if auditor else "codex",
            "backend": backend.name,
            "model": backend.model,
            "effort": backend.effort,
            "timeout_seconds": backend.timeout_seconds,
        }
        if not auditor:
            agent.update(python_path=os.environ.get("AGENT_KERNEL_ARENA_PYTHON"), max_iterations=1)
        result = {"target_gpu_model": self.config.target_gpu_model, "agent": agent}
        if language is not None:
            result["target_language"] = language
        if task_id is not None:
            result["task_id"] = task_id
            result["_task_id"] = task_id  # Shared prompt builder identity for scratch packages.
        if self.config.evaluation_tools and not auditor:
            result["evaluation_tools"] = dict(self.config.evaluation_tools)
        return result

    def _terminal_evidence_current(self, task_id: str, canonical_task: Path) -> bool:
        assert self.state is not None
        record = self.state.task(task_id)
        if record.get("accepted_fingerprint") != stable_fingerprint(snapshot_tree(canonical_task)):
            return False
        if record.get("state") == "platform_deferred":
            return True
        evidence = record.get("audit_evidence")
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
                report = load_report(workspace)
                if report is None:
                    return False
                path = workspace / REPORT_FILENAME
                if hashlib.sha256(path.read_bytes()).hexdigest() != item.get("report_sha256"):
                    return False
                if (report.get("task_id") != task_id or report.get("status") != item.get("status")
                        or not report_is_current(report, item.get("task_package_sha256"))
                        or any(f.get("id") == "QA-AUDIT" for f in report.get("findings", []))):
                    return False
        except (KeyError, TypeError, ValueError, OSError, AttributeError):
            return False
        return True

    @staticmethod
    def _copy_task(source: Path, destination: Path, *, exclude_root: tuple[str, ...] = ()) -> None:
        # Copy the task package, never prior evaluation results or the
        # framework-written workspace files. Leave the original experiment
        # artifacts exactly where their owner stored them.
        root = Path(source)

        def ignore(directory, names):
            base = Path(directory).relative_to(root)
            return [name for name in names if is_generated_path((base / name).as_posix())
                    or (base == Path(".") and name in exclude_root)]
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
            "target_language": self.config.target_language,
            "backend": self.config.backend.name,
            "optimization_iterations": self.config.optimization_iterations,
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
        optimizer = ("none (repair_only)" if self.config.mode == "repair_only"
                     else f"Codex, exactly one iteration per task in "
                          f"{self.config.target_language or 'each task audit language'}")
        return f"""## Summary

- Audited tasks: {len(records)}
- Accepted task changes: {len(changed)}
- QA warnings recorded: {warnings}
- Unresolved QA failures: {len(unresolved)}
- Optimizer: {optimizer}
- Easy-task threshold: reproducible {self.config.easy_speedup_threshold:.1f}x

## Changed tasks

{chr(10).join(f'- `{task}`' for task in changed) or '- None'}

## Unresolved QA failures

{chr(10).join(f'- `{task}`' for task in unresolved) or '- None'}

The full machine-readable report is stored in the local run artifact
`{report_relative}`. Every accepted task change passed a fresh framework-finalized
QA audit (`qa_report.yaml`, status PASS) of the changed package. Promotions and
case changes also checked the original baseline and the actual optimized candidate
through the shared task lifecycle.
"""
