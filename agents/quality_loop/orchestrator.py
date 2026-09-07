# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
from __future__ import annotations

import logging
import os
import shutil
import statistics
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
from agents.task_validator.validation_prompt import build_validation_prompt
from src.evaluator import (
    evaluate_compilation,
    evaluate_correctness,
    evaluate_kernel,
    measure_baseline,
    write_task_result,
)
from src.harness_guard import snapshot_workspace_harness, verify_workspace_harness
from src.perf_helper_materialization import materialize_perf_helpers_in_workspace
from src.preprocessing import _resolve_gfx_arch, setup_workspace
from src.prompt_builder import prompt_builder
from src.testcases import collect_benchmark_methods
from src.eval_tools.config import EvalToolsConfig
from src.eval_tools.contracts import SourceEvidence
from src.eval_tools.evidence import capture_submission_evidence


def _task_slug(task_id: str) -> str:
    return task_id.replace("/", "__")


def _source_paths(config: dict[str, Any]) -> tuple[str, ...]:
    raw = config.get("source_file_path", [])
    if isinstance(raw, str):
        raw = [raw]
    return tuple(str(path) for path in raw if str(path).strip())


def _repo_subdir(config: dict[str, Any]) -> str | None:
    if config.get("repo_subdir"):
        return str(config["repo_subdir"])
    if config.get("image_repo_path"):
        return Path(str(config["image_repo_path"])).name
    if config.get("repo_url"):
        name = str(config["repo_url"]).rstrip("/").rsplit("/", 1)[-1]
        return name[:-4] if name.endswith(".git") else name
    return None


def _filtered_changes(
    before: dict[str, str],
    after: dict[str, str],
    *,
    repo_subdir: str | None,
) -> TreeChanges:
    changes = diff_trees(before, after)
    return TreeChanges(
        added=tuple(p for p in changes.added if not is_generated_path(p, repo_subdir=repo_subdir)),
        modified=tuple(
            p for p in changes.modified if not is_generated_path(p, repo_subdir=repo_subdir)
        ),
        deleted=tuple(
            p for p in changes.deleted if not is_generated_path(p, repo_subdir=repo_subdir)
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
        and all(value > 0 for value in speedups)
        and statistics.median(speedups) >= config.easy_speedup_threshold
        and result.get("pass_compilation") is True
        and result.get("pass_correctness") is True
        and result.get("pass_tool_gate", True) is True
        and result.get("tool_policy_satisfied", True) is True
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
    ):
        self.repo_root = repo_root.resolve()
        self.config = config
        self.logger = logger
        self.backend = backend or CodexBackend(config.backend, logger)
        self.reviewer_backend = reviewer_backend or CodexBackend(config.reviewer, logger)
        self.publisher = publisher or GitHubPublisher(self.repo_root, config.github, logger)
        self.defer_github = defer_github
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
            task_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
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
        required = platform.get("required_arch")
        return not required or (gfx_arch is not None and str(required).strip() == gfx_arch)

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
            if self.state.is_terminal(task_id):
                self.logger.info("Resume: skipping terminal task %s", task_id)
                continue
            task_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            if not self._platform_matches(task_config, gfx_arch):
                self.state.transition(
                    task_id,
                    "platform_deferred",
                    reason=f"requires a different platform than {gfx_arch or 'unknown'}",
                )
                continue
            self.logger.info("quality_loop task %d/%d: %s", index, len(tasks), task_id)
            try:
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
        task_artifacts = self.artifact_dir / "tasks" / _task_slug(task_id)
        original_task = task_artifacts / "original_task"
        candidate_task = task_artifacts / "candidate_task"
        self._reset_path(task_artifacts)
        task_artifacts.mkdir(parents=True)
        shutil.copytree(canonical_task, original_task)
        shutil.copytree(canonical_task, candidate_task)
        original_tree = snapshot_tree(original_task)
        original_validation_status = "FAIL"

        self.state.transition(task_id, "validating")
        validation_workspace, validation = self._validate(
            task_id, candidate_task, task_artifacts / "validation_initial"
        )
        original_validation_status = str(validation.get("overall_status", "FAIL")).upper()
        warnings = _validation_warnings(validation)

        if original_validation_status == "FAIL":
            self.state.transition(task_id, "repairing", warnings=warnings)
            task_config = self._load_task_config(candidate_task)
            before = snapshot_tree(validation_workspace)
            self.backend.run(
                repair_prompt(validation, task_id), validation_workspace, role="repair"
            )
            after = snapshot_tree(validation_workspace)
            changes = _filtered_changes(
                before, after, repo_subdir=_repo_subdir(task_config)
            )
            if changes.empty:
                self._handle_unrepairable(task_id, validation)
                return
            apply_changes(validation_workspace, candidate_task, changes)
            restore_committed_perf_stubs(candidate_task)
            _, validation = self._validate(
                task_id, candidate_task, task_artifacts / "validation_repaired"
            )
            warnings = _validation_warnings(validation)
            if str(validation.get("overall_status", "FAIL")).upper() == "FAIL":
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
                eval_result = evaluate_kernel(
                    optimization_workspace,
                    self._load_task_config(candidate_task),
                    baseline_cases,
                    self.logger,
                )
                baseline_methods = set(collect_benchmark_methods(baseline_cases))
                optimized_methods = set(eval_result.get("optimized_benchmark_methods") or [])
                repeated_valid = bool(
                    eval_result.get("pass_compilation")
                    and eval_result.get("pass_correctness")
                    and baseline_methods
                    and baseline_methods == optimized_methods
                    and int(eval_result.get("valid_baseline_cases", 0)) > 0
                    and eval_result.get("valid_baseline_cases")
                    == eval_result.get("valid_optimized_cases")
                )
                speedups.append(
                    float(eval_result.get("average_speedup") or 0.0)
                    if repeated_valid
                    else 0.0
                )

        hardened = False
        hardening_reason = "candidate did not meet the configured easy-task gate"
        task_config = self._load_task_config(candidate_task)
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
            )

        restore_committed_perf_stubs(candidate_task)
        candidate_tree = snapshot_tree(candidate_task)
        final_changes = _filtered_changes(
            original_tree,
            candidate_tree,
            repo_subdir=_repo_subdir(task_config),
        )
        commit = None
        commit_pending = False
        if not final_changes.empty:
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
        )

    def _validate(
        self, task_id: str, task_dir: Path, stage_dir: Path
    ) -> tuple[Path, dict[str, Any]]:
        workspace = self._make_workspace(task_id, task_dir, stage_dir)
        prompt = build_validation_prompt(
            str(task_dir / "config.yaml"),
            str(workspace),
            self._eval_config(),
        )
        self.backend.run(prompt, workspace, role="validator")
        report_path = workspace / "validation_report.yaml"
        if not report_path.exists():
            report = {
                "task_name": task_id,
                "overall_status": "FAIL",
                "checks": {},
                "summary": "validator backend did not produce validation_report.yaml",
            }
            report_path.write_text(yaml.safe_dump(report), encoding="utf-8")
            return workspace, report
        report = yaml.safe_load(report_path.read_text(encoding="utf-8")) or {}
        if not isinstance(report, dict) or str(report.get("overall_status", "")).upper() not in {
            "PASS",
            "WARN",
            "FAIL",
        }:
            raise RuntimeError(f"invalid validator report for {task_id}: {report_path}")
        return workspace, report

    def _optimize_once(
        self, task_id: str, task_dir: Path, stage_dir: Path
    ) -> tuple[Path, list[Any], dict[str, Any]]:
        workspace = self._make_workspace(task_id, task_dir, stage_dir)
        task_config = self._load_task_config(task_dir)
        eval_tools_config = EvalToolsConfig.from_mapping(self._eval_config())
        submission_evidence = None
        if eval_tools_config.enabled:
            submission_evidence = capture_submission_evidence(
                workspace,
                task_config,
                stage_dir / "submission_evidence",
            )
        original_sources = stage_dir / "original_sources"
        original_sources.mkdir()
        source_manifest: dict[str, str] = {}
        for relative in _source_paths(task_config):
            source = (workspace / relative).resolve()
            if not source.is_relative_to(workspace.resolve()) or not source.is_file():
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
        task_type = str(task_config.get("task_type", ""))
        if task_type == "torch2hip":
            baseline_cases = measure_baseline(workspace, task_config, self.logger)
        else:
            compiled, error = evaluate_compilation(workspace, task_config, self.logger)
            if not compiled:
                raise RuntimeError(f"baseline compilation failed after validation: {error}")
            baseline_cases = measure_baseline(workspace, task_config, self.logger)

        harness = snapshot_workspace_harness(workspace)
        base_prompt = prompt_builder(
            str(task_dir / "config.yaml"),
            str(workspace),
            self._eval_config(),
            self.logger,
        )
        self.backend.run(
            optimizer_prompt(base_prompt, task_id), workspace, role="optimizer"
        )
        verify_workspace_harness(harness)
        if snapshot_tree(original_sources) != original_source_tree:
            raise RuntimeError("optimizer modified the protected original-source snapshot")
        materialize_perf_helpers_in_workspace(workspace, logger=self.logger)
        tool_manager = None
        tool_source_evidence = None
        if eval_tools_config.enabled:
            assert submission_evidence is not None
            submission_evidence.verify()
            tool_source_evidence = SourceEvidence(
                original_root=str(submission_evidence.files_dir),
                original_fingerprint=submission_evidence.fingerprint,
                candidate_fingerprint=submission_evidence.candidate_fingerprint(),
                metadata={
                    "manifest": str(submission_evidence.storage_dir / "manifest.json"),
                    "quality_loop_task": task_id,
                },
            )
            from src.eval_tools.factory import (
                create_default_manager,
                task_artifact_root,
            )

            tool_manager = create_default_manager()
            tool_report_root = task_artifact_root(workspace)
        else:
            tool_report_root = None
        evaluation = evaluate_kernel(
            workspace,
            task_config,
            baseline_cases,
            self.logger,
            tool_manager=tool_manager,
            eval_tools_config=eval_tools_config,
            tool_source_evidence=tool_source_evidence,
            tool_artifact_root=tool_report_root,
            gpu_arch=_resolve_gfx_arch(self.config.target_gpu_model),
        )
        write_task_result(
            workspace,
            evaluation,
            baseline_cases,
            task_id,
            "quality_loop/codex",
            self.logger,
            create_plots=False,
        )
        shutil.copytree(
            original_sources,
            workspace / ".quality_loop_original_sources",
        )
        return workspace, baseline_cases, yaml.safe_load(
            (workspace / "task_result.yaml").read_text(encoding="utf-8")
        )

    def _review(
        self, task_id: str, workspace: Path, result: dict[str, Any]
    ) -> dict[str, Any]:
        output_name = "quality_loop_review.yaml"
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
        review = (
            yaml.safe_load(review_path.read_text(encoding="utf-8"))
            if review_path.exists()
            else None
        )
        if not _review_is_valid(review):
            raise RuntimeError(f"reviewer returned an invalid decision for {task_id}")
        if not (
            result.get("pass_compilation")
            and result.get("pass_correctness")
            and result.get("benchmark_method_consistent")
        ):
            review["accepted"] = False
            review["evidence_sufficient"] = False
            review["summary"] = (
                "Deterministic evaluator gate rejected the candidate. "
                + review["summary"]
            )
        return review

    def _promote_baseline(
        self,
        task_id: str,
        original_task: Path,
        candidate_task: Path,
        optimized_workspace: Path,
        task_artifacts: Path,
    ) -> tuple[bool, str | None]:
        task_config = self._load_task_config(candidate_task)
        sources = _source_paths(task_config)
        if not sources or any(not (candidate_task / path).is_file() for path in sources):
            reason = "task has no promotable committed source baseline"
            self.logger.warning("Task %s %s", task_id, reason)
            return False, reason
        if any(not (optimized_workspace / path).is_file() for path in sources):
            reason = "optimizer omitted a declared source file"
            self.logger.warning("Task %s %s", task_id, reason)
            return False, reason
        source_backup = task_artifacts / "baseline_before_promotion"
        self._reset_path(source_backup)
        source_backup.mkdir(parents=True)
        for relative in sources:
            backup_file = source_backup / relative
            backup_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate_task / relative, backup_file)
        for relative in sources:
            source = optimized_workspace / relative
            if not source.is_file():
                return False, f"optimizer omitted declared source file {relative}"
            destination = candidate_task / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        restore_committed_perf_stubs(candidate_task)

        _, validation = self._validate(
            task_id, candidate_task, task_artifacts / "validation_hardened"
        )
        if str(validation.get("overall_status", "FAIL")).upper() == "FAIL":
            # Restore the previously accepted source files; an unverified faster
            # candidate must never become the task baseline.
            for relative in sources:
                shutil.copy2(source_backup / relative, candidate_task / relative)
            return False, "promoted baseline failed fresh task validation"
        accepted = self._dual_correctness_gate(
            task_id, original_task, candidate_task, task_artifacts / "hardening_gate"
        )
        if not accepted:
            for relative in sources:
                shutil.copy2(source_backup / relative, candidate_task / relative)
            return False, "promoted baseline failed the dual correctness gate"
        return True, None

    def _enhance_cases(
        self,
        task_id: str,
        original_task: Path,
        candidate_task: Path,
        rationale: str,
        task_artifacts: Path,
    ) -> bool:
        case_workspace = self._make_workspace(
            task_id, candidate_task, task_artifacts / "case_candidate"
        )
        task_config = self._load_task_config(candidate_task)
        before = snapshot_tree(case_workspace)
        self.backend.run(
            case_enhancement_prompt(task_id, rationale),
            case_workspace,
            role="case_enhancer",
        )
        after = snapshot_tree(case_workspace)
        changes = _filtered_changes(
            before, after, repo_subdir=_repo_subdir(task_config)
        )
        if changes.empty:
            return False
        if any(
            not is_case_path(path) or Path(path).name == "performance_utils_pytest.py"
            for path in changes.paths
        ):
            self.logger.warning("Rejecting non-case changes from case enhancer: %s", changes.paths)
            return False

        backup = task_artifacts / "candidate_before_cases"
        shutil.copytree(candidate_task, backup)
        apply_changes(case_workspace, candidate_task, changes)
        restore_committed_perf_stubs(candidate_task)
        if not self._dual_correctness_gate(
            task_id, original_task, candidate_task, task_artifacts / "case_gate"
        ):
            self._replace_directory(candidate_task, backup)
            return False
        _, validation = self._validate(
            task_id, candidate_task, task_artifacts / "validation_cases"
        )
        if str(validation.get("overall_status", "FAIL")).upper() == "FAIL":
            self._replace_directory(candidate_task, backup)
            return False
        return True

    def _dual_correctness_gate(
        self, task_id: str, original_task: Path, candidate_task: Path, stage_dir: Path
    ) -> bool:
        task_config = self._load_task_config(candidate_task)
        sources = _source_paths(task_config)
        if not sources or any(not (original_task / path).is_file() for path in sources):
            self.logger.warning(
                "Cannot prove new cases/baseline against a committed original kernel for %s",
                task_id,
            )
            return False
        original_with_cases = stage_dir / "original_with_cases"
        self._reset_path(stage_dir)
        shutil.copytree(candidate_task, original_with_cases)
        for relative in sources:
            destination = original_with_cases / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original_task / relative, destination)

        for label, task_dir in (
            ("original", original_with_cases),
            ("candidate", candidate_task),
        ):
            workspace = self._make_workspace(task_id, task_dir, stage_dir / label)
            config = self._load_task_config(task_dir)
            compiled, _ = evaluate_compilation(workspace, config, self.logger)
            correct, _ = evaluate_correctness(workspace, config, self.logger)
            if not compiled or not correct:
                self.logger.warning(
                    "Dual correctness gate rejected %s (%s): compile=%s correctness=%s",
                    task_id,
                    label,
                    compiled,
                    correct,
                )
                return False
        return True

    def _handle_unrepairable(self, task_id: str, report: dict[str, Any]) -> None:
        assert self.state is not None
        self.state.transition(
            task_id,
            "reported_failure",
            warnings=_validation_warnings(report),
            validation_report=report,
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

    def _eval_config(self) -> dict[str, Any]:
        result = {
            "target_gpu_model": self.config.target_gpu_model,
            "agent": {
                "template": "codex",
                "python_path": os.environ.get("AGENT_KERNEL_ARENA_PYTHON"),
                "compile_timeout": 600,
                "correctness_timeout": 600,
                "performance_timeout": 600,
                "max_iterations": 1,
            },
        }
        if self.config.evaluation_tools:
            result["evaluation_tools"] = dict(self.config.evaluation_tools)
        return result

    @staticmethod
    def _load_task_config(task_dir: Path) -> dict[str, Any]:
        value = yaml.safe_load((task_dir / "config.yaml").read_text(encoding="utf-8")) or {}
        if not isinstance(value, dict):
            raise ValueError(f"invalid task config: {task_dir / 'config.yaml'}")
        return value

    @staticmethod
    def _reset_path(path: Path) -> None:
        if not path.exists() and not path.is_symlink():
            return
        if path.is_symlink() or path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)

    @classmethod
    def _replace_directory(cls, destination: Path, source: Path) -> None:
        cls._reset_path(destination)
        shutil.copytree(source, destination)

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
`{report_relative}`. Every promoted baseline and case change passed the fail-closed
dual correctness gate against the pre-audit kernel.
"""
