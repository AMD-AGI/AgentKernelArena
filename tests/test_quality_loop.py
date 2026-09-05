import json
import logging
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from agents.quality_loop.config import QualityLoopConfig
from agents.quality_loop.filesystem import (
    apply_changes,
    diff_trees,
    is_case_path,
    snapshot_tree,
)
from agents.quality_loop.github import CommandError, GitHubPublisher, parse_github_slug
from agents.quality_loop.orchestrator import QualityLoop, difficulty_is_easy
from agents.quality_loop.state import AuditState, resolve_worktree, validate_run_id


LOGGER = logging.getLogger("quality-loop-test")


class FakePublisher:
    def __init__(self):
        self.commits = []

    def commit_task(self, worktree, task_id):
        self.commits.append(task_id)
        return "abc123"

class RepairBackend:
    def __init__(self, change_on_repair=True):
        self.roles = []
        self.change_on_repair = change_on_repair

    def run(self, prompt, workspace, *, role):
        self.roles.append(role)
        if role == "repair" and self.change_on_repair:
            (workspace / "kernel.py").write_text("def kernel():\n    return 1\n")
        return "ok"


class TamperingReviewerBackend:
    def run(self, prompt, workspace, *, role):
        (workspace / "task_result.yaml").write_text("pass_correctness: false\n")
        (workspace / "quality_loop_review.yaml").write_text(
            yaml.safe_dump(
                {
                    "accepted": True,
                    "logic_equivalent": True,
                    "evidence_sufficient": True,
                    "case_enhancement_needed": False,
                    "case_rationale": "none",
                    "summary": "accepted",
                }
            )
        )
        return "tampered"


class StubQualityLoop(QualityLoop):
    def __init__(self, *args, reports, **kwargs):
        super().__init__(*args, **kwargs)
        self.reports = list(reports)

    def _validate(self, task_id, task_dir, stage_dir):
        stage_dir.mkdir(parents=True, exist_ok=True)
        workspace = stage_dir / "workspace"
        if workspace.exists():
            self._reset_path(workspace)
        workspace.mkdir()
        for path in task_dir.iterdir():
            if path.is_file():
                (workspace / path.name).write_bytes(path.read_bytes())
        report = self.reports.pop(0)
        (workspace / "validation_report.yaml").write_text(yaml.safe_dump(report))
        return workspace, report

    def _optimize_once(self, task_id, task_dir, stage_dir):
        stage_dir.mkdir(parents=True, exist_ok=True)
        workspace = stage_dir / "workspace"
        workspace.mkdir()
        for path in task_dir.iterdir():
            if path.is_file():
                (workspace / path.name).write_bytes(path.read_bytes())
        result = {
            "task_name": task_id,
            "pass_compilation": True,
            "pass_correctness": True,
            "speedup_ratio": 2.0,
            "benchmark_method_consistent": True,
            "valid_baseline_cases": 1,
            "valid_optimized_cases": 1,
        }
        (workspace / "task_result.yaml").write_text(yaml.safe_dump(result))
        return workspace, [], result

    def _review(self, task_id, workspace, result):
        return {
            "accepted": True,
            "logic_equivalent": True,
            "evidence_sufficient": True,
            "case_enhancement_needed": False,
            "case_rationale": "coverage is sufficient",
            "summary": "accepted",
        }


def make_task(root: Path) -> Path:
    task = root / "tasks" / "hip2hip" / "sample"
    task.mkdir(parents=True)
    (task / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "task_type": "hip2hip",
                "source_file_path": ["kernel.py"],
                "target_kernel_functions": ["kernel"],
                "compile_command": ["true"],
                "correctness_command": ["true"],
                "performance_command": ["true"],
            }
        )
    )
    (task / "kernel.py").write_text("def kernel():\n    raise RuntimeError('broken')\n")
    return task


def attach_state(workflow: QualityLoop, root: Path, worktree: Path) -> None:
    workflow.artifact_dir = root / "artifacts" / "run"
    workflow.worktree = worktree
    workflow.state = AuditState.create(
        workflow.artifact_dir / "state.yaml",
        run_id="run",
        config_fingerprint="fingerprint",
        repo_slug="AMD-AGI/AgentKernelArena",
        base_sha="base",
        base_branch="main",
        branch="quality-loop/run",
        worktree=worktree,
    )


class QualityLoopTests(unittest.TestCase):
    def test_config_defaults_to_codex_and_exactly_one_iteration(self):
        config = QualityLoopConfig.from_dict(
            {"tasks": ["all"], "target_gpu_model": "MI300", "quality_loop": {}}
        )
        self.assertEqual(config.backend.name, "codex")
        self.assertEqual(config.reviewer.name, "codex")
        self.assertEqual(config.optimization_iterations, 1)

        with self.assertRaisesRegex(ValueError, "exactly one"):
            QualityLoopConfig.from_dict(
                {
                    "tasks": ["all"],
                    "target_gpu_model": "MI300",
                    "quality_loop": {"optimization_iterations": 2},
                }
            )

        with self.assertRaisesRegex(ValueError, "always drafts"):
            QualityLoopConfig.from_dict(
                {
                    "target_gpu_model": "MI300",
                    "quality_loop": {"github": {"draft_pr": False}},
                }
            )

        with self.assertRaisesRegex(ValueError, "never creates GitHub issues"):
            QualityLoopConfig.from_dict(
                {
                    "target_gpu_model": "MI300",
                    "quality_loop": {"github": {"issue_labels": ["task-bug"]}},
                }
            )

        with self.assertRaisesRegex(ValueError, "exactly one repair"):
            QualityLoopConfig.from_dict(
                {
                    "target_gpu_model": "MI300",
                    "quality_loop": {"max_repair_attempts": 0},
                }
            )

        with self.assertRaisesRegex(ValueError, "top-level tasks field"):
            QualityLoopConfig.from_dict(
                {
                    "target_gpu_model": "MI300",
                    "quality_loop": {"promotion_task_types": ["hip2hip"]},
                }
            )

        with self.assertRaisesRegex(ValueError, "repository-relative"):
            QualityLoopConfig.from_dict(
                {
                    "target_gpu_model": "MI300",
                    "quality_loop": {"artifact_root": "../outside"},
                }
            )

    def test_run_ids_and_relative_worktree_paths_are_container_portable(self):
        with self.assertRaisesRegex(ValueError, "run ID"):
            validate_run_id("../../escape")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = root / ".quality_loop_worktrees" / "run"
            self.assertEqual(
                resolve_worktree(root, ".quality_loop_worktrees/run"),
                expected.resolve(),
            )

    def test_parse_github_slug(self):
        for remote in (
            "git@github.com:AMD-AGI/AgentKernelArena.git",
            "https://github.com/AMD-AGI/AgentKernelArena.git",
            "ssh://git@github.com/AMD-AGI/AgentKernelArena.git",
        ):
            with self.subTest(remote=remote):
                self.assertEqual(parse_github_slug(remote), "AMD-AGI/AgentKernelArena")

    def test_yaml_task_selectors_define_the_complete_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for task_id in ("custom/selected", "hip2hip/not-selected"):
                task = root / "tasks" / task_id
                task.mkdir(parents=True)
                (task / "config.yaml").write_text("task_type: custom\n")
            config = QualityLoopConfig.from_dict(
                {
                    "tasks": ["custom/selected"],
                    "target_gpu_model": "MI300",
                    "quality_loop": {},
                }
            )
            workflow = QualityLoop(root, config, logger=LOGGER)
            self.assertEqual(list(workflow.discover_tasks(root)), ["custom/selected"])

    def test_preflight_rejects_missing_write_permission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = QualityLoopConfig.from_dict(
                {"target_gpu_model": "MI300", "quality_loop": {}}
            )
            publisher = GitHubPublisher(root, config.github, LOGGER)

            def fake_run(args, **kwargs):
                if args[:3] == ["git", "status", "--porcelain"]:
                    stdout = ""
                elif args[:3] == ["git", "remote", "get-url"]:
                    stdout = "git@github.com:AMD-AGI/AgentKernelArena.git\n"
                elif args[:2] == ["gh", "api"]:
                    stdout = json.dumps(
                        {
                            "permissions": {"push": False},
                            "viewer_permission": "READ",
                            "has_issues": True,
                            "default_branch": "main",
                        }
                    )
                else:
                    stdout = ""
                return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

            with mock.patch(
                "agents.quality_loop.github.shutil.which", return_value="/bin/tool"
            ), mock.patch(
                "agents.quality_loop.github.run_command", side_effect=fake_run
            ):
                with self.assertRaisesRegex(RuntimeError, "lacks write permission"):
                    publisher.preflight()

    def test_preflight_stops_on_gh_auth_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = QualityLoopConfig.from_dict(
                {"target_gpu_model": "MI300", "quality_loop": {}}
            )
            publisher = GitHubPublisher(root, config.github, LOGGER)

            def fake_run(args, **kwargs):
                if args[:3] == ["git", "status", "--porcelain"]:
                    return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
                if args[:3] == ["git", "var", "GIT_AUTHOR_IDENT"]:
                    return subprocess.CompletedProcess(
                        args, 0, stdout="Quality Loop <test@example.invalid>\n", stderr=""
                    )
                if args[:3] == ["gh", "auth", "status"]:
                    raise CommandError("not logged in")
                raise AssertionError(f"preflight continued after auth failure: {args}")

            with mock.patch(
                "agents.quality_loop.github.shutil.which", return_value="/bin/tool"
            ), mock.patch(
                "agents.quality_loop.github.run_command", side_effect=fake_run
            ):
                with self.assertRaisesRegex(CommandError, "not logged in"):
                    publisher.preflight()

    def test_isolated_worktree_can_live_under_ignored_runtime_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            remote = parent / "remote.git"
            subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
            root = parent / "repo"
            subprocess.run(["git", "clone", str(remote), str(root)], check=True, capture_output=True)
            subprocess.run(["git", "switch", "-c", "main"], cwd=root, check=True, capture_output=True)
            subprocess.run(
                ["git", "config", "user.email", "quality-loop@example.invalid"],
                cwd=root,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "quality_loop test"], cwd=root, check=True
            )
            (root / ".gitignore").write_text(".quality_loop_worktrees/\n")
            (root / "README.md").write_text("test\n")
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            subprocess.run(["git", "commit", "-m", "base"], cwd=root, check=True, capture_output=True)
            subprocess.run(
                ["git", "push", "--set-upstream", "origin", "main"],
                cwd=root,
                check=True,
                capture_output=True,
            )
            config = QualityLoopConfig.from_dict(
                {"target_gpu_model": "MI300", "quality_loop": {}}
            )
            publisher = GitHubPublisher(root, config.github, LOGGER)
            worktree = root / ".quality_loop_worktrees" / "run"
            publisher.create_worktree(path=worktree, branch="quality-loop/run", base_branch="main")
            self.assertTrue((worktree / "README.md").is_file())
            self.assertEqual(
                subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=root,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout,
                "",
            )

    def test_tree_diff_and_apply_are_task_local(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            destination = root / "destination"
            source.mkdir()
            destination.mkdir()
            (source / "kernel.py").write_text("old\n")
            (destination / "kernel.py").write_text("old\n")
            before = snapshot_tree(source)
            (source / "kernel.py").write_text("new\n")
            (source / "test_kernel.py").write_text("case\n")
            changes = diff_trees(before, snapshot_tree(source))
            apply_changes(source, destination, changes)
            self.assertEqual((destination / "kernel.py").read_text(), "new\n")
            self.assertEqual((destination / "test_kernel.py").read_text(), "case\n")
            self.assertTrue(is_case_path("test_kernel.py"))
            self.assertFalse(is_case_path("kernel.py"))

    def test_pending_diff_verification_rejects_unrecorded_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True)
            subprocess.run(
                ["git", "config", "user.email", "quality-loop@example.invalid"],
                cwd=root,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "quality_loop test"],
                cwd=root,
                check=True,
            )
            (root / "README.md").write_text("base\n")
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            subprocess.run(
                ["git", "commit", "-m", "base"], cwd=root, check=True, capture_output=True
            )
            branch = subprocess.run(
                ["git", "branch", "--show-current"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            base_sha = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            (root / "README.md").write_text("unexpected\n")
            config = QualityLoopConfig.from_dict(
                {"target_gpu_model": "MI300", "quality_loop": {}}
            )
            publisher = GitHubPublisher(root, config.github, LOGGER)
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                publisher.verify_pending_changes(
                    worktree=root,
                    branch=branch,
                    base_sha=base_sha,
                    expected_paths=set(),
                )

    def test_publisher_always_creates_at_most_one_draft_pr(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "artifacts"
            config = QualityLoopConfig.from_dict(
                {"target_gpu_model": "MI300", "quality_loop": {}}
            )
            publisher = GitHubPublisher(root, config.github, LOGGER)
            calls = []
            existing = False

            def fake_run(args, **kwargs):
                nonlocal existing
                calls.append(list(args))
                if args[:2] == ["git", "rev-list"]:
                    stdout = "1\n"
                elif args[:3] == ["gh", "pr", "list"]:
                    stdout = (
                        json.dumps([{"url": "https://example.invalid/pr/1"}])
                        if existing
                        else "[]"
                    )
                elif args[:3] == ["gh", "pr", "create"]:
                    existing = True
                    stdout = "https://example.invalid/pr/1\n"
                else:
                    stdout = ""
                return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")

            with mock.patch(
                "agents.quality_loop.github.run_command", side_effect=fake_run
            ):
                first = publisher.publish_draft_pr(
                    worktree=root,
                    repo_slug="AMD-AGI/AgentKernelArena",
                    branch="quality-loop/run",
                    base_branch="main",
                    title="quality audit",
                    body="summary",
                    artifact_dir=artifact,
                )
                second = publisher.publish_draft_pr(
                    worktree=root,
                    repo_slug="AMD-AGI/AgentKernelArena",
                    branch="quality-loop/run",
                    base_branch="main",
                    title="quality audit",
                    body="summary",
                    artifact_dir=artifact,
                )

            create_calls = [args for args in calls if args[:3] == ["gh", "pr", "create"]]
            self.assertEqual(first, "https://example.invalid/pr/1")
            self.assertEqual(second, first)
            self.assertEqual(len(create_calls), 1)
            self.assertIn("--draft", create_calls[0])
            self.assertFalse(any(args[:2] == ["gh", "issue"] for args in calls))

    def test_publisher_skips_pr_when_run_has_no_commits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "artifacts"
            config = QualityLoopConfig.from_dict(
                {"target_gpu_model": "MI300", "quality_loop": {}}
            )
            publisher = GitHubPublisher(root, config.github, LOGGER)
            calls = []

            def fake_run(args, **kwargs):
                calls.append(list(args))
                if args[:2] == ["git", "rev-list"]:
                    return subprocess.CompletedProcess(
                        args, 0, stdout="0\n", stderr=""
                    )
                raise AssertionError(f"unexpected command after empty run: {args}")

            with mock.patch(
                "agents.quality_loop.github.run_command", side_effect=fake_run
            ):
                result = publisher.publish_draft_pr(
                    worktree=root,
                    repo_slug="AMD-AGI/AgentKernelArena",
                    branch="quality-loop/run",
                    base_branch="main",
                    title="quality audit",
                    body="summary",
                    artifact_dir=artifact,
                )

            self.assertIsNone(result)
            self.assertEqual(calls, [["git", "rev-list", "--count", "origin/main..HEAD"]])
            self.assertFalse(artifact.exists())

    def test_reviewer_cannot_modify_evaluation_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "task_result.yaml").write_text(
                yaml.safe_dump(
                    {
                        "pass_compilation": True,
                        "pass_correctness": True,
                        "benchmark_method_consistent": True,
                    }
                )
            )
            original = workspace / ".quality_loop_original_sources"
            original.mkdir()
            (original / "kernel.py").write_text("def kernel():\n    return 1\n")
            config = QualityLoopConfig.from_dict(
                {"target_gpu_model": "MI300", "quality_loop": {}}
            )
            workflow = QualityLoop(
                root,
                config,
                logger=LOGGER,
                reviewer_backend=TamperingReviewerBackend(),
            )
            with self.assertRaisesRegex(RuntimeError, "protected evaluation evidence"):
                workflow._review(
                    "hip2hip/sample",
                    workspace,
                    {
                        "pass_compilation": True,
                        "pass_correctness": True,
                        "benchmark_method_consistent": True,
                    },
                )

    def test_repair_then_revalidate_commits_only_successful_task_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            worktree = root / "worktree"
            task = make_task(worktree)
            backend = RepairBackend(change_on_repair=True)
            publisher = FakePublisher()
            config = QualityLoopConfig.from_dict(
                {
                    "tasks": ["hip2hip/sample"],
                    "target_gpu_model": "MI300",
                    "quality_loop": {},
                }
            )
            workflow = StubQualityLoop(
                root,
                config,
                logger=LOGGER,
                backend=backend,
                reviewer_backend=backend,
                publisher=publisher,
                reports=[
                    {"overall_status": "FAIL", "checks": {}, "summary": "broken"},
                    {"overall_status": "PASS", "checks": {}, "summary": "fixed"},
                ],
            )
            attach_state(workflow, root, worktree)
            workflow._process_task("hip2hip/sample", task)

            self.assertIn("return 1", (task / "kernel.py").read_text())
            self.assertEqual(workflow.state.task("hip2hip/sample")["state"], "completed")
            self.assertEqual(backend.roles, ["repair"])
            self.assertEqual(publisher.commits, ["hip2hip/sample"])

    def test_unrepairable_failure_is_reported_without_external_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            worktree = root / "worktree"
            task = make_task(worktree)
            original = (task / "kernel.py").read_text()
            backend = RepairBackend(change_on_repair=False)
            publisher = FakePublisher()
            config = QualityLoopConfig.from_dict(
                {
                    "tasks": ["hip2hip/sample"],
                    "target_gpu_model": "MI300",
                    "quality_loop": {},
                }
            )
            workflow = StubQualityLoop(
                root,
                config,
                logger=LOGGER,
                backend=backend,
                reviewer_backend=backend,
                publisher=publisher,
                reports=[{"overall_status": "FAIL", "checks": {}, "summary": "broken"}],
            )
            attach_state(workflow, root, worktree)
            workflow._process_task("hip2hip/sample", task)

            record = workflow.state.task("hip2hip/sample")
            self.assertEqual(record["state"], "reported_failure")
            self.assertEqual(record["validation_report"]["overall_status"], "FAIL")
            self.assertEqual((task / "kernel.py").read_text(), original)
            self.assertEqual(publisher.commits, [])

    def test_container_reports_unrepairable_failure_without_publication_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            worktree = root / "worktree"
            task = make_task(worktree)
            backend = RepairBackend(change_on_repair=False)
            publisher = FakePublisher()
            config = QualityLoopConfig.from_dict(
                {
                    "tasks": ["hip2hip/sample"],
                    "target_gpu_model": "MI300",
                    "quality_loop": {},
                }
            )
            workflow = StubQualityLoop(
                root,
                config,
                logger=LOGGER,
                backend=backend,
                reviewer_backend=backend,
                publisher=publisher,
                defer_github=True,
                reports=[{"overall_status": "FAIL", "checks": {}, "summary": "broken"}],
            )
            attach_state(workflow, root, worktree)
            workflow._process_task("hip2hip/sample", task)

            record = workflow.state.task("hip2hip/sample")
            self.assertEqual(record["state"], "reported_failure")
            self.assertNotIn("issue_request", record)
            self.assertNotIn("issue_url", record)

    def test_container_defers_task_commit_to_host_finalizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            worktree = root / "worktree"
            task = make_task(worktree)
            backend = RepairBackend(change_on_repair=True)
            publisher = FakePublisher()
            config = QualityLoopConfig.from_dict(
                {
                    "tasks": ["hip2hip/sample"],
                    "target_gpu_model": "MI300",
                    "quality_loop": {},
                }
            )
            workflow = StubQualityLoop(
                root,
                config,
                logger=LOGGER,
                backend=backend,
                reviewer_backend=backend,
                publisher=publisher,
                defer_github=True,
                reports=[
                    {"overall_status": "FAIL", "checks": {}, "summary": "broken"},
                    {"overall_status": "PASS", "checks": {}, "summary": "fixed"},
                ],
            )
            attach_state(workflow, root, worktree)
            workflow._process_task("hip2hip/sample", task)

            record = workflow.state.task("hip2hip/sample")
            self.assertTrue(record["commit_pending"])
            self.assertIsNone(record["commit"])
            self.assertEqual(publisher.commits, [])

    def test_warn_is_reported_without_repair(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            worktree = root / "worktree"
            task = make_task(worktree)
            backend = RepairBackend()
            publisher = FakePublisher()
            config = QualityLoopConfig.from_dict(
                {
                    "tasks": ["hip2hip/sample"],
                    "target_gpu_model": "MI300",
                    "quality_loop": {},
                }
            )
            workflow = StubQualityLoop(
                root,
                config,
                logger=LOGGER,
                backend=backend,
                reviewer_backend=backend,
                publisher=publisher,
                reports=[
                    {
                        "overall_status": "WARN",
                        "checks": {
                            "performance": {"status": "WARN", "details": "too few repeats"}
                        },
                        "summary": "warning",
                    }
                ],
            )
            attach_state(workflow, root, worktree)
            workflow._process_task("hip2hip/sample", task)

            record = workflow.state.task("hip2hip/sample")
            self.assertEqual(record["state"], "completed")
            self.assertEqual(record["warnings"], ["performance: too few repeats"])
            self.assertNotIn("repair", backend.roles)

    def test_easy_gate_is_fail_closed_and_task_type_agnostic(self):
        config = QualityLoopConfig.from_dict(
            {"target_gpu_model": "MI300", "quality_loop": {}}
        )
        result = {
            "pass_compilation": True,
            "pass_correctness": True,
            "benchmark_method_consistent": True,
            "valid_baseline_cases": 3,
            "valid_optimized_cases": 3,
        }
        review = {
            "accepted": True,
            "logic_equivalent": True,
            "evidence_sufficient": True,
        }
        self.assertTrue(
            difficulty_is_easy(
                speedups=[5.1, 5.0, 6.0],
                result=result,
                review=review,
                config=config,
            )
        )
        result["benchmark_method_consistent"] = False
        self.assertFalse(
            difficulty_is_easy(
                speedups=[6.0, 6.0, 6.0],
                result=result,
                review=review,
                config=config,
            )
        )


if __name__ == "__main__":
    unittest.main()
