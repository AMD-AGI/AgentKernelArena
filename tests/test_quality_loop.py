"""quality_loop configuration, publication and workflow control flow (CPU only)."""
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
    is_generated_path,
    snapshot_tree,
)
from agents.quality_loop.github import CommandError, GitHubPublisher, parse_github_slug
from agents.quality_loop.orchestrator import QualityLoop, difficulty_is_easy
from agents.quality_loop.state import AuditState, resolve_worktree, validate_run_id


LOGGER = logging.getLogger("quality-loop-test")

TASK_ID = "elementwise/scale/f32"
TASK_YAML = {
    "schema_version": 3,
    "description": "out = 2 * x over a vector.",
    "operator": {"name": "scale", "category": "elementwise"},
    "interface": {
        "axes": {"n": {"kind": "var", "min": 1, "max": 1048576}},
        "inputs": {"x": {"kind": "tensor", "shape": ["n"], "dtype": "float32",
                         "distribution": {"kind": "normal", "mean": 0.0, "std": 1.0}}},
        "outputs": {"out": {"shape": ["n"], "dtype": "float32"}},
    },
    "reference": {"file": "reference.py", "symbol": "reference"},
    "baseline": {"language": "triton", "entry": "baseline/kernel.py",
                 "provenance": {"source": "arena-tests", "revision": "v1",
                                "path": "tests/fixture.py", "license": "Apache-2.0"}},
    "compare": {"out": {"bound": "baseline_relative"}},
    "workloads": "workloads.json",
}
# Three distinct boundary correctness cases and four performance cases
# spanning 128x, as src.task_contract requires.
CASES = [
    {"id": f"n{n}", "axes": {"n": n},
     "checks": ["correctness", "performance"] if perf else ["correctness"], "source": "fixture"}
    for n, perf in ((1, False), (3, False), (1000, False),
                    (1024, True), (4096, True), (16384, True), (131072, True))
]


def write_package(task: Path, *, task_yaml: dict | None = None, cases: list | None = None) -> Path:
    """A minimal valid schema-v3 package (see docs/how-to/add-task.md)."""
    (task / "baseline").mkdir(parents=True, exist_ok=True)
    (task / "task.yaml").write_text(yaml.safe_dump(task_yaml or TASK_YAML, sort_keys=False))
    (task / "workloads.json").write_text(json.dumps({"cases": cases or CASES}, indent=1) + "\n")
    (task / "reference.py").write_text("def reference(*, x):\n    return x.double() * 2\n")
    (task / "baseline/kernel.py").write_text(
        "import triton\n\n\ndef run(*, x, out):\n    raise RuntimeError('fixture baseline')\n")
    (task / "README.md").write_text("Scale fixture.\n")
    return task


def make_task(root: Path, task_id: str = TASK_ID) -> Path:
    return write_package(root / "tasks" / task_id)


def qa_report(status: str, findings: list | None = None) -> dict:
    """A report-shaped mapping; real finalization is exercised in test_quality_loop_v2."""
    return {"status": status, "findings": list(findings or []), "summary": "workflow fixture"}


def finding(check_id: str, severity: str, detail: str) -> dict:
    return {"id": check_id, "severity": severity, "title": check_id, "detail": detail,
            "evidence": {}, "fix": "fixture"}


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
            (workspace / "README.md").write_text("Repaired: documents the boundary cases.\n")
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
    """Workflow fixture: audits, workspaces, optimization and review are stubbed.

    The real audit, materialization and session paths are exercised in
    test_quality_loop_v2 and test_quality_loop_review_evidence.
    """

    def __init__(self, *args, reports, **kwargs):
        super().__init__(*args, **kwargs)
        self.reports = list(reports)

    def _make_workspace(self, task_id, task_dir, stage_dir, *, target_language):
        if stage_dir.exists():
            self._reset_path(stage_dir)
        stage_dir.mkdir(parents=True)
        workspace = stage_dir / "workspace"
        self._copy_task(task_dir, workspace)
        return workspace

    def _audit(self, task_id, task_dir, stage_dir):
        workspace = self._make_workspace(task_id, task_dir, stage_dir, target_language="triton")
        report = self.reports.pop(0)
        (workspace / "qa_report.yaml").write_text(yaml.safe_dump(report))
        return workspace, report

    def _review_repair(self, task_id, original, workspace, findings):
        # Workflow-only fixture; actual reviewer tampering and closure are
        # exercised separately against the controller implementation.
        return dict(accepted=True, contract_preserved=True, scoring_preserved=True,
                    resolved_findings=list(findings), summary="CPU workflow fixture")

    def _optimize_once(self, task_id, task_dir, stage_dir):
        workspace = self._make_workspace(task_id, task_dir, stage_dir, target_language="triton")
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


def attach_state(workflow: QualityLoop, root: Path, worktree: Path) -> None:
    # As in QualityLoop.run: run paths derive from the resolved repository root.
    workflow.artifact_dir = workflow.repo_root / "artifacts" / "run"
    workflow.worktree = worktree.resolve()
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


def stub_workflow(root, backend, publisher, reports, **kwargs):
    config = QualityLoopConfig.from_dict(
        {"tasks": [TASK_ID], "target_gpu_model": "MI300", "quality_loop": {}})
    return StubQualityLoop(root, config, logger=LOGGER, backend=backend, reviewer_backend=backend,
                           publisher=publisher, reports=reports, **kwargs)


BLOCKING = [finding("QA-CONTROLS", "blocking", "the zero_output control passes correctness")]


class QualityLoopTests(unittest.TestCase):
    def test_config_defaults_to_codex_and_exactly_one_iteration(self):
        config = QualityLoopConfig.from_dict(
            {"tasks": ["all"], "target_gpu_model": "MI300", "quality_loop": {}}
        )
        self.assertEqual(config.backend.name, "codex")
        self.assertEqual(config.reviewer.name, "codex")
        self.assertEqual(config.auditor.name, "codex")
        self.assertIsNone(config.target_language)
        self.assertEqual(config.optimization_iterations, 1)

        cases = (
            ({"tasks": ["all"], "quality_loop": {"optimization_iterations": 2}}, "exactly one"),
            ({"quality_loop": {"github": {"draft_pr": False}}}, "always drafts"),
            ({"quality_loop": {"github": {"issue_labels": ["task-bug"]}}}, "never creates GitHub issues"),
            ({"quality_loop": {"max_repair_attempts": 0}}, "integer from 1 to 3"),
            ({"quality_loop": {"promotion_task_types": ["hip2hip"]}}, "top-level tasks field"),
            ({"quality_loop": {"artifact_root": "../outside"}}, "repository-relative"),
            ({"quality_loop": {"validator": {"name": "codex"}}}, "replaced by quality_loop.auditor"),
            ({"quality_loop": {"auditor": {"name": "cursor"}}}, "auditor supports only"),
            ({"quality_loop": {"target_language": "hip"}}, "top level"),
            ({"target_language": "cuda"}, "target_language must be"),
            ({"tasks": [{"path": "gemm", "target_language": "hip"}]}, "selector"),
        )
        for raw, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                QualityLoopConfig.from_dict({"target_gpu_model": "MI300", **raw})
        config = QualityLoopConfig.from_dict({"target_gpu_model": "MI300", "target_language": "gluon",
                                              "quality_loop": {"auditor": {"name": "none"}}})
        self.assertEqual((config.target_language, config.auditor.name), ("gluon", "none"))

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
            for task_id in ("custom/selected", "elementwise/not-selected"):
                make_task(root, task_id)
            # A schema-v2 package (the e2e serving tasks) is not a quality_loop task.
            legacy = root / "tasks" / "e2e" / "serving"
            legacy.mkdir(parents=True)
            (legacy / "config.yaml").write_text("schema_version: 2\n")
            config = QualityLoopConfig.from_dict(
                {
                    "tasks": ["custom/selected"],
                    "target_gpu_model": "MI300",
                    "quality_loop": {},
                }
            )
            workflow = QualityLoop(root, config, logger=LOGGER)
            self.assertEqual(list(workflow.discover_tasks(root)), ["custom/selected"])
            everything = QualityLoop(root, QualityLoopConfig.from_dict(
                {"target_gpu_model": "MI300"}), logger=LOGGER)
            self.assertEqual(list(everything.discover_tasks(root)),
                             ["custom/selected", "elementwise/not-selected"])
            with self.assertRaisesRegex(ValueError, "matched no schema-v3 task"):
                QualityLoop(root, QualityLoopConfig.from_dict(
                    {"tasks": ["e2e"], "target_gpu_model": "MI300"}), logger=LOGGER).discover_tasks(root)

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
            (source / "inputs.py").write_text("old\n")
            (destination / "inputs.py").write_text("old\n")
            before = snapshot_tree(source)
            (source / "inputs.py").write_text("new\n")
            (source / "README.md").write_text("doc\n")
            changes = diff_trees(before, snapshot_tree(source))
            apply_changes(source, destination, changes)
            self.assertEqual((destination / "inputs.py").read_text(), "new\n")
            self.assertEqual((destination / "README.md").read_text(), "doc\n")
            # Framework-written workspace files are never task-package changes.
            for generated in ("config.yaml", "solution/kernel.py", "_aka_eval/evaluate.py",
                              "qa_report.yaml", ".qa_complete", "qa_review.draft.json"):
                self.assertTrue(is_generated_path(generated), generated)
            for task_file in ("task.yaml", "workloads.json", "inputs.py", "baseline/kernel.py"):
                self.assertFalse(is_generated_path(task_file), task_file)

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
            (original / "solution").mkdir()
            (original / "solution/kernel.py").write_text("def kernel():\n    return 1\n")
            config = QualityLoopConfig.from_dict(
                {"target_gpu_model": "MI300", "quality_loop": {}}
            )
            workflow = QualityLoop(
                root,
                config,
                logger=LOGGER,
                reviewer_backend=TamperingReviewerBackend(),
            )
            # Isolate this workspace check; test_quality_loop_review_evidence
            # exercises the actual external TaskSession evidence.
            workflow._sessions[workspace] = mock.Mock()
            with self.assertRaisesRegex(RuntimeError, "protected evaluation evidence"):
                workflow._review(
                    TASK_ID,
                    workspace,
                    {
                        "pass_compilation": True,
                        "pass_correctness": True,
                        "benchmark_method_consistent": True,
                    },
                )

    def test_repair_then_reaudit_commits_only_successful_task_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            worktree = root / "worktree"
            task = make_task(worktree)
            backend = RepairBackend(change_on_repair=True)
            publisher = FakePublisher()
            workflow = stub_workflow(root, backend, publisher, [
                qa_report("FAIL", BLOCKING), qa_report("PASS"), qa_report("PASS")])
            attach_state(workflow, root, worktree)
            workflow._process_task(TASK_ID, task)

            self.assertIn("Repaired", (task / "README.md").read_text())
            record = workflow.state.task(TASK_ID)
            self.assertEqual(record["state"], "completed")
            self.assertEqual(record["changes"], ["README.md"])
            self.assertEqual(backend.roles, ["repair"])
            self.assertEqual(publisher.commits, [TASK_ID])

    def test_unrepairable_failure_is_reported_without_external_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            worktree = root / "worktree"
            task = make_task(worktree)
            original = snapshot_tree(task)
            backend = RepairBackend(change_on_repair=False)
            publisher = FakePublisher()
            workflow = stub_workflow(root, backend, publisher, [qa_report("FAIL", BLOCKING)])
            attach_state(workflow, root, worktree)
            workflow._process_task(TASK_ID, task)

            record = workflow.state.task(TASK_ID)
            self.assertEqual(record["state"], "reported_failure")
            self.assertEqual(record["qa_report"]["status"], "FAIL")
            self.assertEqual(snapshot_tree(task), original)
            self.assertEqual(publisher.commits, [])

    def test_repair_that_changes_the_scored_contract_is_rejected(self):
        class ScoringBackend:
            def run(self, prompt, workspace, *, role):
                # Still a valid package, but an original case now measures another shape.
                cases = json.loads((workspace / "workloads.json").read_text())
                cases["cases"][1]["axes"] = {"n": 5}
                (workspace / "workloads.json").write_text(json.dumps(cases))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            worktree = root / "worktree"
            task = make_task(worktree)
            original = snapshot_tree(task)
            workflow = stub_workflow(root, ScoringBackend(), FakePublisher(),
                                     [qa_report("FAIL", BLOCKING), qa_report("PASS")])
            attach_state(workflow, root, worktree)
            workflow._process_task(TASK_ID, task)
            record = workflow.state.task(TASK_ID)
            self.assertEqual(record["state"], "reported_failure")
            self.assertIn("original case", record["repair_rejection"])
            self.assertEqual(snapshot_tree(task), original)

    def test_container_reports_unrepairable_failure_without_publication_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            worktree = root / "worktree"
            task = make_task(worktree)
            workflow = stub_workflow(root, RepairBackend(change_on_repair=False), FakePublisher(),
                                     [qa_report("FAIL", BLOCKING)], defer_github=True)
            attach_state(workflow, root, worktree)
            workflow._process_task(TASK_ID, task)

            record = workflow.state.task(TASK_ID)
            self.assertEqual(record["state"], "reported_failure")
            self.assertNotIn("issue_request", record)
            self.assertNotIn("issue_url", record)

    def test_container_defers_task_commit_to_host_finalizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            worktree = root / "worktree"
            task = make_task(worktree)
            publisher = FakePublisher()
            workflow = stub_workflow(root, RepairBackend(change_on_repair=True), publisher, [
                qa_report("FAIL", BLOCKING), qa_report("PASS"), qa_report("PASS")], defer_github=True)
            attach_state(workflow, root, worktree)
            workflow._process_task(TASK_ID, task)

            record = workflow.state.task(TASK_ID)
            self.assertTrue(record["commit_pending"])
            self.assertIsNone(record["commit"])
            self.assertEqual(publisher.commits, [])

    def test_warning_is_reported_without_repair(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            worktree = root / "worktree"
            task = make_task(worktree)
            backend = RepairBackend()
            workflow = stub_workflow(root, backend, FakePublisher(), [
                qa_report("PASS", [finding("QA-NONDETERMINISTIC", "warning", "cases ['n4096']")])])
            attach_state(workflow, root, worktree)
            workflow._process_task(TASK_ID, task)

            record = workflow.state.task(TASK_ID)
            self.assertEqual(record["state"], "completed")
            self.assertEqual(record["warnings"], ["QA-NONDETERMINISTIC: cases ['n4096']"])
            self.assertNotIn("repair", backend.roles)

    def test_easy_gate_is_fail_closed_and_language_agnostic(self):
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
