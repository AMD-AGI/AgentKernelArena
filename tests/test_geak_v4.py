"""Offline tests for the GEAK v4 Arena adapter.

These tests exercise handoff validation, result recovery, GPU namespace
mapping, and the single-file patch import boundary.  They intentionally do not
invoke Claude, the Claude Agent SDK, GEAK, a container, or a GPU.
"""

from __future__ import annotations

import importlib
import json
import logging
import subprocess
from pathlib import Path, PurePosixPath
from typing import Callable

import pytest

from agents.geak_v4 import workflow_runner
from src.module_registration import AgentType, load_agent_launcher


geak_launcher = importlib.import_module("agents.geak_v4.launch_agent")


SOURCE = PurePosixPath("src/kernel.py")
ORIGINAL_SOURCE = "value = 1\nsecond = 2\nthird = 3\n"
OPTIMIZED_SOURCE = "value = 4\nsecond = 5\nthird = 6\n"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _handoff(tmp_path: Path) -> tuple[dict[str, object], Path, Path]:
    kernel = tmp_path / "kernel"
    workflow_dir = tmp_path / "geak" / "kernel_workflow"
    eval_dir = tmp_path / "artifacts" / "eval"
    kernel.mkdir(parents=True)
    workflow_dir.mkdir(parents=True)
    (workflow_dir / "kernel_workflow.js").write_text(
        "// offline fixture\n",
        encoding="utf-8",
    )
    handoff: dict[str, object] = {
        "schema_version": workflow_runner.SCHEMA_VERSION,
        "kernel_path": str(kernel),
        "workflow_dir": str(workflow_dir),
        "eval_dir": str(eval_dir),
        "exp_root": str(tmp_path / "artifacts" / "runs"),
        "gpu_ids": "7",
        "budget": 3,
        "min_improve": 0.03,
        "deep_cost": 1,
        # These untrusted values must never override the Arena policy.
        "mode": "author",
        "apply_to_original": True,
    }
    return handoff, kernel, eval_dir


def _workflow_return(
    eval_dir: Path,
    *,
    workload_aligned: bool = False,
) -> dict[str, object]:
    return {
        "eval_dir": str(eval_dir),
        "validation_status": "accepted",
        "final_geomean": 1.20,
        "final_speedup": 1.19,
        "final_patch": str(eval_dir / "final_patch.diff"),
        "workload_aligned": workload_aligned,
    }


def _director_validation(
    eval_dir: Path,
    *,
    validation_status: str = "accepted",
    correctness: str = "pass",
    geomean: object = 1.20,
    weighted: object = 1.50,
) -> dict[str, object]:
    return {
        "validation_status": validation_status,
        "correctness": correctness,
        "director_verified_speedup_geomean": geomean,
        "director_verified_speedup_weighted": weighted,
        "final_patch": str(eval_dir / "final_patch.diff"),
        "applied_to_original": "false",
    }


def _prepare_normalized_result(
    tmp_path: Path,
    *,
    workload_aligned: bool = False,
    validation_status: str = "accepted",
    correctness: str = "pass",
    geomean: object = 1.20,
    weighted: object = 1.50,
) -> tuple[Path, dict[str, object]]:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    _write_json(
        eval_dir / "workflow_return.json",
        _workflow_return(eval_dir, workload_aligned=workload_aligned),
    )
    _write_json(
        eval_dir / "director_validation.json",
        _director_validation(
            eval_dir,
            validation_status=validation_status,
            correctness=correctness,
            geomean=geomean,
            weighted=weighted,
        ),
    )
    (eval_dir / "final_patch.diff").write_text(
        "non-empty offline fixture\n",
        encoding="utf-8",
    )
    return eval_dir, workflow_runner.normalize_result(eval_dir)


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspace"
    source = workspace.joinpath(*SOURCE.parts)
    source.parent.mkdir(parents=True)
    source.write_text(ORIGINAL_SOURCE, encoding="utf-8")
    (workspace / "config.yaml").write_text(
        "task_type: triton2triton\n",
        encoding="utf-8",
    )
    scripts = workspace / "scripts"
    scripts.mkdir()
    (scripts / "test_kernel.py").write_text(
        "def test_kernel(): pass\n",
        encoding="utf-8",
    )
    return workspace


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _write_repo_file(repo: Path, relative: str, content: str | bytes) -> None:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")


def _make_git_patch(
    tmp_path: Path,
    baseline: dict[str, str | bytes],
    mutate: Callable[[Path], None],
    *,
    find_renames: bool = False,
) -> bytes:
    repo = tmp_path / "patch_repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    for relative, content in baseline.items():
        _write_repo_file(repo, relative, content)
    _git(repo, "add", "-A")
    _git(
        repo,
        "-c",
        "user.name=Offline Test",
        "-c",
        "user.email=offline@example.invalid",
        "commit",
        "--allow-empty",
        "-qm",
        "baseline",
    )

    mutate(repo)
    _git(repo, "add", "-A")
    command = ["git", "diff", "--cached", "--binary", "--no-ext-diff"]
    if find_renames:
        command.append("--find-renames")
    command.append("HEAD")
    diff = subprocess.run(
        command,
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout
    assert diff
    return diff


def _import_result(eval_dir: Path, *, status: str = "ok") -> dict[str, object]:
    patch = str(eval_dir / "final_patch.diff")
    return {
        "schema_version": workflow_runner.SCHEMA_VERSION,
        "status": status,
        "validation_status": "accepted",
        "correctness": "pass",
        "applied_to_original": "false",
        "final_speedup": 1.10,
        "eval_dir": str(eval_dir),
        "final_patch": patch,
        "director_final_patch": patch,
        "workflow_final_patch": patch,
    }


def _apply_patch(
    tmp_path: Path,
    patch: bytes,
    *,
    status: str = "ok",
) -> tuple[bool, Path, Path]:
    workspace = _workspace(tmp_path)
    eval_dir = tmp_path / "eval"
    run_dir = tmp_path / "run"
    eval_dir.mkdir()
    run_dir.mkdir()
    (eval_dir / "final_patch.diff").write_bytes(patch)
    applied = geak_launcher._apply_validated_patch(
        result=_import_result(eval_dir, status=status),
        expected_eval_dir=eval_dir,
        workspace=workspace,
        source_path=SOURCE,
        min_improve=0.02,
        run_dir=run_dir,
    )
    return applied, workspace, run_dir


def _workspace_contents(workspace: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(workspace)): path.read_bytes()
        for path in sorted(workspace.rglob("*"))
        if path.is_file()
    }


def test_dry_run_forces_optimize_without_importing_sdk(tmp_path, monkeypatch):
    handoff, _, _ = _handoff(tmp_path)
    handoff_path = tmp_path / "handoff.json"
    result_path = tmp_path / "result.json"
    _write_json(handoff_path, handoff)

    # A dry run must not enter the only function that imports claude_agent_sdk.
    monkeypatch.setattr(
        workflow_runner,
        "invoke_via_sdk",
        lambda *args, **kwargs: pytest.fail("dry-run invoked Claude SDK"),
    )
    assert workflow_runner.main(
        [str(handoff_path), str(result_path), "--dry-run"]
    ) == 0

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "dry_run"
    assert result["workflow_args"]["mode"] == "optimize"
    assert result["workflow_args"]["apply_to_original"] == "false"
    assert result["workflow_args"]["gpu_ids"] == "7"
    assert "Invoke the Workflow tool exactly once" in result["prompt"]
    assert '"apply_to_original": "false"' in result["prompt"]


def test_agent_registry_loads_geak_v4():
    assert AgentType.from_string("geak_v4") is AgentType.GEAK_V4
    assert (
        load_agent_launcher(AgentType.GEAK_V4, logging.getLogger(__name__))
        is geak_launcher.launch_agent
    )


@pytest.mark.parametrize("isolated_field", ["eval_dir", "exp_root"])
def test_handoff_rejects_artifacts_inside_kernel(tmp_path, isolated_field):
    handoff, kernel, _ = _handoff(tmp_path)
    handoff[isolated_field] = str(kernel / "recursive-output")

    with pytest.raises(
        workflow_runner.HandoffError,
        match=rf"{isolated_field} must not be inside kernel_path",
    ):
        workflow_runner.map_workflow_args(handoff)


def test_disposable_input_is_independent_and_omits_run_artifacts(tmp_path):
    workspace = _workspace(tmp_path)
    (workspace / "task_result.yaml").write_text("score: 1\n", encoding="utf-8")
    (workspace / "__pycache__").mkdir()
    (workspace / "__pycache__" / "kernel.pyc").write_bytes(b"cache")
    disposable = tmp_path / "disposable"

    geak_launcher._materialize_disposable_input(workspace, disposable)
    disposable.joinpath(*SOURCE.parts).write_text(
        OPTIMIZED_SOURCE,
        encoding="utf-8",
    )

    assert workspace.joinpath(*SOURCE.parts).read_text() == ORIGINAL_SOURCE
    assert not (disposable / "task_result.yaml").exists()
    assert not (disposable / "__pycache__").exists()


def test_disposable_input_rejects_workspace_symlinks(tmp_path):
    workspace = _workspace(tmp_path)
    (workspace / "src" / "kernel_alias.py").symlink_to("kernel.py")

    with pytest.raises((ValueError, RuntimeError), match="[Ss]ymlink"):
        geak_launcher._materialize_disposable_input(
            workspace,
            tmp_path / "disposable",
        )


def test_artifact_root_must_not_be_a_symlink(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    redirected = tmp_path / "redirected"
    redirected.mkdir()
    (tmp_path / ".workspace_geak_v4").symlink_to(redirected, target_is_directory=True)

    with pytest.raises(RuntimeError, match="real directory"):
        geak_launcher._new_run_paths(workspace)


def test_json_readers_reject_symlinks_and_oversized_files(tmp_path, monkeypatch):
    target = tmp_path / "target.json"
    target.write_text('{"status": "ok"}\n', encoding="utf-8")
    alias = tmp_path / "alias.json"
    alias.symlink_to(target)

    assert workflow_runner._read_json(alias) is None
    assert geak_launcher._read_json(alias) is None

    monkeypatch.setattr(workflow_runner, "_JSON_SIZE_LIMIT", 4)
    monkeypatch.setattr(geak_launcher, "_JSON_SIZE_LIMIT", 4)
    assert workflow_runner._read_json(target) is None
    assert geak_launcher._read_json(target) is None


def test_atomic_json_writers_replace_destination_symlink(tmp_path):
    sentinel = tmp_path / "sentinel.txt"
    sentinel.write_text("SAFE\n", encoding="utf-8")

    launcher_dir = tmp_path / "launcher"
    launcher_dir.mkdir()
    launcher_result = launcher_dir / "result.json"
    launcher_result.symlink_to(sentinel)
    geak_launcher._atomic_write_json(
        launcher_result,
        {"status": "ok"},
        expected_parent_identity=geak_launcher._directory_identity(launcher_dir),
    )

    runner_dir = tmp_path / "runner"
    runner_dir.mkdir()
    runner_result = runner_dir / "result.json"
    runner_result.symlink_to(sentinel)
    workflow_runner._atomic_write_json(runner_result, {"status": "ok"})

    assert sentinel.read_text(encoding="utf-8") == "SAFE\n"
    for result in (launcher_result, runner_result):
        assert not result.is_symlink()
        assert json.loads(result.read_text(encoding="utf-8")) == {"status": "ok"}


def test_workspace_manifest_detects_direct_source_modification(tmp_path):
    workspace = _workspace(tmp_path)
    manifest = geak_launcher._workspace_manifest(workspace)
    workspace.joinpath(*SOURCE.parts).write_text(
        OPTIMIZED_SOURCE,
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="direct mutation.*changed="):
        geak_launcher._verify_workspace_manifest(manifest, workspace)


def test_workspace_manifest_detects_added_file(tmp_path):
    workspace = _workspace(tmp_path)
    manifest = geak_launcher._workspace_manifest(workspace)
    (workspace / "rogue-output.txt").write_text(
        "written outside the disposable input\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="direct mutation.*added="):
        geak_launcher._verify_workspace_manifest(manifest, workspace)


def test_terminal_artifacts_require_complete_schema(tmp_path):
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    _write_json(
        eval_dir / "workflow_return.json",
        {
            "eval_dir": str(eval_dir),
            "validation_status": "accepted",
            "final_patch": "final_patch.diff",
        },
    )
    _write_json(
        eval_dir / "director_validation.json",
        {
            "validation_status": "accepted",
            "correctness": "pass",
            "final_patch": "final_patch.diff",
        },
    )
    assert not workflow_runner._terminal_artifact_exists(eval_dir)

    _write_json(
        eval_dir / "director_validation.json",
        _director_validation(eval_dir),
    )
    assert workflow_runner._terminal_artifact_exists(eval_dir)


def test_full_workflow_return_is_terminal_and_transcript_parser_ignores_noise(
    tmp_path,
):
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    complete = _workflow_return(eval_dir)
    _write_json(eval_dir / "workflow_return.json", complete)
    assert workflow_runner._terminal_artifact_exists(eval_dir)

    wrong_eval = dict(complete, eval_dir=str(tmp_path / "other"))
    transcript = (
        'setup={"enableWorkflows": true}\n'
        + json.dumps(wrong_eval)
        + '\npartial={"eval_dir": '
        + json.dumps(str(eval_dir))
        + "}\n"
        + json.dumps(complete)
    )
    assert workflow_runner._extract_workflow_return(transcript, eval_dir) == complete


def test_completed_background_stream_without_result_fails_immediately():
    state = {
        "background_started": True,
        "terminal_task_seen": True,
        "result_seen": False,
        "producer_done": True,
    }

    error = workflow_runner._completed_producer_error(state, set(), [])

    assert isinstance(error, RuntimeError)
    assert "without a ResultMessage" in str(error)


def test_normalize_non_workload_uses_director_geomean(tmp_path):
    _, result = _prepare_normalized_result(
        tmp_path,
        workload_aligned=False,
        geomean=1.20,
        weighted=9.0,
    )

    assert result["status"] == "ok"
    assert result["final_speedup"] == pytest.approx(1.20)
    assert result["final_geomean"] == pytest.approx(1.20)
    assert result["final_weighted"] == pytest.approx(9.0)


def test_normalize_workload_aligned_uses_director_weighted_speedup(tmp_path):
    _, result = _prepare_normalized_result(
        tmp_path,
        workload_aligned=True,
        geomean=1.20,
        weighted=1.50,
    )

    assert result["status"] == "ok"
    assert result["final_speedup"] == pytest.approx(1.50)


def test_normalize_rejects_non_finite_director_geomean(tmp_path):
    _, result = _prepare_normalized_result(
        tmp_path,
        geomean=float("nan"),
        weighted=1.50,
    )

    assert result["status"] == "error"
    assert result["final_geomean"] is None
    assert "missing or invalid" in result["reason"]


def test_normalize_workload_falls_back_when_weighted_is_non_finite(tmp_path):
    _, result = _prepare_normalized_result(
        tmp_path,
        workload_aligned=True,
        geomean=1.20,
        weighted=float("nan"),
    )

    assert result["status"] == "ok"
    assert result["final_speedup"] == pytest.approx(1.20)
    assert result["final_weighted"] is None


def test_normalize_flagged_candidate_is_rejected(tmp_path):
    _, result = _prepare_normalized_result(
        tmp_path,
        validation_status="flagged",
        correctness="pass",
    )

    assert result["status"] == "rejected"
    assert "did not accept" in result["reason"]


def test_parallel_worker_maps_host_gpu_to_logical_zero(monkeypatch):
    monkeypatch.setenv("AGENT_KERNEL_ARENA_HOST_GPU_ID", "7")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "7")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "7")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setenv("GEAK_V4_GPU_IDS", "7")

    assert geak_launcher._logical_gpu_ids({"gpu_ids": "7"}) == "0"


def test_single_declared_source_accepts_one_regular_kernel(tmp_path):
    workspace = _workspace(tmp_path)

    assert geak_launcher._single_declared_source(
        {"source_file_path": [str(SOURCE)]},
        workspace,
    ) == SOURCE


@pytest.mark.parametrize(
    ("source_value", "message"),
    [
        (["src/kernel.py", "src/other.py"], "exactly one"),
        ("../kernel.py", "safe relative path"),
        ("config.yaml", "protected"),
        ("scripts/test_kernel.py", "protected"),
        ("test_kernel.py", "co-located test/harness"),
        ("test_kernel.cpp", "co-located test/harness"),
        ("test_kernel.hip", "co-located test/harness"),
        ("src/kernel_harness.cu", "co-located test/harness"),
        ("src/kernel_harness.py", "protected"),
    ],
)
def test_single_declared_source_rejects_unsafe_allowlists(
    tmp_path,
    source_value,
    message,
):
    workspace = _workspace(tmp_path)
    (workspace / "test_kernel.py").write_text("kernel = 1\n", encoding="utf-8")
    (workspace / "src" / "kernel_harness.py").write_text(
        "kernel = 1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        geak_launcher._single_declared_source(
            {"source_file_path": source_value},
            workspace,
        )


def test_valid_single_file_patch_is_imported_atomically(tmp_path):
    patch = _make_git_patch(
        tmp_path,
        {str(SOURCE): ORIGINAL_SOURCE},
        lambda repo: _write_repo_file(repo, str(SOURCE), OPTIMIZED_SOURCE),
    )

    applied, workspace, run_dir = _apply_patch(tmp_path, patch)

    assert applied is True
    assert workspace.joinpath(*SOURCE.parts).read_text() == OPTIMIZED_SOURCE
    audit = json.loads(
        (run_dir / "applied_patch.json").read_text(encoding="utf-8")
    )
    assert audit["source_file"] == str(SOURCE)
    assert audit["director_speedup"] == pytest.approx(1.10)


def test_patch_audit_replaces_symlink_without_overwriting_target(tmp_path):
    patch = _make_git_patch(
        tmp_path,
        {str(SOURCE): ORIGINAL_SOURCE},
        lambda repo: _write_repo_file(repo, str(SOURCE), OPTIMIZED_SOURCE),
    )
    workspace = _workspace(tmp_path)
    eval_dir = tmp_path / "eval"
    run_dir = tmp_path / "run"
    eval_dir.mkdir()
    run_dir.mkdir()
    (eval_dir / "final_patch.diff").write_bytes(patch)
    sentinel = tmp_path / "sentinel.txt"
    sentinel.write_text("SAFE\n", encoding="utf-8")
    audit = run_dir / "applied_patch.json"
    audit.symlink_to(sentinel)

    assert geak_launcher._apply_validated_patch(
        result=_import_result(eval_dir),
        expected_eval_dir=eval_dir,
        workspace=workspace,
        source_path=SOURCE,
        min_improve=0.02,
        run_dir=run_dir,
    )

    assert sentinel.read_text(encoding="utf-8") == "SAFE\n"
    assert not audit.is_symlink()
    assert json.loads(audit.read_text(encoding="utf-8"))["source_file"] == str(SOURCE)


def test_valid_patch_applies_inside_an_outer_git_worktree(tmp_path):
    patch = _make_git_patch(
        tmp_path,
        {str(SOURCE): ORIGINAL_SOURCE},
        lambda repo: _write_repo_file(repo, str(SOURCE), OPTIMIZED_SOURCE),
    )
    outer = tmp_path / "outer_repo"
    outer.mkdir()
    _git(outer, "init", "-q")
    workspace = _workspace(outer)
    eval_dir = outer / "eval"
    run_dir = outer / "run"
    eval_dir.mkdir()
    run_dir.mkdir()
    (eval_dir / "final_patch.diff").write_bytes(patch)

    assert geak_launcher._apply_validated_patch(
        result=_import_result(eval_dir),
        expected_eval_dir=eval_dir,
        workspace=workspace,
        source_path=SOURCE,
        min_improve=0.02,
        run_dir=run_dir,
    )
    assert workspace.joinpath(*SOURCE.parts).read_text() == OPTIMIZED_SOURCE


def test_normalize_rejects_patch_not_named_by_director(tmp_path):
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    _write_json(eval_dir / "workflow_return.json", _workflow_return(eval_dir))
    validation = _director_validation(eval_dir)
    validation["final_patch"] = str(eval_dir / "validated_elsewhere.diff")
    _write_json(eval_dir / "director_validation.json", validation)
    (eval_dir / "final_patch.diff").write_text(
        "unvalidated patch\n",
        encoding="utf-8",
    )

    result = workflow_runner.normalize_result(eval_dir)

    assert result["status"] == "error"
    assert "Director artifact is missing or invalid" in result["reason"]


def test_patch_import_rechecks_director_patch_provenance(tmp_path):
    patch = _make_git_patch(
        tmp_path,
        {str(SOURCE): ORIGINAL_SOURCE},
        lambda repo: _write_repo_file(repo, str(SOURCE), OPTIMIZED_SOURCE),
    )
    workspace = _workspace(tmp_path)
    eval_dir = tmp_path / "eval"
    run_dir = tmp_path / "run"
    eval_dir.mkdir()
    run_dir.mkdir()
    (eval_dir / "final_patch.diff").write_bytes(patch)
    result = _import_result(eval_dir)
    result["director_final_patch"] = str(eval_dir / "validated_elsewhere.diff")

    with pytest.raises(RuntimeError, match="director_final_patch"):
        geak_launcher._apply_validated_patch(
            result=result,
            expected_eval_dir=eval_dir,
            workspace=workspace,
            source_path=SOURCE,
            min_improve=0.02,
            run_dir=run_dir,
        )

    assert workspace.joinpath(*SOURCE.parts).read_text() == ORIGINAL_SOURCE


def test_patch_import_rejects_unknown_result_schema(tmp_path):
    patch = _make_git_patch(
        tmp_path,
        {str(SOURCE): ORIGINAL_SOURCE},
        lambda repo: _write_repo_file(repo, str(SOURCE), OPTIMIZED_SOURCE),
    )
    workspace = _workspace(tmp_path)
    eval_dir = tmp_path / "eval"
    run_dir = tmp_path / "run"
    eval_dir.mkdir()
    run_dir.mkdir()
    (eval_dir / "final_patch.diff").write_bytes(patch)
    result = _import_result(eval_dir)
    result["schema_version"] = 999

    with pytest.raises(RuntimeError, match="schema"):
        geak_launcher._apply_validated_patch(
            result=result,
            expected_eval_dir=eval_dir,
            workspace=workspace,
            source_path=SOURCE,
            min_improve=0.02,
            run_dir=run_dir,
        )

    assert workspace.joinpath(*SOURCE.parts).read_text() == ORIGINAL_SOURCE


@pytest.mark.parametrize("status", ["no_gain", "rejected"])
def test_non_accepted_status_never_parses_or_applies_patch(tmp_path, status):
    traversal = (
        b"diff --git a/../escape.py b/../escape.py\n"
        b"--- a/../escape.py\n"
        b"+++ b/../escape.py\n"
        b"@@ -1 +1 @@\n"
        b"-unsafe\n"
        b"+escaped\n"
    )

    applied, workspace, _ = _apply_patch(tmp_path, traversal, status=status)

    assert applied is False
    assert workspace.joinpath(*SOURCE.parts).read_text() == ORIGINAL_SOURCE
    assert not (tmp_path / "escape.py").exists()


def _malicious_patch(tmp_path: Path, kind: str) -> bytes:
    if kind == "multi_file":
        def mutate(repo: Path) -> None:
            _write_repo_file(repo, str(SOURCE), OPTIMIZED_SOURCE)
            _write_repo_file(repo, "config.yaml", "task_type: fake\n")

        return _make_git_patch(
            tmp_path,
            {
                str(SOURCE): ORIGINAL_SOURCE,
                "config.yaml": "task_type: triton2triton\n",
            },
            mutate,
        )
    if kind == "new_file":
        return _make_git_patch(
            tmp_path,
            {},
            lambda repo: _write_repo_file(repo, str(SOURCE), OPTIMIZED_SOURCE),
        )
    if kind == "delete":
        return _make_git_patch(
            tmp_path,
            {str(SOURCE): ORIGINAL_SOURCE},
            lambda repo: (repo / str(SOURCE)).unlink(),
        )
    if kind == "rename":
        def rename(repo: Path) -> None:
            destination = repo / "src" / "renamed.py"
            (repo / str(SOURCE)).rename(destination)

        return _make_git_patch(
            tmp_path,
            {str(SOURCE): ORIGINAL_SOURCE},
            rename,
            find_renames=True,
        )
    if kind == "binary":
        return _make_git_patch(
            tmp_path,
            {str(SOURCE): b"old\x00binary\n"},
            lambda repo: _write_repo_file(
                repo,
                str(SOURCE),
                b"new\x00binary\n",
            ),
        )
    if kind == "traversal":
        return (
            b"diff --git a/../escape.py b/../escape.py\n"
            b"index 1234567..7654321 100644\n"
            b"--- a/../escape.py\n"
            b"+++ b/../escape.py\n"
            b"@@ -1 +1 @@\n"
            b"-unsafe\n"
            b"+escaped\n"
        )
    raise AssertionError(f"unhandled malicious patch kind: {kind}")


@pytest.mark.parametrize(
    "kind",
    ["multi_file", "new_file", "delete", "rename", "binary", "traversal"],
)
def test_malicious_patch_is_rejected_without_workspace_mutation(tmp_path, kind):
    patch = _malicious_patch(tmp_path, kind)
    workspace = _workspace(tmp_path)
    before = _workspace_contents(workspace)
    eval_dir = tmp_path / "eval"
    run_dir = tmp_path / "run"
    eval_dir.mkdir()
    run_dir.mkdir()
    (eval_dir / "final_patch.diff").write_bytes(patch)

    with pytest.raises(RuntimeError):
        geak_launcher._apply_validated_patch(
            result=_import_result(eval_dir),
            expected_eval_dir=eval_dir,
            workspace=workspace,
            source_path=SOURCE,
            min_improve=0.02,
            run_dir=run_dir,
        )

    assert _workspace_contents(workspace) == before
    assert not (tmp_path / "escape.py").exists()
