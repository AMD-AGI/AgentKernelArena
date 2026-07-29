"""Offline tests for the GEAK v4 Arena adapter.

These tests exercise the SDK-free surface: handoff mapping (including the
handoff-driven ``apply_to_original``), on-disk result recovery/normalization, GPU
namespace mapping, and the simplified launcher's handoff construction. They
intentionally do not invoke Claude, the Claude Agent SDK, GEAK, a container, or a
GPU.
"""

from __future__ import annotations

import importlib
import json
import logging
from pathlib import Path

import pytest

from agents.geak_v4 import workflow_runner
from src.module_registration import AgentType, load_agent_launcher


geak_launcher = importlib.import_module("agents.geak_v4.launch_agent")


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
        # The workflow always runs in optimize mode regardless of the handoff.
        "mode": "author",
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
    applied_to_original: str = "true",
) -> dict[str, object]:
    return {
        "validation_status": validation_status,
        "correctness": correctness,
        "director_verified_speedup_geomean": geomean,
        "director_verified_speedup_weighted": weighted,
        "final_patch": str(eval_dir / "final_patch.diff"),
        "applied_to_original": applied_to_original,
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


def _make_task(tmp_path: Path) -> tuple[Path, Path]:
    """Create a minimal task workspace + config.yaml with one kernel source."""
    workspace = tmp_path / "workspace"
    source = workspace / "src" / "kernel.py"
    source.parent.mkdir(parents=True)
    source.write_text("value = 1\n", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        "task_type: hip2hip\nsource_file_path:\n  - src/kernel.py\n",
        encoding="utf-8",
    )
    return workspace, config


# --------------------------------------------------------------------------- #
# Registration + launcher
# --------------------------------------------------------------------------- #
def test_agent_registry_loads_geak_v4():
    assert AgentType.from_string("geak_v4") is AgentType.GEAK_V4
    assert (
        load_agent_launcher(AgentType.GEAK_V4, logging.getLogger(__name__))
        is geak_launcher.launch_agent
    )


def test_declared_sources_accepts_str_and_list(tmp_path):
    workspace, _ = _make_task(tmp_path)
    assert geak_launcher._declared_sources(
        {"source_file_path": "src/kernel.py"}, workspace
    ) == ["src/kernel.py"]
    assert geak_launcher._declared_sources(
        {"source_file_path": ["src/kernel.py"]}, workspace
    ) == ["src/kernel.py"]


def test_declared_sources_empty_when_unset(tmp_path):
    workspace, _ = _make_task(tmp_path)
    assert geak_launcher._declared_sources({}, workspace) == []


def test_declared_sources_fails_when_anchor_missing(tmp_path):
    workspace, _ = _make_task(tmp_path)
    with pytest.raises(FileNotFoundError, match="not found in workspace"):
        geak_launcher._declared_sources(
            {"source_file_path": ["src/missing.py"]}, workspace
        )


def test_launch_agent_rejects_unsupported_task_type(tmp_path):
    workspace, _ = _make_task(tmp_path)
    config = tmp_path / "bad.yaml"
    config.write_text(
        "task_type: repo2repo\nsource_file_path: [src/kernel.py]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="does not support task_type"):
        geak_launcher.launch_agent({}, str(config), str(workspace))


def test_launch_agent_writes_apply_in_place_handoff(tmp_path, monkeypatch):
    workspace, config = _make_task(tmp_path)
    workflow_dir = tmp_path / "geak" / "kernel_workflow"
    workflow_dir.mkdir(parents=True)
    (workflow_dir / "kernel_workflow.js").write_text("// fixture\n", encoding="utf-8")

    monkeypatch.setenv("GEAK_V4_WORKFLOW_DIR", str(workflow_dir))
    monkeypatch.setattr(geak_launcher.shutil, "which", lambda name: "/usr/bin/claude")
    monkeypatch.setattr(
        geak_launcher,
        "load_prompt_builder",
        lambda *args, **kwargs: (lambda *a, **k: "BASE PROMPT"),
    )

    captured: dict[str, object] = {}

    def fake_runner(handoff_path, result_path, *, timeout_seconds, logger):
        handoff = json.loads(Path(handoff_path).read_text(encoding="utf-8"))
        captured["handoff"] = handoff
        Path(result_path).write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "ok",
                    "applied_to_original": "true",
                    "final_speedup": 1.3,
                    "eval_dir": handoff["eval_dir"],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return "runner-output"

    monkeypatch.setattr(geak_launcher, "_run_workflow_runner", fake_runner)

    output = geak_launcher.launch_agent({"gpu_ids": "0"}, str(config), str(workspace))

    handoff = captured["handoff"]
    assert handoff["schema_version"] == 1
    # GEAK edits the workspace directly; Arena's harness guard re-scores it.
    assert handoff["apply_to_original"] == "true"
    assert handoff["kernel_path"] == str(workspace.resolve())
    assert handoff["workflow_dir"] == str(workflow_dir.resolve())
    assert handoff["claude_cli_path"] == "/usr/bin/claude"
    assert "BASE PROMPT" in handoff["task"]
    assert "src/kernel.py" in handoff["task"]

    # Artifacts must live OUTSIDE the scored workspace (hidden sibling dir).
    eval_dir = Path(handoff["eval_dir"]).resolve()
    assert not eval_dir.is_relative_to(workspace.resolve())
    artifact_root = workspace.parent / f".{workspace.name}_geak_v4"
    assert eval_dir.parent.parent == artifact_root.resolve()

    assert "runner-output" in output
    assert '"status": "ok"' in output


def test_launch_agent_errors_without_claude_cli(tmp_path, monkeypatch):
    workspace, config = _make_task(tmp_path)
    workflow_dir = tmp_path / "geak" / "kernel_workflow"
    workflow_dir.mkdir(parents=True)
    (workflow_dir / "kernel_workflow.js").write_text("// fixture\n", encoding="utf-8")
    monkeypatch.setenv("GEAK_V4_WORKFLOW_DIR", str(workflow_dir))
    monkeypatch.setattr(geak_launcher.shutil, "which", lambda name: None)

    with pytest.raises(RuntimeError, match="Claude Code CLI"):
        geak_launcher.launch_agent({}, str(config), str(workspace))


def test_parallel_worker_maps_host_gpu_to_logical_zero(monkeypatch):
    monkeypatch.setenv("AGENT_KERNEL_ARENA_HOST_GPU_ID", "7")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "7")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "7")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setenv("GEAK_V4_GPU_IDS", "7")

    assert geak_launcher._logical_gpu_ids({"gpu_ids": "7"}) == "0"


# --------------------------------------------------------------------------- #
# workflow_runner: handoff mapping + apply_to_original
# --------------------------------------------------------------------------- #
def test_dry_run_defaults_apply_to_original_false(tmp_path, monkeypatch):
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
    assert "apply_to_original is false" in result["prompt"]


def test_apply_to_original_is_handoff_driven(tmp_path):
    handoff, _, _ = _handoff(tmp_path)
    handoff["apply_to_original"] = "true"

    script_path, args = workflow_runner.map_workflow_args(handoff)
    assert args["apply_to_original"] == "true"
    assert args["mode"] == "optimize"

    prompt = workflow_runner.build_prompt(script_path, args)
    assert "apply_to_original is true" in prompt
    assert "the caller owns patch import" not in prompt


def test_invalid_apply_to_original_is_rejected(tmp_path):
    handoff, _, _ = _handoff(tmp_path)
    handoff["apply_to_original"] = "maybe"
    with pytest.raises(workflow_runner.HandoffError, match="apply_to_original"):
        workflow_runner.map_workflow_args(handoff)


@pytest.mark.parametrize("isolated_field", ["eval_dir", "exp_root"])
def test_handoff_rejects_artifacts_inside_kernel(tmp_path, isolated_field):
    handoff, kernel, _ = _handoff(tmp_path)
    handoff[isolated_field] = str(kernel / "recursive-output")

    with pytest.raises(
        workflow_runner.HandoffError,
        match=rf"{isolated_field} must not be inside kernel_path",
    ):
        workflow_runner.map_workflow_args(handoff)


# --------------------------------------------------------------------------- #
# workflow_runner: on-disk readers + terminal artifact detection
# --------------------------------------------------------------------------- #
def test_runner_read_json_rejects_symlink_and_oversize(tmp_path, monkeypatch):
    target = tmp_path / "target.json"
    target.write_text('{"status": "ok"}\n', encoding="utf-8")
    alias = tmp_path / "alias.json"
    alias.symlink_to(target)

    assert workflow_runner._read_json(alias) is None

    monkeypatch.setattr(workflow_runner, "_JSON_SIZE_LIMIT", 4)
    assert workflow_runner._read_json(target) is None


def test_runner_atomic_write_replaces_destination_symlink(tmp_path):
    sentinel = tmp_path / "sentinel.txt"
    sentinel.write_text("SAFE\n", encoding="utf-8")
    runner_dir = tmp_path / "runner"
    runner_dir.mkdir()
    runner_result = runner_dir / "result.json"
    runner_result.symlink_to(sentinel)

    workflow_runner._atomic_write_json(runner_result, {"status": "ok"})

    assert sentinel.read_text(encoding="utf-8") == "SAFE\n"
    assert not runner_result.is_symlink()
    assert json.loads(runner_result.read_text(encoding="utf-8")) == {"status": "ok"}


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


# --------------------------------------------------------------------------- #
# workflow_runner: result normalization
# --------------------------------------------------------------------------- #
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
    assert result["applied_to_original"] == "true"


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


def test_normalize_ok_requires_patch_applied_when_launcher_asks(tmp_path):
    """An accepted gain that never reached the workspace must not report ok."""
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    _write_json(eval_dir / "workflow_return.json", _workflow_return(eval_dir))
    _write_json(
        eval_dir / "director_validation.json",
        _director_validation(eval_dir, applied_to_original="false"),
    )
    (eval_dir / "final_patch.diff").write_text(
        "non-empty offline fixture\n",
        encoding="utf-8",
    )

    relaxed = workflow_runner.normalize_result(eval_dir)
    assert relaxed["status"] == "ok"

    strict = workflow_runner.normalize_result(eval_dir, require_applied=True)
    assert strict["status"] == "error"
    assert strict["applied_to_original"] == "false"
    assert "did not apply the patch to the workspace" in strict["reason"]


def test_normalize_ok_when_patch_applied_and_apply_required(tmp_path):
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    _write_json(eval_dir / "workflow_return.json", _workflow_return(eval_dir))
    _write_json(
        eval_dir / "director_validation.json",
        _director_validation(eval_dir, applied_to_original="true"),
    )
    (eval_dir / "final_patch.diff").write_text(
        "non-empty offline fixture\n",
        encoding="utf-8",
    )

    result = workflow_runner.normalize_result(eval_dir, require_applied=True)
    assert result["status"] == "ok"
    assert result["applied_to_original"] == "true"


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
