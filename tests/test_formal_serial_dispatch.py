import logging
from pathlib import Path
from types import SimpleNamespace

import main as aka_main
from src.campaign import CampaignError
from src.module_registration import AgentType


def _context(tmp_path: Path) -> dict[str, object]:
    return {
        "config": {
            "campaign": {
                "comparison": "apex_vs_codex",
                "attempts": 3,
                "attempt_timeout_seconds": 3600,
                "apex_internal_allowance_seconds": 3600,
                "task_timeout_seconds": 25200,
                "evaluator_allowance_seconds": 3600,
                "selection_policy": "correctness_then_measured_rate_v1",
                "workspace_policy": "fresh_per_attempt",
                "gpu_policy": "deterministic_task_gpu_v1",
                "require_clean_checkouts": True,
            }
        },
        "agent": AgentType.CODEX,
        "agent_launcher": object(),
        "run_directory": tmp_path / "run",
        "timestamp": "20260808_000000",
        "resume_mode": False,
        "logger": logging.getLogger(__name__),
        "task_config_dict": {
            "task/a": "/source/task_a/config.yaml",
            "task/b": "/source/task_b/config.yaml",
        },
    }


def test_formal_serial_dispatch_uses_validated_task_gpu_bindings(
    tmp_path: Path, monkeypatch
) -> None:
    context = _context(tmp_path)
    observed: list[tuple[str, str, str]] = []
    postprocessed: list[str] = []

    monkeypatch.setattr(aka_main, "_build_context", lambda *_args, **_kwargs: context)
    monkeypatch.setattr(
        aka_main,
        "_validated_formal_task_bindings",
        lambda _run, _tasks: {
            "task/a": {
                "assigned_host_gpu_id": "0",
                "config_path": "/sealed/task_a/config.yaml",
                "task_index": 1,
                "total_tasks": 2,
            },
            "task/b": {
                "assigned_host_gpu_id": "0",
                "config_path": "/sealed/task_b/config.yaml",
                "task_index": 2,
                "total_tasks": 2,
            },
        },
    )

    def run_task(**kwargs):
        observed.append(
            (
                kwargs["task_name"],
                kwargs["eval_config"]["assigned_host_gpu_id"],
                kwargs["task_config_dir"],
            )
        )
        workspace = tmp_path / kwargs["task_name"].replace("/", "_")
        return True, workspace

    monkeypatch.setattr(aka_main, "run_task", run_task)
    monkeypatch.setattr(
        aka_main,
        "run_post_processing",
        lambda _agent, paths, _logger, **_kwargs: postprocessed.extend(paths),
    )

    assert aka_main.run_serial(SimpleNamespace()) == 0
    assert observed == [
        ("task/a", "0", "/sealed/task_a/config.yaml"),
        ("task/b", "0", "/sealed/task_b/config.yaml"),
    ]
    assert postprocessed == [
        str(tmp_path / "task_a"),
        str(tmp_path / "task_b"),
    ]


def test_formal_serial_dispatch_fails_before_tasks_on_binding_error(
    tmp_path: Path, monkeypatch
) -> None:
    context = _context(tmp_path)
    called = False

    monkeypatch.setattr(aka_main, "_build_context", lambda *_args, **_kwargs: context)

    def reject_bindings(_run, _tasks):
        raise CampaignError("sealed binding mismatch")

    def run_task(**_kwargs):
        nonlocal called
        called = True
        return False, None

    monkeypatch.setattr(
        aka_main, "_validated_formal_task_bindings", reject_bindings
    )
    monkeypatch.setattr(aka_main, "run_task", run_task)

    assert aka_main.run_serial(SimpleNamespace()) == 1
    assert called is False


def test_formal_serial_resume_preserves_original_manifest_indices(
    tmp_path: Path, monkeypatch
) -> None:
    context = _context(tmp_path)
    context["resume_mode"] = True
    observed: list[tuple[str, int, int]] = []

    monkeypatch.setattr(aka_main, "_build_context", lambda *_args, **_kwargs: context)

    def full_manifest_bindings(_run, tasks):
        assert list(tasks) == ["task/a", "task/b"]
        return {
            "task/a": {
                "assigned_host_gpu_id": "0",
                "config_path": "/sealed/task_a/config.yaml",
                "task_index": 1,
                "total_tasks": 2,
            },
            "task/b": {
                "assigned_host_gpu_id": "0",
                "config_path": "/sealed/task_b/config.yaml",
                "task_index": 2,
                "total_tasks": 2,
            },
        }

    monkeypatch.setattr(
        aka_main,
        "_validated_formal_task_bindings",
        full_manifest_bindings,
    )
    monkeypatch.setattr(
        aka_main,
        "_filter_completed_tasks",
        lambda tasks, *_args: {"task/b": tasks["task/b"]},
    )

    def run_task(**kwargs):
        observed.append(
            (
                kwargs["task_name"],
                kwargs["task_index"],
                kwargs["total_tasks"],
            )
        )
        return True, tmp_path / "task_b"

    monkeypatch.setattr(aka_main, "run_task", run_task)
    monkeypatch.setattr(aka_main, "run_post_processing", lambda *_args, **_kwargs: None)

    assert aka_main.run_serial(SimpleNamespace()) == 0
    assert observed == [("task/b", 2, 2)]
