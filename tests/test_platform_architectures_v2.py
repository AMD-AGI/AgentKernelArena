"""Architecture eligibility and runtime identity; CPU fixtures, not GPU evidence."""
import json
import logging
from types import SimpleNamespace

import pytest

from agents.quality_loop.orchestrator import QualityLoop
from main import should_run_task_for_platform
from src.task_runtime import bind_session_runtime
from src.task_spec import TaskConfigError, TaskSpec


def task(required):
    return TaskSpec.from_mapping({
        "schema_version": 2,
        "candidate": {"language": "hip", "editable": ["kernel.hip"]},
        "evaluation": {"runner": ["python3", "evaluate.py"]},
        "platform_support": {"required_arch": required},
    }, task_id="example/operator")


@pytest.mark.parametrize("required", ["gfx950", ["gfx950"], ["gfx942", "gfx950"]])
@pytest.mark.parametrize("actual", ["gfx942", "gfx950", "gfx1100", None])
def test_scheduling_quality_loop_and_runtime_agree(required, actual, tmp_path, monkeypatch):
    spec = task(required)
    allowed = required if isinstance(required, list) else [required]
    expected = actual in allowed
    assert spec.to_mapping()["platform_support"]["required_arch"] == required
    assert should_run_task_for_platform(spec.task_id, spec.to_mapping(), actual,
                                        logging.getLogger(__name__)) is expected
    assert QualityLoop._platform_matches(spec.to_mapping(), actual) is expected
    monkeypatch.setattr("src.task_runtime._runtime_identity", lambda: {"gpu_arch": actual})
    session = SimpleNamespace(spec=spec, state_directory=tmp_path)
    if expected:
        assert bind_session_runtime(session)["gpu_arch"] == actual
    else:
        with pytest.raises(RuntimeError, match="Task requires"):
            bind_session_runtime(session)
        assert not (tmp_path / "runtime_identity.json").exists()


@pytest.mark.parametrize("invalid", [None, [], ["gfx950", "gfx950"], ["gfx950", 950],
                                     [""], ["gfx950", ["gfx942"]], True, "gfx9*", "gfx942 gfx950"])
def test_invalid_architecture_declarations_fail_before_execution(invalid):
    with pytest.raises(TaskConfigError, match="required_arch"):
        task(invalid)


def test_allowed_architectures_do_not_allow_reusing_baseline_on_other_gpu(tmp_path, monkeypatch):
    session = SimpleNamespace(spec=task(["gfx942", "gfx950"]), state_directory=tmp_path)
    monkeypatch.setattr("src.task_runtime._runtime_identity", lambda: {"gpu_arch": "gfx942"})
    bind_session_runtime(session)
    recorded = (tmp_path / "runtime_identity.json").read_bytes()
    monkeypatch.setattr("src.task_runtime._runtime_identity", lambda: {"gpu_arch": "gfx950"})
    with pytest.raises(RuntimeError, match="Scoring runtime changed"):
        bind_session_runtime(session)
    assert (tmp_path / "runtime_identity.json").read_bytes() == recorded
    assert json.loads(recorded)["gpu_arch"] == "gfx942"


def test_explicit_skip_overrides_architecture_match():
    config = task(["gfx942", "gfx950"]).to_mapping()
    config["platform_support"].update(status="skip", skip_reason="Pending task maintenance")
    assert not should_run_task_for_platform("example/operator", config, "gfx950", logging.getLogger(__name__))
    assert not QualityLoop._platform_matches(config, "gfx950")
