from __future__ import annotations

import json

import pytest
import yaml

from src.eval_tools.contracts import (
    ArtifactKind,
    CapabilityCheck,
    EvaluationPlan,
    EvaluationPolicy,
    EvaluationReport,
    ExecutionRecord,
    ExecutionStatus,
    FindingStatus,
    InstrumentationControl,
    KernelLanguage,
    PolicyDecision,
    TaskProfile,
    ToolCapability,
    ToolEvaluation,
    ToolPlan,
    ToolRunResult,
)
from src.eval_tools.reporting import (
    has_current_plan,
    merge_task_result_data,
    merge_task_result_file,
    serialize_report,
    write_report,
)


def _report(tmp_path):
    profile = TaskProfile(
        task_type="triton2triton",
        language=KernelLanguage.TRITON,
        artifact_kind=ArtifactKind.PYTHON_JIT,
        framework="standalone",
        instrumentation_control=InstrumentationControl.COMPILER_CONTROLLED,
        adapter="triton_python_jit",
        source_available=True,
        source_files=("kernel.py",),
    )
    plan = EvaluationPlan(
        schema_version=1,
        policy=EvaluationPolicy.ADVISORY,
        profile=profile,
        tools=(
            ToolPlan(
                tool="gpu_asan",
                runtime_ref="image@sha256:abc",
                plugin_version="1.0",
            ),
        ),
        fingerprint="f" * 64,
    )
    capability = ToolCapability(
        "gpu_asan",
        CapabilityCheck.ready(),
        CapabilityCheck.ready(),
        CapabilityCheck.ready(image_digest="sha256:abc"),
    )
    execution = ExecutionRecord(
        command=("asan", "runner"),
        returncode=0,
        stdout="large raw stdout",
        stderr="diagnostic",
    )
    result = ToolRunResult(
        tool="gpu_asan",
        execution=ExecutionStatus.COMPLETED,
        finding=FindingStatus.CLEAN,
        artifacts=(str(tmp_path / "gpu_asan" / "stderr.log"),),
        execution_record=execution,
    )
    return EvaluationReport(
        plan=plan,
        evaluations=(ToolEvaluation(capability, result),),
        decision=PolicyDecision(EvaluationPolicy.ADVISORY, True, True),
    )


def test_summary_serialization_omits_raw_output_by_default(tmp_path):
    report = _report(tmp_path)
    summary = serialize_report(report)
    execution = summary["tools"]["gpu_asan"]["result"]["execution_record"]
    assert "stdout" not in execution
    assert "stderr" not in execution
    assert summary["overall_status"] == "clean"
    assert summary["plan_fingerprint"] == "f" * 64
    assert summary["plan"]["tools"][0] == {
        "tool": "gpu_asan",
        "runtime_ref": "image@sha256:abc",
        "plugin_version": "1.0",
        "timeout_s": 3600,
        "options": {},
    }

    verbose = serialize_report(report, include_output=True)
    assert (
        verbose["tools"]["gpu_asan"]["result"]["execution_record"]["stdout"]
        == "large raw stdout"
    )


def test_write_report_accepts_directory_and_writes_valid_json(tmp_path):
    path = write_report(_report(tmp_path), tmp_path / "tool_reports")
    assert path.name == "summary.json"
    loaded = json.loads(path.read_text())
    assert loaded["tools"]["gpu_asan"]["capability"]["effective"]["state"] == "ready"


def test_merge_preserves_legacy_scoring_fields_and_plan_can_be_checked(tmp_path):
    report = _report(tmp_path)
    original = {
        "task_name": "triton2triton/example",
        "pass_compilation": True,
        "pass_correctness": True,
        "speedup_ratio": 1.5,
        "score": 270.0,
    }
    merged = merge_task_result_data(original, report)
    assert merged["score"] == 270.0
    assert merged["pass_correctness"] is True
    assert has_current_plan(merged, "f" * 64)
    assert not has_current_plan(merged, "0" * 64)

    path = tmp_path / "task_result.yaml"
    path.write_text(yaml.safe_dump(original, sort_keys=False))
    merge_task_result_file(path, report)
    loaded = yaml.safe_load(path.read_text())
    assert loaded["speedup_ratio"] == 1.5
    assert loaded["tool_evaluation"]["overall_status"] == "clean"
    assert has_current_plan(path, "f" * 64)


def test_atomic_merge_replaces_destination_symlink_not_its_target(tmp_path):
    report = _report(tmp_path)
    victim = tmp_path / "victim.yaml"
    victim.write_text("secret: keep\n")
    result_path = tmp_path / "task_result.yaml"
    result_path.symlink_to(victim)

    merge_task_result_file(result_path, report)

    assert not result_path.is_symlink()
    assert victim.read_text() == "secret: keep\n"
    assert yaml.safe_load(result_path.read_text())["secret"] == "keep"


def test_non_mapping_task_result_is_rejected(tmp_path):
    path = tmp_path / "task_result.yaml"
    path.write_text("- not\n- a\n- mapping\n")
    with pytest.raises(ValueError, match="not a mapping"):
        merge_task_result_file(path, _report(tmp_path))
