from __future__ import annotations

import pytest

from src.eval_tools.contracts import (
    ArtifactKind,
    CapabilityCheck,
    CapabilityState,
    ExecutionRecord,
    ExecutionStatus,
    Finding,
    FindingSeverity,
    FindingStatus,
    InstrumentationControl,
    KernelLanguage,
    TaskProfile,
    ToolCapability,
    ToolContext,
    ToolInvocation,
    ToolRunResult,
    effective_capability,
)


def _profile() -> TaskProfile:
    return TaskProfile(
        task_type="triton2triton",
        language=KernelLanguage.TRITON,
        artifact_kind=ArtifactKind.PYTHON_JIT,
        framework="standalone",
        instrumentation_control=InstrumentationControl.COMPILER_CONTROLLED,
        adapter="triton_python_jit",
        source_available=True,
        source_files=("kernel.py",),
    )


def test_capability_dimensions_remain_independent_and_effective_is_derived():
    engine = CapabilityCheck.ready(commit="abc")
    adapter = CapabilityCheck.blocked(
        CapabilityState.ADAPTER_REQUIRED, "NEEDS_AOT"
    )
    runtime = CapabilityCheck.blocked(
        CapabilityState.UNAVAILABLE_RUNTIME, "IMAGE_MISSING"
    )

    capability = ToolCapability("rocjitsu", engine, adapter, runtime)

    assert capability.engine.evidence == {"commit": "abc"}
    assert capability.adapter.state == CapabilityState.ADAPTER_REQUIRED
    assert capability.runtime.state == CapabilityState.UNAVAILABLE_RUNTIME
    assert capability.effective.state == CapabilityState.ADAPTER_REQUIRED
    assert capability.effective.reason_code == "NEEDS_AOT"
    assert not capability.ready


def test_not_applicable_precedes_runtime_unavailability():
    effective = effective_capability(
        CapabilityCheck.blocked(CapabilityState.NOT_APPLICABLE, "WRONG_LANGUAGE"),
        CapabilityCheck.ready(),
        CapabilityCheck.blocked(
            CapabilityState.UNAVAILABLE_RUNTIME, "SIDECAR_OFFLINE"
        ),
    )
    assert effective.state == CapabilityState.NOT_APPLICABLE
    assert effective.reason_code == "WRONG_LANGUAGE"


def test_contracts_round_trip_through_plain_dict(tmp_path):
    context = ToolContext(
        workspace=str(tmp_path),
        task_config={"task_type": "triton2triton"},
        profile=_profile(),
        artifact_dir=str(tmp_path / "artifacts"),
        gpu_arch="gfx950",
        env={"A": "1"},
        options={"positive_control_required": True},
    )
    restored_context = ToolContext.from_dict(context.to_dict())
    assert restored_context == context

    invocation = ToolInvocation(
        tool="gpu-asan",
        command=("python3", "runner.py"),
        cwd=str(tmp_path),
        timeout_s=12,
    )
    restored_invocation = ToolInvocation.from_dict(invocation.to_dict())
    assert restored_invocation == invocation
    assert restored_invocation.tool == "gpu_asan"

    execution = ExecutionRecord(
        command=invocation.command,
        returncode=0,
        stdout="safe",
        duration_s=0.5,
    )
    assert execution.status == ExecutionStatus.COMPLETED
    assert ExecutionRecord.from_dict(execution.to_dict()) == execution


def test_finding_status_is_not_inferred_from_process_return_code():
    execution = ExecutionRecord(
        command=("rocjitsu", "launcher"), returncode=0, stdout="RACE ... END_RACE"
    )
    finding = Finding(
        kind="lds_race",
        severity=FindingSeverity.ERROR,
        message="LDS write/read race",
        locations=("kernel.cpp:10",),
    )
    result = ToolRunResult(
        tool="rocjitsu",
        execution=ExecutionStatus.COMPLETED,
        finding=FindingStatus.FOUND,
        findings=(finding,),
        execution_record=execution,
    )

    assert execution.returncode == 0
    assert result.finding == FindingStatus.FOUND
    assert result.findings_count == 1
    assert ToolRunResult.from_dict(result.to_dict(include_output=True)) == result


def test_finding_raw_output_is_only_serialized_when_explicitly_requested():
    finding = Finding(
        kind="oob",
        severity=FindingSeverity.ERROR,
        message="out of bounds",
        raw="very large parser excerpt",
    )
    result = ToolRunResult(
        tool="gpu_asan",
        execution=ExecutionStatus.COMPLETED,
        finding=FindingStatus.FOUND,
        findings=(finding,),
    )

    assert "raw" not in result.to_dict()["findings"][0]
    assert (
        result.to_dict(include_output=True)["findings"][0]["raw"]
        == "very large parser excerpt"
    )


def test_invalid_clean_and_found_results_are_rejected():
    finding = Finding("oob", FindingSeverity.ERROR, "out of bounds")
    with pytest.raises(ValueError, match="CLEAN"):
        ToolRunResult(
            tool="gpu_asan",
            execution=ExecutionStatus.COMPLETED,
            finding=FindingStatus.CLEAN,
            findings=(finding,),
        )
    with pytest.raises(ValueError, match="must contain"):
        ToolRunResult(
            tool="gpu_asan",
            execution=ExecutionStatus.COMPLETED,
            finding=FindingStatus.FOUND,
        )
