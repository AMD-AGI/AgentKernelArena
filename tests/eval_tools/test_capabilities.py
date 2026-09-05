from __future__ import annotations

import pytest

from src.eval_tools.contracts import CapabilityCheck, CapabilityState
from src.eval_tools.task_profile import (
    resolve_builtin_capability,
    resolve_task_profile,
)


def _profile(task_type, source, **extra):
    config = {"task_type": task_type, "source_file_path": [source], **extra}
    return resolve_task_profile(config)


def test_triton_ready_tools_and_rocjitsu_requires_aot_adapter():
    profile = _profile("triton2triton", "kernel.py")
    assert resolve_builtin_capability("triton_fpsan", profile).ready
    assert resolve_builtin_capability("gpu_asan", profile).ready

    rocjitsu = resolve_builtin_capability("rocjitsu", profile)
    assert rocjitsu.engine.state == CapabilityState.READY
    assert rocjitsu.adapter.state == CapabilityState.ADAPTER_REQUIRED
    assert rocjitsu.effective.state == CapabilityState.ADAPTER_REQUIRED


def test_explicit_triton_aot_adapter_enables_rocjitsu():
    profile = resolve_task_profile(
        {
            "task_type": "triton2triton",
            "source_file_path": ["kernel.py"],
            "evaluation_profile": {"adapter": "triton_aot"},
        }
    )
    assert resolve_builtin_capability("rocjitsu", profile).ready


def test_generic_native_launcher_alias_does_not_claim_triton_aot_support():
    profile = resolve_task_profile(
        {
            "task_type": "triton2triton",
            "source_file_path": ["kernel.py"],
            "evaluation_profile": {"adapter": "native_hsaco_launcher"},
        }
    )
    assert (
        resolve_builtin_capability("rocjitsu", profile).adapter.state
        == CapabilityState.ADAPTER_REQUIRED
    )


def test_flydsl_asan_is_unsupported_but_rocjitsu_engine_support_is_retained():
    profile = _profile("torch2flydsl", "kernel.py")
    asan = resolve_builtin_capability("gpu_asan", profile)
    race = resolve_builtin_capability("rocjitsu", profile)

    assert asan.engine.state == CapabilityState.UNSUPPORTED
    assert asan.effective.reason_code == "GPU_ASAN_FLYDSL_PIPELINE"
    assert race.engine.state == CapabilityState.READY
    assert race.adapter.state == CapabilityState.ADAPTER_REQUIRED


def test_hip_source_asan_and_rocjitsu_are_ready_but_hip_fpsan_is_manual():
    profile = _profile("hip2hip", "kernel.hip")
    assert resolve_builtin_capability("gpu_asan", profile).ready
    assert resolve_builtin_capability("rocjitsu", profile).ready
    hip_fpsan = resolve_builtin_capability("hip_fpsan", profile)
    assert hip_fpsan.engine.state == CapabilityState.READY
    assert hip_fpsan.adapter.state == CapabilityState.ADAPTER_REQUIRED


def test_manual_hip_fpsan_override_marks_adapter_ready():
    profile = resolve_task_profile(
        {
            "task_type": "hip2hip",
            "source_file_path": ["kernel.hip"],
            "evaluation_profile": {"adapter": "hip_fpsan_manual"},
        }
    )
    assert resolve_builtin_capability("hip_fpsan", profile).ready


def test_precompiled_hsaco_is_not_mistaken_for_instrumented_code():
    profile = resolve_task_profile(
        {
            "task_type": "repository",
            "repository_language": "hip",
            "source_file_path": ["kernel.hsaco"],
        }
    )
    assert (
        resolve_builtin_capability("gpu_asan", profile).effective.state
        == CapabilityState.UNSUPPORTED
    )
    assert (
        resolve_builtin_capability("rocjitsu", profile).adapter.state
        == CapabilityState.UNSUPPORTED
    )


def test_runtime_unavailability_is_separate_and_does_not_override_not_applicable():
    runtime = CapabilityCheck.blocked(
        CapabilityState.UNAVAILABLE_RUNTIME, "IMAGE_NOT_PRESENT"
    )
    triton = _profile("triton2triton", "kernel.py")
    hip = _profile("hip2hip", "kernel.hip")

    unavailable = resolve_builtin_capability("gpu_asan", triton, runtime)
    not_applicable = resolve_builtin_capability("triton_fpsan", hip, runtime)
    assert unavailable.engine.state == CapabilityState.READY
    assert unavailable.runtime.state == CapabilityState.UNAVAILABLE_RUNTIME
    assert unavailable.effective.state == CapabilityState.UNAVAILABLE_RUNTIME
    assert not_applicable.effective.state == CapabilityState.NOT_APPLICABLE


def test_unknown_tool_has_no_accidental_default_support():
    with pytest.raises(KeyError, match="no built-in"):
        resolve_builtin_capability("future_tool", _profile("hip2hip", "kernel.hip"))
