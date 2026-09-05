from __future__ import annotations

import pytest

from src.eval_tools.contracts import (
    ArtifactKind,
    CapabilityCheck,
    CapabilityState,
    InstrumentationControl,
    KernelLanguage,
)
from src.eval_tools.task_profile import resolve_builtin_capability, resolve_task_profile


@pytest.mark.parametrize(
    ("task_type", "language", "artifact", "control", "adapter"),
    [
        (
            "triton2triton",
            KernelLanguage.TRITON,
            ArtifactKind.PYTHON_JIT,
            InstrumentationControl.COMPILER_CONTROLLED,
            "triton_python_jit",
        ),
        (
            "instruction2triton",
            KernelLanguage.TRITON,
            ArtifactKind.PYTHON_JIT,
            InstrumentationControl.COMPILER_CONTROLLED,
            "triton_python_jit",
        ),
        (
            "torch2flydsl",
            KernelLanguage.FLYDSL,
            ArtifactKind.PYTHON_JIT,
            InstrumentationControl.NONE,
            "flydsl_python_jit",
        ),
        (
            "torch2hip",
            KernelLanguage.HIP,
            ArtifactKind.SOURCE_AOT,
            InstrumentationControl.RECOMPILE,
            "hip_source",
        ),
    ],
)
def test_isolated_task_profile_inference(
    task_type, language, artifact, control, adapter
):
    suffix = ".py" if language != KernelLanguage.HIP else ".hip"
    profile = resolve_task_profile(
        {
            "task_type": task_type,
            "source_file_path": [f"source/kernel{suffix}"],
            "target_kernel_functions": ["kernel"],
        }
    )
    assert profile.language == language
    assert profile.artifact_kind == artifact
    assert profile.instrumentation_control == control
    assert profile.adapter == adapter
    assert profile.source_available


def test_repository_language_is_authoritative_and_aiter_source_is_not_precompiled():
    profile = resolve_task_profile(
        {
            "task_type": "image_kernel",
            "repository_language": "triton",
            "image_repo_path": "/sgl-workspace/aiter",
            "source_file_path": [
                "aiter/ops/triton/_triton_kernels/gemm/basic/gemm.py"
            ],
        }
    )
    assert profile.language == KernelLanguage.TRITON
    assert profile.framework == "aiter"
    assert profile.artifact_kind == ArtifactKind.PYTHON_JIT
    assert profile.source_available


def test_hsaco_source_path_is_precompiled_even_for_hip_repository():
    profile = resolve_task_profile(
        {
            "task_type": "repository",
            "repository_language": "hip",
            "repo_url": "https://example.test/kernels.git",
            "source_file_path": ["build/kernel.hsaco"],
        }
    )
    assert profile.artifact_kind == ArtifactKind.HSACO_PRECOMPILED
    assert profile.instrumentation_control == InstrumentationControl.NONE
    assert not profile.source_available


def test_explicit_profile_override_wins_and_is_auditable():
    profile = resolve_task_profile(
        {
            "task_type": "image_kernel",
            "repository_language": "hip",
            "image_repo_path": "/sgl-workspace/aiter",
            "source_file_path": ["kernel.cu"],
            "evaluation_profile": {
                "language": "hip",
                "artifact_kind": "hsaco_precompiled",
                "framework": "custom-runtime",
                "instrumentation_control": "none",
                "adapter": "native_hsaco_launcher",
                "source_available": False,
                "submission_paths": ["kernel.cu", "capsule/kernel.hsaco"],
            },
        }
    )
    assert profile.artifact_kind == ArtifactKind.HSACO_PRECOMPILED
    assert profile.framework == "custom_runtime"
    assert profile.adapter == "native_hsaco_launcher"
    assert not profile.source_available
    assert set(profile.explicit_overrides) == {
        "language",
        "artifact_kind",
        "framework",
        "instrumentation_control",
        "adapter",
        "source_available",
        "submission_paths",
    }
    assert profile.evidence["submission_paths"] == [
        "kernel.cu",
        "capsule/kernel.hsaco",
    ]


def test_invalid_override_fails_early_instead_of_silently_guessing():
    with pytest.raises(ValueError, match="artifact_kind"):
        resolve_task_profile(
            {
                "task_type": "hip2hip",
                "source_file_path": ["kernel.hip"],
                "evaluation_profile": {"artifact_kind": "magic_binary"},
            }
        )
    with pytest.raises(ValueError, match="must be a mapping"):
        resolve_task_profile(
            {
                "task_type": "hip2hip",
                "source_file_path": ["kernel.hip"],
                "evaluation_profile": "hip",
            }
        )


def test_manual_instrumentation_evidence_is_explicit_and_auditable():
    profile = resolve_task_profile(
        {
            "task_type": "hip2hip",
            "source_file_path": ["kernel.hip"],
            "evaluation_profile": {
                "fpsan_ported": True,
                "rebuilt_from_source": True,
            },
        }
    )
    assert profile.evidence["fpsan_ported"] is True
    assert profile.evidence["rebuilt_from_source"] is True
    assert {"fpsan_ported", "rebuilt_from_source"} <= set(
        profile.explicit_overrides
    )


@pytest.mark.parametrize(
    ("tool", "adapter"),
    [
        ("rocjitsu_waitcheck", "waitcheck_code_object"),
        ("rocjitsu_consan", "consan_native"),
    ],
)
def test_native_rocjitsu_profiles_are_explicitly_ready(tool, adapter):
    profile = resolve_task_profile(
        {
            "task_type": "hip2hip",
            "source_file_path": ["kernel.hip"],
            "evaluation_profile": {"adapter": adapter},
        }
    )
    capability = resolve_builtin_capability(
        tool, profile, CapabilityCheck.ready(target_arch="gfx950")
    )
    assert capability.ready


def test_consan_broad_library_runtime_is_unsupported():
    profile = resolve_task_profile(
        {
            "task_type": "image_kernel",
            "repository_language": "hip",
            "image_repo_path": "/sgl-workspace/aiter",
            "source_file_path": ["kernel.hip"],
            "evaluation_profile": {"adapter": "consan_native"},
        }
    )
    capability = resolve_builtin_capability(
        "rocjitsu_consan", profile, CapabilityCheck.ready(target_arch="gfx950")
    )
    assert capability.engine.state == CapabilityState.UNSUPPORTED
    assert capability.engine.reason_code == "CONSAN_BROAD_LIBRARY_RUNTIME_UNSUPPORTED"
