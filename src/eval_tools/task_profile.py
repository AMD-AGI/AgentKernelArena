"""Resolve task execution profiles and built-in tool capabilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .contracts import (
    ArtifactKind,
    CapabilityCheck,
    CapabilityState,
    InstrumentationControl,
    KernelLanguage,
    TaskProfile,
    ToolCapability,
    ToolName,
)


_LANGUAGE_ALIASES = {
    "triton": KernelLanguage.TRITON,
    "hip": KernelLanguage.HIP,
    "rocm": KernelLanguage.HIP,
    "cuda": KernelLanguage.HIP,
    "flydsl": KernelLanguage.FLYDSL,
    "unknown": KernelLanguage.UNKNOWN,
}

_ARTIFACT_ALIASES = {
    "source": ArtifactKind.SOURCE_AOT,
    "source_aot": ArtifactKind.SOURCE_AOT,
    "native_source": ArtifactKind.SOURCE_AOT,
    "python_jit": ArtifactKind.PYTHON_JIT,
    "jit_python": ArtifactKind.PYTHON_JIT,
    "jit": ArtifactKind.PYTHON_JIT,
    "hsaco": ArtifactKind.HSACO_PRECOMPILED,
    "precompiled": ArtifactKind.HSACO_PRECOMPILED,
    "hsaco_precompiled": ArtifactKind.HSACO_PRECOMPILED,
    "unknown": ArtifactKind.UNKNOWN,
}

_CONTROL_ALIASES = {
    "compiler_controlled": InstrumentationControl.COMPILER_CONTROLLED,
    "compiler": InstrumentationControl.COMPILER_CONTROLLED,
    "recompile": InstrumentationControl.RECOMPILE,
    "rebuild": InstrumentationControl.RECOMPILE,
    "none": InstrumentationControl.NONE,
    "unknown": InstrumentationControl.UNKNOWN,
}


def _string_list(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value if item is not None)
    raise ValueError(f"expected string or list of strings, got {type(value).__name__}")


def _explicit_enum(
    value: Any,
    aliases: Mapping[str, Any],
    field_name: str,
):
    key = str(value).strip().lower().replace("-", "_")
    if key not in aliases:
        raise ValueError(
            f"invalid evaluation_profile.{field_name}={value!r}; "
            f"expected one of {sorted(aliases)}"
        )
    return aliases[key]


def _as_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    raise ValueError(f"evaluation_profile.{field_name} must be a boolean")


def _infer_language(task_type: str, task_config: Mapping[str, Any]) -> KernelLanguage:
    if task_type in {"repository", "image_kernel"}:
        candidate = str(task_config.get("repository_language") or "").strip().lower()
    elif "2" in task_type:
        candidate = task_type.rsplit("2", 1)[-1]
    elif task_type == "instruction2triton":
        candidate = "triton"
    else:
        candidate = task_type
    return _LANGUAGE_ALIASES.get(candidate, KernelLanguage.UNKNOWN)


def _infer_artifact_kind(
    language: KernelLanguage,
    source_files: tuple[str, ...],
) -> ArtifactKind:
    suffixes = {Path(path).suffix.lower() for path in source_files}
    if suffixes and suffixes <= {".hsaco", ".co"}:
        return ArtifactKind.HSACO_PRECOMPILED
    if language in {KernelLanguage.TRITON, KernelLanguage.FLYDSL}:
        return ArtifactKind.PYTHON_JIT
    if language == KernelLanguage.HIP:
        return ArtifactKind.SOURCE_AOT
    return ArtifactKind.UNKNOWN


def _infer_framework(
    task_config: Mapping[str, Any], source_files: tuple[str, ...]
) -> str:
    evidence = [*source_files]
    for key in ("image_repo_path", "repo_url", "repo_subdir"):
        if task_config.get(key):
            evidence.append(str(task_config[key]))
    joined = " ".join(evidence).lower()
    for framework in ("aiter", "sglang", "rocblas", "rccl"):
        if framework in joined:
            return framework
    return "standalone"


def _infer_instrumentation_control(
    language: KernelLanguage, artifact_kind: ArtifactKind
) -> InstrumentationControl:
    if artifact_kind == ArtifactKind.HSACO_PRECOMPILED:
        return InstrumentationControl.NONE
    if language == KernelLanguage.TRITON and artifact_kind == ArtifactKind.PYTHON_JIT:
        return InstrumentationControl.COMPILER_CONTROLLED
    if language == KernelLanguage.HIP and artifact_kind == ArtifactKind.SOURCE_AOT:
        return InstrumentationControl.RECOMPILE
    # Current FlyDSL cannot be assumed to expose arbitrary compiler
    # instrumentation even though it does JIT-compile source.
    if language == KernelLanguage.FLYDSL:
        return InstrumentationControl.NONE
    return InstrumentationControl.UNKNOWN


def _infer_adapter(language: KernelLanguage, artifact_kind: ArtifactKind) -> str | None:
    if artifact_kind == ArtifactKind.HSACO_PRECOMPILED:
        return "precompiled"
    if language == KernelLanguage.HIP and artifact_kind == ArtifactKind.SOURCE_AOT:
        return "hip_source"
    if language == KernelLanguage.TRITON and artifact_kind == ArtifactKind.PYTHON_JIT:
        return "triton_python_jit"
    if language == KernelLanguage.FLYDSL and artifact_kind == ArtifactKind.PYTHON_JIT:
        return "flydsl_python_jit"
    return None


def resolve_task_profile(task_config: Mapping[str, Any]) -> TaskProfile:
    """Infer a task profile, then apply a validated ``evaluation_profile`` override.

    Inference deliberately describes the submission rather than promising that
    a particular tool supports it.  For example an AITER repository containing
    editable Triton source remains ``python_jit``; it is not mislabeled as a
    precompiled AITER operator merely because "aiter" appears in the path.
    """

    if not isinstance(task_config, Mapping):
        raise TypeError("task_config must be a mapping")
    task_type = str(task_config.get("task_type") or "").strip().lower()
    source_files = _string_list(task_config.get("source_file_path"))
    target_functions = _string_list(task_config.get("target_kernel_functions"))

    language = _infer_language(task_type, task_config)
    artifact_kind = _infer_artifact_kind(language, source_files)
    framework = _infer_framework(task_config, source_files)
    control = _infer_instrumentation_control(language, artifact_kind)
    adapter = _infer_adapter(language, artifact_kind)
    source_available = bool(source_files) and artifact_kind != ArtifactKind.HSACO_PRECOMPILED

    override = task_config.get("evaluation_profile") or {}
    if not isinstance(override, Mapping):
        raise ValueError("evaluation_profile must be a mapping when present")
    explicit: list[str] = []

    if "language" in override:
        language = _explicit_enum(override["language"], _LANGUAGE_ALIASES, "language")
        explicit.append("language")
    if "artifact_kind" in override:
        artifact_kind = _explicit_enum(
            override["artifact_kind"], _ARTIFACT_ALIASES, "artifact_kind"
        )
        explicit.append("artifact_kind")
    if "framework" in override:
        framework = str(override["framework"]).strip().lower().replace("-", "_")
        if not framework:
            raise ValueError("evaluation_profile.framework cannot be empty")
        explicit.append("framework")
    if "instrumentation_control" in override:
        control = _explicit_enum(
            override["instrumentation_control"],
            _CONTROL_ALIASES,
            "instrumentation_control",
        )
        explicit.append("instrumentation_control")
    if "adapter" in override:
        adapter_value = override["adapter"]
        adapter = str(adapter_value).strip().lower() if adapter_value is not None else None
        adapter = adapter or None
        explicit.append("adapter")
    if "source_available" in override:
        source_available = _as_bool(override["source_available"], "source_available")
        explicit.append("source_available")

    submission_paths = None
    if "submission_paths" in override:
        submission_paths = _string_list(override["submission_paths"])
        # Path containment is enforced authoritatively by evidence capture; the
        # profile still records that the operator supplied an explicit boundary.
        explicit.append("submission_paths")

    evidence = {
        "repository_language": task_config.get("repository_language"),
        "image_repo_path": task_config.get("image_repo_path"),
        "repo_url": task_config.get("repo_url"),
        "submission_paths": list(submission_paths) if submission_paths is not None else None,
    }
    for evidence_flag in ("fpsan_ported", "rebuilt_from_source"):
        if evidence_flag in override:
            evidence[evidence_flag] = _as_bool(
                override[evidence_flag], evidence_flag
            )
            explicit.append(evidence_flag)

    return TaskProfile(
        task_type=task_type,
        language=language,
        artifact_kind=artifact_kind,
        framework=framework,
        instrumentation_control=control,
        adapter=adapter,
        source_available=source_available,
        source_files=source_files,
        target_functions=target_functions,
        explicit_overrides=tuple(explicit),
        evidence=evidence,
    )


def _not_applicable(tool: str, reason: str) -> ToolCapability:
    return ToolCapability(
        tool=tool,
        engine=CapabilityCheck.blocked(
            CapabilityState.NOT_APPLICABLE, reason, "tool does not apply to this kernel language"
        ),
        adapter=CapabilityCheck.ready(),
        runtime=CapabilityCheck.ready(),
    )


def resolve_builtin_capability(
    tool: str | ToolName,
    profile: TaskProfile,
    runtime: CapabilityCheck | None = None,
) -> ToolCapability:
    """Return the conservative support matrix established by end-to-end probes.

    Concrete plugins may add version/architecture checks, but they should not
    broaden these defaults without new evidence.  Runtime availability is an
    independent input so a missing image cannot be confused with an unsupported
    compiler/adapter.
    """

    tool_name = tool.value if isinstance(tool, ToolName) else str(tool).lower().replace("-", "_")
    runtime_check = runtime or CapabilityCheck.ready()
    ready = CapabilityCheck.ready

    if tool_name == ToolName.TRITON_FPSAN.value:
        if profile.language != KernelLanguage.TRITON:
            capability = _not_applicable(tool_name, "FPSAN_NON_TRITON")
        elif (
            profile.artifact_kind == ArtifactKind.PYTHON_JIT
            and profile.instrumentation_control == InstrumentationControl.COMPILER_CONTROLLED
        ):
            capability = ToolCapability(tool_name, ready(), ready(), runtime_check)
        else:
            capability = ToolCapability(
                tool_name,
                CapabilityCheck.blocked(
                    CapabilityState.UNSUPPORTED,
                    "FPSAN_REQUIRES_TRITON_RECOMPILE",
                    "Triton FpSan cannot instrument a precompiled/non-controlled artifact",
                ),
                ready(),
                runtime_check,
            )
    elif tool_name == ToolName.GPU_ASAN.value:
        if profile.artifact_kind == ArtifactKind.HSACO_PRECOMPILED:
            capability = ToolCapability(
                tool_name,
                CapabilityCheck.blocked(
                    CapabilityState.UNSUPPORTED,
                    "GPU_ASAN_PRECOMPILED_HSACO",
                    "ASan runtime cannot inspect HSACO that was not instrumented at compile time",
                ),
                ready(),
                runtime_check,
            )
        elif profile.language == KernelLanguage.FLYDSL:
            capability = ToolCapability(
                tool_name,
                CapabilityCheck.blocked(
                    CapabilityState.UNSUPPORTED,
                    "GPU_ASAN_FLYDSL_PIPELINE",
                    "current FlyDSL ROCDL pipeline does not run AMD GPU ASan instrumentation",
                ),
                ready(),
                runtime_check,
            )
        elif (
            profile.language == KernelLanguage.TRITON
            and profile.artifact_kind == ArtifactKind.PYTHON_JIT
            and profile.instrumentation_control == InstrumentationControl.COMPILER_CONTROLLED
        ):
            capability = ToolCapability(tool_name, ready(), ready(), runtime_check)
        elif (
            profile.language == KernelLanguage.HIP
            and profile.artifact_kind == ArtifactKind.SOURCE_AOT
            and profile.source_available
            and profile.instrumentation_control == InstrumentationControl.RECOMPILE
        ):
            capability = ToolCapability(tool_name, ready(), ready(), runtime_check)
        else:
            capability = ToolCapability(
                tool_name,
                CapabilityCheck.blocked(
                    CapabilityState.UNSUPPORTED,
                    "GPU_ASAN_NO_INSTRUMENTABLE_SOURCE",
                    "no supported compiler-controlled source path is available",
                ),
                ready(),
                runtime_check,
            )
    elif tool_name == ToolName.ROCJITSU.value:
        if profile.artifact_kind == ArtifactKind.HSACO_PRECOMPILED:
            capability = ToolCapability(
                tool_name,
                ready(),
                CapabilityCheck.blocked(
                    CapabilityState.UNSUPPORTED,
                    "ROCJITSU_PRECOMPILED_RUNTIME_NOT_VALIDATED",
                    "arbitrary/AITER precompiled Python runtime is not a supported replay path",
                ),
                runtime_check,
            )
        elif profile.language == KernelLanguage.HIP and profile.artifact_kind == ArtifactKind.SOURCE_AOT:
            capability = ToolCapability(tool_name, ready(), ready(), runtime_check)
        elif profile.language in {KernelLanguage.TRITON, KernelLanguage.FLYDSL}:
            # The concrete plugin accepts only the language-specific trusted
            # capsule adapter. Generic launcher aliases would overstate support
            # because they do not prove the lowered ABI/capture contract.
            expected = {f"{profile.language.value}_aot"}
            adapter_check = (
                ready()
                if profile.adapter in expected
                else CapabilityCheck.blocked(
                    CapabilityState.ADAPTER_REQUIRED,
                    f"ROCJITSU_{profile.language.value.upper()}_AOT_ADAPTER_REQUIRED",
                    "HSACO engine support exists, but the Python JIT evaluator cannot be wrapped directly",
                )
            )
            capability = ToolCapability(tool_name, ready(), adapter_check, runtime_check)
        else:
            capability = _not_applicable(tool_name, "ROCJITSU_NON_GPU_KERNEL")
    elif tool_name == ToolName.HIP_FPSAN.value:
        if profile.language != KernelLanguage.HIP:
            capability = _not_applicable(tool_name, "HIP_FPSAN_NON_HIP")
        elif profile.artifact_kind != ArtifactKind.SOURCE_AOT or not profile.source_available:
            capability = ToolCapability(
                tool_name,
                CapabilityCheck.blocked(
                    CapabilityState.UNSUPPORTED,
                    "HIP_FPSAN_SOURCE_REQUIRED",
                    "HIP-FpSan is a source library and cannot wrap precompiled code",
                ),
                ready(),
                runtime_check,
            )
        elif profile.adapter in {"hip_fpsan", "hip_fpsan_manual", "fpsan_value_source"}:
            capability = ToolCapability(tool_name, ready(), ready(), runtime_check)
        else:
            capability = ToolCapability(
                tool_name,
                ready(),
                CapabilityCheck.blocked(
                    CapabilityState.ADAPTER_REQUIRED,
                    "HIP_FPSAN_MANUAL_SOURCE_ADAPTER_REQUIRED",
                    "source operations must be explicitly ported to fpsan::Value",
                ),
                runtime_check,
            )
    else:
        raise KeyError(f"no built-in capability matrix for tool {tool_name!r}")

    # _not_applicable constructs an always-ready runtime; replace it with the
    # actual runtime dimension while preserving the semantic engine decision.
    if capability.runtime != runtime_check:
        capability = ToolCapability(
            capability.tool, capability.engine, capability.adapter, runtime_check
        )
    return capability


__all__ = ["resolve_builtin_capability", "resolve_task_profile"]
