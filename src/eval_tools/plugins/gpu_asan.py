"""ROCm GPU AddressSanitizer plugin and build recipe."""

from __future__ import annotations

from pathlib import Path

from ..contracts import (
    ArtifactKind,
    CapabilityCheck,
    CapabilityState,
    InstrumentationControl,
    KernelLanguage,
    ToolCapability,
    ToolContext,
    ToolInvocation,
    ToolName,
    ToolRunResult,
)
from .attestation import BuildAttestation
from .base import (
    blocked_check,
    command_from_context,
    context_path,
    parsed_to_run_result,
    ready_check,
    sidecar_path,
)
from .parsers import parse_gpu_asan


GPU_ASAN_ENV = {
    "HSA_XNACK": "1",
    "HSA_DISABLE_FRAGMENT_ALLOCATOR": "1",
    "AMD_PYTORCH_NO_CUDA_MEMORY_CACHING": "1",
    "PYTORCH_NO_HIP_MEMORY_CACHING": "1",
    "AMDGCN_USE_BUFFER_OPS": "0",
    "ASAN_OPTIONS": "detect_leaks=0,alloc_dealloc_mismatch=0",
}


def _prepend_paths(*values: str | None, inherited: str = "") -> str:
    paths: list[str] = []
    for value in (*values, inherited):
        for item in str(value or "").split(":"):
            if item and item not in paths:
                paths.append(item)
    return ":".join(paths)


def hip_asan_build_flags(target_arch: str) -> tuple[str, ...]:
    arch = target_arch.split(":", 1)[0].strip()
    if not arch.startswith("gfx"):
        raise ValueError(f"invalid AMD GPU architecture: {target_arch!r}")
    return ("-fsanitize=address", "-shared-libsan", f"--offload-arch={arch}:xnack+")


class GpuAsanPlugin:
    name = ToolName.GPU_ASAN.value
    version = "1"

    def assess(self, context: ToolContext, runtime: CapabilityCheck) -> ToolCapability:
        profile = context.profile
        is_triton = profile.language == KernelLanguage.TRITON or profile.framework == "triton"
        is_hip = profile.language == KernelLanguage.HIP

        if profile.language == KernelLanguage.FLYDSL or profile.framework == "flydsl":
            engine = blocked_check(
                CapabilityState.UNSUPPORTED,
                "gpu_asan_flydsl_no_device_instrumentation",
                "FlyDSL 0.2.x does not insert the AMDGPU AddressSanitizer pass.",
            )
        elif profile.framework in {"aiter", "rocblas", "rccl"} or profile.artifact_kind == ArtifactKind.HSACO_PRECOMPILED:
            # Explicit source-rebuild evidence may make a future AITER HIP lane
            # eligible, but the default classification must remain fail closed.
            rebuilt = bool(profile.evidence.get("rebuilt_from_source"))
            if not (is_hip and profile.source_available and rebuilt):
                engine = blocked_check(
                    CapabilityState.UNSUPPORTED,
                    "gpu_asan_precompiled_code_object",
                    "The code object was not rebuilt from source with GPU ASan instrumentation.",
                )
            else:
                engine = ready_check(rebuilt_from_source=True)
        elif not (is_triton or is_hip):
            engine = blocked_check(
                CapabilityState.NOT_APPLICABLE,
                "gpu_asan_language_not_supported",
                f"GPU ASan adapter does not support language={profile.language.value}.",
            )
        elif is_hip and profile.instrumentation_control not in {
            InstrumentationControl.RECOMPILE,
            InstrumentationControl.COMPILER_CONTROLLED,
        }:
            engine = blocked_check(
                CapabilityState.UNSUPPORTED,
                "gpu_asan_recompile_not_controlled",
                "HIP GPU ASan requires control of the source compilation command.",
            )
        else:
            engine = ready_check(xnack_required=True)

        command_configured = bool(context.options.get("command"))
        adapter = (
            ready_check(adapter="triton_gpu_asan" if is_triton else "hip_source_rebuild")
            if command_configured
            else blocked_check(
                CapabilityState.ADAPTER_REQUIRED,
                "gpu_asan_command_missing",
                "A dedicated ASan build/run argv must be configured.",
            )
        )

        evidence = dict(runtime.evidence)
        if runtime.state == CapabilityState.READY and evidence.get("xnack_supported") is False:
            runtime = blocked_check(
                CapabilityState.UNAVAILABLE_RUNTIME,
                "gpu_asan_xnack_unavailable",
                "GPU ASan requires an xnack+ code object and xnack-enabled runtime.",
                **evidence,
            )
        elif runtime.state == CapabilityState.READY and is_triton and evidence.get("triton_asan") is False:
            runtime = blocked_check(
                CapabilityState.UNAVAILABLE_RUNTIME,
                "gpu_asan_triton_pass_missing",
                "The selected Triton build does not expose TRITON_ENABLE_ASAN.",
                **evidence,
            )
        return ToolCapability(self.name, engine, adapter, runtime)

    def build_invocation(self, context: ToolContext) -> ToolInvocation:
        command = command_from_context(context, "command")
        profile = context.profile
        is_triton = profile.language == KernelLanguage.TRITON or profile.framework == "triton"
        artifact_dir = Path(context.artifact_dir)
        env = {**dict(context.env), **GPU_ASAN_ENV, "AKA_EVAL_TOOL": self.name}
        if is_triton:
            cache = artifact_dir / "triton-gpu-asan-cache"
            cache.mkdir(parents=True, exist_ok=True)
            env.update({"TRITON_ENABLE_ASAN": "1", "TRITON_CACHE_DIR": str(cache)})
        # GPU ASan needs both the host compiler-rt runtime and ROCm's ASan-built
        # HIP/HSA/COMGR libraries.  The paths are attested by the sidecar health
        # probe and intentionally never stat'ed in the scoring container.
        host_runtime = sidecar_path(context, "host_asan_preload")
        hip_runtime = sidecar_path(context, "hip_asan_runtime")
        runtime_dir = sidecar_path(context, "asan_runtime_dir")
        host_library_dir = sidecar_path(context, "host_asan_lib_dir")
        normal_rocm_library_dir = sidecar_path(context, "normal_rocm_lib_dir")
        if host_runtime is not None and host_library_dir is None:
            host_library_dir = host_runtime.parent
        env["LD_LIBRARY_PATH"] = _prepend_paths(
            str(host_library_dir) if host_library_dir else None,
            str(runtime_dir) if runtime_dir else None,
            str(normal_rocm_library_dir) if normal_rocm_library_dir else None,
            inherited=env.get("LD_LIBRARY_PATH", ""),
        )
        if is_triton:
            if host_runtime is None or hip_runtime is None:
                raise ValueError(
                    "Triton GPU ASan requires attested host_asan_preload and "
                    "hip_asan_runtime sidecar paths"
                )
            env["LD_PRELOAD"] = _prepend_paths(
                str(host_runtime),
                str(hip_runtime),
                inherited=env.get("LD_PRELOAD", ""),
            )
        env["AKA_BUILD_ATTESTATION_PATH"] = str(
            artifact_dir / "build_attestation.json"
        )
        return ToolInvocation(
            tool=self.name,
            command=command,
            cwd=context.workspace,
            env=env,
            timeout_s=int(context.options.get("timeout_s", 600)),
            artifact_dir=context.artifact_dir,
            metadata={
                "build_flags": list(hip_asan_build_flags(context.gpu_arch or "gfx950")) if not is_triton else [],
                "attestation_path": str(artifact_dir / "build_attestation.json"),
            },
        )

    def parse(self, context: ToolContext, execution) -> ToolRunResult:
        profile = context.profile
        is_triton = profile.language == KernelLanguage.TRITON or profile.framework == "triton"
        attestation_path = context_path(context, "attestation_path") or (
            Path(context.artifact_dir) / "build_attestation.json"
        )
        attested = False
        if attestation_path.is_file():
            attestation = BuildAttestation.load(attestation_path)
            if is_triton:
                valid, _ = attestation.validate(
                    expected_tool=self.name,
                    required_env={"TRITON_ENABLE_ASAN": "1", "HSA_XNACK": "1"},
                )
            else:
                valid, _ = attestation.validate(
                    expected_tool=self.name,
                    required_flags=hip_asan_build_flags(context.gpu_arch or "gfx950"),
                    required_env={"HSA_XNACK": "1"},
                )
            attested = valid
        parsed = parse_gpu_asan(
            execution.stdout,
            execution.stderr,
            execution.returncode,
            attested=attested,
            timed_out=execution.timed_out,
        )
        artifacts = [attestation_path] if attestation_path.is_file() else []
        return parsed_to_run_result(self.name, parsed, execution, artifacts=artifacts)
