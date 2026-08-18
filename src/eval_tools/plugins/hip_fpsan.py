"""HIP-FPSan explicit source-porting plugin."""

from __future__ import annotations

from pathlib import Path

from ..contracts import (
    ArtifactKind,
    CapabilityCheck,
    CapabilityState,
    KernelLanguage,
    ToolCapability,
    ToolContext,
    ToolInvocation,
    ToolName,
    ToolRunResult,
)
from .attestation import BuildAttestation
from .base import (
    artifact_path,
    blocked_check,
    command_from_context,
    parsed_to_run_result,
    ready_check,
    sidecar_path,
)
from .parsers import parse_fpsan_comparison


class HipFpSanPlugin:
    name = ToolName.HIP_FPSAN.value
    version = "2"

    def assess(self, context: ToolContext, runtime: CapabilityCheck) -> ToolCapability:
        profile = context.profile
        if profile.language in {KernelLanguage.TRITON, KernelLanguage.FLYDSL} or profile.framework in {
            "triton",
            "flydsl",
        }:
            engine = blocked_check(
                CapabilityState.NOT_APPLICABLE,
                "hip_fpsan_non_hip_language",
                "HIP-FPSan instruments explicitly ported HIP/C++ source only.",
            )
        elif profile.framework in {"aiter", "rocblas", "rccl"} or profile.artifact_kind == ArtifactKind.HSACO_PRECOMPILED:
            engine = blocked_check(
                CapabilityState.UNSUPPORTED,
                "hip_fpsan_precompiled_or_library_kernel",
                "HIP-FPSan cannot retrofit value semantics into an existing code object/library.",
            )
        elif profile.language != KernelLanguage.HIP:
            engine = blocked_check(
                CapabilityState.NOT_APPLICABLE,
                "hip_fpsan_language_not_supported",
                f"HIP-FPSan does not apply to language={profile.language.value}.",
            )
        else:
            engine = ready_check(explicit_source_port=True)

        ported = bool(profile.evidence.get("fpsan_ported"))
        has_compare = bool(context.options.get("comparison_command") or context.options.get("command"))
        if ported and has_compare:
            adapter = ready_check(adapter="hip_fpsan_compare")
        else:
            adapter = blocked_check(
                CapabilityState.ADAPTER_REQUIRED,
                "hip_fpsan_source_port_required",
                "Both kernel/reference paths must explicitly use fpsan::Value and expose a digest comparison.",
                source_ported=ported,
                comparison_configured=has_compare,
            )
        return ToolCapability(self.name, engine, adapter, runtime)

    @staticmethod
    def build_flags(include_dir: Path) -> tuple[str, ...]:
        # include_dir belongs to the sidecar namespace and must be attested by
        # its runtime health probe; it is intentionally not stat'ed here.
        return (f"-I{include_dir}", "-DAKA_HIP_FPSAN=1")

    def build_invocation(self, context: ToolContext) -> ToolInvocation:
        command = command_from_context(context, "comparison_command", "command")
        include_dir = sidecar_path(context, "include_dir", required=True)
        assert include_dir is not None
        flags = self.build_flags(include_dir)
        attestation_path = artifact_path(
            context, "attestation_path", "build_attestation.json"
        )
        attestation_path.parent.mkdir(parents=True, exist_ok=True)
        env = {
            **dict(context.env),
            "AKA_EVAL_TOOL": self.name,
            "AKA_HIP_FPSAN": "1",
            "FPSAN_INCLUDE_DIR": str(include_dir),
            "AKA_FPSAN_REQUIRE_COMPARISON": "1",
            "AKA_BUILD_ATTESTATION_PATH": str(attestation_path),
        }
        return ToolInvocation(
            tool=self.name,
            command=command,
            cwd=context.workspace,
            env=env,
            timeout_s=int(context.options.get("timeout_s", 600)),
            artifact_dir=context.artifact_dir,
            metadata={
                "build_flags": list(flags),
                "attestation_path": str(attestation_path),
            },
        )

    def parse(self, context: ToolContext, execution) -> ToolRunResult:
        include_dir = sidecar_path(context, "include_dir", required=True)
        assert include_dir is not None
        attestation_path = artifact_path(
            context, "attestation_path", "build_attestation.json"
        )
        attested = False
        if attestation_path.is_file():
            attestation = BuildAttestation.load(attestation_path)
            valid, _ = attestation.validate(
                expected_tool=self.name,
                required_flags=self.build_flags(include_dir),
                required_env={"AKA_HIP_FPSAN": "1"},
            )
            attested = valid and bool(attestation.evidence.get("reference_instrumented")) and bool(
                attestation.evidence.get("candidate_instrumented")
            )
        parsed = parse_fpsan_comparison(
            execution.stdout,
            execution.stderr,
            execution.returncode,
            attested=attested,
            timed_out=execution.timed_out,
        )
        artifacts = [attestation_path] if attestation_path.is_file() else []
        return parsed_to_run_result(self.name, parsed, execution, artifacts=artifacts)
