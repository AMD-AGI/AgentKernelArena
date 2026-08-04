"""Triton FPSan semantic-comparison plugin."""

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
    blocked_check,
    command_from_context,
    context_path,
    parsed_to_run_result,
    ready_check,
)
from .parsers import parse_fpsan_comparison


class TritonFpSanPlugin:
    name = ToolName.TRITON_FPSAN.value
    version = "1"

    def assess(self, context: ToolContext, runtime: CapabilityCheck) -> ToolCapability:
        profile = context.profile
        if profile.language == KernelLanguage.FLYDSL or profile.framework == "flydsl":
            engine = blocked_check(
                CapabilityState.UNSUPPORTED,
                "triton_fpsan_flydsl_pipeline_unavailable",
                "FlyDSL does not compile through Triton's FPSan pass pipeline.",
            )
        elif profile.language != KernelLanguage.TRITON and profile.framework != "triton":
            engine = blocked_check(
                CapabilityState.NOT_APPLICABLE,
                "triton_fpsan_non_triton_task",
                f"Triton FPSan does not apply to language={profile.language.value}.",
            )
        elif profile.artifact_kind == ArtifactKind.HSACO_PRECOMPILED:
            engine = blocked_check(
                CapabilityState.UNSUPPORTED,
                "triton_fpsan_precompiled_code_object",
                "FPSan must run in the Triton compiler; an existing HSACO cannot be instrumented.",
            )
        else:
            engine = ready_check(instrumentation_mode="fpsan")

        has_compare = bool(context.options.get("comparison_command") or context.options.get("command"))
        adapter = (
            ready_check(adapter="triton_fpsan_compare")
            if has_compare
            else blocked_check(
                CapabilityState.ADAPTER_REQUIRED,
                "triton_fpsan_comparison_harness_missing",
                "A reference/candidate FPSan digest comparison command must be configured.",
            )
        )
        if runtime.state == CapabilityState.READY and runtime.evidence.get("triton_fpsan") is False:
            runtime = blocked_check(
                CapabilityState.UNAVAILABLE_RUNTIME,
                "triton_fpsan_runtime_missing",
                "The selected Triton build does not attest FPSan instrumentation support.",
                **dict(runtime.evidence),
            )
        return ToolCapability(self.name, engine, adapter, runtime)

    def build_invocation(self, context: ToolContext) -> ToolInvocation:
        command = command_from_context(context, "comparison_command", "command")
        artifact_dir = Path(context.artifact_dir)
        cache_dir = artifact_dir / "triton-fpsan-cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        env = {
            **dict(context.env),
            "TRITON_INSTRUMENTATION_MODE": "fpsan",
            "TRITON_CACHE_DIR": str(cache_dir),
            "AKA_EVAL_TOOL": self.name,
            "AKA_FPSAN_REQUIRE_COMPARISON": "1",
            "AKA_BUILD_ATTESTATION_PATH": str(
                artifact_dir / "build_attestation.json"
            ),
        }
        return ToolInvocation(
            tool=self.name,
            command=command,
            cwd=context.workspace,
            env=env,
            timeout_s=int(context.options.get("timeout_s", 600)),
            artifact_dir=context.artifact_dir,
            metadata={"attestation_path": str(artifact_dir / "build_attestation.json")},
        )

    def parse(self, context: ToolContext, execution) -> ToolRunResult:
        attestation_path = context_path(context, "attestation_path") or (
            Path(context.artifact_dir) / "build_attestation.json"
        )
        attested = False
        if attestation_path.is_file():
            attestation = BuildAttestation.load(attestation_path)
            valid, _ = attestation.validate(
                expected_tool=self.name,
                required_env={"TRITON_INSTRUMENTATION_MODE": "fpsan"},
                require_artifact=True,
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
