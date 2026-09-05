"""Strict rocJITsu ConSan plugin for native gfx950 launchers."""

from __future__ import annotations

from pathlib import Path

from ..adapters.native_code_object import (
    command_references_file,
    fnv1a64_file,
    resolve_workspace_file,
    sha256_file,
)
from ..contracts import (
    CapabilityCheck,
    CapabilityState,
    KernelLanguage,
    ToolCapability,
    ToolContext,
    ToolInvocation,
    ToolName,
    ToolRunResult,
)
from .base import (
    blocked_check,
    command_from_context,
    parsed_to_run_result,
    ready_check,
    sidecar_path,
)
from .parsers import parse_consan


_ENTRYPOINT = "/opt/aka-eval-tools/src/eval_tools/adapters/consan_entrypoint.py"
_PYTHON = "/opt/venv/bin/python"


def _commands(context: ToolContext) -> tuple[tuple[str, ...], tuple[str, ...]]:
    command = command_from_context(context, "command")
    oracle = command_from_context(context, "oracle_command")
    code_object = resolve_workspace_file(context, "code_object")
    # Use the actual task workspace rather than the code object's directory for
    # relative launcher arguments.
    workspace = Path(context.workspace)
    if not command_references_file(command, code_object, workspace):
        raise ValueError("ConSan command argv must name the selected code_object")
    return command, oracle


class ConSanPlugin:
    name = ToolName.ROCJITSU_CONSAN.value
    version = "1"

    def assess(self, context: ToolContext, runtime: CapabilityCheck) -> ToolCapability:
        profile = context.profile
        if profile.language not in {
            KernelLanguage.HIP,
            KernelLanguage.TRITON,
            KernelLanguage.FLYDSL,
        }:
            engine = blocked_check(
                CapabilityState.NOT_APPLICABLE,
                "consan_non_amdgpu_language",
                "ConSan instruments final AMDGPU code objects only.",
            )
        elif profile.framework in {"aiter", "rocblas", "rccl"}:
            engine = blocked_check(
                CapabilityState.UNSUPPORTED,
                "consan_broad_library_runtime_unsupported",
                "The first integration supports focused native launchers, not broad library runtimes.",
            )
        elif context.gpu_arch != "gfx950":
            engine = blocked_check(
                CapabilityState.UNSUPPORTED,
                "consan_gfx950_not_selected",
                "This integration is qualified only for gfx950.",
                observed_gpu_arch=context.gpu_arch,
            )
        else:
            engine = ready_check(
                target="gfx950", mode="record-replay", policy="strict"
            )

        try:
            code_object = resolve_workspace_file(context, "code_object")
            command, oracle = _commands(context)
        except (OSError, ValueError) as error:
            adapter = blocked_check(
                CapabilityState.ADAPTER_REQUIRED,
                "consan_native_adapter_required",
                "ConSan requires an exact HSACO, a launcher that names it, and a "
                f"separate correctness-oracle argv ({error}).",
            )
        else:
            adapter = ready_check(
                adapter="consan_native",
                code_object=str(code_object),
                code_object_sha256=sha256_file(code_object),
                code_object_fingerprint=fnv1a64_file(code_object),
                command=list(command),
                oracle_command=list(oracle),
            )
        if runtime.state == CapabilityState.READY and runtime.evidence.get(
            "gpu_arch"
        ) not in {"gfx950", "gfx950:xnack+", "gfx950:xnack-"}:
            runtime = blocked_check(
                CapabilityState.UNAVAILABLE_RUNTIME,
                "consan_runtime_gfx950_missing",
                "ConSan sidecar did not attest a reachable gfx950 agent.",
                **dict(runtime.evidence),
            )
        return ToolCapability(self.name, engine, adapter, runtime)

    def build_invocation(self, context: ToolContext) -> ToolInvocation:
        code_object = resolve_workspace_file(context, "code_object")
        code_sha256 = sha256_file(code_object)
        fingerprint = fnv1a64_file(code_object)
        command, oracle = _commands(context)
        hook = sidecar_path(context, "consan_hook", required=True)
        assert hook is not None
        argv: list[str] = [
            _PYTHON,
            "-I",
            _ENTRYPOINT,
            "--hook",
            str(hook),
            "--code-object",
            str(code_object),
            "--expected-sha256",
            code_sha256,
            "--expected-fingerprint",
            fingerprint,
            "--mode",
            "record-replay",
        ]
        for argument in command:
            argv.append(f"--command-arg={argument}")
        for argument in oracle:
            argv.append(f"--oracle-arg={argument}")
        return ToolInvocation(
            tool=self.name,
            command=tuple(argv),
            cwd=context.workspace,
            env={**dict(context.env), "AKA_EVAL_TOOL": self.name},
            timeout_s=int(context.options.get("timeout_s", 600)),
            artifact_dir=context.artifact_dir,
            metadata={
                "code_object": str(code_object),
                "code_object_sha256": code_sha256,
                "code_object_fingerprint": fingerprint,
                "mode": "record-replay",
                "policy": "strict",
                "oracle_configured": True,
            },
        )

    def parse(self, context: ToolContext, execution) -> ToolRunResult:
        code_object = resolve_workspace_file(context, "code_object")
        code_sha256 = sha256_file(code_object)
        fingerprint = fnv1a64_file(code_object)
        parsed = parse_consan(
            execution.stdout,
            execution.stderr,
            execution.returncode,
            expected_sha256=code_sha256,
            expected_fingerprint=fingerprint,
            timed_out=execution.timed_out,
        )
        return parsed_to_run_result(
            self.name,
            parsed,
            execution,
            metadata={
                "code_object": str(code_object),
                "code_object_sha256": code_sha256,
                "code_object_fingerprint": fingerprint,
                "mode": "record-replay",
                "policy": "strict",
                "embedded_waitcheck": "preflight_only",
            },
        )


__all__ = ["ConSanPlugin"]
