"""Static rocJITsu Waitcheck plugin for evaluator-selected native code objects."""

from __future__ import annotations

from ..adapters.native_code_object import resolve_workspace_file, sha256_file
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
    parsed_to_run_result,
    ready_check,
    sidecar_path,
)
from .parsers import parse_waitcheck


_ENTRYPOINT = "/opt/aka-eval-tools/src/eval_tools/adapters/waitcheck_entrypoint.py"
_PYTHON = "/opt/venv/bin/python"


def _kernel_entry(context: ToolContext) -> int:
    raw = context.options.get("kernel_entry")
    if isinstance(raw, bool):
        raise ValueError("kernel_entry must be a non-negative integer")
    try:
        value = int(str(raw), 0) if isinstance(raw, str) else int(raw)
    except (TypeError, ValueError) as error:
        raise ValueError("kernel_entry must be a non-negative integer") from error
    if value < 0:
        raise ValueError("kernel_entry must be a non-negative integer")
    return value


def _expected_kernel(context: ToolContext) -> str:
    value = str(context.options.get("expected_kernel") or "").strip()
    if not value or "\x00" in value:
        raise ValueError("expected_kernel must be a non-empty NUL-free string")
    return value


class WaitcheckPlugin:
    name = ToolName.ROCJITSU_WAITCHECK.value
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
                "waitcheck_non_amdgpu_language",
                "Waitcheck analyzes final AMDGPU code objects only.",
            )
        elif context.gpu_arch != "gfx950":
            engine = blocked_check(
                CapabilityState.UNSUPPORTED,
                "waitcheck_gfx950_not_selected",
                "This integration is qualified only for gfx950.",
                observed_gpu_arch=context.gpu_arch,
            )
        else:
            engine = ready_check(target="gfx950", analysis="static_object_code")

        try:
            code_object = resolve_workspace_file(context, "code_object")
            expected_kernel = _expected_kernel(context)
            entry = _kernel_entry(context)
        except (OSError, ValueError) as error:
            adapter = blocked_check(
                CapabilityState.ADAPTER_REQUIRED,
                "waitcheck_code_object_adapter_required",
                "Waitcheck requires an evaluator-selected HSACO, exact kernel name, "
                f"and .text entry offset ({error}).",
            )
        else:
            adapter = ready_check(
                adapter="waitcheck_code_object",
                code_object=str(code_object),
                code_object_sha256=sha256_file(code_object),
                expected_kernel=expected_kernel,
                kernel_entry=entry,
            )
        return ToolCapability(self.name, engine, adapter, runtime)

    def build_invocation(self, context: ToolContext) -> ToolInvocation:
        code_object = resolve_workspace_file(context, "code_object")
        code_sha256 = sha256_file(code_object)
        expected_kernel = _expected_kernel(context)
        entry = _kernel_entry(context)
        binary = sidecar_path(context, "waitcheck_binary", required=True)
        capi_wrapper = sidecar_path(
            context, "waitcheck_capi_wrapper", required=True
        )
        assert binary is not None and capi_wrapper is not None
        return ToolInvocation(
            tool=self.name,
            command=(
                _PYTHON,
                "-I",
                _ENTRYPOINT,
                "--waitcheck",
                str(binary),
                "--capi-wrapper",
                str(capi_wrapper),
                "--code-object",
                str(code_object),
                "--expected-sha256",
                code_sha256,
                "--target",
                "gfx950",
                "--expected-kernel",
                expected_kernel,
                "--kernel-entry",
                str(entry),
            ),
            cwd=context.workspace,
            env={**dict(context.env), "AKA_EVAL_TOOL": self.name},
            timeout_s=int(context.options.get("timeout_s", 600)),
            artifact_dir=context.artifact_dir,
            metadata={
                "code_object": str(code_object),
                "code_object_sha256": code_sha256,
                "expected_kernel": expected_kernel,
                "kernel_entry": entry,
            },
        )

    def parse(self, context: ToolContext, execution) -> ToolRunResult:
        code_object = resolve_workspace_file(context, "code_object")
        code_sha256 = sha256_file(code_object)
        expected_kernel = _expected_kernel(context)
        entry = _kernel_entry(context)
        parsed = parse_waitcheck(
            execution.stdout,
            execution.stderr,
            execution.returncode,
            expected_sha256=code_sha256,
            expected_target="gfx950",
            expected_kernel=expected_kernel,
            expected_entry=entry,
            timed_out=execution.timed_out,
        )
        return parsed_to_run_result(
            self.name,
            parsed,
            execution,
            metadata={
                "code_object": str(code_object),
                "code_object_sha256": code_sha256,
                "expected_kernel": expected_kernel,
                "kernel_entry": entry,
            },
        )


__all__ = ["WaitcheckPlugin"]
