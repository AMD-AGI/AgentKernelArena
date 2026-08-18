"""rocJITsu race-detector plugin."""

from __future__ import annotations

from pathlib import Path

from ..adapters.replay_capsule import CapsuleValidationError, ReplayCapsule
from ..adapters.rocjitsu_replay import (
    SUPPORTED_ARCH,
    sha256_file,
    validate_replay_identity,
)
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
from .base import (
    INCONCLUSIVE,
    ParseResult,
    artifact_path,
    blocked_check,
    command_from_context,
    parsed_to_run_result,
    ready_check,
    sidecar_path,
)
from .parsers import parse_rocjitsu


_AOT_ADAPTERS = {
    KernelLanguage.TRITON: "triton_aot",
    KernelLanguage.FLYDSL: "flydsl_aot",
}
_SIDECAR_PYTHON = "/opt/venv/bin/python"
_SIDECAR_REPLAY_ENTRYPOINT = (
    "/opt/aka-eval-tools/src/eval_tools/adapters/rocjitsu_replay_entrypoint.py"
)
_SIDECAR_HIPCC = "/opt/rocm/bin/hipcc"


def _expected_aot_adapter(context: ToolContext) -> str | None:
    return _AOT_ADAPTERS.get(context.profile.language)


def _capsule_path_below_workspace(context: ToolContext) -> Path:
    raw = context.options.get("capsule")
    if not raw:
        raise CapsuleValidationError("a replay capsule path is required")
    workspace = Path(context.workspace).resolve(strict=True)
    path = Path(str(raw))
    if not path.is_absolute():
        path = workspace / path
    path = path.resolve(strict=True)
    try:
        path.relative_to(workspace)
    except ValueError as error:
        raise CapsuleValidationError(
            "replay capsule must be located below the selected task workspace"
        ) from error
    if not path.is_file():
        raise CapsuleValidationError("replay capsule path is not a regular file")
    return path


def _load_aot_capsule(context: ToolContext) -> tuple[Path, ReplayCapsule, str]:
    expected_adapter = _expected_aot_adapter(context)
    if expected_adapter is None:
        raise CapsuleValidationError("task language does not use AOT replay")
    if context.profile.adapter != expected_adapter:
        raise CapsuleValidationError(
            "task profile adapter must exactly match the language-specific AOT adapter: "
            f"{context.profile.adapter!r} != {expected_adapter!r}"
        )
    if context.gpu_arch != SUPPORTED_ARCH:
        raise CapsuleValidationError(
            f"rocJITsu AOT replay is verified only for {SUPPORTED_ARCH}"
        )
    if context.options.get("launcher") or context.options.get("command"):
        raise CapsuleValidationError(
            "Triton/FlyDSL rocJITsu replay forbids user-configured launchers"
        )

    path = _capsule_path_below_workspace(context)
    capsule = ReplayCapsule.load(path, verify_files=True)
    validate_replay_identity(
        capsule,
        expected_adapter=expected_adapter,
        expected_arch=SUPPORTED_ARCH,
        expected_kernel=capsule.code_object.kernel_name,
    )
    return path, capsule, sha256_file(path)


def _race_report_path(context: ToolContext) -> Path:
    if _expected_aot_adapter(context) is not None and "race_report" in context.options:
        raise ValueError("AOT replay does not allow a configurable race_report sink")
    return artifact_path(
        context,
        "race_report",
        Path("rocjitsu-report") / "race.log",
    )


class RocJitsuPlugin:
    name = ToolName.ROCJITSU.value
    version = "3"

    def assess(self, context: ToolContext, runtime: CapabilityCheck) -> ToolCapability:
        profile = context.profile
        if profile.framework in {"aiter", "rocblas", "rccl"}:
            engine = blocked_check(
                CapabilityState.UNSUPPORTED,
                "rocjitsu_framework_runtime_unsupported",
                "The current rocJITsu runtime cannot safely execute this Python/library path.",
            )
        elif profile.language in {KernelLanguage.HIP} and profile.artifact_kind != ArtifactKind.HSACO_PRECOMPILED:
            engine = ready_check(isa=profile.language.value)
        elif profile.language in {KernelLanguage.TRITON, KernelLanguage.FLYDSL}:
            # The simulator core handles both generated gfx950 code objects.
            if context.gpu_arch == SUPPORTED_ARCH:
                engine = ready_check(isa=profile.language.value, gpu_arch=SUPPORTED_ARCH)
            else:
                engine = blocked_check(
                    CapabilityState.UNSUPPORTED,
                    "rocjitsu_aot_gfx950_only",
                    f"AOT replay has only been verified on {SUPPORTED_ARCH}.",
                    observed_gpu_arch=context.gpu_arch,
                )
        else:
            engine = blocked_check(
                CapabilityState.UNSUPPORTED,
                "rocjitsu_artifact_not_replayable",
                "No supported native launcher or validated replay capsule is available.",
            )

        if profile.language == KernelLanguage.HIP and profile.artifact_kind != ArtifactKind.HSACO_PRECOMPILED:
            if context.options.get("launcher") or context.options.get("command"):
                adapter = ready_check(adapter="hip_native")
            else:
                adapter = blocked_check(
                    CapabilityState.ADAPTER_REQUIRED,
                    "rocjitsu_hip_native_launcher_missing",
                    "A dedicated native HIP launcher argv must be configured.",
                    expected_adapter="hip_native",
                )
        elif profile.language in {KernelLanguage.TRITON, KernelLanguage.FLYDSL}:
            expected = f"{profile.language.value}_aot"
            try:
                capsule_path, capsule, capsule_sha256 = _load_aot_capsule(context)
            except (CapsuleValidationError, OSError, ValueError) as error:
                adapter = blocked_check(
                    CapabilityState.ADAPTER_REQUIRED,
                    "rocjitsu_python_jit_requires_aot_capsule",
                    "rocJITsu requires a validated, language-matched AOT replay capsule; "
                    f"whole-Python JIT and arbitrary launchers are not accepted ({error}).",
                    expected_adapter=expected,
                )
            else:
                adapter = ready_check(
                    adapter=expected,
                    capsule=str(capsule_path),
                    capsule_sha256=capsule_sha256,
                    code_object_sha256=capsule.code_object.sha256,
                    expected_kernel=capsule.code_object.kernel_name,
                    gpu_arch=capsule.target.gpu_arch,
                )
        else:
            adapter = ready_check()

        if runtime.state == CapabilityState.READY and runtime.evidence.get("config_path") is None:
            runtime = blocked_check(
                CapabilityState.UNAVAILABLE_RUNTIME,
                "rocjitsu_config_missing",
                "rocJITsu requires an architecture-matched simulation config.",
                **dict(runtime.evidence),
            )
        return ToolCapability(self.name, engine, adapter, runtime)

    def build_invocation(self, context: ToolContext) -> ToolInvocation:
        binary = sidecar_path(context, "rocjitsu_binary", required=True)
        config = sidecar_path(context, "config_path", required=True)
        assert binary is not None and config is not None
        # ``binary`` and ``config`` are sidecar-namespace paths.  Their
        # existence/version is established by RuntimeClient.probe().
        report_path = _race_report_path(context)
        if report_path.name != "race.log":
            raise ValueError("race_report must name rocJITsu's race.log sink")
        report_dir = report_path.parent
        report_dir.mkdir(parents=True, exist_ok=True)
        # File sinks are never allowed to inherit evidence from an earlier run.
        report_path.write_text("", encoding="utf-8")
        metadata = {"race_report": str(report_path)}
        if _expected_aot_adapter(context) is not None:
            capsule_path, capsule, capsule_sha256 = _load_aot_capsule(context)
            replay_dir = Path(context.artifact_dir) / "rocjitsu-replay"
            expected_adapter = _expected_aot_adapter(context)
            assert expected_adapter is not None
            command = (
                _SIDECAR_PYTHON,
                "-I",
                _SIDECAR_REPLAY_ENTRYPOINT,
                "--capsule",
                str(capsule_path),
                "--output-dir",
                str(replay_dir),
                "--rocjitsu",
                str(binary),
                "--config",
                str(config),
                "--hipcc",
                _SIDECAR_HIPCC,
                "--expected-adapter",
                expected_adapter,
                "--expected-arch",
                SUPPORTED_ARCH,
                "--expected-kernel",
                capsule.code_object.kernel_name,
                "--expected-capsule-sha256",
                capsule_sha256,
            )
            metadata.update(
                {
                    "adapter": expected_adapter,
                    "capsule": str(capsule_path),
                    "capsule_sha256": capsule_sha256,
                    "code_object_sha256": capsule.code_object.sha256,
                    "expected_kernel": capsule.code_object.kernel_name,
                    "expected_replay_result": "AKA_REPLAY_RESULT pass",
                }
            )
        else:
            launcher = command_from_context(context, "launcher", "command")
            command = (str(binary), "--config", str(config), "--", *launcher)
            metadata["expected_kernel"] = context.options.get("expected_kernel")
        env = {
            **dict(context.env),
            "RJ_RACE": "1",
            "RJ_LOG": "1",
            "RJ_SINKS": "stderr,file",
            "RJ_SINK_DIR": str(report_dir),
            "AKA_EVAL_TOOL": self.name,
        }
        return ToolInvocation(
            tool=self.name,
            command=command,
            cwd=context.workspace,
            env=env,
            timeout_s=int(context.options.get("timeout_s", 600)),
            artifact_dir=context.artifact_dir,
            metadata=metadata,
        )

    def parse(self, context: ToolContext, execution) -> ToolRunResult:
        aot_adapter = _expected_aot_adapter(context)
        report_path = _race_report_path(context)
        report = report_path.read_text(encoding="utf-8", errors="replace") if report_path.is_file() else ""
        combined = "\n".join((execution.stdout, execution.stderr, report))
        metadata = {}
        if aot_adapter is not None:
            try:
                capsule_path, capsule, capsule_sha256 = _load_aot_capsule(context)
            except (CapsuleValidationError, OSError, ValueError) as error:
                parsed = ParseResult(
                    INCONCLUSIVE,
                    reason_code="rocjitsu_capsule_revalidation_failed",
                    details=f"Replay capsule failed post-execution validation: {error}",
                )
                return parsed_to_run_result(
                    self.name,
                    parsed,
                    execution,
                    artifacts=[report_path] if report_path.is_file() else [],
                )
            expected_kernel = capsule.code_object.kernel_name
            dispatch_seen = f'Kernel dispatch: "{expected_kernel}"' in combined
            replay_pass_seen = "AKA_REPLAY_RESULT pass" in combined
            capsule_attestation = (
                "AKA_REPLAY_CAPSULE "
                f"sha256={capsule_sha256} "
                f"code_sha256={capsule.code_object.sha256} "
                f"adapter={aot_adapter} arch={SUPPORTED_ARCH} "
                f"kernel={expected_kernel}"
            )
            capsule_attested = capsule_attestation in combined
            attested = dispatch_seen and replay_pass_seen and capsule_attested
            metadata = {
                "adapter": aot_adapter,
                "capsule": str(capsule_path),
                "capsule_sha256": capsule_sha256,
                "code_object_sha256": capsule.code_object.sha256,
                "expected_kernel": expected_kernel,
                "dispatch_attested": dispatch_seen,
                "replay_result_attested": replay_pass_seen,
                "capsule_attested": capsule_attested,
            }
        else:
            expected_kernel = context.options.get("expected_kernel")
            if expected_kernel:
                dispatch_seen = f'Kernel dispatch: "{expected_kernel}"' in combined
            else:
                dispatch_seen = "[rocjitsu] Kernel dispatch:" in combined
            attested = dispatch_seen
        parsed = parse_rocjitsu(
            execution.stdout,
            execution.stderr,
            execution.returncode,
            attested=attested,
            report_text=report,
            timed_out=execution.timed_out,
        )
        artifacts = [report_path] if report_path.is_file() else []
        if aot_adapter is not None:
            replay_dir = Path(context.artifact_dir) / "rocjitsu-replay"
            artifacts.extend(
                path
                for path in (
                    replay_dir / "replay_launcher.cpp",
                    replay_dir / "replay_launcher",
                )
                if path.is_file()
            )
        return parsed_to_run_result(
            self.name,
            parsed,
            execution,
            artifacts=artifacts,
            metadata=metadata,
        )
