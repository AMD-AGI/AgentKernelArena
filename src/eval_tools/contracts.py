"""Strongly typed contracts for optional evaluation tools.

The centralized evaluator intentionally knows nothing about a tool's concrete
installation or command line.  This module is the narrow boundary shared by
task profiling, tool plugins, runtime transports, and result reporting.  The
contracts use only the standard library so they can also be imported by a
small sidecar process.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Sequence, runtime_checkable


class _StringEnum(str, Enum):
    """``StrEnum`` compatible with the Python 3.10 scoring image."""

    def __str__(self) -> str:
        return self.value


class ToolName(_StringEnum):
    TRITON_FPSAN = "triton_fpsan"
    GPU_ASAN = "gpu_asan"
    ROCJITSU = "rocjitsu"
    ROCJITSU_WAITCHECK = "rocjitsu_waitcheck"
    ROCJITSU_CONSAN = "rocjitsu_consan"
    HIP_FPSAN = "hip_fpsan"


class KernelLanguage(_StringEnum):
    TRITON = "triton"
    HIP = "hip"
    FLYDSL = "flydsl"
    UNKNOWN = "unknown"


class ArtifactKind(_StringEnum):
    SOURCE_AOT = "source_aot"
    PYTHON_JIT = "python_jit"
    HSACO_PRECOMPILED = "hsaco_precompiled"
    UNKNOWN = "unknown"


class InstrumentationControl(_StringEnum):
    COMPILER_CONTROLLED = "compiler_controlled"
    RECOMPILE = "recompile"
    NONE = "none"
    UNKNOWN = "unknown"


class CapabilityState(_StringEnum):
    READY = "ready"
    ADAPTER_REQUIRED = "adapter_required"
    UNSUPPORTED = "unsupported"
    NOT_APPLICABLE = "not_applicable"
    UNAVAILABLE_RUNTIME = "unavailable_runtime"


class ExecutionStatus(_StringEnum):
    NOT_RUN = "not_run"
    COMPLETED = "completed"
    TOOL_ERROR = "tool_error"
    TIMEOUT = "timeout"


class FindingStatus(_StringEnum):
    NOT_EVALUATED = "not_evaluated"
    CLEAN = "clean"
    FOUND = "found"
    INCONCLUSIVE = "inconclusive"


class FindingSeverity(_StringEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class EvaluationPolicy(_StringEnum):
    ADVISORY = "advisory"
    REQUIRED = "required"


def _tool_value(value: str | ToolName) -> str:
    text = value.value if isinstance(value, ToolName) else str(value)
    text = text.strip().lower().replace("-", "_")
    if not text:
        raise ValueError("tool name must be a non-empty string")
    return text


def _enum_value(enum_cls, value):
    if isinstance(value, enum_cls):
        return value
    return enum_cls(str(value).strip().lower())


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, Path)):
        return (str(value),)
    return tuple(str(item) for item in value)


def _plain(value: Any) -> Any:
    """Return a deterministic JSON/YAML-compatible representation."""

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True)
class CapabilityCheck:
    state: CapabilityState
    reason_code: Optional[str] = None
    detail: Optional[str] = None
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", _enum_value(CapabilityState, self.state))
        object.__setattr__(self, "evidence", dict(self.evidence or {}))

    @classmethod
    def ready(cls, **evidence: Any) -> "CapabilityCheck":
        return cls(CapabilityState.READY, evidence=evidence)

    @classmethod
    def blocked(
        cls,
        state: CapabilityState,
        reason_code: str,
        detail: Optional[str] = None,
        **evidence: Any,
    ) -> "CapabilityCheck":
        if state == CapabilityState.READY:
            raise ValueError("blocked capability cannot use state=READY")
        return cls(state, reason_code=reason_code, detail=detail, evidence=evidence)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"state": self.state.value}
        if self.reason_code:
            result["reason_code"] = self.reason_code
        if self.detail:
            result["detail"] = self.detail
        if self.evidence:
            result["evidence"] = _plain(self.evidence)
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CapabilityCheck":
        return cls(
            state=data["state"],
            reason_code=data.get("reason_code"),
            detail=data.get("detail"),
            evidence=data.get("evidence") or {},
        )


def _coerce_check(value: CapabilityCheck | CapabilityState | str) -> CapabilityCheck:
    if isinstance(value, CapabilityCheck):
        return value
    return CapabilityCheck(_enum_value(CapabilityState, value))


def effective_capability(
    engine: CapabilityCheck,
    adapter: CapabilityCheck,
    runtime: CapabilityCheck,
) -> CapabilityCheck:
    """Resolve the externally visible capability from independent dimensions.

    Ordering is deliberate: a tool that is semantically irrelevant should be
    reported as ``not_applicable`` even if its image is not installed; an
    unsupported engine is more fundamental than a missing adapter/runtime.
    """

    checks = (engine, adapter, runtime)
    precedence = (
        CapabilityState.NOT_APPLICABLE,
        CapabilityState.UNSUPPORTED,
        CapabilityState.ADAPTER_REQUIRED,
        CapabilityState.UNAVAILABLE_RUNTIME,
    )
    for state in precedence:
        for check in checks:
            if check.state == state:
                return CapabilityCheck(
                    state=state,
                    reason_code=check.reason_code,
                    detail=check.detail,
                    evidence=check.evidence,
                )
    return CapabilityCheck.ready()


@dataclass(frozen=True)
class ToolCapability:
    tool: str
    engine: CapabilityCheck
    adapter: CapabilityCheck
    runtime: CapabilityCheck
    effective: Optional[CapabilityCheck] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool", _tool_value(self.tool))
        engine = _coerce_check(self.engine)
        adapter = _coerce_check(self.adapter)
        runtime = _coerce_check(self.runtime)
        object.__setattr__(self, "engine", engine)
        object.__setattr__(self, "adapter", adapter)
        object.__setattr__(self, "runtime", runtime)
        resolved = _coerce_check(self.effective) if self.effective is not None else effective_capability(
            engine, adapter, runtime
        )
        object.__setattr__(self, "effective", resolved)

    @property
    def ready(self) -> bool:
        return self.effective is not None and self.effective.state == CapabilityState.READY

    def to_dict(self) -> dict[str, Any]:
        assert self.effective is not None
        return {
            "tool": self.tool,
            "engine": self.engine.to_dict(),
            "adapter": self.adapter.to_dict(),
            "runtime": self.runtime.to_dict(),
            "effective": self.effective.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ToolCapability":
        return cls(
            tool=str(data["tool"]),
            engine=CapabilityCheck.from_dict(data["engine"]),
            adapter=CapabilityCheck.from_dict(data["adapter"]),
            runtime=CapabilityCheck.from_dict(data["runtime"]),
            effective=CapabilityCheck.from_dict(data["effective"]),
        )


@dataclass(frozen=True)
class TaskProfile:
    task_type: str
    language: KernelLanguage
    artifact_kind: ArtifactKind
    framework: str
    instrumentation_control: InstrumentationControl
    adapter: Optional[str]
    source_available: bool
    source_files: tuple[str, ...] = ()
    target_functions: tuple[str, ...] = ()
    explicit_overrides: tuple[str, ...] = ()
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_type", str(self.task_type or "").strip().lower())
        object.__setattr__(self, "language", _enum_value(KernelLanguage, self.language))
        object.__setattr__(self, "artifact_kind", _enum_value(ArtifactKind, self.artifact_kind))
        object.__setattr__(
            self,
            "instrumentation_control",
            _enum_value(InstrumentationControl, self.instrumentation_control),
        )
        framework = str(self.framework or "unknown").strip().lower().replace("-", "_")
        object.__setattr__(self, "framework", framework or "unknown")
        adapter = str(self.adapter).strip().lower() if self.adapter is not None else None
        object.__setattr__(self, "adapter", adapter or None)
        object.__setattr__(self, "source_files", _string_tuple(self.source_files))
        object.__setattr__(self, "target_functions", _string_tuple(self.target_functions))
        object.__setattr__(self, "explicit_overrides", tuple(sorted(set(self.explicit_overrides))))
        object.__setattr__(self, "evidence", dict(self.evidence or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_type": self.task_type,
            "language": self.language.value,
            "artifact_kind": self.artifact_kind.value,
            "framework": self.framework,
            "instrumentation_control": self.instrumentation_control.value,
            "adapter": self.adapter,
            "source_available": self.source_available,
            "source_files": list(self.source_files),
            "target_functions": list(self.target_functions),
            "explicit_overrides": list(self.explicit_overrides),
            "evidence": _plain(self.evidence),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TaskProfile":
        return cls(
            task_type=str(data.get("task_type") or ""),
            language=data.get("language", KernelLanguage.UNKNOWN.value),
            artifact_kind=data.get("artifact_kind", ArtifactKind.UNKNOWN.value),
            framework=str(data.get("framework") or "unknown"),
            instrumentation_control=data.get(
                "instrumentation_control", InstrumentationControl.UNKNOWN.value
            ),
            adapter=data.get("adapter"),
            source_available=bool(data.get("source_available", False)),
            source_files=tuple(data.get("source_files") or ()),
            target_functions=tuple(data.get("target_functions") or ()),
            explicit_overrides=tuple(data.get("explicit_overrides") or ()),
            evidence=data.get("evidence") or {},
        )


@dataclass(frozen=True)
class SourceEvidence:
    """Identity of the immutable original and evaluated candidate sources."""

    original_root: Optional[str] = None
    original_fingerprint: Optional[str] = None
    candidate_fingerprint: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_root": self.original_root,
            "original_fingerprint": self.original_fingerprint,
            "candidate_fingerprint": self.candidate_fingerprint,
            "metadata": _plain(self.metadata),
        }

    @classmethod
    def from_value(
        cls, value: "SourceEvidence | Mapping[str, Any] | None"
    ) -> "SourceEvidence":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        return cls(
            original_root=value.get("original_root"),
            original_fingerprint=value.get("original_fingerprint"),
            candidate_fingerprint=value.get("candidate_fingerprint"),
            metadata=value.get("metadata") or {},
        )


@dataclass(frozen=True)
class ToolContext:
    workspace: str
    task_config: Mapping[str, Any]
    profile: TaskProfile
    artifact_dir: str
    gpu_arch: Optional[str] = None
    runtime_ref: Optional[str] = None
    env: Mapping[str, str] = field(default_factory=dict)
    options: Mapping[str, Any] = field(default_factory=dict)
    source_evidence: SourceEvidence = field(default_factory=SourceEvidence)

    def __post_init__(self) -> None:
        object.__setattr__(self, "workspace", str(Path(self.workspace).absolute()))
        object.__setattr__(self, "artifact_dir", str(Path(self.artifact_dir).absolute()))
        object.__setattr__(self, "task_config", dict(self.task_config or {}))
        object.__setattr__(self, "env", {str(k): str(v) for k, v in (self.env or {}).items()})
        object.__setattr__(self, "options", dict(self.options or {}))
        object.__setattr__(
            self, "source_evidence", SourceEvidence.from_value(self.source_evidence)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "workspace": self.workspace,
            "task_config": _plain(self.task_config),
            "profile": self.profile.to_dict(),
            "artifact_dir": self.artifact_dir,
            "gpu_arch": self.gpu_arch,
            "runtime_ref": self.runtime_ref,
            "env": dict(self.env),
            "options": _plain(self.options),
            "source_evidence": self.source_evidence.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ToolContext":
        return cls(
            workspace=str(data["workspace"]),
            task_config=data.get("task_config") or {},
            profile=TaskProfile.from_dict(data["profile"]),
            artifact_dir=str(data["artifact_dir"]),
            gpu_arch=data.get("gpu_arch"),
            runtime_ref=data.get("runtime_ref"),
            env=data.get("env") or {},
            options=data.get("options") or {},
            source_evidence=SourceEvidence.from_value(data.get("source_evidence")),
        )


@dataclass(frozen=True)
class ToolInvocation:
    tool: str
    command: tuple[str, ...]
    cwd: str
    env: Mapping[str, str] = field(default_factory=dict)
    timeout_s: int = 3600
    artifact_dir: Optional[str] = None
    shell: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool", _tool_value(self.tool))
        object.__setattr__(self, "command", _string_tuple(self.command))
        if not self.command:
            raise ValueError("tool invocation command cannot be empty")
        if int(self.timeout_s) <= 0:
            raise ValueError("tool invocation timeout_s must be positive")
        object.__setattr__(self, "timeout_s", int(self.timeout_s))
        object.__setattr__(self, "cwd", str(Path(self.cwd).absolute()))
        object.__setattr__(self, "env", {str(k): str(v) for k, v in (self.env or {}).items()})
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "command": list(self.command),
            "cwd": self.cwd,
            "env": dict(self.env),
            "timeout_s": self.timeout_s,
            "artifact_dir": self.artifact_dir,
            "shell": self.shell,
            "metadata": _plain(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ToolInvocation":
        return cls(
            tool=str(data["tool"]),
            command=tuple(data.get("command") or ()),
            cwd=str(data["cwd"]),
            env=data.get("env") or {},
            timeout_s=int(data.get("timeout_s", 3600)),
            artifact_dir=data.get("artifact_dir"),
            shell=bool(data.get("shell", False)),
            metadata=data.get("metadata") or {},
        )


# Short alias used by tool plugins.
Invocation = ToolInvocation


@dataclass(frozen=True)
class ExecutionRecord:
    command: tuple[str, ...]
    returncode: Optional[int]
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0
    timed_out: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "command", _string_tuple(self.command))
        object.__setattr__(self, "duration_s", max(0.0, float(self.duration_s)))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    @property
    def status(self) -> ExecutionStatus:
        if self.timed_out:
            return ExecutionStatus.TIMEOUT
        if self.returncode == 0:
            return ExecutionStatus.COMPLETED
        return ExecutionStatus.TOOL_ERROR

    def to_dict(self, *, include_output: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "command": list(self.command),
            "returncode": self.returncode,
            "duration_s": self.duration_s,
            "timed_out": self.timed_out,
            "metadata": _plain(self.metadata),
        }
        if include_output:
            result["stdout"] = self.stdout
            result["stderr"] = self.stderr
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExecutionRecord":
        return cls(
            command=tuple(data.get("command") or ()),
            returncode=data.get("returncode"),
            stdout=str(data.get("stdout") or ""),
            stderr=str(data.get("stderr") or ""),
            duration_s=float(data.get("duration_s", 0.0)),
            timed_out=bool(data.get("timed_out", False)),
            metadata=data.get("metadata") or {},
        )


@dataclass(frozen=True)
class Finding:
    kind: str
    severity: FindingSeverity
    message: str
    locations: tuple[str, ...] = ()
    raw: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.kind).strip():
            raise ValueError("finding kind cannot be empty")
        object.__setattr__(self, "severity", _enum_value(FindingSeverity, self.severity))
        object.__setattr__(self, "locations", _string_tuple(self.locations))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))

    def to_dict(self, *, include_output: bool = False) -> dict[str, Any]:
        result = {
            "kind": self.kind,
            "severity": self.severity.value,
            "message": self.message,
            "locations": list(self.locations),
            "metadata": _plain(self.metadata),
        }
        # ``raw`` is parser output derived from the bounded stdout/stderr logs
        # and can still be tens of MiB.  Durable summaries point at those log
        # artifacts instead of duplicating them unless a caller explicitly
        # requests verbose output.
        if include_output and self.raw is not None:
            result["raw"] = self.raw
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Finding":
        return cls(
            kind=str(data["kind"]),
            severity=data.get("severity", FindingSeverity.ERROR.value),
            message=str(data.get("message") or ""),
            locations=tuple(data.get("locations") or ()),
            raw=data.get("raw"),
            metadata=data.get("metadata") or {},
        )


@dataclass(frozen=True)
class ToolRunResult:
    tool: str
    execution: ExecutionStatus
    finding: FindingStatus
    findings: tuple[Finding, ...] = ()
    summary: Optional[str] = None
    artifacts: tuple[str, ...] = ()
    execution_record: Optional[ExecutionRecord] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool", _tool_value(self.tool))
        object.__setattr__(self, "execution", _enum_value(ExecutionStatus, self.execution))
        object.__setattr__(self, "finding", _enum_value(FindingStatus, self.finding))
        object.__setattr__(self, "findings", tuple(self.findings or ()))
        object.__setattr__(self, "artifacts", _string_tuple(self.artifacts))
        object.__setattr__(self, "metadata", dict(self.metadata or {}))
        if self.finding == FindingStatus.CLEAN and self.findings:
            raise ValueError("a CLEAN result cannot contain findings")
        if self.finding == FindingStatus.FOUND and not self.findings:
            # Parsers should retain at least one structured finding instead of
            # reducing a report to a lossy boolean.
            raise ValueError("a FOUND result must contain at least one finding")

    @property
    def findings_count(self) -> int:
        return len(self.findings)

    def to_dict(self, *, include_output: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "tool": self.tool,
            "execution": self.execution.value,
            "finding": self.finding.value,
            "findings_count": self.findings_count,
            "findings": [
                item.to_dict(include_output=include_output) for item in self.findings
            ],
            "artifacts": list(self.artifacts),
            "metadata": _plain(self.metadata),
        }
        if self.summary:
            result["summary"] = self.summary
        if self.execution_record is not None:
            result["execution_record"] = self.execution_record.to_dict(
                include_output=include_output
            )
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ToolRunResult":
        record_data = data.get("execution_record")
        return cls(
            tool=str(data["tool"]),
            execution=data.get("execution", ExecutionStatus.NOT_RUN.value),
            finding=data.get("finding", FindingStatus.NOT_EVALUATED.value),
            findings=tuple(Finding.from_dict(item) for item in data.get("findings") or ()),
            summary=data.get("summary"),
            artifacts=tuple(data.get("artifacts") or ()),
            execution_record=(
                ExecutionRecord.from_dict(record_data) if isinstance(record_data, Mapping) else None
            ),
            metadata=data.get("metadata") or {},
        )


@dataclass(frozen=True)
class ToolEvaluation:
    capability: ToolCapability
    result: Optional[ToolRunResult] = None

    def __post_init__(self) -> None:
        if self.result is not None and self.result.tool != self.capability.tool:
            raise ValueError("capability and run result refer to different tools")
        if not self.capability.ready and self.result is not None:
            raise ValueError("a non-ready capability must not have a run result")

    def to_dict(self, *, include_output: bool = False) -> dict[str, Any]:
        return {
            "capability": self.capability.to_dict(),
            "result": (
                self.result.to_dict(include_output=include_output)
                if self.result is not None
                else None
            ),
        }


@dataclass(frozen=True)
class ToolPlan:
    tool: str
    runtime_ref: Optional[str] = None
    plugin_version: str = "unknown"
    timeout_s: int = 3600
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool", _tool_value(self.tool))
        if int(self.timeout_s) <= 0:
            raise ValueError("tool timeout_s must be positive")
        object.__setattr__(self, "timeout_s", int(self.timeout_s))
        object.__setattr__(self, "options", dict(self.options or {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "runtime_ref": self.runtime_ref,
            "plugin_version": self.plugin_version,
            "timeout_s": self.timeout_s,
            "options": _plain(self.options),
        }


@dataclass(frozen=True)
class EvaluationPlan:
    schema_version: int
    policy: EvaluationPolicy
    profile: TaskProfile
    tools: tuple[ToolPlan, ...]
    fingerprint: str
    source_evidence: SourceEvidence = field(default_factory=SourceEvidence)

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy", _enum_value(EvaluationPolicy, self.policy))
        object.__setattr__(self, "tools", tuple(self.tools or ()))
        object.__setattr__(
            self, "source_evidence", SourceEvidence.from_value(self.source_evidence)
        )
        if len({item.tool for item in self.tools}) != len(self.tools):
            raise ValueError("evaluation plan contains duplicate tools")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy": self.policy.value,
            "profile": self.profile.to_dict(),
            "tools": [item.to_dict() for item in self.tools],
            "fingerprint": self.fingerprint,
            "source_evidence": self.source_evidence.to_dict(),
        }


@dataclass(frozen=True)
class PolicyDecision:
    policy: EvaluationPolicy
    allowed: bool
    policy_satisfied: bool
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy", _enum_value(EvaluationPolicy, self.policy))
        object.__setattr__(self, "reasons", _string_tuple(self.reasons))

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy.value,
            "allowed": self.allowed,
            "policy_satisfied": self.policy_satisfied,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class EvaluationReport:
    plan: EvaluationPlan
    evaluations: tuple[ToolEvaluation, ...]
    decision: PolicyDecision

    def __post_init__(self) -> None:
        object.__setattr__(self, "evaluations", tuple(self.evaluations or ()))
        expected = [item.tool for item in self.plan.tools]
        actual = [item.capability.tool for item in self.evaluations]
        if expected != actual:
            raise ValueError(
                f"evaluation order/tools do not match plan: expected={expected}, actual={actual}"
            )

    def to_dict(self, *, include_output: bool = False) -> dict[str, Any]:
        return {
            "schema_version": self.plan.schema_version,
            "plan_fingerprint": self.plan.fingerprint,
            # Retain the complete immutable plan, not only its digest.  This is
            # needed to audit the selected image, plugin version, timeout, and
            # adapter options after a run.  The legacy top-level mirrors below
            # remain for consumers introduced with schema version 1.
            "plan": self.plan.to_dict(),
            "policy": self.plan.policy.value,
            "overall_status": self.overall_status,
            "resolved_task_profile": self.plan.profile.to_dict(),
            "source_evidence": self.plan.source_evidence.to_dict(),
            "decision": self.decision.to_dict(),
            "tools": {
                item.capability.tool: item.to_dict(include_output=include_output)
                for item in self.evaluations
            },
        }

    @property
    def overall_status(self) -> str:
        applicable = []
        incomplete = False
        for item in self.evaluations:
            assert item.capability.effective is not None
            if item.capability.effective.state == CapabilityState.NOT_APPLICABLE:
                continue
            applicable.append(item)
            if not item.capability.ready or item.result is None:
                incomplete = True
                continue
            if item.result.finding == FindingStatus.FOUND:
                return "finding"
            if (
                item.result.execution != ExecutionStatus.COMPLETED
                or item.result.finding != FindingStatus.CLEAN
            ):
                incomplete = True
        if not applicable:
            return "not_applicable"
        return "incomplete" if incomplete else "clean"


@runtime_checkable
class RuntimeClient(Protocol):
    """Transport used by :class:`EvalToolManager`; implementations are injectable."""

    def probe(self, tool: str, context: ToolContext) -> CapabilityCheck:
        """Return only the runtime/image availability dimension."""

    def execute(self, invocation: ToolInvocation, context: ToolContext) -> ExecutionRecord:
        """Execute a prepared invocation in its isolated runtime."""


@runtime_checkable
class ToolPlugin(Protocol):
    name: str
    version: str

    def assess(
        self, context: ToolContext, runtime: CapabilityCheck
    ) -> ToolCapability:
        """Resolve engine/adapter support and combine it with runtime availability."""

    def build_invocation(self, context: ToolContext) -> ToolInvocation:
        """Build an invocation only after capability is READY."""

    def parse(self, context: ToolContext, execution: ExecutionRecord) -> ToolRunResult:
        """Parse exit/output/artifacts without equating return code with cleanliness."""


__all__ = [
    "ArtifactKind",
    "CapabilityCheck",
    "CapabilityState",
    "EvaluationPlan",
    "EvaluationPolicy",
    "EvaluationReport",
    "ExecutionRecord",
    "ExecutionStatus",
    "Finding",
    "FindingSeverity",
    "FindingStatus",
    "InstrumentationControl",
    "Invocation",
    "KernelLanguage",
    "PolicyDecision",
    "RuntimeClient",
    "SourceEvidence",
    "TaskProfile",
    "ToolCapability",
    "ToolContext",
    "ToolEvaluation",
    "ToolInvocation",
    "ToolName",
    "ToolPlan",
    "ToolPlugin",
    "ToolRunResult",
    "effective_capability",
]
