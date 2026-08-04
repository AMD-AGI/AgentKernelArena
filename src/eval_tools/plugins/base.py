"""Small, runtime-agnostic contracts shared by sanitizer plugins.

The evaluator owns process isolation.  Plugins only decide whether a tool is
applicable, describe the exact subprocess to run, and parse its output.  This
keeps tool-specific environment variables out of the scoring process and makes
the same plugin usable by a local subprocess client or a container client.

These contracts intentionally use strings for capability/result states.  The
top-level eval-tools contracts expose enums with the same values; keeping the
wire representation primitive avoids an import cycle and makes plugin results
straightforward to serialize.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from ..contracts import (
    CapabilityCheck,
    CapabilityState,
    ExecutionRecord,
    ExecutionStatus,
    Finding,
    FindingSeverity,
    FindingStatus,
    ToolContext,
    ToolInvocation,
    ToolRunResult,
)


READY = CapabilityState.READY
ADAPTER_REQUIRED = CapabilityState.ADAPTER_REQUIRED
UNSUPPORTED = CapabilityState.UNSUPPORTED
NOT_APPLICABLE = CapabilityState.NOT_APPLICABLE
UNAVAILABLE_RUNTIME = CapabilityState.UNAVAILABLE_RUNTIME

PASS = "pass"
FINDING = "finding"
TOOL_ERROR = "tool_error"
INCONCLUSIVE = "inconclusive"


def field_value(value: object, name: str, default: Any = None) -> Any:
    """Read ``name`` from a mapping or a dataclass-like object.

    Core contracts use dataclasses while unit tests and older callers often
    pass dictionaries.  Plugins accept both without weakening validation.
    Enum values are normalized by :func:`text_value` at comparison sites.
    """

    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def text_value(value: object, default: str = "") -> str:
    if value is None:
        return default
    enum_value = getattr(value, "value", value)
    return str(enum_value).strip().lower()


def bool_value(value: object, name: str, default: bool = False) -> bool:
    raw = field_value(value, name, default)
    if isinstance(raw, str):
        return raw.strip().lower() in {"1", "true", "yes", "on", "ready", "supported"}
    return bool(raw)


@dataclass(frozen=True)
class FindingRecord:
    kind: str
    message: str
    severity: str = "error"
    kernel: Optional[str] = None
    location: Optional[str] = None
    raw: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ParseResult:
    status: str
    findings: tuple[FindingRecord, ...] = ()
    reason_code: Optional[str] = None
    details: str = ""
    attested: bool = False

    @property
    def passed(self) -> bool:
        return self.status == PASS


def execute_invocation(invocation: ToolInvocation) -> ExecutionRecord:
    """Execute an invocation without a shell.

    Production orchestration may replace this with a container/runtime client.
    Keeping a safe local implementation makes probes and plugin unit tests
    directly executable and avoids shell interpolation of submission data.
    """

    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in invocation.env.items()})
    started = time.monotonic()
    try:
        completed = subprocess.run(
            invocation.command,
            cwd=invocation.cwd,
            env=env,
            capture_output=True,
            text=True,
            timeout=invocation.timeout_s,
            check=False,
        )
        return ExecutionRecord(
            command=invocation.command,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            duration_s=time.monotonic() - started,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return ExecutionRecord(
            command=invocation.command,
            returncode=None,
            stdout=stdout,
            stderr=stderr,
            duration_s=time.monotonic() - started,
            timed_out=True,
        )


def command_tuple(command: Sequence[str]) -> tuple[str, ...]:
    result = tuple(str(part) for part in command)
    if not result:
        raise ValueError("command must not be empty")
    return result


def ready_check(**evidence: Any) -> CapabilityCheck:
    return CapabilityCheck.ready(**evidence)


def blocked_check(state: CapabilityState, code: str, detail: str, **evidence: Any) -> CapabilityCheck:
    return CapabilityCheck.blocked(state, code, detail, **evidence)


def command_from_context(context: ToolContext, *keys: str) -> tuple[str, ...]:
    """Resolve a trusted argv vector from plugin options.

    Tool commands are deliberately not inferred from correctness shell strings;
    doing so would reintroduce shell ambiguity and could accidentally sanitize a
    reference path instead of the optimized candidate.
    """

    for key in keys or ("command",):
        raw = context.options.get(key)
        if raw:
            if isinstance(raw, str):
                raise ValueError(f"{context.profile.task_type}: {key} must be an argv list, not a shell string")
            return command_tuple(raw)
    raise ValueError(f"missing tool argv option; expected one of {keys or ('command',)}")


def context_path(context: ToolContext, key: str, *, required: bool = False) -> Optional[Path]:
    raw = context.options.get(key)
    if raw is None:
        if required:
            raise ValueError(f"missing required tool option: {key}")
        return None
    path = Path(str(raw))
    if not path.is_absolute():
        path = Path(context.workspace) / path
    return path


def sidecar_path(context: ToolContext, key: str, *, required: bool = False) -> Optional[Path]:
    """Return an absolute path interpreted inside the isolated runtime.

    Never stat this path in the scoring process: the two mount namespaces are
    intentionally different.  Requiring an absolute, NUL-free path prevents
    cwd-dependent interpretation while RuntimeClient.probe supplies existence
    and version evidence.
    """

    raw = context.options.get(key)
    if raw is None:
        if required:
            raise ValueError(f"missing required sidecar tool option: {key}")
        return None
    text = str(raw)
    if "\x00" in text:
        raise ValueError(f"invalid NUL byte in sidecar path option: {key}")
    path = Path(text)
    if not path.is_absolute():
        raise ValueError(f"sidecar path option {key} must be absolute: {text!r}")
    return path


def parsed_to_run_result(
    tool: str,
    parsed: ParseResult,
    execution: ExecutionRecord,
    *,
    artifacts: Sequence[str | Path] = (),
    metadata: Mapping[str, Any] = {},
) -> ToolRunResult:
    if parsed.status == FINDING:
        execution_status = ExecutionStatus.COMPLETED
        finding_status = FindingStatus.FOUND
    elif parsed.status == PASS:
        execution_status = ExecutionStatus.COMPLETED
        finding_status = FindingStatus.CLEAN
    elif parsed.status == INCONCLUSIVE:
        execution_status = ExecutionStatus.TIMEOUT if execution.timed_out else ExecutionStatus.COMPLETED
        finding_status = FindingStatus.INCONCLUSIVE
    else:
        execution_status = ExecutionStatus.TIMEOUT if execution.timed_out else ExecutionStatus.TOOL_ERROR
        finding_status = FindingStatus.INCONCLUSIVE

    findings = tuple(
        Finding(
            kind=item.kind,
            severity=FindingSeverity(item.severity),
            message=item.message,
            locations=tuple(v for v in (item.location,) if v),
            raw=item.raw or None,
            metadata={**dict(item.metadata), **({"kernel": item.kernel} if item.kernel else {})},
        )
        for item in parsed.findings
    )
    result_metadata = {
        "reason_code": parsed.reason_code,
        "attested": parsed.attested,
        **dict(metadata),
    }
    return ToolRunResult(
        tool=tool,
        execution=execution_status,
        finding=finding_status,
        findings=findings,
        summary=parsed.details or parsed.reason_code,
        artifacts=tuple(str(v) for v in artifacts),
        execution_record=execution,
        metadata=result_metadata,
    )
