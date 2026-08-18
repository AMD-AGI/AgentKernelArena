"""Client for an evaluation-tool worker listening on a Unix domain socket."""

from __future__ import annotations

import json
import os
import socket
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .contracts import (
    CapabilityCheck,
    CapabilityState,
    ExecutionRecord,
    ToolContext,
    ToolInvocation,
)


DEFAULT_SOCKET_DIR = "/run/aka-eval-tools"
DEFAULT_RESPONSE_LIMIT_BYTES = 16 * 1024 * 1024
DEFAULT_SCORING_ROOT = "/workspace"
DEFAULT_ARTIFACT_SCORING_ROOT = "/workspace/experiments"
DEFAULT_SIDECAR_INPUT_ROOT = "/input"
DEFAULT_SIDECAR_ARTIFACT_ROOT = "/artifacts"
DEFAULT_RPC_GRACE_SECONDS = 90.0
_IMAGE_FRAMEWORK_ROOT = "/opt/aka-eval-tools"
_BUILTIN_PROBES = {
    "triton_fpsan": ("triton_fpsan_probe.py",),
    "gpu_asan": ("gpu_asan_probe.hip", "triton_asan_probe.py"),
    "rocjitsu": ("rocjitsu_race_probe.hip",),
    "rocjitsu_waitcheck": ("waitcheck_probe.hip",),
    "rocjitsu_consan": ("consan_probe.hip",),
    "hip_fpsan": ("hip_fpsan_probe.hip",),
}


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


class RuntimeRPCError(RuntimeError):
    """A worker rejected a request or returned an invalid RPC response."""

    def __init__(self, message: str, *, code: str = "RPC_ERROR", details: Any = None):
        super().__init__(message)
        self.code = code
        self.details = details


def _safe_tool_id(tool: str) -> str:
    if not isinstance(tool, str) or not tool:
        raise ValueError("tool must be a non-empty string")
    if not all(character.isalnum() or character in "_.-" for character in tool):
        raise ValueError(f"invalid tool id: {tool!r}")
    return tool


def validate_relative_rpc_path(value: str, *, field: str) -> str:
    """Validate a transport path before it reaches the worker.

    The worker performs the authoritative filesystem containment check.  This
    lexical check catches mistakes early and keeps absolute host paths out of RPC
    payloads and artifacts.
    """

    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty relative path")
    if "\x00" in value or "\\" in value:
        raise ValueError(f"{field} contains an invalid character")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must stay below its declared root")
    return path.as_posix()


def socket_path_for_tool(tool: str, socket_dir: str | Path | None = None) -> Path:
    directory = Path(
        socket_dir
        or os.environ.get("AKA_EVAL_TOOL_SOCKET_DIR")
        or DEFAULT_SOCKET_DIR
    )
    return directory / f"{_safe_tool_id(tool)}.sock"


class UnixSocketRuntimeClient:
    """Small newline-delimited JSON RPC client.

    Responses intentionally contain metadata and relative artifact paths rather
    than raw logs.  This keeps the socket message bounded even for noisy tools.
    """

    def __init__(
        self,
        socket_path: str | Path,
        *,
        timeout_seconds: float = 30.0,
        response_limit_bytes: int = DEFAULT_RESPONSE_LIMIT_BYTES,
    ) -> None:
        self.socket_path = Path(socket_path)
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if response_limit_bytes <= 0:
            raise ValueError("response_limit_bytes must be positive")
        self.timeout_seconds = timeout_seconds
        self.response_limit_bytes = response_limit_bytes

    @classmethod
    def for_tool(
        cls,
        tool: str,
        *,
        socket_dir: str | Path | None = None,
        timeout_seconds: float = 30.0,
    ) -> "UnixSocketRuntimeClient":
        return cls(
            socket_path_for_tool(tool, socket_dir),
            timeout_seconds=timeout_seconds,
        )

    def call(
        self,
        method: str,
        params: Mapping[str, Any] | None = None,
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        if not isinstance(method, str) or not method or not method.replace("_", "").isalnum():
            raise ValueError(f"invalid RPC method: {method!r}")
        if params is not None and not isinstance(params, Mapping):
            raise ValueError("RPC params must be a mapping")
        request_id = uuid.uuid4().hex
        request = {
            "id": request_id,
            "method": method,
            "params": dict(params or {}),
        }
        encoded = json.dumps(request, separators=(",", ":"), ensure_ascii=False).encode("utf-8") + b"\n"
        timeout = self.timeout_seconds if timeout_seconds is None else timeout_seconds
        if timeout <= 0:
            raise ValueError("timeout_seconds must be positive")

        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.settimeout(timeout)
                connection.connect(str(self.socket_path))
                connection.sendall(encoded)
                response_bytes = self._receive_line(connection)
        except FileNotFoundError as error:
            raise RuntimeRPCError(
                f"evaluation-tool socket is unavailable: {self.socket_path}",
                code="UNAVAILABLE_RUNTIME",
            ) from error
        except (ConnectionRefusedError, socket.timeout, OSError) as error:
            raise RuntimeRPCError(
                f"evaluation-tool RPC failed for {self.socket_path}: {error}",
                code="UNAVAILABLE_RUNTIME",
            ) from error

        try:
            response = json.loads(response_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeRPCError("worker returned invalid JSON", code="INVALID_RESPONSE") from error
        if not isinstance(response, dict) or response.get("id") != request_id:
            raise RuntimeRPCError("worker returned a mismatched response", code="INVALID_RESPONSE")
        if response.get("ok") is not True:
            error_value = response.get("error")
            if isinstance(error_value, dict):
                message = str(error_value.get("message") or "worker rejected request")
                code = str(error_value.get("code") or "WORKER_ERROR")
                details = error_value.get("details")
            else:
                message = str(error_value or "worker rejected request")
                code = "WORKER_ERROR"
                details = None
            raise RuntimeRPCError(message, code=code, details=details)
        result = response.get("result")
        if not isinstance(result, dict):
            raise RuntimeRPCError("worker result must be an object", code="INVALID_RESPONSE")
        return result

    def _receive_line(self, connection: socket.socket) -> bytes:
        chunks: list[bytes] = []
        received = 0
        while True:
            chunk = connection.recv(min(65536, self.response_limit_bytes + 1 - received))
            if not chunk:
                raise RuntimeRPCError("worker closed the socket without a response", code="INVALID_RESPONSE")
            newline = chunk.find(b"\n")
            if newline >= 0:
                chunks.append(chunk[:newline])
                received += newline
                break
            chunks.append(chunk)
            received += len(chunk)
            if received > self.response_limit_bytes:
                raise RuntimeRPCError("worker response exceeded size limit", code="INVALID_RESPONSE")
        if received > self.response_limit_bytes:
            raise RuntimeRPCError("worker response exceeded size limit", code="INVALID_RESPONSE")
        return b"".join(chunks)

    def health(self) -> dict[str, Any]:
        return self.call("health")

    def execute(
        self,
        request: Mapping[str, Any],
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        return self.call("execute", request, timeout_seconds=timeout_seconds)

    def shutdown(self) -> dict[str, Any]:
        return self.call("shutdown")


def _absolute_path(value: str | Path, *, field: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{field} must be absolute: {path}")
    return path.resolve(strict=False)


def _relative_below(path: Path, root: Path, *, field: str) -> str:
    """Map a scoring-container path to a sidecar-relative transport path."""

    path = path.resolve(strict=False)
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{field} is outside its allowed root {root}: {path}") from error
    # ``Path('.')`` serializes as '.', which is an explicitly supported RPC
    # path and keeps the request independent of the host mount location.
    return validate_relative_rpc_path(relative.as_posix(), field=field)


def _read_bounded_log(path: Path, *, expected_bytes: Any) -> str:
    """Read exactly the worker-bounded log, rejecting mismatched paths/sizes."""

    try:
        size = int(expected_bytes)
    except (TypeError, ValueError) as error:
        raise RuntimeRPCError(
            "worker returned invalid log metadata", code="INVALID_RESPONSE"
        ) from error
    if size < 0:
        raise RuntimeRPCError("worker returned invalid log size", code="INVALID_RESPONSE")
    try:
        with path.open("rb") as stream:
            payload = stream.read(size + 1)
    except OSError as error:
        raise RuntimeRPCError(
            f"could not read evaluation-tool log artifact: {path}",
            code="MISSING_ARTIFACT",
        ) from error
    if len(payload) != size:
        raise RuntimeRPCError(
            f"evaluation-tool log size did not match worker metadata: {path}",
            code="INVALID_RESPONSE",
        )
    return payload.decode("utf-8", errors="replace")


class SidecarRuntimeClient:
    """Typed multi-tool transport used by :class:`EvalToolManager`.

    The scoring container and each tool sidecar see the same immutable input
    and writable artifact trees at different absolute mount points.  Only
    validated paths relative to those trees cross the Unix socket.
    """

    def __init__(
        self,
        *,
        socket_dir: str | Path | None = None,
        scoring_root: str | Path | None = None,
        artifact_scoring_root: str | Path | None = None,
        sidecar_input_root: str | Path = DEFAULT_SIDECAR_INPUT_ROOT,
        sidecar_artifact_root: str | Path = DEFAULT_SIDECAR_ARTIFACT_ROOT,
        rpc_grace_seconds: float = DEFAULT_RPC_GRACE_SECONDS,
        stdout_limit_bytes: int | None = None,
        stderr_limit_bytes: int | None = None,
    ) -> None:
        self.socket_dir = _absolute_path(
            socket_dir
            or os.environ.get("AKA_EVAL_TOOL_SOCKET_DIR")
            or DEFAULT_SOCKET_DIR,
            field="socket_dir",
        )
        self.scoring_root = _absolute_path(
            scoring_root
            or os.environ.get("AKA_EVAL_TOOL_SCORING_ROOT")
            or os.environ.get("AGENT_KERNEL_ARENA_WORKDIR")
            or DEFAULT_SCORING_ROOT,
            field="scoring_root",
        )
        self.artifact_scoring_root = _absolute_path(
            artifact_scoring_root
            or os.environ.get("AKA_EVAL_TOOL_ARTIFACT_SCORING_ROOT")
            or DEFAULT_ARTIFACT_SCORING_ROOT,
            field="artifact_scoring_root",
        )
        self.sidecar_input_root = _absolute_path(
            sidecar_input_root, field="sidecar_input_root"
        )
        self.sidecar_artifact_root = _absolute_path(
            sidecar_artifact_root, field="sidecar_artifact_root"
        )
        if rpc_grace_seconds <= 0:
            raise ValueError("rpc_grace_seconds must be positive")
        for name, limit in (
            ("stdout_limit_bytes", stdout_limit_bytes),
            ("stderr_limit_bytes", stderr_limit_bytes),
        ):
            if limit is not None and (isinstance(limit, bool) or int(limit) < 0):
                raise ValueError(f"{name} must be a non-negative integer")
        self.rpc_grace_seconds = float(rpc_grace_seconds)
        self.stdout_limit_bytes = (
            int(stdout_limit_bytes) if stdout_limit_bytes is not None else None
        )
        self.stderr_limit_bytes = (
            int(stderr_limit_bytes) if stderr_limit_bytes is not None else None
        )

    def _client(self, tool: str, *, timeout_seconds: float = 30.0) -> UnixSocketRuntimeClient:
        return UnixSocketRuntimeClient.for_tool(
            tool,
            socket_dir=self.socket_dir,
            timeout_seconds=timeout_seconds,
        )

    def probe(self, tool: str, context: ToolContext) -> CapabilityCheck:
        try:
            health = self._client(tool).health()
        except (RuntimeRPCError, ValueError) as error:
            code = error.code if isinstance(error, RuntimeRPCError) else "UNAVAILABLE_RUNTIME"
            return CapabilityCheck.blocked(
                CapabilityState.UNAVAILABLE_RUNTIME,
                reason_code=code,
                detail=str(error),
            )
        if health.get("status") not in {"ready", "degraded"} or health.get("tool") != tool:
            return CapabilityCheck.blocked(
                CapabilityState.UNAVAILABLE_RUNTIME,
                reason_code="INVALID_RUNTIME_IDENTITY",
                detail=f"sidecar health response did not identify tool {tool!r}",
                health=health,
            )
        raw_evidence = health.get("evidence", {})
        if not isinstance(raw_evidence, Mapping):
            return CapabilityCheck.blocked(
                CapabilityState.UNAVAILABLE_RUNTIME,
                reason_code="INVALID_RUNTIME_EVIDENCE",
                detail="sidecar health evidence was not an object",
            )
        evidence = dict(raw_evidence)
        evidence["worker_status"] = health.get("status")
        expected_runtime_ref = context.runtime_ref
        actual_runtime_ref = evidence.get("runtime_ref")
        if expected_runtime_ref and actual_runtime_ref != expected_runtime_ref:
            return CapabilityCheck.blocked(
                CapabilityState.UNAVAILABLE_RUNTIME,
                reason_code="RUNTIME_REF_MISMATCH",
                detail=(
                    f"configured runtime_ref {expected_runtime_ref!r} does not match "
                    f"sidecar image ID {actual_runtime_ref!r}"
                ),
                **evidence,
            )
        required_assets = {
            "triton_fpsan": ("triton_fpsan",),
            "gpu_asan": (
                "asan_runtime_dir",
                "hip_asan_runtime",
                "host_asan_preload",
                "host_asan_lib_dir",
                "normal_rocm_lib_dir",
                "xnack_supported",
            ),
            "rocjitsu": ("rocjitsu_binary", "config_path"),
            "rocjitsu_waitcheck": (
                "waitcheck_binary",
                "waitcheck_capi_wrapper",
                "target_arch",
            ),
            "rocjitsu_consan": ("consan_hook", "target_arch", "gpu_arch"),
            "hip_fpsan": ("include_dir", "public_header", "hip_fpsan_headers"),
        }.get(tool, ())
        missing = [name for name in required_assets if not evidence.get(name)]
        if missing:
            return CapabilityCheck.blocked(
                CapabilityState.UNAVAILABLE_RUNTIME,
                reason_code="RUNTIME_ASSET_MISSING",
                detail=f"sidecar is missing required runtime evidence: {', '.join(missing)}",
                runtime="unix_socket_sidecar",
                protocol_version=health.get("protocol_version"),
                **evidence,
            )
        if tool in _BUILTIN_PROBES:
            manifest = evidence.get("probe_manifest")
            framework_valid = bool(
                evidence.get("framework_root") == _IMAGE_FRAMEWORK_ROOT
                and evidence.get("framework_source")
                in {"configured_image", "image_default"}
                and evidence.get("worker_module")
                == f"{_IMAGE_FRAMEWORK_ROOT}/src/eval_tools/worker.py"
                and _is_sha256(evidence.get("worker_module_sha256"))
                and isinstance(manifest, Mapping)
                and all(_is_sha256(manifest.get(name)) for name in _BUILTIN_PROBES[tool])
            )
            if not framework_valid:
                return CapabilityCheck.blocked(
                    CapabilityState.UNAVAILABLE_RUNTIME,
                    reason_code="INVALID_FRAMEWORK_PROVENANCE",
                    detail=(
                        "sidecar worker and startup probes were not loaded from the "
                        "image-owned evaluation framework"
                    ),
                    **evidence,
                )
        raw_required = context.options.get("positive_control_required", False)
        positive_required = (
            raw_required
            if isinstance(raw_required, bool)
            else str(raw_required).strip().lower() in {"1", "true", "yes", "on", "required"}
        )
        positive = evidence.get("positive_control")
        selected_positive = positive
        if tool == "gpu_asan" and isinstance(positive, Mapping):
            controls = positive.get("controls")
            if isinstance(controls, Mapping):
                is_triton = (
                    str(context.profile.language) == "triton"
                    or context.profile.framework == "triton"
                )
                selected_positive = controls.get("triton" if is_triton else "hip")
        if positive_required and (
            not isinstance(selected_positive, Mapping)
            or selected_positive.get("passed") is not True
        ):
            return CapabilityCheck.blocked(
                CapabilityState.UNAVAILABLE_RUNTIME,
                reason_code="POSITIVE_CONTROL_FAILED",
                detail=(
                    "required synthetic positive control did not pass: "
                    + (
                        str(selected_positive.get("detail") or selected_positive.get("kind"))
                        if isinstance(selected_positive, Mapping)
                        else "missing positive-control evidence"
                    )
                ),
                **evidence,
            )
        return CapabilityCheck.ready(
            runtime="unix_socket_sidecar",
            protocol_version=health.get("protocol_version"),
            worker_pid=health.get("pid"),
            **evidence,
        )

    def _map_cwd(self, invocation: ToolInvocation, context: ToolContext) -> dict[str, str]:
        cwd = _absolute_path(invocation.cwd, field="invocation.cwd")
        workspace = _absolute_path(context.workspace, field="context.workspace")
        # Plugins may only execute in the selected task workspace.  Merely being
        # under the broader repository input mount is intentionally insufficient.
        _relative_below(cwd, workspace, field="invocation.cwd")
        relative = _relative_below(cwd, self.scoring_root, field="invocation.cwd")
        return {"root": "input", "path": relative}

    def _map_artifact_dir(
        self, invocation: ToolInvocation, context: ToolContext
    ) -> tuple[str, Path]:
        selected = invocation.artifact_dir or context.artifact_dir
        artifact_dir = _absolute_path(selected, field="invocation.artifact_dir")
        context_artifact_dir = _absolute_path(
            context.artifact_dir, field="context.artifact_dir"
        )
        # An invocation may choose a child directory, but cannot redirect logs
        # into another task's report tree.
        _relative_below(
            artifact_dir,
            context_artifact_dir,
            field="invocation.artifact_dir",
        )
        relative = _relative_below(
            artifact_dir,
            self.artifact_scoring_root,
            field="invocation.artifact_dir",
        )
        return relative, artifact_dir

    def _translate_absolute_path(self, value: str) -> str:
        """Translate a known scoring mount path into its sidecar mount path."""

        if not value.startswith("/") or "\x00" in value:
            return value
        path = Path(value).resolve(strict=False)
        try:
            relative = path.relative_to(self.artifact_scoring_root)
        except ValueError:
            pass
        else:
            return str(self.sidecar_artifact_root / relative)
        try:
            relative = path.relative_to(self.scoring_root)
        except ValueError:
            return value
        return str(self.sidecar_input_root / relative)

    def _translate_argument(self, argument: str) -> str:
        translated = self._translate_absolute_path(argument)
        if translated != argument:
            return translated
        for prefix in ("-I", "-L"):
            if argument.startswith(prefix + "/"):
                return prefix + self._translate_absolute_path(argument[len(prefix) :])
        if "=" in argument:
            prefix, value = argument.split("=", 1)
            if value.startswith("/"):
                return prefix + "=" + self._translate_absolute_path(value)
        return argument

    def _translate_environment(self, environment: Mapping[str, str]) -> dict[str, str]:
        result: dict[str, str] = {}
        for key, value in environment.items():
            # Do not reinterpret colon-separated PATH/LD_LIBRARY_PATH values as
            # one filesystem path. Plugins should pass each tool-image path in
            # its native sidecar form.
            result[key] = (
                self._translate_absolute_path(value)
                if value.startswith("/") and ":" not in value
                else value
            )
        return result

    def execute(self, invocation: ToolInvocation, context: ToolContext) -> ExecutionRecord:
        if invocation.shell:
            raise ValueError(
                "sidecar invocations must use an argv vector; use an explicit "
                "['bash', '-lc', ...] command only in a trusted plugin"
            )
        cwd = self._map_cwd(invocation, context)
        artifact_relative, _artifact_dir = self._map_artifact_dir(invocation, context)
        request: dict[str, Any] = {
            "argv": [self._translate_argument(item) for item in invocation.command],
            "cwd": cwd,
            "artifact_dir": artifact_relative,
            "timeout_s": invocation.timeout_s,
            "env": self._translate_environment({**context.env, **invocation.env}),
        }
        if self.stdout_limit_bytes is not None:
            request["stdout_limit_bytes"] = self.stdout_limit_bytes
        if self.stderr_limit_bytes is not None:
            request["stderr_limit_bytes"] = self.stderr_limit_bytes

        response = self._client(
            invocation.tool,
            timeout_seconds=invocation.timeout_s + self.rpc_grace_seconds,
        ).execute(
            request,
            timeout_seconds=invocation.timeout_s + self.rpc_grace_seconds,
        )
        if response.get("tool") != invocation.tool:
            raise RuntimeRPCError(
                "worker result tool did not match invocation",
                code="INVALID_RESPONSE",
            )
        execution = response.get("execution")
        if not isinstance(execution, Mapping):
            raise RuntimeRPCError(
                "worker response omitted execution metadata", code="INVALID_RESPONSE"
            )

        logs: dict[str, str] = {}
        for stream_name in ("stdout", "stderr"):
            stream = execution.get(stream_name)
            if not isinstance(stream, Mapping):
                raise RuntimeRPCError(
                    f"worker response omitted {stream_name} metadata",
                    code="INVALID_RESPONSE",
                )
            relative_log = validate_relative_rpc_path(
                str(stream.get("path") or ""), field=f"{stream_name}.path"
            )
            log_path = (self.artifact_scoring_root / relative_log).resolve(strict=False)
            _relative_below(log_path, self.artifact_scoring_root, field=f"{stream_name}.path")
            # Also require each returned log to belong to this invocation's
            # artifact directory, not merely some other artifact in the run.
            expected_root = (self.artifact_scoring_root / artifact_relative).resolve(
                strict=False
            )
            _relative_below(log_path, expected_root, field=f"{stream_name}.path")
            logs[stream_name] = _read_bounded_log(
                log_path, expected_bytes=stream.get("bytes_written")
            )

        exit_code = execution.get("exit_code")
        signal_number = execution.get("signal")
        if execution.get("cleanup_required") is True:
            # Background descendants violate the invocation contract even when
            # the leader exited zero and containment successfully removed them.
            returncode = None
        elif exit_code is not None:
            try:
                returncode: int | None = int(exit_code)
            except (TypeError, ValueError) as error:
                raise RuntimeRPCError(
                    "worker returned invalid exit code", code="INVALID_RESPONSE"
                ) from error
        elif signal_number is not None:
            try:
                returncode = -int(signal_number)
            except (TypeError, ValueError) as error:
                raise RuntimeRPCError(
                    "worker returned invalid signal", code="INVALID_RESPONSE"
                ) from error
        else:
            returncode = None

        try:
            duration_s = float(execution.get("duration_ms", 0)) / 1000.0
        except (TypeError, ValueError) as error:
            raise RuntimeRPCError(
                "worker returned invalid duration", code="INVALID_RESPONSE"
            ) from error
        return ExecutionRecord(
            command=invocation.command,
            returncode=returncode,
            stdout=logs["stdout"],
            stderr=logs["stderr"],
            duration_s=duration_s,
            timed_out=bool(execution.get("timed_out", False)),
            metadata={
                **dict(invocation.metadata),
                "runtime": "unix_socket_sidecar",
                "tool": invocation.tool,
                "execution": dict(execution),
            },
        )


# Backward-compatible short alias for direct, single-socket RPC callers.  The
# manager-facing implementation is ``SidecarRuntimeClient``.
RuntimeClient = UnixSocketRuntimeClient


__all__ = [
    "RuntimeClient",
    "RuntimeRPCError",
    "SidecarRuntimeClient",
    "UnixSocketRuntimeClient",
    "socket_path_for_tool",
    "validate_relative_rpc_path",
]
