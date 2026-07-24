#!/usr/bin/env python3
"""Stable GEAK v4 kernel-workflow runner for external orchestrators.

The GEAK JavaScript workflow can only execute inside Claude Code's dynamic
``Workflow`` runtime.  This module keeps that volatile SDK/tool lifecycle out of
the Arena launcher:

* map a versioned handoff onto ``kernel_workflow.js`` arguments;
* pin a known evaluation directory for completion/recovery;
* keep the SDK client alive when Workflow runs as a background task;
* recover the authoritative result from on-disk GEAK artifacts; and
* never let GEAK write directly to the Arena workspace.

The command is intentionally usable in ``--dry-run`` mode without importing the
Claude Agent SDK or contacting a model.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import stat
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
ALLOWED_TOOLS = ["Workflow", "Bash", "Read", "Write"]
VALID_EFFORTS = {"low", "medium", "high", "xhigh", "max"}
DEFAULT_SETTINGS = {"enableWorkflows": True, "ultracode": True}
_JSON_SIZE_LIMIT = 8 * 1024 * 1024
_SDK_OUTPUT_FILE_LIMIT = 8 * 1024 * 1024
_TRANSCRIPT_SIZE_LIMIT = 8 * 1024 * 1024
_TRANSCRIPT_JSON_LINE_LIMIT = 64


class HandoffError(ValueError):
    """The caller supplied an invalid GEAK handoff."""


def _open_directory_fd(
    path: Path | str,
    *,
    parent_fd: int | None = None,
) -> int:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, dir_fd=parent_fd)
    metadata = os.fstat(descriptor)
    if not stat.S_ISDIR(metadata.st_mode):
        os.close(descriptor)
        raise OSError(f"not a directory: {path}")
    return descriptor


def _atomic_write_json(
    path: Path,
    payload: dict[str, Any],
    *,
    directory_fd: int | None = None,
) -> None:
    """Write JSON without following attacker-created file symlinks.

    When ``directory_fd`` is supplied, the write stays pinned to that already
    opened directory even if Workflow renames or replaces its pathname.
    """
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    owned_directory_fd = directory_fd is None
    if directory_fd is None:
        directory_fd = _open_directory_fd(path.parent)

    temporary_fd = -1
    temporary_name = ""
    try:
        proc_directory = Path(f"/proc/self/fd/{directory_fd}")
        temporary_fd, temporary_path = tempfile.mkstemp(
            prefix=f".{path.name}.tmp.",
            dir=proc_directory,
        )
        temporary_name = Path(temporary_path).name
        metadata = os.fstat(temporary_fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise OSError("atomic JSON temporary is not a private regular file")
        with os.fdopen(temporary_fd, "wb", closefd=True) as stream:
            temporary_fd = -1
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(
            temporary_name,
            path.name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
        )
        temporary_name = ""
        os.fsync(directory_fd)
    finally:
        if temporary_fd >= 0:
            os.close(temporary_fd)
        if temporary_name:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        if owned_directory_fd:
            os.close(directory_fd)


def _read_bounded_text(path: Path, size_limit: int) -> str | None:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return None
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size > size_limit
        ):
            return None
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            content = stream.read(size_limit + 1)
        if len(content) > size_limit:
            return None
        return content.decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    finally:
        os.close(descriptor)


def _read_json(path: Path) -> dict[str, Any] | None:
    content = _read_bounded_text(path, _JSON_SIZE_LIMIT)
    if content is None:
        return None
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def load_handoff(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    if value is None:
        raise HandoffError(f"handoff is not a readable JSON object: {path}")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise HandoffError(
            f"unsupported handoff schema_version={value.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    return value


def _absolute_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise HandoffError(f"{field} must be a non-empty absolute path")
    path = Path(value)
    if not path.is_absolute():
        raise HandoffError(f"{field} must be absolute: {path}")
    return path.resolve()


def _positive_int(value: Any, field: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise HandoffError(f"{field} must be an integer") from exc
    if parsed <= 0:
        raise HandoffError(f"{field} must be positive")
    return parsed


def _nonnegative_float(value: Any, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise HandoffError(f"{field} must be numeric") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise HandoffError(f"{field} must be finite and non-negative")
    return parsed


def _gpu_ids(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        value = ",".join(str(item) for item in value)
    text = str(value if value is not None else "0")
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if not parts or any(not part.isdigit() for part in parts):
        raise HandoffError(f"gpu_ids must be comma-separated non-negative integers: {text!r}")
    return ",".join(parts)


def map_workflow_args(handoff: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    """Validate a handoff and return ``(script_path, workflow_args)``."""
    kernel_path = _absolute_path(handoff.get("kernel_path"), "kernel_path")
    workflow_dir = _absolute_path(handoff.get("workflow_dir"), "workflow_dir")
    eval_dir = _absolute_path(handoff.get("eval_dir"), "eval_dir")
    exp_root = _absolute_path(
        handoff.get("exp_root") or str(eval_dir.parent),
        "exp_root",
    )
    script_path = workflow_dir / "kernel_workflow.js"

    if not kernel_path.is_dir():
        raise HandoffError(f"kernel_path is not a directory: {kernel_path}")
    if not script_path.is_file():
        raise HandoffError(f"GEAK kernel workflow not found: {script_path}")
    if eval_dir.exists() and (
        not eval_dir.is_dir() or any(eval_dir.iterdir())
    ):
        raise HandoffError(f"eval_dir must be absent or an empty directory: {eval_dir}")
    for path, field in ((eval_dir, "eval_dir"), (exp_root, "exp_root")):
        try:
            path.relative_to(kernel_path)
        except ValueError:
            pass
        else:
            raise HandoffError(
                f"{field} must not be inside kernel_path; GEAK copies kernel_path "
                f"and would recursively copy its own outputs: {path}"
            )

    args: dict[str, Any] = {
        "kernel_path": str(kernel_path),
        "workflow_dir": str(workflow_dir),
        "eval_dir": str(eval_dir),
        "exp_root": str(exp_root),
        "gpu_ids": _gpu_ids(handoff.get("gpu_ids", "0")),
        "budget": _positive_int(handoff.get("budget", 6), "budget"),
        "min_improve": _nonnegative_float(
            handoff.get("min_improve", 0.02),
            "min_improve",
        ),
        "deep_cost": _positive_int(handoff.get("deep_cost", 2), "deep_cost"),
        "mode": "optimize",
        # Arena owns the task workspace and applies only a policy-checked patch.
        "apply_to_original": "false",
    }
    task = handoff.get("task")
    if task:
        args["task"] = str(task)
    if bool(handoff.get("use_expert_skills", False)):
        args["use_expert_skills"] = "true"
    return script_path, args


def build_prompt(script_path: Path, workflow_args: dict[str, Any]) -> str:
    eval_dir = workflow_args["eval_dir"]
    return (
        "Invoke the Workflow tool exactly once with:\n"
        f'  scriptPath: "{script_path}"\n'
        f"  args: {json.dumps(workflow_args, ensure_ascii=False)}\n"
        "Run the complete GEAK kernel pipeline through independent Director "
        "validation. Do not edit the original kernel_path directly; "
        "apply_to_original is false and the caller owns patch import. When the "
        "Workflow finishes, write its exact full return object as compact JSON to "
        f'"{eval_dir}/workflow_return.json", then print exactly that compact JSON '
        "as the final line and print nothing after it."
    )


def _iter_message_text(message: Any) -> Iterable[str]:
    """Yield text fragments from SDK objects across supported SDK shapes."""
    if message is None:
        return
    if isinstance(message, str):
        if message.strip():
            yield message
        return
    if isinstance(message, dict):
        for key in ("result", "text", "summary"):
            value = message.get(key)
            if isinstance(value, str) and value.strip():
                yield value
        content = message.get("content")
        if isinstance(content, str):
            if content.strip():
                yield content
        elif isinstance(content, (list, tuple)):
            for item in content:
                yield from _iter_message_text(item)
        return

    for attribute in ("result", "text", "summary"):
        value = getattr(message, attribute, None)
        if isinstance(value, str) and value.strip():
            yield value
    content = getattr(message, "content", None)
    if isinstance(content, str):
        if content.strip():
            yield content
    elif isinstance(content, (list, tuple)):
        for item in content:
            yield from _iter_message_text(item)


def _finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _matches_pinned_patch(value: Any, eval_dir: Path) -> bool:
    return isinstance(value, str) and value == str(eval_dir / "final_patch.diff")


def _valid_workflow_return(
    value: Any,
    eval_dir: Path,
    *,
    require_pinned_patch: bool = False,
) -> bool:
    if not isinstance(value, dict):
        return False
    raw_eval_dir = value.get("eval_dir")
    if not isinstance(raw_eval_dir, str):
        return False
    matches = raw_eval_dir == str(eval_dir)
    status = value.get("validation_status")
    patch_contract_ok = (
        not require_pinned_patch
        or status not in {"accepted", "flagged"}
        or _matches_pinned_patch(value.get("final_patch"), eval_dir)
    )
    return (
        matches
        and isinstance(status, str)
        and _finite_number(value.get("final_geomean"))
        and isinstance(value.get("final_patch"), str)
        and patch_contract_ok
        and (
            "workload_aligned" not in value
            or isinstance(value.get("workload_aligned"), bool)
        )
    )


def _valid_director_validation(
    value: Any,
    eval_dir: Path | None = None,
) -> bool:
    valid = (
        isinstance(value, dict)
        and value.get("validation_status") in {"accepted", "flagged"}
        and value.get("correctness") in {"pass", "fail"}
        and _finite_number(value.get("director_verified_speedup_geomean"))
        and value.get("applied_to_original") in {"true", "false"}
        and isinstance(value.get("final_patch"), str)
    )
    return valid and (
        eval_dir is None
        or _matches_pinned_patch(value.get("final_patch"), eval_dir)
    )


def _terminal_artifact_exists(eval_dir: Path) -> bool:
    workflow_return = _read_json(eval_dir / "workflow_return.json")
    director_validation = _read_json(eval_dir / "director_validation.json")
    return _valid_workflow_return(
        workflow_return,
        eval_dir,
        require_pinned_patch=True,
    ) or _valid_director_validation(director_validation, eval_dir)


def _extract_workflow_return(transcript: str, expected_eval_dir: Path) -> dict[str, Any] | None:
    """Read the final compact JSON line without quadratic brace scanning."""
    lines = transcript.splitlines()
    for raw_line in reversed(lines[-_TRANSCRIPT_JSON_LINE_LIMIT:]):
        line = raw_line.strip()
        if not line or len(line.encode("utf-8")) > _JSON_SIZE_LIMIT:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if _valid_workflow_return(
            value,
            expected_eval_dir,
            require_pinned_patch=True,
        ):
            return value
    return None


def _completed_producer_error(
    state: dict[str, bool],
    pending: set[str],
    producer_error: list[BaseException],
) -> BaseException | None:
    if not state["producer_done"]:
        return None
    if producer_error:
        return producer_error[0]
    if pending:
        return RuntimeError(
            "Claude SDK message stream ended with unfinished GEAK tasks: "
            f"{sorted(pending)}"
        )
    if not state["result_seen"]:
        return RuntimeError(
            "Claude SDK message stream ended without a ResultMessage or "
            "a valid GEAK terminal artifact"
        )
    return None


def invoke_via_sdk(
    prompt: str,
    *,
    workflow_dir: Path,
    eval_dir: Path,
    model: str,
    effort: str,
    settings: str,
    cli_path: str,
    timeout_seconds: int,
    done_grace_seconds: float,
    done_poll_seconds: float,
) -> str:
    """Invoke Claude Code while surviving synchronous and background Workflows."""
    try:
        import anyio
        from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
    except ImportError as exc:
        raise RuntimeError(
            "claude_agent_sdk is required for reliable GEAK Workflow lifecycle "
            "handling; run `make docker-setup-geak` first"
        ) from exc

    option_extras: dict[str, Any] = {}
    if effort in VALID_EFFORTS:
        option_extras["effort"] = effort
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        raise RuntimeError(
            "GEAK v4 refuses to run Claude as root without a real OS sandbox; "
            "use the Docker runner's non-root host UID mapping"
        )
    sdk_env = {
        "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
    }

    options = ClaudeAgentOptions(
        model=model,
        allowed_tools=ALLOWED_TOOLS,
        permission_mode="bypassPermissions",
        settings=settings,
        extra_args=option_extras,
        cwd=str(workflow_dir),
        env=sdk_env,
        **({"cli_path": cli_path} if cli_path else {}),
    )

    async def _run() -> str:
        chunks: list[str] = []
        captured_chars = 0
        pending: set[str] = set()
        state = {
            "background_started": False,
            "terminal_task_seen": False,
            "result_seen": False,
            "producer_done": False,
        }

        with anyio.fail_after(timeout_seconds):
            async with ClaudeSDKClient(options=options) as client:
                await client.query(prompt)
                producer_error: list[BaseException] = []

                async def _receive() -> None:
                    nonlocal captured_chars
                    try:
                        async for message in client.receive_messages():
                            for text in _iter_message_text(message):
                                remaining = _TRANSCRIPT_SIZE_LIMIT - captured_chars
                                if remaining > 0:
                                    retained = text[:remaining]
                                    chunks.append(retained)
                                    captured_chars += len(retained)
                                compact = " ".join(text[:2000].split())
                                if compact:
                                    print(
                                        f"[GEAK SDK] {compact[:500]}",
                                        file=sys.stderr,
                                        flush=True,
                                    )

                            name = type(message).__name__
                            if name == "TaskStartedMessage":
                                task_id = getattr(message, "task_id", None)
                                if task_id:
                                    pending.add(str(task_id))
                                    state["background_started"] = True
                            elif name == "TaskNotificationMessage":
                                state["terminal_task_seen"] = True
                                task_id = getattr(message, "task_id", None)
                                if task_id:
                                    pending.discard(str(task_id))
                                output_file = getattr(message, "output_file", None)
                                if output_file:
                                    output = _read_bounded_text(
                                        Path(output_file),
                                        _SDK_OUTPUT_FILE_LIMIT,
                                    )
                                    remaining = _TRANSCRIPT_SIZE_LIMIT - captured_chars
                                    if output and remaining > 0:
                                        retained = output[:remaining]
                                        chunks.append(retained)
                                        captured_chars += len(retained)
                            elif name == "ResultMessage":
                                state["result_seen"] = True
                    except BaseException as exc:
                        producer_error.append(exc)
                    finally:
                        state["producer_done"] = True

                async with anyio.create_task_group() as task_group:
                    task_group.start_soon(_receive)
                    weak_deadline: float | None = None
                    while True:
                        # Match GEAK's lifecycle contract: a task notification
                        # is authoritative over an on-disk marker. The Director
                        # writes its JSON before StructuredOutput returns and
                        # the JS Workflow assembles its final result.
                        if pending and not state["producer_done"]:
                            await anyio.sleep(max(0.1, done_poll_seconds))
                            continue
                        if _terminal_artifact_exists(eval_dir):
                            break
                        if state["result_seen"] and not state["background_started"]:
                            break
                        completion_error = _completed_producer_error(
                            state,
                            pending,
                            producer_error,
                        )
                        if completion_error is not None:
                            raise completion_error

                        weak_terminal = (
                            state["background_started"]
                            and state["result_seen"]
                            and not pending
                            and (
                                state["terminal_task_seen"]
                                or state["producer_done"]
                            )
                        )
                        if weak_terminal and weak_deadline is None:
                            weak_deadline = (
                                time.monotonic() + max(0.0, done_grace_seconds)
                            )
                        if weak_deadline is not None and time.monotonic() >= weak_deadline:
                            break
                        if (
                            state["producer_done"]
                            and not state["background_started"]
                            and not state["result_seen"]
                        ):
                            break
                        await anyio.sleep(max(0.1, done_poll_seconds))
                    task_group.cancel_scope.cancel()
        return "\n".join(chunks)[:_TRANSCRIPT_SIZE_LIMIT]

    return anyio.run(_run)


def _number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def normalize_result(
    eval_dir: Path,
    workflow_return: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the stable runner result from GEAK's authoritative artifacts."""
    disk_return_path = eval_dir / "workflow_return.json"
    disk_return = _read_json(disk_return_path)
    disk_return_present = disk_return_path.exists() or disk_return_path.is_symlink()
    disk_return_valid = _valid_workflow_return(
        disk_return,
        eval_dir,
        require_pinned_patch=True,
    )
    workflow_contract_invalid = disk_return_present and not disk_return_valid
    if disk_return_valid:
        workflow_return = disk_return
    elif not _valid_workflow_return(
        workflow_return,
        eval_dir,
        require_pinned_patch=True,
    ):
        workflow_return = {}
    validation = _read_json(eval_dir / "director_validation.json") or {}

    validation_status = str(
        validation.get("validation_status")
        or workflow_return.get("validation_status")
        or "unknown"
    ).lower()
    correctness = str(validation.get("correctness") or "unknown").lower()
    workload_aligned = workflow_return.get("workload_aligned") is True
    weighted = _number(validation.get("director_verified_speedup_weighted"))
    geomean = _number(validation.get("director_verified_speedup_geomean"))
    workflow_speedup = _number(workflow_return.get("final_speedup"))
    speedup = (
        weighted
        if workload_aligned and weighted is not None
        else (geomean if geomean is not None else workflow_speedup)
    )

    patch_path = eval_dir / "final_patch.diff"
    patch_exists = patch_path.is_file() and patch_path.stat().st_size > 0

    accepted = validation_status == "accepted" and correctness == "pass"
    gained = speedup is not None and speedup > 1.0
    director_valid = _valid_director_validation(validation, eval_dir)
    primary_metric_valid = speedup is not None
    if (
        accepted
        and gained
        and patch_exists
        and director_valid
        and not workflow_contract_invalid
    ):
        status = "ok"
    elif accepted and director_valid and workflow_contract_invalid:
        status = "error"
    elif accepted and director_valid and not primary_metric_valid:
        status = "error"
    elif accepted and director_valid and not gained:
        status = "no_gain"
    elif validation_status == "flagged" or correctness == "fail":
        status = "rejected"
    else:
        status = "error"

    if not director_valid and validation_status in {"accepted", "flagged"}:
        reason = "GEAK Director artifact is missing or invalid"
    elif workflow_contract_invalid:
        reason = "GEAK workflow return artifact is present but invalid"
    elif not accepted:
        reason = (
            f"GEAK validation did not accept the candidate "
            f"(status={validation_status}, correctness={correctness})"
        )
    elif not primary_metric_valid:
        metric = "weighted" if workload_aligned else "geomean"
        reason = f"GEAK Director artifact has no finite {metric} speedup"
    elif not gained:
        reason = f"GEAK did not verify a speedup above 1.0x (speedup={speedup})"
    elif not patch_exists:
        reason = f"GEAK accepted a gain but produced no non-empty patch at {patch_path}"
    else:
        reason = ""

    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "eval_dir": str(eval_dir),
        "validation_status": validation_status,
        "correctness": correctness,
        "workload_aligned": workload_aligned,
        "final_speedup": speedup,
        "final_geomean": geomean,
        "final_weighted": weighted,
        "final_patch": str(patch_path),
        "director_final_patch": validation.get("final_patch"),
        "workflow_final_patch": workflow_return.get("final_patch"),
        "report_path": str(
            workflow_return.get("report_path")
            or eval_dir / "tech_lead_report.md"
        ),
        "budget_used": workflow_return.get("budget_used"),
        "budget_total": workflow_return.get("budget_total"),
        "applied_to_original": validation.get("applied_to_original", "unknown"),
        "reason": reason,
    }


def run_handoff(handoff: dict[str, Any]) -> dict[str, Any]:
    script_path, workflow_args = map_workflow_args(handoff)
    eval_dir = Path(workflow_args["eval_dir"])
    eval_dir.parent.mkdir(parents=True, exist_ok=True)
    prompt = build_prompt(script_path, workflow_args)

    timeout_seconds = _positive_int(
        handoff.get("timeout_seconds", 43200),
        "timeout_seconds",
    )
    model = str(handoff.get("model") or "claude-opus-4-8")
    effort = str(handoff.get("effort") or "ultracode")
    settings_value = handoff.get("settings", DEFAULT_SETTINGS)
    settings = (
        settings_value
        if isinstance(settings_value, str)
        else json.dumps(settings_value)
    )
    cli_path = str(
        handoff.get("claude_cli_path")
        or os.environ.get("GEAK_CLAUDE_BIN")
        or shutil.which("claude")
        or ""
    ).strip()
    if not cli_path:
        raise RuntimeError("Claude Code CLI not found; cannot run GEAK Workflow")
    done_grace = _nonnegative_float(
        handoff.get("done_grace_seconds", 1800),
        "done_grace_seconds",
    )
    done_poll = _nonnegative_float(
        handoff.get("done_poll_seconds", 5),
        "done_poll_seconds",
    )

    run_directory_fd = _open_directory_fd(eval_dir.parent)
    try:
        transcript = ""
        invocation_error: Exception | None = None
        try:
            transcript = invoke_via_sdk(
                prompt,
                workflow_dir=script_path.parent,
                eval_dir=eval_dir,
                model=model,
                effort=effort,
                settings=settings,
                cli_path=cli_path,
                timeout_seconds=timeout_seconds,
                done_grace_seconds=done_grace,
                done_poll_seconds=done_poll,
            )
        except Exception as exc:  # disk recovery below may still prove completion
            invocation_error = exc

        parsed_return = _extract_workflow_return(transcript, eval_dir)
        if parsed_return:
            eval_directory_fd = _open_directory_fd(
                eval_dir.name,
                parent_fd=run_directory_fd,
            )
            try:
                try:
                    os.stat(
                        "workflow_return.json",
                        dir_fd=eval_directory_fd,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    _atomic_write_json(
                        eval_dir / "workflow_return.json",
                        parsed_return,
                        directory_fd=eval_directory_fd,
                    )
            finally:
                os.close(eval_directory_fd)

        if _terminal_artifact_exists(eval_dir):
            result = normalize_result(eval_dir, parsed_return)
            if invocation_error:
                result["recovered_after_error"] = type(invocation_error).__name__
            return result
        if invocation_error:
            raise invocation_error
        raise RuntimeError(
            "GEAK Workflow exited without workflow_return.json or "
            f"director_validation.json under {eval_dir}"
        )
    finally:
        os.close(run_directory_fd)


def _dry_run_result(
    handoff: dict[str, Any],
    script_path: Path,
    workflow_args: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "dry_run",
        "script_path": str(script_path),
        "workflow_args": workflow_args,
        "prompt": build_prompt(script_path, workflow_args),
        "model": str(handoff.get("model") or "claude-opus-4-8"),
        "effort": str(handoff.get("effort") or "ultracode"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run GEAK v4 kernel_workflow")
    parser.add_argument("handoff", type=Path)
    parser.add_argument("result", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate/map the handoff without importing the SDK or invoking Claude",
    )
    namespace = parser.parse_args(argv)

    result_directory_fd = _open_directory_fd(namespace.result.parent)
    try:
        result: dict[str, Any]
        try:
            handoff = load_handoff(namespace.handoff)
            if namespace.dry_run:
                script_path, workflow_args = map_workflow_args(handoff)
                result = _dry_run_result(handoff, script_path, workflow_args)
            else:
                result = run_handoff(handoff)
        except Exception as exc:
            result = {
                "schema_version": SCHEMA_VERSION,
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            _atomic_write_json(
                namespace.result,
                result,
                directory_fd=result_directory_fd,
            )
            print(json.dumps(result, ensure_ascii=False), flush=True)
            return 1

        _atomic_write_json(
            namespace.result,
            result,
            directory_fd=result_directory_fd,
        )
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 1 if result.get("status") == "error" else 0
    finally:
        os.close(result_directory_fd)


if __name__ == "__main__":
    raise SystemExit(main())
