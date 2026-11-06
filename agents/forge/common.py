# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""GPU identity, engine result and supervised subprocess helpers for the Forge adapter."""

from __future__ import annotations

import json
import logging
import os
import re
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

FORGE_RESULT_SENTINEL = "__FORGE_RESULT__"

_GPU_TYPE_ALIASES = {
    # Arena historically accepts the family-style names below for the X SKUs.
    # KernelForge addresses KB records by exact hardware model, so normalize the
    # aliases before they become distinct, non-interoperable recipe identities.
    "mi300": "mi300x",
    "mi325": "mi325x",
}


def _normalize_gfx_arch(arch: str) -> str:
    """Normalize rocminfo/config variants to the Forge KB architecture token."""
    match = re.search(r"gfx[0-9a-f]+", str(arch or "").lower())
    if not match:
        raise ValueError(f"Invalid AMD GPU architecture: {arch!r}")
    return match.group(0)


def _resolve_gpu_arch(eval_config: dict[str, Any]) -> str:
    """Resolve the gfx arch for the forge run, reusing Arena's shared resolution.

    Priority mirrors ``src.preprocessing.setup_rocm_env`` so ``--gpu-target`` always
    matches the arch Arena actually compiles/runs for:

      1. the real hardware arch reported by ``rocminfo`` (most reliable); then
      2. the configured ``target_gpu_model`` looked up in the shared architecture
         map in ``default_cheatsheet.yaml`` (the single source of truth — covers
         MI300/MI325/MI355X, RDNA4->gfx1201, ...).

    An unknown model FAILS EXPLICITLY instead of silently assuming an arch: a
    mismatched arch (e.g. handing RDNA4 the gfx942 profile) produces invalid build
    flags, misleading agent guidance, and kernels tuned for the wrong ISA.
    """
    from src.preprocessing import _detect_gfx_arch_from_rocminfo, _resolve_gfx_arch

    detected = _detect_gfx_arch_from_rocminfo()
    if detected:
        return _normalize_gfx_arch(detected)

    model = str(eval_config.get("target_gpu_model", "")).strip()
    arch = _resolve_gfx_arch(model)
    if arch:
        return _normalize_gfx_arch(arch)

    raise ValueError(
        f"Cannot resolve a gfx arch for target_gpu_model={model!r}: it was not "
        "detected via rocminfo and is not defined under 'architecture' in "
        "src/prompts/cheatsheet/default_cheatsheet.yaml. Add the model there (with "
        "its gfx_arch) or run on the target GPU so rocminfo can report it."
    )


def _resolve_gpu_type(eval_config: dict[str, Any]) -> str:
    """Return Arena's hardware model in Forge's canonical KB token form."""
    raw = str(eval_config.get("target_gpu_model") or "").strip().lower()
    if not raw or not re.fullmatch(r"[a-z0-9][a-z0-9._+-]*", raw):
        raise ValueError(
            "target_gpu_model must be a non-empty hardware model token "
            f"for Forge KB identity; got {eval_config.get('target_gpu_model')!r}"
        )
    return _GPU_TYPE_ALIASES.get(raw, raw)


def _process_group_exists(pgid: int) -> bool:
    """Return whether a process group still has signalable members."""
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_process_group_exit(
    process: subprocess.Popen,
    pgid: int,
    timeout: float,
) -> bool:
    """Wait for the complete group to exit, reaping its leader along the way."""
    deadline = time.monotonic() + timeout
    while True:
        process.poll()
        if not _process_group_exists(pgid):
            return True

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(0.1, remaining))


def _terminate_process_group(
    process: subprocess.Popen,
    logger: logging.Logger,
    term_timeout: float = 10,
    kill_timeout: float = 5,
) -> None:
    """Signal the supervised Forge process group and wait for shutdown.

    The subprocess is launched in its own session (``start_new_session=True``),
    so its PID is the process-group leader. Signal the supervisor and members
    still in its group; the supervisor's subreaper cleans up descendants that
    created separate sessions. killpg alone cannot reach those descendants.
    """
    try:
        pgid = os.getpgid(process.pid)
    except ProcessLookupError:
        return  # already exited

    def _signal_group(sig: int) -> bool:
        try:
            os.killpg(pgid, sig)
            return True
        except ProcessLookupError:
            return False  # group already gone

    if not _signal_group(signal.SIGTERM):
        return

    if _wait_for_process_group_exit(process, pgid, term_timeout):
        return

    logger.warning("Force killing forge process group (SIGKILL)")
    if not _signal_group(signal.SIGKILL):
        return

    if not _wait_for_process_group_exit(process, pgid, kill_timeout):
        logger.warning("Forge process group did not exit even after SIGKILL")


def _read_forge_result(result_json: Path, stdout: str) -> dict[str, Any] | None:
    """Read the structured result file, then fall back to the stdout sentinel."""
    try:
        result = json.loads(result_json.read_text())
        if isinstance(result, dict):
            return result
    except (OSError, json.JSONDecodeError):
        pass
    matches = re.findall(
        re.escape(FORGE_RESULT_SENTINEL)
        + r"(.*?)"
        + re.escape(FORGE_RESULT_SENTINEL),
        stdout,
        flags=re.DOTALL,
    )
    for payload in reversed(matches):
        try:
            result = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(result, dict):
            return result
    return None


def run_forge_subprocess(
    cmd_parts: list[str],
    *,
    workspace: str,
    env: dict[str, str],
    timeout_seconds: int,
    logger: logging.Logger,
) -> tuple[subprocess.Popen, list[str], list[str], bool]:
    """Stream a KernelForge subprocess to the log and hard-kill it on timeout.

    The adapter supplies a supervised entry point, launched with shell=False
    in a new session. On timeout, signal its group and let the supervisor reap
    the complete child tree, including descendants in separate sessions, before
    returning to candidate delivery. Callers must retain that supervision:
    creating a session here does not keep all future descendants in one group.
    """
    process = subprocess.Popen(
        cmd_parts,
        shell=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=workspace,
        env=env,
        bufsize=1,
        start_new_session=True,
    )

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    def read_stream(stream, sink, prefix, log_func):
        try:
            for line in iter(stream.readline, ""):
                if not line:
                    break
                text = line.rstrip()
                if text:
                    sink.append(text)
                    log_func(f"{prefix} {text}")
        finally:
            stream.close()

    threads = [
        threading.Thread(
            target=read_stream,
            args=(process.stdout, stdout_lines, "[FORGE]", logger.info),
            daemon=True,
        ),
        threading.Thread(
            target=read_stream,
            args=(process.stderr, stderr_lines, "[FORGE STDERR]", logger.warning),
            daemon=True,
        ),
    ]
    for thread in threads:
        thread.start()

    timed_out = False
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        logger.warning(
            f"Forge run timed out after {timeout_seconds}s; terminating process group"
        )
        _terminate_process_group(process, logger)
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.error(
                "Forge process leader did not exit after process-group termination"
            )

    for thread in threads:
        thread.join(timeout=1)

    return process, stdout_lines, stderr_lines, timed_out
