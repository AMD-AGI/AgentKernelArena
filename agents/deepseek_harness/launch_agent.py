# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Run the pinned DeepSeek Harness headless CLI in an Arena task workspace."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Any

import yaml

from agents import register_agent
from agents.run_budget import append_run_budget
from src.module_registration import AgentType, load_prompt_builder
from src.process_control import stop_process_group as _stop_process_group
from src.prompts.agent_budget import time_budget_guidance
from src.runtime_env import build_subprocess_env
from src.tools.agent_cost import report_agent_usage
from src.tools.agent_time_budget import (
    SessionOutcome, raise_for_sessions, resolve_budget_mode, run_sessions,
)


def _load_config(eval_config: dict[str, Any] | None = None) -> dict[str, Any]:
    config = yaml.safe_load(Path(__file__).with_name("agent_config.yaml").read_text())
    if not isinstance(config, dict):
        raise ValueError("DeepSeek Harness agent_config.yaml must be a mapping")
    overrides = (eval_config or {}).get("agent", {})
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, dict):
        raise ValueError("DeepSeek Harness agent must be a mapping")
    for name in (
        "model", "reasoning_effort", "protocol", "base_url", "max_tokens",
        "max_iterations", "timeout_seconds", "python_path", "budget_mode",
    ):
        if name in overrides:
            config[name] = overrides[name]
    for name in ("cli_version", "model"):
        if not isinstance(config.get(name), str) or not config[name].strip():
            raise ValueError(f"DeepSeek Harness {name} must be a nonempty string")
    for name in ("timeout_seconds", "max_tokens", "max_iterations"):
        value = config.get(name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"DeepSeek Harness {name} must be a positive integer")
    if config.get("reasoning_effort") not in {"off", "low", "high", "max"}:
        raise ValueError("Unsupported DeepSeek Harness reasoning_effort")
    if config.get("protocol") not in {"messages", "chat-completions"}:
        raise ValueError("Unsupported DeepSeek Harness protocol")
    config["budget_mode"] = resolve_budget_mode(config)
    return config


def _environment(config: dict[str, Any]) -> dict[str, str]:
    env = build_subprocess_env(config.get("python_path"))
    # Do not inherit a host profile, experimental tools mode, or telemetry policy.
    for name in list(env):
        if name.startswith("DSH_"):
            del env[name]
    env["DSH_PERMISSION_MODE"] = "danger-full-access"
    env["DSH_TELEMETRY_MODE"] = "DISABLED"
    env["DSH_TELEMETRY_DISABLED"] = "1"
    return env


def _preflight(config: dict[str, Any], env: dict[str, str]) -> tuple[str, str]:
    binary = shutil.which("dsh", path=env.get("PATH"))
    if binary is None:
        raise RuntimeError(
            "dsh not found; install the pinned version described in "
            "agents/deepseek_harness/README.md"
        )
    if not env.get("DEEPSEEK_API_KEY", "").strip():
        raise RuntimeError("DeepSeek Harness requires DEEPSEEK_API_KEY")
    result = subprocess.run(
        [binary, "--version"], env=env, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=15, check=False,
    )
    version = result.stdout.strip()
    if result.returncode != 0 or version != config["cli_version"]:
        # Do not echo arbitrary CLI output, which may include credentials.
        raise RuntimeError(
            "DeepSeek Harness CLI version does not match cli_version in "
            "agents/deepseek_harness/agent_config.yaml; reinstall the pinned version"
        )
    return binary, version


def check_installation() -> str:
    """Offline Docker preflight: binary/version and credential presence only."""
    config = _load_config()
    _, version = _preflight(config, _environment(config))
    return f"dsh_version={version} credential=present (API access not checked)"


def _patch(config: dict[str, Any]) -> list[dict[str, Any]]:
    provider = {
        "apiKeyEnv": "DEEPSEEK_API_KEY",
        "protocol": config["protocol"],
        "reasoningEffort": config["reasoning_effort"],
        "maxTokens": config["max_tokens"],
    }
    if config.get("base_url"):
        provider["baseURL"] = config["base_url"]
    return [
        {"id": "agent-default-model", "config": {
            "provider": "deepseek-official", "model": config["model"],
        }},
        {"id": "llm-deepseek", "config": provider},
        # Separate from OTel: the base bundle contributes session logs to API calls.
        {"id": "session-log-deepseek", "config": {"enabled": False}},
    ]


def _final_usage(lines: list[str]) -> dict[str, Any] | None:
    """Token usage from the --json final event, if this CLI version emits one.

    The pinned CLI does not document usage or dollars; absence is reported as
    unknown rather than estimated.
    """
    usage = None
    for line in lines:
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and event.get("type") == "final" and isinstance(event.get("usage"), dict):
            usage = event["usage"]
    return usage


def _session_id(lines: list[str]) -> str | None:
    """Read the opaque identity from the headless stream's opening event."""
    for line in lines:
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and event.get("type") == "session":
            value = event.get("sessionId")
            if isinstance(value, str) and value.strip():
                return value
    return None


def _run(
    command: list[str], workspace: Path, state: Path,
    env: dict[str, str], timeout: float, logger: logging.Logger, model: str | None = None,
) -> tuple[str, SessionOutcome]:
    """Run one headless process and stop its tool children before returning."""
    started = time.monotonic()
    outputs: dict[str, list[str]] = {"stdout": [], "stderr": []}
    secret = env.get("DEEPSEEK_API_KEY", "")

    def capture(stream, name: str) -> None:
        with stream, (state / f"{name}.log").open("w", encoding="utf-8") as log:
            for line in stream:
                redacted = line.replace(secret, "[REDACTED]") if secret else line
                outputs[name].append(redacted)
                log.write(redacted)
                log.flush()
                logger.info("[DSH %s] %s", name, redacted.rstrip()[:500])

    # File-backed stdin avoids argv length limits and blocking writes of large prompts.
    with (state / "prompt.txt").open(encoding="utf-8") as prompt:
        process = subprocess.Popen(
            command, cwd=workspace, env=env, stdin=prompt,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            start_new_session=True,
        )
    threads = [
        threading.Thread(target=capture, args=(process.stdout, "stdout"), daemon=True),
        threading.Thread(target=capture, args=(process.stderr, "stderr"), daemon=True),
    ]
    for thread in threads:
        thread.start()
    timed_out = False
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
    finally:
        _stop_process_group(process)
        for thread in threads:
            thread.join(timeout=5)
        report_agent_usage(
            usage=_final_usage(outputs["stdout"]), usage_scope="invocation",
            session_id=_session_id(outputs["stdout"]),
            source="deepseek_harness --json final", model=model, timed_out=timed_out,
            seconds=time.monotonic() - started,
        )
    output = "".join(outputs["stdout"]) + "".join(outputs["stderr"])
    return output, SessionOutcome(
        timed_out=timed_out, returncode=process.returncode,
        session_id=_session_id(outputs["stdout"]), seconds=time.monotonic() - started,
    )


@register_agent("deepseek_harness")
def launch_agent(eval_config: dict, task_config_dir: str, workspace: str) -> str:
    config = _load_config(eval_config)
    env = _environment(config)
    binary, version = _preflight(config, env)
    logger = logging.getLogger(__name__)
    root = Path(workspace).resolve(strict=True)
    builder = load_prompt_builder(AgentType.DEEPSEEK_HARNESS, logger)
    prompt = builder(task_config_dir, str(root), eval_config, logger)
    if config["budget_mode"] == "time":
        prompt = prompt.rstrip() + time_budget_guidance()
    else:
        prompt += f"\n\nFor this optimization, iterate up to {config['max_iterations']} versions."
    prompt += (
        f"\nUse this Python interpreter: `{env['AGENT_KERNEL_ARENA_PYTHON']}`. "
        "Run pytest through that interpreter with `-m pytest`.\n"
    )
    prompt = append_run_budget(prompt, config["timeout_seconds"])
    # Fresh for each task launch/retry; only this launch's continuations share it.
    state = Path(tempfile.mkdtemp(prefix=".deepseek_harness-", dir=root))
    env["DSH_HOME"] = str(state / "home")
    patch_path = state / "cordis.patch.yml"
    patch_path.write_text(yaml.safe_dump(_patch(config), sort_keys=False), encoding="utf-8")
    (state / "invocation.json").write_text(json.dumps({
        "cli_version": version, "agent_config": config,
        "endpoint_override": bool(config.get("base_url") or env.get("DEEPSEEK_BASE_URL")),
    }, indent=2) + "\n", encoding="utf-8")
    logger.info("DeepSeek Harness version=%s model=%s state=%s", version, config["model"], state)
    outputs: list[str] = []

    def invoke(prompt_text: str, resume_id: str | None, timeout: float) -> SessionOutcome:
        # Preserve the original iterations-mode artifact paths. Each time-mode
        # invocation gets its own prompt/logs and uses the same isolated home.
        invocation = state
        if config["budget_mode"] == "time":
            invocation = state / f"session-{len(outputs) + 1:04d}"
            invocation.mkdir()
            (invocation / "invocation.json").write_text(json.dumps({
                "resume_session_id": resume_id, "timeout_seconds": timeout,
            }, indent=2) + "\n", encoding="utf-8")
        (invocation / "prompt.txt").write_text(prompt_text, encoding="utf-8")
        command = [binary, "--profile", "headless", "--patch", str(patch_path), "--json"]
        if resume_id is not None:
            command.extend(["--session-id", resume_id])
        output, outcome = _run(command, root, invocation, env, timeout, logger, model=config["model"])
        outputs.append(output)
        if resume_id is not None and outcome.session_id not in (None, resume_id):
            raise RuntimeError(f"DeepSeek Harness resumed a different session; see {invocation}")
        return outcome

    runs = run_sessions(
        invoke, mode=config["budget_mode"], prompt=prompt,
        timeout_seconds=config["timeout_seconds"], logger=logger,
    )
    raise_for_sessions(runs, mode=config["budget_mode"], agent_label="DeepSeek Harness",
                       timeout_seconds=config["timeout_seconds"])
    return "\n".join(outputs)
