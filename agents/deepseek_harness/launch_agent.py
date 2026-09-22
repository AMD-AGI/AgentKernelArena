# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Run the pinned DeepSeek Harness headless CLI in an Arena task workspace."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from typing import Any

import yaml

from agents import register_agent
from src.module_registration import AgentType, load_prompt_builder
from src.runtime_env import build_subprocess_env


def _load_config() -> dict[str, Any]:
    config = yaml.safe_load(Path(__file__).with_name("agent_config.yaml").read_text())
    if not isinstance(config, dict):
        raise ValueError("DeepSeek Harness agent_config.yaml must be a mapping")
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


def _stop_process_group(process: subprocess.Popen[str]) -> None:
    """Stop tool children too, including after the headless parent has exited."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        process.poll()  # Reap the leader before testing whether the group survived.
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait(timeout=5)


def _run(
    command: list[str], workspace: Path, state: Path,
    env: dict[str, str], timeout: int, logger: logging.Logger,
) -> str:
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
    if timed_out:
        raise TimeoutError(f"DeepSeek Harness exceeded {timeout}s; see {state}")
    if process.returncode != 0:
        raise RuntimeError(
            f"DeepSeek Harness exited with code {process.returncode}; see {state}"
        )
    return "".join(outputs["stdout"]) + "".join(outputs["stderr"])


@register_agent("deepseek_harness")
def launch_agent(eval_config: dict, task_config_dir: str, workspace: str) -> str:
    config = _load_config()
    env = _environment(config)
    binary, version = _preflight(config, env)
    logger = logging.getLogger(__name__)
    root = Path(workspace).resolve(strict=True)
    builder = load_prompt_builder(AgentType.DEEPSEEK_HARNESS, logger)
    prompt = builder(task_config_dir, str(root), eval_config, logger)
    prompt += (
        f"\n\nFor this optimization, iterate up to {config['max_iterations']} versions."
        f"\nUse this Python interpreter: `{env['AGENT_KERNEL_ARENA_PYTHON']}`. "
        "Run pytest through that interpreter with `-m pytest`.\n"
    )
    # Unique on every invocation, including retries. Never reuse host sessions/settings.
    state = Path(tempfile.mkdtemp(prefix=".deepseek_harness-", dir=root))
    env["DSH_HOME"] = str(state / "home")
    (state / "prompt.txt").write_text(prompt, encoding="utf-8")
    patch_path = state / "cordis.patch.yml"
    patch_path.write_text(yaml.safe_dump(_patch(config), sort_keys=False), encoding="utf-8")
    (state / "invocation.json").write_text(json.dumps({
        "cli_version": version, "agent_config": config,
        "endpoint_override": bool(config.get("base_url") or env.get("DEEPSEEK_BASE_URL")),
    }, indent=2) + "\n", encoding="utf-8")
    logger.info("DeepSeek Harness version=%s model=%s state=%s", version, config["model"], state)
    return _run(
        [binary, "--profile", "headless", "--patch", str(patch_path), "--json"],
        root, state, env, config["timeout_seconds"], logger,
    )
