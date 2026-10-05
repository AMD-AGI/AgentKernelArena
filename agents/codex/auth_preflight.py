"""Provider-aware Codex access checks without logging provider secrets."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile

import yaml

try:
    import tomllib
except ImportError:  # Python 3.10 images may provide the TOML backport.
    import tomli as tomllib


class CodexAuthError(RuntimeError):
    """A sanitized error suitable for the Docker preflight log."""


def _run_inference(command: list[str], directory: str, timeout: int) -> subprocess.CompletedProcess:
    # npm's Codex launcher spawns a native child. Kill the process group on a
    # timeout so the provider request cannot outlive the failed preflight.
    with subprocess.Popen(
        command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, cwd=directory, start_new_session=True,
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate()
            raise
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def _user_config() -> dict:
    home = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex")))
    path = home / "config.toml"
    try:
        return tomllib.loads(path.read_text()) if path.is_file() else {}
    except (OSError, ValueError):
        # TOML diagnostics can include the offending credential line.
        raise CodexAuthError("Unable to read Codex configuration; details withheld.") from None


def _model_settings(config_path: str | None, user_config: dict) -> tuple[str | None, str | None]:
    """Match the settings used by the selected framework agent launcher."""
    model = user_config.get("model")
    effort = user_config.get("model_reasoning_effort")
    if not config_path:
        return model, effort
    try:
        run = yaml.safe_load(Path(config_path).read_text()) or {}
        agent = run.get("agent", {})
        if not isinstance(agent, dict):
            return model, effort
        template = agent.get("template")
        if template not in {"codex", "task_validator"}:
            return model, effort
        defaults_path = Path(__file__).parents[1] / template / "agent_config.yaml"
        defaults = yaml.safe_load(defaults_path.read_text()) or {}
        if template == "task_validator":
            # Keep the validator's existing per-run precedence, including null
            # inheriting its agent defaults rather than the CLI configuration.
            model = agent.get("model") or defaults.get("model") or model
            effort = agent.get("effort") or defaults.get("effort") or effort
        else:
            model = defaults.get("model") or model
            effort = defaults.get("effort") or effort
    except (OSError, ValueError, AttributeError, yaml.YAMLError):
        raise CodexAuthError("Unable to resolve the configured Codex model.") from None
    return model, effort


def check_codex_access(config_path: str | None = None, *, timeout: int = 120) -> str:
    """Check OpenAI login or prove custom-provider access with real inference.

    Provider credential presence is never accepted as proof of access. Raw CLI
    stdout/stderr, config values, URLs, and HTTP diagnostics are not returned.
    """
    config = _user_config()
    provider = config.get("model_provider", "openai")
    if not isinstance(provider, str):
        raise CodexAuthError("Invalid Codex provider selection.")
    providers = config.get("model_providers", {})
    definition = providers.get(provider) if isinstance(providers, dict) else None
    if definition is not None and not isinstance(definition, dict):
        raise CodexAuthError("Invalid Codex provider configuration.")
    if definition is None and provider != "openai":
        raise CodexAuthError("Selected custom Codex provider is not defined.")
    requires_login = (definition or {}).get("requires_openai_auth", definition is None)
    if not isinstance(requires_login, bool):
        raise CodexAuthError("Codex requires_openai_auth must be boolean.")
    if requires_login:
        try:
            result = subprocess.run(
                ["codex", "login", "status"], capture_output=True, text=True,
                check=False, timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired):
            raise CodexAuthError("Codex OpenAI login check could not complete.") from None
        if result.returncode:
            raise CodexAuthError("Codex provider requires OpenAI authentication; login check failed.")
        return "codex_status=OpenAI_login_verified"

    model, effort = _model_settings(config_path, config)
    sentinel = "AKA_CODEX_PROVIDER_ACCESS_OK"
    with tempfile.TemporaryDirectory(prefix="aka-codex-access-") as directory:
        output = Path(directory) / "response.txt"
        command = [
            "codex", "exec", "--ephemeral", "--skip-git-repo-check",
            "--sandbox", "read-only", "--cd", directory,
            "--output-last-message", str(output),
            "-c", 'approval_policy="never"',
            "-c", "project_doc_max_bytes=0",
            "-c", "features.memories=false",
            "-c", "features.shell_tool=false",
            "-c", "features.multi_agent=false",
            "-c", 'web_search="disabled"',
        ]
        # Prevent unrelated MCP connections during the benign inference check.
        for name in config.get("mcp_servers", {}):
            command.extend(["-c", f"mcp_servers.{json.dumps(name)}.enabled=false"])
        if model:
            command.extend(["--model", str(model)])
        if effort:
            command.extend(["-c", f"model_reasoning_effort={json.dumps(str(effort))}"])
        command.append(
            "This is a provider connectivity check. Do not call tools or inspect files. "
            f"Reply with exactly {sentinel} and no other text."
        )
        try:
            result = _run_inference(command, directory, timeout)
        except subprocess.TimeoutExpired:
            raise CodexAuthError(f"Codex custom-provider inference timed out after {timeout}s.") from None
        except OSError:
            raise CodexAuthError("Codex custom-provider inference could not start.") from None
        if result.returncode:
            raise CodexAuthError(
                f"Codex custom-provider inference failed (exit {result.returncode}); "
                "provider output withheld to protect credentials."
            )
        try:
            response = output.read_text().strip()
        except OSError:
            response = ""
        if response != sentinel:
            raise CodexAuthError("Codex custom-provider inference did not return the expected response.")
    return "codex_status=custom_provider_inference_verified"
