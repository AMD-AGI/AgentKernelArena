"""Explicit native Codex login support for the pinned Forge SDK backend."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil


def install_cli_auth() -> None:
    """Keep Forge's backend/SDK; let the Codex CLI consume its own OAuth file.

    Called only after the upstream source pin is verified and the run explicitly
    selects codex_auth_mode=cli. Never reinterpret an OAuth token as an API key.
    """
    from kernelforge.agent_backends import codex

    source = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))) / "auth.json"
    try:
        auth = json.loads(source.read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError("Forge CLI authentication requires a readable Codex auth.json") from exc
    if auth.get("auth_mode") != "chatgpt" or not isinstance(auth.get("tokens"), dict):
        raise RuntimeError("Forge codex_auth_mode=cli requires a native ChatGPT login")
    if any(os.environ.get(name) for name in ("OPENAI_BASE_URL", "OPENAI_API_KEY")):
        raise RuntimeError("Forge CLI authentication cannot be mixed with an API gateway/key")

    def native_provider(gateway):
        if gateway.has_endpoint or gateway.has_key:
            raise RuntimeError("Forge CLI authentication cannot override a configured gateway")
        return ['model_provider="openai"']

    original_environment = codex.CodexBackend._child_environment
    original_thread_options = codex.CodexBackend._thread_start_options
    def thread_options(self, sdk, spec):
        options = original_thread_options(self, sdk, spec)
        options["model_provider"] = "openai"
        return options

    def child_environment(self, base=None):
        env = original_environment(self, base)
        destination = Path(env["CODEX_HOME"]) / "auth.json"
        if destination.resolve() == source.resolve():
            raise RuntimeError("Forge SDK authentication HOME must be isolated from the login source")
        if not destination.exists():
            # Existing files belong to this SDK session and can contain refreshed
            # login state. Never overwrite them or copy mutable state back.
            with open(destination, "x", opener=lambda path, flags: os.open(path, flags, 0o600)) as target:
                with source.open() as origin:
                    shutil.copyfileobj(origin, target)
        return env

    codex._provider_overrides = native_provider
    codex.CodexBackend._child_environment = child_environment
    codex.CodexBackend._thread_start_options = thread_options
