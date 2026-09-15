"""The deprecated template is an alias, not a separate task integration."""
import importlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_deprecated_template_forwards_the_exact_standard_arguments(monkeypatch):
    module = importlib.import_module("agents.forge_operator2flydsl.launch_agent")
    calls = []
    monkeypatch.setattr(module, "_launch", lambda *args: calls.append(args) or "output")
    config = {"agent": {"workflow": "auto"}}
    assert module.launch_agent(config, "task/config.yaml", "workspace") == "output"
    assert calls == [(config, "task/config.yaml", "workspace")]


def test_alias_does_not_invoke_provider_specific_exports():
    source = (ROOT / "agents/forge_operator2flydsl/postprocessing.py").read_text()
    assert "backfill_solutions" not in source
    assert "general_post_processing(workspace_paths, logger)" in source
