"""Legacy names select canonical integrations, with identical configuration."""
import importlib
import logging

import pytest

from src.module_registration import AgentType, load_agent_launcher, load_post_processing_handler


@pytest.mark.parametrize("alias,canonical", [("geak_v4", "geak"), ("forge_operator2flydsl", "forge")])
def test_legacy_names_share_launcher_configuration_and_postprocessing(alias, canonical):
    selected = AgentType.from_string(alias)
    assert selected is AgentType.from_string(canonical)
    assert AgentType.from_string(alias.upper().replace("_", "-")) is selected
    logger = logging.getLogger(__name__)
    assert load_agent_launcher(selected, logger) is importlib.import_module(f"agents.{canonical}.launch_agent").launch_agent
    assert load_post_processing_handler(selected, logger) is load_post_processing_handler(AgentType.from_string(canonical), logger)
    if canonical == "geak":
        from agents.geak.launch_agent import load_options as options
    else:
        from agents.forge.adapter import _config as options
    overrides = {"model": "test-model", "timeout_seconds": 1200}
    assert options({"agent": {"template": alias, **overrides}}) == options({"agent": {"template": canonical, **overrides}})
