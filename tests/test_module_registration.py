"""Agent selection and loading checks; no agent processes or GPUs are started."""

import importlib
import logging

import pytest

from src.module_registration import (
    AgentType,
    load_agent_launcher,
    load_post_processing_handler,
)


@pytest.mark.parametrize("name", ["geak_v3", "geak_v3_triton", "mini_swe_triton", "task_validator"])
@pytest.mark.parametrize("spelling", ["original", "hyphenated", "uppercase"])
def test_retired_agent_templates_are_rejected(name, spelling):
    selected = name
    if spelling == "hyphenated":
        selected = name.replace("_", "-")
    elif spelling == "uppercase":
        selected = name.upper()

    with pytest.raises(ValueError, match="Invalid agent type") as error:
        AgentType.from_string(selected)
    assert name not in str(error.value).split("Valid options are:")[1]


@pytest.mark.parametrize("name", [
    "cursor", "claude_code", "codex", "deepseek_harness", "qa_auditor", "geak", "geak_v4", "forge",
    "forge_operator2flydsl",
])
def test_remaining_agents_load_with_their_postprocessors(name):
    logger = logging.getLogger(__name__)
    agent = AgentType.from_string(name)
    assert AgentType.from_string(name.upper().replace("_", "-")) is agent
    launcher = load_agent_launcher(agent, logger)
    assert launcher is importlib.import_module(f"agents.{agent.value}.launch_agent").launch_agent

    if name == "qa_auditor":
        from agents.qa_auditor.postprocessing import audit_post_processing

        expected = audit_post_processing
    else:
        from src.postprocessing import general_post_processing

        expected = general_post_processing
    assert load_post_processing_handler(agent, logger) is expected
