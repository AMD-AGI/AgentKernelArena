"""Public engine selection must reach the registered GEAK launcher."""
import logging

from src.module_registration import AgentType, load_agent_launcher, load_post_processing_handler


def test_public_geak_loads_the_real_launcher_and_common_postprocessing():
    from agents.geak.launch_agent import launch_agent
    from src.postprocessing import general_post_processing

    assert AgentType.from_string("geak") is AgentType.GEAK
    assert load_agent_launcher(AgentType.GEAK, logging.getLogger(__name__)) is launch_agent
    assert load_post_processing_handler(AgentType.GEAK, logging.getLogger(__name__)) is general_post_processing
