# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
import logging
from enum import Enum
from typing import Callable
from agents import AGENT_REGISTRY


class AgentType(Enum):
    """Enumeration of supported agent types."""
    CURSOR = "cursor"
    CLAUDE_CODE = "claude_code"
    SWE_AGENT = "swe_agent"
    SINGLE_LLM_CALL = "single_llm_call"
    OPENEVOLVE = "openevolve"
    GEAK_OPTIMAGENTV2 = "geak_optimagentv2"
    GEAK_HIP = "geak_hip"
    OURLLM_KERNEL2KERNEL = "geak_ourllm_kernel2kernel"
    TASK_VALIDATOR = "task_validator"

    @classmethod
    def from_string(cls, agent_string: str) -> 'AgentType':
        """
        Convert string to AgentType enum.

        Args:
            agent_string: String representation of agent name

        Returns:
            AgentType enum

        Raises:
            ValueError: If agent_string is not a valid agent type
        """
        # Normalize the string (lowercase, replace underscores with hyphens for matching)
        normalized = agent_string.lower().replace('_', '-')

        for agent_type in cls:
            if agent_type.value.replace('_', '-') == normalized:
                return agent_type

        # If no match found, raise error with available options
        valid_options = [agent.value for agent in cls]
        raise ValueError(f"Invalid agent type: '{agent_string}'. Valid options are: {valid_options}")



def load_agent_launcher(agent_type: AgentType, logger: logging.Logger) -> Callable[..., str]:
    """
    Dynamically load agent launcher function from agent registry.

    Args:
        agent_type: AgentType enum
        logger: Logger instance

    Returns:
        Agent launcher function
    """
    agent_name = agent_type.value

    # Import all agent modules to trigger registration
    try:
        if agent_type == AgentType.CURSOR:
            from agents.cursor import launch_agent  # noqa: F401
        elif agent_type == AgentType.CLAUDE_CODE:
            from agents.claude_code import launch_agent  # noqa: F401
        elif agent_type == AgentType.SINGLE_LLM_CALL:
            from agents.single_llm_call import launch_agent  # noqa: F401
        elif agent_type == AgentType.OPENEVOLVE:
            from agents.openevolve import launch_agent  # noqa: F401
        elif agent_type == AgentType.GEAK_OPTIMAGENTV2:
            from agents.geak_optimagentv2 import launch_agent  # noqa: F401
        elif agent_type == AgentType.SWE_AGENT:
            from agents.SWE_agent import launch_agent  # noqa: F401
        elif agent_type == AgentType.GEAK_HIP:
            from agents.geak_hip import launch_agent  # noqa: F401
        elif agent_type == AgentType.OURLLM_KERNEL2KERNEL:
            from agents.geak_ourllm_kernel2kernel import launch_agent  # noqa: F401
        elif agent_type == AgentType.TASK_VALIDATOR:
            from agents.task_validator import launch_agent  # noqa: F401
    except ImportError as e:
        logger.error(f"Failed to import agent {agent_name}: {e}")
        raise

    # Get agent from registry
    if agent_name not in AGENT_REGISTRY:
        raise ValueError(f"Agent '{agent_name}' not found in registry. Available agents: {list(AGENT_REGISTRY.keys())}")

    logger.info(f"Loaded agent: {agent_name}")
    return AGENT_REGISTRY[agent_name]


def load_post_processing_handler(agent_type: AgentType, logger: logging.Logger) -> Callable[[list, logging.Logger], None]:
    """
    Dynamically load post-processing function based on agent type.

    Args:
        agent_type: AgentType enum
        logger: Logger instance

    Returns:
        Post-processing function for the agent

    Raises:
        NotImplementedError: If agent doesn't have post-processing support
    """
    from src.postprocessing import general_post_processing

    agent_name = agent_type.value

    # Map agents to their post-processing functions
    if agent_type == AgentType.TASK_VALIDATOR:
        from agents.task_validator.validation_postprocessing import validation_post_processing
        logger.info(f"Using validation_post_processing for agent: {agent_name}")
        return validation_post_processing
    elif agent_type in [AgentType.CURSOR, AgentType.CLAUDE_CODE, AgentType.SWE_AGENT, AgentType.GEAK_OPTIMAGENTV2, AgentType.GEAK_HIP, AgentType.OPENEVOLVE, AgentType.SINGLE_LLM_CALL, AgentType.OURLLM_KERNEL2KERNEL]:
        logger.info(f"Using general_post_processing for agent: {agent_name}")
        return general_post_processing
    else:
        raise NotImplementedError(f"Post-processing not implemented for agent: {agent_name}")


def load_prompt_builder(agent_type: AgentType, logger: logging.Logger) -> Callable:
    """
    Dynamically load prompt builder function based on agent type.

    Args:
        agent_type: AgentType enum
        logger: Logger instance

    Returns:
        Prompt builder function

    Raises:
        NotImplementedError: If agent doesn't have prompt builder support
    """
    from src.prompt_builder import prompt_builder

    agent_name = agent_type.value

    # Map agents to their prompt builder functions
    if agent_type in [AgentType.CURSOR, AgentType.CLAUDE_CODE, AgentType.SWE_AGENT, AgentType.OURLLM_KERNEL2KERNEL]:
        logger.info(f"Using standard prompt_builder for agent: {agent_name}")
        return prompt_builder
    else:
        raise NotImplementedError(f"Prompt builder not implemented for agent: {agent_name}")
