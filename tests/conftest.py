"""Shared pytest fixtures."""
import pytest


@pytest.fixture(autouse=True)
def default_agent_sandbox_mode(monkeypatch):
    """Run fixture tasks with the agent sandbox off unless a test selects a mode.

    The default ``auto`` mode confines agents wherever Landlock is available, and
    fixture launchers run in-process, so they could never attest the policy.
    tests/test_agent_sandbox.py overrides this fixture to test the real default.
    """
    monkeypatch.setattr("src.agent_sandbox.DEFAULT_MODE", "none")
