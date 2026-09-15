"""PORT phase accounting fixtures; no GPU or numerical qualification claims."""
import asyncio
from dataclasses import dataclass
import json
from unittest.mock import patch

import pytest

from agents.forge.port_budget import bound_port


@dataclass
class PortResult:
    ok: bool
    attempts: int
    error_tail: str = ""


@pytest.mark.parametrize("upstream_stop,previous,expected", [(2000, None, 1400),
                         (1300, None, 1300), (2000, 1250, 1250)])
def test_port_reserves_search_and_restores_phase(tmp_path, upstream_stop, previous, expected):
    plan = dict(deadline_unix=2000, agent_config={"initialization_budget_fraction": .4},
                result=str(tmp_path / "result.json"))
    if previous is not None:
        plan["phase_deadline_unix"] = previous
    baseline = dict(plan)
    received = []
    async def native(spec, driver_path, config, *, stop_at_unix=None):
        received.append((spec, driver_path, config))
        assert stop_at_unix == expected == plan["phase_deadline_unix"]
        return PortResult(ok=True, attempts=2)
    with patch("agents.forge.port_budget.time.time", return_value=1000):
        result = asyncio.run(bound_port(native, plan)("declared", "public bridge", "config",
                             stop_at_unix=upstream_stop))
    assert result.ok and received == [("declared", "public bridge", "config")]
    assert plan == baseline
    evidence = json.loads((tmp_path / "port_budget.json").read_text())
    assert evidence["phase_deadline_unix"] == expected
    assert evidence["campaign_deadline_unix"] == 2000
    assert evidence["status"] == "PASS" and evidence["attempts"] == 2


@pytest.mark.parametrize("mode", ["phase_timeout", "compile_failure", "exception"])
def test_failed_port_never_becomes_success_and_keeps_failure_evidence(tmp_path, mode):
    plan = dict(deadline_unix=2000, agent_config={"initialization_budget_fraction": .4},
                result=str(tmp_path / "result.json"))
    now = [1000]
    async def native(spec, driver_path, config, *, stop_at_unix=None):
        if mode == "exception":
            raise RuntimeError("actual runtime failure")
        if mode == "phase_timeout":
            now[0] = stop_at_unix
        return PortResult(ok=False, attempts=2, error_tail="actual compiler diagnostic")
    with patch("agents.forge.port_budget.time.time", side_effect=lambda: now[0]):
        if mode == "exception":
            with pytest.raises(RuntimeError, match="actual runtime failure"):
                asyncio.run(bound_port(native, plan)(None, None, None))
        else:
            result = asyncio.run(bound_port(native, plan)(None, None, None))
            assert not result.ok
            assert ("reserved" in result.error_tail) == (mode == "phase_timeout")
            assert "actual compiler diagnostic" in result.error_tail
    evidence = json.loads((tmp_path / "port_budget.json").read_text())
    assert evidence["status"] == "FAILED"
    assert "phase_deadline_unix" not in plan
    if mode != "phase_timeout":
        assert "actual" in evidence["error"]
