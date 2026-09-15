"""Shared-clock regressions; native compatibility tests use the pinned engine."""
from dataclasses import dataclass
import asyncio
import inspect
import json
import os
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

import pytest

from agents.forge.deadline import SessionBudgetExceeded, bound_agent, bound_session


@dataclass
class Session:
    timeout_sec: int | None = 1200
    user_prompt: str = "## Session deadline\nYou have 20 minutes.\n## Prior experience\nkeep this"


def test_session_cannot_consume_checkpoint_reserve():
    original = Session()
    with patch("agents.forge.deadline.time.time", return_value=1000):
        bounded = bound_session(original, {"deadline_unix": 1300})
    assert bounded.timeout_sec == 180
    assert "180 seconds" in bounded.user_prompt
    assert "20 minutes" not in bounded.user_prompt
    assert "## Prior experience\nkeep this" in bounded.user_prompt
    assert original.timeout_sec == 1200


def test_shorter_provider_limit_and_initialization_phase_are_preserved():
    with patch("agents.forge.deadline.time.time", return_value=1000):
        original = Session(timeout_sec=30)
        assert bound_session(original, {"deadline_unix": 1300}) is original
        result = bound_session(Session(), {"deadline_unix": 5000, "phase_deadline_unix": 1040})
        assert result.timeout_sec == 40
        with pytest.raises(TimeoutError, match="finalization reserve"):
            bound_session(Session(), {"deadline_unix": 1119})
        with pytest.raises(TimeoutError):
            bound_session(Session(), {"deadline_unix": 900, "phase_deadline_unix": 1100})


def test_total_budget_cancels_repeated_short_turns_and_preserves_callable_metadata():
    events, sink = [], {}
    async def native(kernel, history, session_sink=None):
        try:
            for _ in range(100):
                events.append("resume")
                await asyncio.sleep(.025)
            return "should not complete"
        finally:
            events.append("unwound")
    native.backend_name = "codex"
    native.backend_model = "test-only"
    wrapped = bound_agent(native, {"deadline_unix": 5000}, .1)
    assert inspect.signature(wrapped) == inspect.signature(native)
    assert wrapped.backend_name == "codex" and wrapped.backend_model == "test-only"
    with patch("agents.forge.deadline.time.time", return_value=1000):
        with pytest.raises(SessionBudgetExceeded, match="total session budget"):
            asyncio.run(wrapped("kernel", "history", session_sink=sink))
    assert 1 < events.count("resume") < 100
    assert events[-1] == "unwound"
    assert sink["end_reason"] == "session_timeout" and sink["gate_passed"] is False


def test_agent_time_is_recomputed_for_each_invocation_and_does_not_relabel_inner_errors():
    calls = []
    async def native(kernel, history, session_sink=None):
        calls.append(kernel)
        if kernel == "fail":
            raise TimeoutError("task compile timeout evidence")
        return "finished"
    wrapped = bound_agent(native, {"deadline_unix": 1130}, 600)
    with patch("agents.forge.deadline.time.time", return_value=1000):
        assert asyncio.run(wrapped("valid", "")) == "finished"
        sink = {}
        with pytest.raises(TimeoutError, match="task compile timeout evidence"):
            asyncio.run(wrapped("fail", "", sink))
        assert "end_reason" not in sink
    with patch("agents.forge.deadline.time.time", return_value=1011):
        with pytest.raises(TimeoutError, match="finalization reserve"):
            asyncio.run(wrapped("never starts", ""))
    assert calls == ["valid", "fail"]


def test_pinned_native_loop_admission_uses_absolute_clock(tmp_path):
    python = os.environ.get("AKA_FORGE_PROBE_PYTHON")
    if not python:
        pytest.skip("Set AKA_FORGE_PROBE_PYTHON to the pinned Hyperloom[forge] interpreter")
    root = Path(__file__).resolve().parents[1]
    script = r'''
from types import SimpleNamespace
from unittest.mock import patch
from agents.forge import upstream, deadline
from kernelforge.loop import runner
upstream.probe()
original = runner.IterationLoop
deadline.install()
bounded = runner.IterationLoop
deadline.install()
assert runner.IterationLoop is bounded
loop = object.__new__(bounded)
loop.start_time = 1000
loop.ic = SimpleNamespace(max_time_hours=1, deadline_unix=1300, budget_reserve_sec=60)
with patch('time.time', return_value=1250):
    # Original would claim 55+ minutes and admit another costly search.
    assert original._time_remaining(loop) == 3350
    assert loop._time_remaining() == 50
    assert loop._is_budget_exhausted()
with patch('time.time', return_value=1400):
    assert loop._time_remaining() == 0
    assert loop._is_budget_exhausted()
loop.ic.deadline_unix = 10000
with patch('time.time', return_value=4550):
    assert loop._time_remaining() == 50  # wall budget still wins if shorter
loop.ic.deadline_unix = None
with patch('time.time', return_value=1250):
    assert loop._time_remaining() == original._time_remaining(loop)

received = []
def capture(self, ic, tracker, config, resume):
    received.append(ic)
with patch.object(original, '__init__', capture), patch('time.time', return_value=1000):
    ic = runner.IterationConfig(kernel_file='kernel.py', driver_script='driver.py',
                                max_time_hours=1, deadline_unix=2200)
    bounded(ic, None)
    assert received[-1].budget_reserve_sec == 120
    assert ic.budget_reserve_sec == 1800
    assert received[-1].validate_stage_timeout_sec == ic.validate_stage_timeout_sec
    assert received[-1].bench_timeout_sec == ic.bench_timeout_sec
'''
    result = subprocess.run([python, "-c", script], cwd=root,
                            env=dict(os.environ, PYTHONPATH=str(root)),
                            text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("mode", ["resumes", "gate_process", "tamper_on_resume", "safety_rejection"])
def test_pinned_implementer_timeout_unwinds_and_finalizes_integrity(tmp_path, mode):
    python = os.environ.get("AKA_FORGE_PROBE_PYTHON")
    if not python:
        pytest.skip("Set AKA_FORGE_PROBE_PYTHON to the pinned Hyperloom[forge] interpreter")
    from test_forge_v2 import fixture_task, ROOT
    from agents.forge import adapter, bridge

    context, plan, path = fixture_task(tmp_path, language="triton")
    plan.update(deadline_unix=time.time()+600,
                agent_config={"session_timeout_seconds": 1})
    path.write_text(json.dumps(plan))
    engine = Path(plan["engine_root"])
    (engine / "arena_forge_driver.py").write_text(bridge.render_driver(path, ROOT))
    adapter._initialize_git(engine)
    # Only provider replies and numerical results below are scripted. The real
    # pinned implementer, outer resume loop, integrity gate and subprocess
    # cancellation run on CPU. This is not GPU qualification.
    script = r'''
import asyncio, json, os, sys
from pathlib import Path
from agents.forge import upstream
from agents.forge.deadline import SessionBudgetExceeded
from kernelforge.config import Config
from kernelforge.orchestrator import agent
from kernelforge.loop import insession_gate
from kernelforge.agent_backends.base import AgentCapabilities, AgentRunResult
from kernelforge.mcp_server.tools._subprocess import communicate_process_group

plan = json.loads(Path(os.environ['ARENA_FORGE_PLAN']).read_text())
root = Path(plan['engine_root'])
mode = os.environ['CPU_TIMEOUT_MODE']
resumes, children, unwound = [], [], []
class Rejection(RuntimeError):
    agent_safety_rejection = True
class Provider:
    name = 'codex'
    capabilities = AgentCapabilities(stop_hooks=False, resumable=True, workspace_guard=True)
    def __init__(self, runtime): self.runtime = runtime
    async def run(self, spec, usage=None):
        if mode == 'safety_rejection': raise Rejection('test backend safety rejection')
        return AgentRunResult(text='PLAN: test only', subtype='success', session_id='cpu-only',
                              file_changes=[], num_turns=1)
    async def resume(self, spec, session_id, feedback, usage=None):
        resumes.append(spec.timeout_sec)
        if mode == 'tamper_on_resume':
            (root/'runner.py').write_text('protected task runner changed by CPU fixture')
        try:
            await asyncio.sleep(20 if mode == 'tamper_on_resume' else .1)
            return await self.run(spec, usage)
        finally:
            unwound.append(True)

async def correctness(**kwargs):
    if mode == 'gate_process':
        proc = await asyncio.create_subprocess_exec(sys.executable, '-c',
            'import time; time.sleep(60)', start_new_session=True,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        children.append(proc)
        await communicate_process_group(proc, timeout=60)
    return {'passed': False, 'message': 'CPU fixture rejection; resume required'}

agent.create_registered_backend = lambda runtime, **kwargs: Provider(runtime)
insession_gate.test_correctness = correctness
upstream.install_hooks(plan)
config = Config(workspace=str(root), agent_backend='codex', agent_model='cpu-fixture',
                agent_precheck=False)
fn = agent.make_agent_fn(config, 'CPU fixture', kernel_backend_name='triton',
                         insession_gate=True, driver_script=str(root/'arena_forge_driver.py'),
                         session_timeout_sec=600, max_blocks=100)
assert fn.backend_name == 'codex' and fn.backend_model == 'cpu-fixture'
sink = {}
original = (root/'runner.py').read_bytes()
try:
    asyncio.run(fn(str(root/'source/kernel.py'), '', session_sink=sink))
except Rejection:
    assert mode == 'safety_rejection'
    assert sink['integrity_violation'] is True
    assert 'backend workspace safety rejection' in sink['integrity_reason']
except SessionBudgetExceeded as error:
    assert 'total session budget' in str(error), str(error)
    assert sink['end_reason'] == 'session_timeout' and sink['gate_passed'] is False
    if mode == 'resumes':
        assert 2 <= len(resumes) < 100 and unwound
    elif mode == 'gate_process':
        assert len(children) == 1 and children[0].returncode is not None
    elif mode == 'tamper_on_resume':
        assert resumes and unwound
        assert sink['integrity_violation'] is True
        sink['integrity_restore']()
        assert (root/'runner.py').read_bytes() == original
    else: raise
else:
    raise AssertionError('unbounded repeated session completed unexpectedly')
assert (root/'source/kernel.py').read_text() == '2'
assert (Path(plan['template'])/'runner.py').read_bytes() == original
if mode == 'resumes':
    # The real PORT loop treats asyncio.TimeoutError as the entire phase ending.
    # A shorter per-attempt budget must leave its remaining attempts available.
    from types import SimpleNamespace
    from agents.forge.deadline import bound_agent
    from kernelforge.rewrite_by_flydsl import port_loop
    import time
    calls = []
    async def expires(*args, **kwargs):
        calls.append(True)
        await asyncio.sleep(20)
    agent.make_agent_fn = lambda **kwargs: bound_agent(expires, plan, .05)
    port_loop.build_port_program_md = lambda *args: 'CPU fixture'
    spec = SimpleNamespace(snr_threshold=30, flydsl_kernel=str(root/'source/kernel.py'),
        builder_symbol='declared', source_kernel=str(root/'source/helper.py'),
        source_kernel_name='helper.py', op_name='cpu-fixture')
    result = asyncio.run(port_loop.run_port_loop(spec, 'unused', config,
                         max_attempts=2, stop_at_unix=time.time()+10))
    assert not result.ok and result.attempts == 2 and len(calls) == 2
    calls.clear()
    result = asyncio.run(port_loop.run_port_loop(spec, 'unused', config,
                         max_attempts=2, stop_at_unix=time.time()+.025))
    assert not result.ok and result.attempts == 1 and len(calls) == 1
    assert 'finalization reserve' in result.error_tail
print('PINNED_SESSION_TIMEOUT_CPU_PASS')
'''
    run = subprocess.run([python, "-c", script], cwd=engine,
                         env=dict(os.environ, PYTHONPATH=str(ROOT), ARENA_FORGE_PLAN=str(path),
                                  CPU_TIMEOUT_MODE=mode), text=True, capture_output=True, timeout=30)
    assert run.returncode == 0, run.stdout[-4000:] + run.stderr[-5000:]
    assert "PINNED_SESSION_TIMEOUT_CPU_PASS" in run.stdout
