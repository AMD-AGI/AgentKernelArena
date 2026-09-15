"""Public action deadlines through the pinned native loop and in-session gate."""
import json
import os
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

import pytest

from agents.forge.action_budget import driver_limits
from agents.forge import adapter
from src.task_spec import TaskSpec
from test_forge_v2 import ROOT, fixture_task


def task_spec():
    return TaskSpec.from_mapping({
        "schema_version": 2,
        "candidate": {"language": "hip", "editable": ["impl.hip"]},
        "evaluation": {"runner": ["python3", "evaluate.py"],
            "candidate": {"compile": {"timeout_s": 11},
                          "correctness": {"timeout_s": 400},
                          "performance": {"timeout_s": 700}},
            "baseline": {"compile": {"timeout_s": 20},
                         "performance": {"timeout_s": 1400}}},
    }, task_id="arbitrary/operator")


def test_complete_driver_sequence_and_distinct_baseline_fit_declared_ceilings():
    spec = task_spec()
    with patch("agents.forge.action_budget.time.time", return_value=1000):
        assert driver_limits(spec, {"deadline_unix": 5000}) == {
            "build_timeout_sec": 11, "validate_stage_timeout_sec": 411,
            "bench_timeout_sec": 1420}
    # Public per-action limits are not rewritten by the native outer envelope.
    assert spec.action("candidate", "correctness").timeout_s == 400
    assert spec.action("candidate", "performance").timeout_s == 700


@pytest.mark.parametrize("phase", [None, 1005])
def test_driver_ceilings_never_extend_campaign_or_initialization(phase):
    plan = {"deadline_unix": 1100}
    if phase is not None:
        plan["phase_deadline_unix"] = phase
    with patch("agents.forge.action_budget.time.time", return_value=1000):
        limits = driver_limits(task_spec(), plan)
        assert max(limits.values()) == (5 if phase else 100)
    with patch("agents.forge.action_budget.time.time", return_value=1100):
        with pytest.raises(TimeoutError, match="no remaining"):
            driver_limits(task_spec(), plan)


def test_pinned_native_loop_and_agent_gate_receive_public_action_envelope(tmp_path):
    python = os.environ.get("AKA_FORGE_PROBE_PYTHON")
    if not python:
        pytest.skip("Set AKA_FORGE_PROBE_PYTHON to the pinned Hyperloom[forge] interpreter")
    context, plan, plan_path = fixture_task(tmp_path, language="hip")
    document = json.loads(context.path.read_text())
    document["task_config"]["evaluation"] = task_spec().to_mapping()["evaluation"]
    context.path.write_text(json.dumps(document))
    plan["deadline_unix"] = time.time() + 9000
    plan["agent_config"] = {"session_timeout_seconds": 1800}
    plan_path.write_text(json.dumps(plan))
    adapter._initialize_git(Path(plan["engine_root"]))
    script = r'''
import inspect, json, os
from functools import wraps
from pathlib import Path
from unittest.mock import patch
from agents.forge import upstream
from kernelforge.config import Config
from kernelforge.orchestrator import agent
from kernelforge.loop import runner
plan = json.loads(Path(os.environ['ARENA_FORGE_PLAN']).read_text())
native_factory = agent.make_agent_fn
signature = inspect.signature(native_factory)
received = []
@wraps(native_factory)
def capture(*args, **kwargs):
    received.append(signature.bind_partial(*args, **kwargs).arguments)
    async def implementer(kernel, history, session_sink=None):
        raise AssertionError('No provider or GPU call in this compatibility test')
    return implementer
agent.make_agent_fn = capture
original_loop = runner.IterationLoop
upstream.install_hooks(plan)
agent.make_agent_fn(config=Config(workspace=plan['engine_root']), program_md='CPU fixture',
                   kernel_backend_name='hip', bench_timeout_sec=300)
assert received[0]['validation_timeout_sec'] == 411
assert received[0]['bench_timeout_sec'] == 1420
assert received[0]['session_timeout_sec'] == 1800
configs = []
def collect(self, config, *args, **kwargs):
    configs.append(config)
with patch.object(original_loop, '__init__', collect):
    original = runner.IterationConfig(kernel_file='impl.hip', driver_script='driver.py',
                    max_time_hours=3, deadline_unix=plan['deadline_unix'])
    runner.IterationLoop(original, None)
assert original.bench_timeout_sec == 300
assert configs[0].bench_timeout_sec == 1420
assert configs[0].validate_stage_timeout_sec == 411
assert configs[0].build_timeout_sec == 11
assert configs[0].deadline_unix == original.deadline_unix
'''
    run = subprocess.run([python, "-c", script], cwd=ROOT,
                         env=dict(os.environ, PYTHONPATH=str(ROOT), ARENA_FORGE_PLAN=str(plan_path)),
                         text=True, capture_output=True, timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr
