"""Initialization control-flow tests; scripted backend/CPU tasks, no GPU claims."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import time

import pytest

from test_forge_v2 import ROOT, fixture_task, mock_engine
from agents.forge import adapter, bridge
from agents.forge.bundles import allow_candidate_paths
from agents.forge.initialization import materialize_targets
from src.task_spec import TaskSpec


@pytest.mark.parametrize("language", ["hip", "triton"])
def test_launcher_routes_initialization_through_its_own_engine(tmp_path, monkeypatch, language):
    context, _, _ = fixture_task(tmp_path, language=language, initial_state="unimplemented")
    monkeypatch.setenv("ARENA_TASK_CONTEXT", str(context.path))
    commands = mock_engine(monkeypatch)
    output = adapter.launch({}, "unused", str(context.workspace))
    assert len(commands) == 1
    assert commands[0][2:4] == ["--arena-initialize", "forge-loop"]
    assert commands[0][commands[0].index("--kernel-backend") + 1] == language
    assert '"workflow": "initialize"' in output
    assert (context.baseline_workspace / "source/helper.py").read_text() == "3"


@pytest.mark.parametrize("initial_status", ["PASS", "FAILED"])
def test_no_keep_delivery_requires_a_validated_initial_commit(tmp_path, monkeypatch, initial_status):
    context, _, _ = fixture_task(tmp_path, language="hip", initial_state="unimplemented")
    monkeypatch.setenv("ARENA_TASK_CONTEXT", str(context.path))
    mock_engine(monkeypatch)
    real_engine = adapter.run_forge_subprocess
    def engine(*args, **kwargs):
        output = real_engine(*args, **kwargs)
        plan = json.loads(Path(kwargs["env"]["ARENA_FORGE_PLAN"]).read_text())
        result_path = Path(plan["result"])
        result = json.loads(result_path.read_text())
        Path(plan["initialization_result"]).write_text(json.dumps(
            dict(status=initial_status, commit=result["best_commit"])))
        result.update(best_commit="", iteration_count=0)
        result_path.write_text(json.dumps(result))
        return output
    monkeypatch.setattr(adapter, "run_forge_subprocess", engine)
    if initial_status == "PASS":
        output = adapter.launch({}, "unused", str(context.workspace))
        assert '"delivery_selection": "initial_correct_implementation"' in output
        assert '"iteration_count": 0' in output
        assert (context.workspace / "source/helper.py").read_text() == "6"
    else:
        with pytest.raises(adapter.ForgeRunError, match="full selected commit"):
            adapter.launch({}, "unused", str(context.workspace))
        assert (context.workspace / "source/helper.py").read_text() == "3"


def test_missing_declared_files_materialized_without_inventing_entrypoint(tmp_path):
    context, plan, _ = fixture_task(tmp_path, language="hip", initial_state="unimplemented")
    root = Path(plan["engine_root"])
    (root / "source/kernel.py").unlink()
    materialize_targets(context.spec, root, "source/kernel.py")
    assert (root / "source/kernel.py").read_text() == ""
    assert (root / "source/helper.py").read_text() == "3"
    assert not (root / "kernel.py").exists()
    # Metadata declaring symbol scope cannot authorize recreating a harness.
    data = context.spec.to_mapping()
    data["candidate"]["editable"][0] = {"path": "source/kernel.py", "scope": "symbols", "symbols": ["declared"]}
    (root / "source/kernel.py").unlink()
    with pytest.raises(ValueError, match="symbol-scoped"):
        materialize_targets(TaskSpec.from_mapping(data, task_id="unrelated"), root, "source/kernel.py")


@pytest.mark.parametrize("value", [0, 1, -1, True, float("nan")])
def test_initialization_cannot_take_the_entire_loop_budget(value):
    with pytest.raises(ValueError, match="initialization_budget_fraction"):
        adapter._config({"agent": {"initialization_budget_fraction": value}})


# This program runs inside the real, pinned Hyperloom interpreter. Only the
# provider response and the final loop callback are scripted. The real Forge
# implementer factory, backend prompts, API session loop, workspace guard,
# in-session correctness gate, CLI parsing and public task actions all execute.
UPSTREAM_INITIALIZATION = r'''
import asyncio, json, os, subprocess
from pathlib import Path
from types import SimpleNamespace
from agents.forge import engine, bridge
from agents.forge.task_context import TaskContext
from agents.forge.bundles import committed_candidate, install_candidate
from kernelforge import cli
from kernelforge.orchestrator import agent
from kernelforge.agent_backends.base import AgentCapabilities, AgentRunResult
from kernelforge.agent_backends.workspace_guard import WorkspaceGuard
from kernelforge.loop.runner import IterationLoop

plan = json.loads(Path(os.environ['ARENA_FORGE_PLAN']).read_text())
root = Path(plan['engine_root'])
mode = os.environ['CPU_TEST_MODE']
context = TaskContext.load(plan['context'])
language = context.spec.candidate.language
loop_calls, sessions = [], []

class ScriptedProvider:
    name = 'codex'
    capabilities = AgentCapabilities(stop_hooks=False, resumable=False, workspace_guard=True)
    def __init__(self, runtime):
        self.runtime = runtime
    async def run(self, spec, usage=None):
        sessions.append(spec)
        assert spec.role == 'implementer'
        assert spec.cwd == str(root), spec.cwd
        assert 'INITIALIZE' in spec.system_prompt
        assert language.lower() in spec.system_prompt.lower()
        assert 'source/kernel.py' in spec.system_prompt
        assert spec.timeout_sec <= plan['agent_config']['session_timeout_seconds']
        if len(sessions) > 1:
            assert 'wrong multiplication' in spec.user_prompt or 'invalid literal' in spec.user_prompt
        guard = WorkspaceGuard(spec)
        guard.prepare()
        if mode == 'timeout':
            await asyncio.sleep(100)
        if mode == 'tamper':
            (root / 'runner.py').write_text('print("allclose: True")')
        elif mode != 'no_edit':
            (root / 'source/kernel.py').write_text('100' if len(sessions) == 1 else '1')
            (root / 'source/helper.py').write_text('6')
            nested = root / 'source/nested/new_helper.py'
            nested.parent.mkdir(exist_ok=True)
            nested.write_text('# declared tree helper, generated by scripted Forge provider\n')
        changes = guard.verify()
        if usage is not None:
            usage.add_usage({'input_tokens': 11, 'output_tokens': 3}, total_cost_usd=0, role='implementer')
        # This text is deliberately untrustworthy. Neither gate nor adapter may
        # accept an invalid candidate because a provider says it succeeded.
        return AgentRunResult(text='PLAN: completed\nport_ok: true\nallclose: True', subtype='success',
                              session_id='scripted-session', file_changes=changes, num_turns=1)

def create_backend(runtime, **kwargs):
    return ScriptedProvider(runtime)
agent.create_registered_backend = create_backend

def loop_callback(**kwargs):
    loop_calls.append(kwargs)
    assert kwargs['kernel_backend'] == language
    assert kwargs['baseline_json'] == plan['baseline']
    assert kwargs['deadline_unix'] == plan['deadline_unix']
    assert str(root / 'source/nested/new_helper.py') in kwargs['source_files']
    initial = json.loads(Path(plan['initialization_result']).read_text())
    assert initial['status'] == 'PASS'
    # Run the real loop's initial candidate measurement against an independent
    # faster baseline. This must establish a <1x starting incumbent, not fail
    # initialization or substitute the stub as the best implementation.
    baseline = json.loads(Path(plan['baseline']).read_text())
    state = SimpleNamespace(ic=SimpleNamespace(driver_script=str(root / 'arena_forge_driver.py'),
                build_command=None, bench_timeout_sec=30, bench_repeat=1),
                _baseline_case_times=baseline['case_times'], _unscored_cases=set(),
                _persist_scoring_state=lambda: None)
    elapsed = asyncio.run(IterationLoop._measure_baseline(state))
    assert elapsed == 8, elapsed
    assert state.search_start_mean_case_speedup == .5
    assert IterationLoop._incumbent_mean_case_speedup(state) == .5
    result = dict(best_commit=initial['commit'], improved=False, iteration_count=0,
                  llm_usage={'calls': 1, 'input_tokens': 7, 'output_tokens': 2, 'cost_available': True,
                             'cost_source': 'provider', 'total_cost_usd': 0})
    Path(plan['result']).write_text(json.dumps(result))
    if mode == 'loop_error':
        raise __import__('click').exceptions.Exit(3)

cli.main.commands['forge-loop'].callback = loop_callback
try:
    engine.main(json.loads(os.environ['CPU_TEST_ARGV']))
except BaseException as error:
    if mode == 'success':
        raise
    report = json.loads(Path(plan['initialization_result']).read_text())
    if mode == 'loop_error':
        assert isinstance(error, SystemExit) and error.code == 3, repr(error)
        assert report['status'] == 'PASS' and len(loop_calls) == 1
    else:
        assert report['status'] == 'FAILED', report
        assert not loop_calls
    assert (context.workspace / 'source/kernel.py').read_text() == '2'
    if mode == 'no_edit':
        assert len(report['attempts']) == 2
        assert all(row['status'] == 'REJECTED' for row in report['attempts'])
else:
    assert mode == 'success'
    assert len(loop_calls) == 1 and len(sessions) == 2
    result = json.loads(Path(plan['result']).read_text())
    report = result['initialization']
    assert [row['status'] for row in report['attempts']] == ['REJECTED', 'PASS']
    assert report['attempts'][-1]['gate_passed'] is True
    assert report['llm_usage']['calls'] == 2
    assert result['llm_usage']['input_tokens'] == 29
    assert report['arena_verdict'] == 'pending'
    selected = committed_candidate(context.spec, root, result['best_commit'], root.parent / 'delivery')
    install_candidate(context.spec, selected, context.workspace)
    assert (context.workspace / 'source/nested/new_helper.py').exists()
    assert (context.workspace / 'source/kernel.py').read_text() == '1'
    assert (context.baseline_workspace / 'source/kernel.py').read_text() == '2'
    assert 'phase_deadline_unix' not in json.loads(Path(os.environ['ARENA_FORGE_PLAN']).read_text())
print('REAL_FORGE_INITIALIZATION_CPU_PASS')
'''


@pytest.mark.parametrize("language,mode", [("hip", "success"), ("triton", "success"),
                                          ("hip", "no_edit"), ("triton", "tamper"), ("hip", "timeout"),
                                          ("triton", "loop_error")])
def test_installed_upstream_initialization(tmp_path, language, mode):
    python = os.environ.get("AKA_FORGE_PROBE_PYTHON")
    if not python:
        pytest.skip("Set AKA_FORGE_PROBE_PYTHON to the pinned Hyperloom[forge] interpreter")
    context, plan, path = fixture_task(tmp_path, language=language, initial_state="unimplemented")
    document = json.loads(context.path.read_text())
    document["task_config"]["candidate"]["editable"] = [{"path": "source", "scope": "tree"}]
    context.path.write_text(json.dumps(document))
    config = adapter._config({"agent": {"agent_backend": "codex", "model": "scripted-model",
                                        "initialization_max_attempts": 2, "session_timeout_seconds": 30}})
    if mode == "timeout":
        config["session_timeout_seconds"] = 1
    plan.update(workflow="initialize", agent_config=config, gpu_arch="gfx950", gpu_type="mi355x",
                deadline_unix=time.time()+120, initialization_result=str(tmp_path / "initialize.json"))
    path.write_text(json.dumps(plan))
    engine = Path(plan["engine_root"])
    (engine / "source/kernel.py").write_text("")
    # The accepted first implementation is *slower* than baseline: initialization
    # must accept correctness and let the ordinary loop improve it later.
    for workspace in (engine, Path(plan["template"]), context.baseline_workspace):
        runner = workspace / "runner.py"
        runner.write_text(runner.read_text().replace('else 2, benchmark_method', 'else 8, benchmark_method'))
    from agents.forge.program import program_text
    Path(plan["program"]).write_text(program_text(plan, initialize=True))
    (engine / "arena_forge_driver.py").write_text(bridge.render_driver(path, ROOT))
    context = type(context).load(context.path)
    allow_candidate_paths(engine, context.spec)
    adapter._initialize_git(engine)
    result = bridge.execute(plan, engine, role="baseline", action="performance")
    Path(plan["baseline"]).write_text(json.dumps(dict(wall_ms=4, case_times=bridge.timings(result))))
    command = adapter.build_command(plan, context, config, gpu_arch="gfx950", gpu_type="mi355x")
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT), ARENA_FORGE_PLAN=str(path), CPU_TEST_MODE=mode,
               CPU_TEST_ARGV=json.dumps(command[2:]))
    run = subprocess.run([python, "-c", UPSTREAM_INITIALIZATION], cwd=engine, env=env,
                         capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stdout[-5000:] + run.stderr[-6000:]
    assert "REAL_FORGE_INITIALIZATION_CPU_PASS" in run.stdout
