"""Native deadline cancellation must revert the attempt and publish its real best."""
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agents.forge.deadline import _active_round_budget, bound_session


def test_sessions_reserve_assessment_in_addition_to_delivery():
    from test_forge_deadline import Session
    loop = SimpleNamespace(_time_remaining=lambda: 880,
                           _measurement_estimate_sec=lambda: 600)
    token = _active_round_budget.set(loop)
    try:
        with patch("agents.forge.deadline.time.time", return_value=1000):
            assert bound_session(Session(), {"deadline_unix": 2000}).timeout_sec == 280
            # PORT owns a separate phase and must not inherit a loop's reserve.
            assert bound_session(Session(), {"deadline_unix": 2000,
                "phase_deadline_unix": 1500}).timeout_sec == 500
            loop._time_remaining = lambda: 590
            with pytest.raises(TimeoutError):
                bound_session(Session(), {"deadline_unix": 2000})
    finally:
        _active_round_budget.reset(token)


def test_pinned_admission_and_session_share_complete_assessment_estimate():
    python = os.environ.get("AKA_FORGE_PROBE_PYTHON")
    if not python:
        pytest.skip("Set AKA_FORGE_PROBE_PYTHON to the pinned Hyperloom[forge] interpreter")
    root = Path(__file__).resolve().parents[1]
    script = r'''
from types import SimpleNamespace
from unittest.mock import patch
from agents.forge import upstream, deadline, round_budget
from kernelforge.loop import runner
upstream.probe()
deadline.install()
original = runner.IterationLoop
round_budget.install()
bounded = runner.IterationLoop
before, after = object.__new__(original), object.__new__(bounded)
for loop in (before, after):
    loop.start_time = 1000
    loop.ic = SimpleNamespace(max_time_hours=2, deadline_unix=3260,
                              budget_reserve_sec=60, lanes=1)
    loop.run_state = SimpleNamespace(round_costs=SimpleNamespace(recent=[]))
    loop.state_store = SimpleNamespace(append_event=lambda event: None)
with patch('time.time', return_value=1000), \
     patch.object(original, '_measurement_estimate_sec', return_value=1000):
    assert before._time_remaining() == 2260
    assert after._time_remaining() == 2140
    assert before._admit_next_round(2) == 1
    assert after._admit_next_round(2) is None
    assert after.termination_reason == 'round_budget_exhausted'
    for loop in (before, after): loop.ic.deadline_unix = 2800
    assert before._admit_dispatch(2)
    assert not after._admit_dispatch(2)
    token = deadline._active_round_budget.set(after)
    try:
        assert deadline._available({'deadline_unix': 2800}) == 680
        assert after._analysis_deadline_unix() == 1680
    finally: deadline._active_round_budget.reset(token)
    # Complete observed cost includes all gates (not just the last benchmark).
    after._arena_initial_measurement = 900
    assert after._measurement_estimate_sec() == 1500
    after._arena_assessment_high_water = 2000
    assert after._measurement_estimate_sec() == 2500
'''
    result = subprocess.run([python, "-c", script], cwd=root,
        env=dict(os.environ, PYTHONPATH=str(root)), text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("stage", ["validation", "benchmark", "canonical", "first_incumbent"])
def test_pinned_native_late_assessment_reverts_and_finishes(tmp_path, stage):
    python = os.environ.get("AKA_FORGE_PROBE_PYTHON")
    if not python:
        pytest.skip("Set AKA_FORGE_PROBE_PYTHON to the pinned Hyperloom[forge] interpreter")
    root = Path(__file__).resolve().parents[1]
    # Real native loop, Git/checkpoint/event/publication and subprocess cleanup.
    # Only provider edits and GPU numerical/measurement responses are CPU fixtures.
    script = r'''
import asyncio, copy, json, os, subprocess, sys, time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from agents.forge import upstream, deadline, incumbent, round_budget
from kernelforge.loop import runner
from kernelforge.config import Config
from kernelforge.tracker import ExperimentTracker
from kernelforge.mcp_server.tools._subprocess import communicate_process_group
upstream.probe()
deadline.install()
incumbent.install()
round_budget.install()
adapted = runner.IterationLoop
round_budget.install()
assert adapted is runner.IterationLoop
directory = Path(os.environ['ROUND_TEST_DIR'])
stage = os.environ['ROUND_TEST_STAGE']
first = stage == 'first_incumbent'
(directory/'kernel.py').write_text('def kernel(): return 1\n')
(directory/'driver.py').write_text('print("CPU fixture only")\n')
for argv in [['git','init'],['git','config','user.email','test@example.invalid'],
             ['git','config','user.name','CPU Test'],['git','add','kernel.py','driver.py'],
             ['git','commit','-m','CPU initial fixture']]:
    subprocess.run(argv, cwd=directory, check=True, capture_output=True)
initial = subprocess.check_output(['git','rev-parse','HEAD'], cwd=directory, text=True).strip()
config = runner.IterationConfig(kernel_file=str(directory/'kernel.py'),
    driver_script=str(directory/'driver.py'), workspace_dir=str(directory),
    baseline_case_times={'a': 1.0}, pristine_baseline_wall_ms=1.0,
    target_wall_ms=.001, max_time_hours=2, deadline_unix=time.time()+7200)
loop = adapted(config, ExperimentTracker(directory/'experiments'),
               Config(workspace=str(directory), experiments_dir=directory/'experiments'))
report = SimpleNamespace(all_passed=True, results=[SimpleNamespace(passed=True,
    stage=1, stage_name='full', snr_db=None)], summary=lambda: 'CPU complete correctness PASS')
canonical = SimpleNamespace(passed=True, outcome='', detail='', output='')
attempts, children, benchmarks, canonicals = [], [], [], []
def bench(ms):
    measurement = {'success': True, 'median_ms': ms, 'case_times': {'a': ms}}
    return {**measurement, 'measurements': [copy.deepcopy(measurement) for _ in range(3)]}

async def blocked():
    proc = await asyncio.create_subprocess_exec(sys.executable, '-c',
        'import time; time.sleep(60)', start_new_session=True,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    children.append(proc)
    await communicate_process_group(proc, timeout=60)
    raise AssertionError('unbounded assessment unexpectedly finished')

async def implement(kernel, history, session_sink=None):
    attempts.append(True)
    Path(kernel).write_text('def kernel(): return %d\n' % (len(attempts)+1))
    if first or len(attempts) == 2:
        # The round was admitted with adequate time, but a changed candidate
        # takes unexpectedly long. Exercise the real cancellation boundary.
        loop.ic.deadline_unix = time.time() + round_budget.FINALIZATION_RESERVE_SEC + .25
    return 'CPU candidate fixture'

def late(): return first or len(attempts) == 2
async def validate(**kw):
    if late() and stage == 'validation': await blocked()
    return report
async def measure(**kw):
    assert kw['measurements'] == 3
    benchmarks.append(len(attempts))
    if not attempts:
        return {'success': False, 'message': 'CPU failed initial timing'} if first else bench(1.0)
    if late() and stage == 'benchmark': await blocked()
    return bench(50.0 if first else .5 / len(attempts))
async def accept(*args, **kw):
    canonicals.append(len(attempts))
    if late() and stage in ('canonical', 'first_incumbent'): await blocked()
    return canonical

async def run():
    with patch.object(runner, 'force_jit_rebuild'), \
         patch.object(runner, 'run_validation_pipeline', validate), \
         patch.object(runner, 'measure_wallclock', measure), \
         patch.object(runner, 'check_registers', AsyncMock(return_value={'success': False})), \
         patch.object(runner, 'accept_candidate', accept):
        results = await loop.run(agent_fn=implement)
    assert time.time() < loop.ic.deadline_unix - 100
    assert len(children) == 1 and children[0].returncode is not None
    assert deadline._active_round_budget.get() is None
    expired = results[-1]
    assert not expired.kept and not expired.validation_passed and not expired.commit_hash
    assert expired.validation_outcome == 'timeout' and not expired.crashed
    assert 'unfinished attempt cancelled' in expired.validation_summary
    assert loop.termination_reason == 'budget_exhausted'
    events = loop.state_store.read_events()
    trials = [e for e in events if e.get('type') == 'iteration_result']
    assert trials[-1]['decision'] == 'REVERT_VALIDATION_TIMEOUT'
    assert not trials[-1].get('commit_hash')
    assert events[-1]['type'] == 'run_terminated'
    assert events[-1]['reason'] == 'budget_exhausted'
    head = subprocess.check_output(['git','rev-parse','HEAD'], cwd=directory, text=True).strip()
    if first:
        assert len(results) == 1 and head == initial
        assert not loop.run_state.best.commit_hash and incumbent.needs_incumbent()
        assert (directory/'kernel.py').read_text() == 'def kernel(): return 1\n'
        assert benchmarks == [0, 1] and canonicals == [1]
    else:
        assert len(results) == 2 and results[0].kept
        selected = results[0].commit_hash
        assert selected == head and selected != initial
        assert trials[0]['decision'] == 'KEEP' and trials[0]['commit_hash'] == selected
        manifest = json.loads(loop.best_publisher.manifest_path.read_text())
        assert manifest['commit_hash'] == selected and manifest['mean_case_speedup'] == 2.0
        assert loop.run_state.best.commit_hash == selected
        assert (directory/'kernel.py').read_text() == 'def kernel(): return 2\n'
        assert subprocess.check_output(['git','show', selected+':kernel.py'], cwd=directory) == \
               (directory/'kernel.py').read_bytes()
        assert benchmarks[:2] == [0, 1] and canonicals[0] == 1
    # A genuine inner action timeout keeps its original exception/evidence.
    loop.ic.deadline_unix = time.time()+7200
    with patch.object(runner, 'force_jit_rebuild'), \
         patch.object(runner, 'run_validation_pipeline', AsyncMock(side_effect=TimeoutError('inner action evidence'))):
        try: await loop.run_one_iteration(9)
        except TimeoutError as error: assert str(error) == 'inner action evidence'
        else: raise AssertionError('inner timeout was mislabeled')
asyncio.run(run())
print('PINNED_ROUND_FINALIZATION_CPU_PASS')
'''
    result = subprocess.run([python, "-c", script], cwd=root,
        env=dict(os.environ, PYTHONPATH=str(root), ROUND_TEST_DIR=str(tmp_path),
                 ROUND_TEST_STAGE=stage), text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout[-7000:] + result.stderr[-5000:]
    assert "PINNED_ROUND_FINALIZATION_CPU_PASS" in result.stdout
