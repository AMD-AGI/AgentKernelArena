"""Missing candidate timing must not invent a baseline-speed incumbent."""
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from agents.forge.incumbent import scoreable_trial


def trial():
    return SimpleNamespace(
        validation_passed=True, crashed=False, integrity_violation=False,
        workspace_contention="", wall_ms=50.0, mean_case_speedup=.02,
        bench_detail={"success": True, "case_coverage_complete": True,
                      "case_times": {"a": 50.0, "b": 50.0},
                      "measurements": [{"case_times": {"a": 50.0, "b": 50.0}} for _ in range(3)]},
    )


@pytest.mark.parametrize("failure", ["correctness", "crash", "integrity", "contention",
    "benchmark", "measurement_failure", "coverage", "scalar_only", "missing_case", "excluded_case", "nonfinite", "zero", "boolean"])
def test_first_candidate_still_needs_complete_validated_timing(failure):
    result = trial()
    assert scoreable_trial(result, {"a": 1.0, "b": 1.0})
    if failure == "correctness": result.validation_passed = False
    elif failure == "crash": result.crashed = True
    elif failure == "integrity": result.integrity_violation = True
    elif failure == "contention": result.workspace_contention = "foreign process"
    elif failure == "benchmark": result.bench_detail["success"] = False
    elif failure == "measurement_failure": result.bench_detail["measurements"][0]["success"] = False
    elif failure == "coverage": result.bench_detail["case_coverage_complete"] = False
    elif failure == "scalar_only": result.bench_detail["measurements"] = []
    elif failure == "missing_case": del result.bench_detail["measurements"][1]["case_times"]["b"]
    elif failure == "excluded_case": result.bench_detail["measurements"][1]["unscored_cases"] = ["b"]
    elif failure == "nonfinite": result.bench_detail["measurements"][1]["case_times"]["a"] = float("inf")
    elif failure == "zero": result.mean_case_speedup = 0.0
    elif failure == "boolean": result.wall_ms = True
    assert not scoreable_trial(result, {"a": 1.0, "b": 1.0})


def test_pinned_native_first_measured_candidate_keeps_full_gates(tmp_path):
    python = os.environ.get("AKA_FORGE_PROBE_PYTHON")
    if not python:
        pytest.skip("Set AKA_FORGE_PROBE_PYTHON to the pinned Hyperloom[forge] interpreter")
    root = Path(__file__).resolve().parents[1]
    script = r'''
import asyncio, copy, os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from agents.forge import upstream, incumbent
from kernelforge.loop import runner, insession_gate
upstream.probe()
original = runner.IterationLoop
original_gate = insession_gate.InSessionGate
incumbent.install()
adapted = runner.IterationLoop
incumbent.install()
assert adapted is runner.IterationLoop
directory = Path(os.environ['INCUMBENT_TEST_DIR'])
(directory/'kernel.py').write_text('def kernel(): return 1\n')
config = runner.IterationConfig(kernel_file=str(directory/'kernel.py'),
    driver_script=str(directory/'driver.py'), workspace_dir=str(directory),
    baseline_case_times={'a': 1.0}, pristine_baseline_wall_ms=1.0)
loop = adapted(config, None)
loop.run_state = SimpleNamespace(baseline_case_times={}, best_case_times={},
    unscored_cases=[], search_start_mean_case_speedup=None,
    baseline_wall_ms=None, best=SimpleNamespace(commit_hash='', wall_ms=None))
loop.state_store = SimpleNamespace(save=lambda state: None)
loop._best_case_times = {'a': 1.0}  # reproduce native startup's external anchor copy
baseline = copy.deepcopy(loop._baseline_case_times)
failed = {'success': False, 'message': 'captured candidate failed'}
measurement = {'success': True, 'median_ms': 50.0, 'case_times': {'a': 50.0}}
bench = {**measurement, 'measurements': [copy.deepcopy(measurement) for _ in range(3)]}
report = SimpleNamespace(all_passed=True, results=[SimpleNamespace(passed=True,
    stage=1, stage_name='full', snr_db=None)], summary=lambda: 'full task correctness PASS')
canonical = SimpleNamespace(passed=True, outcome='', detail='', output='')
async def run():
    with patch.object(runner, 'measure_wallclock', AsyncMock(return_value=failed)):
        assert await loop._measure_baseline() is None
    assert loop._baseline_case_times == baseline
    assert loop._best_case_times == {} and loop._incumbent_mean_case_speedup() is None
    assert incumbent.needs_incumbent()
    received=[]
    def gate_init(self, *args, **kwargs): received.append(kwargs)
    with patch.object(original_gate, '__init__', gate_init):
        insession_gate.InSessionGate(driver_script='driver.py', snr_threshold=30)
    assert received[-1]['correctness_only'] is True

    with patch.object(runner, 'force_jit_rebuild'), \
         patch.object(runner, 'run_validation_pipeline', AsyncMock(return_value=report)) as validate, \
         patch.object(runner, 'measure_wallclock', AsyncMock(side_effect=lambda **kw: copy.deepcopy(bench))) as measure, \
         patch.object(runner, 'check_registers', AsyncMock(return_value={'success': False})), \
         patch.object(runner, 'accept_candidate', AsyncMock(return_value=canonical)) as accept:
        old = await original.run_one_iteration(loop, 1)
        assert old.validation_passed and not old.kept and old.mean_case_speedup == .02
        assert accept.await_count == 0
        result = await loop.run_one_iteration(1)
        assert result.kept and result.validation_passed and result.mean_case_speedup == .02
        assert measure.await_args.kwargs['measurements'] == 3
        assert validate.await_count == 2 and accept.await_count == 1
        assert result.bench_detail['arena_speedup_improvement_claimed'] is False
        assert loop._baseline_case_times == baseline and incumbent.needs_incumbent()
        # The normal native commit path promotes only after a successful commit.
        loop._promote_best(result)
        loop.best_mean_case_speedup = result.mean_case_speedup
        assert not incumbent.needs_incumbent() and loop._best_case_times == {'a': 50.0}
        with patch.object(original_gate, '__init__', gate_init):
            insession_gate.InSessionGate(driver_script='driver.py', snr_threshold=30)
        assert not received[-1].get('correctness_only')
        # The unchanged performance rule rejects an equal later candidate.
        later = await loop.run_one_iteration(2)
        assert not later.kept and accept.await_count == 1

        loop._mark_unmeasured()
        accept.return_value = SimpleNamespace(passed=False, outcome='canonical_failure',
            detail='task rejection', output='full rejection evidence')
        rejected = await loop.run_one_iteration(3)
        assert not rejected.kept and not rejected.validation_passed
        assert rejected.error_output == 'full rejection evidence'
        assert incumbent.needs_incumbent()
        calls = accept.await_count
        report.all_passed=False; report.failed_stage=1; report.failed_outcome='numeric'; report.failed_output='mismatch'
        rejected = await loop.run_one_iteration(4)
        assert not rejected.kept and not rejected.validation_passed and accept.await_count == calls
        report.all_passed=True
        bench['measurements'][1]['case_times'] = {}
        rejected = await loop.run_one_iteration(5)
        assert not rejected.kept and accept.await_count == calls

    # A resume without any measured candidate must not rehydrate anchor times
    # as a candidate. A durable measured KEEP remains a normal incumbent.
    loop.run_state.best_case_times = {'a': 1.0}
    loop._restore_scoring_state()
    assert loop._best_case_times == {} and incumbent.needs_incumbent()
    other = adapted(config, None)
    other.run_state = SimpleNamespace(best_case_times={'a': 50.0}, unscored_cases=[],
        search_start_mean_case_speedup=None, baseline_wall_ms=None,
        best=SimpleNamespace(commit_hash='measured-keep', wall_ms=50.0))
    other._restore_scoring_state()
    assert other._best_case_times == {'a': 50.0} and not incumbent.needs_incumbent()
asyncio.run(run())

# Exercise the actual native commit/publication/event path as well. Provider,
# GPU measurements and correctness calls are CPU fixtures, never GPU evidence.
import json, subprocess
from kernelforge.config import Config
from kernelforge.tracker import ExperimentTracker
native = directory/'native_commit';native.mkdir()
(native/'kernel.py').write_text('def kernel(): return 1\n')
(native/'driver.py').write_text('print("CPU test fixture")\n')
for argv in [['git','init'],['git','config','user.email','test@example.invalid'],
             ['git','config','user.name','CPU Test'],['git','add','kernel.py','driver.py'],
             ['git','commit','-m','initial fixture']]:
    subprocess.run(argv,cwd=native,check=True,capture_output=True)
config = runner.IterationConfig(kernel_file=str(native/'kernel.py'),
    driver_script=str(native/'driver.py'), workspace_dir=str(native),
    baseline_case_times={'a':1.0}, pristine_baseline_wall_ms=1.0,
    target_wall_ms=100, max_time_hours=2, budget_reserve_sec=60)
loop = adapted(config, ExperimentTracker(native/'experiments'),
               Config(workspace=str(native), experiments_dir=native/'experiments'))
async def implement(kernel, history, session_sink=None):
    assert incumbent.needs_incumbent()
    Path(kernel).write_text('def kernel(): return 2\n')
    return 'CPU recovery candidate'
report.all_passed=True
measurement = {'success':True,'median_ms':50.0,'case_times':{'a':50.0}}
bench = {**measurement,'measurements':[copy.deepcopy(measurement)for _ in range(3)]}
async def complete():
    with patch.object(runner, 'force_jit_rebuild'), \
         patch.object(runner, 'run_validation_pipeline', AsyncMock(return_value=report)), \
         patch.object(runner, 'measure_wallclock', AsyncMock(side_effect=[failed,copy.deepcopy(bench)])), \
         patch.object(runner, 'check_registers', AsyncMock(return_value={'success':False})), \
         patch.object(runner, 'accept_candidate', AsyncMock(return_value=canonical)):
        results=await loop.run(agent_fn=implement)
    assert len(results)==1 and results[0].kept and results[0].commit_hash
    commit=results[0].commit_hash
    committed=subprocess.check_output(['git','show',commit+':kernel.py'],cwd=native)
    assert committed==b'def kernel(): return 2\n'
    events=loop.state_store.read_events()
    trials=[e for e in events if e.get('type')=='iteration_result']
    assert len(trials)==1 and trials[0]['decision']=='KEEP'
    assert trials[0]['commit_hash']==commit and trials[0]['mean_case_speedup']==.02
    manifest=json.loads(loop.best_publisher.manifest_path.read_text())
    assert manifest['commit_hash']==commit and manifest['baseline_wall_ms']==1.0
    assert manifest['mean_case_speedup']==.02
    assert loop.search_start_mean_case_speedup==.02 and not incumbent.needs_incumbent()
asyncio.run(complete())
'''
    run = subprocess.run([python, "-c", script], cwd=root,
                         env=dict(os.environ, PYTHONPATH=str(root), INCUMBENT_TEST_DIR=str(tmp_path)),
                         capture_output=True, text=True, timeout=90)
    assert run.returncode == 0, run.stdout + run.stderr
