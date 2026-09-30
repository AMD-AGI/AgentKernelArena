"""CPU contract checks; these do not establish GPU correctness or performance."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.measurement import paired_throughput
from src.run_budget import agent_budget, budget_policy, open_budget, reserve_from_validation
from src.serving_runtime import gpu_groups, selected_serving_tasks, validate_runtime_lock
from src.task_protocol import parse_command_result, performance_cases, TaskProtocolError
from src.perf_helper_materialization import materialize_perf_helpers_in_workspace

ROOT = Path(__file__).resolve().parents[1]


def report(rate=100):
    row = dict(test_case_id='serving', status='PASS', shape=[8,128,128], dtype='bfloat16',
               params=dict(num_requests=10, output_tokens_per_request=20),
               execution_time_ms=200/rate*1000, benchmark_method='serving_wall_clock',
               metrics=dict(completed_requests=10, output_tokens=200, duration_s=200/rate,
                            output_tokens_per_s=rate), metadata=dict(runtime_fingerprint='abc'))
    return dict(protocol='arena-eval-v1', role='candidate', action='performance', status='PASS', cases=[row])


def parse(data, kind='serving'):
    return parse_command_result('ARENA_EVAL_RESULT='+json.dumps(data), role='candidate', action='performance',
                                returncode=0, measurement=kind)


def test_serving_is_opt_in_and_rejects_changed_work():
    assert parse(report()).passed
    with pytest.raises(TaskProtocolError):
        parse(report(), 'kernel')
    for field,value in [('completed_requests',9),('output_tokens',199),('duration_s',3),('output_tokens_per_s',101)]:
        bad = report()
        bad['cases'][0]['metrics'][field] = value
        with pytest.raises(TaskProtocolError):
            parse(bad)
    for latency in (float('nan'), float('inf'), -1):
        bad = report()
        bad['cases'][0]['metrics']['p99_tpot_ms'] = latency
        with pytest.raises(TaskProtocolError):
            parse(bad)
    bad = report()
    bad['cases'][0]['params']['input_tokens'] = 128
    bad['cases'][0]['metrics']['input_tokens'] = 1279
    with pytest.raises(TaskProtocolError, match='input token'):
        parse(bad)


def test_explicit_paired_metric_uses_median_and_locks_runtime():
    base = performance_cases(parse(report()))
    candidate = [performance_cases(parse(report(rate))) for rate in [101,102,180]]
    result = paired_throughput([(base, rows) for rows in candidate])
    assert result['speedup_ratio'] == pytest.approx(1.02)
    assert result['paired_ratios']['serving'] == pytest.approx([1.01,1.02,1.8])
    candidate[0][0].metadata['runtime_fingerprint'] = 'other'
    with pytest.raises(ValueError, match='runtime'):
        paired_throughput([(base,candidate[0])])
    with pytest.raises(ValueError, match='between pairs'):
        paired_throughput([(base,base),(candidate[0],candidate[0])])


def test_budget_persists_and_run_limit_overrides_template(tmp_path, monkeypatch):
    monkeypatch.setattr('src.run_budget.time.time', lambda: 1000)
    policy = budget_policy(dict(budget=dict(task_wall_time_s=86400,final_evaluation_reserve_s=7200)))
    path = tmp_path/'budget.json'
    record = open_budget(path,policy)
    monkeypatch.setattr('src.run_budget.time.time', lambda: 4600)
    assert open_budget(path,policy) == record
    assert agent_budget(dict(agent={}),record)['agent']['timeout_seconds'] == 75600
    assert agent_budget(dict(agent=dict(timeout_seconds=10)),record)['agent']['timeout_seconds'] == 10
    with pytest.raises(ValueError):
        open_budget(path,None)
    record['effective_reserve_s'] = 86000
    with pytest.raises(TimeoutError):
        agent_budget(dict(agent={}),record)


def test_groups_only_address_assigned_gpus():
    assert gpu_groups(dict(resources=dict(gpu_groups=[[0,1],[2,3]])), ['2','3','6','7']) == [['2','3'],['6','7']]
    for value in ([[0],[0]], [[4]], [[True]], [[]]):
        with pytest.raises(ValueError):
            gpu_groups(dict(resources=dict(gpu_groups=value)), ['2','3','6','7'])


@pytest.mark.parametrize('source', ['model', 'magpie', 'inferencex'])
@pytest.mark.parametrize('revision', ['main', 'v1.0', 'abc1234', None])
def test_runtime_lock_rejects_floating_or_incomplete_revisions(source, revision):
    lock = json.loads((ROOT/'tasks/e2e/qwen3_0_6b_sglang/runtime.lock.json').read_text())
    target = lock['model'] if source == 'model' else lock['dependencies'][source]
    target['revision'] = revision
    with pytest.raises(ValueError, match='fixed 40-character commit'):
        validate_runtime_lock(lock)


@pytest.mark.parametrize('key,value', [('version', True), ('version', 2),
    ('gpu_count', True), ('gpu_count', 0), ('minimum_final_evaluation_s', -1)])
def test_runtime_lock_rejects_invalid_version_or_resource_limits(key, value):
    lock = json.loads((ROOT/'tasks/e2e/qwen3_0_6b_sglang/runtime.lock.json').read_text())
    lock[key] = value
    with pytest.raises(ValueError):
        validate_runtime_lock(lock)


def test_small_task_runtime_and_materialization(tmp_path):
    selected = selected_serving_tasks(ROOT/'example_configs/e2e_qwen3_codex_mi355x.yaml', ROOT)
    source, spec, lock = selected['e2e/qwen3_0_6b_sglang']
    assert lock['gpu_count'] == 1
    assert spec.to_mapping()['evaluation']['measurement']['pairs'] == 3
    import shutil
    shutil.copytree(source, tmp_path/'task')
    materialize_perf_helpers_in_workspace(tmp_path/'task')
    assert (tmp_path/'task/scripts/_aka_serving.py').is_file()
    assert materialize_perf_helpers_in_workspace(tmp_path/'task') == []
    import subprocess, sys
    run = subprocess.run([sys.executable,'scripts/evaluate.py','validate-task'], cwd=tmp_path/'task',
                         capture_output=True, text=True)
    manifest = parse_command_result(run.stdout,role='task',action='validate-task',returncode=run.returncode,measurement='serving')
    assert manifest.passed
    assert manifest.cases[0]['params']['num_requests'] == 80
    assert manifest.metadata['candidate_state'] == 'implemented'
    (tmp_path/'task/source/rmsnorm.py').write_text('def rms_norm(*args): pass\n')
    invalid = subprocess.run([sys.executable,'scripts/evaluate.py','validate-task'], cwd=tmp_path/'task',
                             capture_output=True,text=True)
    assert invalid.returncode != 0 and 'locked production source' in invalid.stdout



def test_clean_executor_rejects_nonaction_before_docker(tmp_path):
    from src.serving_runtime import CleanExecutor
    selected = selected_serving_tasks(ROOT/'example_configs/e2e_qwen3_codex_mi355x.yaml', ROOT)
    executor = CleanExecutor(selected, ROOT, tmp_path/'artifacts', '2', tmp_path/'cache')
    with pytest.raises(ValueError):
        executor.execute(dict(task_id='e2e/qwen3_0_6b_sglang',role='candidate',action='shell',phase='candidate_evaluation'))
    with pytest.raises(ValueError, match='Allocated'):
        CleanExecutor(selected,ROOT,tmp_path/'other','2,3',tmp_path/'cache')


def test_task_kernel_policy_rejects_environment_and_flag_mutation(tmp_path):
    import importlib.util
    source = ROOT/'tasks/e2e/qwen3_0_6b_sglang/scripts/checks.py'
    spec = importlib.util.spec_from_file_location('serving_contract',source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = tmp_path
    (tmp_path/'source').mkdir()
    candidate = tmp_path/'source/rmsnorm.py'
    for text in ['import os\nos.environ["TP"]="2"',
                 'import torch\ntorch.backends.cuda.matmul.allow_tf32=True',
                 'exec("print(1)")']:
        candidate.write_text(text)
        with pytest.raises(ValueError):
            module.source_policy()
    candidate.write_text('import triton\nimport triton.language as tl\n@triton.jit\ndef kernel(x):\n    tl.store(x, 0)\n')
    module.source_policy(target_only=True)
    candidate.write_text('import aiter\n')
    module.source_policy()
    with pytest.raises(ValueError):
        module.source_policy(target_only=True)
    candidate.write_text('import torch as t\ndef rms_norm(x,w,e): return t.nn.functional.rms_norm(x, (1024,), w, e)\n')
    with pytest.raises(ValueError,match='allocation and metadata'):
        module.source_policy(target_only=True)


def test_reserve_scales_with_measured_startup(tmp_path,monkeypatch):
    monkeypatch.setattr('src.run_budget.time.time',lambda: 1000)
    (tmp_path/'runtime.lock.json').write_text(json.dumps(dict(minimum_final_evaluation_s=600)))
    spec=SimpleNamespace(to_mapping=lambda:dict(evaluation=dict(measurement=dict(runtime_lock='runtime.lock.json',pairs=3))))
    results={('task_validation','baseline',action):SimpleNamespace(commands=[SimpleNamespace(elapsed_s=seconds)])
             for action,seconds in [('compile',30),('correctness',100),('performance',500)]}
    session=SimpleNamespace(spec=spec,workspace=tmp_path,results=results)
    record=dict(policy=dict(final_evaluation_reserve_s=1200),deadline_epoch=10000)
    reserved=reserve_from_validation(record,session)
    assert reserved['effective_reserve_s'] == 4695
    assert record['policy']['final_evaluation_reserve_s'] == 1200
    record['deadline_epoch']=2000
    with pytest.raises(TimeoutError):
        reserve_from_validation(record,session)


def test_trusted_templates_cannot_live_in_agent_checkout(tmp_path):
    from src.serving_runtime import CleanExecutor
    with pytest.raises(ValueError, match='outside'):
        CleanExecutor({}, tmp_path, tmp_path/'logs', '0', tmp_path/'cache')


def test_server_hook_does_not_import_framework_in_compiler_subprocess(tmp_path):
    import os, subprocess, sys
    hook = ROOT/'tasks/e2e/qwen3_0_6b_sglang/scripts'
    # The CPU environment intentionally has no torch or SGLang installed.
    env = {**os.environ, 'AKA_INSTALL_KERNEL':'1', 'PYTHONPATH':str(hook)}
    run = subprocess.run([sys.executable,'-c',"import sys; assert 'torch' not in sys.modules; assert 'sglang' not in sys.modules; print('registered')"],
                         cwd=tmp_path,env=env,capture_output=True,text=True)
    assert run.returncode == 0 and run.stdout.strip() == 'registered'
    assert 'Error in sitecustomize' not in run.stderr


def test_resumed_baseline_must_match_trusted_template(tmp_path,monkeypatch):
    import shutil
    from src.serving_runtime import CleanExecutor
    from src.task_spec import load_task_spec
    source=tmp_path/'source'
    shutil.copytree(ROOT/'tasks/e2e/qwen3_0_6b_sglang',source)
    checkout=tmp_path/'checkout'
    baseline=checkout/'run/baseline'
    shutil.copytree(source,baseline)
    (baseline/'source/rmsnorm.py').write_text('def changed(): pass\n')
    lock=json.loads((source/'runtime.lock.json').read_text())
    spec=load_task_spec(source/'config.yaml',task_id='e2e/test')
    monkeypatch.setattr('src.perf_helper_materialization.materialize_perf_helpers_in_workspace',lambda *args,**kwargs:None)
    executor=CleanExecutor({'e2e/test':(source,spec,lock)},checkout,tmp_path/'trusted','2',tmp_path/'cache')
    with pytest.raises(ValueError,match='Frozen baseline'):
        executor.execute(dict(task_id='e2e/test',workspace='/workspace/run/baseline',role='baseline',
                              action='performance',phase='candidate_evaluation',timeout=100))


def test_model_logprobs_require_complete_finite_aligned_values():
    import importlib.util
    spec = importlib.util.spec_from_file_location('serving_adapter',ROOT/'src/tools/perf/serving.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [[None,10,None],[-1.01,20,None],[-2.0,30,None]]
    assert module.logprob_error([10,20,30],[-1,-2],rows) == pytest.approx(.01)
    for bad in (rows[:-1], [rows[0],[-1.01,21,None],rows[2]],
                [rows[0],[float('nan'),20,None],rows[2]]):
        with pytest.raises(ValueError):
            module.logprob_error([10,20,30],[-1,-2],bad)


def test_final_baseline_action_uses_frozen_role_and_requires_validation():
    from src.task_session import TaskSession
    session = SimpleNamespace(initial_validation=SimpleNamespace(accepted=True),
                              _execute=lambda *args: args)
    assert TaskSession.baseline_action(session,'performance') == ('baseline','performance','candidate_evaluation')
    session.initial_validation.accepted = False
    with pytest.raises(RuntimeError,match='accepted initial'):
        TaskSession.baseline_action(session,'performance')


def test_runtime_identity_ignores_only_transport_and_diagnostic_ids():
    import importlib.util
    spec = importlib.util.spec_from_file_location('serving_adapter',ROOT/'src/tools/perf/serving.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = [dict(server_settings=dict(port=10000,tp_size=1),
                  runtime_settings=dict(SGLANG_RUN_ID='run-a',SGLANG_USE_AITER='1'))]
    initial = module.runtime_identity('locked',calls)[0]
    calls[0]['server_settings']['port'] = 20000
    calls[0]['runtime_settings']['SGLANG_RUN_ID'] = 'run-b'
    assert module.runtime_identity('locked',calls)[0] == initial
    calls[0]['runtime_settings']['SGLANG_USE_AITER'] = '0'
    assert module.runtime_identity('locked',calls)[0] != initial
    calls[0]['runtime_settings']['SGLANG_USE_AITER'] = '1'
    calls[0]['server_settings']['tp_size'] = 2
    assert module.runtime_identity('locked',calls)[0] != initial
