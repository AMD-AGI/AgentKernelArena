"""CPU contract checks; these do not establish GPU correctness or performance."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.measurement import LIMIT_DEFAULTS, lock_limits, paired_throughput
from src.run_budget import agent_budget, budget_policy, open_budget, reserve_from_validation
from src.serving_runtime import gpu_groups, selected_serving_tasks, validate_runtime_lock
from src.task_protocol import parse_command_result, performance_cases, TaskProtocolError
from src.perf_helper_materialization import materialize_perf_helpers_in_workspace

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def serving_adapter():
    import importlib.util
    spec = importlib.util.spec_from_file_location('serving_adapter', ROOT/'src/tools/perf/serving.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_serving_port_avoids_ephemeral_and_busy_auxiliary(serving_adapter, monkeypatch):
    module = serving_adapter
    monkeypatch.setattr(module.Path, 'read_text', lambda self: '32768 60999')
    monkeypatch.setattr(module.random, 'SystemRandom', lambda: SimpleNamespace(shuffle=lambda values: None))
    bound = []
    class Socket:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def bind(self, address):
            bound.append(address[1])
            if address[1] == 20240:
                raise OSError('auxiliary occupied')
    monkeypatch.setattr(module.socket, 'socket', Socket)
    port = module.serving_port()
    assert port == 10241
    assert bound == [10240, 20240, 10241, 20241]
    assert all(not 32768 <= p <= 60999 for p in bound)


def test_serving_port_exhaustion_fails_closed(serving_adapter, monkeypatch):
    monkeypatch.setattr(serving_adapter.Path, 'read_text', lambda self: '1024 65535')
    with pytest.raises(RuntimeError, match='outside the ephemeral range'):
        serving_adapter.serving_port()


@pytest.mark.parametrize('error', ['[Errno 98] error while attempting to bind: address already in use',
                                  'Address already in use'])
def test_serving_bind_failure_cleans_up_even_if_parent_alive(serving_adapter, tmp_path, monkeypatch, error):
    module = serving_adapter
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'serving_port', lambda: 12000)
    monkeypatch.setattr(module, 'launch_command', lambda *args: (['server'], 'health'))
    process = SimpleNamespace(poll=lambda: None)
    def launch(*args, **kwargs):
        kwargs['stdout'].write(error)
        kwargs['stdout'].flush()
        return process
    monkeypatch.setattr(module.subprocess, 'Popen', launch)
    cleaned = []
    def stop(proc, log):
        cleaned.append(proc)
        log.close()
    monkeypatch.setattr(module, 'stop_server', stop)
    monkeypatch.setattr(module, 'request', lambda *args, **kwargs: pytest.fail('must not probe a conflicting server'))
    with pytest.raises(RuntimeError, match='Serving port bind failed'):
        module.start_server({'envs': {}}, {'server_startup_timeout_s': 600})
    assert cleaned == [process]
    assert (tmp_path/'server.log').read_text() == error


def test_serving_readiness_requests_share_startup_deadline(serving_adapter, tmp_path, monkeypatch):
    module = serving_adapter
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'serving_port', lambda: 12000)
    monkeypatch.setattr(module, 'launch_command', lambda *args: (['server'], 'health'))
    monkeypatch.setattr(module.subprocess, 'Popen', lambda *args, **kwargs: SimpleNamespace(poll=lambda: None))
    now, timeouts, cleaned = [0.0], [], []
    monkeypatch.setattr(module.time, 'monotonic', lambda: now[0])
    monkeypatch.setattr(module.time, 'sleep', lambda seconds: now.__setitem__(0, now[0] + seconds))
    def request(port, endpoint, *, timeout):
        timeouts.append(timeout)
        now[0] += timeout
        raise TimeoutError('unresponsive endpoint')
    def stop(proc, log):
        cleaned.append(proc)
        log.close()
    monkeypatch.setattr(module, 'request', request)
    monkeypatch.setattr(module, 'stop_server', stop)
    with pytest.raises(TimeoutError, match='Model startup exceeded 8 seconds'):
        module.start_server({'envs': {}}, {'server_startup_timeout_s': 8})
    assert timeouts == [5, 2]
    assert now == [8] and len(cleaned) == 1


def test_nonroot_image_path_preserves_tools_and_missing_tool_semantics(tmp_path, monkeypatch):
    import importlib.util
    import os
    import subprocess
    spec = importlib.util.spec_from_file_location('serving_adapter', ROOT/'src/tools/perf/serving.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    private, first, second = (tmp_path/name for name in ('private', 'first', 'second'))
    for directory in (private, first, second):
        directory.mkdir()
    for directory, output in ((first, 'first'), (second, 'second')):
        tool = directory/'compiler'
        tool.write_text('#!/bin/sh\necho '+output+'\n')
        tool.chmod(0o755)
    private.chmod(0)
    # Root CI can search mode-000 directories; model the scoring container UID.
    if os.geteuid() == 0:
        access = os.access
        monkeypatch.setattr(os, 'access', lambda path, mode: str(path) != str(private) and access(path, mode))
    monkeypatch.setenv('PATH', os.pathsep.join(map(str, (private, first, tmp_path/'missing', second))))
    try:
        module.normalize_executable_path()
        assert os.environ['PATH'] == os.pathsep.join(map(str, (first, second)))
        assert subprocess.check_output(['compiler'], text=True).strip() == 'first'
        with pytest.raises(FileNotFoundError):
            subprocess.run(['absent-optional-compiler'], check=True)
    finally:
        private.chmod(0o700)


def report(rate=100, *, kernel_sha256='a' * 64):
    row = dict(test_case_id='serving', status='PASS', shape=[8,128,128], dtype='bfloat16',
               params=dict(num_requests=10, output_tokens_per_request=20),
               execution_time_ms=200/rate*1000, benchmark_method='serving_wall_clock',
               metrics=dict(completed_requests=10, output_tokens=200, duration_s=200/rate,
                            output_tokens_per_s=rate), metadata=dict(runtime_fingerprint='abc', kernel_sha256=kernel_sha256))
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
    candidate = [performance_cases(parse(report(rate, kernel_sha256='b' * 64))) for rate in [101,102,180]]
    result = paired_throughput([(base, rows) for rows in candidate])
    assert result['speedup_ratio'] == pytest.approx(1.02)
    assert result['paired_ratios']['serving'] == pytest.approx([1.01,1.02,1.8])
    spread = result['uncertainty']['serving']
    assert spread['ratio_min'] == pytest.approx(1.01) and spread['ratio_max'] == pytest.approx(1.8)
    assert spread['baseline_throughput_cv'] == 0
    assert spread['candidate_throughput_cv'] > 0.3 and spread['median_within_noise'] is True
    candidate[0][0].metadata['runtime_fingerprint'] = 'other'
    with pytest.raises(ValueError, match='runtime'):
        paired_throughput([(base,candidate[0])])
    with pytest.raises(ValueError, match='between pairs'):
        paired_throughput([(base,base),(candidate[0],candidate[0])])


@pytest.mark.parametrize('rates,expected', [([101,102,103], 1.0), ([97,98,99], 0.98)])
def test_unchanged_kernel_keeps_raw_measurements_without_gain_credit(rates, expected):
    baseline = performance_cases(parse(report()))
    pairs = [(baseline, performance_cases(parse(report(rate)))) for rate in rates]
    result = paired_throughput(pairs)
    assert result['speedup_ratio'] == pytest.approx(expected)
    assert result['raw_speedup_ratio'] == pytest.approx(rates[1] / 100)
    assert result['paired_ratios']['serving'] == pytest.approx([r / 100 for r in rates])
    assert result['kernel_changed'] == {'serving': False}
    assert result['scored_case_ratios']['serving'] == pytest.approx(expected)


@pytest.mark.parametrize('bad', [None, '', 'not-a-hash'])
def test_serving_pair_requires_source_identity(bad):
    baseline = performance_cases(parse(report()))
    candidate = performance_cases(parse(report(110, kernel_sha256=bad)))
    with pytest.raises(ValueError, match='kernel hashes'):
        paired_throughput([(baseline, candidate)])


def test_serving_rejects_changing_candidate_between_pairs():
    baseline = performance_cases(parse(report()))
    candidates = [performance_cases(parse(report(110, kernel_sha256=digit * 64))) for digit in 'bc']
    with pytest.raises(ValueError, match='kernel identity changed'):
        paired_throughput([(baseline, rows) for rows in candidates])


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


@pytest.mark.parametrize('seed', [True, -1, 2**31, 1.5, '1729'])
def test_runtime_lock_rejects_invalid_server_seed(seed):
    lock = json.loads((ROOT/'tasks/e2e/qwen3_next_80b_sglang/runtime.lock.json').read_text())
    lock['server_seed'] = seed
    with pytest.raises(ValueError, match='server_seed'):
        validate_runtime_lock(lock)


def test_large_task_locks_seed_in_launch_and_rank_evidence(tmp_path, monkeypatch):
    import importlib.util
    import yaml
    task = ROOT/'tasks/e2e/qwen3_next_80b_sglang'
    spec = importlib.util.spec_from_file_location('large_task_checks', task/'scripts/checks.py')
    checks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checks)
    lock = json.loads((task/'runtime.lock.json').read_text())
    validate_runtime_lock(lock)
    command, _ = checks.launch_command(yaml.safe_load((task/'benchmark.yaml').read_text())['benchmark'], 10001)
    assert command[command.index('--random-seed')+1] == str(lock['server_seed'])
    spec = importlib.util.spec_from_file_location('serving_adapter', ROOT/'src/tools/perf/serving.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    import shutil, sys
    from src.tools.perf import serving_bundle
    shutil.copytree(task/'source', tmp_path/'source')
    monkeypatch.setitem(sys.modules, '_aka_serving_bundle', serving_bundle)
    bundle = serving_bundle.Bundle(tmp_path, lock)
    rows = [dict(rank=rank, function='model_forward', observer=True,
                 bundle_sha256=bundle.digest, server_settings=dict(random_seed=lock['server_seed']))
            for rank in range(lock['gpu_count'])]
    rows.extend(dict(rank=0, function=p['module']+':'+p['symbol'], observer=False,
                     bundle_sha256=bundle.digest) for p in bundle.patches)
    log = tmp_path/'rank_calls.jsonl'
    log.write_text('\n'.join(map(json.dumps, rows)))
    assert len(module.audit(lock)) == 4 + len(bundle.patches)
    rows[2]['server_settings']['random_seed'] += 1
    log.write_text('\n'.join(map(json.dumps, rows)))
    with pytest.raises(ValueError, match='random seed'):
        module.audit(lock)


def test_small_task_runtime_and_materialization(tmp_path, monkeypatch):
    # This checks the local CPU manifest, including when pytest itself runs in
    # an agent container. A live serving socket would route this temporary path
    # into the host executor, which correctly rejects paths outside its mount.
    monkeypatch.delenv('AKA_SERVING_SOCKET', raising=False)
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
    assert manifest.cases[0]['params']['num_requests'] == 2000
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
    assert reserve_from_validation(record,session)['effective_reserve_s'] == 4695
    assert record['deadline_epoch'] == 2000


def test_trusted_templates_cannot_live_in_agent_checkout(tmp_path):
    from src.serving_runtime import CleanExecutor
    with pytest.raises(ValueError, match='outside'):
        CleanExecutor({}, tmp_path, tmp_path/'logs', '0', tmp_path/'cache')


def test_server_hook_does_not_import_framework_in_compiler_subprocess(tmp_path):
    import os, subprocess, sys
    import shutil
    task = tmp_path/'hook-task'
    shutil.copytree(ROOT/'tasks/e2e/qwen3_0_6b_sglang', task)
    from src.perf_helper_materialization import materialize_perf_helpers_in_workspace
    materialize_perf_helpers_in_workspace(task)
    hook = task/'scripts'
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


@pytest.mark.parametrize('container_root', ['/workspace', '/custom/worker root'])
def test_clean_executor_ignores_candidate_import_shadows_and_harness_edits(tmp_path, monkeypatch, container_root):
    import shutil
    from src.serving_runtime import CleanExecutor
    from src.task_spec import load_task_spec

    source = ROOT/'tasks/e2e/qwen3_0_6b_sglang'
    checkout = tmp_path/'checkout'
    candidate = checkout/'run/candidate'
    shutil.copytree(source, candidate)
    shutil.copytree(ROOT/'src/tools/perf', checkout/'src/tools/perf')
    shutil.copy2(ROOT/'src/measurement.py', checkout/'src/measurement.py')
    lock = json.loads((source/'runtime.lock.json').read_text())
    spec = load_task_spec(source/'config.yaml', task_id='e2e/test')
    trusted = tmp_path/'trusted'
    executor = CleanExecutor({'e2e/test': (source, spec, lock)}, checkout, trusted, '2', tmp_path/'cache',
                             container_root=container_root)
    # None of these agent-controlled files may enter the clean action directory.
    extra_paths = ['scripts/yaml.py', 'scripts/usercustomize.py', 'sitecustomize.py']
    (candidate/'source/extra.py').write_text('def helper(x): return x\n')
    (candidate/'source/rmsnorm.py').unlink()
    (candidate/'source/replacement.py').write_text('def kernel(x): return x\n')
    for relative in extra_paths:
        (candidate/relative).write_text("raise RuntimeError('unexpected import')\n")
    (candidate/'scripts/evaluate.py').write_text("print('forged success')\n")
    (candidate/'runtime.lock.json').write_text('{}\n')
    submitted = candidate/'source/replacement.py'
    submitted.write_text(submitted.read_text() + '\n# submitted candidate\n')

    class BeforeContainerLaunch(Exception):
        pass

    def inspect_submission(*args):
        actions = [path for path in trusted.iterdir() if path.name != 'templates']
        assert len(actions) == 1
        action = actions[0]
        assert all(not (action/relative).exists() for relative in extra_paths)
        for relative in ['scripts/evaluate.py', 'runtime.lock.json']:
            assert (action/relative).read_bytes() == (source/relative).read_bytes()
        assert (action/'source/replacement.py').read_bytes() == submitted.read_bytes()
        assert not (action/'source/rmsnorm.py').exists()
        assert (action/'source/extra.py').read_bytes() == (candidate/'source/extra.py').read_bytes()
        raise BeforeContainerLaunch

    monkeypatch.setattr(executor.runtime_verifier, 'verify', inspect_submission)
    with pytest.raises(BeforeContainerLaunch):
        executor.execute(dict(task_id='e2e/test', workspace=container_root+'/run/candidate', role='candidate',
                              action='correctness', phase='candidate_evaluation', timeout=100))


@pytest.mark.parametrize('workspace', ['/workspace/run', '/custom/root-other/run', 'run',
    '/custom/root/../escape', '/custom/root/tasks/task', '/custom/root', '/custom/root/link'])
def test_custom_container_root_keeps_workspace_containment(tmp_path, monkeypatch, workspace):
    from src.serving_runtime import CleanExecutor
    checkout = tmp_path/'checkout'
    (checkout/'tasks/task').mkdir(parents=True)
    outside = tmp_path/'outside'
    outside.mkdir()
    (checkout/'link').symlink_to(outside, target_is_directory=True)
    executor = CleanExecutor({}, checkout, tmp_path/'trusted', '0', tmp_path/'cache',
                             container_root='/custom/root')
    action = SimpleNamespace(commands=[['runner']], timeout_s=100)
    executor.tasks['test'] = (tmp_path, SimpleNamespace(action=lambda *_: action), {})
    monkeypatch.setattr('src.serving_runtime.subprocess.run', lambda *_a, **_k: pytest.fail('must not launch Docker'))
    with pytest.raises(ValueError):
        executor.execute(dict(task_id='test', workspace=workspace, role='candidate', action='correctness',
                              phase='candidate_evaluation', timeout=10))


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


def test_correctness_client_shares_deadline_without_retry(monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location('serving_adapter', ROOT/'src/tools/perf/serving.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    now = [100.0]
    calls = []
    monkeypatch.setattr(module.time, 'monotonic', lambda: now[0])
    def request(port, path, value, *, timeout):
        calls.append((port, path, value, timeout))
        now[0] += value['elapsed']
        if value.get('fail'):
            raise OSError('connection lost')
        return {'answer': 42}
    monkeypatch.setattr(module, 'request', request)
    send = module.deadline_client(2400)
    assert send(1234, 'generate', {'elapsed': 180}) == {'answer': 42}
    assert send(1234, 'generate', {'elapsed': 20}) == {'answer': 42}
    assert [row[3] for row in calls] == [2400, 2220]
    with pytest.raises(OSError, match='connection lost'):
        send(1234, 'generate', {'elapsed': 0, 'fail': True})
    assert len(calls) == 3
    with pytest.raises(TimeoutError, match='deadline'):
        send(1234, 'generate', {'elapsed': 2201})
    with pytest.raises(TimeoutError, match='deadline'):
        send(1234, 'generate', {'elapsed': 0})
    assert len(calls) == 4  # Expired queued work never sends another request.


def test_model_correctness_samples_preserve_prompt_and_decode_path(monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location('serving_adapter', ROOT/'src/tools/perf/serving.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    requests = []
    response = dict(output_ids=[30,40], meta_info=dict(
        input_token_logprobs=[[None,10,None],[-1,20,None]],
        output_token_logprobs=[[-2,30,None],[-3,40,None]]))
    def request(port, endpoint, body):
        requests.append((port, endpoint, body))
        return response
    monkeypatch.setattr(module, 'request', request)
    candidate = module.generation_sample(10001, [10,20], 2)
    reference = module.generation_sample(10002, [10,20], 2)
    assert requests[0][1:] == requests[1][1:]
    assert requests[1][2]['input_ids'] == [10,20]
    assert requests[1][2]['sampling_params']['max_new_tokens'] == 2
    expected = [row[0] for row in reference['actual'][1:]]
    assert module.logprob_error(candidate['tokens'], expected, reference['actual']) == 0
    response['output_ids'] = [31,40]
    response['meta_info']['output_token_logprobs'][0][1] = 31
    divergent = module.generation_sample(10002, [10,20], 2)
    with pytest.raises(ValueError, match='tokens differ'):
        module.logprob_error(candidate['tokens'], expected, divergent['actual'])
    response['output_ids'] = [30]
    with pytest.raises(ValueError, match='decode length'):
        module.generation_sample(10002, [10,20], 2)


@pytest.mark.parametrize('ratio', [1, .8])
def test_model_correctness_covers_scored_batch_lengths(ratio):
    import importlib.util
    spec = importlib.util.spec_from_file_location('serving_adapter', ROOT/'src/tools/perf/serving.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = dict(precision='fp16', envs=dict(CONC=8, TP=1, ISL=128, OSL=128,
                  NUM_PROMPTS=2000, RANDOM_RANGE_RATIO=ratio))
    prompts = [[10,20], [30,40,50]]
    plan = module.model_check_plan(config, {'correctness': {'output_tokens': 16}}, prompts)
    assert plan[0] == dict(kind='smoke', concurrency=1, inputs=prompts, outputs=[16,16])
    workload = module.identity(config)['params']
    batch = plan[1]
    assert batch['concurrency'] == 8 and len(batch['inputs']) == 8
    assert [len(tokens) for tokens in batch['inputs']] == workload.get('input_lens', [128]*8)[:8]
    assert batch['outputs'] == workload.get('output_lens', [128]*8)[:8]


def test_model_correctness_sends_concurrent_requests_and_keeps_order():
    import importlib.util
    import threading
    spec = importlib.util.spec_from_file_location('serving_adapter', ROOT/'src/tools/perf/serving.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    entered = threading.Barrier(3)
    def request(port, endpoint, body):
        # A sequential implementation times out here, even if it labels the
        # diagnostics as concurrent. Variable decode lengths must survive.
        entered.wait(timeout=5)
        tokens = body['input_ids']
        outputs = [30] * body['sampling_params']['max_new_tokens']
        return dict(output_ids=outputs, meta_info=dict(
            input_token_logprobs=[[None,tokens[0],None]] + [[-1,t,None] for t in tokens[1:]],
            output_token_logprobs=[[-2,t,None] for t in outputs]))
    plan = [dict(kind='workload', concurrency=3, inputs=[[10,20],[11,21,22],[12,23]], outputs=[2,3,4])]
    rows = module.generation_checks(10001, plan, request)
    assert [row['input_tokens'] for row in rows] == plan[0]['inputs']
    assert [row['output_length'] for row in rows] == [2,3,4]
    assert all(row['concurrency'] == 3 and row['check_kind'] == 'workload' for row in rows)
    assert [len(row['actual']) for row in rows] == [4,6,6]


def test_no_logprob_mode_checks_every_generated_token_against_independent_argmax():
    import importlib.util
    spec = importlib.util.spec_from_file_location('serving_adapter', ROOT/'src/tools/perf/serving.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    requests = []
    def request(port, endpoint, body):
        requests.append(body)
        # No diagnostic probabilities are returned on the actual scoring mode.
        return dict(output_ids=[30,40])
    plan = [dict(kind='workload', concurrency=1, inputs=[[10,20]], outputs=[2])]
    sample = module.generation_checks(10001, plan, request, return_logprob=False)[0]
    assert requests[0]['return_logprob'] is False
    assert 'logprob_start_len' not in requests[0] and 'actual' not in sample
    assert sample['tokens'] == [10,20,30,40] and sample['return_logprob'] is False
    # Input-token likelihood need not be maximal. Both generated positions must
    # be greedy under the independent reference, allowing only locked near ties.
    reference = dict(values=[-8., -.1, -.22], best=[-.1, -.1, -.2])
    assert module.greedy_reference_error(sample, reference, .1) == pytest.approx(.02)
    for position in (1, 2):
        wrong = copy.deepcopy(reference)
        wrong['values'][position] = -5.
        with pytest.raises(module.ModelNumericalFailure, match='argmax'):
            module.greedy_reference_error(sample, wrong, .1)
    for wrong in (dict(values=[-1.], best=[0.]),
                  dict(values=[-8., -.1, float('nan')], best=[-.1, -.1, -.2])):
        with pytest.raises(module.ModelNumericalFailure):
            module.greedy_reference_error(sample, wrong, .1)


@pytest.mark.parametrize('tokens', [[True, 30], [-1, 30], [30], '30'])
def test_no_logprob_mode_rejects_invalid_or_truncated_outputs(tokens):
    import importlib.util
    spec = importlib.util.spec_from_file_location('serving_adapter', ROOT/'src/tools/perf/serving.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(module.ModelNumericalFailure):
        module.generation_sample(10001, [10,20], 2, lambda *args: dict(output_ids=tokens), return_logprob=False)


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
    calls = [dict(server_settings=dict(port=10000,tp_size=1,_runtime_instance_id='instance-a'),
                  runtime_settings=dict(SGLANG_RUN_ID='run-a',SGLANG_USE_AITER='1'))]
    initial = module.runtime_identity('locked',calls)[0]
    calls[0]['server_settings']['port'] = 20000
    calls[0]['runtime_settings']['SGLANG_RUN_ID'] = 'run-b'
    calls[0]['server_settings']['_runtime_instance_id'] = 'instance-b'
    assert module.runtime_identity('locked',calls)[0] == initial
    calls[0]['runtime_settings']['SGLANG_USE_AITER'] = '0'
    assert module.runtime_identity('locked',calls)[0] != initial
    calls[0]['runtime_settings']['SGLANG_USE_AITER'] = '1'
    calls[0]['server_settings']['tp_size'] = 2
    assert module.runtime_identity('locked',calls)[0] != initial


def test_uncertainty_separates_a_clear_gain_from_noise():
    def rows(rate, role):
        return performance_cases(parse(report(rate, kernel_sha256=role * 64)))
    tight = paired_throughput([(rows(b, 'a'), rows(c, 'b')) for b, c in [(100,110), (100.5,110.5), (99.5,109.5)]])
    spread = tight['uncertainty']['serving']
    assert spread['baseline_throughput_cv'] < 0.01 and spread['candidate_throughput_cv'] < 0.01
    assert spread['median_within_noise'] is False
    assert tight['gain_observed_in_all_pairs'] is True


def test_lock_limits_default_and_reject_unknown_or_nonpositive_values():
    assert lock_limits({}) == LIMIT_DEFAULTS
    assert lock_limits({'limits': {'client_timeout_s': 900}})['client_timeout_s'] == 900
    for bad in ({'bogus': 1}, {'client_timeout_s': 0}, {'idle_vram_limit_mib': 1.5}, {'minimum_measurement_s': True}):
        with pytest.raises(ValueError):
            lock_limits({'limits': bad})
    lock = json.loads((ROOT/'tasks/e2e/qwen3_0_6b_sglang/runtime.lock.json').read_text())
    assert lock['limits']['minimum_measurement_s'] == 30
    lock['limits']['server_startup_timeout_s'] = -5
    with pytest.raises(ValueError, match='positive integer'):
        validate_runtime_lock(lock)


def test_task_lock_matches_the_committed_workload():
    import hashlib
    task = ROOT/'tasks/e2e/qwen3_0_6b_sglang'
    lock = json.loads((task/'runtime.lock.json').read_text())
    assert hashlib.sha256((task/'benchmark.yaml').read_bytes()).hexdigest() == lock['workload_sha256']
    assert hashlib.sha256((task/'source/rmsnorm.py').read_bytes()).hexdigest() == lock['initial_candidate_sha256']
    import yaml
    envs = yaml.safe_load((task/'benchmark.yaml').read_text())['benchmark']['envs']
    # 2000 requests at the qualified ~4100 tokens/s keep the window about twice the 30 s floor.
    assert envs['NUM_PROMPTS'] * envs['OSL'] / 4100 > 2 * lock['limits']['minimum_measurement_s']


def test_serving_task_without_budget_fails_before_workspace_setup(tmp_path, monkeypatch):
    import logging
    from src import task_run

    def refuse(*args, **kwargs):
        raise AssertionError('workspace setup must not start without a budget')

    monkeypatch.setattr(task_run, 'setup_workspace', refuse)
    with pytest.raises(ValueError, match='budget.task_wall_time_s'):
        task_run.run_task_v2(eval_config={'agent': {'template': 'codex'}}, agent=SimpleNamespace(value='codex'),
                             agent_launcher=refuse, task_name='e2e/qwen3_0_6b_sglang',
                             task_config_dir=str(ROOT/'tasks/e2e/qwen3_0_6b_sglang/config.yaml'),
                             run_directory=tmp_path, timestamp='t', logger=logging.getLogger('test'))
    assert not (tmp_path/'.budgets').exists()


def test_exhausted_budget_becomes_a_materialization_timeout(monkeypatch):
    from src.task_materialization import MaterializationTimeout, _Deadline
    monkeypatch.setenv('ARENA_TASK_DEADLINE', '1')
    with pytest.raises(MaterializationTimeout, match='budget exhausted'):
        _Deadline(60)


def test_sweep_outcome_blocks_acceptance(tmp_path, monkeypatch):
    import logging
    from src import task_run
    monkeypatch.delenv('AGENT_KERNEL_ARENA_DEDICATED_CONTAINER', raising=False)
    record = task_run._sweep_before_final_evaluation(
        dict(evaluation=dict(measurement=dict(kind='serving'))), tmp_path, logging.getLogger('test'))
    assert record['status'] == 'unavailable' and record['required'] is True
    assert json.loads((tmp_path/'process_sweep.json').read_text())['status'] == 'unavailable'
    kernel = task_run._sweep_before_final_evaluation(dict(evaluation={}), tmp_path, logging.getLogger('test'))
    assert kernel['required'] is False
    monkeypatch.setenv('AGENT_KERNEL_ARENA_DEDICATED_CONTAINER', '1')
    monkeypatch.setattr('src.process_control.sweep_foreign_processes',
                        lambda **kwargs: dict(status='leftovers', terminated=[dict(pid=9, comm='sleep')], survivors=[9]))
    swept = task_run._sweep_before_final_evaluation(dict(evaluation={}), tmp_path, logging.getLogger('test'))
    assert swept['status'] == 'leftovers' and swept['survivors'] == [9]


def test_variable_request_lengths_reject_same_total_different_work():
    data = report()
    row = data['cases'][0]
    row['params'].pop('output_tokens_per_request')
    row['params'].update(output_lens=[19, 21]*5, input_lens=[9, 11]*5)
    row['metrics'].update(output_lens=[19, 21]*5, input_lens=[9, 11]*5, input_tokens=100)
    assert parse(data).passed
    for field in ('input_lens', 'output_lens'):
        bad = copy.deepcopy(data)
        bad['cases'][0]['metrics'][field].reverse()
        with pytest.raises(TaskProtocolError, match='per-request'):
            parse(bad)
    for values in ([20]*9, [True]+[20]*9, [0]+[20]*9):
        bad = copy.deepcopy(data)
        bad['cases'][0]['params']['output_lens'] = values
        with pytest.raises(TaskProtocolError):
            parse(bad)
    bad = copy.deepcopy(data)
    bad['cases'][0]['params']['output_tokens_per_request'] = 20
    with pytest.raises(TaskProtocolError, match='unambiguous'):
        parse(bad)


def test_large_task_preserves_recipe_and_protects_harness_outside_source_tree(tmp_path, monkeypatch):
    monkeypatch.delenv('AKA_SERVING_SOCKET', raising=False)
    import importlib.util
    import shutil
    import subprocess
    import sys
    from src.task_spec import load_task_spec
    from src.harness_guard import snapshot_workspace_harness, verify_workspace_harness
    task = ROOT/'tasks/e2e/qwen3_next_80b_sglang'
    imported = ROOT/'tasks/e2e/sglang_mi355x/benchmark_sglang_qwen3_next_80b_fp8/benchmark_sglang_qwen3_next_80b_fp8.yaml'
    assert (task/'benchmark.yaml').read_bytes() == imported.read_bytes()
    shutil.copytree(task, tmp_path/'task')
    materialize_perf_helpers_in_workspace(tmp_path/'task')
    run = subprocess.run([sys.executable, 'scripts/evaluate.py', 'validate-task'], cwd=tmp_path/'task', text=True, capture_output=True)
    manifest = parse_command_result(run.stdout,role='task',action='validate-task',returncode=run.returncode,measurement='serving')
    assert manifest.passed, run.stdout
    params = manifest.cases[0]['params']
    assert params['num_requests'] == 320 and params['tp'] == 4
    assert len(set(params['output_lens'])) > 100
    assert all(819 <= n <= 1024 for n in params['output_lens'])
    spec = load_task_spec(tmp_path/'task/config.yaml', task_id='e2e/qwen3_next_80b_sglang')
    snapshot = snapshot_workspace_harness(tmp_path/'task', task_spec=spec)
    source = tmp_path/'task/source/gdn.py'
    source.write_text(source.read_text()+'\n# replacement source is editable\n')
    verify_workspace_harness(snapshot, discard_added=False)
    runner = tmp_path/'task/scripts/checks.py'
    runner.write_text(runner.read_text()+'\n# protected harness mutation\n')
    with pytest.raises(Exception):
        verify_workspace_harness(snapshot, discard_added=False)


def test_shared_model_directory_is_hashed_on_every_verification(tmp_path):
    from src.prepare_serving_runtime import file_digest, verify
    model = tmp_path/'shared-model'
    model.mkdir()
    weight = model/'weight.bin'
    weight.write_bytes(b'original model')
    runtime = tmp_path/'runtime'
    runtime.mkdir()
    (runtime/'model').symlink_to(model, target_is_directory=True)
    identity = {'model': {'revision': 'fixed'}}
    (runtime/'artifact_manifest.json').write_text(json.dumps(dict(identity=identity,files={'model/weight.bin':file_digest(weight)})))
    verify(runtime,identity)
    weight.write_bytes(b'modified model')
    with pytest.raises(ValueError,match='content changed'):
        verify(runtime,identity)


def test_runtime_verification_cache_rejects_restored_mtime(tmp_path,monkeypatch):
    import os
    from src import prepare_serving_runtime as runtime
    (tmp_path/'model').mkdir()
    weight=tmp_path/'model/weight.bin'
    weight.write_bytes(b'initial')
    identity={'model':{'revision':'fixed'}}
    (tmp_path/'artifact_manifest.json').write_text(json.dumps(dict(identity=identity,files={'model/weight.bin':runtime.file_digest(weight)})))
    checker=runtime.VerifiedRuntime()
    checker.verify(tmp_path,identity)
    hashed=[]
    digest=runtime.file_digest
    def observed_digest(path):
        hashed.append(path)
        return digest(path)
    monkeypatch.setattr(runtime,'file_digest',observed_digest)
    checker.verify(tmp_path,identity)
    assert weight in hashed  # Equal metadata must not skip content verification.
    original_fingerprint=checker.fingerprint(tmp_path)
    # Reproduce a coarse ctime tick deterministically, on any test filesystem.
    monkeypatch.setattr(checker,'fingerprint',lambda directory: original_fingerprint)
    stamp=weight.stat()
    weight.write_bytes(b'changed')
    os.utime(weight,ns=(stamp.st_atime_ns,stamp.st_mtime_ns))
    with pytest.raises(ValueError,match='assets changed'):
        checker.verify(tmp_path,identity)


def test_shared_dependencies_are_mounted_at_container_visible_paths(tmp_path):
    from src.serving_runtime import runtime_mounts
    runtime = tmp_path/'runtime'
    runtime.mkdir()
    for name in ('model', 'magpie', 'inferencex'):
        asset = tmp_path/('shared-'+name)
        asset.mkdir()
        (runtime/name).symlink_to(asset, target_is_directory=True)
    mounts = runtime_mounts(runtime, {'dependencies': {'magpie': {}, 'inferencex': {}}})
    assert f'{tmp_path}/shared-model:/task/model:ro' in mounts
    assert f'{tmp_path}/shared-magpie:/task/runtime/magpie:ro' in mounts
    assert f'{tmp_path}/shared-inferencex:/task/runtime/inferencex:ro' in mounts
    assert not any(value.endswith(':/task/runtime:ro') for value in mounts)
    assert all(value.endswith(':ro') for value in mounts if value != '-v')
    with pytest.raises(ValueError, match='mount name'):
        runtime_mounts(runtime, {'dependencies': {'../escape': {}}})



def test_large_task_rejects_serving_dtype_drift():
    import importlib.util
    task = ROOT/'tasks/e2e/qwen3_next_80b_sglang'
    spec = importlib.util.spec_from_file_location('large_task_dtypes', task/'scripts/checks.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = ('torch.bfloat16', 'torch.float32', 'torch.bfloat16')
    module.audit_kernel_dtypes(*expected)
    for position in range(3):
        for wrong in ('torch.float16', 'torch.float8_e4m3fn', None):
            values = list(expected)
            values[position] = wrong
            with pytest.raises(ValueError, match='FP32 recurrent state'):
                module.audit_kernel_dtypes(*values)


def test_gdn_small_outputs_reject_scaling_noise_and_cached_zero():
    import importlib.util
    import torch
    spec = importlib.util.spec_from_file_location('gdn_checks', ROOT/'tasks/e2e/qwen3_next_80b_sglang/scripts/checks.py')
    checks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checks)
    lock = json.loads((ROOT/'tasks/e2e/qwen3_next_80b_sglang/runtime.lock.json').read_text())
    torch.manual_seed(1729)
    expected, _ = checks.oracle(checks.make_case(2, device='cpu'))
    rule = lock['correctness']
    assert checks.compare_tensor(expected, expected, rule)['relative_rms'] == 0
    for actual in (expected * 0.8, expected + 0.015, torch.zeros_like(expected)):
        with pytest.raises((AssertionError, ValueError)):
            checks.compare_tensor(actual, expected, rule)
    # Elementwise absolute tolerance alone still misses small systematic bias.
    small = torch.full((128,), 0.0002)
    biased = small * 0.8
    torch.testing.assert_close(biased, small, rtol=rule['rtol'], atol=rule['atol'])
    with pytest.raises(ValueError, match='relative RMS'):
        checks.compare_tensor(biased, small, rule)
    specs = checks.case_specs()
    assert {1012, 1016} <= {x.get('pool_size') for x in specs}
    assert any(x['batch'] == 512 and x.get('active') == 32 for x in specs)
    assert any(x.get('normalize') is False for x in specs)


def test_model_quality_gate_checks_answers_coverage_and_regression(tmp_path, monkeypatch):
    import importlib.util
    import copy
    task = ROOT/'tasks/e2e/qwen3_next_80b_sglang'
    spec = importlib.util.spec_from_file_location('large_task_quality', task/'scripts/quality.py')
    quality = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(quality)
    monkeypatch.setattr(quality, 'ROOT', tmp_path)
    (tmp_path/'quality.json').write_bytes((task/'quality.json').read_bytes())
    lock = json.loads((task/'runtime.lock.json').read_text())
    cases = quality.load_cases(lock)['samples']
    expected = {row['question']: row['answer'] for row in cases}
    def request(port, path, value):
        assert path == 'v1/chat/completions'
        assert value['max_tokens'] == 8192 and value['temperature'] == 0
        assert 'chat_template_kwargs' not in value  # Preserve default reasoning.
        question = value['messages'][0]['content'].rsplit('Question: ', 1)[-1].split('\nEnd your response', 1)[0]
        return dict(choices=[dict(message=dict(content='</think>\n'+expected[question]), finish_reason='stop')],
                    usage=dict(completion_tokens=100))
    report = quality.run(1234, lock, request, 'candidate')
    assert quality.validate(report, report, lock)['candidate_accuracy'] == 1
    assert len(json.loads((tmp_path/'candidate_accuracy_report.json').read_text())['results']) == 64
    bad = copy.deepcopy(report)
    bad['results'][0]['response'] = 'The answer is 42'  # Missing strict final-answer marker.
    bad['results'][0].update(actual=None, correct=False)
    bad['score'] = 63/64
    assert quality.validate(bad, report, lock)['candidate_accuracy'] == 63/64
    bad['results'][1]['response'] = 'No final answer'
    bad['results'][1].update(actual=None, correct=False)
    bad['score'] = 62/64
    assert quality.validate(bad, report, lock)['candidate_accuracy'] == 62/64
    for row in bad['results'][2:4]:
        row.update(response='No final answer', actual=None, correct=False)
    bad['score'] = 60/64
    with pytest.raises(ValueError, match='regressed'):
        quality.validate(bad, report, lock)
    bad['score'] = 1
    with pytest.raises(ValueError, match='answer evidence'):
        quality.validate(bad, report, lock)
    bad = copy.deepcopy(report)
    bad['results'][0]['id'] = -1
    with pytest.raises(ValueError, match='identities'):
        quality.validate(bad, report, lock)
    bad = copy.deepcopy(report)
    bad['results'][0]['expected'] = 'fabricated'
    with pytest.raises(ValueError, match='answer evidence'):
        quality.validate(bad, report, lock)
    bad = copy.deepcopy(report)
    bad['results'][0]['finish_reason'] = 'length'
    with pytest.raises(ValueError, match='answer evidence'):
        quality.validate(bad, report, lock)
    bad['results'][0].update(actual=None, correct=False)
    bad['score'] = 63/64
    assert quality.validate(bad, report, lock)['candidate_accuracy'] == 63/64
    (tmp_path/'quality.json').write_text('{}')
    with pytest.raises(ValueError, match='differs from lock'):
        quality.load_cases(lock)
    assert quality.answer('Reasoning 123. #### 1,234.50') == '1234.5'
    assert quality.answer('#### 2\nCorrection: #### -3') == '-3'
    assert quality.answer('#### NaN') is None
    assert quality.answer('<think>Draft: #### 42') is None
    assert quality.answer('<think>Draft: #### 42</think>\nNo final answer') is None
    assert quality.answer('<think>Draft: #### 42</think>\n#### 7') == '7'
    assert quality.answer('Draft: #### 42\nStill reasoning') is None
    assert quality.completion_answer('Draft: #### 42', 'stop', None) == '42'  # Instruct does not require a thinking tag.
    assert quality.completion_answer('</think>\n#### 42', 'length', None) is None
    assert quality.completion_answer('#### 42', 'stop', 'separated reasoning') == '42'


@pytest.mark.parametrize("value", [0, 1, 2, 101, True, 1.5, "1", None])
def test_invalid_run_pair_count(value):
    with pytest.raises(ValueError, match="final_evaluation_pairs"):
        budget_policy({"budget": {"task_wall_time_s": 86400,
            "final_evaluation_reserve_s": 7200, "final_evaluation_pairs": value}})


def test_pair_policy_persists_and_reserve_uses_actual_count(tmp_path, monkeypatch):
    monkeypatch.setattr("src.run_budget.time.time", lambda: 1000)
    policy = budget_policy({"budget": {"task_wall_time_s": 86400,
        "final_evaluation_reserve_s": 1200, "final_evaluation_pairs": 5}})
    record = open_budget(tmp_path/"budget.json", policy)
    with pytest.raises(ValueError, match="resume"):
        open_budget(tmp_path/"budget.json", {**policy, "final_evaluation_pairs": 3})
    (tmp_path/"runtime.lock.json").write_text(json.dumps({"minimum_final_evaluation_s": 600}))
    spec = SimpleNamespace(to_mapping=lambda: {"evaluation": {"measurement": {
        "runtime_lock": "runtime.lock.json", "pairs": 3}}})
    results = {("task_validation", "baseline", action): SimpleNamespace(
        commands=[SimpleNamespace(elapsed_s=seconds)])
        for action, seconds in [("compile", 30), ("correctness", 100), ("performance", 500)]}
    session = SimpleNamespace(spec=spec, workspace=tmp_path, results=results)
    assert reserve_from_validation(record, session)["effective_reserve_s"] == 7695
    assert spec.to_mapping()["evaluation"]["measurement"]["pairs"] == 3


def test_single_pair_has_no_noise_estimate():
    baseline = performance_cases(parse(report()))
    candidate = performance_cases(parse(report(101, kernel_sha256="b"*64)))
    result = paired_throughput([(baseline, candidate)])
    assert result["speedup_ratio"] == pytest.approx(1.01)
    spread = result["uncertainty"]["serving"]
    assert spread["status"] == "not_estimable_from_one_pair"
    assert spread["baseline_throughput_cv"] is None
    assert spread["candidate_throughput_cv"] is None
    assert spread["median_within_noise"] is None


@pytest.mark.parametrize('declared,override', [(3, 1), (5, 3)])
def test_run_cannot_reduce_task_measurement_count(declared, override):
    from src.measurement import serving_pair_count
    config = {'evaluation': {'measurement': {'pairs': declared}}}
    with pytest.raises(ValueError, match='final_evaluation_pairs'):
        serving_pair_count(config, override)


@pytest.mark.parametrize('task', ['qwen3_0_6b_sglang', 'qwen3_next_80b_sglang'])
def test_task_policy_uses_the_backend_neutral_bundle(tmp_path, monkeypatch, task):
    import importlib.util
    import shutil
    import sys
    from src.tools.perf import serving_bundle
    source = ROOT / 'tasks/e2e' / task
    shutil.copytree(source, tmp_path / 'task')
    spec = importlib.util.spec_from_file_location('task_checks', source / 'scripts/checks.py')
    checks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checks)
    checks.ROOT = tmp_path / 'task'
    monkeypatch.setitem(sys.modules, '_aka_serving_bundle', serving_bundle)
    checks.source_policy()
    candidate = next((checks.ROOT / 'source').glob('*.py'))
    candidate.write_text(candidate.read_text() + '\nimport os\n')
    with pytest.raises(ValueError):
        checks.source_policy()


@pytest.mark.parametrize("override,count", [(None, 3), (5, 5)])
def test_evaluator_runs_requested_complete_pairs_after_correctness(tmp_path, monkeypatch, override, count):
    from src.evaluator import evaluate_kernel
    calls = []
    def action(role, kind):
        calls.append((role, kind))
        if kind == "performance":
            result = parse(report(100 if role == "baseline" else 101,
                kernel_sha256=("a" if role == "baseline" else "b")*64))
        else:
            result = SimpleNamespace(passed=True, reason=None)
        return SimpleNamespace(result=result)
    session = SimpleNamespace(candidate_action=lambda kind: action("candidate", kind),
        baseline_action=lambda kind: action("baseline", kind))
    config = {"evaluation": {"measurement": {"kind": "serving", "pairs": 3}}}
    result = evaluate_kernel(tmp_path, config, [], task_session=session,
        final_evaluation_pairs=override)
    assert calls[:2] == [("candidate", "compile"), ("candidate", "correctness")]
    assert len([x for x in calls if x[1] == "performance"]) == 2*count
    assert result["serving_measurement"]["pairs"] == count
    assert result["serving_measurement"]["declared_pairs"] == 3
    assert result["serving_measurement"]["pair_count_override"] == override


@pytest.mark.parametrize('prepared',[False,True])
@pytest.mark.parametrize('category',['model_quality','model_numerical'])
def test_model_failure_keeps_manifest_identity_and_category(tmp_path, monkeypatch, capsys, prepared, category):
    import sys
    import shutil
    from src.tools.perf import serving
    from src import measurement
    from src.task_protocol import parse_command_result
    task=tmp_path/'task'
    shutil.copytree(ROOT/'tasks/e2e/qwen3_0_6b_sglang',task)
    monkeypatch.setattr(serving,'ROOT',task)
    monkeypatch.setitem(sys.modules,'_aka_measurement',measurement)
    monkeypatch.delenv('AKA_SERVING_SOCKET',raising=False)
    monkeypatch.setattr(sys,'argv',['evaluate.py','candidate','correctness'])
    def fail(*args):
        if prepared:
            row=serving.identity(args[1])
            row['params']['request_plan_sha256']='a'*64
            args[0].case_identity=row
        error=serving.ModelQualityFailure if category=='model_quality' else serving.ModelNumericalFailure
        raise error('Candidate model failed: accuracy regression')
    monkeypatch.setattr(serving,'evaluate',fail)
    assert serving.main()==1
    parsed=parse_command_result(capsys.readouterr().out,role='candidate',action='correctness',returncode=1,measurement='serving')
    assert not parsed.passed
    assert len(parsed.cases)==1 and parsed.cases[0]['status']=='FAIL'
    assert parsed.metadata['failure_category']==category
    if prepared: assert parsed.cases[0]['params']['request_plan_sha256']=='a'*64
    assert 'accuracy regression' in parsed.reason


def test_bundle_patch_count_does_not_change_runtime_identity():
    from src.tools.perf.serving import runtime_identity
    observers=[dict(rank=i, observer=True,server_settings={'tp_size':2,'port':200+i},runtime_settings={'ROCR_VISIBLE_DEVICES':'0,1'}) for i in range(2)]
    patches=[dict(rank=0,observer=False,function='a'),dict(rank=1,observer=False,function='b')]
    assert runtime_identity('lock',observers)==runtime_identity('lock',observers+patches)


@pytest.mark.parametrize('rate,scored', [(101.23,1.0), (104.99,1.0), (105,1.05), (112,1.12), (99,0.99)])
def test_serving_five_percent_gain_threshold_keeps_all_samples(rate, scored):
    base = performance_cases(parse(report()))
    candidate = performance_cases(parse(report(rate, kernel_sha256='b' * 64)))
    result = paired_throughput([(base, candidate)]*30, minimum_speedup_ratio=1.05)
    assert result['speedup_ratio'] == pytest.approx(scored)
    assert result['raw_speedup_ratio'] == pytest.approx(rate/100)
    assert len(result['paired_ratios']['serving']) == 30
    assert result['significance'] == 'not_assessed'


def test_hundred_pairs_allowed_and_invalid_threshold_rejected():
    from src.measurement import validate_pair_count
    assert validate_pair_count(100) == 100
    for value in [True, 0.9, float('nan'), float('inf')]:
        with pytest.raises(ValueError):
            paired_throughput([], minimum_speedup_ratio=value)


def production_fixture():
    import importlib.util
    spec=importlib.util.spec_from_file_location('production_serving_check',ROOT/'src/tools/perf/serving.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    lock={'correctness':{'production_tolerances':dict(prefix_max_abs=.5,prefix_mean_abs=.2,
        prefix_p99_abs=.5,decode_max_regret=.5,decode_mean_regret=.2,decode_p99_regret=.5)}}
    diagnostic=dict(input_tokens=[10,20,30],actual=[[None,10],[-1.1,20],[-2.1,30]],
        reference_rows=[[None,10],[-1.,20],[-2.,30]])
    plain=dict(input_tokens=[10,20,30],tokens=[10,20,30,40,50],output_length=2,
        reference=dict(values=[-1.,-2.,-.4,-.3],best=[-1.,-2.,-.3,-.2]))
    return module,lock,diagnostic,plain


def test_production_normal_variation_retains_per_case_bounds():
    module,lock,diagnostic,plain=production_fixture()
    result=module.production_numerical_checks([diagnostic],[plain],lock)
    assert result['prefix_cases'][0]['mean']==pytest.approx(.1)
    assert result['decode_cases'][0]['max']==pytest.approx(.1)
    assert result['thresholds']==lock['correctness']['production_tolerances']


@pytest.mark.parametrize('corruption',['prefix','decode','nonfinite','missing_token','missing_reference','misaligned'])
def test_production_wrong_computation_or_missing_coverage_fails(corruption):
    module,lock,diagnostic,plain=production_fixture()
    if corruption=='prefix': diagnostic['actual'][1][0]=-2
    if corruption=='decode': plain['reference']['values'][-1]=-4
    if corruption=='nonfinite': plain['reference']['best'][-1]=float('nan')
    if corruption=='missing_token': plain['tokens'].pop()
    if corruption=='missing_reference': plain['reference']['best'].pop()
    if corruption=='misaligned': diagnostic['reference_rows'][1][1]=21
    with pytest.raises(ValueError): module.production_numerical_checks([diagnostic],[plain],lock)


def test_production_errors_cannot_be_diluted_by_other_cases():
    import copy
    module,lock,diagnostic,plain=production_fixture()
    diagnostics=[copy.deepcopy(diagnostic) for _ in range(100)]
    plains=[copy.deepcopy(plain) for _ in range(100)]
    plains[-1]['reference']['values'][-1]=-2
    with pytest.raises(ValueError,match='decode case 99'):
        module.production_numerical_checks(diagnostics,plains,lock)
    for bad in ({}, {'prefix_max_abs':1}, {**lock['correctness']['production_tolerances'],'decode_max_regret':True}):
        with pytest.raises(ValueError,match='tolerances'):
            module.production_tolerances({'correctness':{'production_tolerances':bad}})


def test_production_teacher_reference_scores_actual_continuation():
    module,lock,diagnostic,plain=production_fixture()
    def request(port,path,body):
        assert body['input_ids']==plain['tokens'] and body['top_logprobs_num']==1
        assert body['sampling_params']==dict(temperature=0,max_new_tokens=1,ignore_eos=True)
        return {'meta_info':dict(input_token_logprobs=[[None,10],[-1,20],[-2,30],[-.4,40],[-.3,50]],
            input_top_logprobs=[None,[[-1,20]],[[-2,30]],[[-.3,41]],[[-.2,51]]])}
    assert module.production_teacher_reference(1,plain,request)==plain['reference']


def test_vllm_derived_capacity_is_retained_but_not_a_configured_override():
    from src.tools.perf.serving import runtime_identity
    calls=[dict(backend='vllm',server_settings=dict(model_config={'dtype':'float16'},
        cache_config={'num_gpu_blocks':100,'num_cpu_blocks':0,'num_gpu_blocks_override':None,
                      'block_size':16,'cache_dtype':'fp8'}),runtime_settings={})]
    expected=runtime_identity('lock',calls)[0]
    calls[0]['server_settings']['cache_config']['num_gpu_blocks']=101
    assert runtime_identity('lock',calls)[0]==expected
    assert calls[0]['server_settings']['cache_config']['num_gpu_blocks']==101
    for name,value in [('num_gpu_blocks_override',102),('block_size',32),('cache_dtype','float16')]:
        changed=copy.deepcopy(calls)
        changed[0]['server_settings']['cache_config'][name]=value
        assert runtime_identity('lock',changed)[0]!=expected


def test_vllm_compilation_observations_are_retained_without_treating_sources_as_flags():
    from src.tools.perf.serving import runtime_identity
    config=dict(mode=3,custom_ops=['all'],pass_config={'fuse_allreduce_rms':False},
        traced_files=['original.py'],compilation_time=10.0,encoder_compilation_time=0.0,
        enabled_custom_ops={'rms_norm':24},disabled_custom_ops={})
    calls=[dict(backend='vllm',server_settings=dict(compilation_config=config),runtime_settings={})]
    fingerprint=runtime_identity('lock',calls)[0]
    config.update(traced_files=['original.py','new_fusion.py'],compilation_time=20.0,
                  enabled_custom_ops={'rms_norm':12})
    assert runtime_identity('lock',calls)[0]==fingerprint
    assert config['traced_files']==['original.py','new_fusion.py']
    for key,value in [('mode',2),('custom_ops',[]),('pass_config',{'fuse_allreduce_rms':True})]:
        changed=copy.deepcopy(calls);changed[0]['server_settings']['compilation_config'][key]=value
        assert runtime_identity('lock',changed)[0]!=fingerprint


def test_vllm_auto_cache_source_key_keeps_configured_input_root_and_flags():
    from src.tools.perf.serving import runtime_identity
    cache='/tmp/.cache/vllm/torch_compile_cache/'
    calls=[dict(backend='vllm',observer=True,configuration_inputs={'compilation_cache_dir':''},
        server_settings=dict(model_config={'dtype':'float16'},compilation_config=dict(
            cache_dir=cache+'aaaaaaaaaa',mode=3,backend='inductor',cudagraph_mode=[2,1],custom_ops=['none'])),
        runtime_settings={})]
    expected=runtime_identity('lock',calls)[0]
    calls[0]['server_settings']['compilation_config']['cache_dir']=cache+'bbbbbbbbbb'
    assert runtime_identity('lock',calls)[0]==expected
    assert calls[0]['server_settings']['compilation_config']['cache_dir']==cache+'bbbbbbbbbb'
    for key,value in [('mode',0),('backend','eager'),('cudagraph_mode',[]),('custom_ops',['all']),
                      ('cache_dir','/other/torch_compile_cache/bbbbbbbbbb')]:
        changed=copy.deepcopy(calls);changed[0]['server_settings']['compilation_config'][key]=value
        assert runtime_identity('lock',changed)[0]!=expected
    changed=copy.deepcopy(calls);changed[0]['configuration_inputs']['compilation_cache_dir']=cache+'bbbbbbbbbb'
    assert runtime_identity('lock',changed)[0]!=expected
    changed=copy.deepcopy(calls);changed[0]['server_settings']['model_config']['dtype']='bfloat16'
    assert runtime_identity('lock',changed)[0]!=expected


@pytest.mark.parametrize('configured', [None,'/explicit/cache'])
def test_vllm_cache_without_auto_provenance_or_with_override_stays_exact(configured):
    from src.tools.perf.serving import runtime_identity
    calls=[dict(backend='vllm',observer=True,server_settings=dict(compilation_config=dict(
        cache_dir='/tmp/.cache/vllm/torch_compile_cache/aaaaaaaaaa')),runtime_settings={})]
    if configured is not None:
        calls[0]['configuration_inputs']=dict(compilation_cache_dir=configured)
    expected=runtime_identity('lock',calls)[0]
    calls[0]['server_settings']['compilation_config']['cache_dir']='/tmp/.cache/vllm/torch_compile_cache/bbbbbbbbbb'
    assert runtime_identity('lock',calls)[0]!=expected


def test_native_fp32_reference_accepts_near_ties_and_retains_wrong_token_evidence(serving_adapter, tmp_path, monkeypatch):
    module=serving_adapter
    monkeypatch.setattr(module,'ROOT',tmp_path)
    samples=[dict(tokens=[1,2,3],input_tokens=[1],output_length=2,return_logprob=False),
             dict(tokens=[1,4,5],input_tokens=[1],output_length=2,return_logprob=False)]
    references=[dict(values=[-.02,-.03],best=[0.,0.]),dict(values=[-.01,-.04],best=[0.,0.])]
    calls=[]
    def reference(sequences,*,with_best):
        calls.append((sequences,with_best))
        return copy.deepcopy(references)
    monkeypatch.setattr(module,'model_reference',reference)
    lock={'correctness':{'model_logprob_atol':.1}}
    result=module.fp32_token_checks(copy.deepcopy(samples),lock)
    assert result['generated_tokens_checked']==4 and result['max_greedy_regret']==pytest.approx(.04)
    assert result['reference_dtype']=='float32' and result['logprob_checks'] is False
    assert calls==[([[1,2,3],[1,4,5]],True)]
    references[1]['values'][1]=-5.
    with pytest.raises(module.ModelNumericalFailure,match='reference argmax'):
        module.fp32_token_checks(copy.deepcopy(samples),lock)
    evidence=json.loads((tmp_path/'no_logprob_correctness.json').read_text())
    assert evidence[1]['tokens']==[1,4,5] and evidence[1]['reference']['values'][1]==-5.
    references.pop()
    with pytest.raises(module.ModelNumericalFailure,match='incomplete request coverage'):
        module.fp32_token_checks(copy.deepcopy(samples),lock)
    with pytest.raises(module.ModelNumericalFailure,match='actual no-logprob samples'):
        module.fp32_token_checks([{**samples[0],'return_logprob':True}],lock)


def test_atom_runtime_identity_keeps_compute_settings_and_raw_transport_evidence():
    from src.tools.perf.serving import runtime_identity
    config=dict(compilation_config=dict(compilation_time=1,cache_dir='/tmp/.cache/atom/torch_compile_cache/aaaaaaaaaa',
        mode=3,custom_ops=['all']),parallel_config=dict(tensor_parallel_size=4,pipeline_parallel_size=1,
        data_parallel_size=1,data_parallel_master_port=29500,enable_expert_parallel=False,
        control_address='ipc:///tmp/one',data_parallel_base_port=12345))
    calls=[dict(backend='atom',server_settings=config,runtime_settings={},configuration_inputs={'compilation_cache_dir':''})]
    fingerprint=runtime_identity('lock',calls)[0]
    config['compilation_config'].update(compilation_time=2,cache_dir='/tmp/.cache/atom/torch_compile_cache/bbbbbbbbbb')
    config['parallel_config'].update(control_address='ipc:///tmp/two',data_parallel_base_port=54321)
    assert runtime_identity('lock',calls)[0]==fingerprint
    assert config['parallel_config']['control_address']=='ipc:///tmp/two'
    for key,value in [('tensor_parallel_size',8),('pipeline_parallel_size',2),('data_parallel_size',2),
                      ('enable_expert_parallel',True),('data_parallel_master_port',30000)]:
        changed=copy.deepcopy(calls);changed[0]['server_settings']['parallel_config'][key]=value
        assert runtime_identity('lock',changed)[0]!=fingerprint
    for key,value in [('mode',0),('custom_ops',[]),('cache_dir','/explicit/cache')]:
        changed=copy.deepcopy(calls);changed[0]['server_settings']['compilation_config'][key]=value
        assert runtime_identity('lock',changed)[0]!=fingerprint
    changed=copy.deepcopy(calls);changed[0]['runtime_settings']['ATOM_DP_BASE_PORT']='12345'
    assert runtime_identity('lock',changed)[0]!=fingerprint


def test_initial_catalog_manifest_already_binds_realized_native_requests(serving_adapter,tmp_path,monkeypatch):
    import sys
    module=serving_adapter
    monkeypatch.setattr(module,'ROOT',tmp_path)
    workload=tmp_path/'benchmark.yaml';workload.write_text('immutable fixture')
    calls=[]
    def workload_identity(config,lock,limits,row):
        calls.append('prepared')
        row['params'].update(request_plan_sha256='a'*64,input_lens=[129])
        row['params'].pop('input_tokens',None)
        return row
    monkeypatch.setitem(sys.modules,'checks',SimpleNamespace(source_policy=lambda:None,workload_identity=workload_identity))
    bundle=SimpleNamespace(digest='empty',check_sources=lambda:{})
    monkeypatch.setitem(sys.modules,'_aka_serving_bundle',SimpleNamespace(Bundle=lambda *args:bundle))
    config=dict(precision='bf16',envs=dict(NUM_PROMPTS=1,CONC=1,TP=1,ISL=128,OSL=128))
    lock=dict(catalog_version=1,scenario='text',workload_sha256=module.digest(workload),initial_bundle_sha256='empty')
    args=SimpleNamespace(action='validate-task')
    rows,_=module.evaluate(args,config,lock,{})
    assert calls==['prepared']
    assert rows[0]['params']['request_plan_sha256']=='a'*64
    assert rows[0]['params']['input_lens']==[129] and 'input_tokens' not in rows[0]['params']
    rows[0]['params']['input_lens'][0]=1
    assert args.case_identity['params']['input_lens']==[129]
