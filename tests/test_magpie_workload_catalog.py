"""Catalog/protocol tests use synthetic bindings, never imply GPU qualification."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest
import yaml

from src import magpie_catalog as catalog
from src.measurement import validate_serving_case
from src.task_spec import load_task_spec
from src.tools.perf import serving_backends as backends, serving_scenarios as scenarios, serving_bundle

ROOT = Path(__file__).resolve().parents[1]
CATALOG = catalog.load_catalog()


def binding(entry):
    info = catalog.describe(entry)
    recipe = yaml.safe_load(entry['source'])['benchmark']
    value = dict(image='registry.invalid/test@sha256:' + 'a'*64,
        gpu_count=recipe.get('envs', {}).get('TP', 1), gpu_arch='gfx950',
        model=dict(id=recipe['model'], revision='1'*40),
        model_architectures=['FixtureModel'], server_topology={'TP': 1},
        correctness=dict(model_reference='production_numerical', model_samples=2, output_tokens=2,
            production_tolerances=dict(prefix_max_abs=.1, prefix_mean_abs=.1, prefix_p99_abs=.1,
                decode_max_regret=.1, decode_mean_regret=.1, decode_p99_regret=.1)),
        dependencies=dict(magpie=dict(repository=CATALOG['repository'], revision=CATALOG['revision']),
            inferencex=dict(repository='https://example.invalid/source', revision='2'*40, subdirectory='inferencex-e2e')),
        benchmark_scripts={'benchmarks/sglang_mi355x.sh': '3'*64},
        client_runtime=dict(kind='image',executable='/usr/bin/python3'),
        inputs={'correctness.json': ['A fresh test input.', 'A different test input.']})
    if recipe['model'].startswith('/'):
        value['source_model_alias'] = recipe['model']
        value['model']['id'] = 'fixture/model'
    script = recipe.get('benchmark_script', '')
    if script and not script.startswith(('sglang_', 'vllm_', 'atom_')):
        value['server_launch'] = dict(argv=['python3', '-m', 'fixture', '--port', '{port}'], env={}, health_path='/health')
    if info['scenario'] == 'agentx':
        value['request_manifest'] = dict(dataset_sha256='4'*64, duration_s=120,
            requests=[dict(id='trace:0', input_tokens=8, output_tokens=2)])
        value['assets'] = {'traces.jsonl': dict(url='https://example.invalid/traces', sha256='4'*64)}
        value['agentx'] = dict(recipe_fingerprint='5'*64,
            launch_spec=dict(argv=['python3', '-m', 'fixture', '--port', '{port}'], env={}, health_path='/health'),
            client=dict(argv=['aiperf', 'profile', '--scenario', 'inferencex-agentx-mvp', '--benchmark-duration', '120',
                '--concurrency', str(recipe['envs']['CONC']), '--url', 'http://127.0.0.1:{port}'], env={}))
        value['dependencies']['aiperf'] = dict(repository='https://example.invalid/aiperf', revision='6'*40)
        value['correctness']['trace_probe_indices'] = [0]
    if info['scenario'] == 'multimodal':
        value['correctness']['model_reference'] = 'production_serving'
        value['correctness']['model_logprob_atol'] = .01
        count=int(recipe['envs'].get('NUM_PROMPTS',int(recipe['envs']['CONC'])*10))
        value['request_manifest'] = dict(dataset_sha256='4'*64,
            input_lens=[8+(i%2) for i in range(count)],output_lens=[int(recipe['envs']['OSL'])]*count)
        cases = [dict(messages=[dict(role='user', content=[dict(type='image_url', image_url={'url':'data:image/png;base64,fixture'}),
            dict(type='text', text='Describe this image.')])], output_tokens=2)]*2
        value['inputs']['media_checks.json'] = cases
        value['correctness']['media_sha256'] = catalog.sha256((json.dumps(cases, indent=2, ensure_ascii=False) + '\n').encode())
    return value


def test_catalog_exact_current_snapshot_and_all_scenarios():
    assert len(CATALOG['entries']) >= 50
    assert {catalog.describe(e)['scenario'] for e in CATALOG['entries']} == {'text','agentx','multimodal'}
    assert {catalog.describe(e)['framework'] for e in CATALOG['entries']} == {'sglang','vllm','atom'}
    for entry in CATALOG['entries']:
        assert catalog.sha256(entry['source'].encode()) == entry['sha256']


@pytest.mark.parametrize('entry', CATALOG['entries'], ids=lambda entry: entry['id'])
def test_every_recipe_materializes_standalone_contract(tmp_path, entry):
    profile = binding(entry)
    dest = catalog.materialize(CATALOG, entry['id'], profile, tmp_path/'task')
    spec = load_task_spec(dest/'config.yaml', task_id='e2e/magpie/'+entry['id'])
    assert spec.candidate.language == 'gpu'
    assert spec.candidate.editable[0].scope == 'tree'
    lock = json.loads((dest/'runtime.lock.json').read_text())
    bundle = serving_bundle.Bundle(dest, lock)
    assert bundle.patches == [] and bundle.digest == lock['initial_bundle_sha256']
    assert bundle.check_sources()['replacements'] == []
    assert (dest/'upstream.yaml').read_text() == entry['source']
    normalized = yaml.safe_load((dest/'benchmark.yaml').read_text())['benchmark']
    original = yaml.safe_load(entry['source'])['benchmark']
    for key, val in original.get('envs', {}).items():
        if key not in {'BENCHMARK_BASE_URL','PORT','ENABLE_PROFILIER','ENABLE_PROFILE','PROFILE','MODEL_PATH'}:
            assert normalized['envs'][key] == val
    assert lock['candidate_bundle']['observer']['classes'] == ['FixtureModel']
    assert lock['candidate_bundle']['scope'] == 'whole_model'
    assert len(lock['candidate_bundle']['module_prefixes']) >= 3
    assert (dest/'scripts/_aka_serving_scenarios.py').is_file()
    assert yaml.safe_load((dest/'config.yaml').read_text())['evaluation']['measurement']['pairs'] == 3
    # Text manifests now bind native realized requests and require the pinned
    # model tokenizer/client even for initialization. Synthetic CPU bindings
    # must fail closed instead of pretending to qualify that runtime.
    env = {k:v for k,v in os.environ.items() if k != 'AKA_SERVING_SOCKET'}
    completed = subprocess.run([sys.executable, 'scripts/evaluate.py', 'validate-task'],
        cwd=dest, env=env, capture_output=True, text=True, timeout=30)
    if catalog.describe(entry)['scenario'] == 'text':
        assert completed.returncode != 0
        assert 'runtime/inferencex' in completed.stdout + completed.stderr
    else:
        from src.task_protocol import parse_command_result
        result = parse_command_result(completed.stdout, role='task', action='validate-task',
            returncode=completed.returncode, measurement='serving')
        assert result.passed, completed.stdout + completed.stderr
        assert result.cases[0]['params']['gpu_count'] == profile['gpu_count']
    with pytest.raises(FileExistsError):
        catalog.materialize(CATALOG, entry['id'], profile, dest)


@pytest.mark.parametrize('mutate', [
    lambda v: v.update(image='moving:latest'),
    lambda v: v['model'].update(revision='main'),
    lambda v: v.update(model_architectures=[]),
    lambda v: v.update(benchmark_scripts={}),
    lambda v: v['correctness'].update(model_reference='answers_only'),
])
def test_incomplete_binding_cannot_materialize(tmp_path, mutate):
    entry = CATALOG['entries'][-1]
    value = binding(entry); mutate(value)
    with pytest.raises(ValueError): catalog.materialize(CATALOG, entry['id'], value, tmp_path/'task')
    assert not (tmp_path/'task').exists()


def test_model_cannot_change_during_binding():
    entry = CATALOG['entries'][-1]; value = binding(entry)
    value['model']['id'] = 'other/model'
    with pytest.raises(ValueError, match='Runtime model'):
        catalog.normalized_recipe(entry, value)


def test_short_diagnostic_workload_can_increase_measurement_samples_before_search(tmp_path):
    entry=next(e for e in CATALOG['entries'] if e['id']=='benchmark_vllm_r9700_qwen35_9b_mm_tracelens')
    profile=binding(entry)
    original=yaml.safe_load(entry['source'])['benchmark']['envs']
    profile['measurement_num_prompts']=2000
    normalized=catalog.normalized_recipe(entry,profile)['envs']
    assert normalized['NUM_PROMPTS']==2000
    assert all(normalized[k]==v for k,v in original.items() if k not in {'NUM_PROMPTS','PROFILE','ENABLE_PROFILE','ENABLE_PROFILIER'})
    with pytest.raises(ValueError,match='complete prepared sample count'):
        catalog.materialize(CATALOG,entry['id'],profile,tmp_path/'incomplete')
    profile['request_manifest']['input_lens']=[8]*2000
    profile['request_manifest']['output_lens']=[int(original['OSL'])]*2000
    task=catalog.materialize(CATALOG,entry['id'],profile,tmp_path/'task')
    assert json.loads((task/'provenance.json').read_text())['measurement_samples']==dict(
        original=2,prepared=2000,fixed_before_optimization=True)
    profile['measurement_num_prompts']=1
    with pytest.raises(ValueError,match='never decrease'):catalog.normalized_recipe(entry,profile)
    trace=next(e for e in CATALOG['entries'] if catalog.describe(e)['scenario']=='agentx')
    with pytest.raises(ValueError,match='fixed duration'):catalog.normalized_recipe(trace,{**binding(trace),'measurement_num_prompts':2000})


def test_new_and_unknown_upstream_contract_fails_visibly():
    entry = copy.deepcopy(CATALOG['entries'][0])
    entry['source'] = yaml.safe_dump({'benchmark':dict(framework='unknown', model='fixture', precision='fp8')})
    with pytest.raises(ValueError, match='No serving backend'): catalog.describe(entry)


def trace_record(index=0, *, error=None):
    return dict(metadata=dict(conversation_id='trace::session', turn_index=0, request_id=str(index),
        request_start_ns=index*10**9, request_end_ns=(index+120)*10**9, benchmark_phase='profiling',
        was_cancelled=False, cancellation_time_ns=None),
        metrics=dict(input_sequence_length={'value':8},output_sequence_length={'value':2}), error=error)


def test_trace_duration_mode_accounts_native_work_without_fixing_throughput():
    entry = next(e for e in CATALOG['entries'] if catalog.describe(e)['scenario']=='agentx')
    lock = binding(entry)
    config = catalog.normalized_recipe(entry, lock)
    row = scenarios.manifest_identity(config, {**lock,'scenario':'agentx'}, 'float4')
    for count in (1,2):
        metrics = scenarios.trace_metrics([trace_record(i) for i in range(count)],lock['request_manifest'])
        measured = {**row, 'metrics':metrics, 'execution_time_ms':metrics['duration_s']*1000}
        validate_serving_case(measured)
        assert metrics['completed_requests']==count
    for invalid in ([trace_record(error='failed')], [trace_record(),trace_record()]):
        with pytest.raises(ValueError): scenarios.trace_metrics(invalid, lock['request_manifest'])
    bad = trace_record(); bad['metrics']['output_sequence_length']['value'] = 1
    with pytest.raises(ValueError, match='token work'): scenarios.trace_metrics([bad],lock['request_manifest'])


@pytest.mark.parametrize('cancelled,cancel_time', [
    (True, None), (True, 1), (False, 1), (None, None), (0, None),
])
def test_trace_rejects_cancelled_or_unproven_completion_even_with_matching_counts(cancelled, cancel_time):
    record = trace_record()
    record['metadata'].update(was_cancelled=cancelled, cancellation_time_ns=cancel_time)
    manifest = dict(requests=[dict(id='trace:0', input_tokens=8, output_tokens=2)])
    with pytest.raises(ValueError, match='cancelled or incomplete'):
        scenarios.trace_metrics([record], manifest)


@pytest.mark.parametrize('field,value', [
    ('request_count', {'avg': 2}), ('completed_request_count', {'avg': 2}),
    ('error_request_count', {'avg': 1}), ('context_overflow_count', {'avg': 1}),
    ('skipped_context_overflow_count', {'avg': 1}), ('request_count', {'avg': 1.5}),
    ('request_count', {'avg': True}), ('request_count', {'avg': None}),
    ('request_count', None), ('was_cancelled', True),
])
def test_trace_export_reconciles_native_counters_and_rejects_skipped_work(field, value):
    metrics = dict(completed_requests=1)
    valid = dict(was_cancelled=False, request_count={'avg': 1.0})
    scenarios.verify_trace_export(valid, metrics)  # Native zero-error counters can be absent.
    with pytest.raises(ValueError):
        scenarios.verify_trace_export({**valid, field: value}, metrics)


def test_trace_rejects_native_context_overflow_skip_with_matching_counts():
    record = trace_record()
    record['metadata']['context_overflow_skip'] = True
    with pytest.raises(ValueError, match='skipped request work'):
        scenarios.trace_metrics([record], dict(requests=[dict(id='trace:0', input_tokens=8, output_tokens=2)]))


def test_trace_shortened_window_rejected():
    entry = next(e for e in CATALOG['entries'] if catalog.describe(e)['scenario']=='agentx')
    lock = binding(entry)
    row = scenarios.manifest_identity(catalog.normalized_recipe(entry,lock),{**lock,'scenario':'agentx'}, 'float4')
    metrics = scenarios.trace_metrics([trace_record()],lock['request_manifest'])
    metrics['duration_s'] = 1; metrics['output_tokens_per_s'] = 2
    with pytest.raises(ValueError, match='fixed measurement window'):
        validate_serving_case({**row,'metrics':metrics,'execution_time_ms':1000})


def test_media_reporting_adaptation_cannot_change_compute_flags(tmp_path):
    folder = tmp_path/'benchmarks'; folder.mkdir()
    script = folder/'vllm_gfx12_mm.sh'
    script.write_text('vllm bench serve \\\n    --save-result\n')
    scenarios.compose_client(tmp_path,dict(scenario='multimodal',benchmark_script=script.name))
    assert script.read_text() == 'vllm bench serve \\\n    --save-detailed\n    --save-result\n'
    with pytest.raises(ValueError): scenarios.compose_client(tmp_path,dict(scenario='multimodal',benchmark_script=script.name))


def test_completion_checks_no_logprob_and_teacher_forcing_paths():
    calls=[]
    def send(port,path,body,**kwargs):
        calls.append(body)
        if not body['echo']: return dict(choices=[dict(token_ids=[3,4])])
        return dict(choices=[dict(logprobs=dict(tokens=['token_id:1','token_id:2','token_id:3'],
            token_logprobs=[None,-.2,-.3],top_logprobs=[None,{'token_id:2':-.2},{'token_id:3':-.3}]))])
    plain=backends.completion_request(send,Path('/task'),1,'generate',dict(input_ids=[1], sampling_params=dict(max_new_tokens=2)))
    assert plain['output_ids']==[3,4] and calls[-1]['logprobs'] is None
    teacher=backends.completion_request(send,Path('/task'),1,'generate',dict(input_ids=[1,2], return_logprob=True,
        top_logprobs_num=1, sampling_params=dict(max_new_tokens=1)))
    assert teacher['meta_info']['input_top_logprobs']==[None,[[-.2,2]]]


def test_atom_native_ids_are_reported_without_changing_ordinary_responses(monkeypatch):
    module=types.ModuleType('atom.entrypoints.openai.serving_completion')
    module.build_completion_response=lambda *args:types.SimpleNamespace(choices=[{'text':'native'}], usage={'completion_tokens':2})
    parent=types.ModuleType('atom.entrypoints.openai');parent.serving_completion=module
    monkeypatch.setitem(sys.modules,parent.__name__,parent)
    monkeypatch.setitem(sys.modules,module.__name__,module)
    lock=dict(framework='atom',correctness={'model_reference':'production_tokens'})
    backends.install_protocol(lock)
    function=module.build_completion_response
    backends.install_protocol(lock)
    assert function is module.build_completion_response
    ordinary=function('id','model',{'token_ids':[3,4]})
    assert ordinary.choices==[{'text':'native'}]
    from array import array
    result=function('id','model',{'prompt_token_ids':[1,2],'token_ids':array('I',[3,4])})
    assert result.choices==[{'text':'native','token_ids':[3,4]}]
    assert function('id','model',{'prompt_token_ids':[1,2],'token_ids':[3,4]}).choices==result.choices
    for bad_ids in (None, array('f',[3,4]), [3,-1]):
        with pytest.raises(ValueError,match='actual generated token IDs'):
            function('id','model',{'prompt_token_ids':[1,2],'token_ids':bad_ids})
    with pytest.raises(ValueError,match='contradict'):
        function('id','model',{'prompt_token_ids':[1,2],'token_ids':[3]})
    request=dict(input_ids=[1,2],sampling_params={'max_new_tokens':2})
    send=lambda *args,**kwargs:dict(choices=[{'token_ids':[3,4]}],prompt_token_ids=[1,2])
    assert backends.completion_request(send,Path('/task'),1,'generate',request,require_prompt_ids=True)['output_ids']==[3,4]
    with pytest.raises(ValueError,match='prompt token IDs'):
        backends.completion_request(lambda *args,**kwargs:dict(choices=[{'token_ids':[3,4]}]),
            Path('/task'),1,'generate',request,require_prompt_ids=True)


def test_atom_token_policy_is_explicit_exact_and_cannot_replace_other_policies(tmp_path):
    entry=next(e for e in CATALOG['entries'] if catalog.describe(e)['framework']=='atom')
    profile=binding(entry)
    profile['correctness'].update(model_reference='production_tokens',token_match='exact')
    task=catalog.materialize(CATALOG,entry['id'],profile,tmp_path/'task')
    assert 'install_protocol(lock)' in (task/'scripts/sitecustomize.py').read_text()
    profile['correctness']['token_match']='approximate'
    with pytest.raises(ValueError,match='exact equality'):
        catalog.validate_binding(profile,catalog.describe(entry))
    profile['correctness']['token_match']='exact'
    with pytest.raises(ValueError,match='independent model numerical reference'):
        catalog.validate_binding(profile,dict(framework='sglang',scenario='text'))


def test_atom_fp32_token_policy_requires_a_frozen_finite_regret_bound(tmp_path):
    entry=next(e for e in CATALOG['entries'] if catalog.describe(e)['framework']=='atom')
    profile=binding(entry)
    profile['correctness'].update(model_reference='transformers_tokens',model_logprob_atol=.1)
    task=catalog.materialize(CATALOG,entry['id'],profile,tmp_path/'task')
    assert json.loads((task/'runtime.lock.json').read_text())['correctness']['model_reference']=='transformers_tokens'
    for tolerance in (None, 0, float('nan')):
        profile['correctness']['model_logprob_atol']=tolerance
        with pytest.raises(ValueError,match='positive finite numerical tolerance'):
            catalog.validate_binding(profile,catalog.describe(entry))


def test_native_token_reference_rejects_wrong_compute_missing_or_reordered_work():
    from src.tools.perf.serving import exact_token_checks,ModelNumericalFailure
    row=dict(input_tokens=[1,2],tokens=[1,2,3,4],output_length=2,check_kind='short',concurrency=1,return_logprob=False)
    assert exact_token_checks([copy.deepcopy(row)],[row])==dict(policy='exact_native_tokens',samples=1,generated_tokens_checked=2,logprob_checks=False)
    for key,value in [('tokens',[1,2,0,0]),('input_tokens',[2,1]),('output_length',1),('return_logprob',True)]:
        wrong=copy.deepcopy(row);wrong[key]=value
        with pytest.raises(ModelNumericalFailure):exact_token_checks([wrong],[row])
    with pytest.raises(ModelNumericalFailure):exact_token_checks([row],[])


def test_prepared_client_runtime_is_fixed_and_never_installed(tmp_path,monkeypatch):
    import io,tarfile
    archive=tmp_path/'assets/client.tar.gz';archive.parent.mkdir()
    with tarfile.open(archive,'w:gz') as stream:
        content=b'#!/bin/sh\nexit 0\n'
        item=tarfile.TarInfo('python/bin/python3.12');item.size=len(content);item.mode=0o755
        stream.addfile(item,io.BytesIO(content))
    lock=dict(client_runtime=dict(asset='client.tar.gz',executable='python/bin/python3.12'),
        assets={'client.tar.gz':dict(sha256=catalog.sha256(archive.read_bytes()))})
    monkeypatch.setattr(scenarios,'_CLIENT_RUNTIMES',{})
    destination=tmp_path/'extracted';destination.mkdir()
    monkeypatch.setattr(scenarios.tempfile,'mkdtemp',lambda **kwargs:str(destination))
    executable=Path(scenarios.client_python(tmp_path,lock))
    assert executable.read_bytes()==content
    assert scenarios.client_python(tmp_path,lock)==str(executable)
    executable.write_bytes(b'changed runtime')
    with pytest.raises(ValueError,match='changed during'):scenarios.client_python(tmp_path,lock)


@pytest.mark.parametrize('name,link',[('../escape',''),('python/link','/etc/passwd')])
def test_client_runtime_rejects_archive_traversal(tmp_path,monkeypatch,name,link):
    import io,tarfile
    archive=tmp_path/'assets/client.tar.gz';archive.parent.mkdir()
    with tarfile.open(archive,'w:gz') as stream:
        item=tarfile.TarInfo(name)
        if link:item.type=tarfile.SYMTYPE;item.linkname=link
        stream.addfile(item,io.BytesIO())
    lock=dict(client_runtime=dict(asset='client.tar.gz',executable='python/bin/python3.12'),
        assets={'client.tar.gz':dict(sha256=catalog.sha256(archive.read_bytes()))})
    monkeypatch.setattr(scenarios,'_CLIENT_RUNTIMES',{})
    destination=tmp_path/'extracted';destination.mkdir()
    monkeypatch.setattr(scenarios.tempfile,'mkdtemp',lambda **kwargs:str(destination))
    with pytest.raises(ValueError,match='Unsafe|escapes'):scenarios.client_python(tmp_path,lock)


def test_client_runtime_preserves_in_root_relative_links_on_older_python(tmp_path,monkeypatch):
    import io,tarfile
    archive=tmp_path/'assets/client.tar.gz';archive.parent.mkdir()
    with tarfile.open(archive,'w:gz') as stream:
        content=b'#!/bin/sh\nexit 0\n'
        item=tarfile.TarInfo('python/bin/python3.12');item.size=len(content);item.mode=0o755
        stream.addfile(item,io.BytesIO(content))
        item=tarfile.TarInfo('python/links/interpreter');item.type=tarfile.SYMTYPE;item.linkname='../bin/python3.12'
        stream.addfile(item)
    original=tarfile.data_filter
    def old_filter(member,destination):
        if member.issym():raise ValueError('Old CPython symlink-base bug')
        return original(member,destination)
    monkeypatch.setattr(tarfile,'data_filter',old_filter)
    monkeypatch.setattr(scenarios,'_CLIENT_RUNTIMES',{})
    destination=tmp_path/'extracted';destination.mkdir()
    monkeypatch.setattr(scenarios.tempfile,'mkdtemp',lambda **kwargs:str(destination))
    lock=dict(client_runtime=dict(asset='client.tar.gz',executable='python/links/interpreter'),
        assets={'client.tar.gz':dict(sha256=catalog.sha256(archive.read_bytes()))})
    assert Path(scenarios.client_python(tmp_path,lock)).read_bytes()==content
    assert (destination/'python/links/interpreter').is_symlink()


def test_compile_preflights_native_client_without_bootstrapping(tmp_path,monkeypatch):
    from src.tools.perf import catalog_checks as checks
    monkeypatch.setattr(checks,'ROOT',tmp_path)
    client=tmp_path/'client';(client/'benchmarks').mkdir(parents=True)
    script=client/'benchmarks/magpie_bench_remote_compat.sh'
    script.write_text('magpie_infx_client_ready() { test "$PIP_NO_INDEX" = 1 && test "$UV_OFFLINE" = 1 && "$1" -c "import sys; assert sys.version_info >= (3, 12)"; }\n')
    monkeypatch.setitem(sys.modules,'_aka_serving',types.SimpleNamespace(prepare_client=lambda lock:client))
    monkeypatch.setitem(sys.modules,'_aka_serving_scenarios',scenarios)
    lock=dict(scenario='text',image='fixture@sha256:'+'a'*64,client_runtime=dict(kind='image',executable=sys.executable))
    assert checks.prepare_runtime(lock)['prepared_client_runtime']=='PASS'
    script.write_text('magpie_infx_client_ready() { return 1; }\nmagpie_infx_client_python() { touch should-not-bootstrap; }\n')
    with pytest.raises(ValueError,match='no bootstrap'):checks.prepare_runtime(lock)
    assert not (client/'should-not-bootstrap').exists()


def test_timed_python_sources_and_extra_files_are_checked(tmp_path,monkeypatch):
    from src.tools.perf import serving
    monkeypatch.setattr(serving,'ROOT',tmp_path)
    monkeypatch.setitem(sys.modules,'_aka_serving_scenarios',scenarios)
    source=tmp_path/'runtime/inferencex/client';(source/'benchmarks').mkdir(parents=True)
    (source/'benchmarks/benchmark_lib.sh').write_text('native benchmark')
    (source/'timed_client.py').write_text('print("native client")\n')
    native=source/'infx/bench_serving/benchmark_serving.py';native.parent.mkdir(parents=True)
    native.write_text('def main():\n    goodput_config_dict = check_goodput_args(args)\n')
    helper=tmp_path/'scripts/_aka_serving_request_plan.py';helper.parent.mkdir()
    helper.write_bytes((ROOT/'src/tools/perf/serving_request_plan.py').read_bytes())
    scripts=tmp_path/'runtime/magpie/Magpie/scripts/benchmark';scripts.mkdir(parents=True)
    (scripts/'sglang_fixture.sh').write_text('native launcher')
    lock=dict(catalog_version=1,scenario='text',dependencies={'inferencex':{'subdirectory':'client'}},
        benchmark_scripts={'benchmarks/benchmark_lib.sh':catalog.sha256(b'native benchmark'),
                           'benchmarks/sglang_fixture.sh':catalog.sha256(b'native launcher')})
    client=serving.prepare_client(lock)
    target=client/'timed_client.py';target.write_text('print("changed client")\n')
    with pytest.raises(ValueError,match='Timed client source'):serving.prepare_client(lock)
    target.write_bytes((source/'timed_client.py').read_bytes())
    (client/'extra.py').write_text('unexpected runtime source')
    with pytest.raises(ValueError,match='Timed client source'):serving.prepare_client(lock)


def test_effective_settings_bind_the_actual_readonly_model(monkeypatch):
    config=types.ModuleType('vllm.config')
    config.get_current_vllm_config=lambda:dict(model_config={'model':'/task/model'},parallel_config={'tensor_parallel_size':4,'enable_expert_parallel':True})
    monkeypatch.setitem(sys.modules,'vllm.config',config)
    value=backends.runtime_settings({'framework':'vllm'})
    assert value['parallel_config']==dict(tensor_parallel_size=4,enable_expert_parallel=True)
    config.get_current_vllm_config=lambda:dict(model_config={'model':'other-model'})
    with pytest.raises(ValueError,match='pinned read-only'):backends.runtime_settings({'framework':'vllm'})


def test_vllm_observes_live_configuration_after_loading_context_exits(monkeypatch):
    module=types.ModuleType('vllm.config')
    def inactive():raise AssertionError('No active loading context')
    module.get_current_vllm_config=inactive
    monkeypatch.setitem(sys.modules,'vllm.config',module)
    config=dict(model_config={'model':'/task/model','dtype':'float16'},parallel_config={'tensor_parallel_size':4},
                compilation_config={'cache_dir':''})
    child=types.SimpleNamespace(vllm_config=config)
    model=types.SimpleNamespace(modules=lambda:[child])
    assert backends.runtime_settings({'framework':'vllm'},model)['parallel_config']['tensor_parallel_size']==4
    inputs=backends.runtime_inputs({'framework':'vllm'},model)
    config['compilation_config']['cache_dir']='/tmp/.cache/vllm/torch_compile_cache/aaaaaaaaaa'
    assert inputs==dict(compilation_cache_dir='')
    assert backends.runtime_inputs({'framework':'vllm'},model)['compilation_cache_dir'].endswith('aaaaaaaaaa')
    config['parallel_config']['tensor_parallel_size']=2
    assert backends.runtime_settings({'framework':'vllm'},model)['parallel_config']['tensor_parallel_size']==2


def test_model_observation_preserves_fullgraph_compilation(tmp_path,monkeypatch):
    import torch
    (tmp_path/'compiled_fixture.py').write_text('import torch\nclass FixtureModel(torch.nn.Module):\n    def forward(self,x): return x*2+1\n')
    source=tmp_path/'source';source.mkdir()
    (source/'patches.json').write_text(json.dumps(dict(version=1,patches=[])))
    lock=dict(framework='vllm',gpu_count=1,limits=dict(max_candidate_source_bytes=4096),candidate_bundle=dict(root='source',manifest='patches.json',
        module_prefixes=['compiled_fixture'],allow_empty=True,scope='whole_model',
        observer=dict(kind='model_classes',classes=['FixtureModel'],entrypoint='__call__')))
    monkeypatch.syspath_prepend(str(tmp_path))
    calls=[];hooks=sys.meta_path[:];cache=dict(value='')
    monkeypatch.setitem(sys.modules,'_aka_serving_backends',types.SimpleNamespace(
        runtime_inputs=lambda lock,model:dict(compilation_cache_dir=cache['value'])))
    monkeypatch.setattr(serving_bundle,'record',lambda bundle,label,observer=False,model=None,configuration_inputs=None:
        calls.append((model,configuration_inputs)))
    def compile_graph(graph, examples):
        cache['value']='/tmp/.cache/vllm/torch_compile_cache/aaaaaaaaaa'
        return graph.forward
    try:
        serving_bundle.install(tmp_path,lock)
        module=importlib.import_module('compiled_fixture');model=module.FixtureModel()
        model.forward=torch.compile(model.forward,backend=compile_graph,fullgraph=True)
        assert model(torch.tensor(2)).item()==5
        assert model(torch.tensor(3)).item()==7
        assert calls==[(model,dict(compilation_cache_dir=''))]
        assert cache['value'].endswith('aaaaaaaaaa')
    finally:
        sys.meta_path[:]=hooks;sys.modules.pop('compiled_fixture',None)


def test_replacement_tracing_preserves_fullgraph_and_is_not_reported_as_gpu_execution(tmp_path,monkeypatch):
    import torch
    (tmp_path/'compiled_compute_fixture.py').write_text('def compute(x): return x+1\n')
    source=tmp_path/'source';source.mkdir()
    (source/'fused.py').write_text('def fused(x): return x*x+1\n')
    (source/'patches.json').write_text(json.dumps(dict(version=1,patches=[dict(
        module='compiled_compute_fixture',symbol='compute',file='fused.py',callable='fused')])))
    lock=dict(limits=dict(max_candidate_source_bytes=4096),candidate_bundle=dict(root='source',manifest='patches.json',
        module_prefixes=['compiled_compute_fixture'],scope='whole_model',observer=dict(kind='model_classes',classes=['UnusedModel'])))
    monkeypatch.syspath_prepend(str(tmp_path))
    calls=[];hooks=sys.meta_path[:]
    monkeypatch.setattr(serving_bundle,'record',lambda bundle,label,**kwargs:calls.append(kwargs))
    try:
        serving_bundle.install(tmp_path,lock)
        module=importlib.import_module('compiled_compute_fixture')
        compiled=torch.compile(module.compute,backend='eager',fullgraph=True)
        assert compiled(torch.tensor(2)).item()==5
        assert compiled(torch.tensor(3)).item()==10
        assert calls==[dict(execution='compiled_trace')]
    finally:
        sys.meta_path[:]=hooks;sys.modules.pop('compiled_compute_fixture',None)


def test_generic_observer_covers_new_compute_and_direct_model_forward(tmp_path,monkeypatch):
    (tmp_path/'fixture_engine.py').write_text('class Layer:\n    def forward(self,x): return x\nclass FixtureModel:\n    def forward(self,x): return Layer().forward(x)\n')
    source=tmp_path/'source';source.mkdir()
    (source/'fusion.py').write_text('def compute(self,x): return x*2+1\n')
    patches=[dict(module='fixture_engine',symbol='Layer.forward',file='fusion.py',callable='compute')]
    (source/'patches.json').write_text(json.dumps(dict(version=1,patches=patches)))
    lock=dict(gpu_count=1,limits=dict(max_candidate_source_bytes=4096),candidate_bundle=dict(root='source',manifest='patches.json',
        module_prefixes=['fixture_engine'],allow_empty=True,observer=dict(kind='model_classes',classes=['FixtureModel'])))
    monkeypatch.syspath_prepend(str(tmp_path))
    calls=[];monkeypatch.setattr(serving_bundle,'record',lambda bundle,label,observer=False: calls.append((label,observer)))
    hooks=sys.meta_path[:]
    try:
        serving_bundle.install(tmp_path,lock)
        import fixture_engine
        assert fixture_engine.FixtureModel().forward(3)==7
        assert fixture_engine.FixtureModel().forward(8)==17
        assert calls==[('fixture_engine:Layer.forward',False),('model_forward',True)]
        originals=serving_bundle.original_sources(serving_bundle.Bundle(tmp_path,lock))
        assert 'return x' in originals[0]['source']
        assert 'class FixtureModel' in originals[0]['module_source']
    finally:
        sys.meta_path[:]=hooks;sys.modules.pop('fixture_engine',None)


def test_media_checks_use_actual_image_body_and_both_generation_modes(tmp_path,monkeypatch):
    monkeypatch.setitem(sys.modules,'_aka_serving_backends',backends)
    case=dict(messages=[dict(role='user',content=[dict(type='image_url',image_url={'url':'data:image/png;base64,fixture'})])],output_tokens=2)
    calls=[]
    def send(port,path,body):
        calls.append(body)
        return dict(usage={'completion_tokens':2},choices=[dict(token_ids=[1,2],logprobs={'content':[
            dict(token='token_id:1',logprob=-.1),dict(token='token_id:2',logprob=-.2)]})])
    plan=[dict(media=[case])]
    for mode in (True,False):
        rows=scenarios.media_generations(tmp_path,1,plan,send,return_logprob=mode)
        assert rows[0]['tokens']==[0,1,2] and calls[-1]['logprobs'] is mode
        assert calls[-1]['messages']==case['messages']
    def missing(*args):return dict(usage={'completion_tokens':2},choices=[dict(token_ids=[1])])
    with pytest.raises(ValueError,match='token IDs'):scenarios.media_generations(tmp_path,1,plan,missing)


def test_new_kernel_dsl_is_not_limited_to_seed_operator_backend(tmp_path):
    entry=CATALOG['entries'][-1]
    dest=catalog.materialize(CATALOG,entry['id'],binding(entry),tmp_path/'task')
    (dest/'source/new_fusion.py').write_text('import flydsl\nimport wave_lang\n\ndef compute(x): return x\n')
    lock=json.loads((dest/'runtime.lock.json').read_text())
    evidence=serving_bundle.Bundle(dest,lock).check_sources()
    assert 'new_fusion.py' in evidence['source_hashes']


def test_editable_engine_import_and_optional_namespaces(tmp_path,monkeypatch):
    import importlib.abc
    file=tmp_path/'editable_source.py'
    file.write_text('class FixtureModel:\n    def forward(self,x): return x+1\n')
    class Editable(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if fullname=='fixture_editable_engine':
                return importlib.util.spec_from_file_location(fullname,file)
            return None
    source=tmp_path/'source';source.mkdir()
    (source/'patches.json').write_text(json.dumps(dict(version=1,patches=[])))
    lock=dict(gpu_count=1,limits=dict(max_candidate_source_bytes=4096),candidate_bundle=dict(root='source',manifest='patches.json',
        module_prefixes=['fixture_editable_engine'],allow_empty=True,observer=dict(kind='model_classes',classes=['FixtureModel'])))
    hooks=sys.meta_path[:];calls=[]
    monkeypatch.setattr(serving_bundle,'record',lambda bundle,label,observer=False:calls.append((label,observer)))
    try:
        sys.meta_path.append(Editable())
        serving_bundle.install(tmp_path,lock)
        module=importlib.import_module('fixture_editable_engine')
        assert module.FixtureModel().forward(2)==3
        assert calls==[('model_forward',True)]
    finally:
        sys.meta_path[:]=hooks;sys.modules.pop('fixture_editable_engine',None)


def test_random_seed_is_fixed_without_replacing_engine_options():
    assert backends.seeded_arguments(['--dtype','float16'],1729)==['--dtype','float16','--random-seed','1729']
    assert backends.seeded_arguments(['--random-seed=7'],7)==['--random-seed=7']
    with pytest.raises(ValueError,match='conflict'):backends.seeded_arguments(['--random-seed','3'],7)


def test_whole_model_allows_compute_helpers_beyond_forward(tmp_path):
    source=tmp_path/'source';source.mkdir()
    (source/'fusion.py').write_text('def fused_gate(x): return x*x\n')
    (source/'patches.json').write_text(json.dumps(dict(version=1,patches=[dict(
        module='sglang.srt.models.fixture',symbol='Layer.fused_gate',file='fusion.py',callable='fused_gate')])))
    lock=dict(limits=dict(max_candidate_source_bytes=4096),candidate_bundle=dict(
        root='source',manifest='patches.json',module_prefixes=['sglang'],scope='whole_model'))
    candidate=serving_bundle.Bundle(tmp_path,lock)
    assert candidate.function(candidate.patches[0])(3)==9
    lock['candidate_bundle'].pop('scope')
    with pytest.raises(ValueError,match='forward compute boundary'):serving_bundle.Bundle(tmp_path,lock)


def test_preparation_freezes_custom_model_files_and_offline_assets(tmp_path, monkeypatch):
    import io
    from src import prepare_serving_runtime as runtime
    names = ['config.json', 'model.safetensors', 'tokenizer.model', 'chat_template.jinja',
             'modeling_fixture.py', 'weights.bin', 'LICENSE', 'README.md']
    details = dict(sha='a'*40, siblings=[dict(rfilename=name) for name in names])
    monkeypatch.setattr(runtime.urllib.request, 'urlopen',
        lambda url: io.BytesIO(json.dumps(details).encode()))
    acquired = []
    def acquire(url, target):
        acquired.append(url)
        target.write_bytes(b'offline fixture')
    monkeypatch.setattr(runtime, 'acquire', acquire)
    lock = dict(model=dict(id='fixture/model', revision='a'*40), dependencies={},
        assets={'traces/turns.jsonl': dict(url='https://example.invalid/frozen',
            sha256=catalog.sha256(b'offline fixture'))})
    destination = runtime.prepare(lock, tmp_path/'cache')
    assert {p.name for p in (destination/'model').iterdir()} == set(names)-{'README.md'}
    assert (destination/'assets/traces/turns.jsonl').read_bytes() == b'offline fixture'
    runtime.verify(destination, runtime.asset_identity(lock))
    assert runtime.prepare(lock, tmp_path/'cache') == destination
    assert len(acquired) == len(names)
    (destination/'assets/traces/turns.jsonl').write_bytes(b'different trace')
    with pytest.raises(ValueError, match='content changed'):
        runtime.prepare(lock, tmp_path/'cache')


def test_wrong_offline_asset_is_not_cached(tmp_path, monkeypatch):
    import io
    from src import prepare_serving_runtime as runtime
    monkeypatch.setattr(runtime.urllib.request, 'urlopen',
        lambda url: io.BytesIO(json.dumps(dict(sha='a'*40,siblings=[])).encode()))
    monkeypatch.setattr(runtime, 'acquire', lambda url,target: target.write_bytes(b'wrong bytes'))
    lock = dict(model=dict(id='fixture/model',revision='a'*40),dependencies={},
        assets={'trace.jsonl': dict(url='https://example.invalid/frozen',sha256='b'*64)})
    with pytest.raises(ValueError,match='fixed digest'):runtime.prepare(lock,tmp_path/'cache')
    assert not [p for p in (tmp_path/'cache').iterdir() if not p.name.startswith('.')]


def test_provided_asset_location_is_host_only_and_content_verified(tmp_path, monkeypatch):
    import io
    from src import prepare_serving_runtime as runtime
    monkeypatch.setattr(runtime.urllib.request, 'urlopen',
        lambda url: io.BytesIO(json.dumps(dict(sha='a'*40, siblings=[])).encode()))
    asset = tmp_path/'private-host-location'
    asset.write_bytes(b'prepared environment')
    lock = dict(model=dict(id='fixture/model', revision='a'*40), dependencies={},
        assets={'client.tar.gz': dict(source='provided', sha256=catalog.sha256(asset.read_bytes()))})
    with pytest.raises(ValueError, match='--asset-file'):
        runtime.prepare(lock, tmp_path/'cache')
    with pytest.raises(ValueError, match='not declared'):
        runtime.prepare(lock, tmp_path/'cache', {'typo': asset})
    dest = runtime.prepare(lock, tmp_path/'cache', {'client.tar.gz': asset})
    assert (dest/'assets/client.tar.gz').read_bytes() == b'prepared environment'
    assert str(asset) not in (dest/'artifact_manifest.json').read_text()
    assert runtime.prepare(lock, tmp_path/'cache') == dest
    asset.write_bytes(b'changed bytes')
    with pytest.raises(ValueError, match='fixed digest'):
        runtime.prepare(lock, tmp_path/'other-cache', {'client.tar.gz': asset})
    (dest/'assets/client.tar.gz').write_bytes(b'changed cached bytes')
    with pytest.raises(ValueError, match='content changed'):
        runtime.prepare(lock, tmp_path/'cache')


@pytest.mark.parametrize('asset', [
    dict(url='file:///private/host/artifact', sha256='a'*64),
    dict(url='http://example.invalid/artifact', sha256='a'*64),
    dict(source='provided', url='https://example.invalid/artifact', sha256='a'*64),
    dict(source='provided', sha256='bad'),
])
def test_asset_descriptor_rejects_private_paths_and_ambiguous_sources(asset):
    from src.prepare_serving_runtime import validate_assets
    with pytest.raises(ValueError): validate_assets({'artifact': asset})


@pytest.mark.parametrize('base_module,native_module', [
    ('vllm.model_executor.layers.quantization.base_config', 'vllm.model_executor.layers.quantization.fp8'),
    ('atom.config', 'atom.config'),
    ('vllm.model_executor.layers.quantization.base_config', 'third_party_quant_plugin'),
])
def test_native_quantization_configuration_keeps_all_precision_fields(monkeypatch, base_module, native_module):
    module = types.ModuleType(base_module)
    class QuantizationConfig: pass
    module.QuantizationConfig = QuantizationConfig
    monkeypatch.setitem(sys.modules, module.__name__, module)
    class NativeConfig(QuantizationConfig):
        def __init__(self):
            self.weight_block_size = [128, 128]
            self.activation_scheme = 'dynamic'
            self._precision_mode = 'fp8'
    NativeConfig.__module__ = native_module
    value = NativeConfig()
    before = backends._plain(value)
    assert before['fields'] == vars(value)
    assert before['configuration_type'].endswith('.NativeConfig')
    value.weight_block_size[0] = 64
    assert before != backends._plain(value)
    value._precision_mode = object()
    with pytest.raises(ValueError, match='Unsupported effective configuration'):
        backends._plain(value)


def test_native_configuration_enum_is_not_dropped_or_stringified():
    meta = type('pybind11_type', (type,), {'__module__': 'pybind11_builtins'})
    class NativeChoice(metaclass=meta):
        __members__ = {}
        def __init__(self, name, value): self.name, self.value = name, value
        def __int__(self): return self.value
    NativeChoice.__module__ = 'module_aiter_core'
    plain = backends._plain(NativeChoice('No', 0))
    assert plain['name'] == 'No' and plain['value'] == 0
    assert plain['enum_type'].endswith('.NativeChoice')
    assert plain != backends._plain(NativeChoice('per_Tensor', 1))
    class NativeObject(metaclass=meta):
        name, value = 'No', 0
    with pytest.raises(ValueError, match='Unsupported effective configuration'):
        backends._plain(NativeObject())


def test_generic_hook_preserves_python_module_entrypoint_protocol(tmp_path):
    source=tmp_path/'source';source.mkdir()
    (source/'patches.json').write_text(json.dumps(dict(version=1,patches=[])))
    lock=dict(limits=dict(max_candidate_source_bytes=4096),candidate_bundle=dict(
        root='source',manifest='patches.json',module_prefixes=['fixture_engine'],allow_empty=True,
        observer=dict(kind='model_classes',classes=['FixtureModel'])))
    (tmp_path/'runtime.lock.json').write_text(json.dumps(lock))
    (tmp_path/'fixture_engine.py').write_text('print("MODULE_ENTRYPOINT_EXECUTED")\n')
    (tmp_path/'sitecustomize.py').write_text(
        'import json\nfrom pathlib import Path\nfrom src.tools.perf.serving_bundle import install\n'
        'root=Path(__file__).parent\ninstall(root,json.loads((root/"runtime.lock.json").read_text()))\n')
    env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(tmp_path),str(ROOT)]))
    completed=subprocess.run([sys.executable,'-m','fixture_engine'],cwd=tmp_path,env=env,
        capture_output=True,text=True,timeout=30)
    assert completed.returncode==0,completed.stderr
    assert completed.stdout.strip()=='MODULE_ENTRYPOINT_EXECUTED'
    assert 'Error in sitecustomize' not in completed.stderr


def test_native_trace_integral_float_counts_are_exactly_normalized():
    manifest=dict(requests=[dict(id='trace:0',input_tokens=8,output_tokens=2)])
    record=trace_record()
    record['metrics']['input_sequence_length']['value']=8.0
    record['metrics']['output_sequence_length']['value']=2.0
    result=scenarios.trace_metrics([record],manifest)
    assert result['input_lens']==[8] and type(result['input_lens'][0]) is int
    for invalid in (1.5,True,float('nan'),float('inf'),'2'):
        record['metrics']['output_sequence_length']['value']=invalid
        with pytest.raises(ValueError,match='token work'):scenarios.trace_metrics([record],manifest)
    record['metrics']['output_sequence_length']['value']=2
    record['metadata']['request_end_ns']=-1
    with pytest.raises(ValueError,match='timestamps'):scenarios.trace_metrics([record],manifest)


def test_modern_sglang_observes_resolved_overrides_not_retired_or_raw_args(monkeypatch):
    context=types.ModuleType('sglang.srt.runtime_context')
    actual=dict(model_path='/task/model',tp_size=4,enable_ep_moe=True,random_seed=1729)
    context.get_context=lambda:types.SimpleNamespace(resolved_server_args_dict=lambda:actual)
    monkeypatch.setitem(sys.modules,'sglang.srt.runtime_context',context)
    assert backends.runtime_settings({'framework':'sglang'})==actual
    actual['tp_size']=2
    assert backends.runtime_settings({'framework':'sglang'})['tp_size']==2
    actual['model_path']='/other/model'
    with pytest.raises(ValueError,match='pinned read-only'):
        backends.runtime_settings({'framework':'sglang'})


def test_configuration_serialization_excludes_noninput_runtime_handles():
    from dataclasses import dataclass,field
    @dataclass
    class EngineConfig:
        tensor_parallel_size:int=4
        cache_dtype:str='fp8'
        torch_dtype:str=field(default='float16',init=False)
        static_forward_context:dict=field(default_factory=lambda:{'layer':object()},init=False)
    assert backends._plain(EngineConfig())==dict(tensor_parallel_size=4,cache_dtype='fp8',torch_dtype='float16')


@pytest.mark.parametrize('engine,module,symbol',[
    ('sglang','sglang.srt.server_args','get_global_server_args'),
    ('sglang','sglang.srt.runtime_context','get_context'),
    ('sglang','sglang.srt.arg_groups.fixture','resolved_dict'),
    ('vllm','vllm.config','get_current_vllm_config'),
    ('atom','atom.config','get_current_atom_config'),
])
def test_configuration_observer_cannot_be_declared_as_compute(tmp_path,engine,module,symbol):
    entry=next(e for e in CATALOG['entries'] if catalog.describe(e)['framework']==engine)
    task=catalog.materialize(CATALOG,entry['id'],binding(entry),tmp_path/'task')
    (task/'source/config_spoof.py').write_text('def fake(): return {}\n')
    (task/'source/patches.json').write_text(json.dumps(dict(version=1,patches=[dict(
        module=module,symbol=symbol,file='config_spoof.py',callable='fake')])))
    lock=json.loads((task/'runtime.lock.json').read_text())
    with pytest.raises(ValueError,match='observation modules are protected'):
        serving_bundle.Bundle(task,lock)


def test_whole_model_accepts_private_compute_and_stateful_cache_helpers(tmp_path):
    source=tmp_path/'source';source.mkdir()
    (source/'fusion.py').write_text('''import torch
class State:
    def __init__(self, value):
        self.value = value
    def set_kv_buffer(self, value):
        self.value.copy_(value)
def _compute(x):
    state = State(torch.zeros_like(x))
    state.set_kv_buffer(x)
    return getattr(state, 'value')
''')
    (source/'patches.json').write_text(json.dumps(dict(version=1,patches=[dict(
        module='sglang.srt.layers._cache',symbol='Cache.set_kv_buffer',file='fusion.py',callable='_compute')])))
    lock=dict(limits=dict(max_candidate_source_bytes=4096),candidate_bundle=dict(
        root='source',manifest='patches.json',module_prefixes=['sglang'],scope='whole_model',
        protected_modules=['sglang.srt.runtime_context']))
    candidate=serving_bundle.Bundle(tmp_path,lock)
    candidate.check_sources()
    import torch
    for values in ([1.,2.],[3.,4.]):
        x=torch.tensor(values)
        assert torch.equal(candidate.function(candidate.patches[0])(x),x)


@pytest.mark.parametrize('count_env',[{'NUM_PROMPTS':1},{'CONC':1}])
def test_native_plan_preserves_tokenization_and_precedes_timing(tmp_path, monkeypatch, count_env):
    from src.tools.perf import serving_request_plan as reporter
    root=tmp_path;client=root/'client';native=client/'infx/bench_serving/benchmark_serving.py'
    native.parent.mkdir(parents=True);(client/'benchmarks').mkdir()
    native.write_text('''import json, os
from pathlib import Path
def check_goodput_args(args): return {}
def main(args=None):
    count = int(os.environ.get('NUM_PROMPTS', str(int(os.environ.get('CONC', '1'))*10)))
    input_requests = [('native prompt', 129, 128, None)] * count
    goodput_config_dict = check_goodput_args(args)
    Path(os.environ['RESULT_DIR'], 'http-was-called').touch()
    Path(os.environ['RESULT_DIR'], 'inferencex_result.json').write_text(json.dumps(dict(
        duration=31, completed=count, total_input_tokens=129*count, total_output_tokens=128*count,
        output_throughput=128*count/31, p99_tpot_ms=1, input_lens=[129]*count, output_lens=[128]*count)))
main()
''')
    script=client/'benchmarks/fixture.sh'
    script.write_text('exec "$MAGPIE_BENCHMARK_PYTHON" -m infx.bench_serving.benchmark_serving\n')
    lock=dict(scenario='text',image='fixture@sha256:'+'a'*64,client_runtime=dict(kind='image',executable=sys.executable))
    scenarios.compose_client(client, lock, Path(reporter.__file__))
    import_code=native.read_text()
    assert import_code.index('emit_request_plan(input_requests)') < import_code.index('goodput_config_dict =')
    monkeypatch.setitem(sys.modules,'_aka_serving',types.SimpleNamespace(prepare_client=lambda lock:client))
    config=dict(envs=dict(ISL=128,OSL=128,**count_env),benchmark_script='fixture.sh')
    count=count_env.get('NUM_PROMPTS',10)
    row=dict(params=dict(num_requests=count,input_tokens=128,output_tokens_per_request=128))
    limits=dict(client_timeout_s=10,minimum_measurement_s=30)
    scenarios.text_identity(root,config,lock,limits,row)
    assert not (root/'http-was-called').exists()
    assert row['params']['input_lens']==[129]*count and 'input_tokens' not in row['params']
    result,_=scenarios.performance(root,config,lock,limits,client,1234,row)
    assert result['metrics']['input_tokens']==129*count and (root/'http-was-called').exists()
    # Same token counts but different input bytes are a different workload.
    native.write_text(import_code.replace("'native prompt'", "'changed prompt'"))
    with pytest.raises(ValueError,match='changed the prepared native request plan'):
        scenarios.performance(root,config,lock,limits,client,1234,row)
    native.write_text(import_code.replace('input_lens=[129]', 'input_lens=[128]'))
    with pytest.raises(ValueError,match='per-request input_lens'):
        scenarios.performance(root,config,lock,limits,client,1234,row)


@pytest.mark.parametrize('raw', [b'',b'{}',b'{"version":true,"requests":[]}',
    b'{"version":1,"requests":[]}',
    json.dumps(dict(version=1,requests=[dict(prompt_sha256='a'*64,input_tokens=True,output_tokens=2)])).encode()])
def test_missing_or_invalid_native_request_plan_fails_closed(raw):
    with pytest.raises(ValueError): scenarios.read_request_plan(raw,1)


def test_text_plan_adapter_rejects_changed_upstream_anchor():
    with pytest.raises(ValueError,match='contract changed'):
        scenarios.composed_script(b'print("changed")','infx/bench_serving/benchmark_serving.py',dict(scenario='text'))
