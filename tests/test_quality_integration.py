"""Concrete native, paired-variable and ordinary accepted-score pathways."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from src import evaluator, performance
from src.benchmark_quality import comparison_quality, validate_case_quality
from src.native_baseline import as_test_cases, metric_summary, validate_paired_measurements
from src.quality_measurements import portable_test_cases, validated_work_ids, KIMI_FIELDS, MLA_FIELDS
from src.score import task_result_scoring
from src.task_contract import finalize_report, fingerprint
from src.testcases import load_performance_results, save_performance_results


def manifest(variable=False):
    tensor={'role':'input','shape':[2],'strides':[1],'storage_offset':0,'dtype':'float32','device_type':'cuda'}
    case={'case_id':'case','occurrences':100,'calls_per_sample':1,
          'fixture':{'path':'fixtures/input.json','sha256':'e'*64},
          'tensors':{'x':tensor,'out':{**tensor,'role':'output'}},'scalars':{}}
    if variable:case['work_distribution']={'reviewed_test_work_values':[1,1000]}
    return {'schema_version':1,'runtime_image':'image@sha256:'+'d'*64,'cases':[case],
        'measurement':{'method':'cuda_graph','warmup_iterations':10,'benchmark_iterations':100,
          'correctness_seeds':[0,1],'negative_controls':['no_op','wrong_output'],
          'refresh_inputs':'each_replay','initialize_outputs':'each_replay','validate_outputs':'each_replay'}}


def raw_row(case,samples):
    return {'case':case,'correct':True,'samples_ms':list(samples),'warmup_iterations':10,
            'fresh_input_resets':100,'output_initializations':100,'oracle_checks':100,'benchmark_method':'cuda_graph'}


def paired_fixture(tmp_path, *, source='a', variable=True, spike=False):
    m=manifest(variable);case=m['cases'][0];sources={'source/kernel.py':source*64}
    request={'schema_version':1,'phase':'performance','request_id':'request-'+source,
             'manifest_sha256':fingerprint(m),'source_sha256':sources,'challenge_seed':1200,'package_sha256':source*64}
    entries=[]
    for seed in range(1200,1310):
        entries.append({'input_seed':seed,'work_value':1 if seed%2 else 1000,
                        'actual_control_hashes':{},'private_cpu_storage_bytes':8} if variable else
                       {'input_seed':seed,'fixture':case['fixture']})
    schedule={'schema_version':1,'case_id':'case','manifest_sha256':fingerprint(m),
              'warmup_inputs':entries[:10],'measured_inputs':entries[10:]}
    values=[x['work_value'] for x in entries[10:]] if variable else [1]*100
    samples=list(values)
    if spike:samples[1]=1000000
    outer=raw_row(case,samples);candidate=deepcopy(outer);reference=raw_row(case,[2*v for v in values])
    for row in [outer,candidate,reference]:row['paired_schedule_sha256']=fingerprint(schedule)
    cb={'leg':'candidate','source_sha256':sources,'module':'private_candidate','gpu_binding':'hip'}
    rb={'leg':'reference','source_sha256':{'source/kernel.py':'a'*64},'module':'private_reference','gpu_binding':'hip'}
    pair={'case_id':'case','candidate_snapshot_before_reference':True,'reference_reused_as_oracle':True,
          'output_conformance':True,'checked_pair_count':110,'reference_output_clear_calls':110,
          'graph_setup':{'warmup_invocations_per_leg':3,'capture_invocations_per_leg':1,
                        'capture_on_warmed_stream':True,'separate_graphs_and_outputs':True},
          'candidate_binding':cb,'reference_binding':rb,'input_schedule':schedule,
          'legs':{'candidate_port':candidate,'protected_reference':reference}}
    outer['paired_reference']=deepcopy(pair)
    provenance={'baseline_kind':'native_production','reference_source_sha256':rb['source_sha256'],'gpu_binding':'hip',
        'input_signature_fields':{'case':sorted(KIMI_FIELDS) if variable else ['fixture','input_seed']},
        'input_receipt_contracts':{'case':{'kind':'task_local_paired_inputs_v1' if variable else 'fixed_fixture_v1'}}}
    if variable:
        ut=tmp_path/'ut';ut.mkdir(exist_ok=True)
        validator=ut/'validate_inputs.py'
        validator.write_text('''def validate(root, report, manifest, request, provenance):
    schedule = report['paired_reference_comparison']['cases'][0]['input_schedule']
    for i, row in enumerate(schedule['warmup_inputs'] + schedule['measured_inputs']):
        seed = request['challenge_seed'] + i
        expected = {'input_seed': seed, 'work_value': 1 if seed % 2 else 1000,
                    'actual_control_hashes': {}, 'private_cpu_storage_bytes': 8}
        if row != expected:
            raise ValueError('independent work reconstruction rejected')
    return True
''')
        provenance['paired_input_validator']={'schema':'task-local-paired-input-validator-v1',
            'validator':'ut/validate_inputs.py','function':'validate',
            'files_sha256':{'ut/validate_inputs.py':hashlib.sha256(validator.read_bytes()).hexdigest()}}
    policy={'schema_version':2,'kind':'native_production','native_source_manifest':'provenance/PAIRED-REFERENCE.json'}
    digest=fingerprint(provenance)
    report=finalize_report({'schema_version':1,'status':'ok','request':request,'cases':[outer],
        'compiled_specializations':[{'case_id':'case','candidate_binding':cb,'reference_binding':rb,'invoked_and_synchronized':True}]},m,request)
    report['paired_reference_comparison']={'schema':'paired-reference-comparison-v1','schema_version':1,
        'status':'ok','score_input':True,'diagnostic_only':False,'baseline_kind':'native_production',
        'request':request,'source_hashes':sources,'manifest_sha256':fingerprint(m),'runtime_image':m['runtime_image'],
        'native_source_manifest_sha256':digest,'challenge_seed':1200,'anti_cheat_attestation':False,
        'fresh_trusted_host_retest_required':True,'cases':[pair]}
    return json.loads(json.dumps(report)),m,request,sources,digest,policy,provenance


def validate_pair(tmp_path, args):
    return validate_paired_measurements(*args,task_root=tmp_path)


def test_paired_variable_classes_are_reconstructed_before_quality_and_all_samples_survive(tmp_path):
    first=as_test_cases(validate_pair(tmp_path,paired_fixture(tmp_path,source='a')),is_baseline=True)
    second=as_test_cases(validate_pair(tmp_path,paired_fixture(tmp_path,source='b')))
    result=metric_summary(first,second)
    assert result['benchmark_quality']['status']=='pass' and result['native_speedup_ratio']==2
    proof=second[0].metadata['benchmark_quality_case']
    assert len(proof['series']['candidate']['raw_samples_ms'])==100
    assert len(proof['series']['candidate']['work_classes'])==2
    assert {row['sample_count'] for row in proof['series']['candidate']['work_classes']}=={50}
    # Global 1000x work variation is not called fixed-work instability.
    assert proof['status']=='pass'


def test_forged_work_labels_cannot_split_an_outlier_into_its_own_quality_class(tmp_path,monkeypatch):
    args=paired_fixture(tmp_path,source='b',spike=True)
    report=args[0];pair=report['paired_reference_comparison']['cases'][0]
    pair['input_schedule']['measured_inputs'][1]['work_value']=1000000
    digest=fingerprint(pair['input_schedule'])
    for leg in pair['legs'].values():leg['paired_schedule_sha256']=digest
    report['cases'][0]['paired_schedule_sha256']=digest
    report['cases'][0]['paired_reference']=deepcopy(pair)
    from src import quality_measurements
    monkeypatch.setattr(quality_measurements,'validated_work_ids',lambda *a:pytest.fail('projected classes before semantic validation'))
    with pytest.raises(ValueError,match='independent work reconstruction rejected'):validate_pair(tmp_path,args)


def test_paired_failure_rejects_whole_primary_and_secondary_without_losing_raw_means(tmp_path):
    first=as_test_cases(validate_pair(tmp_path,paired_fixture(tmp_path,source='a')),is_baseline=True)
    second=as_test_cases(validate_pair(tmp_path,paired_fixture(tmp_path,source='b',spike=True)))
    result=metric_summary(first,second)
    assert result['benchmark_quality']['status']=='reject'
    assert result['native_speedup_ratio'] is result['port_to_port_speedup_ratio'] is None
    assert result['native_baseline_cases'][0]['speedup'] is None
    assert result['raw_native_speedup_ratio']>0 and result['native_baseline_cases'][0]['candidate_ms']>0
    assert not result['accepted_gain'] and not result['production_kernel_improvement']
    assert len(second[0].metadata['benchmark_quality_case']['series']['candidate']['raw_samples_ms'])==100


def test_paired_unchanged_source_is_control_even_when_faster_than_native(tmp_path):
    first=as_test_cases(validate_pair(tmp_path,paired_fixture(tmp_path)),is_baseline=True)
    second=as_test_cases(validate_pair(tmp_path,paired_fixture(tmp_path)))
    result=metric_summary(first,second)
    assert result['native_speedup_ratio'] is None and result['raw_native_speedup_ratio']==2
    assert result['benchmark_quality']['comparison_status']=='unchanged_source_control'


def test_projection_rejects_unknown_task_local_and_address_only_strata():
    case={'case_id':'test','work_distribution':{}}
    with pytest.raises(ValueError,match='unsupported'):
        validated_work_ids(case,{'measured_inputs':[{'input_seed':i,'address_sha256':'a'*64} for i in range(100)]},
                           {'kind':'task_local_paired_inputs_v1'},['input_seed','address_sha256'])
    # The supported MLA projection uses validated length, not changing indices.
    ids=validated_work_ids({'case_id':'mla','distribution_recipe':{}},
        {'measured_inputs':[{'length':192,'kv_indices_sha256':str(i)} for i in range(100)]},
        {'kind':'task_local_paired_inputs_v1'},MLA_FIELDS)
    assert len(set(ids))==1


@pytest.mark.parametrize('mode',['changed','unchanged','harness_only','spike','missing_quality'])
def test_ordinary_portable_evaluation_and_saved_score_obey_quality(tmp_path,monkeypatch,mode):
    m=manifest();(tmp_path/'cases.json').write_text(json.dumps(m));(tmp_path/'source').mkdir()
    source=tmp_path/'source/kernel.py';source.write_text('VALUE=1\n');(tmp_path/'build').mkdir()
    config={'task_type':'instruction2triton','performance_command':['benchmark'],
            'source_file_path':['source/kernel.py'],'trusted_evaluation':{'schema_version':1}}
    calls=[]
    def run(*a,**k):
        calls.append(1);assert not (tmp_path/'build/performance_report.json').exists()
        src={'source/kernel.py':hashlib.sha256(source.read_bytes()).hexdigest()}
        request={'phase':'performance','request_id':'request-'+str(len(calls)),
                 'manifest_sha256':fingerprint(m),'source_sha256':src,'challenge_seed':42}
        samples=[2.0]*100 if len(calls)==1 else [1.0]*100
        if mode=='spike' and len(calls)==2:samples[27]=10000.0
        report=finalize_report({'schema_version':1,'status':'ok','request':request,
                               'cases':[raw_row(m['cases'][0],samples)]},m,request)
        (tmp_path/'build/performance_report.json').write_text(json.dumps(report))
        return True,'Performance: 0.000001 ms',''
    monkeypatch.setattr(performance,'run_command',run);monkeypatch.setattr(performance,'force_jit_rebuild',lambda *a:{})
    fresh_before=performance.measure_baseline(tmp_path,config)
    before=load_performance_results(tmp_path,'baseline_perf.yaml',validated_cases=fresh_before)
    assert len(before[0].metadata['benchmark_quality_case']['series']['candidate']['raw_samples_ms'])==100
    if mode not in ('unchanged','harness_only'):source.write_text('VALUE=2\n')
    if mode=='harness_only':(tmp_path/'unrelated_harness.txt').write_text('changed')
    if mode=='missing_quality':before[0].metadata.pop('benchmark_quality_case')
    monkeypatch.setattr(evaluator,'evaluate_compilation',lambda *a,**k:(True,None))
    monkeypatch.setattr(evaluator,'evaluate_correctness',lambda *a,**k:(True,None))
    result=evaluator.evaluate_kernel(tmp_path,config,before)
    evaluator.write_task_result(tmp_path,result,before,'portable','test',create_plots=False)
    saved=yaml.safe_load((tmp_path/'task_result.yaml').read_text())
    assert len(calls)==2
    if mode=='changed':
        assert saved['speedup_ratio']==2 and saved['accepted_gain']
        assert task_result_scoring(str(tmp_path))==320
    else:
        assert saved['speedup_ratio'] is None and not saved['accepted_gain']
        assert task_result_scoring(str(tmp_path))==120
    assert saved['valid_optimized_cases']==1 and saved['best_optimized_execution_time']>0
    after=load_performance_results(tmp_path,'optimized_perf.yaml')
    assert len(after[0].metadata['benchmark_quality_case']['series']['candidate']['raw_samples_ms'])==100


def test_saved_quality_cannot_be_relabelled_pass_or_have_samples_removed(tmp_path):
    cases=as_test_cases(validate_pair(tmp_path,paired_fixture(tmp_path,source='b',spike=True)))
    save_performance_results(cases,tmp_path,'result.yaml')
    restored=load_performance_results(tmp_path,'result.yaml')
    restored[0].metadata['benchmark_quality_case']['status']='pass'
    with pytest.raises(ValueError):validate_case_quality(restored[0])
    restored=load_performance_results(tmp_path,'result.yaml')
    restored[0].metadata['benchmark_quality_case']['series']['candidate']['raw_samples_ms'].pop()
    with pytest.raises(ValueError):validate_case_quality(restored[0])


def test_unpaired_variable_portable_route_fails_closed_before_scoring():
    m=manifest(variable=True);sources={'source/kernel.py':'a'*64}
    request={'phase':'performance','request_id':'test','manifest_sha256':fingerprint(m),'source_sha256':sources}
    report=finalize_report({'schema_version':1,'status':'ok','request':request,'cases':[raw_row(m['cases'][0],[1.0]*100)]},m,request)
    with pytest.raises(ValueError,match='variable work requires'):portable_test_cases(report,m,request,sources)


@pytest.mark.parametrize('accepted',[None,0.0])
def test_final_writer_never_recovers_explicit_rejection_from_aggregate_times(tmp_path,accepted):
    from src.testcases import TestCaseResult
    baseline=[TestCaseResult('case',execution_time_ms=8.0,metadata={'benchmark_method':'cuda_graph'})]
    result={'pass_compilation':True,'pass_correctness':True,'average_speedup':accepted,
            'best_optimized_execution_time':2.0,'benchmark_method_consistent':True,'workload_consistent':True}
    evaluator.write_task_result(tmp_path,result,baseline,'test','test',create_plots=False)
    saved=yaml.safe_load((tmp_path/'task_result.yaml').read_text())
    assert saved['speedup_ratio']==accepted
    assert task_result_scoring(str(tmp_path))==120


def test_ordinary_native_quant_adapter_validates_and_preserves_both_raw_legs(monkeypatch):
    from src import native_baseline
    from src.quality_measurements import load_quality_measurements, quality_required
    from src.tools import trusted_native_eval as native
    from test_trusted_native_eval import report
    root=Path(__file__).resolve().parents[1]/'tasks/headkernel_sg520/deepseek-v4-pro__per_group_quant_fp8'
    config=yaml.safe_load((root/'config.yaml').read_text());m=json.loads((root/'cases.json').read_text())
    payload=report('performance',native.identities(root),m['cases'],2.0)
    original=native_baseline.read_task_file
    monkeypatch.setattr(native_baseline,'read_task_file',lambda r,p:json.dumps(payload).encode()
                        if p=='build/performance_report.json' else original(r,p))
    cases=load_quality_measurements(root,config)
    assert quality_required(config) and len(cases)==3
    for row in cases:
        proof=validate_case_quality(row)
        assert {len(s['raw_samples_ms']) for s in proof['series'].values()}=={100}
    quality=comparison_quality(cases,cases)
    assert quality['comparison_status']=='unchanged_source_control' and not quality['accepted_gain']


def test_direct_case_scoring_cannot_bypass_quality_or_drop_the_same_case_from_both_legs():
    from src.testcases import calculate_average_speedup
    from test_native_baseline_scoring import measured
    before=as_test_cases(measured(),is_baseline=True)
    changed=as_test_cases(measured(source_hash='e'*64,native_ms=4,port_ms=2,candidate_ms=2))
    assert calculate_average_speedup(before,changed)==2.0
    assert calculate_average_speedup(before,as_test_cases(measured()))==0.0
    with pytest.raises(ValueError,match='omitted declared manifest cases'):
        comparison_quality(before[:1],changed[:1])
    assert calculate_average_speedup(before[:1],changed[:1])==0.0


@pytest.mark.parametrize('scenario',['stable','spike','missing_report','stale_source'])
def test_framework_finalizer_independently_overrides_model_benchmark_pass(tmp_path,monkeypatch,scenario):
    from agents.task_validator.report_schema import finalize_report as finalize_validation, validation_report_is_complete
    from test_task_validator import _valid_raw_report
    from src import harness_guard
    m=manifest();(tmp_path/'cases.json').write_text(json.dumps(m));(tmp_path/'source').mkdir();(tmp_path/'build').mkdir()
    source=tmp_path/'source/kernel.py';source.write_text('VALUE=1\n')
    config={'source_file_path':['source/kernel.py'],'trusted_evaluation':{'schema_version':1}}
    (tmp_path/'config.yaml').write_text(yaml.safe_dump(config))
    request={'phase':'performance','request_id':'validation','manifest_sha256':fingerprint(m),
             'source_sha256':{'source/kernel.py':hashlib.sha256(source.read_bytes()).hexdigest()}}
    samples=[1.0]*100
    if scenario=='spike':samples[27]=10000.0
    report=finalize_report({'schema_version':1,'status':'ok','request':request,
                           'cases':[raw_row(m['cases'][0],samples)]},m,request)
    if scenario!='missing_report':(tmp_path/'build/performance_report.json').write_text(json.dumps(report))
    if scenario=='stale_source':source.write_text('VALUE=2\n')
    (tmp_path/'validation_report.yaml').write_text(yaml.safe_dump(_valid_raw_report()))
    monkeypatch.setattr(harness_guard,'describe_workspace_harness',lambda root:{
        'enforced_during_optimization':True,'protected_paths':['config.yaml']})
    final=finalize_validation(tmp_path,expected_task_name='hip2hip/example')
    assert validation_report_is_complete(tmp_path)
    check=final['checks']['benchmark_integrity']
    assert check['benchmark_quality_required'] is True
    assert check['benchmark_quality']['accepted_gain'] is False
    if scenario=='stable':
        assert final['overall_status']=='PASS' and check['case_count']==1
        assert check['benchmark_quality']['comparison_status']=='task_validation_only'
    else:
        assert final['overall_status']=='FAIL' and check['status']=='FAIL'
        assert check['benchmark_quality']['status']=='reject'


def _invent_singleton_cache_classes(cases):
    from src.benchmark_quality import assess_series
    for case in cases:
        proof=case.metadata['benchmark_quality_case']
        work_ids=['invented-singleton-'+str(i) for i in range(100)]
        proof['validated_work_ids']=work_ids
        for name,row in list(proof['series'].items()):
            raw=row['raw_samples_ms']
            proof['series'][name]={**assess_series(raw,work_ids=work_ids),'raw_samples_ms':raw}
        proof['status']='pass'


def test_review_work_class_bypass_roundtrip_cannot_turn_rejection_into_21_98x(tmp_path):
    """QUALITY-RELOAD-WORK-1: exact retained-array/singleton-label attack."""
    from src.testcases import calculate_average_speedup
    before=as_test_cases(validate_pair(tmp_path,paired_fixture(tmp_path,source='a')),is_baseline=True)
    args=paired_fixture(tmp_path,source='b')
    pair=args[0]['paired_reference_comparison']['cases'][0]
    pair['legs']['protected_reference']['samples_ms'][1]=1000000
    args[0]['cases'][0]['paired_reference']=deepcopy(pair)
    after=as_test_cases(validate_pair(tmp_path,args))
    original=metric_summary(before,after)
    assert original['benchmark_quality']['status']=='reject'
    assert original['raw_native_speedup_ratio']==pytest.approx(21.97998001998002)
    original_arrays=[deepcopy(c.metadata['benchmark_quality_case']['series']) for c in before+after]
    save_performance_results(before,tmp_path,'before.yaml');save_performance_results(after,tmp_path,'after.yaml')
    cached_before=load_performance_results(tmp_path,'before.yaml')
    cached_after=load_performance_results(tmp_path,'after.yaml')
    _invent_singleton_cache_classes(cached_before+cached_after)
    save_performance_results(cached_before,tmp_path,'before.yaml');save_performance_results(cached_after,tmp_path,'after.yaml')
    edited_before=load_performance_results(tmp_path,'before.yaml')
    edited_after=load_performance_results(tmp_path,'after.yaml')
    for case,original_series in zip(edited_before+edited_after,original_arrays):
        for name,row in case.metadata['benchmark_quality_case']['series'].items():
            assert row['raw_samples_ms']==original_series[name]['raw_samples_ms']
    with pytest.raises(ValueError,match='authoritative quality context is missing'):
        metric_summary(edited_before,edited_after)
    assert calculate_average_speedup(edited_before,edited_after)==0.0
    # Neither an edited cache nor an authentic-but-different receipt can serve
    # as its own reconstruction/authority context.
    assert load_performance_results(tmp_path,'before.yaml',validated_cases=edited_before)==[]
    assert load_performance_results(tmp_path,'before.yaml',validated_cases=before)==[]
    assert load_performance_results(tmp_path,'after.yaml',validated_cases=after)==[]
    # The same edit on live cases is checked against the immutable host record.
    live=deepcopy(before+after);_invent_singleton_cache_classes(live)
    with pytest.raises(ValueError,match='authoritative semantic validation'):
        metric_summary(live[:1],live[1:])


def test_review_case_coverage_bypass_roundtrip_cannot_accept_partial_2x(tmp_path):
    """QUALITY-RELOAD-CASES-2: edited cached declarations are not a manifest."""
    from src.testcases import calculate_average_speedup
    from test_native_baseline_scoring import measured
    before=as_test_cases(measured(),is_baseline=True)
    after=as_test_cases(measured(source_hash='e'*64,native_ms=2,port_ms=1,candidate_ms=1))
    assert metric_summary(before,after)['native_speedup_ratio']==2.0
    manifest_hash=before[0].metadata['benchmark_quality_case']['manifest_sha256']
    assert manifest_hash=='24d067f5aec344fc2fa6c6cb58c9e106755847988b00aa171097e9995d4b76a9'
    save_performance_results(before,tmp_path,'before.yaml');save_performance_results(after,tmp_path,'after.yaml')
    cached_before=load_performance_results(tmp_path,'before.yaml')[:1]
    cached_after=load_performance_results(tmp_path,'after.yaml')[:1]
    for case in cached_before+cached_after:
        case.metadata['benchmark_quality_case']['case_ids']=[case.test_case_id]
        assert case.metadata['benchmark_quality_case']['manifest_sha256']==manifest_hash
    save_performance_results(cached_before,tmp_path,'before.yaml');save_performance_results(cached_after,tmp_path,'after.yaml')
    edited_before=load_performance_results(tmp_path,'before.yaml')
    edited_after=load_performance_results(tmp_path,'after.yaml')
    with pytest.raises(ValueError,match='authoritative quality context is missing'):
        metric_summary(edited_before,edited_after)
    assert calculate_average_speedup(edited_before,edited_after)==0.0
    assert load_performance_results(tmp_path,'before.yaml',validated_cases=before)==[]
    # Passing only one authentic case also fails: its host context still owns
    # the full two-case manifest, independent of either supplied list.
    assert load_performance_results(tmp_path,'before.yaml',validated_cases=before[:1])==[]
    live_before=deepcopy(before[:1]);live_after=deepcopy(after[:1])
    for case in live_before+live_after:case.metadata['benchmark_quality_case']['case_ids']=[case.test_case_id]
    with pytest.raises(ValueError,match='authoritative semantic validation'):
        metric_summary(live_before,live_after)


def test_unedited_cache_is_diagnostic_without_context_and_exact_with_fresh_context(tmp_path):
    from src.testcases import calculate_average_speedup
    from test_native_baseline_scoring import measured
    before=as_test_cases(measured(),is_baseline=True)
    after=as_test_cases(measured(source_hash='e'*64,native_ms=2,port_ms=1,candidate_ms=1))
    save_performance_results(before,tmp_path,'before.yaml');save_performance_results(after,tmp_path,'after.yaml')
    plain_before=load_performance_results(tmp_path,'before.yaml');plain_after=load_performance_results(tmp_path,'after.yaml')
    assert len(plain_before)==len(plain_after)==2
    assert calculate_average_speedup(plain_before,plain_after)==0.0
    restored_before=load_performance_results(tmp_path,'before.yaml',validated_cases=before)
    restored_after=load_performance_results(tmp_path,'after.yaml',validated_cases=after)
    assert metric_summary(restored_before,restored_after)['native_speedup_ratio']==2.0
    assert '_benchmark_quality_binding' not in (tmp_path/'before.yaml').read_text()


def test_serialized_measured_dict_cannot_recreate_authority_from_cached_declarations(tmp_path):
    measured=validate_pair(tmp_path,paired_fixture(tmp_path,source='b'))
    assert as_test_cases(measured)
    cached=json.loads(json.dumps(measured))
    with pytest.raises(ValueError,match='freshly validated quality record'):
        as_test_cases(cached)


def test_cached_flags_cannot_opt_out_of_authority_checks(tmp_path):
    from src.testcases import calculate_average_speedup
    from test_native_baseline_scoring import measured
    before=as_test_cases(measured(),is_baseline=True)
    after=as_test_cases(measured(source_hash='e'*64,native_ms=2,port_ms=1,candidate_ms=1))
    save_performance_results(before,tmp_path,'before.yaml');save_performance_results(after,tmp_path,'after.yaml')
    for name in ['before.yaml','after.yaml']:
        data=yaml.safe_load((tmp_path/name).read_text())
        for row in data['test_cases']:
            for key in list(row):
                if key.startswith('benchmark_quality') or key=='baseline_kind':row.pop(key)
        (tmp_path/name).write_text(yaml.safe_dump(data))
    assert calculate_average_speedup(load_performance_results(tmp_path,'before.yaml'),
                                     load_performance_results(tmp_path,'after.yaml'))==0.0
    live=deepcopy(before+after)
    for case in live:
        case.metadata.pop('benchmark_quality_required')
        case.metadata.pop('baseline_kind')
    assert calculate_average_speedup(live[:2],live[2:])==0.0
