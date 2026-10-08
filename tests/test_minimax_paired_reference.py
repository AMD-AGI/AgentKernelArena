"""CPU graph lifecycle and shared-consumer compatibility, not GPU validation."""
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import types

import pytest
from paired_cpu_backend import run_cpu_pair, ROOT

SNAPSHOT = Path(os.environ['MINIMAX_PAIRED_CONSUMER_SNAPSHOT']) if os.environ.get('MINIMAX_PAIRED_CONSUMER_SNAPSHOT') else ROOT/'src'


def consumer():
    package='minimax_consumer_snapshot'
    if package+'.native_baseline' in sys.modules:
        return sys.modules[package+'.native_baseline']
    module=types.ModuleType(package);module.__path__=[str(SNAPSHOT)];sys.modules[package]=module
    for name in ('task_contract','testcases','native_baseline'):
        spec=importlib.util.spec_from_file_location(package+'.'+name,SNAPSHOT/(name+'.py'))
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def run_consumer_pair(seed=91):
    value=os.environ.get('MINIMAX_PAIRED_INPUT_FIXTURE_TASK')
    if not value:
        pytest.skip('Set MINIMAX_PAIRED_INPUT_FIXTURE_TASK to a staged task with protected control fixtures')
    task=Path(value)
    manifest=json.loads((task/'cases.json').read_text())
    case=manifest['cases'][0]
    manifest['cases']=[case]
    manifest['capture']['required_case_ids']=[case['case_id']]
    manifest['capture']['target_calls']=manifest['capture']['represented_calls']=case['occurrences']
    provenance=json.loads((task/'provenance/PAIRED-REFERENCE.json').read_text())
    import paired_input_validation
    signatures=paired_input_validation.reconstruct_inputs(task,manifest,{'challenge_seed':seed},provenance)[case['case_id']]
    source={name:hashlib.sha256((task/name).read_bytes()).hexdigest() for name in provenance['reference_source_sha256']}
    result=run_cpu_pair(seed,task=task,signatures=signatures,source_override=source)
    result.fixture_task=task
    result.fixture_provenance=provenance
    return result


def comparison(result):
    row=result.row; request=result.request
    provenance={'baseline_kind':'protected_reference','reference_source_sha256':result.identities['reference']['source_sha256'],
                'gpu_binding':result.identities['reference']['gpu_binding'],
                'input_signature_fields':{row['case']['case_id']:list(row['realized_input_signatures'][0])},
                'input_receipt_contracts':{row['case']['case_id']:{'kind':'task_local_paired_inputs_v1'}}}
    if hasattr(result,'fixture_provenance'):
        for key in ('paired_input_validator','input_receipt_contract'):
            provenance[key]=deepcopy(result.fixture_provenance[key])
    provenance_sha=hashlib.sha256(json.dumps(provenance,sort_keys=True).encode()).hexdigest()
    record={'schema_version':1,'schema':'paired-reference-comparison-v1','status':'ok',
        'score_input':True,'diagnostic_only':False,'baseline_kind':'protected_reference',
        'request':request,'source_hashes':request['source_sha256'],'manifest_sha256':request['manifest_sha256'],
        'runtime_image':result.manifest['runtime_image'],'native_source_manifest_sha256':provenance_sha,
        'challenge_seed':request['challenge_seed'],'cases':[row['paired_reference']],
        'anti_cheat_attestation':False,'fresh_trusted_host_retest_required':True}
    report={'schema_version':1,'status':'ok','request':request,'cases':[row],
        'paired_reference_comparison':record,'compiled_specializations':[{
            'case_id':row['case']['case_id'],'candidate_binding':result.identities['candidate'],
            'reference_binding':result.identities['reference'],'invoked_and_synchronized':True}]}
    policy={'schema_version':2,'kind':'protected_reference','native_source_manifest':'provenance/PAIRED-REFERENCE.json'}
    return report,policy,provenance,provenance_sha


def validate(result,report=None):
    built,policy,provenance,sha=comparison(result)
    return consumer().validate_paired_measurements(report or built,result.manifest,result.request,
        result.request['source_sha256'],sha,policy,provenance,task_root=getattr(result,'fixture_task',None))


def test_real_producer_has_symmetric_graph_setup_and_110_deferred_pairs():
    result=run_cpu_pair()
    assert result.row['samples_ms']==[2.0]*100
    assert result.row['paired_reference']['legs']['protected_reference']['samples_ms']==[3.0]*100
    assert result.events.count('candidate_replay')==result.events.count('reference_replay')==110
    for role in ('candidate','reference'):
        assert [v['capture'] for v in result.setup[role]]==[False,False,False,True]
        assert len({v['stream'] for v in result.setup[role]})==1
    assert result.setup['candidate'][0]['stream']!=result.setup['reference'][0]['stream']
    assert result.events.index('candidate_snapshot')<result.events.index('reference_setup_replay')
    starts=[i for i,e in enumerate(result.events) if e=='prepare']+[len(result.events)]
    for start,end in zip(starts,starts[1:]):
        events=result.events[start:end]
        assert events.index('candidate_snapshot')<events.index('reference_restore')
        assert events.index('candidate_immutable')<events.index('reference_replay')
        assert events.index('reference_snapshot')<events.index('reference_clear')
        assert events.index('reference_clear')<events.index('independent_math_and_compare')
    assert result.outputs['reference'].storage.scrubbed


@pytest.mark.parametrize('failure',['candidate_input','restore','reference_input','compare','reference_setup'])
def test_failure_paths_scrub_reference_outputs(failure):
    with pytest.raises(ValueError):run_cpu_pair(failure=failure)


@pytest.mark.parametrize('task_name', ['minimax-m3__decode_score_kernel', 'minimax-m3__gqa_share_sparse_decode_kernel'])
@pytest.mark.parametrize('fail_at', [1, 2, 4])
@pytest.mark.parametrize('failure_kind', ['launch_count', 'launch_contract'])
def test_real_operator_scrubs_new_reference_output_before_failed_attestation(task_name, fail_at, failure_kind):
    from paired_cpu_backend import run_real_operator_failure
    result = run_real_operator_failure(ROOT/'tasks/headkernel'/task_name, fail_at, failure_kind)
    assert result['all_reference_storage_bytes_scrubbed']


@pytest.mark.parametrize('task_name', ['minimax-m3__decode_score_kernel', 'minimax-m3__gqa_share_sparse_decode_kernel'])
@pytest.mark.parametrize('fail_at', [1, 2])
def test_real_operator_preserves_original_failure_if_reference_cleanup_raises(task_name, fail_at):
    from paired_cpu_backend import run_real_operator_failure
    result = run_real_operator_failure(ROOT/'tasks/headkernel'/task_name, fail_at, cleanup_failure=True)
    assert result['original_attestation_exception_preserved']


@pytest.mark.parametrize('task_name', ['minimax-m3__decode_score_kernel', 'minimax-m3__gqa_share_sparse_decode_kernel'])
def test_reference_output_ownership_does_not_change_candidate_invocation(task_name):
    from paired_cpu_backend import run_real_operator_failure
    result = run_real_operator_failure(ROOT/'tasks/headkernel'/task_name, reference=False)
    assert not result['all_reference_storage_bytes_scrubbed']


def test_real_cpu_storage_scrub_covers_padding_and_deduplicates_aliases():
    import torch
    import minimax_paired
    import paired_reference
    storage=torch.arange(64,dtype=torch.uint8)
    calls=[]
    def raw(value):
        calls.append(value.untyped_storage().data_ptr())
        return minimax_paired.raw_storage(value)
    paired_reference.clear_outputs({'leaves':minimax_paired.leaves,'raw_storage':raw},
                                   (None,storage[8:16],storage[24:32]))
    assert len(calls)==1
    assert torch.equal(storage,torch.full_like(storage,0xAA))


def test_real_cpu_storage_cannot_alias_across_candidate_and_reference():
    import torch
    import minimax_paired
    import paired_reference
    shared=torch.empty(8)
    with pytest.raises(ValueError,match='storage aliases'):
        paired_reference.assert_disjoint_legs({'leaves':minimax_paired.leaves},
            {'input':shared},torch.empty(2),{'input':shared},torch.empty(2))


def test_exact_shared_consumer_accepts_producer_and_suppresses_unpaired_secondary():
    a,b=run_consumer_pair(91),run_consumer_pair(123)
    c=consumer()
    left=c.as_test_cases(validate(a),is_baseline=True)
    right=c.as_test_cases(validate(b),is_baseline=False)
    metric=c.metric_summary(left,right)
    assert metric['native_speedup_ratio']==1.5
    assert metric['baseline_kind']=='protected_reference'
    assert metric['port_to_port_speedup_ratio'] is None
    assert metric['secondary_comparison_status']=='unpaired_workload'
    assert metric['production_kernel_improvement'] is False


@pytest.mark.parametrize('change',[
    lambda r:r['paired_reference_comparison'].update(challenge_seed=-1),
    lambda r:r['paired_reference_comparison']['cases'][0].update(reference_output_clear_calls=109),
    lambda r:r['paired_reference_comparison']['cases'][0]['legs']['protected_reference']['samples_ms'].pop(),
    lambda r:r['paired_reference_comparison']['cases'][0]['input_schedule']['measured_inputs'][0].update(input_seed=0),
    lambda r:r['paired_reference_comparison']['cases'][0]['legs']['protected_reference'].update(paired_schedule_sha256='0'*64),
    lambda r:r['paired_reference_comparison']['cases'][0]['graph_setup'].update(warmup_invocations_per_leg=2),
])
def test_consumer_rejects_unpaired_or_incomplete_evidence(change):
    result=run_consumer_pair();validate(result);report=comparison(result)[0];change(report)
    with pytest.raises(ValueError):validate(result,report)


def test_consumer_reconstructs_rehashed_address_and_wrong_observed_variant():
    from collections import Counter
    from evaluation_contract import fingerprint
    from workload_controls import RecordedControls
    result = run_consumer_pair()
    validate(result)
    for corruption in ('observed_variant', 'seq_lens', 'req_to_token'):
        report = deepcopy(comparison(result)[0])
        outer = report['cases'][0]
        pair = report['paired_reference_comparison']['cases'][0]
        item = pair['input_schedule']['measured_inputs'][0]
        if corruption == 'observed_variant':
            distribution = RecordedControls(result.manifest['cases'][0])
            variant = next(value for value in distribution.variant_ids if value != item['variant_id'])
            item['variant_id'] = item['actual_tensor_controls_sha256'] = variant
            receipt = outer['workload_control_sampling']
            receipt['measured_variant_ids'][0] = variant
            selected = receipt['warmup_variant_ids'] + receipt['measured_variant_ids']
            receipt['measured_histogram'] = dict(sorted(Counter(receipt['measured_variant_ids']).items()))
            receipt['schedule_fingerprint'] = fingerprint({'case_id': pair['case_id'],
                'seed': result.request['challenge_seed'], 'distribution_sha256': distribution.distribution_sha256,
                'selected': selected})
            receipt['measured_variant_count'] = len(set(receipt['measured_variant_ids']))
            receipt['untimed_variant_ids'] = sorted(set(distribution.variant_ids)-set(receipt['measured_variant_ids']))
        else:
            item['addressing'][corruption]['sha256'] = '0'*64
        outer['realized_input_signatures'][10] = deepcopy(item)
        digest = fingerprint(pair['input_schedule'])
        for leg in pair['legs'].values():
            leg['paired_schedule_sha256'] = digest
        outer['paired_schedule_sha256'] = digest
        outer['paired_reference'] = deepcopy(pair)
        with pytest.raises(ValueError, match='draws differ|protected reconstruction'):
            validate(result, report)
