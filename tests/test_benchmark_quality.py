"""Retained real outlier, source controls, and genuine variable-work examples."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from src.benchmark_quality import assess_comparison, assess_series, gate_native_measurement

BEFORE = 'a' * 64
AFTER = 'b' * 64
FIXTURE = json.loads((Path(__file__).parent / 'fixtures/timing_quality/ds_quant_observed_outlier.json').read_text())


def case(reference=None, candidate=None, name='fixed'):
    return {'case_id': name, 'work_kind': 'fixed',
            'reference_samples_ms': reference if reference is not None else [2.0] * 100,
            'candidate_samples_ms': candidate if candidate is not None else [1.0] * 100}


def assess(cases, changed=True):
    return assess_comparison(cases, reference_source=BEFORE, candidate_source=AFTER if changed else BEFORE)


def test_actual_observed_68ms_sample_rejects_without_trimming_or_changing_raw_mean():
    entry=case(FIXTURE['reference_samples_ms'], FIXTURE['candidate_samples_ms'])
    untouched=deepcopy(entry)
    result=assess([entry])
    stats=result['cases'][0]['reference']['work_classes'][0]
    assert result['status']=='reject' and result['accepted_arithmetic_mean_speedup'] is None
    assert result['accepted_gain'] is False
    assert stats['reasons']==['dominant_extreme_replay']
    assert stats['max_ms']==68.0001449584961 and stats['sample_count']==100
    assert stats['mean_ms']==pytest.approx(0.7334084195271134)
    assert result['raw_arithmetic_mean_speedup']==pytest.approx(16.68180578185655)
    assert entry==untouched


def test_stable_changed_source_can_earn_gain():
    result=assess([case()])
    assert result['status']=='pass' and result['source_changed'] is True
    assert result['accepted_gain'] is True and result['accepted_arithmetic_mean_speedup']==2.0


def test_unchanged_source_cannot_earn_gain_even_with_stable_favorable_timings():
    result=assess([case()],changed=False)
    assert result['status']=='pass' and result['comparison_status']=='unchanged_source_control'
    assert result['raw_arithmetic_mean_speedup']==2.0
    assert result['accepted_arithmetic_mean_speedup'] is None and result['accepted_gain'] is False


def test_harness_only_change_is_not_executable_source_change():
    entry=case();entry['reference_package_sha256']='c'*64;entry['candidate_package_sha256']='d'*64
    result=assess_comparison([entry],reference_source={'source/kernel.cu':BEFORE},
                            candidate_source={'source/kernel.cu':BEFORE})
    assert not result['gain_eligible'] and result['raw_arithmetic_mean_speedup']==2.0


def test_extreme_fixed_work_two_regimes_reject_even_without_dominant_single_sample():
    result=assess([case([1.0]*50+[100.0]*50)])
    assert result['status']=='reject'
    assert 'extreme_replay_spread' in result['cases'][0]['reference']['work_classes'][0]['reasons']


def test_moderate_fixed_work_noise_is_not_a_catastrophic_failure():
    result=assess([case([1.0,1.5,2.0,2.5]*25,[0.8,1.2,1.6,2.0]*25)])
    assert result['status']=='pass'


def test_real_variable_work_can_have_a_rare_expensive_draw_without_false_rejection():
    # This would fail a naive fixed-work dispersion check. Both legs perform
    # the same validated work sequence; the rare high-work draw stays scored.
    left=[1.0]*99+[1000.0];right=[0.5]*99+[500.0]
    work=['short']*99+['long']
    entry={**case(left,right),'work_kind':'paired_variable',
           'reference_work_ids':work,'candidate_work_ids':list(work)}
    result=assess([entry])
    assert assess_series(left)['status']=='reject'
    assert result['status']=='pass' and result['accepted_arithmetic_mean_speedup']==2.0
    assert result['cases'][0]['reference']['sample_count']==100
    assert result['cases'][0]['reference']['raw_mean_ms']==10.99
    assert result['cases'][0]['reference']['singleton_work_classes']==1


def test_legitimate_variant_specific_speedups_are_not_compared_as_same_work():
    work=['small']*50+['large']*50
    entry={**case([1.0]*50+[1000.0]*50,[0.9]*50+[1.0]*50),
           'work_kind':'paired_variable','reference_work_ids':work,'candidate_work_ids':list(work)}
    result=assess([entry])
    assert result['status']=='pass' and result['accepted_gain'] is True


def test_variable_work_still_rejects_an_extreme_spike_within_a_repeated_work_class():
    work=['short']*90+['long']*10
    samples=[1.0]*89+[1000.0]+[10.0]*10
    entry={**case(samples,[0.5]*90+[5.0]*10),'work_kind':'paired_variable',
           'reference_work_ids':work,'candidate_work_ids':list(work)}
    result=assess([entry])
    assert result['status']=='reject' and not result['accepted_gain']


@pytest.mark.parametrize('mode',['missing','different','short','empty_label'])
def test_variable_work_requires_complete_paired_validated_classes(mode):
    entry={**case(),'work_kind':'paired_variable','reference_work_ids':['a']*100,
           'candidate_work_ids':['a']*100}
    if mode=='missing':entry.pop('reference_work_ids')
    elif mode=='different':entry['candidate_work_ids'][0]='b'
    elif mode=='short':entry['reference_work_ids']=entry['candidate_work_ids']=['a']*99
    else:entry['reference_work_ids']=entry['candidate_work_ids']=['']*100
    with pytest.raises(ValueError):assess([entry])


@pytest.mark.parametrize('samples',[[1.0]*99,[1.0]*101,[True]*100,[0.0]*100,[float('inf')]*100])
def test_incomplete_or_invalid_samples_fail_closed(samples):
    with pytest.raises(ValueError):assess([case(samples)])


@pytest.mark.parametrize('source',[None,{},'package-name',{'source/kernel.cu':'bad'}])
def test_missing_executable_source_identity_fails_closed(source):
    with pytest.raises(ValueError):assess_comparison([case()],reference_source=source,candidate_source=AFTER)


def test_one_bad_case_rejects_the_whole_comparison():
    result=assess([case(name='first'),case(FIXTURE['reference_samples_ms'],name='second'),case(name='third')])
    assert len(result['cases'])==3 and result['status']=='reject'
    assert result['accepted_arithmetic_mean_speedup'] is None
    assert all(r['reference']['sample_count']==r['candidate']['sample_count']==100 for r in result['cases'])


def native_fixture(changed=False, outlier=False):
    reference=[];candidate=[];rows=[]
    for index in range(3):
        left=FIXTURE['reference_samples_ms'] if outlier and index==1 else [2.0]*100
        right=FIXTURE['candidate_samples_ms'] if outlier and index==1 else [1.0]*100
        a=sum(left)/100;b=sum(right)/100
        rows.append({'test_case_id':str(index),'reference_ms':a,'candidate_ms':b,'speedup':a/b})
        reference.append({'case_id':str(index),'timings':{'candidate_native':{'samples_ms':list(left)},'production_native':{'samples_ms':[1.0]*100}}})
        candidate.append({'case_id':str(index),'timings':{'candidate_native':{'samples_ms':list(right)},'production_native':{'samples_ms':[1.0]*100}}})
    m={'status':'measured','reference_source_sha256':BEFORE,'candidate_source_sha256':AFTER if changed else BEFORE,
       'cases':rows,'arithmetic_mean_speedup':sum(r['speedup'] for r in rows)/3}
    return m,{'paired_cases':reference},{'paired_cases':candidate}


def test_native_adapter_preserves_full_diagnostics_and_suppresses_invalid_accepted_fields():
    args=native_fixture(changed=True,outlier=True);untouched=deepcopy(args)
    result=gate_native_measurement(*args)
    assert args==untouched and result['status']=='rejected_timing_quality'
    assert result['arithmetic_mean_speedup'] is None and result['raw_arithmetic_mean_speedup']==args[0]['arithmetic_mean_speedup']
    assert all(r['speedup'] is None and r['raw_speedup']>0 for r in result['cases'])
    assert [(r['reference_ms'],r['candidate_ms']) for r in result['cases']]==[(r['reference_ms'],r['candidate_ms']) for r in args[0]['cases']]


def test_native_adapter_keeps_stable_same_source_as_control_without_speedup():
    result=gate_native_measurement(*native_fixture())
    assert result['status']=='measured' and result['arithmetic_mean_speedup'] is None
    assert result['raw_arithmetic_mean_speedup']==2.0 and result['gain_eligible'] is False


def test_unstable_production_diagnostic_cannot_be_dropped_to_salvage_a_comparison():
    args=native_fixture(changed=True)
    args[1]['paired_cases'][2]['timings']['production_native']['samples_ms']=[1.0]*99+[1000.0]
    result=gate_native_measurement(*args)
    assert result['status']=='rejected_timing_quality' and result['accepted_gain'] is False


def test_adapter_rejects_partial_case_sets_before_quality_can_accept():
    args=native_fixture(changed=True);args[2]['paired_cases'].pop()
    with pytest.raises(ValueError):gate_native_measurement(*args)


@pytest.mark.parametrize('change',['failed_status','changed_mean'])
def test_quality_gate_cannot_launder_failed_or_forged_measurement(change):
    args=native_fixture(changed=True)
    if change=='failed_status':args[0]['status']='failed'
    else:args[0]['cases'][0]['reference_ms']=0.000001
    with pytest.raises(ValueError):gate_native_measurement(*args)


@pytest.mark.parametrize('mode',['stable_same_source','stable_changed_source','unstable_changed_source'])
def test_native_retest_finishes_all_six_phases_before_gate_and_keeps_every_report(tmp_path,monkeypatch,mode):
    import hashlib
    import shutil
    from src.tools import trusted_native_eval as native
    from test_trusted_native_eval import report
    cases=[{'case_id':str(i),'shape':[1,i+1],'trace_call_count':1} for i in range(3)]
    image='example@sha256:'+'d'*64
    agent=tmp_path/'agent';agent.mkdir();candidate=agent/'kernel.cu'
    candidate.write_bytes(b'stock' if mode=='stable_same_source' else b'changed')
    repo=tmp_path/'repo';repo.mkdir();calls=[]
    def extract(repo,commit,task_path,destination):
        (destination/'source').mkdir(parents=True)
        (destination/native.SOURCE).write_bytes(b'stock')
    def identity(task):
        digest=hashlib.sha256((task/native.SOURCE).read_bytes()).hexdigest()
        return {'package_sha256':digest,'source_tree_sha256':digest}
    def run(image,task,build,render_device,phase,log,timeout,jit_source=None):
        calls.append((task.name,phase))
        payload=report(phase,identity(task),cases,2.0 if task.name=='reference' else 1.0)
        if phase=='performance' and task.name=='reference' and mode=='unstable_changed_source':
            values=FIXTURE['reference_samples_ms']
            timing={'samples_ms':list(values),'mean_ms':sum(values)/100,'min_ms':min(values),'max_ms':max(values)}
            payload['paired_cases'][1]['timings']['candidate_native']=timing
            payload['test_cases'][1]['execution_time_ms']=timing['mean_ms']
        return payload
    monkeypatch.setattr(native,'extract_task',extract)
    monkeypatch.setattr(native,'validate_contract',lambda task:(image,{'cases':cases}))
    monkeypatch.setattr(native,'validate_candidate',lambda *a:None)
    monkeypatch.setattr(native,'identities',identity)
    monkeypatch.setattr(native,'copy_payload',lambda a,b:shutil.copytree(a,b))
    monkeypatch.setattr(native,'seed_image_cache',lambda *a,**k:{'complete_parity':True})
    monkeypatch.setattr(native,'run_mode',run)
    monkeypatch.setattr(native.subprocess,'check_output',lambda *a,**k:json.dumps([{'RepoDigests':[image],'Id':'sha256:'+'d'*64}]))
    output=tmp_path/'output'
    result=native.trusted_retest(repo=repo,commit='1'*40,task_path='task',candidate=candidate,
        agent_workspace=agent,output=output,render_device='/dev/dri/renderD128',scratch_dir=tmp_path/'scratch')
    assert calls==[(leg,phase) for leg in ['reference','candidate'] for phase in native.MODES]
    assert all((output/(leg+'_'+phase+'.json')).is_file() for leg,phase in calls)
    assert (output/'trusted_measurement.json').is_file()
    if mode=='unstable_changed_source':
        assert result['status']=='rejected_timing_quality' and result['arithmetic_mean_speedup'] is None
        retained=json.loads((output/'reference_performance.json').read_text())
        assert retained['paired_cases'][1]['timings']['candidate_native']['samples_ms']==FIXTURE['reference_samples_ms']
    elif mode=='stable_same_source':
        assert result['status']=='measured' and not result['accepted_gain'] and result['arithmetic_mean_speedup'] is None
    else:
        assert result['accepted_gain'] and result['arithmetic_mean_speedup']==2.0
