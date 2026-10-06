"""CPU simulation of actual paired callbacks and fail-closed scoring consumers."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from contextlib import contextmanager

import pytest
import yaml

from src import evaluator
from src.native_baseline import as_test_cases, load_native_measurements, metric_summary, validate_paired_measurements
from src.task_contract import checked_replays, finalize_report, fingerprint
from src.testcases import load_performance_results, save_performance_results

TASKS=Path(__file__).resolve().parents[1]/'tasks/headkernel'
TASK=TASKS/'deepseek-v4-pro__moe_stage2_down_proj_reduce_opus_a8w4'
sys.path.insert(0,str(TASK/'ut'))
import paired_reference as pair
from snapshots import raw_storage


def fixture_manifest():
    tensor={'role':'input','shape':[1],'strides':[1],'storage_offset':0,'dtype':'float32','device_type':'cuda'}
    case={'case_id':'cpu-paired-example','occurrences':1,'calls_per_sample':1,
          'fixture':{'path':'fixtures/example.json','sha256':'f'*64},
          'tensors':{'a':tensor,'output':{**tensor,'role':'output'}},'scalars':{}}
    manifest={'schema_version':1,'runtime_image':'image@sha256:'+'b'*64,'cases':[case],'tolerance':0.02,
        'measurement':{'method':'cuda_graph','warmup_iterations':10,'benchmark_iterations':100,
            'correctness_seeds':[42,43,44],'negative_controls':['no_op','wrong_output'],
            'refresh_inputs':'each_replay','initialize_outputs':'each_replay','validate_outputs':'each_replay'}}
    return manifest,case


def simulate(monkeypatch, *, attack=None, seed=123, provided_out=False):
    import torch
    events=[];state={'stream':0,'capture':None,'phase':'setup','last_leg':None};stream_ids=[]
    class Stream:
        def __init__(self):self.ident=len(stream_ids)+1;stream_ids.append(self)
        def wait_stream(self,other):events.append(('wait',self.ident,other.ident))
        def synchronize(self):events.append(('stream_sync',self.ident))
    default=type('DefaultStream',(),{'ident':0,'wait_stream':lambda self,other:events.append(('wait',0,other.ident))})()
    @contextmanager
    def stream_context(stream):
        previous=state['stream'];state['stream']=stream.ident
        try:yield
        finally:state['stream']=previous
    class Graph:
        def replay(self):events.append(('replay',self.leg));self.call()
    @contextmanager
    def capture(graph,stream):
        with stream_context(stream):
            state['capture']=graph
            try:yield
            finally:state['capture']=None
    class Event:
        def __init__(self,**kwargs):pass
        def record(self):events.append(('event',))
        def synchronize(self):pass
        def elapsed_time(self,other):return 2.0 if state['last_leg']=='candidate' else 3.0
    monkeypatch.setattr(torch.cuda,'Stream',Stream);monkeypatch.setattr(torch.cuda,'current_stream',lambda:default)
    monkeypatch.setattr(torch.cuda,'stream',stream_context);monkeypatch.setattr(torch.cuda,'graph',capture)
    monkeypatch.setattr(torch.cuda,'CUDAGraph',Graph);monkeypatch.setattr(torch.cuda,'Event',Event)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    inputs={'a':torch.ones(1)};reference_inputs={'a':torch.ones(1)}
    output=torch.zeros(1);reference_output=torch.zeros(1)
    if provided_out:inputs['out']=output;reference_inputs['out']=reference_output
    def leaves(value):
        if torch.is_tensor(value):return [value]
        if isinstance(value,dict):return sum((leaves(v) for v in value.values()),[])
        if isinstance(value,(list,tuple)):return sum((leaves(v) for v in value),[])
        return []
    def snapshots(values):return {k:raw_storage(v).clone() for k,v in values.items()}
    def restore(values,truth):
        events.append(('restore','reference' if values is reference_inputs else 'candidate'))
        for k,v in truth.items():raw_storage(values[k]).copy_(v)
    def check(values,truth):
        leg='reference' if values is reference_inputs else 'candidate';events.append(('input_check',leg))
        for k,v in truth.items():
            if k!='out' and not torch.equal(raw_storage(values[k]),v):raise AssertionError('input mutation')
    def clone(value):
        if value is None:return None
        events.append(('snapshot','reference' if value is reference_output else 'candidate'))
        return value.clone()
    def candidate(**kwargs):
        events.append(('invoke','candidate',state['phase'],state['stream']));state['last_leg']='candidate'
        if attack=='mutated_input':kwargs['a'].add_(1)
        if attack!='no_op':output.copy_(kwargs['a']*(0 if attack=='wrong_output' else 2))
        return output
    def reference(**kwargs):
        events.append(('invoke','reference',state['phase'],state['stream']));state['last_leg']='reference'
        reference_output.copy_(kwargs['a']*2)
        if attack=='reference_input':kwargs['a'].add_(1)
        return reference_output
    def invoke(fn,values):
        if state['capture'] is not None:
            state['capture'].call=lambda:fn(**values)
            state['capture'].leg='candidate' if fn is candidate else 'reference'
        return fn(**values)
    comparisons=[]
    def compare(actual,expected,values,tolerance,*,expected_inputs):
        assert bool((raw_storage(reference_output)==0xAA).all()),'reference golden remains resident'
        comparisons.append((actual.clone(),expected.clone()))
        if not torch.equal(actual,expected):raise AssertionError('wrong output')
    api={'leaves':leaves,'storage_snapshots':snapshots,'restore_storages':restore,
         'assert_immutable_inputs':check,'cpu_clone':clone,'invoke':invoke,'runtime_abi':lambda *args:({},{}),
         'observe_case':lambda case,*args:case,'checked_replays':checked_replays,'compare_native_outputs':compare}
    manifest,case=fixture_manifest();truth=snapshots(inputs)
    def reset(input_seed):
        state['phase']='checked';events.append(('prepare',input_seed))
        assert bool((raw_storage(reference_output)==0xAA).all()),'reference not cleared before candidate'
        restore(inputs,truth);inputs['a'].fill_(input_seed)
        return snapshots(inputs)
    source={'source/kernel.py':'a'*64}
    identity={'leg':'candidate','source_sha256':source,'gpu_binding':'hip','module':'private_candidate'}
    ref_identity={**identity,'leg':'reference','module':'private_reference'}
    request={'schema_version':1,'request_id':'cpu-'+str(seed),'phase':'performance','manifest_sha256':fingerprint(manifest),
             'package_sha256':'c'*64,'source_sha256':source,'challenge_seed':seed}
    row=pair.paired_performance(api,case,manifest,request,inputs,reference_inputs,candidate,reference,truth,
        torch.zeros(1) if provided_out else None,identity,ref_identity,reset)
    report=finalize_report({'schema_version':1,'status':'ok','request':request,'compiled':True,
        'compiled_specializations':[{'case_id':case['case_id'],'candidate_binding':identity,
            'reference_binding':ref_identity,'invoked_and_synchronized':True}], 'cases':[row]},manifest,request)
    return report,manifest,request,events,comparisons


def attach(tmp_path,report,manifest,request,kind='native_production'):
    (tmp_path/'provenance').mkdir(exist_ok=True)
    provenance={'baseline_kind':kind,'reference_source_sha256':request['source_sha256'],'gpu_binding':'hip',
                'input_signature_fields':{case['case_id']:['fixture','input_seed'] for case in manifest['cases']},
                'input_receipt_contracts':{case['case_id']:{'kind':'fixed_fixture_v1'} for case in manifest['cases']}}
    path=tmp_path/'provenance/PAIRED-REFERENCE.json';path.write_text(json.dumps(provenance))
    pair.attach_comparison(tmp_path,report,manifest,request)
    policy={'schema_version':2,'kind':kind,'native_source_manifest':'provenance/PAIRED-REFERENCE.json'}
    return policy,provenance,hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize('provided_out',[False,True])
def test_real_callback_order_counts_stream_capture_and_clear(monkeypatch,provided_out):
    report,manifest,request,events,comparisons=simulate(monkeypatch,provided_out=provided_out)
    row=report['cases'][0];paired=row['paired_reference']
    assert row['samples_ms']==[2.0]*100
    assert paired['legs']['protected_reference']['samples_ms']==[3.0]*100
    assert len(comparisons)==110
    for leg in ['candidate','reference']:
        calls=[e for e in events if e[:2]==('invoke',leg)]
        assert len(calls)==114
        setup=[e for e in calls if e[2]=='setup'];assert len(setup)==4
        assert len({e[3] for e in setup})==1 and setup[0][3]!=0
        assert len([e for e in calls if e[2]=='checked'])==110
    assert {e[3] for e in events if e[:2]==('invoke','candidate') and e[2]=='setup'} != {
        e[3] for e in events if e[:2]==('invoke','reference') and e[2]=='setup'}
    starts=[i for i,e in enumerate(events) if e[0]=='prepare']+[len(events)]
    for left,right in zip(starts,starts[1:]):
        frame=events[left:right]
        assert frame.index(('snapshot','candidate')) < frame.index(('restore','reference'))
        assert frame.index(('input_check','candidate')) < frame.index(('replay','reference'))


@pytest.mark.parametrize('attack',['no_op','wrong_output','mutated_input','reference_input'])
def test_invalid_work_fails_closed(monkeypatch,attack):
    with pytest.raises(AssertionError):simulate(monkeypatch,attack=attack)


@pytest.mark.parametrize('kind',['native_production','protected_reference'])
def test_generic_primary_kind_and_unpaired_secondary_suppression(tmp_path,monkeypatch,kind):
    report,manifest,request,_,_=simulate(monkeypatch,seed=123)
    policy,provenance,digest=attach(tmp_path,report,manifest,request,kind)
    first=validate_paired_measurements(report,manifest,request,request['source_sha256'],digest,policy,provenance)
    report2,manifest2,request2,_,_=simulate(monkeypatch,seed=456)
    policy2,provenance2,digest2=attach(tmp_path,report2,manifest2,request2,kind)
    second=validate_paired_measurements(report2,manifest2,request2,request2['source_sha256'],digest2,policy2,provenance2)
    before=as_test_cases(first,is_baseline=True);after=as_test_cases(second)
    summary=metric_summary(before,after)
    assert summary['native_speedup_ratio']==1.5 and summary['port_to_port_speedup_ratio'] is None
    assert summary['secondary_comparison_status']=='unpaired_workload'
    assert summary['production_kernel_improvement']==(kind=='native_production')
    assert metric_summary(before,as_test_cases(first))['port_to_port_speedup_ratio']==1.0
    save_performance_results(before,tmp_path,'baseline.yaml')
    reloaded=load_performance_results(tmp_path,'baseline.yaml')
    assert reloaded[0].metadata['paired_input_schedule_sha256']==before[0].metadata['paired_input_schedule_sha256']
    monkeypatch.setattr(evaluator,'evaluate_compilation',lambda *a,**k:(True,None))
    monkeypatch.setattr(evaluator,'evaluate_correctness',lambda *a,**k:(True,None))
    monkeypatch.setattr(evaluator,'measure_performance',lambda *a,**k:after)
    result=evaluator.evaluate_kernel(tmp_path,{'scoring_baseline':policy,'trusted_evaluation':{'schema_version':1}},before)
    evaluator.write_task_result(tmp_path,result,before,'cpu-paired','test',create_plots=False)
    final=yaml.safe_load((tmp_path/'task_result.yaml').read_text())
    assert final['baseline_kind']==kind and final['speedup_ratio']==1.5
    assert final['production_kernel_improvement']==(kind=='native_production')
    assert final['port_to_port_speedup_ratio'] is None


@pytest.mark.parametrize('attack',['missing','request','kind','warmups','seed','schedule','candidate_samples',
    'reference_samples','reference_source','alias_binding','clears','extra_oracle','snapshot_order','capture_stream',
    'host_timing','anti_cheat_claim','missing_raw_controls'])
def test_paired_evidence_mutations_rejected(tmp_path,monkeypatch,attack):
    report,manifest,request,_,_=simulate(monkeypatch)
    policy,provenance,digest=attach(tmp_path,report,manifest,request)
    paired=report['paired_reference_comparison'];row=paired['cases'][0]
    if attack=='missing':report.pop('paired_reference_comparison')
    if attack=='request':paired['request']={**request,'request_id':'foreign'}
    if attack=='kind':paired['baseline_kind']='protected_reference'
    if attack=='warmups':row['input_schedule']['warmup_inputs'].pop()
    if attack=='seed':row['input_schedule']['measured_inputs'][0]['input_seed']+=1
    if attack=='schedule':row['legs']['protected_reference']['paired_schedule_sha256']='0'*64
    if attack=='candidate_samples':row['legs']['candidate_port']['samples_ms']=([9.0]*100)
    if attack=='reference_samples':row['legs']['protected_reference']['samples_ms'].pop()
    if attack=='reference_source':row['reference_binding']['source_sha256']={'source/kernel.py':'f'*64}
    if attack=='alias_binding':row['reference_binding']['module']=row['candidate_binding']['module']
    if attack=='clears':row['reference_output_clear_calls']-=1
    if attack=='extra_oracle':row['reference_reused_as_oracle']=False
    if attack=='snapshot_order':row['candidate_snapshot_before_reference']=False
    if attack=='capture_stream':row['graph_setup']['capture_on_warmed_stream']=False
    if attack=='host_timing':row['legs']['protected_reference']['benchmark_method']='host'
    if attack=='anti_cheat_claim':paired['anti_cheat_attestation']=True
    if attack=='missing_raw_controls':
        for item in row['input_schedule']['warmup_inputs']+row['input_schedule']['measured_inputs']:item.pop('fixture')
        for leg in row['legs'].values():leg['paired_schedule_sha256']=fingerprint(row['input_schedule'])
        report['cases'][0]['paired_schedule_sha256']=fingerprint(row['input_schedule'])
    with pytest.raises(ValueError):validate_paired_measurements(report,manifest,request,request['source_sha256'],digest,policy,provenance)


def test_task_reference_kinds_counts_and_original_policies():
    for name,count,kind in [('moe_stage1_grouped_gemm_silu_flydsl',10,'protected_reference'),
                           ('moe_stage1_grouped_gemm_silu_opus_a8w4',4,'native_production'),
                           ('moe_stage2_down_proj_reduce_opus_a8w4',14,'native_production')]:
        root=TASKS/('deepseek-v4-pro__'+name);config=yaml.safe_load((root/'config.yaml').read_text())
        manifest=json.loads((root/'cases.json').read_text());base=json.loads((root/'provenance/BASE-CASES.json').read_text())
        assert config['scoring_baseline']['kind']==kind and len(manifest['cases'])==count
        assert manifest['cases'][:len(base['cases'])]==base['cases']
        assert manifest['measurement']==base['measurement'] and manifest['tolerance']==base['tolerance']


@pytest.mark.parametrize('name',['moe_stage1_grouped_gemm_silu_flydsl','moe_stage1_grouped_gemm_silu_opus_a8w4',
                                  'moe_stage2_down_proj_reduce_opus_a8w4'])
def test_real_task_schema_and_source_provenance_accept_complete_cpu_rows(name):
    root=TASKS/('deepseek-v4-pro__'+name);config=yaml.safe_load((root/'config.yaml').read_text())
    manifest=json.loads((root/'cases.json').read_text());prov=json.loads((root/'provenance/PAIRED-REFERENCE.json').read_text())
    registry=json.loads((root/'provenance/WORK-DISTRIBUTIONS.json').read_text())
    spec=importlib.util.spec_from_file_location('actual_schedule_recipe',root/'ut/work_distribution.py')
    recipe=importlib.util.module_from_spec(spec);spec.loader.exec_module(recipe)
    source={path:hashlib.sha256((root/path).read_bytes()).hexdigest() for path in config['source_file_path']}
    request={'schema_version':1,'request_id':'cpu-real-contract','phase':'performance','manifest_sha256':fingerprint(manifest),
             'package_sha256':'c'*64,'source_sha256':source,'challenge_seed':1200}
    cb={'leg':'candidate','source_sha256':source,'gpu_binding':prov['gpu_binding'],'module':'private_candidate'}
    rb={'leg':'reference','source_sha256':prov['reference_source_sha256'],'gpu_binding':prov['gpu_binding'],'module':'private_reference'}
    rows=[];compiled=[]
    for case in manifest['cases']:
        entries=[]
        for index in range(110):
            seed=1200+index
            if 'distribution_group' in case:
                group=registry['groups'][case['distribution_group']];observed=recipe.weighted_variant(group,seed)
                entries.append({'input_seed':seed,'variant_id':observed['variant_id'],'num_valid_ids':observed['num_valid_ids'],
                    'route_seed':int(hashlib.sha256((str(seed)+':'+observed['variant_id']).encode()).hexdigest()[:16],16),
                    'routing_origin':'generated_legal_routes_not_actual_other_rank_capture',
                    'activation_origin':'fresh_seeded_fp8_values_with_remapped_captured_token_scales',
                    'fresh_stage1_reference_output':group['seam']=='moe2'})
            else:entries.append({'input_seed':seed,'fixture':case['fixture']})
        schedule={'schema_version':1,'case_id':case['case_id'],'manifest_sha256':fingerprint(manifest),
                  'warmup_inputs':entries[:10],'measured_inputs':entries[10:]}
        common={'case':case,'correct':True,'warmup_iterations':10,'fresh_input_resets':100,
                'output_initializations':100,'oracle_checks':100,'benchmark_method':'cuda_graph','paired_schedule_sha256':fingerprint(schedule)}
        row={**common,'samples_ms':[1.0]*100}
        row['paired_reference']={'case_id':case['case_id'],'candidate_binding':cb,'reference_binding':rb,
            'candidate_snapshot_before_reference':True,'reference_reused_as_oracle':True,'output_conformance':True,
            'input_schedule':schedule,'reference_output_clear_calls':110,'checked_pair_count':110,
            'graph_setup':{'warmup_invocations_per_leg':3,'capture_invocations_per_leg':1,
                'capture_on_warmed_stream':True,'separate_graphs_and_outputs':True},
            'legs':{'candidate_port':{**common,'samples_ms':[1.0]*100},'protected_reference':{**common,'samples_ms':[2.0]*100}}}
        if 'distribution_group' in case:
            row['observed_work_distribution']={'histogram_sha256':group['histogram_sha256'],
                'frequency_basis':'actual eight-rank call counts',
                'sampling':'integer histogram draws from the protected request challenge seed',
                'schedule_sha256':fingerprint([entry['variant_id'] for entry in entries]),
                'shared_private_request_seed_required_for_paired_comparison':True,
                'warmup_variants':entries[:10],
                'performance_samples':[dict(entry,device_time_ms=sample) for entry,sample in zip(entries[10:],row['samples_ms'])],
                'distinct_measured_settings':len({entry['variant_id'] for entry in entries[10:]}),
                'observed_setting_count':len(group['histogram']),'all_observed_settings_timed':False,
                'actual_other_rank_routing_recovered':False,
                'scope':'sampled observed work-count distribution with representative generated routes'}
        rows.append(row);compiled.append({'case_id':case['case_id'],'candidate_binding':cb,'reference_binding':rb,'invoked_and_synchronized':True})
    report=finalize_report({'schema_version':1,'status':'ok','request':request,'compiled_specializations':compiled,'cases':rows},manifest,request)
    pair.attach_comparison(root,report,manifest,request)
    measured=load_native_measurements(root,config,report=report,request=request)
    assert measured['baseline_kind']==config['scoring_baseline']['kind']
    assert len(measured['candidate'])==len(manifest['cases'])
    assert all(row['execution_time_ms']==2.0 for row in measured['native'])


@pytest.mark.parametrize('stage',['decode','prefill'])
def test_stage2_distribution_dispatch_prepares_fresh_input_once_per_checked_pair(tmp_path,monkeypatch,stage):
    import torch
    import distribution_runner
    manifest=json.loads((TASK/'cases.json').read_text())
    base=json.loads((TASK/'provenance/BASE-CASES.json').read_text())
    registry=json.loads((TASK/'provenance/WORK-DISTRIBUTIONS.json').read_text())
    case=next(c for c in manifest['cases'] if c.get('distribution_group') and c['stage']==stage)
    (tmp_path/'build').mkdir();instances=[]
    class Provider:
        def __init__(self,*args):
            self.producer_calls=0;self.draws=[];self.producer_identity={'CPU_only':True};instances.append(self)
        def refresh(self,seed,forced=None):
            self.producer_calls+=1
            self.draws.append({'input_seed':seed,'variant_id':'cpu-'+str(seed),'num_valid_ids':[forced or 42,64]})
            return {'CPU_only':seed}
    def paired(*args,**kwargs):
        reset=args[12];signature=kwargs['current_input_signature']
        for index in range(110):
            reset(1000+index);assert signature()['input_seed']==1000+index
        return {'case':case,'samples_ms':[1.0]*100}
    monkeypatch.setattr(distribution_runner,'Provider',Provider)
    monkeypatch.setattr(distribution_runner,'paired_performance',paired)
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    api={'fixture':lambda *args:({'a':torch.ones(1)},None),'storage_snapshots':lambda inputs:{},
         'cpu_clone':lambda value:None,'invoke':lambda *args:torch.ones(1),
         'verify_after_snapshot':lambda *args:None,'runtime_abi':lambda *args:({},{}),
         'observe_case':lambda case,*args:case}
    result,_=distribution_runner.run_case(api,tmp_path,manifest,base,registry,case,'performance',{'challenge_seed':1000},
        None,None,None,None,{}, {})
    assert instances[0].producer_calls==111
    assert result['stage1_input_generation']['fresh_calls']==110
    assert result['stage1_input_generation']['captured_parent_output_reused'] is False
    assert len(result['observed_work_distribution']['performance_samples'])==100


@pytest.mark.parametrize('attack',['unknown_work','different_observed_work','missing_observed_receipt',
                                  'receipt_time','receipt_work','fixed_fixture'])
def test_rehashed_real_task_schedule_forgery_is_rejected(monkeypatch,attack):
    captured={};original=load_native_measurements
    def retain(root,config,**kwargs):
        captured.update(root=root,config=config,report=copy.deepcopy(kwargs['report']),request=kwargs['request'])
        return original(root,config,**kwargs)
    monkeypatch.setitem(globals(),'load_native_measurements',retain)
    test_real_task_schema_and_source_provenance_accept_complete_cpu_rows('moe_stage2_down_proj_reduce_opus_a8w4')
    report=captured['report'];root=captured['root']
    target=next(row for row in report['cases'] if bool(row['case'].get('distribution_group'))==(attack!='fixed_fixture'))
    paired=next(row for row in report['paired_reference_comparison']['cases'] if row['case_id']==target['case']['case_id'])
    signature=paired['input_schedule']['measured_inputs'][0]
    if attack=='missing_observed_receipt':target.pop('observed_work_distribution')
    elif attack=='receipt_time':target['observed_work_distribution']['performance_samples'][0]['device_time_ms']=99.0
    elif attack=='receipt_work':target['observed_work_distribution']['performance_samples'][0]['num_valid_ids']=[1,64]
    elif attack=='fixed_fixture':signature['fixture']={'path':'fixtures/foreign.json','sha256':'0'*64}
    else:
        if attack=='unknown_work':signature.update(num_valid_ids=[1,64],variant_id='0'*64)
        else:
            registry=json.loads((root/'provenance/WORK-DISTRIBUTIONS.json').read_text())
            histogram=registry['groups'][target['case']['distribution_group']]['histogram']
            other=next(row for row in histogram if row['variant_id']!=signature['variant_id'])
            signature.update(num_valid_ids=other['num_valid_ids'],variant_id=other['variant_id'])
        signature['route_seed']=int(hashlib.sha256((str(signature['input_seed'])+':'+signature['variant_id']).encode()).hexdigest()[:16],16)
        receipt=target['observed_work_distribution']
        receipt['performance_samples'][0]={**signature,'device_time_ms':target['samples_ms'][0]}
        receipt['schedule_sha256']=fingerprint([row['variant_id'] for row in receipt['warmup_variants']+receipt['performance_samples']])
        receipt['distinct_measured_settings']=len({row['variant_id'] for row in receipt['performance_samples']})
    digest=fingerprint(paired['input_schedule']);target['paired_schedule_sha256']=digest
    for leg in paired['legs'].values():leg['paired_schedule_sha256']=digest
    target['paired_reference']=copy.deepcopy(paired)
    with pytest.raises(ValueError):original(root,captured['config'],report=report,request=captured['request'])


@pytest.mark.parametrize('failure',['immutable','abi'])
def test_capture_setup_failure_scrubs_entire_output_storage(monkeypatch,failure):
    import torch
    from contextlib import nullcontext
    backing=torch.arange(64,dtype=torch.uint8);output=backing[8:24].view(torch.float32)
    stream=type('Stream',(),{'wait_stream':lambda self,other:None,'synchronize':lambda self:None})()
    monkeypatch.setattr(torch.cuda,'Stream',lambda:stream);monkeypatch.setattr(torch.cuda,'current_stream',lambda:stream)
    monkeypatch.setattr(torch.cuda,'stream',lambda stream:nullcontext());monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    monkeypatch.setattr(torch.cuda,'CUDAGraph',lambda:object());monkeypatch.setattr(torch.cuda,'graph',lambda *a,**k:nullcontext())
    def invoke(*args):backing.fill_(17);return output
    def immutable(*args):
        if failure=='immutable':raise ValueError('setup immutable failure')
    def observe(*args):raise ValueError('setup ABI failure')
    api={'leaves':lambda value:[value] if value is not None else [],'invoke':invoke,'restore_storages':lambda *args:None,
         'assert_immutable_inputs':immutable,'cpu_clone':lambda value:value.clone(),'runtime_abi':lambda *args:({},{}),'observe_case':observe}
    with pytest.raises(ValueError):pair.capture_native_graph(api,{},None,{}, {})
    assert bool((backing==0xAA).all())


@pytest.mark.parametrize('attack',[None,'structured_success','changed_dependency','candidate_in_closure','candidate_import','cached_candidate_import','false_verdict'])
def test_pinned_cpu_validator_hook_rejects_candidate_imports_and_tampering(tmp_path,monkeypatch,attack):
    from src.native_baseline import _validate_task_input_receipts
    import types
    ut=tmp_path/'ut';ut.mkdir();helper=ut/'validate_inputs.py'
    candidate=ut/'candidate_module.py';candidate.write_text('raise AssertionError("candidate code executed")\n')
    helper.write_text(('import candidate_module\n' if attack in ('candidate_import','cached_candidate_import') else '')+
        'def validate(root, report, manifest, request, provenance):\n    return '+('False' if attack=='false_verdict' else "{'status': 'ok'}" if attack=='structured_success' else 'True')+'\n')
    files={'ut/validate_inputs.py':hashlib.sha256(helper.read_bytes()).hexdigest()}
    if attack=='candidate_in_closure':files['ut/candidate_module.py']=hashlib.sha256(candidate.read_bytes()).hexdigest()
    if attack=='changed_dependency':helper.write_text(helper.read_text()+'# changed\n')
    if attack=='cached_candidate_import':
        cached=types.ModuleType('candidate_module');cached.__file__=str(candidate)
        monkeypatch.setitem(sys.modules,'candidate_module',cached)
    provenance={'paired_input_validator':{'schema':'task-local-paired-input-validator-v1','validator':'ut/validate_inputs.py',
        'function':'validate','files_sha256':files}}
    if attack in (None,'structured_success'):
        _validate_task_input_receipts(tmp_path,{}, {}, {},provenance,{'ut/candidate_module.py':'a'*64})
    else:
        with pytest.raises(ValueError):_validate_task_input_receipts(tmp_path,{}, {}, {},provenance,{'ut/candidate_module.py':'a'*64})
    assert not (ut/'__pycache__').exists()


@pytest.mark.parametrize('mode',['entry','helper','relative_package_helper'])
def test_matching_pyc_cannot_replace_verified_entry_or_helper_source(tmp_path,mode):
    from src.native_baseline import _validate_task_input_receipts
    import os
    import py_compile
    ut=tmp_path/'ut';ut.mkdir();entry=ut/'validate_inputs.py';target=entry
    source='def validate(root, report, manifest, request, provenance):\n    return False\n'
    forged='def validate(root, report, manifest, request, provenance):\n    return True \n'
    if mode=='helper':
        entry.write_text('from support import validate\n');target=ut/'support.py'
    elif mode=='relative_package_helper':
        package=ut/'validators';package.mkdir();entry=package/'__init__.py'
        entry.write_text('from .support import validate\n');target=package/'support.py'
    assert len(source)==len(forged)
    target.write_text(forged);stamp=target.stat()
    bytecode=Path(py_compile.compile(str(target),doraise=True));cached=bytecode.read_bytes()
    target.write_text(source);os.utime(target,ns=(stamp.st_atime_ns,stamp.st_mtime_ns))
    files={p.relative_to(tmp_path).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in ut.rglob('*.py')}
    provenance={'paired_input_validator':{'schema':'task-local-paired-input-validator-v1',
        'validator':entry.relative_to(tmp_path).as_posix(),'function':'validate','files_sha256':files}}
    with pytest.raises(ValueError,match='validator rejected'):
        _validate_task_input_receipts(tmp_path,{}, {}, {},provenance,{'source/candidate.py':'a'*64})
    assert target.read_text()==source
    assert bytecode.read_bytes()==cached  # Integrity does not depend on deleting caches.


def test_ambient_cached_helper_is_replaced_by_verified_source_then_restored(tmp_path,monkeypatch):
    from src.native_baseline import _validate_task_input_receipts
    import types
    ut=tmp_path/'ut';ut.mkdir();entry=ut/'validate_inputs.py';helper=ut/'support.py'
    entry.write_text('from support import validate\n')
    helper.write_text('def validate(*args):\n    return False\n')
    ambient=types.ModuleType('support');ambient.__file__='/ambient/site-packages/support.py'
    ambient.validate=lambda *args:True;monkeypatch.setitem(sys.modules,'support',ambient)
    files={p.relative_to(tmp_path).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in [entry,helper]}
    provenance={'paired_input_validator':{'schema':'task-local-paired-input-validator-v1',
        'validator':'ut/validate_inputs.py','function':'validate','files_sha256':files}}
    with pytest.raises(ValueError,match='validator rejected'):
        _validate_task_input_receipts(tmp_path,{}, {}, {},provenance,{'source/candidate.py':'a'*64})
    assert sys.modules['support'] is ambient


def test_helper_executes_captured_verified_bytes_even_if_file_changes_after_check(tmp_path):
    from src.native_baseline import _validate_task_input_receipts
    ut=tmp_path/'ut';ut.mkdir();entry=ut/'validate_inputs.py';helper=ut/'support.py'
    source='def validate(root, report, *args):\n    report["executed"] = "verified"\n    return False\n'
    forged='def validate(root, report, *args):\n    report["executed"] = "replaced"\n    return True\n'
    helper.write_text(source)
    entry.write_text('from pathlib import Path\nPath(__file__).with_name("support.py").write_text('+repr(forged)+')\nfrom support import validate\n')
    files={p.relative_to(tmp_path).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in [entry,helper]}
    provenance={'paired_input_validator':{'schema':'task-local-paired-input-validator-v1',
        'validator':'ut/validate_inputs.py','function':'validate','files_sha256':files}}
    report={}
    with pytest.raises(ValueError,match='validator rejected'):
        _validate_task_input_receipts(tmp_path,report, {}, {},provenance,{'source/candidate.py':'a'*64})
    assert report['executed']=='verified'
