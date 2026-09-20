"""CPU controls for compact GLM contracts and protected worker boundaries."""
from __future__ import annotations
import base64
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import zlib

import pytest
from head_kernel_generated_test_utils import generated_helper, generated_task
import torch

ROOT = Path(__file__).resolve().parents[1]
TASKS = {p.parent.name:p.parent for p in (ROOT/'tasks/head_kernels/glm-5.3-flash').rglob('config.yaml')
         if p.parent.name in ('elementwise_copy_cluster','fused_moe_kernel')}


def common_path(tmp_path, filename):
    """Test the finalized integration dependency without vendoring a private guard."""
    target=tmp_path/'common-support'/filename
    target.parent.mkdir(parents=True,exist_ok=True)
    support=ROOT/'tasks/head_kernels/_support'
    target.write_bytes((support/filename).read_bytes())
    return target



def load(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


contract=load('generated_contract',generated_helper("glm", "generated_contract.py"))
worker=load('generated_worker',generated_helper("glm", "generated_worker.py"))
controller=load('glm_controller_test',generated_helper("glm", "generated_correctness.py"))
extractor=load('glm_extract_test',generated_helper("glm", "extract_contract.py"))


def tensor(shape=(2,3), recipe='generated_numeric', dtype='torch.float32', group='a', **kw):
    size=1;stride=[]
    for dim in reversed(shape):stride.insert(0,size);size*=dim
    item=torch.empty((),dtype=getattr(torch,dtype.removeprefix('torch.'))).element_size()
    return {'tensor':True,'shape':list(shape),'stride':stride,'storage_offset':0,
            'storage_group':group,'storage_nbytes':size*item,'tensor_attrs':{},
            'requires_grad':False,'recipe':recipe,'dtype':dtype,**kw}


def compact(blob, task='elementwise_copy_cluster'):
    return {'task':task,'blob':blob,'activation_std':1.25}


def test_committed_contracts_keep_every_recorded_random_and_timed_case():
    expected={'elementwise_copy_cluster':(7,8,8),'fused_moe_kernel':(3,4,0)}
    for name, task in TASKS.items():
        value=contract.load_contract(task/'ut')
        meta=json.loads((task/'ut/meta.json').read_text())
        profiles=meta['generated_inputs']['profiles']
        assert tuple(len(profiles[p]) for p in ('recorded','random','timed'))==expected[name]
        assert value['correctness_case_count']==meta['num_cases']
        assert not json.loads((task/'scripts/artifacts.json').read_text())
        assert 'reference_io_sha256' not in meta
        assert meta['tol']==(1e-6 if name=='elementwise_copy_cluster' else 0.02)
        for path in task.rglob('reference_io.pt'):pytest.fail(f'unexpected archive {path}')
    moe=contract.load_contract(TASKS['fused_moe_kernel']/'ut')
    assert moe['blob']['shared']['w1']['shape']==[288,512,4096]
    assert moe['blob']['shared']['w2']['shape']==[288,4096,256]
    assert moe['blob']['static']['block_shape']['items']==[128,128]
    assert moe['blob']['static']['inplace'] is False
    assert moe['blob']['static']['routed_scaling_factor']==2.5
    assert moe['blob']['static']['swiglu_limit']==10.0


def test_actual_copy_records_retain_layout_aliases_and_math_outputs():
    task=TASKS['elementwise_copy_cluster'];value=contract.load_contract(task/'ut')
    cases=load('_glm_copy_cases_test',task/'ut/cases.py')
    contract.prepare_cases(cases,task/'ut',991,torch,'cpu')
    rows=contract.recorded_cases(cases,value)
    assert len(rows)==7
    aliases=[]
    for row in rows:
        scale=row['args']['scale'];out=contract.reference_copy(row['args'])
        encoded=contract.encode_output(out,torch)
        contract.require_output_contract(encoded,row['output_contract'])
        assert torch.equal(out,scale)
        aliases.append(contract.alias_contract(out,row['args'],torch))
    assert aliases[4]==aliases[5]==[[0,0]]  # singleton rows preserve the true alias.
    assert aliases[0]==aliases[-1]==[]


def test_generated_values_are_deterministic_and_preserve_storage_views():
    desc=tensor(shape=(2,2),stride=[3,1],storage_offset=1,storage_nbytes=24,
                tensor_attrs={'dispatch_flag':True})
    value=compact({'a':desc,'alias':copy.deepcopy(desc)})
    one=contract.build_blob(value,77,torch,'cpu');repeat=contract.build_blob(value,77,torch,'cpu')
    changed=contract.build_blob(value,78,torch,'cpu')
    assert one['a'].stride()==(3,1) and one['a'].storage_offset()==1
    assert one['a'].dispatch_flag is True
    assert one['a'].data_ptr()==one['alias'].data_ptr()
    assert torch.equal(one['a'],repeat['a']) and not torch.equal(one['a'],changed['a'])


def test_fp8_generation_and_scale_recipe_are_finite_and_valid():
    value=compact({'w1':tensor((2,128,128),'finite_fp8_weight','torch.float8_e4m3fn'),
                   'w1_scale':tensor((2,1,1),'positive_block_scale',group='scale')},'fused_moe_kernel')
    data=contract.build_blob(value,19,torch,'cpu')
    assert data['w1'].dtype==torch.float8_e4m3fn
    assert torch.isfinite(data['w1'].float()).all()
    assert data['w1'].float().abs().max()<=448
    assert (data['w1_scale']>0).all() and torch.isfinite(data['w1_scale']).all()


def test_every_real_routing_tensor_roundtrips_its_exact_bytes():
    value=contract.load_contract(TASKS['fused_moe_kernel']/'ut')
    leaves=[]
    def collect(node):
        if isinstance(node,dict):
            if node.get('recipe')=='captured_routing':leaves.append(node)
            else:
                for child in node.values():collect(child)
        elif isinstance(node,list):
            for child in node:collect(child)
    collect(value['blob'])
    assert len(leaves)==12
    for leaf in leaves:
        actual=contract.build_blob(compact({'route':leaf},'fused_moe_kernel'),99,torch,'cpu')['route']
        raw=zlib.decompress(base64.b64decode(leaf['data']))
        assert actual.contiguous().view(torch.uint8).numpy().tobytes()==raw
        assert actual.stride()==tuple(leaf['stride'])
        if actual.dtype==torch.int32:assert int(actual.min())>=0 and int(actual.max())<288


@pytest.mark.parametrize('dt',[torch.int32,torch.int64,torch.bool])
def test_integer_and_boolean_outputs_require_exact_equality(dt):
    left=torch.tensor([100000,200000],dtype=dt) if dt!=torch.bool else torch.tensor([True,False])
    right=left.clone();right[0]=right[0]-1 if dt!=torch.bool else False
    assert not contract.compare_output(contract.encode_output(left,torch),contract.encode_output(right,torch),0.02,torch)


def test_parent_rejects_wrong_values_metadata_aliases_and_missing_timed_probe():
    payload={'id':'x','inputs':[],'output':contract.encode_output(torch.tensor([2.]),torch),'aliases':[]}
    for change in ('value','stride','alias'):
        altered=copy.deepcopy(payload)
        if change=='value':altered['output']=contract.encode_output(torch.tensor([3.]),torch)
        elif change=='stride':altered['output']['stride']=[2]
        else:altered['aliases']=[[0,0]]
        with pytest.raises(RuntimeError,match='mismatch'):
            controller.compare_rows([payload],[altered],contract,0.02,torch)
    timed={'id':'x','inputs':[],'checks':[payload]*3}
    with pytest.raises(RuntimeError,match='A/B/A'):
        controller.compare_rows([timed],[{**timed,'checks':[payload]*2}],contract,0.02,torch,timed=True)


def test_parent_rejects_stale_or_partial_worker_results():
    value={'schema_version':1,'profile':'recorded','seed':9,'reference':False,'rows':[{'id':'x'}]}
    def proc(data):return SimpleNamespace(returncode=0,stdout=controller.PREFIX+json.dumps(data),stderr='')
    assert controller.parse_worker(proc(value),'recorded',9,False,['x'])==[{'id':'x'}]
    for altered in ({**value,'seed':8},{**value,'reference':True},{**value,'rows':[]}):
        with pytest.raises(RuntimeError):controller.parse_worker(proc(altered),'recorded',9,False,['x'])


def make_guard(tmp_path, monkeypatch):
    monitor=load('runtime_integrity',common_path(tmp_path,'runtime_integrity.py'))
    benchmark=load('_aka_benchmark',ROOT/'src/tools/perf/aka_benchmark.py')
    harness=load('harness_lib',TASKS['elementwise_copy_cluster']/'ut/harness_lib.py')
    monkeypatch.setitem(sys.modules,'generated_contract',contract)
    monkeypatch.setitem(sys.modules,'generated_worker',worker)
    guard=monitor.RuntimeIntegrity(tmp_path,torch,benchmark,harness,trusted_modules={'generated_contract':contract,'generated_worker':worker})
    guard.install()
    return monitor,guard


@pytest.mark.parametrize('attack',['checker','encoder','worker','module','code'])
def test_loaded_helper_attestation_rejects_runtime_monkeypatches(tmp_path,monkeypatch,attack):
    monitor,guard=make_guard(tmp_path,monkeypatch)
    old_checker=contract.require_inputs_unchanged;old_encoder=contract.encode_output
    old_worker=worker.invoke_checked;old_code=old_checker.__code__
    try:
        if attack=='checker':contract.require_inputs_unchanged=lambda *a:None
        elif attack=='encoder':contract.encode_output=lambda *a:{'stride':[1]}
        elif attack=='worker':worker.invoke_checked=lambda *a:{'output':None}
        elif attack=='module':sys.modules['generated_contract']=ModuleType('generated_contract')
        else:old_checker.__code__=(lambda *a:None).__code__
        with pytest.raises(monitor.IntegrityError):guard.check()
    finally:
        contract.require_inputs_unchanged=old_checker;contract.encode_output=old_encoder
        worker.invoke_checked=old_worker;old_checker.__code__=old_code
        sys.modules['generated_contract']=contract
        guard.close()


def test_input_mutation_is_rejected_with_real_attested_helpers(tmp_path,monkeypatch):
    monitor,guard=make_guard(tmp_path,monkeypatch)
    x=torch.ones(2)
    def malicious(values):
        values['scale'].zero_()
        return values['scale'].clone()
    try:
        with pytest.raises(RuntimeError,match='read-only GLM input'):
            worker.invoke_checked(malicious,{'scale':x},contract,torch,[])
    finally:guard.close()


def test_no_candidate_golden_file_or_reference_payload_in_generated_protocol():
    source=(generated_helper("glm", "generated_correctness.py")).read_text()
    assert 'torch.save' not in source and 'torch.load' not in source
    assert '_benchmark_reference' not in source
    assert '--reference' in source  # a role flag, not a path or expected answer
    candidate=(generated_helper("glm", "generated_worker.py")).read_text()
    assert 'expected[' not in candidate and 'reference_path' not in candidate


def test_extractor_omits_numerical_and_golden_values():
    x=torch.randn(2,3)
    result=extractor.extract_blob({'records':[{'sig':'s','regime':'decode','args':(x,),
                 'kwargs':{},'output':x.t().contiguous().t()}]},
                 {'num_cases':1,'reference_io_sha256':'a'*64},'elementwise_copy_cluster',torch)
    record=result['blob']['records']['items'][0]
    assert record['args']['items'][0]['recipe']=='generated_numeric'
    assert record['output']['recipe']=='runtime_reference'
    assert 'data' not in record['output'] and 'data' not in record['args']['items'][0]


def test_task_source_and_abi_bytes_are_unchanged_from_published_base():
    for task in TASKS.values():
        paths=list((task/'source').glob('*.py'))+[task/'scripts/source_abi.json']
        for path in paths:
            relative=path.relative_to(ROOT).as_posix()
            expected=subprocess.check_output(['git','show','b66e373d:'+relative],cwd=ROOT)
            assert path.read_bytes()==expected


def test_candidate_checker_or_encoder_attack_is_caught_at_call_boundary(tmp_path,monkeypatch):
    monitor,guard=make_guard(tmp_path,monkeypatch)
    original=contract.require_inputs_unchanged
    def attack(values):
        sys.modules['generated_contract'].require_inputs_unchanged=lambda *a:None
        values['scale'].zero_()
        return values['scale'].clone()
    try:
        with pytest.raises(monitor.IntegrityError):
            worker.invoke_checked(attack,{'scale':torch.ones(2)},contract,torch,[])
    finally:
        contract.require_inputs_unchanged=original
        guard.close()


def test_exact_timed_replay_outputs_are_compared_only_in_parent(monkeypatch):
    bench=load('_bench',ROOT/'tasks/head_kernels/_support/_bench.py')
    monkeypatch.setattr(worker,'check',lambda:None)
    settings=[]
    def fake_benchmark(fn, *, warmup, repetition, prepare_fn, timed_run):
        settings.append((warmup,repetition))
        prepare_fn();output=fn()
        def replay():
            prepare_fn()
            output.copy_(fn())
            return output
        timed_run._bind(replay,output)
        return [0.2]*100,{'benchmark_method':'cuda_graph','benchmark_samples':100}
    monkeypatch.setitem(sys.modules,'_aka_benchmark',SimpleNamespace(benchmark_cuda_graph_or_events_samples=fake_benchmark))
    def row():return {'sig':'scale','regime':'decode','m':2,'args':{'scale':torch.arange(6.).reshape(2,3)}}
    baseline=worker.timed_row(row(),contract.reference_copy,SimpleNamespace(),{'tol':1e-6},True,contract,torch)
    candidate=worker.timed_row(row(),contract.reference_copy,SimpleNamespace(),{'tol':1e-6},False,contract,torch)
    controller.compare_rows([baseline],[candidate],contract,1e-6,torch,timed=True)
    assert settings==[(10,100)]
    assert candidate['timing']['benchmark_output_validation']=='exact_timed_graph_replay'
    def stale_benchmark(fn, *, warmup, repetition, prepare_fn, timed_run):
        prepare_fn();out=fn()
        def replay():
            prepare_fn()
            return out
        timed_run._bind(replay,out)
        return [0.2]*100,{'benchmark_method':'cuda_graph','benchmark_samples':100}
    monkeypatch.setitem(sys.modules,'_aka_benchmark',SimpleNamespace(benchmark_cuda_graph_or_events_samples=stale_benchmark))
    stale=worker.timed_row(row(),contract.reference_copy,SimpleNamespace(),{'tol':1e-6},False,contract,torch)
    with pytest.raises(RuntimeError,match='timed replay mismatch'):
        controller.compare_rows([baseline],[stale],contract,1e-6,torch,timed=True)


def test_real_independent_cpu_worker_process_detects_wrong_candidate(tmp_path):
    # These small workers use the same generator/checker/encoder/worker objects
    # and pre-import guard as the GPU wrapper, without claiming GPU validation.
    task=tmp_path/'task';ut=task/'ut';ut.mkdir(parents=True);(task/'source').mkdir()
    desc=tensor((2,3));output=tensor((2,3),recipe='runtime_reference',group='out',stride=[1,2])
    data={'schema_version':1,'source_reference_sha256':'a'*64,'task':'elementwise_copy_cluster',
          'correctness_case_ids':['capture'],'correctness_case_count':1,
          'blob':{'records':{'sequence':'list','items':[{'sig':'capture','regime':'decode',
           'args':{'sequence':'tuple','items':[desc]},'kwargs':{},'output':output}]},'shared':{}}}
    (ut/'generated_cases.json').write_text(json.dumps(data))
    meta={'num_cases':1,'random_draws':3,'tol':1e-6,'archival_capture':{'reference_io_sha256':'a'*64},
          'generated_inputs':{'contract_file':'generated_cases.json','contract_sha256':contract.digest(ut/'generated_cases.json'),
                              'captured_case_ids':['capture']}}
    (ut/'meta.json').write_text(json.dumps(meta))
    (ut/'cases.py').write_text('_C={}\ndef _blob(): return _C["blob"]\ndef call(args):\n    from candidate import run\n    return run(args)\n')
    candidate=task/'source/candidate.py'
    candidate.write_text('def run(args):\n    x=args["scale"]\n    return x.t().contiguous().t()\n')
    program='''
import importlib.util,json,pathlib,sys
import torch
root=pathlib.Path(sys.argv[1]);task=pathlib.Path(sys.argv[2]);reference=sys.argv[3]=='1'
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
 sys.modules[name]=module;spec.loader.exec_module(module);return module
base=next((root/'tasks/head_kernels/glm-5.3-flash').rglob('ut/generated_contract.py')).parent.parent
contract=load('generated_contract',base/'ut/generated_contract.py')
worker=load('generated_worker',base/'scripts/generated_worker.py')
cases=load('_glm_generated_cases',task/'ut/cases.py')
bench=load('_aka_benchmark',root/'src/tools/perf/aka_benchmark.py')
harness=load('harness_lib',pathlib.Path(sys.argv[4]))
monitor=load('runtime_integrity',pathlib.Path(sys.argv[5]))
guard=monitor.RuntimeIntegrity(task,torch,bench,harness,trusted_modules={'generated_contract':contract,'generated_worker':worker,'_glm_generated_cases':cases})
guard.install()
try:
 sys.path.insert(0,str(task/'source'))
 result=worker.run_profile(task/'ut','recorded',55,reference,device='cpu')
 guard.check()
 print(worker.PREFIX+json.dumps(result))
finally:guard.close()
'''
    def execute(reference):
        return subprocess.run([sys.executable,'-B','-c',program,str(ROOT),str(task),str(int(reference)),
              str(TASKS['elementwise_copy_cluster']/'ut/harness_lib.py'),str(common_path(tmp_path,'runtime_integrity.py'))],capture_output=True,text=True,check=True)
    expected=['oracle0_m2_decode']
    ref=controller.parse_worker(execute(True),'recorded',55,True,expected)
    good=controller.parse_worker(execute(False),'recorded',55,False,expected)
    controller.compare_rows(ref,good,contract,1e-6,torch)
    candidate.write_text('def run(args):\n    x=args["scale"]\n    return x.t().contiguous().t()+1\n')
    wrong=controller.parse_worker(execute(False),'recorded',55,False,expected)
    with pytest.raises(RuntimeError,match='correctness mismatch'):
        controller.compare_rows(ref,wrong,contract,1e-6,torch)




def test_output_instance_cannot_forge_values_or_strides():
    out=torch.ones(2,3)
    out.stride=lambda:(1,2)
    with pytest.raises(TypeError,match='shadowed methods'):
        contract.encode_output(out,torch)
    del out.stride
    out.detach=lambda:torch.zeros(2,3)
    with pytest.raises(TypeError,match='shadowed methods'):
        contract.encode_output(out,torch)


def test_compact_contract_tampering_and_workspace_copy_are_protected(tmp_path):
    import logging
    from src.preprocessing import setup_workspace
    from src.harness_guard import snapshot_workspace_harness, verify_workspace_harness
    task=TASKS['elementwise_copy_cluster']
    workspace=setup_workspace(str(task/'config.yaml'),tmp_path/'run','test',logging.getLogger(__name__),
                              task_name='head_kernels/glm-generated-copy')
    assert (workspace/'scripts/_aka_benchmark.py').is_file()
    assert (workspace/'ut/generated_cases.json').read_bytes()==(task/'ut/generated_cases.json').read_bytes()
    assert not (workspace/'ut/reference_io.pt').exists()
    snapshot=snapshot_workspace_harness(workspace,task_root=task)
    (workspace/'ut/generated_cases.json').write_text('{}')
    with pytest.raises(RuntimeError,match='checksum'):
        contract.load_contract(workspace/'ut')
    with pytest.raises(RuntimeError):verify_workspace_harness(snapshot)


def test_moe_scoring_is_blocked_before_any_worker_and_no_counter_becomes_payload(monkeypatch):
    task=TASKS['fused_moe_kernel'];meta=json.loads((task/'ut/meta.json').read_text())
    assert meta['generated_inputs']['profiles']['timed']==[]
    scoring=meta['generated_inputs']['workload_scoring']
    assert scoring['enabled'] is False
    assert scoring['retained_payload_m']==[19,1,8192]
    assert scoring['missing_payload_m']==[64,16384]
    assert [r['m'] for r in meta['generated_inputs']['semantic_call_probes']]==[19,1,8192]
    assert all(r['scored'] is False for r in meta['generated_inputs']['semantic_call_probes'])
    reports=[]
    runner=SimpleNamespace(UT_DIR=task/'ut',write_report=lambda name,data:reports.append(data))
    monkeypatch.setattr(controller,'run_correctness',lambda *a:pytest.fail('must fail before worker/kernel execution'))
    assert controller.run_performance(runner,{},60)==[]
    assert reports[-1]['status']=='fail'
    assert 'authentic pre-call capture' in reports[-1]['error']


def test_semantic_retained_routing_is_verbatim_and_inplace_proof_is_required(monkeypatch):
    task=TASKS['fused_moe_kernel'];meta=json.loads((task/'ut/meta.json').read_text())
    compact_data=contract.load_contract(task/'ut')
    records=[]
    for original in compact_data['blob']['cases']['items']:
        record={'m':original['m'],'regime':original['regime'],'sig':original['sig'],
                'hidden_states':object()}
        for name in ['topk_ids','topk_weights']:
            desc=original[name]
            record[name]=contract.build_blob({**compact_data,'blob':{'value':desc}},3,torch,'cpu')['value']
        records.append(record)
    def args(hs,weights,ids):return {'hidden_states':hs,'topk_weights':weights,'topk_ids':ids,'inplace':False}
    monkeypatch.setattr(torch,'randint',lambda *a,**k:pytest.fail('retained semantic routing must not be resampled'))
    rows=contract.semantic_moe_call_cases({'cases':records},args,meta)
    assert len(rows)==3
    for row,record in zip(rows,records):
        assert row['args']['topk_ids'] is record['topk_ids']
        assert row['args']['topk_weights'] is record['topk_weights']
        assert row['args']['inplace'] is True
        assert row['mutable_inputs']==['hidden_states'] and row['scored'] is False
    records[0]['sig']=records[0]['sig'].replace('|None|None|True|str|','|None|None|False|str|',1)
    with pytest.raises(ValueError,match='in-place'):
        contract.semantic_moe_call_cases({'cases':records},args,meta)


def test_inplace_probe_allows_only_declared_value_writes():
    hidden=torch.ones(2,3);route=torch.tensor([[0],[1]])
    saved=contract.snapshot_inputs((),{'hidden_states':hidden,'topk_ids':route},torch)
    hidden.add_(4)
    contract.require_inputs_unchanged(saved,torch,writable=(hidden,))
    route.add_(1)
    with pytest.raises(RuntimeError,match='read-only'):
        contract.require_inputs_unchanged(saved,torch,writable=(hidden,))


def test_declared_common_boundary_has_all_real_aliases_and_no_private_guard():
    import yaml
    for task in TASKS.values():
        cfg=yaml.safe_load((task/'config.yaml').read_text())
        aliases=cfg['headkernel']['trusted_worker_modules']
        assert aliases['generated_contract']=='ut/generated_contract.py'
        assert aliases['generated_worker']=='scripts/generated_worker.py'
        assert aliases['_glm_generated_cases']==aliases['_headkernel_cases']=='ut/cases.py'
        assert aliases['_bench']=='scripts/_bench.py'
        assert cfg['headkernel']['common_boundary_revision']=='a7bf289b'
        for name in ['generated_integrity.py','generated_trusted_worker.py','generated_runtime_preflight.py']:
            assert not (task/'scripts'/name).exists()


def test_copy_producer_stride_proof_is_conditional_and_external_layout_branch_is_flagged():
    proof=json.loads((TASKS['elementwise_copy_cluster']/'ut/copy_layout_proof.json').read_text())
    assert proof['standard_quant_producer']['transpose_scale'] is False
    assert len(proof['unproven_branches'])==2
    for m,g in proof['timed_shapes']:
        row_major=torch.arange(m*g,dtype=torch.float32).reshape(m,g)
        copied=contract.reference_copy({'scale':row_major})
        assert copied.stride()==(1,m)
        assert not contract.alias_contract(copied,{'scale':row_major},torch)
        # The pre-quantized caller branch may already supply the desired layout;
        # equal shape alone cannot establish the same copy work or alias contract.
        already_column_major=copied
        viewed=contract.reference_copy({'scale':already_column_major})
        assert viewed.stride()==(1,m)
        assert contract.alias_contract(viewed,{'scale':already_column_major},torch)==[[0,0]]


def test_moe_performance_cli_fails_before_runtime_or_gpu_initialization():
    result=subprocess.run([sys.executable,str(TASKS['fused_moe_kernel']/'scripts/generated_task_runner.py'),
                           'performance','--timeout','1'],capture_output=True,text=True)
    assert result.returncode==1
    assert 'authentic pre-call capture' in result.stdout
    assert 'Environment:' not in result.stdout
