"""CPU tests for the exact attention replay and native source/cache boundaries."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
from types import ModuleType, SimpleNamespace

import pytest

torch=pytest.importorskip('torch')
TASK=Path(__file__).resolve().parents[1]/'tasks/headkernel/qwen3.8-2.4t__paged_attention_decode'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module);return module


inputs=load('_attention_inputs',TASK/'scripts/_attention_inputs.py')
contract=load('_graph_contract',TASK/'scripts/_graph_contract.py')
bench=load('qwen_attention_bench',TASK/'scripts/_bench.py')
cpp=load('qwen_attention_cpp_guard',TASK/'ut/cpp_body_guard.py')


def small_args():
    return (torch.zeros((2,2,2),dtype=torch.bfloat16), torch.zeros(64,dtype=torch.uint8),
        torch.zeros((2,2,2),dtype=torch.bfloat16),
        torch.tensor([[1,0],[0,1],[-1,0],[0,-1]],dtype=torch.bfloat16).reshape(4,1,1,2),
        torch.tensor([[2,4],[6,8],[10,12],[14,16]],dtype=torch.bfloat16).reshape(4,1,1,2),
        1.0,torch.tensor([0,2,4],dtype=torch.int32),torch.tensor([1,0,3,2],dtype=torch.int32),
        torch.ones(3,dtype=torch.int32),1,44,None,'auto','NHD',0.0,
        torch.ones(1),torch.ones(1),None,256)


def test_zero_query_is_exact_ragged_value_mean():
    args=small_args();source=inputs.CPUInputs(torch,args,5)
    output=source.reference(args[2])
    assert output.tolist()==[[[4.,6.],[4.,6.]],[[12.,14.],[12.,14.]]]


def test_softmax_uses_captured_page_mapping_and_scale():
    args=list(small_args());args[2][0,0]=torch.tensor([1.,0.],dtype=torch.bfloat16)
    source=inputs.CPUInputs(torch,args,5);output=source.reference(args[2])
    expected=torch.tensor([(6+2*math.e)/(1+math.e),(8+4*math.e)/(1+math.e)],dtype=torch.bfloat16)
    assert output[0,0].equal(expected)


def test_fresh_queries_and_all_challenges_keep_kv_and_metadata_fixed():
    args=small_args();source=inputs.CPUInputs(torch,args,8)
    before={name:args[index].clone() for name,index in inputs.IMMUTABLE.items() if name!='query'}
    first,_=source.next();second,_=source.next()
    assert not first.equal(second)
    for kind in inputs.CHALLENGES:
        query,expected=source.next(kind)
        assert query.device.type==expected.device.type=='cpu'
        assert torch.isfinite(expected).all()
        if kind=='zero_query':assert query.eq(0).all()
    assert all(args[inputs.IMMUTABLE[name]].equal(value) for name,value in before.items())


@pytest.mark.parametrize('change',['bad_page','truncated_indptr','wrong_block','gpu_truth'])
def test_invalid_or_different_captured_contract_rejects(change):
    args=list(small_args())
    if change=='bad_page':args[7][0]=999
    elif change=='truncated_indptr':args[6][-1]=3
    elif change=='wrong_block':args[9]=16
    else:args[2]=torch.empty_like(args[2],device='meta')
    with pytest.raises(ValueError):inputs.CPUInputs(torch,args,1)


def compare(actual,expected,tolerance):
    error=float((actual.float()-expected.float()).abs().max())
    return bool(torch.allclose(actual,expected,rtol=tolerance,atol=tolerance)),error


def state_for(monkeypatch,mode='correct'):
    fixed=small_args();device=tuple(x.clone() if isinstance(x,torch.Tensor) else x for x in fixed)
    oracle=inputs.CPUInputs(torch,fixed,2);cache=[]
    def call(payload):
        args=payload['positional']
        if mode=='noop':return args[0]
        result=oracle.reference(args[2])
        if mode=='memoized':
            if not cache:cache.append(result.clone())
            result=cache[0]
        args[0].copy_(result);args[1].zero_()
        if mode=='mutate_kv':args[3][0].add_(1)
        return args[0]
    monkeypatch.setattr(torch.cuda,'is_current_stream_capturing',lambda:False)
    return bench.ReplayState(torch,SimpleNamespace(call=call),SimpleNamespace(correct=compare),
                             device,fixed,0.02,31000)


def test_output_and_scratch_are_poisoned_before_every_replay(monkeypatch):
    state=state_for(monkeypatch)
    state.prepare()
    assert state.args[0].isnan().all() and state.args[1].eq(255).all()
    state.run();state.prepare()
    assert state.checked_invocations==1
    assert state.args[0].isnan().all() and state.args[1].eq(255).all()
    state.run();state.validate();assert state.checked_invocations==2


@pytest.mark.parametrize('mode',['noop','mutate_kv'])
def test_skipped_output_or_changed_kv_rejects(monkeypatch,mode):
    state=state_for(monkeypatch,mode);state.prepare();state.run()
    with pytest.raises(RuntimeError):state.validate()


def test_private_output_memoization_fails_after_query_changes(monkeypatch):
    state=state_for(monkeypatch,'memoized')
    state.prepare();state.run();state.validate()
    state.prepare();state.run()
    with pytest.raises(RuntimeError,match='failed tolerance'):state.validate()


MARKERS=['__inline__ __device__ void\n_paged_attention_kernel(',
         '__inline__ __device__ void _paged_attention_ll4mi_reduce_kernel(']


def changed_header(original,body):
    start,_end=cpp.span(original,MARKERS[0])
    return original[:start]+body+original[start:]


def test_actual_native_device_body_edit_is_allowed():
    original=(TASK/'source/pa_kernels.cuh').read_text()
    cpp.validate_cpp(changed_header(original,'\nconst int candidate_value = 1;\n'),original,MARKERS)


@pytest.mark.parametrize('body',['\n}\nvoid host_callback() {\n',
    '\n#include "replacement.h"\n','\n__attribute__((constructor)) void attack();\n'])
def test_cpp_body_cannot_escape_into_host_or_preprocessing(body):
    original=(TASK/'source/pa_kernels.cuh').read_text()
    with pytest.raises(ValueError):cpp.validate_cpp(changed_header(original,body),original,MARKERS)


@pytest.fixture
def native_package(tmp_path,monkeypatch):
    root=tmp_path/'task';shutil.copytree(TASK,root,symlinks=True,ignore=shutil.ignore_patterns('build','__pycache__'))
    build=tmp_path/'compiled';calls=[]
    def compile_template(template,name,includes,**kwargs):
        folder=build/kwargs['func_name'];(folder/'include').mkdir(parents=True,exist_ok=True)
        source=next(Path(p) for p in includes if Path(p).name=='pa_kernels.cuh')
        shutil.copy2(source,folder/'include/pa_kernels.cuh')
        calls.append(kwargs['func_name'])
        return lambda *args:None
    utils=ModuleType('csrc.cpp_itfs.utils')
    utils.AITER_CORE_DIR=str(tmp_path/'image');utils.BUILD_DIR=str(build)
    utils.compile_template_op=compile_template;utils.str_to_bool=lambda value:value=='true'
    for name in ('csrc','csrc.cpp_itfs'):
        package=ModuleType(name);package.__path__=[];monkeypatch.setitem(sys.modules,name,package)
    monkeypatch.setitem(sys.modules,'csrc.cpp_itfs.utils',utils)
    module=load('private_native_attention_test',root/'ut/native/pa_ragged.py')
    return root,module,calls


def test_native_build_name_changes_with_submitted_device_body(native_package):
    root,module,calls=native_package
    module.compile(8,256,1,'__hip_bfloat16','__hip_bfloat16','auto','__hip_bfloat16',1)
    source=root/'source/pa_kernels.cuh'
    source.write_text(changed_header(source.read_text(),'\nconst int candidate_value = 1;\n'))
    module.compile(8,256,1,'__hip_bfloat16','__hip_bfloat16','auto','__hip_bfloat16',1)
    assert len(calls)==2 and calls[0]!=calls[1]
    receipt=json.loads((root/'build/native_compile_report.json').read_text())
    assert receipt['compiled_header_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()


def test_native_route_cannot_change_recorded_abi(native_package):
    _root,module,calls=native_package
    with pytest.raises(ValueError,match='recorded ABI'):
        module.compile(8,256,1,'__hip_bfloat16','__hip_bfloat16','auto','__hip_bfloat16',16)
    assert not calls


def test_wrong_cached_header_is_not_accepted(native_package,monkeypatch):
    root,module,_calls=native_package
    def wrong(*args,**kwargs):
        folder=Path(module.BUILD_DIR)/kwargs['func_name']/'include';folder.mkdir(parents=True)
        (folder/'pa_kernels.cuh').write_text('unrelated device program')
        return lambda *args:None
    monkeypatch.setattr(module,'compile_template_op',wrong)
    with pytest.raises(RuntimeError,match='Compiled device header differs'):
        module.compile(8,256,1,'__hip_bfloat16','__hip_bfloat16','auto','__hip_bfloat16',1)


@pytest.mark.parametrize('attack',['host_callback','alias_replacement'])
def test_frozen_host_wrapper_and_source_alias_reject(native_package,attack):
    root,_module,_calls=native_package
    load('cpp_body_guard',root/'ut/cpp_body_guard.py')
    guard=load('attention_frozen_source_guard',root/'ut/source_guard.py')
    guard.validate_sources(root,root)
    if attack=='host_callback':
        path=root/'source/attention_candidate.py'
        path.write_text(path.read_text()+'\ntorch.cuda.Event.elapsed_time = lambda *args: 0.001\n')
    else:
        path=root/'ut/kernel_src/attention_candidate.py'
        body=path.read_bytes();path.unlink();path.write_bytes(body)
    with pytest.raises(ValueError,match='Frozen host/build source|source routing alias'):
        guard.validate_sources(root,root)


def complete_report():
    policy=contract.contract(TASK);case=policy['cases'][0]
    pins=contract.provenance(TASK,policy)
    validation={'checked_invocations':118,'timing_checked_invocations':114,'fresh_input_sets':120,
        'input_seed':12345,'expected_device':'cpu','fresh_inputs_per_replay':True,
        'measured_graph_validated':True,'immutable_inputs_unchanged':True,'returned_aliases_checked':True,
        'tolerance':0.02,'max_rel_err':0.001,
        'correctness_challenges':[{'kind':k,'passed':True} for k in policy['challenges']],
        **policy['validation_contract']}
    source=pins['source_sha256']['source/pa_kernels.cuh']
    request={'schema_version':1,'phase':'performance','request_id':'c'*48,**pins,
             'manifest_sha256':hashlib.sha256((TASK/'ut/meta.json').read_bytes()).hexdigest()}
    return {'timer':'cuda_graph','warmup':10,'iters':100,**pins,
        'request':request,
        'source_binding':{'launcher_source_sha256':hashlib.sha256((TASK/'source/attention_candidate.py').read_bytes()).hexdigest(),
            'device_source_sha256':source,'device_kernel':'paged_attention_ll4mi_QKV_mfma16_kernel'},
        'cases':[{'sig':case['test_case_id'],'params':case['params'],
            'benchmark_method':'cuda_graph','benchmark_warmup':10,'benchmark_samples':100,
            'benchmark_effective_repeats':1,'samples_ms':[1.0]*100,'mean_ms':1.0,'median_ms':1.0,'min_ms':1.0,
            'validation':validation}]}


@pytest.mark.parametrize('attack',['none','missing_case','missing_sample','no_poison','wrong_abi','wrong_binary','missing_request','wrong_metadata'])
def test_complete_attention_report_is_required(attack):
    report=complete_report();row=report['cases'][0]
    if attack=='none':
        assert len(contract.benchmark_cases(report,TASK))==1
        return
    if attack=='missing_case':report['cases']=[]
    elif attack=='missing_sample':row['samples_ms'].pop()
    elif attack=='no_poison':row['validation']['poisoned_buffers']=[]
    elif attack=='wrong_abi':row['params']={**row['params'],'block_size':16}
    elif attack=='wrong_binary':report['source_binding']['device_source_sha256']='0'*64
    elif attack=='missing_request':report.pop('request')
    else:report['request']['manifest_sha256']='0'*64
    with pytest.raises(ValueError):contract.benchmark_cases(report,TASK)
