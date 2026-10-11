"""CPU regressions for recurrent math, per-replay reset and complete reports."""
import copy
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

torch = pytest.importorskip('torch')
TASK = Path(__file__).resolve().parents[1]/'tasks/headkernel/qwen3.8-2.4t__fused_recurrent_gated_delta_rule_decode'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


inputs = load('_recurrent_inputs', TASK/'scripts/_recurrent_inputs.py')
contract = load('_graph_contract', TASK/'scripts/_graph_contract.py')
bench = load('qwen_recurrent_graph_bench', TASK/'scripts/_bench.py')
GEO = {'H':1, 'HV':2, 'K':2, 'V':1, 'qkv_dim':6, 'state_pool_slots':3,
       'scale':0.5, 'use_qk_l2norm_in_kernel':False}
SPEC = {'B':2, 'm':2, 'regime':'decode', 'sig':'small_test', 'state_indices':[1,2]}


def fixed_args():
    source = inputs.CPUInputs(torch, SPEC, GEO, 3)
    args, _ = source.next()
    args['mixed_qkv'][:] = torch.tensor([1,0,0,1,7,11], dtype=torch.bfloat16)
    args['a'].zero_(); args['b'].zero_(); args['A_log'].zero_(); args['dt_bias'].zero_()
    args['initial_state'].copy_(torch.arange(12,dtype=torch.float32).reshape(3,2,1,2))
    args['initial_state'][1,0,0] = torch.tensor([2.,3.])
    args['initial_state'][1,1,0] = torch.tensor([4.,5.])
    args['ssm_state_indices'][:] = torch.tensor([1,-1],dtype=torch.int32)
    return args


def test_closed_form_state_update_and_inactive_row():
    args = fixed_args()
    before = args['initial_state'].clone()
    out, state = inputs.cpu_reference(torch,args)
    assert out[0,0,:,0].tolist() == [0.5,1.0]
    assert out[1].eq(0).all()
    assert state[1,:,0].tolist() == [[1.,4.25],[2.,6.75]]
    assert torch.equal(state[[0,2]],before[[0,2]])
    assert torch.equal(args['initial_state'],before), 'Oracle mutated its input state'


def test_native_beta_rounds_through_bfloat16_before_state_update():
    args = fixed_args()
    args['initial_state'].zero_()
    args['mixed_qkv'][:] = torch.tensor([1,0,1,0,1,1],dtype=torch.bfloat16)
    args['b'].fill_(0.2)
    _out,state = inputs.cpu_reference(torch,args)
    beta = torch.sigmoid(args['b'][0,0].float()).to(torch.bfloat16).float()
    assert state[1,0,0,0] == beta
    assert state[1,0,0,0] != torch.sigmoid(args['b'][0,0].float())


def test_oracle_refuses_non_cpu_truth_inputs():
    args = fixed_args()
    args['a'] = torch.empty_like(args['a'],device='meta')
    with pytest.raises(ValueError,match='CPU inputs only'):
        inputs.cpu_reference(torch,args)


def test_fresh_inputs_preserve_contract_and_cover_numerical_challenges():
    source = inputs.CPUInputs(torch,SPEC,GEO,7)
    first,_ = source.next(); second,_ = source.next()
    assert not first['mixed_qkv'].equal(second['mixed_qkv'])
    assert not first['initial_state'].equal(second['initial_state'])
    assert first['ssm_state_indices'].tolist() == second['ssm_state_indices'].tolist() == [1,2]
    for kind in inputs.CHALLENGES:
        args,expected = source.next(kind)
        assert all(x.device.type=='cpu' and torch.isfinite(x).all() for x in expected)
        if kind=='zero_query_key': assert expected[0].eq(0).all()
        if kind=='zero_state': assert expected[1][0].eq(0).all()
        if kind=='inactive_last_row': assert expected[0][-1].eq(0).all()


def test_recorded_state_routes_cannot_be_replaced_by_metadata():
    args=fixed_args()  # The second recorded row is inactive, unlike SPEC.
    case={'args':{'kwargs':args},'sig':SPEC['sig'],'regime':SPEC['regime'],'m':SPEC['m']}
    expected=inputs.case_params(args,SPEC,GEO)
    with pytest.raises(RuntimeError,match='Captured recurrent state routing'):
        bench.case_params(case,SPEC,GEO,expected)


def compare(actual, expected, tolerance):
    error = float((actual.float()-expected.float()).abs().max())
    return bool(torch.allclose(actual,expected,atol=tolerance,rtol=tolerance)),error


def replay_state(monkeypatch, call):
    args,_ = inputs.CPUInputs(torch,SPEC,GEO,100).next()
    monkeypatch.setattr(torch.cuda,'is_current_stream_capturing',lambda:False)
    return bench.ReplayState(torch,SimpleNamespace(call=call),SimpleNamespace(correct=compare),
                             args,SPEC,GEO,0.02,31000)


def correct_call(payload):
    args = payload['kwargs']
    out,state = inputs.cpu_reference(torch,args)
    args['out'].copy_(out); args['initial_state'].copy_(state)
    return args['out'],args['initial_state']


def test_previous_replay_is_checked_before_fresh_state_reset(monkeypatch):
    state = replay_state(monkeypatch,correct_call)
    state.prepare()
    assert state.args['out'].isnan().all()
    assert state.args['initial_state'].equal(state.cpu_inputs['initial_state'])
    state.run()
    state.prepare()
    assert state.checked_invocations == 1
    assert state.args['out'].isnan().all()
    assert state.args['initial_state'].equal(state.cpu_inputs['initial_state'])
    state.run(); state.validate()
    assert state.checked_invocations == 2


def test_private_first_result_cache_fails_on_second_actual_replay(monkeypatch):
    cache=[]
    def memoized(payload):
        args=payload['kwargs']
        if not cache: cache.extend(x.clone() for x in correct_call(payload))
        args['out'].copy_(cache[0]); args['initial_state'].copy_(cache[1])
        return args['out'],args['initial_state']
    state=replay_state(monkeypatch,memoized)
    state.prepare(); state.run(); state.validate()
    state.prepare(); state.run()
    with pytest.raises(RuntimeError,match='unselected state slot|failed tolerance'):
        state.validate()


def test_untouched_state_corruption_is_not_hidden_by_output_correctness(monkeypatch):
    def corrupt(payload):
        result=correct_call(payload)
        result[1][0].add_(1)
        return result
    state=replay_state(monkeypatch,corrupt)
    state.prepare(); state.run()
    with pytest.raises(RuntimeError,match='unselected state slot'):
        state.validate()


def valid_report():
    policy=contract.contract(TASK)
    rows=[]
    for case in policy['cases']:
        rows.append({'sig':case['test_case_id'],'params':case['params'],
            'benchmark_method':'cuda_graph','benchmark_warmup':10,'benchmark_samples':100,
            'benchmark_effective_repeats':1,'samples_ms':[1.0]*100,'mean_ms':1.0,'median_ms':1.0,'min_ms':1.0,
            'validation':{'checked_invocations':118,'timing_checked_invocations':114,'fresh_input_sets':120,
                'input_seed':12345,
                'expected_device':'cpu','fresh_inputs_per_replay':True,'measured_graph_validated':True,
                'immutable_inputs_unchanged':True,'returned_aliases_checked':True,'tolerance':0.02,
                'max_rel_err':0.001,'max_mixed_error_ratio':0.1,'correctness_challenges':[{'kind':k,'passed':True} for k in policy['challenges']],
                **policy['validation_contract']}})
    pins=contract.provenance(TASK,policy)
    source=pins['source_sha256']['source/fused_recurrent.py']
    request={'schema_version':1,'phase':'performance','request_id':'c'*48,**pins,
             'manifest_sha256':hashlib.sha256((TASK/'ut/meta.json').read_bytes()).hexdigest()}
    return {'timer':'cuda_graph','warmup':10,'iters':100,'cases':rows,**pins,
            'request':request,
            'source_binding':{'launcher_source_sha256':source,'device_source_sha256':source,
                              'device_kernel':'fused_recurrent_gated_delta_rule_packed_decode_kernel'}}


@pytest.mark.parametrize('attack',['missing_case','dropped_sample','gpu_truth','no_reset','wrong_source','wrong_mean','event','missing_request','wrong_metadata','bool_dimension'])
def test_incomplete_or_forged_report_contract_rejects(attack):
    report=valid_report();row=report['cases'][0]
    if attack=='missing_case':report['cases'].pop()
    elif attack=='dropped_sample':row['samples_ms'].pop()
    elif attack=='gpu_truth':row['validation']['expected_device']='cuda'
    elif attack=='no_reset':row['validation']['restored_buffers']=[]
    elif attack=='wrong_source':report['source_sha256']['source/fused_recurrent.py']='0'*64
    elif attack=='wrong_mean':row['mean_ms']=0.001
    elif attack=='event':row['benchmark_method']='cuda_event'
    elif attack=='missing_request':report.pop('request')
    elif attack=='wrong_metadata':report['request']['manifest_sha256']='0'*64
    else:row['params']={**row['params'],'m':True}
    with pytest.raises(ValueError):contract.benchmark_cases(report,TASK)


def test_complete_report_retains_both_ordered_raw_series():
    report=valid_report(); rows=contract.benchmark_cases(report,TASK)
    assert len(rows)==2
    assert [r['samples_ms'] for r in rows]==[r['samples_ms'] for r in report['cases']]


def test_diagnostic_relative_error_is_not_the_mixed_tolerance_threshold():
    report=valid_report()
    report['cases'][0]['validation'].update(max_rel_err=0.06,max_mixed_error_ratio=0.4)
    assert len(contract.benchmark_cases(report,TASK))==2
    report['cases'][0]['validation']['max_mixed_error_ratio']=1.01
    with pytest.raises(ValueError,match='mixed-tolerance'):
        contract.benchmark_cases(report,TASK)


@pytest.mark.parametrize('body',[
    'torch.cuda.Event.elapsed_time = lambda *args: 0.001',
    'value = globals()',
    'import os',
    'value = tl.__dict__',
])
def test_editable_jit_body_cannot_execute_host_callbacks(body):
    load('cpp_body_guard',TASK/'ut/cpp_body_guard.py')
    guard=load('closed_recurrent_source_guard',TASK/'ut/source_guard.py')
    original=(TASK/'source/fused_recurrent.py').read_text()
    tree=ast.parse(original)
    kernel=next(n for n in tree.body if isinstance(n,ast.FunctionDef)
                and n.name=='fused_recurrent_gated_delta_rule_packed_decode_kernel')
    kernel.body=ast.parse(body).body
    with pytest.raises(ValueError):
        guard.validate_python(ast.unparse(tree),original,[kernel.name],'triton')
