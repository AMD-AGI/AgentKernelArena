"""Real CPU negative controls for the four public vLLM MoE task contracts."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / 'tasks/triton2triton/vllm'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def function(path, name, **namespace):
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)
    env = {'torch': torch, **namespace}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), env)
    return env[name]


CONTRACT = load(TASKS/'triton_batched_moe/_contract_checks.py', '_moe_contract_test')
TIMER = load(ROOT/'src/tools/perf/aka_benchmark.py', '_moe_timer_test')


@pytest.mark.parametrize('failure', [None, 'original', 'stale', 'mutate', 'replay_mutate', 'dtype', 'shape', 'nonfinite'])
def test_actual_measured_output_and_replay_fail_closed(failure):
    x = torch.tensor([2., -3.], dtype=torch.float16)
    original = x.clone()
    calls = []
    def reference(saved):
        calls.append('reference')
        return saved['A'] * 4
    def fn():
        if failure == 'mutate':
            x.zero_()
        return x * 4
    def benchmark(call, *, timed_run, **options):
        assert options == {'warmup': 10, 'repetition': 100}
        calls.append('timer')
        actual = call()
        if failure == 'original': actual.zero_()
        if failure == 'dtype': actual = actual.float()
        if failure == 'shape': actual = actual[:1]
        if failure == 'nonfinite': actual.fill_(float('inf'))
        def replay():
            calls.append('replay')
            if failure == 'stale': return original * 4
            if failure == 'replay_mutate': x.zero_()
            actual.copy_(call())
            return actual
        timed_run._bind(replay, actual)
        return 0.25, {'benchmark_method': 'cuda_graph'}
    def run():
        with patch.dict(sys.modules, {'_aka_benchmark': TIMER}):
            return CONTRACT.checked_benchmark(benchmark, fn, inputs={'A': x}, reference=reference,
                check=CONTRACT.compare_output, perturb=CONTRACT.perturb_activation,
                warmup=10, repetition=100)
    if failure is None:
        ms, metadata = run()
        assert ms == 0.25
        assert metadata['benchmark_original_output_checked'] and metadata['benchmark_replay_checked']
        assert calls == ['reference','timer','reference','replay']
    else:
        with pytest.raises(AssertionError): run()
    torch.testing.assert_close(x, original, atol=0, rtol=0)
    assert calls[0] == 'reference'


def test_eager_input_mutation_cannot_change_oracle():
    x = torch.tensor([4.], dtype=torch.float16)
    def bad():
        x.zero_()
        return x.clone()
    with pytest.raises(AssertionError, match='Read-only input changed'):
        CONTRACT.checked_call(bad, inputs={'A':x}, reference=lambda s:s['A']*2, check=CONTRACT.compare_output)
    assert x.item()==4


def test_batched_known_answer_and_inactive_rows():
    ref = function(TASKS/'triton_batched_moe/scripts/task_runner.py', 'reference')
    inputs = {'A':torch.tensor([[[1.,2.],[3.,4.]], [[7.,8.],[9.,10.]]], dtype=torch.float16),
              'B':torch.tensor([[[5.,6.],[7.,8.]], [[1.,1.],[2.,2.]]], dtype=torch.float16),
              'counts':torch.tensor([1,0], dtype=torch.int32)}
    expected = torch.tensor([[[17.,23.],[0.,0.]],[[0.,0.],[0.,0.]]], dtype=torch.float16)
    torch.testing.assert_close(ref(inputs),expected,atol=0,rtol=0)
    for bad in [torch.zeros_like(expected),expected.flip(0),expected+1]:
        with pytest.raises(CONTRACT.NumericalMismatch): CONTRACT.compare_output(bad,expected)


def test_mmk_independent_known_answer():
    ref = function(TASKS/'triton_moe_mmk/scripts/task_runner.py', 'reference')
    inputs={'A':torch.tensor([[1.,2.],[-1.,3.]], dtype=torch.float16),
            'B':torch.tensor([[5.,-2.,1.],[7.,4.,3.]], dtype=torch.float16)}
    expected=torch.tensor([[19.,6.,7.],[16.,14.,8.]],dtype=torch.float16)
    torch.testing.assert_close(ref(inputs),expected,atol=0,rtol=0)
    with pytest.raises(CONTRACT.NumericalMismatch): CONTRACT.compare_output(torch.zeros_like(expected),expected)


@pytest.mark.parametrize('short,wrapper', [('batched_moe','batched_moe_gemm'),('moe_mmk','moe_matmul'),('fused_moe','fused_moe'),('fused_moe_gptq_awq','fused_moe_gptq_awq')])
def test_wrapper_cannot_bypass_declared_kernel(tmp_path, short, wrapper):
    from src.harness_guard import snapshot_workspace_harness, verify_workspace_harness
    task=TASKS/('triton_'+short)
    workspace=tmp_path/'task';shutil.copytree(task,workspace)
    snapshot=snapshot_workspace_harness(workspace)
    path=workspace/'source'/('triton_'+short+'.py');text=path.read_text()
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name==wrapper)
    lines=text.splitlines(keepends=True)
    lines[node.body[-1].lineno-1:node.body[-1].end_lineno]=['    return torch.zeros_like(A)\n']
    path.write_text(''.join(lines))
    with pytest.raises(RuntimeError): verify_workspace_harness(snapshot)


@pytest.mark.parametrize('short', ['batched_moe','moe_mmk','fused_moe','fused_moe_gptq_awq'])
def test_original_scored_cases_and_timing_intact(short):
    task=TASKS/('triton_'+short)
    manifest=json.loads((task/'workloads.json').read_text())
    measured=[r for r in manifest['cases'] if 'performance' in r['checks']]
    assert [r['test_case_id'] for r in measured]==[f'perf{i}' for i in range(1,6)]
    assert [r['params']['configuration'] for r in measured]==manifest['input_table']
    tree=ast.parse((task/'scripts/task_runner.py').read_text())
    assignments={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
                 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['WARMUP_ITERATIONS','BENCHMARK_ITERATIONS']}
    assert assignments=={'WARMUP_ITERATIONS':10,'BENCHMARK_ITERATIONS':100}


def test_mmk_partial_tile_pointer_indices_stay_within_matrix():
    from types import SimpleNamespace
    path=TASKS/'triton_moe_mmk/source/triton_moe_mmk.py'
    kernel=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='moe_mmk_kernel')
    expr=next(n.value for n in kernel.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='offs_n')
    actual=eval(compile(ast.Expression(expr),str(path),'eval'),{'pid_n':1,'BLOCK_N':64,'N':70,'tl':SimpleNamespace(arange=torch.arange)})
    assert actual.min()>=0 and actual.max()<70
    assert torch.equal(actual[:6],torch.arange(64,70))
    # The original operator precedence used local modulo, yielding indices 70..127.
    legacy=64+torch.arange(64)%70
    assert (legacy>=70).sum()==58


def test_batched_weight_load_masks_both_partial_dimensions():
    path=TASKS/'triton_batched_moe/source/triton_batched_moe.py'
    tree=ast.parse(path.read_text())
    load=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='b')
    mask=next(k.value for k in load.value.keywords if k.arg=='mask')
    actual=eval(compile(ast.Expression(mask),str(path),'eval'),{'offs_k':torch.arange(32),'K':35,'k':1,'BLOCK_K':32,'offs_n':torch.arange(64),'cta_n_size':6})
    assert actual.shape==(32,64) and actual.sum()==18
    assert not actual[:,6:].any() and not actual[3:].any()


def quant_oracles():
    path=TASKS/'triton_fused_moe_gptq_awq/scripts/task_runner.py'
    original=function(path,'reference_fused_moe_int4')
    return function(path,'reference',reference_fused_moe_int4=original), function(path,'control_inputs')


QUANT_ACCURACY = load(TASKS/'triton_fused_moe_gptq_awq/_numerical_contract.py', '_quant_accuracy_test')


def quant_contract_factory():
    path=TASKS/'triton_fused_moe_gptq_awq/scripts/task_runner.py'
    ref,_=quant_oracles()
    return function(path,'numerical_contract',reference=ref,
        reference_and_bound=QUANT_ACCURACY.reference_and_bound,
        assert_accuracy=QUANT_ACCURACY.assert_accuracy,NumericalMismatch=CONTRACT.NumericalMismatch,
        check_output=function(path,'check_output',compare_output=CONTRACT.compare_output),
        check_control_output=function(path,'check_control_output',compare_output=CONTRACT.compare_output))


def test_int4_reference_has_independent_exact_known_answer():
    ref,_=quant_oracles()
    inputs={'A':torch.tensor([[2.,0.,0.,0.],[0.,0.,0.,-3.]],dtype=torch.float16),
            'qweight':torch.tensor([[[0xA3,0xB4,0xC5,0xD6],[0xE7,0xF8,0x19,0x2A]]],dtype=torch.uint8),
            'scales':torch.ones(1,2,4,dtype=torch.float16),
            'zeros':torch.tensor([[[0x21,0x43],[0x21,0x43]]],dtype=torch.uint8),
            'ids':torch.zeros(2,1,dtype=torch.int32)}
    options={'use_int4':True,'group_size':2,'mul_routed_weight':False}
    expected=torch.tensor([[4.,4.,4.,4.],[-39.,-39.,6.,6.]],dtype=torch.float16)
    torch.testing.assert_close(ref(inputs,options),expected,atol=0,rtol=0)
    with pytest.raises(CONTRACT.NumericalMismatch): CONTRACT.compare_output(torch.zeros_like(expected),expected,atol=1.,rtol=.5)


@pytest.mark.parametrize('control',['int4_explicit','int4_default','int8_explicit','int8_default'])
def test_quant_controls_reject_zero_with_unchanged_large_tolerance(control):
    ref,inputs_for=quant_oracles();inputs,options=inputs_for(control,'cpu')
    expected=ref(inputs,options)
    assert expected.shape==(15,70) and expected.dtype==torch.float16
    with pytest.raises(CONTRACT.NumericalMismatch): CONTRACT.compare_output(torch.zeros_like(expected),expected,atol=1.,rtol=.5)
    # A scalar, float64 loop independently verifies packing, grouping and routes.
    actual=[];a=inputs['A'];q=inputs['qweight'];sc=inputs['scales'];ids=inputs['ids'];zp=inputs.get('zeros');w=inputs.get('weights')
    for m in range(a.shape[0]):
        for lane in range(ids.shape[1]):
            expert=int(ids[m,lane]);out=[]
            for n in range(70):
                value=0.
                if 0<=expert<q.shape[0]:
                    for k in range(a.shape[1]):
                        group=k//options['group_size']
                        quant=int(q[expert,k//2 if options['use_int4'] else k,n])
                        if options['use_int4']: quant=(quant>>(4*(k%2)))&15
                        if zp is None: zero=8 if options['use_int4'] else 128
                        else:
                            zero=int(zp[expert,group,n//2 if options['use_int4'] else n])
                            if options['use_int4']: zero=(zero>>(4*(n%2)))&15
                        value+=float(a[m,k])*(quant-zero)*float(sc[expert,group,n])
                    if options['mul_routed_weight'] and w is not None: value*=float(w[m*ids.shape[1]+lane])
                out.append(value)
            actual.append(out)
    torch.testing.assert_close(expected,torch.tensor(actual,dtype=torch.float16),atol=0,rtol=0)
    # The scalar oracle's dyadic results suffer no rounding when stored in FP16.
    torch.testing.assert_close(expected.double(),torch.tensor(actual,dtype=torch.float64),atol=0,rtol=0)
    ideal,bound=QUANT_ACCURACY.reference_and_bound(inputs,options)
    torch.testing.assert_close(ideal,torch.tensor(actual,dtype=torch.float64),atol=0,rtol=0)
    QUANT_ACCURACY.assert_accuracy(expected,ideal,bound,CONTRACT.NumericalMismatch)


def test_quant_control_rejects_wrong_nibble_zero_point_and_routing():
    ref,inputs_for=quant_oracles();inputs,options=inputs_for('int4_explicit','cpu');expected=ref(inputs,options)
    wrong_nibbles={**inputs,'qweight':((inputs['qweight'].to(torch.int32)>>4)|(inputs['qweight'].to(torch.int32)<<4)).to(torch.uint8)}
    no_zero={k:v for k,v in inputs.items() if k!='zeros'}
    wrong=[ref(wrong_nibbles,options),ref(no_zero,options),ref(inputs,{**options,'mul_routed_weight':False})]
    ideal,bound=QUANT_ACCURACY.reference_and_bound(inputs,options)
    for output in wrong:
        with pytest.raises(CONTRACT.NumericalMismatch): CONTRACT.compare_output(output,expected,atol=1.,rtol=.5)
        with pytest.raises(CONTRACT.NumericalMismatch):
            QUANT_ACCURACY.assert_accuracy(output,ideal,bound,CONTRACT.NumericalMismatch)


@pytest.mark.parametrize('control',['int4_explicit','int4_default','int8_explicit','int8_default'])
def test_exact_quant_control_rejects_scaling_and_single_element_errors(control):
    path=TASKS/'triton_fused_moe_gptq_awq/scripts/task_runner.py'
    check=function(path,'check_control_output',compare_output=CONTRACT.compare_output)
    legacy=function(path,'check_output',compare_output=CONTRACT.compare_output)
    ref,inputs_for=quant_oracles();inputs,options=inputs_for(control,'cpu')
    expected=ref(inputs,options)
    check(expected.clone(),expected)
    # Preserve and expose the old scored-case gate instead of silently changing it.
    legacy(expected*0.5,expected)
    one_element=expected.clone();one_element[0,0]+=0.0625
    invalid_expert=expected.clone();invalid_expert[3,0]=0.0625
    for wrong in [expected*0.5,expected*1.25,-expected,one_element,invalid_expert]:
        with pytest.raises(CONTRACT.NumericalMismatch): check(wrong,expected)
    # Exercise the real correctness dispatch, not only the new comparison helper.
    run=function(path,'run_correctness',load_module=lambda:None,CONTROL_CASES=(control,),
        control_inputs=lambda name,device:inputs_for(name,'cpu'),reference=ref,
        invoke=lambda mod,data,opts:ref(data,opts)*0.5,checked_call=CONTRACT.checked_call,
        numerical_contract=quant_contract_factory())
    ok,error=run(control=control)
    assert not ok and isinstance(error,CONTRACT.NumericalMismatch)


@pytest.mark.parametrize('index',range(5))
def test_all_original_quant_shapes_have_input_derived_accuracy_gate(index):
    shapes=json.loads((TASKS/'triton_fused_moe_gptq_awq/workloads.json').read_text())['input_table']
    m,k,e,n,topk,group_size=shapes[index]
    torch.manual_seed(42+index)
    inputs={'A':torch.randn(m,k,dtype=torch.float16)*0.1,
        'qweight':torch.randint(0,255,(e,k//2,n),dtype=torch.int32).to(torch.uint8),
        'scales':torch.randn(e,k//group_size,n,dtype=torch.float16).abs()*0.01+0.001,
        'zeros':torch.randint(0,255,(e,k//group_size,n//2),dtype=torch.int32).to(torch.uint8),
        'ids':torch.randint(0,e,(m,topk),dtype=torch.int32),
        'weights':torch.randn(m*topk,dtype=torch.float32).abs()}
    options={'use_int4':True,'group_size':group_size,'mul_routed_weight':True}
    ref,_=quant_oracles();oracle,check=quant_contract_factory()(options)
    expected=oracle(inputs);check(expected.clone(),expected)
    ideal,bound=QUANT_ACCURACY.reference_and_bound(inputs,options)
    assert bound.shape==expected.shape and (bound>=0).all()
    assert torch.linalg.vector_norm(bound) < 0.02*torch.linalg.vector_norm(ideal)
    # A legitimate FP16-dequantize / FP32-dot implementation is admitted by
    # the derived bound; this numerical model is not a measured GPU baseline.
    q=inputs['qweight'].to(torch.int32)
    unpack=torch.stack((q&15,q>>4),dim=2).reshape(e,k,n)
    zp=inputs['zeros'].to(torch.int32)
    zero=torch.stack((zp&15,zp>>4),dim=-1).reshape(e,k//group_size,n)
    dequant=((unpack-zero.repeat_interleave(group_size,dim=1))*
             inputs['scales'].float().repeat_interleave(group_size,dim=1)).half()
    rounded=torch.empty_like(expected)
    for token in range(m):
        for lane in range(topk):
            row=token*topk+lane;expert=int(inputs['ids'][token,lane])
            rounded[row]=(inputs['A'][token].float()@dequant[expert].float()*inputs['weights'][row]).half()
    check(rounded,expected)
    q=inputs['qweight'].to(torch.int32)
    wrong_nibbles={**inputs,'qweight':((q>>4)|(q<<4)).to(torch.uint8)}
    wrong_zero={key:value for key,value in inputs.items() if key!='zeros'}
    wrong_route={**inputs,'ids':(inputs['ids']+1)%e}
    wrong=[expected*0.5,expected+0.25,ref(wrong_nibbles,options),
           ref(wrong_zero,options),ref(wrong_route,options)]
    # All outputs are still required to meet the original broad allclose gate;
    # the new independent criterion rejects even the half/offset examples it admits.
    CONTRACT.compare_output(wrong[0],expected,atol=1.,rtol=.5)
    CONTRACT.compare_output(wrong[1],expected,atol=1.,rtol=.5)
    for actual in wrong:
        with pytest.raises(CONTRACT.NumericalMismatch):
            QUANT_ACCURACY.assert_accuracy(actual,ideal,bound,CONTRACT.NumericalMismatch)


@pytest.mark.parametrize('failure',['original','replay'])
def test_quant_performance_dispatch_rejects_half_scaled_measured_path(failure):
    path=TASKS/'triton_fused_moe_gptq_awq/scripts/task_runner.py'
    shapes=json.loads((path.parent.parent/'workloads.json').read_text())['input_table']
    ref,_=quant_oracles()
    def benchmark(fn, *, timed_run, **options):
        assert options=={'warmup':10,'repetition':100,'use_cuda_graph':False,
                        'fallback_reason':'fused_moe_host_routing_and_dynamic_allocations'}
        actual=fn()
        if failure=='original': actual*=0.5
        def replay():
            actual.copy_(fn()*0.5 if failure=='replay' else fn())
            return actual
        timed_run._bind(replay,actual)
        return 0.25,{'benchmark_method':'cuda_event_fallback'}
    run=function(path,'run_performance',load_module=lambda:None,TEST_SHAPES=shapes[:1],
        WARMUP_ITERATIONS=10,BENCHMARK_ITERATIONS=100,numerical_contract=quant_contract_factory(),
        checked_benchmark=CONTRACT.checked_benchmark,perturb_activation=CONTRACT.perturb_activation,
        _benchmark_cuda_graph_or_events=benchmark,invoke=lambda mod,data,opts:ref(data,opts))
    randn,randint=torch.randn,torch.randint
    def cpu_randn(*args,**kwargs): return randn(*args,**{**kwargs,'device':'cpu'})
    def cpu_randint(*args,**kwargs): return randint(*args,**{**kwargs,'device':'cpu'})
    with patch.object(torch,'randn',cpu_randn),patch.object(torch,'randint',cpu_randint),patch.dict(sys.modules,{'_aka_benchmark':TIMER}):
        rows=run()
    assert len(rows)==1 and rows[0]['execution_time_ms']==-1.
    assert rows[0]['failure_kind']=='numerical_mismatch'
    assert 'Arithmetic accuracy bound exceeded' in rows[0]['error']


@pytest.mark.parametrize('index',range(5))
def test_quant_correctness_dispatch_rejects_half_scaled_original_case(index):
    path=TASKS/'triton_fused_moe_gptq_awq/scripts/task_runner.py'
    shapes=json.loads((path.parent.parent/'workloads.json').read_text())['input_table']
    ref,_=quant_oracles()
    run=function(path,'run_correctness',load_module=lambda:None,TEST_SHAPES=shapes,
        numerical_contract=quant_contract_factory(),checked_call=CONTRACT.checked_call,
        invoke=lambda mod,data,opts:ref(data,opts)*0.5)
    randn,randint=torch.randn,torch.randint
    def cpu_randn(*args,**kwargs): return randn(*args,**{**kwargs,'device':'cpu'})
    def cpu_randint(*args,**kwargs): return randint(*args,**{**kwargs,'device':'cpu'})
    with patch.object(torch,'randn',cpu_randn),patch.object(torch,'randint',cpu_randint):
        ok,error=run(case_index=index)
    assert not ok and isinstance(error,CONTRACT.NumericalMismatch)
    assert 'Arithmetic accuracy bound exceeded' in str(error)


def test_int8_exact_known_answer_including_default_zero():
    ref,_=quant_oracles()
    inputs={'A':torch.tensor([[2.,-1.]],dtype=torch.float16),
            'qweight':torch.tensor([[[130,124],[131,136]]],dtype=torch.uint8),
            'scales':torch.ones(1,1,2,dtype=torch.float16), 'ids':torch.zeros(1,1,dtype=torch.int32)}
    options={'use_int4':False,'group_size':2,'mul_routed_weight':True}
    torch.testing.assert_close(ref(inputs,options),torch.tensor([[1.,-16.]],dtype=torch.float16),atol=0,rtol=0)


def test_fused_oracle_optional_weights_and_invalid_experts():
    path=TASKS/'triton_fused_moe/scripts/task_runner.py'
    original=function(path,'reference_fused_moe');ref=function(path,'reference',reference_fused_moe=original)
    inputs={'A':torch.tensor([[1.,2.]],dtype=torch.float16),
            'B':torch.tensor([[[3.,4.]],[[5.,6.]]],dtype=torch.float16),
            'ids':torch.tensor([[0,1,-1,2]],dtype=torch.int32)}
    expected=torch.tensor([[11.],[17.],[0.],[0.]],dtype=torch.float16)
    torch.testing.assert_close(ref(inputs,{'mul_routed_weight':True}),expected,atol=0,rtol=0)
    routed={**inputs,'weights':torch.tensor([-2.,3.,7.,9.])}
    torch.testing.assert_close(ref(routed,{'mul_routed_weight':False}),expected,atol=0,rtol=0)
    weighted=ref(routed,{'mul_routed_weight':True})
    torch.testing.assert_close(weighted,torch.tensor([[-22.],[51.],[0.],[0.]],dtype=torch.float16),atol=0,rtol=0)
    with pytest.raises(CONTRACT.NumericalMismatch): CONTRACT.compare_output(expected,weighted)


def test_quant_weight_load_uses_existing_partial_k_mask():
    path=TASKS/'triton_fused_moe_gptq_awq/source/triton_fused_moe_gptq_awq.py'
    loads=[n.value for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.Assign)
           and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='b'
           and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Attribute) and n.value.func.attr=='load']
    assert len(loads)==1
    mask=next(k.value for k in loads[0].keywords if k.arg=='mask')
    # Apply the actual source expression to a final 16-valid/16-invalid K tile.
    selected=eval(compile(ast.Expression(mask),str(path),'eval'),{'k_mask':torch.arange(32)[:,None]<16})
    assert selected[:16].all() and not selected[16:].any()


@pytest.mark.parametrize('error,kind', [(CONTRACT.NumericalMismatch('known numerical defect'),'numerical_mismatch'),
                                      (AssertionError('wrong output dtype'),'correctness_failure')])
def test_public_envelope_keeps_numerical_and_contract_failures_distinct(error,kind):
    from types import SimpleNamespace
    from src.task_protocol import parse_command_result
    task=TASKS/'triton_fused_moe_gptq_awq'
    adapter=load(task/'_arena_eval.py','_quant_moe_adapter_test')
    manifest=json.loads((task/'workloads.json').read_text())
    harness=SimpleNamespace(TEST_SHAPES=manifest['input_table'],
        CONTROL_CASES=tuple(row['params']['control'] for row in manifest['cases'] if 'control' in row['params']),
        run_correctness=lambda **kwargs:(False,error))
    with patch.object(adapter,'load_harness',return_value=harness):
        report=adapter.evaluate('baseline','correctness')
    parsed=parse_command_result('ARENA_EVAL_RESULT='+json.dumps(report),role='baseline',action='correctness',returncode=1)
    assert parsed.status=='FAIL' and parsed.failure_kind==kind
    assert len(parsed.cases)==9 and all(row['failure_kind']==kind for row in parsed.cases)
