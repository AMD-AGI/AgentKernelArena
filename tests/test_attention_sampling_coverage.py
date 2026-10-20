"""Finite CPU source/oracle models; native Triton and graphs require fresh GPU checks."""
import ast
from contextlib import contextmanager
import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT/'tasks/triton2triton/vllm'
CP = 'triton_correct_attn_cp_out'
GROUP = 'triton_decode_attn_grouped_stage1'
REJECT = 'triton_rejection_sample'


def load(path, name='_coverage_cpu'):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    cwd = Path.cwd()
    try:
        spec.loader.exec_module(module)
    finally:
        import os
        os.chdir(cwd)
    return module


def functions(path, names, namespace):
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    for node in nodes:
        node.decorator_list = []
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace


class Launch:
    def __init__(self, call): self.call = call
    def __getitem__(self, grid):
        return lambda *args,**kwargs:self.call(grid,*args,**kwargs)


def cp_wrapper(defect=None):
    def device(grid, out, destination, lses, lse_out, *strides, HEAD_DIM, N_ROUNDED, IS_BASE_E):
        assert out.ndim == lses.ndim == 3
        assert grid == (out.shape[0],out.shape[1],1)
        assert HEAD_DIM == out.shape[-1] and N_ROUNDED == lses.shape[0]
        rank = strides[-1]
        natural = IS_BASE_E if defect != 'base_e_only' else True
        if defect == 'rank_zero': rank = 0
        scale = 1.0 if natural else math.log(2)
        logits = lses.float()*scale
        # Independent normalized-weight identity, rather than copying the
        # protected oracle's max/subtract/exp/sum implementation.
        destination.copy_((out.float()*torch.softmax(logits,dim=0)[rank,:, :,None]).to(out.dtype))
        lse_out.copy_(torch.logsumexp(logits,dim=0)/scale)
    namespace = {'torch':torch,'_correct_attn_cp_out_kernel':Launch(device)}
    functions(TASKS/CP/'source/triton_correct_attn_cp_out.py',{'correct_attn_cp_out'},namespace)
    return SimpleNamespace(correct_attn_cp_out=namespace['correct_attn_cp_out'])


@pytest.mark.parametrize('case',('base2_nonzero_rank','basee_4d_trailing_lse','base2_4d_middle_lse'))
@pytest.mark.parametrize('defect',(None,'base_e_only','rank_zero'))
def test_cp_real_wrapper_controls_use_normalized_layouts_full_oracle_and_readonly(case,defect,monkeypatch):
    task=TASKS/CP;checks=load(task/'_arena_checks.py');h=load(task/'scripts/task_runner.py')
    args,kwargs=checks.control_inputs(h,case,'cpu');config=checks.CONTRACT_CASES[case]
    assert list(args[0].shape)==config['input_shapes']['out']
    assert list(args[1].shape)==config['input_shapes']['lses']
    plan=checks.CallPlan(h,args,kwargs);model=cp_wrapper(defect)
    output=model.correct_attn_cp_out(*args,**kwargs)
    wrong=defect=='rank_zero' or defect=='base_e_only' and not kwargs['is_base_e']
    if wrong:
        with pytest.raises(AssertionError):plan.check(output)
    else:
        plan.check(output);plan.unchanged()
        assert list(output[0].shape)==config['output_shapes']['corrected']
        assert list(output[1].shape)==config['output_shapes']['final_lse']
    # Actual protected run_contract_correctness dispatch, with only native JIT
    # execution replaced by the explicitly labeled CPU device model above.
    h.load_module=lambda:model
    original=checks.control_inputs
    monkeypatch.setattr(checks,'control_inputs',lambda h,c:original(h,c,'cpu'))
    monkeypatch.setattr(checks,'checked_candidate_call',lambda m,fn,*a,**kw:fn(*a,**kw))
    checks.install_controls(h)
    if wrong:
        with pytest.raises(AssertionError):h.run_contract_correctness(case)
    else:
        ok,evidence=h.run_contract_correctness(case)
        assert ok and evidence['full_outputs_checked'] and evidence['readonly_inputs_checked']


def head_mapping(source, heads, kv_heads):
    """Execute the actual wrapper grid and actual kernel head-index AST on CPU."""
    tree=ast.parse(source.read_text())
    kernel=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='_fwd_grouped_kernel_stage1')
    prefix=[]
    for node in kernel.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='offs_d' for t in node.targets):break
        prefix.append(node)
    code=compile(ast.Module(body=prefix,type_ignores=[]),'actual_grouped_head_indices','exec')
    observed=[];grids=[]
    def capture(grid,*args,**kw):
        grids.append(grid)
        for pid in range(grid[1]):
            tl=SimpleNamespace(constexpr=int,cdiv=lambda a,b:(a+b-1)//b,arange=torch.arange,
                               program_id=lambda axis:(0,pid,0)[axis])
            scope={**kw,'tl':tl};exec(code,scope)
            for head in scope['cur_head'][scope['mask_h']].tolist():
                observed.append((head,int(scope['cur_kv_head'])))
    ns={'triton':SimpleNamespace(next_power_of_2=lambda x:1<<(x-1).bit_length(),cdiv=lambda a,b:(a+b-1)//b),
        '_fwd_grouped_kernel_stage1':Launch(capture)}
    functions(source,{'decode_grouped_att_m_fwd'},ns)
    q=torch.empty(1,heads,64);k=torch.empty(16,kv_heads,64);v=torch.empty_like(k)
    ns['decode_grouped_att_m_fwd'](q,k,v,torch.empty(1,heads,1,65),torch.zeros(1,16,dtype=torch.int32),
                                 torch.tensor([16],dtype=torch.int32),1,1/8,16)
    return observed,grids[0]


@pytest.mark.parametrize('group',tuple(range(1,33))+(40,48,64))
@pytest.mark.parametrize('kv_heads',(1,2,4))
def test_grouped_actual_grid_and_kernel_index_prefix_cover_each_head_once(group,kv_heads):
    heads=group*kv_heads
    observed,grid=head_mapping(TASKS/GROUP/'source/_fwd_grouped_kernel_stage1.py',heads,kv_heads)
    assert grid==(1,kv_heads*((group+15)//16),1)
    assert len(observed)==heads
    assert sorted(observed)==[(head,head//group) for head in range(heads)]


def grouped_reference_using_mapping(h,args,kwargs,mapping):
    q,k,v,out,pages,lengths,splits,scale,page_size=args
    result=out.clone()
    for head,kv_head in mapping:
        for batch,length in enumerate(lengths.tolist()):
            block=(length+splits-1)//splits
            for split in range(splits):
                begin,end=split*block,min((split+1)*block,length)
                if begin>=end:continue
                positions=torch.arange(begin,end)
                locations=pages[batch,positions//page_size].long()*page_size+positions%page_size
                scores=k[locations,kv_head].float()@q[batch,head].float()*scale
                if kwargs['logit_cap']>0:scores=kwargs['logit_cap']*torch.tanh(scores/kwargs['logit_cap'])
                result[batch,head,split,:-1]=torch.softmax(scores,dim=0)@v[locations,kv_head].float()
                result[batch,head,split,-1]=torch.logsumexp(scores,dim=0)
    return result


@pytest.mark.parametrize('case',('kv_group_17_two_heads','kv_group_24_two_heads'))
def test_grouped_new_controls_match_independent_attention_reference(case):
    task=TASKS/GROUP;checks=load(task/'_arena_checks.py');h=load(task/'scripts/task_runner.py')
    args,kwargs=checks.control_inputs(h,case,'cpu');config=checks.CONTRACT_CASES[case]
    assert args[0].shape[1]//args[1].shape[1] in (17,24)
    assert list(args[0].shape)==config['input_shapes']['q']
    mapping,_=head_mapping(task/'source/_fwd_grouped_kernel_stage1.py',args[0].shape[1],args[1].shape[1])
    output=grouped_reference_using_mapping(h,args,kwargs,mapping)
    plan=checks.CallPlan(h,args,kwargs);plan.check((output,));plan.unchanged()


class Pointer:
    def __init__(self,tensor,index=0):self.tensor,self.index=tensor,index
    def __add__(self,index):return Pointer(self.tensor,self.index+int(index))
    def __sub__(self,index):return self + (-int(index))


def rejection_wrapper(*,wrong_uniform_span=False):
    source=TASKS/REJECT/'source/triton_rejection_sample.py';tree=ast.parse(source.read_text())
    kernel=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_rejection_sample_kernel')
    kernel.decorator_list=[]
    if wrong_uniform_span:
        assignment=next(n for n in kernel.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='num_tokens' for t in n.targets))
        assignment.value=ast.Name(id='sampled_stride',ctx=ast.Load())
    state={'request':0}
    tl=SimpleNamespace(program_id=lambda _axis:state['request'],
        load=lambda pointer:int(pointer.tensor.reshape(-1)[pointer.index]),
        store=lambda pointer,value:pointer.tensor.reshape(-1).__setitem__(pointer.index,int(value)))
    scope={'tl':tl};exec(compile(ast.fix_missing_locations(ast.Module(body=[kernel],type_ignores=[])),'actual_scalar_Triton_body','exec'),scope)
    def device(grid,*args,**kwargs):
        assert kwargs=={'num_warps':1}
        values=[Pointer(v) if isinstance(v,torch.Tensor) else v for v in args]
        for request in range(grid[0]):state['request']=request;scope['_rejection_sample_kernel'](*values)
    ns={'torch':torch,'_rejection_sample_kernel':Launch(device)}
    functions(source,{'rejection_sample'},ns)
    return SimpleNamespace(rejection_sample=ns['rejection_sample'])


@pytest.mark.parametrize('case_index',(6,7))
@pytest.mark.parametrize('wrong_uniform_span',(False,True))
def test_ragged_real_wrapper_and_scalar_kernel_body_against_independent_prefix_oracle(case_index,wrong_uniform_span,monkeypatch):
    task=TASKS/REJECT;contract=load(task/'_arena_contract.py');replay=load(task/'_arena_replay.py')
    monkeypatch.setitem(sys.modules,'_arena_replay',replay)
    h=load(task/'scripts/task_runner.py');model=rejection_wrapper(wrong_uniform_span=wrong_uniform_span)
    function=replay.Recorder(h,contract).wrap(model.rejection_sample)
    if wrong_uniform_span:
        with pytest.raises((AssertionError,IndexError)):
            contract.run_ragged_control(h,function,'cpu',case_index)
    else:
        evidence=contract.run_ragged_control(h,function,'cpu',case_index)
        assert evidence['independent_scalar_oracle_checked']
        assert evidence['num_sampled']==contract.RAGGED_CONTROLS[case_index]['expected_counts']
        args=contract.ragged_inputs(case_index);expected=contract.ragged_scalar_answer(args)
        old=h.reference(*args);contract.check(h,old,expected,args)
        assert (args[2][1:]-args[2][:-1]).tolist()==contract.RAGGED_CONTROLS[case_index]['spans']


@pytest.mark.parametrize('task_name', (CP,GROUP,REJECT))
def test_all_original_scored_rows_timing_and_reference_source_are_exact(task_name):
    task=TASKS/task_name
    data=json.loads((task/'workloads.json').read_text())
    assert [r['test_case_id'] for r in data['cases'] if 'performance' in r['checks']]==['perf1','perf2','perf3','perf4','perf5']
    assert all(r['checks']==['correctness'] for r in data['cases'][5:])
    # Complete original runner bytes are pinned, including all math/input draws,
    # generated regions, 10 warmups and 100 reported samples.
    import hashlib
    expected={CP:'1c5b87ae973ffbf660cd520343c990c83e8bbb4f3b6d57e16f49e826408a12fd',
              GROUP:'d2dacb2807331c827edcbc97e8419ee17db41992470e6574be2262a9e4b425ad',
              REJECT:'2493d8af2b59535711b8af41e639338ed150a7e2590d01dfe16997be2dbd5860'}
    if task_name!=REJECT:
        from pr107_integration_helpers import original_runner
        raw=original_runner((task/'scripts/task_runner.py').read_text()).encode()
        assert hashlib.sha256(raw).hexdigest()==expected[task_name]
    if task_name==REJECT:
        assert hashlib.sha256((task/'scripts/task_runner.py').read_bytes()).hexdigest()==expected[task_name]
    runner=load(task/'scripts/task_runner.py')
    assert runner.WARMUP_ITERATIONS==10 and runner.BENCHMARK_ITERATIONS==100


@pytest.mark.parametrize('task_name',(CP,GROUP,REJECT))
def test_manifest_controls_are_bound_and_real_adapter_dispatches_failure(task_name,monkeypatch):
    task=TASKS/task_name;adapter=load(task/'_arena_eval.py');data=json.loads((task/'workloads.json').read_text())
    calls=[]
    if task_name in (CP,GROUP):
        checks=load(task/'_arena_checks.py')
        def bad(case):calls.append(case);return False,'finite CPU control sentinel'
        h=SimpleNamespace(TEST_SHAPES=data['input_table'],CONTRACT_CASES=checks.CONTRACT_CASES,
                          run_correctness=lambda **kw:(True,None),run_contract_correctness=bad)
    else:
        contract=load(task/'_arena_contract.py')
        def check(*,case_index):
            calls.append(case_index)
            return case_index<6,'finite CPU ragged sentinel'
        h=SimpleNamespace(TEST_SHAPES=data['input_table'],RAGGED_CONTROLS=contract.RAGGED_CONTROLS,run_correctness=check)
    monkeypatch.setattr(adapter,'load_harness',lambda:h)
    result=adapter.evaluate('baseline','correctness')
    assert result['status']=='FAIL'
    assert all(row['status']=='PASS' for row in result['cases'][:5])
    if task_name in (CP,GROUP):assert calls==list(h.CONTRACT_CASES)
    else:assert calls==list(range(8))
    tampered=copy.deepcopy(data)
    if task_name in (CP,GROUP):tampered['cases'][-1]['params']['seed']+=1
    else:tampered['cases'][-1]['params']['spans'][0]+=1
    monkeypatch.setattr(adapter,'load_manifest',lambda:tampered)
    result=adapter.evaluate('baseline','correctness')
    assert result['status']=='FAIL' and 'manifest' in result['reason'].lower()


@pytest.mark.parametrize('relative',__import__('reviewed_contract_history').TRITON_DEVICE_BOUNDARY_FILES)
def test_all_45_boundary_config_and_doc_inverses_require_exact_current_bytes(relative):
    import hashlib
    from reviewed_contract_history import TRITON_DEVICE_BOUNDARY_FILES,historical_triton_device_boundary_file
    source=(ROOT/relative).read_text()
    restored=historical_triton_device_boundary_file(relative,source)
    assert hashlib.sha256(restored.encode()).hexdigest()==TRITON_DEVICE_BOUNDARY_FILES[relative]['before_sha256']
    with pytest.raises(AssertionError):
        historical_triton_device_boundary_file(relative,source+'\n# unreviewed bytes\n')


@pytest.mark.parametrize('task_name',(CP,GROUP,REJECT))
def test_actual_harness_guard_retains_device_edits_and_protects_host_wrapper(task_name,tmp_path):
    import shutil
    from src.harness_guard import snapshot_workspace_harness,verify_workspace_harness
    from src.task_spec import load_task_spec
    workspace=tmp_path/task_name;shutil.copytree(TASKS/task_name,workspace)
    spec=load_task_spec(workspace/'config.yaml',task_id='triton2triton/vllm/'+task_name)
    snapshot=snapshot_workspace_harness(workspace,task_spec=spec)
    source=workspace/spec.candidate.editable[0].path;initial=source.read_text()
    target=spec.candidate.editable[0].symbols[0]
    tree=ast.parse(initial);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==target)
    lines=initial.splitlines(True);lines.insert(node.end_lineno,'    pass\n');good=''.join(lines)
    good+='\n\n@triton.jit\ndef _coverage_device_helper(value):\n    return value\n'
    source.write_text(good);verify_workspace_harness(snapshot,discard_added=False)
    host={CP:'correct_attn_cp_out',GROUP:'decode_grouped_att_m_fwd',REJECT:'rejection_sample'}[task_name]
    tree=ast.parse(good);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==host)
    lines=good.splitlines(True);lines.insert(node.end_lineno,'    pass\n');source.write_text(''.join(lines))
    with pytest.raises(RuntimeError):verify_workspace_harness(snapshot,discard_added=False)
    source.write_text(good);verify_workspace_harness(snapshot,discard_added=False)


@pytest.mark.parametrize('case_index',(6,7))
@pytest.mark.parametrize('wrong_uniform_span',(False,True))
def test_ragged_actual_installed_correctness_dispatch_executes_new_controls(case_index,wrong_uniform_span,monkeypatch):
    task=TASKS/REJECT;contract=load(task/'_arena_contract.py');replay=load(task/'_arena_replay.py')
    monkeypatch.setitem(sys.modules,'_arena_replay',replay)
    h=load(task/'scripts/task_runner.py');model=rejection_wrapper(wrong_uniform_span=wrong_uniform_span)
    h.load_module=lambda:model
    calls=[];original=contract.run_ragged_control
    def on_cpu(harness,function,device,index):
        assert device=='cuda'
        calls.append(index)
        return original(harness,function,'cpu',index)
    monkeypatch.setattr(contract,'run_ragged_control',on_cpu)
    replay.install(h,contract)
    ok,evidence=h.run_correctness(case_index=case_index)
    assert calls==[case_index]
    assert ok is (not wrong_uniform_span)
    if ok:assert evidence['independent_scalar_oracle_checked']
    else:assert 'IndexError' in evidence
