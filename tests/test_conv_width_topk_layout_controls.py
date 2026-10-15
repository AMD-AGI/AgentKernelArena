"""Actual wrapper/adapter/guard CPU checks; no Triton/GPU qualification."""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest
import torch
from src.harness_guard import snapshot_workspace_harness, verify_workspace_harness
from src.task_spec import load_task_spec

ROOT = Path(__file__).resolve().parents[1]
NAMES = ('triton_causal_conv1d_update', 'triton_topk_log_softmax')
OLD = {
 NAMES[0]: ('d2b5e46b76d5b1943dad2be4977a19618757a3a1c44fcc5cc83f5d4c6da72094',
            '1f74141ac07231fadc259c64d2eac6373953520f18ea4cb2df9bcb5608d1e75d',
            'b6c595ebddc7fbf8255326b526f68028330ef19db31ecd5b408f70b26be5c287',
            '1513bd96d909ca09a0382548d5090aad915a96007899317a96f5b62a30982487'),
 NAMES[1]: ('239f478dc699e10a65d25cd64d72f85d904112fc78114936f117bf3859a0d009',
            'ad05c55bc0c853fd4bdaaa202e1175e50bc6afccb1a167d6adf8ae7efdc0761b',
            '181d6ce2e536410426bd501c3c3d195bf61424e1a1693ed24e2ce1378e71f5d3',
            '4bfe4e94771e529f6128197e16ddb90b8645099a4288623e5057e365855fa152'),
}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(params=NAMES)
def task(request, monkeypatch):
    path = ROOT/'tasks/triton2triton/vllm'/request.param
    monkeypatch.syspath_prepend(str(path))
    monkeypatch.chdir(path)
    for name in ('_width_controls', '_layout_controls'):
        monkeypatch.delitem(sys.modules, name, raising=False)
    kind = 'width' if request.param == NAMES[0] else 'layout'
    controls = load(path/f'_{kind}_controls.py', f'_test_{kind}_controls')
    adapter = load(path/'_arena_eval.py', '_geometry_adapter')
    h = adapter.load_harness()
    monkeypatch.setattr(adapter, 'load_harness', lambda: h)
    return path, kind, controls, h, adapter


def cpu_cuda(monkeypatch):
    for name in ('arange','tensor','empty','zeros','full'):
        original = getattr(torch, name)
        def factory(*args, _original=original, **kwargs):
            if kwargs.get('device') == 'cuda':kwargs['device']='cpu'
            return _original(*args, **kwargs)
        monkeypatch.setattr(torch,name,factory)


def topk_module(source, *, fault=False, seen=None):
    tree=ast.parse(source)
    wrapper=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='compute_token_logprobs')
    class Kernel:
        def __getitem__(self,grid):
            def call(out,logits,row_stride,ids,topk,vocab,**kwargs):
                if seen is not None:seen.append((logits,ids,grid,kwargs))
                # Same address arithmetic as the real unchanged device kernel.
                x=torch.as_strided(logits,size=logits.shape,stride=(row_stride,1))
                t=torch.as_strided(ids,size=ids.shape,stride=(topk,1))
                out.copy_(torch.log_softmax(x.float(),-1).gather(1,t))
                if fault and vocab==1033:out.zero_()
            return call
    ns={'torch':torch,'triton':SimpleNamespace(next_power_of_2=lambda n:1<<(n-1).bit_length()),'_topk_log_softmax_kernel':Kernel()}
    exec(compile(ast.Module(body=[wrapper],type_ignores=[]),'<actual topk wrapper>', 'exec'),ns)
    return SimpleNamespace(compute_token_logprobs=ns['compute_token_logprobs'])


def conv_module(source, *, fault=None):
    wrapper=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='causal_conv1d_update')
    class Kernel:
        def __getitem__(self,grid):
            def call(x,weight,bias,state,indices,*args,**kwargs):
                saved=state.clone(); width=kwargs['KERNEL_WIDTH']
                for seq,slot in enumerate(indices.tolist()):
                    history=torch.cat((saved[slot],x[seq].clone()),-1)
                    y=torch.nn.functional.conv1d(history[None],weight[:,None],bias,groups=x.shape[1])[0]
                    if kwargs['SILU_ACTIVATION']:y=torch.nn.functional.silu(y)
                    x[seq].copy_(y);state[slot].copy_(history[:,-state.shape[-1]:])
                if fault=='output':x.zero_()
                if fault=='state':state.copy_(saved)
                if fault=='inactive_slot':state[1].zero_()
                if fault=='weight':weight.zero_()
                if fault==f'width{width}':x.zero_()
            return call
    ns={'torch':torch,'PAD_SLOT_ID':-1,'triton':SimpleNamespace(next_power_of_2=lambda n:1<<(n-1).bit_length(),cdiv=lambda n,d:(n+d-1)//d),'_causal_conv1d_update_kernel':Kernel()}
    exec(compile(ast.Module(body=[wrapper],type_ignores=[]),'<actual conv wrapper>', 'exec'),ns)
    return SimpleNamespace(causal_conv1d_update=ns['causal_conv1d_update'])


def test_all_old_bytes_and_cases_are_preserved_after_exact_additions(task):
    path,kind,controls,h,adapter=task
    digest=lambda raw:hashlib.sha256(raw).hexdigest()
    runner=(path/'scripts/task_runner.py').read_text()
    dispatch=('    if case_index is not None and case_index >= 20000:\n'
              f'        from _{kind}_controls import run_control\n'
              '        return run_control(case_index - 20000, load_module)\n')
    assert runner.count(dispatch)==1
    assert digest(runner.replace(dispatch,'',1).encode())==OLD[path.name][0]
    data=deepcopy(adapter.load_manifest());assert len(data.pop(kind+'_controls'))==3
    assert data['cases'][-3:]==[{'test_case_id':'control-'+kind+'-'+c['name'],'dtype':'float32,int32' if kind=='width' else 'float32,int64','params':{'case_index':20000+i,'configuration':c},'checks':['correctness'],'status':'PASS'} for i,c in enumerate(controls.CASES)]
    data['cases']=data['cases'][:-3]
    assert digest((json.dumps(data,indent=2)+'\n').encode())==OLD[path.name][1]
    source=(path/'source'/(path.name+'.py')).read_text()
    if kind=='layout':
        addition='    logits = logits.contiguous()\n    token_ids = token_ids.to(torch.int64).contiguous()'
        assert source.count(addition)==1
        source=source.replace(addition,'    token_ids = token_ids.to(torch.int64)',1)
    assert digest(source.encode())==OLD[path.name][2]
    assert digest((path/'_arena_checks.py').read_bytes())==OLD[path.name][3]
    assert (h.WARMUP_ITERATIONS,h.BENCHMARK_ITERATIONS)==(10,100)
    assert sum('performance' in row['checks'] for row in data['cases'])==5


def test_independent_reference_validation_and_explicit_geometry(task):
    _,kind,c,h,_=task;c.validate_reference(h)
    if kind=='width':
        assert [row['shape'][-1] for row in c.CASES]==[2,5,6]
        for i in range(3):
            x,state,w,bias,activation,ids=c.make_case(i,'cpu')
            assert state.stride(1)==1 and len(set(ids.tolist()))==x.shape[0]
            assert 1 not in ids.tolist()
    else:
        assert [(row['strided_logits'],row['strided_ids']) for row in c.CASES]==[(True,False),(False,True),(True,True)]
        for i in range(3):
            x,ids=c.make_case(i,'cpu')
            assert (x.stride(-1)==2) is c.CASES[i]['strided_logits']
            assert (ids.stride(-1)==2) is c.CASES[i]['strided_ids']
            assert ids.dtype==torch.int64


@pytest.mark.parametrize('index',range(3))
def test_real_wrappers_and_new_control_gate_accept_correct_device_model(task,monkeypatch,index):
    path,kind,c,h,_=task;cpu_cuda(monkeypatch)
    text=(path/'source'/(path.name+'.py')).read_text()
    mod=conv_module(text) if kind=='width' else topk_module(text)
    assert c.run_control(index,lambda:mod)==(True,None)
    # Run the actual new public runner dispatch as well.
    monkeypatch.setattr(h,'load_module',lambda:mod)
    assert h.run_correctness(case_index=20000+index)==(True,None)


@pytest.mark.parametrize('index',range(3))
def test_actual_topk_layout_repair_fails_before_and_passes_after(task,index):
    path,kind,c,_,_=task
    if kind!='layout':pytest.skip('topk layout contract')
    source=(path/'source'/(path.name+'.py')).read_text()
    before=source.replace('    logits = logits.contiguous()\n    token_ids = token_ids.to(torch.int64).contiguous()', '    token_ids = token_ids.to(torch.int64)')
    logits,ids=c.make_case(index,'cpu');saved=(logits.clone(),ids.clone())
    expected=c.reference(logits,ids)
    old=topk_module(before).compute_token_logprobs(logits,ids)
    current=topk_module(source).compute_token_logprobs(logits,ids)
    assert not torch.allclose(old,expected,atol=.01,rtol=.01)
    torch.testing.assert_close(current,expected,atol=.01,rtol=.01)
    assert torch.equal(logits,saved[0]) and torch.equal(ids,saved[1])


def test_contiguous_topk_scored_inputs_preserve_launch_pointers(task):
    path,kind,_,_,_=task
    if kind!='layout':pytest.skip('topk layout contract')
    x=torch.randn(2,256);ids=torch.tensor([[0,17,255],[255,0,17]],dtype=torch.int64);seen=[]
    source=(path/'source'/(path.name+'.py')).read_text()
    topk_module(source,seen=seen).compute_token_logprobs(x,ids)
    actual_x,actual_ids,grid,options=seen[0]
    assert actual_x.data_ptr()==x.data_ptr() and actual_ids.data_ptr()==ids.data_ptr()
    assert grid==(2,) and options=={'BLOCK_SIZE':1024,'PADDED_TOPK':4}


@pytest.mark.parametrize('index',range(3))
def test_new_controls_reject_wrong_device_computation(task,monkeypatch,index):
    path,kind,c,_,_=task;cpu_cuda(monkeypatch)
    text=(path/'source'/(path.name+'.py')).read_text()
    mod=conv_module(text,fault='output') if kind=='width' else topk_module(text,fault=True)
    ok,reason=c.run_control(index,lambda:mod)
    assert not ok and 'AssertionError' in reason


@pytest.mark.parametrize('width',[2,5,6])
def test_conv_width_conditioned_device_fault_rejects_exact_control(task,monkeypatch,width):
    path,kind,c,_,_=task
    if kind!='width':pytest.skip('convolution width contract')
    cpu_cuda(monkeypatch)
    mod=conv_module((path/'source'/(path.name+'.py')).read_text(),fault=f'width{width}')
    results=[c.run_control(index,lambda:mod)[0] for index in range(3)]
    assert results==[case['shape'][-1]!=width for case in c.CASES]


@pytest.mark.parametrize('fault',['state','inactive_slot','weight'])
def test_conv_controls_reject_state_and_readonly_defects(task,monkeypatch,fault):
    path,kind,c,_,_=task
    if kind!='width':pytest.skip('convolution state contract')
    cpu_cuda(monkeypatch);mod=conv_module((path/'source'/(path.name+'.py')).read_text(),fault=fault)
    assert not c.run_control(1,lambda:mod)[0]


@pytest.mark.parametrize('role',['baseline','candidate'])
def test_real_adapter_dispatches_all_new_controls_for_both_roles(task,monkeypatch,role):
    path,kind,c,h,a=task;cpu_cuda(monkeypatch)
    text=(path/'source'/(path.name+'.py')).read_text()
    mod=conv_module(text) if kind=='width' else topk_module(text)
    monkeypatch.setattr(h,'load_module',lambda:mod)
    original=h.run_correctness;seen=[]
    def run(*,case_index):
        seen.append(case_index)
        return original(case_index=case_index) if case_index>=20000 else (True,None)
    monkeypatch.setattr(h,'run_correctness',run)
    result=a.evaluate(role,'correctness')
    assert result['status']=='PASS',result
    assert seen==[row['params']['case_index'] for row in result['cases']]
    assert seen[-3:]==[20000,20001,20002]


@pytest.mark.parametrize('drift',['table','row','missing'])
def test_adapter_rejects_control_descriptor_drift(task,monkeypatch,drift):
    _,kind,_,h,a=task;data=deepcopy(a.load_manifest())
    if drift=='table':data[kind+'_controls'][0]['shape'][0]=99
    if drift=='row':data['cases'][-1]['checks']=['correctness','performance']
    if drift=='missing':data['cases'].pop()
    monkeypatch.setattr(a,'load_manifest',lambda:data)
    monkeypatch.setattr(h,'load_module',lambda:pytest.fail('Drift reached candidate'))
    result=a.evaluate('candidate','correctness');assert result['status']=='FAIL'
    assert f'protected {kind}' in result['reason'].lower()


@pytest.mark.parametrize('change',['kernel','helper','wrapper','imports','module','controls'])
def test_actual_topk_guard_boundary(tmp_path,change):
    path=ROOT/'tasks/triton2triton/vllm/triton_topk_log_softmax';work=tmp_path/'task'
    shutil.copytree(path,work,ignore=shutil.ignore_patterns('__pycache__'))
    spec=load_task_spec(work/'config.yaml',task_id='triton2triton/vllm/triton_topk_log_softmax')
    edit,=spec.candidate.editable
    assert edit.scope=='symbols' and edit.symbols==('_topk_log_softmax_kernel',) and edit.allow_new_helpers
    guard=snapshot_workspace_harness(work,task_spec=spec);source=work/edit.path
    tree=ast.parse(source.read_text())
    if change in ('kernel','wrapper'):
        name='_topk_log_softmax_kernel' if change=='kernel' else 'compute_token_logprobs'
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name);node.body.append(ast.Pass())
    elif change=='helper':tree.body.extend(ast.parse('@triton.jit\ndef reduction_helper(x):\n    return x\n').body)
    elif change=='imports':tree.body.extend(ast.parse('import os\n').body)
    elif change=='module':tree.body.extend(ast.parse('print("module execution")\n').body)
    elif change=='controls':(work/'_layout_controls.py').write_text('CASES=[]\n')
    source.write_text(ast.unparse(ast.fix_missing_locations(tree)))
    if change in ('kernel','helper'):verify_workspace_harness(guard,discard_added=False)
    else:
        with pytest.raises(RuntimeError):verify_workspace_harness(guard,discard_added=False)


def test_actual_conv_width_helper_is_protected(tmp_path):
    path=ROOT/'tasks/triton2triton/vllm/triton_causal_conv1d_update';work=tmp_path/'task'
    shutil.copytree(path,work,ignore=shutil.ignore_patterns('__pycache__'))
    spec=load_task_spec(work/'config.yaml',task_id='triton2triton/vllm/triton_causal_conv1d_update')
    guard=snapshot_workspace_harness(work,task_spec=spec)
    (work/'_width_controls.py').write_text('CASES=[]\n')
    with pytest.raises(RuntimeError):verify_workspace_harness(guard,discard_added=False)


@pytest.mark.parametrize('index',range(3))
def test_conv_oracle_cpu_contraction_and_expected_device_transfer(monkeypatch,index):
    """Device proxies prove placement without claiming CUDA execution."""
    path=ROOT/'tasks/triton2triton/vllm/triton_causal_conv1d_update/_width_controls.py'
    controls=load(path,'_cpu_oracle_placement')
    x,state,weight,bias,activation,ids=controls.make_case(index,'cpu')
    history=torch.cat((state.index_select(0,ids.long()),x),-1)
    expected=torch.nn.functional.conv1d(history.double(),weight[:,None].double(),
        None if bias is None else bias.double(),groups=x.shape[1])
    if activation=='silu':expected=torch.nn.functional.silu(expected)
    expected=expected.float()
    expected_state=state.clone()
    expected_state.index_copy_(0,ids.long(),history[...,-state.shape[-1]:])
    copies=[];transfers=[];contractions=[]

    class DeviceProxy:
        def __init__(self,value):self.value=value
        @property
        def device(self):return torch.device('cuda')
        @property
        def dtype(self):return self.value.dtype
        def detach(self):return self
        def cpu(self):
            copies.append(self.value)
            return self.value.clone()

    original_einsum,original_to=torch.einsum,torch.Tensor.to
    def cpu_einsum(equation,*operands):
        assert equation=='bdtw,dw->bdt'
        assert all(value.device.type=='cpu' and value.dtype==torch.float64 for value in operands)
        contractions.append(tuple(value.shape for value in operands))
        return original_einsum(equation,*operands)
    def fake_transfer(value,*args,**kwargs):
        if kwargs.get('device')==torch.device('cuda'):
            transfers.append((tuple(value.shape),kwargs['dtype']))
            kwargs['device']=torch.device('cpu')
        return original_to(value,*args,**kwargs)
    monkeypatch.setattr(torch,'einsum',cpu_einsum)
    monkeypatch.setattr(torch.Tensor,'to',fake_transfer)
    args=[DeviceProxy(value) if value is not None else None for value in (x,state,weight,bias)]
    output,cache=controls.reference(*args,activation,DeviceProxy(ids))
    assert len(copies)==(4 if bias is None else 5) and len(contractions)==1
    assert transfers==[(tuple(x.shape),x.dtype),(tuple(state.shape),state.dtype)]
    torch.testing.assert_close(output,expected,atol=0,rtol=0)
    assert torch.equal(cache,expected_state)
