"""CPU contract regressions; synthetic measurements below are not GPU validation."""
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch

from src.task_protocol import CaseManifest, parse_command_result
from src.task_spec import load_task_spec

ROOT = Path(__file__).resolve().parents[1]
VLLM = sorted((ROOT/'tasks/triton2triton/vllm').glob('*/config.yaml'))
BASE = '5c9f8ef2'


def module_at(path, monkeypatch):
    monkeypatch.chdir(path.parent)
    spec = importlib.util.spec_from_file_location('_isolated_task_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def result_record(result):
    return parse_command_result('ARENA_EVAL_RESULT='+json.dumps(result), role=result['role'],
                                action=result['action'], returncode=0 if result['status']=='PASS' else 1)


@pytest.mark.parametrize('path', VLLM, ids=lambda p:p.parent.name)
def test_vllm_v2_preserves_all_original_cases_checks_sources_and_helpers(path):
    task = path.parent
    spec = load_task_spec(path, task_id=task.relative_to(ROOT/'tasks').as_posix())
    assert spec.candidate.initial_state == 'implemented'
    assert spec.candidate.language == 'triton' and spec.baseline.kind == 'initial_candidate'
    relative = (task/'scripts/task_runner.py').relative_to(ROOT).as_posix()
    before = subprocess.check_output(['git','show',f'{BASE}:{relative}'], cwd=ROOT, text=True)
    after = (ROOT/relative).read_text()
    bt, at = ast.parse(before), ast.parse(after)
    bf = {n.name:n for n in bt.body if isinstance(n,ast.FunctionDef)}
    af = {n.name:n for n in at.body if isinstance(n,ast.FunctionDef)}
    correction = af['run_correctness']
    correction.args = deepcopy(bf['run_correctness'].args)
    loop = next(n for n in ast.walk(correction) if isinstance(n,ast.For))
    assert ast.unparse(loop.body[0].test).startswith('case_index is not None')
    loop.body.pop(0)
    assert ast.dump(correction, include_attributes=False) == ast.dump(bf['run_correctness'], include_attributes=False)
    for name in bf.keys()-{'run_correctness'}:
        if task.name == 'triton_pack_bitmatrix' and name == 'reference_pack_bitmatrix':
            # Explicit semantic repair: old oracle counted tile-padding slots
            # as expert31. Known answers and old-source controls below cover it.
            continue
        expected = ast.get_source_segment(before, bf[name])
        if task.name == 'triton_prepare_mrope_positions' and name == 'run_performance':
            # The old decode-labelled cases always executed prefill. Preserve
            # everything except the two now scenario-dependent input fields.
            expected = expected.replace(
                '            prefill_lens = torch.full((max_num_reqs,), max_model_len,',
                '            # Honor the declared prefill/decode case, as correctness does.\n'
                '            prefill_lens = torch.full((max_num_reqs,), max_model_len if is_prefill else 10,',
            ).replace(
                'num_computed_tokens = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)',
                'num_computed_tokens = torch.full((max_num_reqs,), 0 if is_prefill else 50, dtype=torch.int32, device=device)',
            )
        if task.name == 'triton_topk_topp' and name == 'compare_masked_logits':
            # Require the added metadata/NaN/+inf rejection while preserving
            # the original finite-value tolerances and mask mismatch allowance.
            expected = expected.replace(
                '    import torch\n',
                '    import torch\n\n'
                '    if (got.shape, got.dtype, got.device) != (ref.shape, ref.dtype, ref.device):\n'
                "        return False, 'output shape/dtype/device mismatch'\n"
                '    if torch.isnan(got).any() or torch.isposinf(got).any():\n'
                "        return False, 'only negative infinity is a valid masked logit'\n",
                1,
            )
        assert expected == ast.get_source_segment(after, af[name])
    manifest = json.loads((task/'workloads.json').read_text())
    for source, targets in manifest['candidate_symbols'].items():
        nodes = {n.name: n for n in ast.parse((task/source).read_text()).body
                 if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for target in targets:
            decorators = [d.func if isinstance(d, ast.Call) else d
                          for d in nodes[target['name']].decorator_list]
            is_triton_jit = any(isinstance(d, ast.Attribute) and d.attr == 'jit'
                               and isinstance(d.value, ast.Name) and d.value.id == 'triton'
                               for d in decorators)
            assert target['jit'] is is_triton_jit, (source, target['name'])
    scored_ids = [row['test_case_id'] for row in manifest['cases'] if 'performance' in row['checks']]
    additional_scored_ids = {
        'triton_bad_words': ['perf_prefix_routing'],
        'triton_logit_bias': ['perf_combined_filtering'],
        'triton_penalties': ['perf_speculative_penalties'],
        'triton_topk_topp': ['perf_top_p_only', 'perf_combined_topk_topp'],
    }
    assert scored_ids == [f'perf{i}' for i in range(1, 6)] + additional_scored_ids.get(task.name, [])
    assert all('correctness' in row['checks'] for row in manifest['cases'])
    assert manifest['migration']['original_harness_sha256'] == hashlib.sha256(before.encode()).hexdigest()
    for edit in spec.candidate.editable:
        source = task/edit.path
        original = subprocess.check_output(['git','show',f'{BASE}:{source.relative_to(ROOT).as_posix()}'],cwd=ROOT)
        if task.name == 'triton_pack_bitmatrix':
            old = b'div[:, :, None] == offs[None, None, :], (one << rem)[:, :, None], 0'
            new = (b'mask[:, :, None] & (indices[:, :, None] >= 0) & (div[:, :, None] == offs[None, None, :]),\n'
                   b'            (one << rem)[:, :, None], 0')
            assert original.count(old) == 1
            original = original.replace(old, new)
        assert source.read_bytes() == original
    assert 'Evaluation contract' in (task/'README.md').read_text()


def test_vllm_per_case_failure_and_incomplete_measurement_rejected(monkeypatch):
    adapter = module_at(VLLM[0].parent/'_arena_eval.py', monkeypatch)
    manifest = adapter.load_manifest()
    harness = SimpleNamespace(**{manifest['case_table']:manifest['input_table']})
    harness.run_correctness = lambda case_index: (case_index != 3, 'negative control')
    harness.run_performance = lambda: []
    monkeypatch.setattr(adapter,'load_harness',lambda:harness)
    correctness = adapter.evaluate('candidate','correctness')
    assert correctness['status']=='FAIL'
    assert [r['status'] for r in correctness['cases']]==['PASS','PASS','PASS','FAIL','PASS']
    manifest_result = {'protocol':'arena-eval-v1','role':'task','action':'validate-task',
                       'status':'PASS','cases':manifest['cases']}
    CaseManifest.from_result(result_record(manifest_result)).validate(result_record(correctness))
    perf = adapter.evaluate('candidate','performance')
    assert perf['status']=='FAIL' and all(r['status']=='FAIL' for r in perf['cases'])
    assert perf['failure_kind'] != 'numerical_mismatch'


def test_vllm_candidate_stub_is_never_baseline_fallback(tmp_path,monkeypatch):
    adapter = module_at(VLLM[0].parent/'_arena_eval.py',monkeypatch)
    data = adapter.load_manifest()
    monkeypatch.setattr(adapter,'ROOT',tmp_path)
    for source, targets in data['candidate_symbols'].items():
        p=tmp_path/source;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text('\n'.join('@triton.jit\ndef '+t['name']+'():\n    pass\n' for t in targets))
    assert adapter.inspect_candidate(data)=='unimplemented'
    with pytest.raises(ValueError,match='no baseline fallback'):
        adapter.inspect_candidate(data,require_implemented=True)


def _vllm_called_jit_tasks():
    tasks = []
    for config in VLLM:
        task = config.parent
        manifest = json.loads((task/'workloads.json').read_text())
        for source, targets in manifest['candidate_symbols'].items():
            symbols = {target['name'] for target in targets}
            if any(isinstance(node, ast.FunctionDef) and node.name in symbols
                   and any(isinstance(d, ast.Call) and ast.unparse(d.func) == 'triton.jit'
                           for d in node.decorator_list)
                   for node in ast.parse((task/source).read_text()).body):
                tasks.append(config)
                break
    return tasks


@pytest.mark.parametrize('config', _vllm_called_jit_tasks(), ids=lambda p: p.parent.name)
def test_vllm_called_jit_kernel_cannot_be_replaced_with_plain_python(config, tmp_path, monkeypatch):
    adapter = module_at(config.parent/'_arena_eval.py', monkeypatch)
    data = adapter.load_manifest()
    assert adapter.inspect_candidate(data, require_implemented=True) == 'implemented'
    for source, targets in data['candidate_symbols'].items():
        path = tmp_path/source
        path.parent.mkdir(parents=True, exist_ok=True)
        tree = ast.parse((config.parent/source).read_text())
        symbols = {target['name'] for target in targets}
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name in symbols:
                node.decorator_list = []
                # A nonempty function used to bypass the false manifest flag.
                node.body = ast.parse('return 1').body
        path.write_text(ast.unparse(ast.fix_missing_locations(tree)))
    monkeypatch.setattr(adapter, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='must remain a Triton JIT kernel'):
        adapter.inspect_candidate(data, require_implemented=True)
    result = adapter.evaluate('candidate', 'compile')
    assert result_record(result).status == 'FAIL'
    assert 'must remain a Triton JIT kernel' in result['reason']



def test_staged_write_reference_known_answer_and_untouched_cells(monkeypatch):
    task = ROOT/'tasks/triton2triton/vllm/triton_apply_write'
    harness = module_at(task/'scripts/task_runner.py', monkeypatch)
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    initial = torch.full((3, 5), -7, dtype=torch.int32)
    indices = torch.tensor([2, 0], dtype=torch.int32)
    starts = torch.tensor([1, 3], dtype=torch.int32)
    contents = torch.tensor([10, 11, 20], dtype=torch.int32)
    cumulative = torch.tensor([2, 3], dtype=torch.int32)
    expected = torch.tensor([[-7,-7,-7,20,-7],[-7,-7,-7,-7,-7],[-7,10,11,-7,-7]], dtype=torch.int32)
    torch.testing.assert_close(harness.reference_apply_write(initial,indices,starts,contents,cumulative),expected)
    checks.check_output(expected,initial,indices,starts,contents,cumulative,harness.reference_apply_write)
    invalid = expected.clone(); invalid[1,0] = 0
    with pytest.raises(AssertionError,match='untouched cells'):
        checks.check_output(invalid,initial,indices,starts,contents,cumulative,harness.reference_apply_write)
    with pytest.raises(AssertionError,match='dtype'):
        checks.check_output(expected.float(),initial,indices,starts,contents,cumulative,harness.reference_apply_write)


@pytest.mark.parametrize('mode', ['correct','incorrect_timed','stale','no_write','uniform_segments','zero_starts','identity_mapping'])
def test_staged_write_replay_checks_changed_mapping_offsets_and_lengths(monkeypatch, mode):
    task = ROOT/'tasks/triton2triton/vllm/triton_apply_write'
    harness = module_at(task/'scripts/task_runner.py', monkeypatch)
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    harness._TimedRun = SimpleNamespace
    output = torch.zeros(6, 12, dtype=torch.int32)
    output_work = output.clone()
    write_indices = torch.arange(4, dtype=torch.int32)
    write_starts = torch.zeros(4, dtype=torch.int32)
    write_contents = torch.arange(1, 13, dtype=torch.int32)
    cu_lens = torch.tensor([3,6,9,12], dtype=torch.int32)
    original = [v.clone() for v in (write_indices,write_starts,write_contents,cu_lens)]
    def fn():
        output_work.copy_(harness.reference_apply_write(output_work,write_indices,write_starts,write_contents,cu_lens))
    def prepare():
        output_work.copy_(output)
    observed = []
    def benchmark(measured, *, timed_run, **kwargs):
        observed.append({k:v for k,v in kwargs.items() if k != 'prepare_fn'})
        for actual,expected in zip((write_indices,write_starts,write_contents,cu_lens),original):
            torch.testing.assert_close(actual,expected)
        kwargs['prepare_fn'](); captured = measured(); cached = captured.clone()
        if mode == 'incorrect_timed': captured.zero_()
        def replay():
            kwargs['prepare_fn']()
            indices = original[0] if mode == 'identity_mapping' else write_indices
            starts = original[1] if mode == 'zero_starts' else write_starts
            cumulative = original[3] if mode == 'uniform_segments' else cu_lens
            if mode == 'stale': captured.copy_(cached)
            elif mode != 'no_write':
                captured.copy_(harness.reference_apply_write(output,indices,starts,write_contents,cumulative))
            return captured
        timed_run.outputs = captured; timed_run.rerun = replay
        return 0.25, {'benchmark_method':'cuda_graph'}
    options = dict(warmup=10,repetition=100,target_ms=20.0,prepare_fn=prepare)
    if mode == 'correct':
        ms,metadata = checks.checked_benchmark(harness,benchmark,fn,**options)
        assert ms == 0.25 and metadata['perturbed_input_replay_checked']
    else:
        with pytest.raises(AssertionError): checks.checked_benchmark(harness,benchmark,fn,**options)
    assert observed == [dict(warmup=10,repetition=100,target_ms=20.0)]


def test_staged_write_adapter_installs_checks(monkeypatch):
    adapter = module_at(ROOT/'tasks/triton2triton/vllm/triton_apply_write/_arena_eval.py',monkeypatch)
    assert adapter.load_harness().run_performance.__module__ == '_staged_write_checks'


@pytest.mark.parametrize('mode', ['correct','incorrect_timed','stale','no_write','omit_last','truncate_tail','wrong_source',
                                 'zero_source_and_dest_timed','zero_source_and_dest_replay',
                                 'mutate_source_only_timed','mutate_source_only_replay'])
def test_batch_memcpy_checks_every_timed_destination_and_replay(monkeypatch, mode):
    task = ROOT/'tasks/triton2triton/vllm/triton_batch_memcpy'
    checks = module_at(task/'_arena_checks.py',monkeypatch)
    harness = SimpleNamespace(_TimedRun=SimpleNamespace)
    sources = [torch.tensor([1,2,3],dtype=torch.uint8), torch.tensor([11,12,13,14,15],dtype=torch.uint8)]
    initial_sources = [source.clone() for source in sources]
    destinations = [torch.zeros_like(value) for value in sources]
    def fn():
        for dst,src in zip(destinations,sources): dst.copy_(src)
    observed = []
    def benchmark(measured, *, timed_run, **kwargs):
        observed.append(kwargs)
        for source, initial in zip(sources,initial_sources):
            assert torch.equal(source,initial)
        outputs = measured(); cached = [value.clone() for value in outputs]
        if mode == 'incorrect_timed': outputs[-1].zero_()
        if mode == 'zero_source_and_dest_timed':
            sources[-1].zero_(); outputs[-1].zero_()
        if mode == 'mutate_source_only_timed': sources[-1].zero_()
        def replay():
            for i,(out,src) in enumerate(zip(outputs,sources)):
                if mode == 'no_write' or mode == 'omit_last' and i == len(outputs)-1: continue
                if mode == 'stale': out.copy_(cached[i])
                elif mode == 'truncate_tail': out[:-1].copy_(src[:-1])
                elif mode == 'wrong_source': out.fill_(sources[1-i][0])
                else: out.copy_(src)
            if mode == 'zero_source_and_dest_replay':
                sources[-1].zero_(); outputs[-1].zero_()
            if mode == 'mutate_source_only_replay': sources[-1].zero_()
            return outputs
        timed_run.outputs = outputs; timed_run.rerun = replay
        return 0.25, {'benchmark_method':'cuda_graph'}
    if mode == 'correct':
        ms,metadata=checks.checked_benchmark(harness,benchmark,fn,sources,destinations,warmup=10,repetition=100)
        assert ms==0.25 and metadata['perturbed_input_replay_checked'] and metadata['source_buffers_unchanged']
    else:
        with pytest.raises(AssertionError):
            checks.checked_benchmark(harness,benchmark,fn,sources,destinations,warmup=10,repetition=100)
    assert observed == [dict(warmup=10,repetition=100)]


def test_batch_memcpy_retains_original_inputs_and_restores_hooks(monkeypatch):
    task = ROOT/'tasks/triton2triton/vllm/triton_batch_memcpy'
    checks = module_at(task/'_arena_checks.py',monkeypatch)
    sources=[torch.tensor([1,255,0],dtype=torch.uint8)]
    destinations=[torch.zeros_like(sources[0])]
    sentinel=(sources,destinations,object(),object(),object())
    calls=[]
    def make_inputs(*args,**kwargs): calls.append((args,kwargs));return sentinel
    benchmark=object()
    h=SimpleNamespace(make_inputs=make_inputs,_benchmark_cuda_graph_or_events=benchmark,
                      run_correctness=lambda **kwargs:(True,None))
    def run():
        assert h.make_inputs(4,8,device='cuda') is sentinel
        return h._benchmark_cuda_graph_or_events('protected-fn',warmup=10,repetition=100)
    def checked(actual_h,actual_b,fn,src,dst,**kwargs):
        assert actual_h is h and actual_b is benchmark and fn=='protected-fn'
        assert src is sources and dst is destinations
        assert kwargs==dict(warmup=10,repetition=100)
        raise AssertionError('simulated output mismatch')
    h.run_performance=run;monkeypatch.setattr(checks,'checked_benchmark',checked)
    checks.install(h)
    with pytest.raises(AssertionError,match='simulated output mismatch'): h.run_performance()
    assert h.make_inputs is make_inputs and h._benchmark_cuda_graph_or_events is benchmark
    assert calls==[((4,8),dict(device='cuda'))]


def test_batch_memcpy_adapter_installs_checks(monkeypatch):
    adapter=module_at(ROOT/'tasks/triton2triton/vllm/triton_batch_memcpy/_arena_eval.py',monkeypatch)
    harness=adapter.load_harness()
    assert harness.run_performance.__module__=='_batch_memcpy_checks'
    assert harness.run_correctness.__module__=='_batch_memcpy_checks'


@pytest.mark.parametrize('case_index', [None, 0, 4])
@pytest.mark.parametrize('mode', ['correct','zero_source_and_dest','mutate_source_only','incorrect_output'])
def test_batch_memcpy_original_correctness_uses_pristine_sources(monkeypatch, case_index, mode):
    task=ROOT/'tasks/triton2triton/vllm/triton_batch_memcpy'
    harness=module_at(task/'scripts/task_runner.py',monkeypatch)
    checks=module_at(task/'_arena_checks.py',monkeypatch)
    original_inputs=harness.make_inputs
    current=[]
    seen=[]
    def make_inputs(batch,size,device):
        assert device=='cuda'
        inputs=original_inputs(batch,size,device='cpu')
        current[:]=inputs[:2]
        seen.append((batch,size))
        return inputs
    def candidate(src_ptrs,dst_ptrs,sizes):
        sources,destinations=current
        for source,destination in zip(sources,destinations): destination.copy_(source)
        if mode=='zero_source_and_dest':
            sources[-1].zero_(); destinations[-1].zero_()
        elif mode=='mutate_source_only': sources[-1].zero_()
        elif mode=='incorrect_output': destinations[-1].zero_()
    monkeypatch.setattr(harness,'make_inputs',make_inputs)
    monkeypatch.setattr(harness,'load_module',lambda:SimpleNamespace(batch_memcpy=candidate))
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    checks.install(harness)
    ok,error=harness.run_correctness(case_index=case_index)
    assert ok is (mode=='correct')
    if mode=='zero_source_and_dest': assert 'caller-owned source' in error
    if mode=='correct' or mode=='zero_source_and_dest':
        assert seen==[value for i,value in enumerate(harness.TEST_SHAPES)
                      if case_index is None or case_index==i]
    assert harness.make_inputs is make_inputs

def bmm_known_inputs():
    a=torch.tensor([[[1.,2.],[3.,4.]],[[-1.,2.],[0.,3.]]],dtype=torch.float16)
    b=torch.tensor([[[5.,6.],[7.,8.]],[[4.,2.],[6.,1.]]],dtype=torch.float16)
    expected=torch.tensor([[[19.,22.],[43.,50.]],[[8.,0.],[18.,3.]]],dtype=torch.float16)
    return a,b,expected


@pytest.mark.parametrize('mode',['correct','zero_source_and_output','wrong_batch','shape','dtype','device','nonfinite'])
def test_bmm_correctness_preserves_inputs_and_full_output_contract(monkeypatch,mode):
    checks=module_at(ROOT/'tasks/triton2triton/vllm/triton_bmm/_arena_checks.py',monkeypatch)
    a,b,expected=bmm_known_inputs()
    torch.testing.assert_close(checks.reference(a,b),expected,atol=0,rtol=0)
    def candidate(a,b):
        output=expected.clone()
        if mode=='zero_source_and_output':a.zero_();output.zero_()
        elif mode=='wrong_batch':output[1].copy_(output[0])
        elif mode=='shape':output=output[:1]
        elif mode=='dtype':output=output.float()
        elif mode=='device':output=torch.empty_like(output,device='meta')
        elif mode=='nonfinite':output.fill_(float('inf'))
        return output
    module=SimpleNamespace(bmm_triton=candidate)
    load=lambda:module
    harness=SimpleNamespace(load_module=load)
    with checks.checked_modules(harness):
        if mode=='correct':torch.testing.assert_close(harness.load_module().bmm_triton(a,b),expected)
        else:
            with pytest.raises(AssertionError):harness.load_module().bmm_triton(a,b)
    assert harness.load_module is load and module.bmm_triton is candidate


@pytest.mark.parametrize('mode',['correct','incorrect_timed','stale','no_write','changing_wrong',
                                'zero_source_and_output_timed','zero_source_and_output_replay'])
def test_bmm_exact_timed_replay_and_pristine_input_oracles(monkeypatch,mode):
    checks=module_at(ROOT/'tasks/triton2triton/vllm/triton_bmm/_arena_checks.py',monkeypatch)
    a,b,expected=bmm_known_inputs()
    mod=SimpleNamespace(bmm_triton=checks.reference)
    original=mod.bmm_triton
    initial=(a.clone(),b.clone())
    harness=SimpleNamespace(_TimedRun=SimpleNamespace)
    observed=[]
    def fn():mod.bmm_triton(a,b)
    def benchmark(measured,*,timed_run,**kwargs):
        observed.append(kwargs)
        for value,saved in zip((a,b),initial):assert torch.equal(value,saved)
        output=measured();cached=output.clone()
        torch.testing.assert_close(output,expected)
        if mode=='incorrect_timed':output.zero_()
        if mode=='zero_source_and_output_timed':a.zero_();output.zero_()
        def replay():
            if mode=='correct':output.copy_(checks.reference(a,b))
            elif mode=='stale':output.copy_(cached)
            elif mode=='changing_wrong':output.fill_(a.flatten()[0])
            elif mode=='zero_source_and_output_replay':a.zero_();output.zero_()
            return output
        timed_run.outputs=output;timed_run.rerun=replay
        return 0.25,dict(benchmark_method='cuda_graph')
    if mode=='correct':
        ms,metadata=checks.checked_benchmark(harness,benchmark,fn,warmup=10,repetition=100)
        assert ms==0.25 and metadata['perturbed_input_replay_checked'] and metadata['source_buffers_unchanged']
    else:
        with pytest.raises(AssertionError):checks.checked_benchmark(harness,benchmark,fn,warmup=10,repetition=100)
    assert observed==[dict(warmup=10,repetition=100)] and mod.bmm_triton is original


def test_bmm_adapter_installs_correctness_and_timed_checks(monkeypatch):
    adapter=module_at(ROOT/'tasks/triton2triton/vllm/triton_bmm/_arena_eval.py',monkeypatch)
    harness=adapter.load_harness()
    assert harness.run_correctness.__module__==harness.run_performance.__module__=='_bmm_checks'


def test_rms_reference_has_independent_known_answers(monkeypatch):
    runner = ROOT/'tasks/triton2triton/vllm/triton_rms_norm/scripts/task_runner.py'
    harness = module_at(runner,monkeypatch)
    x = torch.tensor([[3.,4.],[0.,0.]])
    w = torch.tensor([2.,0.5])
    expected = torch.tensor([[6./(12.5+1e-6)**0.5, 2./(12.5+1e-6)**0.5],[0.,0.]])
    result = harness.reference_rms_norm(x,w)
    torch.testing.assert_close(result,expected)
    assert not torch.allclose(result,torch.ones_like(result),atol=1e-2,rtol=1e-2)


@pytest.mark.parametrize('invalid', ['shape', 'dtype', 'device', 'nonfinite', 'values'])
def test_rms_output_contract_rejects_invalid_outputs(monkeypatch, invalid):
    task = ROOT/'tasks/triton2triton/vllm/triton_rms_norm'
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    harness = module_at(task/'scripts/task_runner.py', monkeypatch)
    x = torch.tensor([[3., 4.], [4., 3.]])
    weight = torch.tensor([2., 0.5])
    expected = harness.reference_rms_norm(x, weight)
    checks.check_output(expected, x, weight, 1e-6, harness.reference_rms_norm)
    output = {'shape': expected[:1], 'dtype': expected.double(),
              'device': torch.empty_like(expected, device='meta'),
              'nonfinite': torch.full_like(expected, float('inf')),
              'values': torch.zeros_like(expected)}[invalid]
    with pytest.raises(AssertionError):
        checks.check_output(output, x, weight, 1e-6, harness.reference_rms_norm)


@pytest.mark.parametrize('mode', ['correct', 'incorrect_timed', 'stale', 'no_write', 'changing_wrong'])
def test_rms_exact_timed_output_and_changed_input_replay(monkeypatch, mode):
    """CPU simulation of captured buffers; this does not claim GPU execution."""
    task = ROOT/'tasks/triton2triton/vllm/triton_rms_norm'
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    harness = module_at(task/'scripts/task_runner.py', monkeypatch)
    harness._TimedRun = SimpleNamespace
    x = torch.tensor([[3., 4.], [4., 3.]])
    weight = torch.tensor([2., 0.5])
    eps = 1e-6
    mod = SimpleNamespace(rms_norm=harness.reference_rms_norm)
    original = mod.rms_norm
    seen = []

    def fn():
        # Match the original harness, including its discarded return value.
        mod.rms_norm(x, weight, eps=eps)

    def benchmark(measured, *, timed_run, **kwargs):
        seen.append(kwargs)
        output = measured()
        cached = output.clone()
        if mode == 'incorrect_timed':
            output.zero_()

        def replay():
            if mode == 'correct':
                output.copy_(harness.reference_rms_norm(x, weight, eps))
            elif mode == 'stale':
                output.copy_(cached)
            elif mode == 'changing_wrong':
                output.fill_(x[0, 0])
            return output

        timed_run.outputs = output
        timed_run.rerun = replay
        return 0.25, {'benchmark_method': 'cuda_graph'}

    if mode == 'correct':
        ms, metadata = checks.checked_benchmark(harness, benchmark, fn, warmup=10, repetition=100)
        assert ms == 0.25
        assert metadata['timed_output_checked'] and metadata['perturbed_input_replay_checked']
    else:
        with pytest.raises(AssertionError):
            checks.checked_benchmark(harness, benchmark, fn, warmup=10, repetition=100)
    assert mod.rms_norm is original
    assert seen == [{'warmup': 10, 'repetition': 100}]


def test_rms_adapter_installs_output_checks(monkeypatch):
    adapter = module_at(ROOT/'tasks/triton2triton/vllm/triton_rms_norm/_arena_eval.py', monkeypatch)
    harness = adapter.load_harness()
    assert harness.run_correctness.__module__ == '_rms_output_checks'
    assert harness.run_performance.__module__ == '_rms_output_checks'


@pytest.mark.parametrize('mode', ['correct', 'incorrect_timed', 'stale', 'no_write', 'changing_wrong'])
def test_geak_gemm_exact_timed_output_and_changed_input_replay(monkeypatch, mode):
    checks = module_at(ROOT/'tasks/triton2triton/geak_eval/L3/gemm/_arena_checks.py', monkeypatch)
    monkeypatch.setitem(__import__('sys').modules, '_aka_benchmark', SimpleNamespace(TimedRun=SimpleNamespace))
    x = torch.tensor([[1., 2.], [3., 4.]])
    w = torch.tensor([[2., 1.], [4., 3.]])
    bias = torch.tensor([1., 2.])
    expected = torch.tensor([[5., 12.], [11., 26.]])
    checks.check_output(expected, x, w, bias)
    metadata = {'benchmark_method': 'cuda_graph'}
    seen = []

    def fn():
        return torch.nn.functional.linear(x, w, bias)

    def benchmark(measured, *, timed_run, **kwargs):
        seen.append(kwargs)
        output = measured()
        cached = output.clone()
        if mode == 'incorrect_timed':
            output.zero_()

        def replay():
            if mode == 'correct':
                output.copy_(torch.nn.functional.linear(x, w, bias))
            elif mode == 'stale':
                output.copy_(cached)
            elif mode == 'changing_wrong':
                output.fill_(x[0, 0])
            return output

        timed_run.outputs = output
        timed_run.rerun = replay
        return 0.5, metadata

    if mode == 'correct':
        ms, returned = checks.checked_benchmark(benchmark, fn, warmup=50, repetition=200)
        assert ms == 0.5 and returned is metadata
        assert metadata['timed_output_checked'] and metadata['perturbed_input_replay_checked']
    else:
        with pytest.raises(AssertionError):
            checks.checked_benchmark(benchmark, fn, warmup=50, repetition=200)
        assert 'perturbed_input_replay_checked' not in metadata
    assert seen == [{'warmup': 50, 'repetition': 200}]


def feed_forward_reference():
    path=ROOT/'tasks/triton2triton/geak_eval/L1/llama_ff_triton/test_kernel_harness.py'
    tree=ast.parse(path.read_text());definition=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='reference_ff')
    namespace={'torch':torch}
    exec(compile(ast.Module(body=[definition],type_ignores=[]),str(path),'exec'),namespace)
    return namespace['reference_ff']


def test_feed_forward_reference_independent_known_answer():
    import math
    reference=feed_forward_reference()
    x=torch.tensor([[[3.,4.]]]);rms_w=torch.tensor([2.,0.5]);w=torch.diag(torch.tensor([1.,-4.]))
    a=6./math.sqrt(12.5+1e-6);b=-8./math.sqrt(12.5+1e-6)
    expected=torch.tensor([[[a*a/(1+math.exp(-a)),b*b/(1+math.exp(-b))]]])
    torch.testing.assert_close(reference(x,w,w,rms_w),expected)


@pytest.mark.parametrize('mode',['correct','incorrect_timed','stale','no_write','changing_wrong'])
def test_feed_forward_timed_replay_restores_diagnostic_peer_inputs(monkeypatch,mode):
    checks=module_at(ROOT/'tasks/triton2triton/geak_eval/L1/llama_ff_triton/_arena_checks.py',monkeypatch)
    monkeypatch.setitem(__import__('sys').modules,'_aka_benchmark',SimpleNamespace(TimedRun=SimpleNamespace))
    h=SimpleNamespace(reference_ff=feed_forward_reference())
    x=torch.tensor([[[3.,4.]]]);rms_w=torch.tensor([2.,0.5])
    w1=torch.diag(torch.tensor([1.,-4.]));w3=w1.clone()
    original_x,original_rms=x.clone(),rms_w.clone()
    def fn(): return h.reference_ff(x,w1,w3,rms_w)
    seen=[]
    def benchmark(measured,*,timed_run,**kwargs):
        seen.append(kwargs);output=measured();cached=output.clone()
        if mode=='incorrect_timed':output.zero_()
        def replay():
            if mode=='correct':output.copy_(h.reference_ff(x,w1,w3,rms_w))
            elif mode=='stale':output.copy_(cached)
            elif mode=='changing_wrong':output.fill_(x.flatten()[0])
            return output
        timed_run.outputs=output;timed_run.rerun=replay
        return 0.25, {'benchmark_method':'cuda_graph'}
    if mode=='correct':
        ms,metadata=checks.checked_benchmark(h,benchmark,fn,warmup=50,repetition=200)
        assert ms==0.25 and metadata['perturbed_input_replay_checked']
    else:
        with pytest.raises(AssertionError): checks.checked_benchmark(h,benchmark,fn,warmup=50,repetition=200)
    torch.testing.assert_close(x,original_x,atol=0,rtol=0)
    torch.testing.assert_close(rms_w,original_rms,atol=0,rtol=0)
    assert seen==[dict(warmup=50,repetition=200)]


def test_feed_forward_diagnostic_reference_remains_original_call(monkeypatch):
    checks=module_at(ROOT/'tasks/triton2triton/geak_eval/L1/llama_ff_triton/_arena_checks.py',monkeypatch)
    monkeypatch.setitem(__import__('sys').modules,'_aka_benchmark',SimpleNamespace(TimedRun=SimpleNamespace))
    ref_fn=lambda: 'diagnostic output'
    fn=lambda:ref_fn()
    seen=[]
    def benchmark(actual,**kwargs):
        assert actual is fn
        seen.append(kwargs)
        return 0.5,dict(benchmark_method='cuda_graph')
    assert checks.checked_benchmark(None,benchmark,fn,warmup=50,repetition=200)[0]==0.5
    assert seen==[dict(warmup=50,repetition=200)]


@pytest.mark.parametrize('value', [1., float('inf'), float('nan')])
def test_geak_gemm_correctness_rejects_nonfinite_output(monkeypatch, value):
    checks = module_at(ROOT/'tasks/triton2triton/geak_eval/L3/gemm/_arena_checks.py', monkeypatch)
    original = lambda: torch.tensor([[value]])
    harness = SimpleNamespace(gemm_a16w16=original)
    visited = []

    def run(indices):
        visited.extend(indices)
        # Even a harness comparing matching infinities must not bypass the gate.
        torch.testing.assert_close(harness.gemm_a16w16(), original(), equal_nan=True)

    harness.run_correctness = run
    if value == 1.:
        assert checks.checked_correctness(harness, [0, 1, 2]) is None
    else:
        with pytest.raises(AssertionError, match='finite tensor'):
            checks.checked_correctness(harness, [0, 1, 2])
    assert visited == [0, 1, 2] and harness.gemm_a16w16 is original


@pytest.mark.parametrize('mode', ['correct', 'incorrect_timed', 'stale', 'no_write', 'changing_wrong'])
def test_awq_packed_known_answers_and_exact_replay(monkeypatch, mode):
    task = ROOT/'tasks/triton2triton/vllm/triton_awq_dequantize'
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    harness = module_at(task/'scripts/task_runner.py', monkeypatch)
    harness._TimedRun = SimpleNamespace
    qweight = torch.tensor([[0x76543210], [-19088744]], dtype=torch.int32)  # second word: 0xfedcba98
    scales = torch.arange(1, 9, dtype=torch.float16).repeat(2, 1)
    zeros = torch.tensor([[0x11111111], [-2004318072]], dtype=torch.int32)  # 0x88888888
    expected = torch.tensor([[-1, 6, 0, 16, 5, 30, 14, 48],
                             [0, 8, 3, 20, 10, 36, 21, 56]], dtype=torch.float16)
    checks.check_output(expected, qweight, scales, zeros, harness.reference_awq_dequantize)
    with pytest.raises(AssertionError):
        checks.check_output(expected.flip(-1), qweight, scales, zeros, harness.reference_awq_dequantize)
    original = lambda q, s, z: harness.reference_awq_dequantize(q, s, z, 1)
    mod = SimpleNamespace(awq_dequantize_triton=original)
    seen = []

    def fn():
        mod.awq_dequantize_triton(qweight, scales, zeros)

    def benchmark(measured, *, timed_run, **kwargs):
        seen.append(kwargs)
        output = measured()
        cached = output.clone()
        if mode == 'incorrect_timed': output.zero_()

        def replay():
            if mode == 'correct': output.copy_(original(qweight, scales, zeros))
            elif mode == 'stale': output.copy_(cached)
            elif mode == 'changing_wrong': output.fill_(int(qweight[0, 0]) & 15)
            return output

        timed_run.outputs, timed_run.rerun = output, replay
        return 0.25, {'benchmark_method': 'cuda_graph'}

    if mode == 'correct':
        ms, metadata = checks.checked_benchmark(harness, benchmark, fn, warmup=10, repetition=100)
        assert ms == 0.25 and metadata['perturbed_input_replay_checked']
    else:
        with pytest.raises(AssertionError):
            checks.checked_benchmark(harness, benchmark, fn, warmup=10, repetition=100)
    assert mod.awq_dequantize_triton is original
    assert seen == [{'warmup': 10, 'repetition': 100}]


@pytest.mark.parametrize('mode', ['correct', 'incorrect_timed', 'stale', 'no_write', 'wrong_ids', 'wrong_weights'])
def test_shared_expert_append_known_answers_and_both_timed_outputs(monkeypatch, mode):
    task = ROOT/'tasks/triton2triton/geak_eval/L1/fused_append_shared_experts'
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    monkeypatch.setitem(__import__('sys').modules, '_aka_benchmark', SimpleNamespace(TimedRun=SimpleNamespace))
    node = next(n for n in ast.parse((task/'test_kernel_harness.py').read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'reference_fused_append')
    namespace = {'torch': torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'protected-reference', 'exec'), namespace)
    reference = namespace['reference_fused_append']
    topk_ids = torch.tensor([[3, 7], [1, 0]], dtype=torch.int32)
    topk_weights = torch.tensor([[0.5, 0.25], [0.125, 0.75]])
    cfg = {'S': 1, 'N': 8, 'scale_factor': 1.0}
    expected = (torch.tensor([[3, 7, 8], [1, 0, 8]], dtype=torch.int32),
                torch.tensor([[0.5, 0.25, 1.], [0.125, 0.75, 1.]]))
    checks.check_output(expected, topk_ids, topk_weights, cfg, reference)
    original = lambda ids, weights, s, factor, N: reference(ids, weights, s, factor, N)
    harness = SimpleNamespace(reference_fused_append=reference, fused_append_shared_experts=original)
    seen = []

    def fn():
        harness.fused_append_shared_experts(topk_ids, topk_weights, cfg['S'], cfg['scale_factor'], N=cfg['N'])

    def benchmark(measured, *, timed_run, **kwargs):
        seen.append(kwargs)
        outputs = measured()
        cached = tuple(out.clone() for out in outputs)
        if mode == 'incorrect_timed': outputs[0].zero_()

        def replay():
            if mode in ('correct', 'wrong_ids', 'wrong_weights'):
                expected = reference(topk_ids, topk_weights, cfg['S'], cfg['scale_factor'], cfg['N'])
                for out, ref in zip(outputs, expected): out.copy_(ref)
                if mode == 'wrong_ids': outputs[0][:, -1].zero_()
                if mode == 'wrong_weights': outputs[1][:, -1].zero_()
            elif mode == 'stale':
                for out, ref in zip(outputs, cached): out.copy_(ref)
            return outputs

        timed_run.outputs, timed_run.rerun = outputs, replay
        return 0.5, {'benchmark_method': 'cuda_graph'}

    if mode == 'correct':
        ms, metadata = checks.checked_benchmark(harness, benchmark, fn, warmup=50, repetition=200)
        assert ms == 0.5 and metadata['perturbed_input_replay_checked']
    else:
        with pytest.raises(AssertionError):
            checks.checked_benchmark(harness, benchmark, fn, warmup=50, repetition=200)
    assert harness.fused_append_shared_experts is original
    assert seen == [{'warmup': 50, 'repetition': 200}]


ROCM = sorted([*(ROOT/'tasks/triton2triton/rocmbench').rglob('config.yaml'),
               *(ROOT/'tasks/instruction2triton').rglob('config.yaml')])


@pytest.mark.parametrize('relative', ['tasks/instruction2triton/rocmbench/triton_multreduce_matmul_kernel',
                                    'tasks/triton2triton/rocmbench/hard/triton_multreduce_matmul_kernel'])
def test_multreduce_declares_actual_fixed_launch_entrypoint(relative, monkeypatch, tmp_path):
    task = ROOT/relative
    spec = load_task_spec(task/'config.yaml', task_id=relative.removeprefix('tasks/'))
    assert {e.symbol for e in spec.candidate.entrypoints} == {'triton_matmul_kernel', 'triton_multreduce_matmul_kernel'}
    calls = []
    class Kernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs): calls.append((grid, args, kwargs))
            return launch
    wrapper = pure_functions(task/'triton_multreduce_matmul_kernel.py', ['multreduce_matmul_triton_wrapper'],
        dict(triton=SimpleNamespace(cdiv=lambda a, b: (a+b-1)//b), triton_matmul_kernel=Kernel()))
    a = torch.ones(2, 4); b = torch.ones(4, 2); c = torch.empty(2, 2)
    result = wrapper.multreduce_matmul_triton_wrapper(a, b, c, None, 2, 2, 4, 2, 2, 4, False, 4, 2)
    assert result is c and len(calls) == 1
    assert calls[0][2] == dict(BLOCK_SIZE_M=2, BLOCK_SIZE_N=2, BLOCK_SIZE_K=4,
                               USE_BIAS=False, USE_DOT=False, EVEN_K=True, num_warps=4, num_stages=2)
    # The real candidate inspector now rejects a missing timed core even when
    # the formerly sole declared autotuned wrapper is still present.
    adapter = module_at(task/'_arena_eval.py', monkeypatch)
    data = json.loads((task/'workloads.json').read_text())
    nodes = ast.parse((task/data['source']).read_text())
    for node in nodes.body:
        if isinstance(node, ast.FunctionDef) and node.name == 'triton_matmul_kernel':
            node.body = [ast.Pass()]
    (tmp_path/data['source']).write_text(ast.unparse(ast.fix_missing_locations(nodes)))
    monkeypatch.setattr(adapter, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='partially implemented'):
        adapter.inspect_candidate(data, require_implemented=True)


@pytest.mark.parametrize('relative', ['tasks/instruction2triton/rocmbench/test_matmul_MXFP',
                                    'tasks/triton2triton/rocmbench/hard/test_matmul_MXFP'])
def test_mxfp_unscaled_reference_preserves_fp32_operands(relative, monkeypatch):
    reference = module_at(ROOT/relative/'_arena_reference.py', monkeypatch)
    # Exactly representable FP32 operands cancel after an erroneous FP16 cast.
    a = torch.tensor([[1.000244140625, -1.]], dtype=torch.float32)
    b = torch.tensor([[1024.], [1024.]], dtype=torch.float32)
    output = torch.tensor([[0.25]], dtype=torch.float16)
    context = dict(is_scaled_mode=False, a_tensor=a, b_tensor=b, output_buffer=output)
    check = reference.prepare(context, None)
    check(output)
    early_cast = a.half() @ b.half()
    assert early_cast.item() == 0
    output.copy_(early_cast)
    with pytest.raises(reference.NumericalMismatch): check(output)
    output.fill_(0.25)
    a.zero_(); b.zero_()
    check(output)  # The expected answer was frozen before the candidate call.
    output.zero_()
    with pytest.raises(reference.NumericalMismatch): check(output)
    context.update(a_tensor=torch.tensor([[1., 2.]], dtype=torch.float16),
                   b_tensor=torch.tensor([[3.], [4.]], dtype=torch.float16))
    output.fill_(11)
    reference.prepare(context, None)(output)
    with pytest.raises(RuntimeError, match='Scaled MXFP'):
        reference.prepare({**context, 'is_scaled_mode': True}, None)


@pytest.mark.parametrize('relative', ['tasks/instruction2triton/rocmbench/test_matmul_MXFP',
                                    'tasks/triton2triton/rocmbench/hard/test_matmul_MXFP'])
def test_mxfp_scaled_matrix_reference_respects_32_element_groups(relative):
    source = ROOT/relative/'test_matmul_MXFP.py'
    ref = pure_functions(source, ['mxfp_to_bf16_torch', 'dot_scale_ref'])
    x = torch.full((2, 32), 0x22, dtype=torch.uint8)
    scales = torch.tensor([[127, 128], [126, 129]], dtype=torch.uint8)
    y = torch.full((64, 1), 0x3c, dtype=torch.uint8)  # E5M2 one.
    output = ref.dot_scale_ref(x, scales, y, 'e2m1', 'e5m2')
    expected = torch.tensor([[96.], [144.]], dtype=torch.bfloat16)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(torch.zeros_like(output), expected, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize('relative', ['tasks/instruction2triton/rocmbench/test_matmul_MXFP',
                                    'tasks/triton2triton/rocmbench/hard/test_matmul_MXFP'])
@pytest.mark.parametrize('mode', ['correct', 'zero', 'compiler_error'])
def test_mxfp_scaled_pipeline_runs_on_hip_and_rejects_zero_kernel(relative, mode):
    source = ROOT/relative/'test_matmul_MXFP.py'
    ref = pure_functions(source, ['mxfp_to_bf16_torch', 'dot_scale_ref'])
    launches = []
    class Kernel:
        def __getitem__(self, grid):
            def launch(a, scale_a, b, output, *args, **kwargs):
                launches.append((tuple(a.shape), tuple(scale_a.shape), tuple(b.shape), kwargs))
                if mode == 'compiler_error': raise RuntimeError('unsupported compiler lowering')
                if mode == 'zero': output.zero_()
                else: output.copy_(ref.dot_scale_ref(a, scale_a, b, kwargs['a_type'], kwargs['b_type']))
            return launch
    harness = pure_functions(source, ['test_pipeline_matmul'], dict(
        pytest=pytest, set_seed=lambda: torch.manual_seed(42), check_capabilities=lambda: None,
        is_cuda=lambda: False, is_hopper=lambda: False, is_hip_mi200=lambda: False,
        triton=SimpleNamespace(cdiv=lambda a, b: (a+b-1)//b), matmul_kernel=Kernel(),
        dot_scale_ref=ref.dot_scale_ref, result_gold={}))
    request = SimpleNamespace(node=SimpleNamespace(name='scaled-pipeline'))
    run = lambda: harness.test_pipeline_matmul(True, request, device='cpu')
    if mode == 'correct': run()
    elif mode == 'compiler_error':
        with pytest.raises(RuntimeError, match='unsupported compiler lowering'): run()
    else:
        with pytest.raises(AssertionError): run()
        # The original small-scale check passed; the second known-answer launch
        # must reject zero, retaining the same tolerance and all original cases.
        assert len(launches) == 2
    assert len(launches) == (1 if mode == 'compiler_error' else 2)
    assert all(row[:3] == ((512, 128), (512, 8), (256, 512)) for row in launches)
    assert all(row[3] == dict(NUM_STAGES=4, a_type='e2m1', b_type='e5m2') for row in launches)


@pytest.mark.parametrize('relative', ['tasks/instruction2triton/rocmbench/gemm',
                                    'tasks/triton2triton/rocmbench/hard/gemm'])
def test_rocm_gemm_scope_retains_original_scored_cases_and_numerical_gate(relative,monkeypatch):
    task=ROOT/relative
    source=task/'gemm.py'
    original=subprocess.check_output(['git','show',f'{BASE}:{source.relative_to(ROOT).as_posix()}'],cwd=ROOT)
    assert source.read_bytes()==original
    shapes=pure_functions(source,['get_x_vals']).get_x_vals()
    assert shapes==[(1024*v,1024*v,1024*v) for v in range(1,9)]+[
        (4864,4096,8192),(9728,8192,65536),(4864,8192,4160)]
    rows=json.loads((task/'workloads.json').read_text())['cases']
    assert len(rows)==22 and sum('performance' in r['checks'] for r in rows)==11
    for function,suffix in [('test_correctness',''),('test_performance','_str')]:
        selected=[r['params']['arguments'] for r in rows if r['params']['function']==function]
        assert {(r['M'],r['N'],r['K']) for r in selected}==set(shapes)
        assert len(selected)==11
        for args in selected:
            assert all(args[name+suffix]=='fp16' for name in ('in_dtype_a','in_dtype_b','out_dtype'))
            assert args['col_a'] is False and args['col_b'] is False
    # Independent small known answer, then a negative control, through both
    # original protected correctness and the added performance-input oracle.
    a=torch.tensor([[1.,2.],[3.,4.]],dtype=torch.float16)
    b=torch.tensor([[5.,6.],[7.,8.]],dtype=torch.float16)
    expected=torch.tensor([[19.,22.],[43.,50.]],dtype=torch.float16)
    context={'current_scale_a8_b8':None,'a':a,'b':b,'c':expected.clone()}
    reference=module_at(task/'_arena_reference.py',monkeypatch)
    check=reference.prepare(context,None)
    check(None)
    context['c'].zero_()
    with pytest.raises(reference.NumericalMismatch):check(None)
    def generator(m,n,dtype,col,seed,device):
        assert dtype==torch.float16 and col is False and device=='cuda'
        value=a if seed==1 else b
        return value,value.float(),None
    result=expected.clone()
    def candidate(a,b,c,*args,**kwargs):
        assert kwargs==dict(a_scale=None,b_scale=None,scale_a8_b8=None,activation='')
        c.copy_(result)
    protected=pure_functions(source,['get_x_vals','test_correctness'],dict(
        pytest=pytest,set_seed=lambda:None,name_to_torch_types={'fp16':torch.float16},gen_input=generator,
        dtype_is_8_bit=lambda dtype:False,matmul=candidate,result_gold={}))
    args=(2,2,2,False,False,'fp16','fp16','fp16',SimpleNamespace(node=SimpleNamespace(name='known-answer')))
    protected.test_correctness(*args)
    result.zero_()
    with pytest.raises(AssertionError):protected.test_correctness(*args)


@pytest.mark.parametrize('path', ROCM, ids=lambda p:p.parent.relative_to(ROOT/'tasks').as_posix())
def test_rocm_v2_preserves_original_source_and_complete_parameter_manifest(path):
    task=path.parent
    spec=load_task_spec(path,task_id=task.relative_to(ROOT/'tasks').as_posix())
    assert spec.candidate.initial_state=='implemented' and spec.baseline.kind=='initial_candidate'
    assert all(edit.scope=='symbols' for edit in spec.candidate.editable)
    data=json.loads((task/'workloads.json').read_text())
    source=task/data['source']
    original=subprocess.check_output(['git','show',f'{BASE}:{source.relative_to(ROOT).as_posix()}'],cwd=ROOT)
    expected_source = original
    if task.name == 'test_kernel_sub':
        historical_skip = b'    pytest.skip("Skipping ASTSource compile-in-subprocess check on Triton 3.3 due to known API/compiler instability; numerical correctness tests cover kernel behavior.")\n'
        assert original.count(historical_skip) == 1
        expected_source = original.replace(historical_skip, b'')
    if task.name == 'rmsnorm_fwd':
        skip = b'    # Ensure in_dtype and out_dtype are compatible for RMSNorm (usually they are the same for x and y)\n    # For benchmarking, let\'s assume in_dtype is the primary type for x and g, and y.\n    if in_dtype_str != out_dtype_str:\n         pytest.skip(f"Skipping perf test where in_dtype {in_dtype_str} != out_dtype {out_dtype_str} for simplicity.")\n\n'
        assert expected_source.count(skip) == 1
        expected_source = expected_source.replace(skip,b'').replace(
            b'y_buffer = torch.empty_like(x) # Output buffer for forward',
            b'y_buffer = torch.empty_like(x, dtype=arg_to_torch_dtype[out_dtype_str]) # Declared output dtype').replace(
            b'baseline_callable = lambda: torch_rmsnorm_fwd(x, g, ZERO_CENTERED_GAMMA, current_dtype, eps)',
            b'baseline_callable = lambda: torch_rmsnorm_fwd(x, g, ZERO_CENTERED_GAMMA, arg_to_torch_dtype[out_dtype_str], eps)').replace(
            b'rms = torch.sqrt(torch.sum(x_f32 * x_f32, dim=-1) * 1 / N)',
            b'rms = torch.sqrt(torch.sum(x_f32 * x_f32, dim=-1) * 1 / N + epsilon)')
    if task.name == 'test_matmul_MXFP':
        expected_source = expected_source.replace(
            b'    if scale and not is_cuda():\n        pytest.skip("NYI: scale_dot just implemented in CUDA")\n', b'').replace(
            b'    x_upcast = mxfp_to_bf16_torch(x, scale, type_x)',
            b'    x_grouped = x.reshape(*scale.shape, -1)\n    x_upcast = mxfp_to_bf16_torch(x_grouped, scale, type_x).reshape(x.shape[0], -1)')
        # Only the separately exercised known-answer diagnostic is additional;
        # all old source bytes, including gates/parameters, remain protected.
        after = source.read_text()
        start = after.index('\n    # Unscored known-answer control at an ordinary E8M0 scale.')
        end = after.index('\n\n# Define these globally', start)
        extra = after[start:end].rstrip('\n') + '\n'
        anchor = b'    torch.testing.assert_close(ref_out, output, atol=atol, rtol=rtol, equal_nan=scale)\n'
        expected_source = expected_source.replace(anchor, anchor + extra.encode())
    assert source.read_bytes() == expected_source
    assert hashlib.sha256(original).hexdigest()==data['migration']['original_source_sha256']
    rows=data['cases']
    assert len(rows)==len({row['test_case_id'] for row in rows})
    assert all('correctness' in row['checks'] for row in rows)
    assert any(row['checks']==['correctness','performance'] for row in rows)
    assert {entry.symbol for entry in spec.candidate.entrypoints}==set(data['kernel_symbols'])
    assert 'task_type' not in spec.to_mapping()
    ast.parse((task/'_arena_eval.py').read_text())
    ast.parse((task/'_arena_reference.py').read_text())



@pytest.mark.parametrize('exitcode,timed_out', [(0,False),(1,False),(None,True)])
@pytest.mark.parametrize('task', ['tasks/triton2triton/rocmbench/easy/test_kernel_sub',
                                'tasks/instruction2triton/rocmbench/test_kernel_sub'])
def test_kernel_sub_declared_compilation_case_executes_and_rejects_child_failure(task,exitcode,timed_out):
    # Run the protected test body against a controlled process, verifying that
    # a skipped/nonzero/timed-out compilation cannot become successful evidence.
    tree=ast.parse((ROOT/task/'test_kernel_sub.py').read_text())
    definition=next(node for node in tree.body if isinstance(node,ast.FunctionDef)
                    and node.name=='test_compile_kernel_sub_in_subproc')
    definition.decorator_list=[]
    calls=[]
    def compile_target(): pass
    class Process:
        def __init__(self, target):
            assert target is compile_target
            self.exitcode=exitcode
        def start(self): calls.append('start')
        def join(self, timeout=None): calls.append(('join',timeout))
        def is_alive(self): return timed_out
        def terminate(self): calls.append('terminate')
    namespace=dict(multiprocessing=SimpleNamespace(set_start_method=lambda *a,**k:None,Process=Process),
                   set_seed=lambda:None,pytest=pytest,torch=torch,result_gold={},
                   compile_kernel_sub_for_test=compile_target)
    exec(compile(ast.Module(body=[definition],type_ignores=[]),str(ROOT/task/'test_kernel_sub.py'),'exec'),namespace)
    run=lambda: namespace[definition.name]('fresh-cache',SimpleNamespace(node=SimpleNamespace(name='compile-case')))
    if timed_out:
        with pytest.raises(pytest.fail.Exception,match='Process timed out'): run()
        assert 'terminate' in calls
    elif exitcode != 0:
        with pytest.raises(AssertionError): run()
    else:
        run()
        assert namespace['result_gold']['compile_case'].item()==1.0
    assert calls[:2]==['start',('join',60)]


@pytest.mark.parametrize('task',['tasks/instruction2triton/rocmbench/rmsnorm_fwd','tasks/triton2triton/rocmbench/medium/rmsnorm_fwd'])
@pytest.mark.parametrize('epsilon', [1e-6, 1e-5])
@pytest.mark.parametrize('zero_centered', [True, False])
def test_rms_direct_reference_uses_declared_epsilon_on_near_zero_rows(task, epsilon, zero_centered):
    import math
    source = ROOT/task/'rmsnorm_fwd.py'
    ref = pure_functions(source, ['torch_rmsnorm_fwd'])
    x = torch.tensor([[0., 0.], [0.001, 0.001]])
    g = torch.tensor([[1., 2.]])
    y, rsigma = ref.torch_rmsnorm_fwd(x, g, zero_centered, torch.float32, epsilon)
    expected_rs = torch.tensor([1/math.sqrt(epsilon), 1/math.sqrt(1e-6+epsilon)])
    gain = [2., 3.] if zero_centered else [1., 2.]
    expected_y = torch.tensor([[0., 0.], [0.001*expected_rs[1]*v for v in gain]])
    torch.testing.assert_close(y, expected_y, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(rsigma, expected_rs, atol=1e-5, rtol=1e-5)
    assert torch.isfinite(y).all() and torch.isfinite(rsigma).all()
    if epsilon == 1e-6:
        default_y, default_rs = ref.torch_rmsnorm_fwd(x, g, zero_centered, torch.float32)
        torch.testing.assert_close(default_y, y)
        torch.testing.assert_close(default_rs, rsigma)
    before = subprocess.check_output(['git', 'show', f'{BASE}:{source.relative_to(ROOT).as_posix()}'], cwd=ROOT, text=True)
    node = next(n for n in ast.parse(before).body if isinstance(n, ast.FunctionDef) and n.name == 'torch_rmsnorm_fwd')
    scope = {'torch': torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), scope)
    old_y, old_rs = scope['torch_rmsnorm_fwd'](x, g, zero_centered, torch.float32, epsilon)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(old_rs, expected_rs, atol=1e-5, rtol=1e-5)
    assert not torch.isfinite(old_y[0]).all()


@pytest.mark.parametrize('task',['tasks/instruction2triton/rocmbench/rmsnorm_fwd','tasks/triton2triton/rocmbench/medium/rmsnorm_fwd'])
@pytest.mark.parametrize('in_name',['fp16','bf16','fp32'])
@pytest.mark.parametrize('out_name',['fp16','bf16','fp32'])
@pytest.mark.parametrize('zero_centered',[False,True])
def test_rms_forward_declared_dtype_pairs_execute_original_performance_body(monkeypatch,task,in_name,out_name,zero_centered):
    import inspect,math
    oracle_module=module_at(ROOT/task/'_arena_reference.py',monkeypatch)
    mapping={'fp16':torch.float16,'bf16':torch.bfloat16,'fp32':torch.float32}
    calls=[]
    class CpuTorch:
        def __getattr__(self,name):
            value=getattr(torch,name)
            if name in ('randn','rand','empty','empty_like'):
                def cpu(*args,**kwargs):
                    kwargs['device']='cpu'
                    return value(*args,**kwargs)
                return cpu
            return value
    def run_kernel(x,g,y,rsigma,*args):
        assert x.dtype==mapping[in_name] and y.dtype==mapping[out_name]
        zero,eps=args[5],args[-1]
        for row in range(x.shape[0]):
            inv=1/math.sqrt(sum(float(v)**2 for v in x[row])/x.shape[1]+eps)
            rsigma[row]=inv
            for col in range(x.shape[1]):
                y[row,col]=float(x[row,col])*inv*(float(g[col])+int(zero))
        calls.append('kernel')
        return y
    class Benchmark:
        def __init__(self,op_callable,config,**kwargs):
            assert config==(10,100)
            self.context=dict(inspect.currentframe().f_back.f_locals)
            self.op=op_callable
        def run_benchmark(self,**kwargs):
            output=self.op()
            oracle_module.prepare(self.context,None)(output)
            invalid={**self.context,'y_buffer':output.double()}
            with pytest.raises(ValueError,match='dtype'): oracle_module.prepare(invalid,None)(output)
            output.zero_()
            with pytest.raises(oracle_module.NumericalMismatch): oracle_module.prepare(self.context,None)(output)
    tree=ast.parse((ROOT/task/'rmsnorm_fwd.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='test_performance');fn.decorator_list=[]
    namespace=dict(torch=CpuTorch(),pytest=pytest,arg_to_torch_dtype=mapping,set_seed=lambda:torch.manual_seed(42),
                   get_num_sms=lambda:2,triton=SimpleNamespace(next_power_of_2=lambda n:1<<(n-1).bit_length()),
                   rmsnorm=run_kernel,PytestBenchmarker=Benchmark,do_bench_config=lambda warm_up,repetition:(warm_up,repetition),
                   OP_NAME_FOR_BENCHMARK='fixture',calculate_rmsnorm_fwd_gbps=None,calculate_rmsnorm_fwd_tflops=None)
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(ROOT/task/'rmsnorm_fwd.py'),'exec'),namespace)
    namespace['test_performance'](2,8,zero_centered,in_name,out_name,None)
    assert calls==['kernel']

def oracle(name,monkeypatch):
    return module_at(ROOT/'tasks/instruction2triton/rocmbench'/name/'_arena_reference.py',monkeypatch)


def test_philox_known_counter_and_uniform_negative_control(monkeypatch):
    reference=oracle('test_randn',monkeypatch)
    # Philox4x32-10 zero counter/key known answer (first lane).
    assert reference.philox32(0,1).tolist()==[0x6627e8d5]
    output=torch.zeros(16)
    check=reference.prepare({'seed_val':0,'N_elements':16,'x_output_buffer':output},None)
    with pytest.raises(reference.NumericalMismatch):check(output)


def test_swizzle_matches_original_explicit_known_order(monkeypatch):
    reference=oracle('test_triton_swizzle2d',monkeypatch)
    expected=reference.swizzle_reference(5,7,3,dtype=torch.int32,device='cpu')
    task=ROOT/'tasks/instruction2triton/rocmbench/test_triton_swizzle2d/test_triton_swizzle2d.py'
    tree=ast.parse(task.read_text());test=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='test_swizzle2d')
    assignment=next(n for n in ast.walk(test) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='expected_order' for t in n.targets))
    original_order=ast.literal_eval(assignment.value.func.value.args[0])
    assert expected.tolist()==original_order
    with pytest.raises(reference.NumericalMismatch):reference.compare(expected,torch.arange(35).reshape(5,7),exact=True,check_dtype=False)


def test_rms_backward_reference_matches_autograd_and_rejects_wrong_gradients(monkeypatch):
    reference=oracle('rmsnorm_bwd',monkeypatch)
    x=torch.tensor([[1.,2.,3.],[2.,-1.,4.]],requires_grad=True)
    g=torch.tensor([0.2,0.4,0.8],requires_grad=True)
    go=torch.tensor([[1.,0.,2.],[-1.,2.,0.5]])
    y=x*torch.rsqrt((x*x).mean(-1,keepdim=True)+1e-5)*(g+1)
    y.backward(go)
    dx=x.grad.clone();dg=g.grad.clone()
    context={'x':x.detach(),'g':g.detach(),'grad_output':go,'ZERO_CENTERED_GAMMA':True,'eps':1e-5,
             'dx_bench':dx,'dg_tmp_bench':torch.stack([dg,torch.zeros_like(dg)])}
    check=reference.prepare(context,None);check(None)
    dx.zero_()
    with pytest.raises(reference.NumericalMismatch):check(None)


def test_oracle_distinguishes_output_contract_failure_from_numerical_error(monkeypatch):
    reference=oracle('test_add_kernel',monkeypatch)
    with pytest.raises(ValueError,match='shape/device'):
        reference.compare(torch.zeros(2),torch.zeros(3),atol=1e-3,rtol=1e-2)
    with pytest.raises(ValueError,match='nonfinite'):
        reference.compare(torch.tensor([float('nan')]),torch.zeros(1),atol=1e-3,rtol=1e-2)


@pytest.mark.parametrize('relative',['tasks/instruction2triton/rocmbench/test_add_kernel','tasks/triton2triton/rocmbench/easy/test_add_kernel'])
@pytest.mark.parametrize('mode',['correct','incorrect_timed','stale','no_write','changing_wrong','event_fallback','replay_raises',
                                'zero_inputs_and_output_timed','zero_inputs_and_output_replay'])
def test_add_canonical_samples_observe_exact_replay_and_reject_wrong_output(monkeypatch,relative,mode):
    # Use the actual PytestBenchmarker configuration and statistics helper,
    # with a CPU graph simulator only at the canonical device-timing boundary.
    task=ROOT/relative
    adapter=module_at(task/'_arena_eval.py',monkeypatch)
    reference=module_at(task/'_arena_reference.py',monkeypatch)
    monkeypatch.setitem(__import__('sys').modules,'_arena_reference',reference)
    class Timed:
        outputs=None
        def _bind(self,rerun,outputs):self.outputs=outputs;self.rerun=rerun
    calls=[]
    def timer(fn,*,timed_run=None,**kwargs):
        assert timed_run is not None
        assert kwargs['warmup']==10 and kwargs['repetition']==100
        calls.append(kwargs)
        if mode=='event_fallback':raise RuntimeError('timed_run requires an observable CUDA-graph replay')
        output=fn();saved=output.clone()
        if mode=='incorrect_timed':output.zero_()
        def replay():
            if mode=='replay_raises':raise RuntimeError('injected replay failure')
            if mode=='correct' or mode.startswith('zero_inputs'):fn()
            elif mode=='stale':output.copy_(saved)
            elif mode=='changing_wrong':output.fill_(123)
            return output
        timed_run._bind(replay,output)
        return [0.25]*100,dict(benchmark_method='cuda_graph',benchmark_warmup=10,benchmark_samples=100)
    monkeypatch.setitem(__import__('sys').modules,'_aka_benchmark',SimpleNamespace(
        TimedRun=Timed,benchmark_cuda_graph_or_events_samples=timer))
    helper=module_at(ROOT/'src/tools/perf/performance_utils_pytest.py',monkeypatch)
    monkeypatch.setitem(__import__('sys').modules,'performance_utils_pytest',helper)
    plugin=SimpleNamespace(action='performance',current_row={'test_case_id':'cpu-replay'},exercised=set())
    Checked=adapter.benchmark_type(helper.PytestBenchmarker,plugin,None)
    def make():
        x=torch.tensor([2.,4.,6.]);y=torch.tensor([1.,3.,5.]);output=torch.empty_like(x)
        launches=[]
        def launch():
            launches.append(None)
            output.copy_(x+y)
            if (mode=='zero_inputs_and_output_timed' and len(launches)>=2 or
                    mode=='zero_inputs_and_output_replay' and len(launches)>=3):
                x.zero_();y.zero_();output.zero_()
            return 'compiled-kernel-handle'
        return Checked(op_callable=launch,op_name='add',config=helper.do_bench_config(warm_up=10,repetition=100))
    benchmark=make();original=benchmark.op_callable
    inputs=tuple(benchmark.context[name] for name in ('x','y'))
    pristine=tuple(value.clone() for value in inputs)
    if mode=='correct':
        record=benchmark.run_benchmark(current_params_dict={})
        assert record['timing_ms']['mean']==0.25
        assert plugin.current_row['metadata']['perturbed_input_replay_checked']
        assert plugin.current_row['metadata']['device_timing']['benchmark_samples']==100
        assert plugin.exercised=={'cpu-replay'}
        # Compare effective public-timer arguments with the actual old helper
        # path, including defaults hidden behind the task's direct sample call.
        import inspect
        prior=[]
        monkeypatch.setattr(helper,'benchmark_cuda_graph_or_events_samples',
                            lambda fn,**options:(prior.append(options) or [0.25],{}))
        helper._measure_times(original,benchmark.config,prepare_fn=benchmark.prepare_fn,
                              use_cuda_graph=benchmark.use_cuda_graph,fallback_reason=benchmark.fallback_reason)
        canonical=module_at(ROOT/'src/tools/perf/aka_benchmark.py',monkeypatch)
        signature=inspect.signature(canonical.benchmark_cuda_graph_or_events_samples)
        def effective(options):
            bound=signature.bind_partial(None,**options);bound.apply_defaults()
            return {k:v for k,v in bound.arguments.items() if k not in ('fn','timed_run')}
        assert effective(calls[0])==effective(prior[0])
    elif mode=='replay_raises':
        with pytest.raises(RuntimeError,match='injected replay failure'):
            benchmark.run_benchmark(current_params_dict={})
    elif mode=='event_fallback':
        with pytest.raises(RuntimeError,match='observable'):benchmark.run_benchmark(current_params_dict={})
    else:
        with pytest.raises((reference.NumericalMismatch,ValueError)):
            benchmark.run_benchmark(current_params_dict={})
    assert all(torch.equal(value,expected) for value,expected in zip(inputs,pristine))
    assert all(benchmark.context[name] is value for name,value in zip(('x','y'),inputs))
    if mode!='correct':assert not plugin.exercised
    assert benchmark.op_callable is original and len(calls)==1


@pytest.mark.parametrize('path', [p for p in ROCM if p.parent.name!='test_add_kernel'], ids=lambda p:p.parent.relative_to(ROOT/'tasks').as_posix())
def test_rocm_timing_evidence_retains_canonical_fallback_reason(monkeypatch, path):
    adapter = module_at(path.parent/'_arena_eval.py', monkeypatch)
    expected = torch.tensor([2.])
    def prepare(context, module):
        return lambda value: torch.testing.assert_close(value, expected)
    monkeypatch.setitem(__import__('sys').modules, '_arena_reference', SimpleNamespace(prepare=prepare))
    plugin = SimpleNamespace(action='performance', current_row={'test_case_id':'cpu-fixture'}, exercised=set())
    record = {'timing_ms': {'mean': 0.25}, 'benchmark_method':'cuda_event_fallback',
              'benchmark_fallback_reason':'protected invocation uses a GPU scalar on the host',
              'benchmark_samples':100, 'benchmark_warmup':10, 'benchmark_effective_repeats':1,
              'params':{'not_timing_metadata':True}}
    class Base:
        def __init__(self, op_callable, **kwargs):
            self.op_callable = op_callable
            self.prepare_fn = None
        def run_benchmark(self, **kwargs):
            self.op_callable()
            return record
    benchmark = adapter.benchmark_type(Base, plugin, None)(op_callable=lambda: expected.clone())
    benchmark.run_benchmark()
    assert plugin.current_row['metadata']['device_timing'] == {
        k:v for k,v in record.items() if k.startswith('benchmark_')}
    assert plugin.current_row['metadata']['timed_output_checked']
    assert plugin.current_row['execution_time_ms'] == 0.25


@pytest.mark.parametrize('relative', ['tasks/instruction2triton/rocmbench/moe_gemm',
                                     'tasks/triton2triton/rocmbench/hard/moe_gemm'])
def test_moe_launcher_selects_events_before_attempting_unsupported_capture(monkeypatch, relative):
    adapter = module_at(ROOT/relative/'_arena_eval.py', monkeypatch)
    expected = torch.tensor([2.])
    monkeypatch.setitem(__import__('sys').modules, '_arena_reference', SimpleNamespace(
        prepare=lambda context,module: lambda value: torch.testing.assert_close(value, expected)))
    plugin = SimpleNamespace(action='performance', current_row={'test_case_id':'cpu-fixture'}, exercised=set())
    options = {}
    class Base:
        def __init__(self, op_callable, use_cuda_graph=True, fallback_reason=None):
            options.update(use_cuda_graph=use_cuda_graph, fallback_reason=fallback_reason)
            self.op_callable = op_callable
            self.prepare_fn = None
        def run_benchmark(self, **kwargs):
            if options['use_cuda_graph']:
                raise RuntimeError('The protected .item() launcher cannot be captured')
            self.op_callable()
            return {'timing_ms':{'mean':0.25}, 'benchmark_method':'cuda_event_fallback',
                    'benchmark_fallback_reason':options['fallback_reason']}
    benchmark = adapter.benchmark_type(Base,plugin,None)(op_callable=lambda:expected.clone())
    benchmark.run_benchmark()
    assert options['use_cuda_graph'] is False and '.item()' in options['fallback_reason']
    assert plugin.current_row['metadata']['device_timing']['benchmark_fallback_reason'] == options['fallback_reason']
    assert plugin.exercised == {'cpu-fixture'}


def test_skip_is_not_passing_protocol_evidence(monkeypatch):
    adapter=module_at(ROCM[0].parent/'_arena_eval.py',monkeypatch)
    data=json.loads((ROCM[0].parent/'workloads.json').read_text())
    plugin=adapter.ReportPlugin(data,'correctness')
    key=next(iter(plugin.rows));plugin.node_rows['item']=key
    plugin.pytest_runtest_logreport(SimpleNamespace(nodeid='item',when='setup',failed=False,skipped=True,longrepr='unsupported device'))
    plugin.pytest_runtest_logreport(SimpleNamespace(nodeid='item',when='teardown',failed=False,skipped=False))
    assert plugin.rows[key]['status']=='FAIL' and plugin.rows[key]['failure_kind']=='not_executed'


def test_rocm_real_pytest_lifecycle_and_manifest_drift(tmp_path,monkeypatch):
    """CPU functions exercise real pytest setup/call/teardown and collection hooks."""
    import sys
    adapter=module_at(ROCM[0].parent/'_arena_eval.py',monkeypatch)
    monkeypatch.setattr(adapter,'ROOT',tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('PYTEST_DISABLE_PLUGIN_AUTOLOAD','1')
    source='''import pytest
from types import SimpleNamespace
triton=SimpleNamespace(jit=lambda f:f)
@triton.jit
def kernel():
    return 1
@pytest.mark.parametrize('x',[1,2])
def test_value(x):
    assert x > 0
'''
    (tmp_path/'test_cpu_migration.py').write_text(source)
    rows=[{'test_case_id':adapter.identity('test_value',{'x':x}),
           'params':{'function':'test_value','arguments':{'x':x}},
           'checks':['correctness'],'status':'PASS'} for x in (1,2)]
    data={'source':'test_cpu_migration.py','kernel_symbols':['kernel'],'cases':rows}
    (tmp_path/'workloads.json').write_text(json.dumps(data))
    result=adapter.evaluate('candidate','correctness')
    assert result['status']=='PASS',result
    assert all(r['status']=='PASS' and r['metrics']['original_pytest_passed'] for r in result['cases'])
    # A manifest cannot silently lose one original case.
    data['cases'].pop();(tmp_path/'workloads.json').write_text(json.dumps(data))
    result=adapter.evaluate('task','validate-task')
    assert result['status']=='FAIL' and 'manifest' in result['reason']
    monkeypatch.delitem(sys.modules,'test_cpu_migration',raising=False)


def test_rocm_missing_candidate_fails_every_applicable_case(tmp_path,monkeypatch):
    adapter=module_at(ROCM[0].parent/'_arena_eval.py',monkeypatch)
    data=json.loads((ROCM[0].parent/'workloads.json').read_text())
    monkeypatch.setattr(adapter,'ROOT',tmp_path)
    (tmp_path/'workloads.json').write_text(json.dumps(data))
    result=adapter.evaluate('candidate','performance')
    assert result['status']=='FAIL'
    assert result['cases'] and all(r['status']=='FAIL' and 'performance' in r['checks'] for r in result['cases'])
    result_record(result)


GEAK = sorted((ROOT/'tasks/triton2triton/geak_eval').rglob('config.yaml'))


@pytest.mark.parametrize('path', GEAK, ids=lambda p:p.parent.name)
def test_geak_v2_preserves_original_functions_and_freezes_the_complete_manifest(path):
    task=path.parent
    spec=load_task_spec(path,task_id=task.relative_to(ROOT/'tasks').as_posix())
    assert spec.candidate.initial_state=='implemented' and spec.baseline.kind=='initial_candidate'
    assert spec.candidate.language=='triton'
    assert all(edit.scope=='symbols' for edit in spec.candidate.editable)
    data=json.loads((task/'workloads.json').read_text())
    source=(task/data['source']).relative_to(ROOT).as_posix()
    original=subprocess.check_output(['git','show',f'{BASE}:{source}'],cwd=ROOT)
    assert (ROOT/source).read_bytes()==original
    before=subprocess.check_output(['git','show',f'{BASE}:{task.relative_to(ROOT).as_posix()}/test_kernel_harness.py'],cwd=ROOT,text=True)
    after=(task/'test_kernel_harness.py').read_text()
    assert data['migration']['original_harness_sha256']==hashlib.sha256(before.encode()).hexdigest()
    assert data['migration']['original_source_sha256']==hashlib.sha256(original).hexdigest()
    olddefs={n.name:n for n in ast.parse(before).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    newdefs={n.name:n for n in ast.parse(after).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    bootstrap={'_find_baseline_kernel_dir','_load_baseline_triton','_resolve_geak_kernel_dir','_register_geak_aliases'}
    for name in olddefs.keys()-bootstrap-({'e8m0_to_f32'} if data['migration'].get('reference_fixes') else set()):
        assert ast.get_source_segment(before,olddefs[name])==ast.get_source_segment(after,newdefs[name]),name
    assert 'os.environ.get("GEAK_WORK_DIR"' not in after
    assert 'os.environ.get("GEAK_REPO_ROOT"' not in after
    perfrows=[r for r in data['cases'] if 'performance' in r['checks']]
    assert [r['params']['configuration'] for r in perfrows]==data['input_tables']['performance']
    assert [r['params']['case_index'] for r in perfrows]==list(range(len(perfrows)))
    configs=[r['params']['configuration'] for r in data['cases']]
    assert all(v in configs for v in data['input_tables']['original_correctness'])
    assert all('correctness' in r['checks'] for r in data['cases'])
    result_record({'protocol':'arena-eval-v1','role':'task','action':'validate-task','status':'PASS','cases':data['cases']})
    for role in ('baseline','candidate'):
        # Real command smoke: AST compilation only, no JIT/GPU execution.
        run=subprocess.run([__import__('sys').executable,'_arena_eval.py',role,'compile'],cwd=task,text=True,capture_output=True)
        parsed=parse_command_result(run.stdout,role=role,action='compile',returncode=run.returncode)
        assert parsed.status=='PASS',run.stdout+run.stderr


@pytest.mark.parametrize('stream_kind',['captured','text','reconfigure_error'])
def test_quant_sort_harness_bootstrap_supports_captured_stdout(monkeypatch,stream_kind):
    import io,sys
    from contextlib import redirect_stdout
    path=ROOT/'tasks/triton2triton/geak_eval/L3/fused_mxfp4_quant_moe_sort/test_kernel_harness.py'
    tree=ast.parse(path.read_text())
    # Execute the actual entire import/bootstrap prefix up to GPU dependencies.
    end=next(i for i,node in enumerate(tree.body) if isinstance(node,ast.Import)
             and any(alias.name=='torch' for alias in node.names))
    prefix=compile(ast.Module(body=tree.body[:end],type_ignores=[]),str(path),'exec')
    monkeypatch.setitem(sys.modules,'_aka_benchmark',SimpleNamespace(benchmark_cuda_graph_or_events_samples=None))
    class FaultyStream(io.StringIO):
        def reconfigure(self,**kwargs):raise RuntimeError('broken text stream')
    stream=(io.TextIOWrapper(io.BytesIO()) if stream_kind=='text' else
            FaultyStream() if stream_kind=='reconfigure_error' else io.StringIO())
    try:
        with redirect_stdout(stream):
            if stream_kind=='reconfigure_error':
                with pytest.raises(RuntimeError,match='broken text stream'):exec(prefix,{})
            else:exec(prefix,{})
        if stream_kind=='text':assert stream.line_buffering
        if stream_kind=='captured':
            before=subprocess.check_output(['git','show',f'{BASE}:{path.relative_to(ROOT).as_posix()}'],cwd=ROOT,text=True)
            old=next(node for node in ast.parse(before).body if isinstance(node,ast.Expr)
                     and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute)
                     and node.value.func.attr=='reconfigure')
            with pytest.raises(AttributeError,match='reconfigure'):
                exec(compile(ast.Module(body=[old],type_ignores=[]),str(path),'exec'),
                     {'sys':SimpleNamespace(stdout=stream)})
    finally:stream.close()


def _quant_sort_cpu_harness():
    """Original quantizer/decoder and benchmark orchestration; CPU sort fixture."""
    from types import ModuleType
    path = ROOT/'tasks/triton2triton/geak_eval/L3/fused_mxfp4_quant_moe_sort/test_kernel_harness.py'
    h = ModuleType('_quant_sort_cpu')
    quant = pure_functions(path, ['_torch_dynamic_mxfp4_quant'])
    decoder = pure_functions(path, ['mxfp4_to_f32'], {'torch': SimpleNamespace(
        float32=torch.float32, tensor=lambda values, **kwargs: torch.tensor(
            values, **{**kwargs, 'device': 'cpu'}))})
    conversion = pure_functions(path, ['e8m0_to_f32', 'convert_mxfp4_to_fp32'],
        {'SCALE_GROUP_SIZE': 32, 'mxfp4_to_f32': decoder.mxfp4_to_f32})
    h.dynamic_mxfp4_quant = quant._torch_dynamic_mxfp4_quant
    h.convert_mxfp4_to_fp32 = conversion.convert_mxfp4_to_fp32
    h._fp4x2 = h._fp8_e8m0 = torch.uint8
    def reference(x, sorted_ids, token_num, topk, q_dtype_a, local, valid, block):
        packed, scales = quant._torch_dynamic_mxfp4_quant(x)
        rows = (sorted_ids & 0xffffff) * topk + (sorted_ids >> 24)
        return packed, scales[rows], scales
    h.run_fused_dynamic_mxfp4_quant_moe_sort_ref = reference
    def candidate(x, sorted_ids, num_valid_ids, token_num, topk, block_size=32):
        # Independent output generation on CPU; no Triton execution claimed.
        packed, scales, _ = reference(x, sorted_ids, token_num, topk, None, None,
                                      num_valid_ids, block_size)
        return packed, scales
    h.fused_dynamic_mxfp4_quant_moe_sort = candidate
    x = torch.tensor([[0., .5, 1., 1.5, 2., 3., 4., 6.]]).repeat(4, 4)
    x.mul_(torch.tensor([1., 2., -1., -2.])[:, None])
    inp = dict(x=x, sorted_ids=torch.tensor([1 << 24, 1], dtype=torch.int64),
               num_valid_ids=torch.tensor([2, 2], dtype=torch.int64), token_num=2,
               topk=2, block_size_M=128)
    h._make_inputs = lambda config: inp
    h._cfg_label = lambda config: 'CPU synthetic quant-sort orchestration'
    h.ALL_CONFIGS = [object()]
    h.WARMUP, h.ITERATIONS = 50, 200
    h.math = __import__('math')
    tree = ast.parse(path.read_text())
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_benchmark']
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), h.__dict__)
    return h, inp


def test_quant_sort_reference_known_packed_nibbles_and_scales():
    h, inp = _quant_sort_cpu_harness()
    packed, scales = h.dynamic_mxfp4_quant(inp['x'])
    assert packed[0].tolist() == [0x10, 0x32, 0x54, 0x76] * 4
    assert scales[:, 0].tolist() == [127, 128, 127, 128]
    decoded = h.convert_mxfp4_to_fp32(packed, scales)
    torch.testing.assert_close(decoded, inp['x'], atol=0, rtol=0)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(h.convert_mxfp4_to_fp32(torch.zeros_like(packed), scales),
                                   inp['x'], atol=0.1, rtol=0.1)


@pytest.mark.parametrize('mode', ['correct', 'wrong_data', 'wrong_scales', 'missing',
                                 'dtype', 'shape', 'mutate_source'])
def test_quant_sort_correctness_requires_both_outputs_and_pristine_inputs(monkeypatch, mode):
    task = ROOT/'tasks/triton2triton/geak_eval/L3/fused_mxfp4_quant_moe_sort'
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    h, inp = _quant_sort_cpu_harness()
    original = h.fused_dynamic_mxfp4_quant_moe_sort
    def candidate(*args, **kwargs):
        if mode == 'mutate_source': args[0].zero_()
        packed, scales = original(*args, **kwargs)
        if mode == 'wrong_data': packed.zero_()
        if mode == 'wrong_scales': scales.zero_()
        if mode == 'missing': return (packed,)
        if mode == 'dtype': packed = packed.float()
        if mode == 'shape': scales = scales[:1]
        return packed, scales
    h.fused_dynamic_mxfp4_quant_moe_sort = candidate
    with checks.checked_correctness(h):
        def run():
            return h.fused_dynamic_mxfp4_quant_moe_sort(inp['x'], inp['sorted_ids'],
                inp['num_valid_ids'], inp['token_num'], inp['topk'], inp['block_size_M'])
        if mode == 'correct': run()
        else:
            with pytest.raises(AssertionError): run()
    assert h.fused_dynamic_mxfp4_quant_moe_sort is candidate


@pytest.mark.parametrize('mode', ['correct', 'wrong_timed', 'stale', 'no_write',
                                 'wrong_replay', 'mutate_timed', 'mutate_replay', 'raise_replay'])
def test_quant_sort_actual_benchmark_captures_both_outputs_and_restores_inputs(monkeypatch, mode):
    import sys
    task = ROOT/'tasks/triton2triton/geak_eval/L3/fused_mxfp4_quant_moe_sort'
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    monkeypatch.setitem(sys.modules, '_aka_benchmark', SimpleNamespace(TimedRun=SimpleNamespace))
    h, inp = _quant_sort_cpu_harness()
    original = h.fused_dynamic_mxfp4_quant_moe_sort
    pristine = checks.snapshots(inp)
    observed = []
    def benchmark(fn, *, timed_run, **kwargs):
        observed.append(kwargs)
        checks.unchanged(inp, pristine)
        outputs = fn()
        cached = [value.clone() for value in outputs]
        if mode == 'wrong_timed': outputs[0].zero_()
        if mode == 'mutate_timed': inp['x'].zero_()
        def replay():
            if mode == 'raise_replay': raise RuntimeError('replay exploded')
            new = fn()
            if mode == 'mutate_replay': inp['sorted_ids'].zero_()
            if mode == 'stale': new = cached
            if mode == 'no_write': return outputs
            if mode == 'wrong_replay': new[1].zero_()
            for output, value in zip(outputs, new): output.copy_(value)
            return outputs
        timed_run.outputs, timed_run.rerun = outputs, replay
        return 0.125, {'benchmark_method': 'cuda_graph'}
    h.benchmark_cuda_graph_or_events = lambda fn, **kwargs: checks.checked_benchmark(h, benchmark, fn, **kwargs)
    if mode == 'correct':
        assert h.run_benchmark([0]) == pytest.approx(0.125)
    else:
        with pytest.raises((AssertionError, RuntimeError)): h.run_benchmark([0])
    assert observed == [dict(warmup=50, repetition=200)]
    checks.unchanged(inp, pristine)
    assert h.fused_dynamic_mxfp4_quant_moe_sort is original


def test_quant_sort_action_adapter_retains_replay_metadata(monkeypatch):
    import sys
    task = ROOT/'tasks/triton2triton/geak_eval/L3/fused_mxfp4_quant_moe_sort'
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    h, inp = _quant_sort_cpu_harness()
    def benchmark(fn, *, timed_run, **kwargs):
        timed_run.outputs, timed_run.rerun = fn(), fn
        return 0.25, {'benchmark_method': 'cuda_graph'}
    h.benchmark_cuda_graph_or_events = benchmark
    monkeypatch.setitem(sys.modules, '_aka_benchmark', SimpleNamespace(TimedRun=SimpleNamespace))
    monkeypatch.setitem(sys.modules, 'test_kernel_harness', h)
    monkeypatch.setitem(sys.modules, '_arena_checks', checks)
    actions = module_at(task/'_arena_actions.py', monkeypatch)
    adapter = module_at(task/'_arena_eval.py', monkeypatch)
    measurements = adapter.capture_performance(actions, {'cases': [{'checks': ['performance']}],
                                                        'timing_calls_per_case': ['candidate']})
    ms, metadata = measurements[0]
    assert ms == 0.25
    assert metadata['timed_output_checked'] and metadata['perturbed_input_replay_checked']
    assert metadata['source_buffers_unchanged']
    assert h.benchmark_cuda_graph_or_events is benchmark


def _fp4_gemm_cpu_harness():
    from types import ModuleType
    path = ROOT/'tasks/triton2triton/geak_eval/L3/gemm_a16wfp4/test_kernel_harness.py'
    decoder = pure_functions(path, ['mxfp4_to_f32'], {'torch': SimpleNamespace(
        float32=torch.float32, tensor=lambda values, **kwargs: torch.tensor(
            values, **{**kwargs, 'device': 'cpu'}))})
    ref = pure_functions(path, ['e8m0_to_f32', 'run_torch_reference'],
        {'SCALE_GROUP_SIZE': 32, 'mxfp4_to_f32': decoder.mxfp4_to_f32})
    h = ModuleType('_fp4_gemm_cpu')
    h.run_torch_reference = ref.run_torch_reference
    h.DTYPE = torch.bfloat16
    h.ATOL = h.RTOL = 1e-2
    h.ALL_SHAPES = h.HARNESS_SHAPES = [(1, 2, 32)]
    h.is_fp4_avail = lambda: True
    h._label = str
    h._shape_indices = lambda shapes: [0]
    h.math = __import__('math')
    h.torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None, empty_cache=lambda: None),
                              testing=torch.testing)
    x = torch.arange(1, 33, dtype=h.DTYPE).reshape(1, 32)
    w = torch.tensor([[0x22]*16, [0xaa]*16], dtype=torch.uint8)
    scales = torch.tensor([[127], [128]], dtype=torch.uint8)
    h.generate_inputs = lambda *args: (x, w, w, scales, scales)
    h.gemm_a16wfp4 = lambda x, w, scales, **kw: ref.run_torch_reference(x, w, scales, kw['dtype'])
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)
             and n.name in ('run_correctness', 'run_benchmark')]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), h.__dict__)
    return h, (x, w, scales)


def test_fp4_gemm_original_reference_known_answer_and_postcall_mutation_control(monkeypatch):
    task = ROOT/'tasks/triton2triton/geak_eval/L3/gemm_a16wfp4'
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    h, inputs = _fp4_gemm_cpu_harness()
    expected = torch.tensor([[528., -1056.]], dtype=torch.bfloat16)
    torch.testing.assert_close(h.run_torch_reference(*inputs, h.DTYPE), expected, atol=0, rtol=0)
    original = h.gemm_a16wfp4
    def corrupt(x, w, scales, **kwargs):
        x.zero_(); w.zero_()
        return original(x, w, scales, **kwargs)
    h.gemm_a16wfp4 = corrupt
    # Execute the real old orchestration, which incorrectly accepts corruption.
    assert h.run_correctness()['correct'] is True
    h, inputs = _fp4_gemm_cpu_harness()
    h.gemm_a16wfp4 = corrupt
    with checks.checked_correctness(h):
        result = h.run_correctness()
    assert result['correct'] is False and result['num_failed'] == 1
    assert 'read-only inputs' in result['failures'][0]['error']


@pytest.mark.parametrize('mode', ['correct', 'wrong_value', 'dtype', 'shape', 'nonfinite'])
def test_fp4_gemm_correctness_retains_output_contract(monkeypatch, mode):
    checks = module_at(ROOT/'tasks/triton2triton/geak_eval/L3/gemm_a16wfp4/_arena_checks.py', monkeypatch)
    h, inputs = _fp4_gemm_cpu_harness()
    original = h.gemm_a16wfp4
    def candidate(*args, **kwargs):
        output = original(*args, **kwargs)
        if mode == 'wrong_value': output.zero_()
        if mode == 'dtype': output = output.float()
        if mode == 'shape': output = output[:, :1]
        if mode == 'nonfinite': output.fill_(float('nan'))
        return output
    h.gemm_a16wfp4 = candidate
    with checks.checked_correctness(h): result = h.run_correctness()
    assert result['correct'] is (mode == 'correct')
    assert h.gemm_a16wfp4 is candidate


@pytest.mark.parametrize('mode', ['correct', 'wrong_timed', 'stale', 'no_write', 'wrong_replay',
                                 'mutate_timed', 'mutate_replay', 'raise_replay'])
def test_fp4_gemm_original_benchmark_replay_and_source_restoration(monkeypatch, mode):
    import sys
    checks = module_at(ROOT/'tasks/triton2triton/geak_eval/L3/gemm_a16wfp4/_arena_checks.py', monkeypatch)
    monkeypatch.setitem(sys.modules, '_aka_benchmark', SimpleNamespace(TimedRun=SimpleNamespace))
    h, inputs = _fp4_gemm_cpu_harness()
    pristine = checks.snapshots(inputs)
    options = []
    def benchmark(fn, *, timed_run, **kwargs):
        options.append(kwargs)
        checks.unchanged(inputs, pristine)
        output = fn(); cached = output.clone()
        if mode == 'wrong_timed': output.zero_()
        if mode == 'mutate_timed': inputs[0].zero_()
        def replay():
            if mode == 'raise_replay': raise RuntimeError('replay failed')
            if mode == 'stale': output.copy_(cached)
            elif mode != 'no_write': output.copy_(fn())
            if mode == 'wrong_replay': output.zero_()
            if mode == 'mutate_replay': inputs[1].zero_()
            return output
        timed_run.outputs, timed_run.rerun = output, replay
        return 0.125, {'benchmark_method': 'cuda_graph'}
    h.benchmark_cuda_graph_or_events = lambda fn, **kwargs: checks.checked_benchmark(h, benchmark, fn, **kwargs)
    if mode == 'correct': h.run_benchmark()
    else:
        with pytest.raises((AssertionError, RuntimeError)): h.run_benchmark()
    assert options == [dict(warmup=50, repetition=200)]
    checks.unchanged(inputs, pristine)


def test_geak_boolean_integer_and_skipped_result_contracts(monkeypatch):
    adapter=module_at(GEAK[0].parent/'_arena_eval.py',monkeypatch)
    adapter.require_success(None,'none',1)
    adapter.require_success(0,'zero',1)
    adapter.require_success(True,'bool',1)
    for value,kind in [(None,'bool'),(False,'zero'),(True,'zero'),(0,'bool'),
                       ({'correct':True,'num_correct':0,'num_failed':0,'skipped':True},'dict'),
                       ({'correct':True,'num_correct':1,'num_failed':0},'dict')]:
        with pytest.raises(RuntimeError):adapter.require_success(value,kind,2)


def test_geak_performance_uses_current_candidate_timing_and_rejects_missing_calls(monkeypatch):
    adapter=module_at(GEAK[0].parent/'_arena_eval.py',monkeypatch)
    measurements=iter([17.,3.,19.,5.])
    h=SimpleNamespace(benchmark_cuda_graph_or_events=lambda *a,**k:(next(measurements),{'benchmark_method':'cuda_graph'}))
    actions=SimpleNamespace(h=h,performance=lambda:[h.benchmark_cuda_graph_or_events(None) for _ in range(4)])
    data={'cases':[{'checks':['correctness','performance']},{'checks':['correctness','performance']}],
          'timing_calls_per_case':['reference','candidate']}
    assert [ms for ms,meta in adapter.capture_performance(actions,data)]==[3.,5.]
    # A stale saved report has no role in supplying the missing current call.
    h.benchmark_cuda_graph_or_events=lambda *a,**k:(2.,{'benchmark_method':'cuda_graph'})
    actions.performance=lambda:h.benchmark_cuda_graph_or_events(None)
    with pytest.raises(RuntimeError,match='Missing/extra'):adapter.capture_performance(actions,data)


def test_geak_failure_envelope_and_manifest_drift(monkeypatch):
    adapter=module_at(GEAK[0].parent/'_arena_eval.py',monkeypatch)
    data=json.loads((GEAK[0].parent/'workloads.json').read_text())
    actions=SimpleNamespace(inputs=lambda:data['input_tables'],validate=lambda:None,
                            correctness=lambda require:require(False,'bool',len(data['cases'])))
    monkeypatch.setattr(adapter,'load_actions',lambda:actions)
    result=adapter.evaluate('candidate','correctness')
    assert result['status']=='FAIL' and all(row['status']=='FAIL' for row in result['cases'])
    assert result['failure_kind']!='numerical_mismatch'
    result_record(result)
    actions.inputs=lambda:{'performance':[],'original_correctness':[]}
    result=adapter.evaluate('task','validate-task')
    assert result['status']=='FAIL' and 'manifest' in result['reason']


def pure_functions(path, names, namespace=None):
    """Load protected CPU reference functions without importing GPU dependencies."""
    tree=ast.parse(path.read_text())
    selected=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    assert {n.name for n in selected}==set(names)
    scope={'torch':torch,**(namespace or {})}
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),scope)
    return SimpleNamespace(**scope)


def test_geak_rope_references_have_independent_known_answers():
    path=ROOT/'tasks/triton2triton/geak_eval/L3/fused_qk_rope_cache_mla/test_kernel_harness.py'
    from enum import IntEnum
    ref=pure_functions(path,['RotateStyle','rotate_half_neox','rotate_half_gptj','ref_rope_sbhd_fwd'],{'IntEnum':IntEnum})
    x=torch.tensor([[1.,2.,3.,4.]])
    torch.testing.assert_close(ref.rotate_half_neox(x),torch.tensor([[-3.,-4.,1.,2.]]))
    torch.testing.assert_close(ref.rotate_half_gptj(x),torch.tensor([[-2.,1.,-4.,3.]]))
    got=ref.ref_rope_sbhd_fwd(x,torch.full((1,4),torch.pi/2),ref.RotateStyle.GPTJ,False,False)
    expected=torch.tensor([[-2.,1.,-4.,3.]])
    torch.testing.assert_close(got,expected)
    with pytest.raises(AssertionError):torch.testing.assert_close(got,x)


def test_geak_fp4_reference_decodes_independent_known_values():
    path=ROOT/'tasks/triton2triton/geak_eval/L3/gemm_a16wfp4/test_kernel_harness.py'
    # Only the lookup-table allocation is redirected to CPU; arithmetic is unchanged.
    cpu_torch=SimpleNamespace(float32=torch.float32,
        tensor=lambda values,**kwargs:torch.tensor(values,**{**kwargs,'device':'cpu'}))
    ref=pure_functions(path,['mxfp4_to_f32'],{'torch':cpu_torch})
    # Low/high nibbles enumerate the E2M1 positive and negative values.
    packed=torch.tensor([[0x10,0x32,0x54,0x76,0x98,0xba,0xdc,0xfe]],dtype=torch.uint8)
    expected=torch.tensor([[0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.]])
    got=ref.mxfp4_to_f32(packed)
    torch.testing.assert_close(got,expected)
    with pytest.raises(AssertionError):torch.testing.assert_close(got,torch.zeros_like(expected))


@pytest.mark.parametrize('name',['gemm_a16wfp4','fused_mxfp4_quant_moe_sort'])
def test_e8m0_reference_all_encodings_and_negative_control(name):
    path=ROOT/'tasks/triton2triton/geak_eval/L3'/name/'test_kernel_harness.py'
    ref=pure_functions(path,['e8m0_to_f32'])
    codes=torch.arange(256,dtype=torch.uint8)
    got=ref.e8m0_to_f32(codes)
    expected=codes.view(torch.float8_e8m0fnu).float()
    torch.testing.assert_close(got,expected,atol=0,rtol=0,equal_nan=True)
    assert got[126]==0.5 and got[134]==128 and torch.isnan(got[255])
    before=subprocess.check_output(['git','show',f'{BASE}:{path.relative_to(ROOT).as_posix()}'],cwd=ROOT,text=True)
    node=next(n for n in ast.parse(before).body if isinstance(n,ast.FunctionDef) and n.name=='e8m0_to_f32')
    scope={'torch':torch};exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),scope)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(scope['e8m0_to_f32'](codes),expected,atol=0,rtol=0,equal_nan=True)


@pytest.mark.parametrize('task',[VLLM[0].parent,ROCM[0].parent,GEAK[0].parent],ids=['vllm','rocm','geak'])
def test_bad_diagnostic_evidence_still_emits_failure(task,monkeypatch,capsys):
    adapter=module_at(task/'_arena_eval.py',monkeypatch)
    code=adapter.emit_result({'protocol':'arena-eval-v1','role':'candidate','action':'performance',
                             'status':'PASS','cases':[],'metadata':{'bad':float('nan')}})
    parsed=parse_command_result(capsys.readouterr().out,role='candidate',action='performance',returncode=code)
    assert parsed.status=='FAIL' and parsed.failure_kind=='invalid_evidence'


def test_vllm_additional_original_correctness_cases_are_manifested():
    for name in ['triton_bad_words','triton_logit_bias']:
        task=ROOT/'tasks/triton2triton/vllm'/name
        data=json.loads((task/'workloads.json').read_text())
        correctness_only=[r for r in data['cases'] if r['checks']==['correctness']]
        # Newly added controls supplement, rather than replace, the original
        # targeted multi-token/allowlist cases dispatched with index -1.
        extra=[r for r in correctness_only if r['params']['case_index']==-1]
        controls=[r for r in correctness_only if r['params']['case_index']!=-1]
        assert len(controls)==1 and controls[0]['test_case_id']=='contract_controls'
        assert controls[0]['params']['case_index']==5
        before=subprocess.check_output(['git','show',f'{BASE}:{task.relative_to(ROOT).as_posix()}/scripts/task_runner.py'],cwd=ROOT,text=True)
        tree=ast.parse(before)
        if name=='triton_bad_words':
            assignment=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='multi_cases' for t in n.targets))
            assert [r['shape'] for r in extra]==[list(v) for v in ast.literal_eval(assignment.value)]
            assert [r['params']['seed'] for r in extra]==[1234,1235,1236]
        else:
            assert len(extra)==1 and extra[0]['shape']==[2,256,16] and extra[0]['params']['seed']==777
        assert all(r['params']['case_index']==-1 for r in extra)


def test_all_rocm_manifests_match_original_cpu_collected_pytest_parameters():
    """Import decorators with inert GPU stubs; execute no kernels or test bodies."""
    import sys
    fixture=ROOT/'tests/fixtures/triton_migration/collect_rocm_cpu.py'
    run=subprocess.run([sys.executable,str(fixture)],cwd=ROOT,text=True,capture_output=True,check=True)
    collected=json.loads(run.stdout)
    assert len(collected)==61
    for task,functions in collected.items():
        expected={}
        for fn in functions:
            for params in fn['cases']:
                content=json.dumps(params,sort_keys=True,separators=(',',':'))
                case=fn['function']+'/'+hashlib.sha256(content.encode()).hexdigest()[:20]
                expected[case]={'function':fn['function'],'arguments':params}
        data=json.loads((ROOT/task/'workloads.json').read_text())
        assert {r['test_case_id']:r['params'] for r in data['cases']}==expected,task
        assert 'built-in method' not in json.dumps(data)


DECORATOR_HELPERS={
    'gemm':{'leaky_relu'},
    'layernorm':{'get_autotune_config'},
    'rmsnorm_fwd':{'get_autotune_config'},
    'softmax':{'get_autotune_config'},
    'multreduce_matmul_dot_kernel':{'get_triton_dot_autotune_configs','get_triton_autotune_key','get_triton_heuristics','triton_matmul_kernel'},
    # triton_matmul_kernel is the actual fixed-launch performance entrypoint.
    'triton_multreduce_matmul_kernel':{'get_triton_multreduce_autotune_configs','get_triton_autotune_key','get_triton_heuristics'},
}


@pytest.mark.parametrize('path',[p for p in ROCM if p.parent.name in DECORATOR_HELPERS],ids=lambda p:p.parent.relative_to(ROOT/'tasks').as_posix())
def test_rocm_implementation_helpers_editable_without_exposing_tests_or_references(path,tmp_path):
    import shutil
    from src.harness_guard import snapshot_workspace_harness,verify_workspace_harness
    workspace=tmp_path/'task'
    shutil.copytree(path.parent,workspace,ignore=shutil.ignore_patterns('__pycache__'))
    spec=load_task_spec(workspace/'config.yaml',task_id=path.parent.name)
    helper_names=DECORATOR_HELPERS[path.parent.name]
    entry_names={e.symbol for e in spec.candidate.entrypoints}
    assert not (helper_names & entry_names), 'Implementation helpers must not become required entrypoints'
    assert helper_names <= set(spec.candidate.editable[0].symbols)
    assert set(spec.candidate.editable[0].symbols)==entry_names|helper_names
    snapshot=snapshot_workspace_harness(workspace)
    source=workspace/spec.candidate.editable[0].path
    text=source.read_text();tree=ast.parse(text);lines=text.splitlines(keepends=True)
    insertions=[n.body[0].lineno-1 for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in helper_names]
    for index in sorted(insertions,reverse=True):lines.insert(index,'    _arena_helper_boundary_probe = 1\n')
    source.write_text(''.join(lines))
    verify_workspace_harness(snapshot)
    # Ordinary test/reference code remains protected in that same file.
    tree=ast.parse(source.read_text())
    test=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name.startswith('test_'))
    lines=source.read_text().splitlines(keepends=True)
    lines.insert(test.body[0].lineno-1,'    _arena_forbidden_test_change = 1\n')
    source.write_text(''.join(lines))
    with pytest.raises(RuntimeError):verify_workspace_harness(snapshot)


def conv_fwd_inputs():
    x=torch.tensor([[1.,2.,3.,4.,5.,6.],[7.,8.,9.,10.,11.,12.]])
    weight=torch.tensor([[1.,2.,3.],[1.,-1.,2.]])
    bias=torch.tensor([1.,-1.])
    state=torch.tensor([[[10.,20.],[30.,40.]],[[50.,60.],[70.,80.]]])
    starts=torch.tensor([0,3,6],dtype=torch.int32)
    indices=torch.tensor([0,1],dtype=torch.int32)
    has_init=torch.ones(2,dtype=torch.int32)
    return x,weight,bias,state,starts,indices,has_init


def independent_conv_fwd(x,weight,bias,state,starts,indices,has_init,activation=None):
    # An independent grouped-convolution implementation for the CPU controls.
    result=torch.empty_like(x)
    for i,slot in enumerate(indices.tolist()):
        start,end=starts[i:i+2].tolist()
        prefix=state[slot].clone() if has_init[i] else torch.zeros_like(state[slot])
        sequence=torch.cat((prefix,x[:,start:end]),dim=1)
        values=torch.nn.functional.conv1d(sequence.unsqueeze(0),weight.unsqueeze(1),bias,groups=x.shape[0])[0]
        if activation in ('silu','swish'):values=torch.nn.functional.silu(values)
        result[:,start:end]=values
        state[slot].copy_(sequence[:,-state.shape[-1]:])
    return result


def test_conv_fwd_original_reference_and_state_have_independent_known_answers(monkeypatch):
    task=ROOT/'tasks/triton2triton/vllm/triton_causal_conv1d_fwd'
    harness=module_at(task/'scripts/task_runner.py',monkeypatch)
    checks=module_at(task/'_arena_checks.py',monkeypatch)
    args=conv_fwd_inputs()
    expected=torch.tensor([[54.,29.,15.,183.,84.,33.],[3.,48.,16.,9.,91.,22.]])
    state_expected=torch.tensor([[[2.,3.],[8.,9.]],[[5.,6.],[11.,12.]]])
    actual=checks.references(harness,*args,None)
    assert torch.equal(actual[0],expected) and torch.equal(actual[1],state_expected)
    output=independent_conv_fwd(*args)
    assert torch.equal(output,expected) and torch.equal(args[3],state_expected)


@pytest.mark.parametrize('mode',['correct','wrong_state','no_state_write','wrong_output','dtype','shape','nonfinite','mutate_input'])
def test_conv_fwd_correctness_checks_output_state_and_pristine_input_oracles(monkeypatch,mode):
    task=ROOT/'tasks/triton2triton/vllm/triton_causal_conv1d_fwd'
    harness=module_at(task/'scripts/task_runner.py',monkeypatch)
    checks=module_at(task/'_arena_checks.py',monkeypatch)
    args=conv_fwd_inputs()
    def candidate(x,w,b,state,starts,indices,has_init,activation=None):
        saved=state.clone()
        out=independent_conv_fwd(x,w,b,state,starts,indices,has_init,activation)
        if mode=='wrong_state':state[1,0,0]=0
        elif mode=='no_state_write':state.copy_(saved)
        elif mode=='wrong_output':out.zero_()
        elif mode=='dtype':out=out.double()
        elif mode=='shape':out=out[:1]
        elif mode=='nonfinite':state[0,0,0]=float('nan')
        elif mode=='mutate_input':x.zero_();out.zero_();state.zero_()
        return out
    module=SimpleNamespace(causal_conv1d_fwd=candidate)
    original_load=lambda:module
    harness.load_module=original_load
    with checks.checked_modules(harness):
        if mode=='correct':harness.load_module().causal_conv1d_fwd(*args)
        else:
            with pytest.raises(AssertionError):harness.load_module().causal_conv1d_fwd(*args)
    assert harness.load_module is original_load and module.causal_conv1d_fwd is candidate


@pytest.mark.parametrize('mode',['correct','wrong_timed_state','stale','no_write','wrong_replay_state','wrong_replay_output','mutate_input','replay_raises'])
def test_conv_fwd_actual_timed_output_state_and_replay_restore(monkeypatch,mode):
    task=ROOT/'tasks/triton2triton/vllm/triton_causal_conv1d_fwd'
    harness=module_at(task/'scripts/task_runner.py',monkeypatch)
    checks=module_at(task/'_arena_checks.py',monkeypatch)
    harness._TimedRun=SimpleNamespace
    x,weight,bias_t,conv_states,query_start_loc,cache_indices,has_init=conv_fwd_inputs()
    out=torch.empty_like(x);activation='silu'
    initial_conv_states=conv_states.clone()
    batch_ptr=torch.tensor([0,1],dtype=torch.int32)
    token_chunk_offset_ptr=torch.zeros(2,dtype=torch.int32)
    source=(x,weight,bias_t,conv_states,query_start_loc,cache_indices,has_init,initial_conv_states,batch_ptr,token_chunk_offset_ptr)
    pristine=tuple(value.clone() for value in source)
    def fn():
        assert batch_ptr.numel()==token_chunk_offset_ptr.numel()==2
        out.copy_(independent_conv_fwd(x,weight,bias_t,conv_states,query_start_loc,cache_indices,has_init,activation))
    def prepare():conv_states.copy_(initial_conv_states)
    options=[]
    def benchmark(measured,*,timed_run,**kwargs):
        options.append(kwargs)
        kwargs['prepare_fn']();outputs=measured();saved=tuple(value.clone() for value in outputs)
        if mode=='wrong_timed_state':conv_states.zero_()
        def replay():
            kwargs['prepare_fn']()
            if mode=='replay_raises':raise RuntimeError('injected replay failure')
            if mode=='stale':
                for value,cached in zip(outputs,saved):value.copy_(cached)
            elif mode!='no_write':
                measured()
                if mode=='wrong_replay_state':conv_states.zero_()
                elif mode=='wrong_replay_output':out.zero_()
                elif mode=='mutate_input':x.zero_();out.zero_();conv_states.zero_()
            return outputs
        timed_run.outputs=outputs;timed_run.rerun=replay
        return 0.25,dict(benchmark_method='cuda_graph')
    call=lambda:checks.checked_benchmark(harness,benchmark,fn,warmup=10,repetition=100,prepare_fn=prepare)
    if mode=='correct':
        ms,metadata=call()
        assert ms==0.25 and metadata['cached_state_checked'] and metadata['perturbed_input_replay_checked']
    elif mode=='replay_raises':
        with pytest.raises(RuntimeError,match='injected replay failure'):call()
    else:
        with pytest.raises(AssertionError):call()
    assert options==[dict(warmup=10,repetition=100,prepare_fn=prepare)]
    assert all(torch.equal(value,saved) for value,saved in zip(source,pristine))


def test_conv_fwd_adapter_installs_output_state_and_timing_checks(monkeypatch):
    adapter=module_at(ROOT/'tasks/triton2triton/vllm/triton_causal_conv1d_fwd/_arena_eval.py',monkeypatch)
    harness=adapter.load_harness()
    assert harness.run_correctness.__module__==harness.run_performance.__module__=='_conv_fwd_checks'


def conv_update_inputs():
    x=torch.tensor([[[1.],[7.]],[[4.],[10.]]])
    state=torch.tensor([[[10.,20.],[30.,40.]],[[50.,60.],[70.,80.]]])
    weight=torch.tensor([[1.,2.,3.],[1.,-1.,2.]])
    bias=torch.tensor([1.,-1.]);indices=torch.tensor([0,1],dtype=torch.int32)
    return x,state,weight,bias,indices


def independent_conv_update(x,state,weight,bias=None,activation=None,conv_state_indices=None):
    for sequence,slot in enumerate(conv_state_indices.tolist()):
        history=torch.cat((state[slot].clone(),x[sequence].clone()),dim=-1)
        value=torch.nn.functional.conv1d(history.unsqueeze(0),weight.unsqueeze(1),bias,groups=x.shape[1])[0]
        if activation in ('silu','swish'):value=torch.nn.functional.silu(value)
        x[sequence].copy_(value)
        state[slot].copy_(history[:,-state.shape[-1]:])
    return x


def test_conv_update_original_reference_and_shifted_state_known_answers(monkeypatch):
    task=ROOT/'tasks/triton2triton/vllm/triton_causal_conv1d_update'
    harness=module_at(task/'scripts/task_runner.py',monkeypatch)
    checks=module_at(task/'_arena_checks.py',monkeypatch)
    x,state,weight,bias,indices=conv_update_inputs()
    expected=torch.tensor([[[54.],[3.]],[[183.],[9.]]])
    state_expected=torch.tensor([[[20.,1.],[40.,7.]],[[60.,4.],[80.,10.]]])
    result=checks.references(harness,x,state,weight,bias,None,indices)
    assert torch.equal(result[0],expected) and torch.equal(result[1],state_expected)
    output=independent_conv_update(x,state,weight,bias=bias,conv_state_indices=indices)
    assert torch.equal(output,expected) and torch.equal(state,state_expected)


@pytest.mark.parametrize('mode',['correct','no_state_write','wrong_state','wrong_output','out_of_place','dtype','nonfinite','mutate_weight'])
def test_conv_update_correctness_covers_state_and_in_place_output(monkeypatch,mode):
    task=ROOT/'tasks/triton2triton/vllm/triton_causal_conv1d_update'
    harness=module_at(task/'scripts/task_runner.py',monkeypatch)
    checks=module_at(task/'_arena_checks.py',monkeypatch)
    x,state,weight,bias,indices=conv_update_inputs()
    def candidate(x,state,weight,**kwargs):
        saved=state.clone()
        result=independent_conv_update(x,state,weight,**kwargs)
        if mode=='no_state_write':state.copy_(saved)
        elif mode=='wrong_state':state[1,0,0]=0
        elif mode=='wrong_output':result.zero_()
        elif mode=='out_of_place':result=result.clone()
        elif mode=='dtype':result=result.double()
        elif mode=='nonfinite':state[0,0,0]=float('nan')
        elif mode=='mutate_weight':weight.zero_();result.zero_();state.zero_()
        return result
    module=SimpleNamespace(causal_conv1d_update=candidate)
    original_load=lambda:module
    harness.load_module=original_load
    with checks.checked_modules(harness):
        call=lambda:harness.load_module().causal_conv1d_update(x,state,weight,bias=bias,conv_state_indices=indices)
        if mode=='correct':call()
        else:
            with pytest.raises(AssertionError):call()
    assert harness.load_module is original_load and module.causal_conv1d_update is candidate


@pytest.mark.parametrize('mode',['correct','wrong_timed_state','stale','no_write','wrong_replay_state','wrong_replay_output','mutate_weight','replay_raises'])
def test_conv_update_actual_timed_buffers_and_replay_restore(monkeypatch,mode):
    task=ROOT/'tasks/triton2triton/vllm/triton_causal_conv1d_update'
    harness=module_at(task/'scripts/task_runner.py',monkeypatch)
    checks=module_at(task/'_arena_checks.py',monkeypatch)
    harness._TimedRun=SimpleNamespace
    x,conv_state,weight,bias_t,conv_state_indices=conv_update_inputs()
    x_work=x.clone();conv_state_work=conv_state.clone();activation='silu'
    inputs=(x,conv_state,weight,bias_t,conv_state_indices,x_work,conv_state_work)
    saved=tuple(value.clone() for value in inputs)
    def fn():
        return independent_conv_update(x_work,conv_state_work,weight,bias=bias_t,
                                       activation=activation,conv_state_indices=conv_state_indices)
    def prepare():x_work.copy_(x);conv_state_work.copy_(conv_state)
    options=[]
    def benchmark(measured,*,timed_run,**kwargs):
        options.append(kwargs);kwargs['prepare_fn']()
        outputs=measured();cached=tuple(value.clone() for value in outputs)
        if mode=='wrong_timed_state':conv_state_work.zero_()
        def replay():
            kwargs['prepare_fn']()
            if mode=='replay_raises':raise RuntimeError('injected replay failure')
            if mode=='stale':
                for value,original in zip(outputs,cached):value.copy_(original)
            elif mode!='no_write':
                measured()
                if mode=='wrong_replay_state':conv_state_work.zero_()
                elif mode=='wrong_replay_output':x_work.zero_()
                elif mode=='mutate_weight':weight.zero_();x_work.zero_();conv_state_work.zero_()
            return outputs
        timed_run.outputs=outputs;timed_run.rerun=replay
        return 0.25,dict(benchmark_method='cuda_graph')
    call=lambda:checks.checked_benchmark(harness,benchmark,fn,warmup=10,repetition=100,target_ms=20.0,prepare_fn=prepare)
    if mode=='correct':
        ms,metadata=call()
        assert ms==0.25 and metadata['cached_state_checked'] and metadata['perturbed_input_replay_checked']
    elif mode=='replay_raises':
        with pytest.raises(RuntimeError,match='injected replay failure'):call()
    else:
        with pytest.raises(AssertionError):call()
    assert options==[dict(warmup=10,repetition=100,target_ms=20.0,prepare_fn=prepare)]
    assert all(torch.equal(value,original) for value,original in zip(inputs,saved))


def test_conv_update_adapter_installs_output_state_and_timing_checks(monkeypatch):
    adapter=module_at(ROOT/'tasks/triton2triton/vllm/triton_causal_conv1d_update/_arena_eval.py',monkeypatch)
    harness=adapter.load_harness()
    assert harness.run_correctness.__module__==harness.run_performance.__module__=='_conv_update_checks'


def index_conversion_inputs():
    req=torch.tensor([1,0],dtype=torch.int32)
    table=torch.tensor([[10,20],[30,40]],dtype=torch.int32)
    tokens=torch.tensor([[0,3,4,7],[1,-1,8,12]],dtype=torch.int32)
    return req,table,tokens


def independent_index_conversion(req,table,tokens,BLOCK_SIZE=4,BLOCK_N=4,return_valid_counts=False):
    block=torch.div(tokens,BLOCK_SIZE,rounding_mode='floor')
    valid=(tokens>=0)&(block<table.shape[1])
    bases=table[req.long().unsqueeze(1),block.clamp(0,table.shape[1]-1).long()]
    output=torch.where(valid,bases*BLOCK_SIZE+tokens.remainder(BLOCK_SIZE),-1).to(torch.int32)
    return (output,valid.sum(1).to(torch.int32)) if return_valid_counts else output


def test_index_conversion_reference_counts_and_oob_known_answers(monkeypatch):
    task=ROOT/'tasks/triton2triton/vllm/triton_convert_req_to_global_index'
    harness=module_at(task/'scripts/task_runner.py',monkeypatch)
    checks=module_at(task/'_arena_checks.py',monkeypatch)
    args=index_conversion_inputs()
    expected=torch.tensor([[120,123,160,163],[41,-1,-1,-1]],dtype=torch.int32)
    assert torch.equal(checks.reference(harness,args,4),expected)
    result=independent_index_conversion(*args,return_valid_counts=True)
    assert torch.equal(result[0],expected) and torch.equal(result[1],torch.tensor([4,1],dtype=torch.int32))


@pytest.mark.parametrize('mode',['correct','wrong_counted_output','wrong_counts','dtype','shape','ignore_oob','mutate_inputs'])
@pytest.mark.parametrize('counted',[False,True])
def test_index_conversion_correctness_checks_both_results_and_oob(monkeypatch,mode,counted):
    task=ROOT/'tasks/triton2triton/vllm/triton_convert_req_to_global_index'
    harness=module_at(task/'scripts/task_runner.py',monkeypatch)
    checks=module_at(task/'_arena_checks.py',monkeypatch)
    inputs=index_conversion_inputs()
    def candidate(req,table,tokens,**kwargs):
        result=independent_index_conversion(req,table,tokens,**kwargs)
        output=result[0] if isinstance(result,tuple) else result
        if mode=='wrong_counted_output' and counted:output.zero_()
        elif mode=='wrong_counts' and counted:result[1].zero_()
        elif mode=='dtype':
            output=output.long();result=(output,result[1]) if counted else output
        elif mode=='shape':
            output=output[:1];result=(output,result[1]) if counted else output
        elif mode=='ignore_oob':output[tokens>=table.shape[1]*kwargs['BLOCK_SIZE']]=0
        elif mode=='mutate_inputs':table.zero_();output.zero_()
        return result
    module=SimpleNamespace(convert_req_to_global_index=candidate)
    original_load=lambda:module;harness.load_module=original_load
    with checks.checked_modules(harness):
        call=lambda:harness.load_module().convert_req_to_global_index(*inputs,BLOCK_SIZE=4,BLOCK_N=4,return_valid_counts=counted)
        if mode=='correct' or not counted and mode in ('wrong_counted_output','wrong_counts'):call()
        else:
            with pytest.raises(AssertionError):call()
    assert harness.load_module is original_load and module.convert_req_to_global_index is candidate


@pytest.mark.parametrize('mode',['correct','wrong_timed','stale','no_write','wrong_replay','ignore_oob','mutate_inputs','replay_raises'])
def test_index_conversion_actual_captured_replay_and_pristine_restore(monkeypatch,mode):
    task=ROOT/'tasks/triton2triton/vllm/triton_convert_req_to_global_index'
    harness=module_at(task/'scripts/task_runner.py',monkeypatch)
    checks=module_at(task/'_arena_checks.py',monkeypatch);harness._TimedRun=SimpleNamespace
    req_id,block_table,token_indices=index_conversion_inputs();bs=4
    inputs=(req_id,block_table,token_indices);pristine=tuple(value.clone() for value in inputs)
    mod=SimpleNamespace(convert_req_to_global_index=independent_index_conversion)
    original=mod.convert_req_to_global_index
    def fn():mod.convert_req_to_global_index(req_id,block_table,token_indices,BLOCK_SIZE=bs,BLOCK_N=4)
    options=[]
    def benchmark(measured,*,timed_run,**kwargs):
        options.append(kwargs);output=measured();cached=output.clone()
        if mode=='wrong_timed':output.zero_()
        def replay():
            if mode=='replay_raises':raise RuntimeError('injected replay failure')
            if mode=='stale':output.copy_(cached)
            elif mode!='no_write':
                output.copy_(measured())
                if mode=='wrong_replay':output.zero_()
                elif mode=='ignore_oob':output[token_indices>=block_table.shape[1]*bs]=0
                elif mode=='mutate_inputs':block_table.zero_();output.zero_()
            return output
        timed_run.outputs=output;timed_run.rerun=replay
        return 0.25,dict(benchmark_method='cuda_graph')
    call=lambda:checks.checked_benchmark(harness,benchmark,fn,warmup=10,repetition=100)
    if mode=='correct':
        ms,metadata=call()
        assert ms==0.25 and metadata['out_of_bounds_checked'] and metadata['perturbed_input_replay_checked']
    elif mode=='replay_raises':
        with pytest.raises(RuntimeError,match='injected replay failure'):call()
    else:
        with pytest.raises(AssertionError):call()
    assert options==[dict(warmup=10,repetition=100)] and mod.convert_req_to_global_index is original
    assert all(torch.equal(value,saved) for value,saved in zip(inputs,pristine))


def test_index_conversion_adapter_installs_correctness_and_timing_checks(monkeypatch):
    adapter=module_at(ROOT/'tasks/triton2triton/vllm/triton_convert_req_to_global_index/_arena_eval.py',monkeypatch)
    harness=adapter.load_harness()
    assert harness.run_correctness.__module__==harness.run_performance.__module__=='_index_conversion_checks'


def identity_inputs():
    hidden=torch.tensor([[2.,-4.],[3.,5.]],dtype=torch.float16)
    scales=torch.tensor([[0.25,0.5],[0.5,0.75]],dtype=torch.float32)
    expected=torch.tensor([[1.5,-3.],[3.75,6.25]],dtype=torch.float16)
    return hidden,scales,expected


def test_identity_reference_independent_known_answer(monkeypatch):
    task=ROOT/'tasks/triton2triton/vllm/triton_compute_identity'
    checks=module_at(task/'_arena_checks.py',monkeypatch)
    harness=module_at(task/'scripts/task_runner.py',monkeypatch)
    hidden,scales,expected=identity_inputs()
    assert torch.equal(checks.reference(hidden,scales,2),expected)
    assert torch.equal(harness.reference_compute_identity(hidden,scales,2),expected)


@pytest.mark.parametrize('mode',['correct','shape','dtype','nonfinite','wrong_output','mutate_inputs'])
def test_identity_correctness_requires_output_contract_and_pristine_inputs(monkeypatch,mode):
    checks=module_at(ROOT/'tasks/triton2triton/vllm/triton_compute_identity/_arena_checks.py',monkeypatch)
    hidden,scales,expected=identity_inputs()
    def candidate(a,b,top_k):
        out=expected.clone()
        if mode=='shape':out=out[:1]
        elif mode=='dtype':out=out.float()
        elif mode=='nonfinite':out[0,0]=float('nan')
        elif mode=='wrong_output':out.zero_()
        elif mode=='mutate_inputs':a.zero_();b.zero_();out.zero_()
        return out
    mod=SimpleNamespace(compute_identity=candidate);load=lambda:mod
    harness=SimpleNamespace(load_module=load)
    with checks.checked_modules(harness):
        if mode=='correct':assert torch.equal(harness.load_module().compute_identity(hidden,scales,2),expected)
        else:
            with pytest.raises(AssertionError):harness.load_module().compute_identity(hidden,scales,2)
    assert harness.load_module is load and mod.compute_identity is candidate


@pytest.mark.parametrize('mode',['correct','wrong_timed','stale','no_write','wrong_replay','mutate_inputs','replay_raises'])
def test_identity_actual_timed_output_replay_and_input_restore(monkeypatch,mode):
    checks=module_at(ROOT/'tasks/triton2triton/vllm/triton_compute_identity/_arena_checks.py',monkeypatch)
    hidden_states,expert_scales,expected=identity_inputs();top_k=2
    original=lambda a,b,k:sum((a.float()*b[:,i:i+1] for i in range(k))).to(a.dtype)
    mod=SimpleNamespace(compute_identity=original)
    harness=SimpleNamespace(_TimedRun=SimpleNamespace)
    pristine=(hidden_states.clone(),expert_scales.clone())
    def fn():mod.compute_identity(hidden_states,expert_scales,top_k)
    options=[]
    def benchmark(measured,*,timed_run,**kwargs):
        options.append(kwargs);output=measured();cached=output.clone()
        assert torch.equal(output,expected)
        if mode=='wrong_timed':output.zero_()
        def replay():
            if mode=='replay_raises':raise RuntimeError('injected replay failure')
            if mode=='stale':output.copy_(cached)
            elif mode!='no_write':
                output.copy_(measured())
                if mode=='wrong_replay':output.fill_(123.)
                elif mode=='mutate_inputs':hidden_states.zero_();expert_scales.zero_();output.zero_()
            return output
        timed_run.outputs=output;timed_run.rerun=replay
        return 0.25,dict(benchmark_method='cuda_graph')
    call=lambda:checks.checked_benchmark(harness,benchmark,fn,warmup=10,repetition=100)
    if mode=='correct':
        ms,metadata=call();assert ms==0.25 and metadata['perturbed_input_replay_checked']
    elif mode=='replay_raises':
        with pytest.raises(RuntimeError,match='injected replay failure'):call()
    else:
        with pytest.raises(AssertionError):call()
    assert options==[dict(warmup=10,repetition=100)] and mod.compute_identity is original
    assert all(torch.equal(value,saved) for value,saved in zip((hidden_states,expert_scales),pristine))


def test_identity_adapter_installs_correctness_and_timing_checks(monkeypatch):
    adapter=module_at(ROOT/'tasks/triton2triton/vllm/triton_compute_identity/_arena_eval.py',monkeypatch)
    harness=adapter.load_harness()
    assert harness.run_correctness.__module__==harness.run_performance.__module__=='_identity_checks'


def eagle_inputs():
    return (torch.tensor([10,11,12,13],dtype=torch.int32),
            torch.tensor([100,101,102,103],dtype=torch.int32),
            torch.tensor([99],dtype=torch.int32),
            torch.tensor([0,4],dtype=torch.int32),torch.tensor([1],dtype=torch.int32))


@pytest.mark.parametrize('shift',[False,True])
def test_eagle_scalar_reference_has_six_independent_known_outputs(monkeypatch,shift):
    task=ROOT/'tasks/triton2triton/vllm/triton_copy_and_expand_eagle_inputs'
    harness=module_at(task/'scripts/task_runner.py',monkeypatch)
    checks=module_at(task/'_arena_checks.py',monkeypatch)
    expected=[torch.zeros(16,dtype=torch.int32),torch.zeros(16,dtype=torch.int32),
              torch.zeros(16,dtype=torch.bool),torch.zeros(16,dtype=torch.bool),
              torch.tensor([1,2] if shift else [2,3],dtype=torch.int32),
              torch.arange(4,dtype=torch.int32) if shift else torch.zeros(4,dtype=torch.int32)]
    if shift:
        expected[0][:5]=torch.tensor([11,99,-2,-1,-1]);expected[1][:5]=torch.tensor([100,101,102,0,0])
        expected[2][3:5]=True;expected[3][2]=True
    else:
        expected[0][:6]=torch.tensor([10,11,99,-2,-1,-1]);expected[1][:6]=torch.tensor([100,101,102,103,0,0])
        expected[2][4:6]=True;expected[3][3]=True
    result=checks.reference(harness,eagle_inputs(),-1,-2,2,shift)
    assert len(result)==6
    for value,gold in zip(result,expected):assert value.dtype==gold.dtype and torch.equal(value,gold)


@pytest.mark.parametrize('output_index',range(6))
@pytest.mark.parametrize('mode',['dtype','wrong_value'])
def test_eagle_correctness_checks_each_output_dtype_and_values(monkeypatch,output_index,mode):
    task=ROOT/'tasks/triton2triton/vllm/triton_copy_and_expand_eagle_inputs'
    harness=module_at(task/'scripts/task_runner.py',monkeypatch)
    checks=module_at(task/'_arena_checks.py',monkeypatch)
    def candidate(*args):
        result=list(harness.reference_copy_and_expand(*args[:-1]))
        if mode=='dtype':result[output_index]=result[output_index].long()
        elif result[output_index].dtype==torch.bool:result[output_index].logical_not_()
        else:result[output_index].fill_(123)
        return tuple(result)
    mod=SimpleNamespace(copy_and_expand_eagle_inputs=candidate);load=lambda:mod
    harness.load_module=load
    with checks.checked_modules(harness):
        with pytest.raises(AssertionError):harness.load_module().copy_and_expand_eagle_inputs(*eagle_inputs(),-1,-2,2,True,11)
    assert harness.load_module is load and mod.copy_and_expand_eagle_inputs is candidate


@pytest.mark.parametrize('mode',['correct','missing_output','extra_output','shape','mutate_inputs'])
def test_eagle_correctness_rejects_incomplete_tuples_and_input_mutation(monkeypatch,mode):
    task=ROOT/'tasks/triton2triton/vllm/triton_copy_and_expand_eagle_inputs'
    harness=module_at(task/'scripts/task_runner.py',monkeypatch)
    checks=module_at(task/'_arena_checks.py',monkeypatch)
    def candidate(*args):
        result=harness.reference_copy_and_expand(*args[:-1])
        if mode=='missing_output':return result[:-1]
        if mode=='extra_output':return (*result,result[0])
        if mode=='shape':return (result[0][:1],*result[1:])
        if mode=='mutate_inputs':args[0].zero_()
        return result
    mod=SimpleNamespace(copy_and_expand_eagle_inputs=candidate);load=lambda:mod;harness.load_module=load
    with checks.checked_modules(harness):
        call=lambda:harness.load_module().copy_and_expand_eagle_inputs(*eagle_inputs(),-1,-2,2,False,11)
        if mode=='correct':call()
        else:
            with pytest.raises(AssertionError):call()
    assert harness.load_module is load and mod.copy_and_expand_eagle_inputs is candidate


@pytest.mark.parametrize('mode',['correct','wrong_timed','stale','no_write','wrong_replay','mutate_inputs','replay_raises'])
def test_eagle_actual_six_timed_outputs_and_replay_restore(monkeypatch,mode):
    task=ROOT/'tasks/triton2triton/vllm/triton_copy_and_expand_eagle_inputs'
    harness=module_at(task/'scripts/task_runner.py',monkeypatch)
    checks=module_at(task/'_arena_checks.py',monkeypatch);harness._TimedRun=SimpleNamespace
    tt,tp,nt,qsl,qel=eagle_inputs();nps=2;tpr=4
    inputs=(tt,tp,nt,qsl,qel);pristine=tuple(value.clone() for value in inputs)
    original=lambda *args:harness.reference_copy_and_expand(*args[:-1])
    mod=SimpleNamespace(copy_and_expand_eagle_inputs=original)
    def fn():mod.copy_and_expand_eagle_inputs(tt,tp,nt,qsl,qel,-1,-2,nps,False,tpr+nps+5)
    options=[]
    def benchmark(measured,*,timed_run,**kwargs):
        options.append(kwargs);outputs=measured();cached=tuple(value.clone() for value in outputs)
        if mode=='wrong_timed':outputs[4].zero_()
        def replay():
            if mode=='replay_raises':raise RuntimeError('injected replay failure')
            if mode=='stale':
                for value,saved in zip(outputs,cached):value.copy_(saved)
            elif mode!='no_write':
                for value,new in zip(outputs,measured()):value.copy_(new)
                if mode=='wrong_replay':outputs[4].zero_()
                elif mode=='mutate_inputs':tt.zero_()
            return outputs
        timed_run.outputs=outputs;timed_run.rerun=replay
        return 0.25,dict(benchmark_method='cuda_graph')
    call=lambda:checks.checked_benchmark(harness,benchmark,fn,warmup=10,repetition=100)
    if mode=='correct':
        ms,metadata=call();assert ms==0.25 and metadata['all_six_outputs_checked'] and metadata['perturbed_input_replay_checked']
    elif mode=='replay_raises':
        with pytest.raises(RuntimeError,match='injected replay failure'):call()
    else:
        with pytest.raises(AssertionError):call()
    assert options==[dict(warmup=10,repetition=100)] and mod.copy_and_expand_eagle_inputs is original
    assert all(torch.equal(value,saved) for value,saved in zip(inputs,pristine))


def test_eagle_adapter_installs_correctness_and_timing_checks(monkeypatch):
    adapter=module_at(ROOT/'tasks/triton2triton/vllm/triton_copy_and_expand_eagle_inputs/_arena_eval.py',monkeypatch)
    harness=adapter.load_harness()
    assert harness.run_correctness.__module__==harness.run_performance.__module__=='_eagle_checks'


def slot_mapping_inputs():
    return (torch.tensor([2,0],dtype=torch.int32),torch.tensor([0,1,4],dtype=torch.int32),
            torch.tensor([3,4,1,7],dtype=torch.int64),
            torch.tensor([[10,20],[30,40],[50,60]],dtype=torch.int32))


def independent_slot_mapping(mapping,starts,positions,table,block_size,max_num_tokens):
    owners=torch.repeat_interleave(mapping.long(),(starts[1:]-starts[:-1]).long())
    return (table[owners,torch.div(positions,block_size,rounding_mode='floor')].long()*block_size+
            positions.remainder(block_size))


def test_slot_mapping_reference_ragged_permuted_known_answer(monkeypatch):
    task=ROOT/'tasks/triton2triton/vllm/triton_compute_slot_mappings'
    harness=module_at(task/'scripts/task_runner.py',monkeypatch)
    checks=module_at(task/'_arena_checks.py',monkeypatch)
    args=slot_mapping_inputs();expected=torch.tensor([203,80,41,83],dtype=torch.int64)
    assert torch.equal(checks.reference(harness,args,4),expected)
    assert torch.equal(independent_slot_mapping(*args,4,68),expected)


@pytest.mark.parametrize('mode',['correct','ignore_mapping','uniform_segments','zero_based_positions','dtype','wrong_output','mutate_inputs'])
def test_slot_mapping_correctness_sensitizes_routing_and_boundaries(monkeypatch,mode):
    task=ROOT/'tasks/triton2triton/vllm/triton_compute_slot_mappings'
    harness=module_at(task/'scripts/task_runner.py',monkeypatch)
    checks=module_at(task/'_arena_checks.py',monkeypatch)
    args=(torch.tensor([0,1],dtype=torch.int32),torch.tensor([0,2,4],dtype=torch.int32),
          torch.tensor([0,1,0,1]),slot_mapping_inputs()[3])
    def candidate(mapping,starts,positions,table,bs,max_tokens):
        m,s,p=mapping,starts,positions
        if mode=='ignore_mapping':m=torch.arange(mapping.numel(),dtype=mapping.dtype)
        elif mode=='uniform_segments':s=torch.arange(mapping.numel()+1,dtype=starts.dtype)*(positions.numel()//mapping.numel())
        elif mode=='zero_based_positions':p=torch.arange(positions.numel())%(positions.numel()//mapping.numel())
        output=independent_slot_mapping(m,s,p,table,bs,max_tokens)
        if mode=='dtype':output=output.int()
        elif mode=='wrong_output':output.zero_()
        elif mode=='mutate_inputs':table.zero_();output.zero_()
        return output
    module=SimpleNamespace(compute_slot_mappings=candidate);load=lambda:module;harness.load_module=load
    with checks.checked_modules(harness):
        call=lambda:harness.load_module().compute_slot_mappings(*args,4,68)
        if mode=='correct':call()
        else:
            with pytest.raises(AssertionError):call()
    assert harness.load_module is load and module.compute_slot_mappings is candidate


@pytest.mark.parametrize('mode',['correct','wrong_timed','stale','no_write','wrong_replay','mutate_inputs','replay_raises'])
def test_slot_mapping_exact_timed_replay_and_input_restore(monkeypatch,mode):
    task=ROOT/'tasks/triton2triton/vllm/triton_compute_slot_mappings'
    harness=module_at(task/'scripts/task_runner.py',monkeypatch)
    checks=module_at(task/'_arena_checks.py',monkeypatch);harness._TimedRun=SimpleNamespace
    idx_mapping,query_start_loc,positions,block_table=slot_mapping_inputs();block_size=4;max_num_tokens=68
    inputs=(idx_mapping,query_start_loc,positions,block_table);pristine=tuple(value.clone() for value in inputs)
    mod=SimpleNamespace(compute_slot_mappings=independent_slot_mapping);original=mod.compute_slot_mappings
    def fn():mod.compute_slot_mappings(idx_mapping,query_start_loc,positions,block_table,block_size,max_num_tokens)
    options=[]
    def benchmark(measured,*,timed_run,**kwargs):
        options.append(kwargs);output=measured();cached=output.clone()
        if mode=='wrong_timed':output.zero_()
        def replay():
            if mode=='replay_raises':raise RuntimeError('injected replay failure')
            if mode=='stale':output.copy_(cached)
            elif mode!='no_write':
                output.copy_(measured())
                if mode=='wrong_replay':output.zero_()
                elif mode=='mutate_inputs':block_table.zero_();output.zero_()
            return output
        timed_run.outputs=output;timed_run.rerun=replay
        return 0.25,dict(benchmark_method='cuda_graph')
    call=lambda:checks.checked_benchmark(harness,benchmark,fn,warmup=10,repetition=100)
    if mode=='correct':
        ms,metadata=call();assert ms==0.25 and metadata['ragged_permuted_mapping_checked']
    elif mode=='replay_raises':
        with pytest.raises(RuntimeError,match='injected replay failure'):call()
    else:
        with pytest.raises(AssertionError):call()
    assert options==[dict(warmup=10,repetition=100)] and mod.compute_slot_mappings is original
    assert all(torch.equal(value,saved) for value,saved in zip(inputs,pristine))


def test_slot_mapping_adapter_installs_correctness_and_timing_checks(monkeypatch):
    adapter=module_at(ROOT/'tasks/triton2triton/vllm/triton_compute_slot_mappings/_arena_eval.py',monkeypatch)
    harness=adapter.load_harness()
    assert harness.run_correctness.__module__==harness.run_performance.__module__=='_slot_mapping_checks'


def grammar_inputs():
    logits=torch.arange(99,dtype=torch.float32).reshape(3,33)/10
    indices=torch.tensor([2,0],dtype=torch.int32)
    bits=torch.tensor([[-2147483647,1],[2,0]],dtype=torch.int32)
    return logits,indices,bits


def independent_grammar(logits,indices,bits,vocab):
    columns=torch.arange(vocab)
    words=bits.long()[:,columns//32]
    keep=((words>>(columns%32))&1).bool()
    logits[indices.long(),:vocab]=torch.where(keep,logits[indices.long(),:vocab],float('-inf'))


def test_grammar_signed_mask_reference_and_unselected_row_known_answers(monkeypatch):
    task=ROOT/'tasks/triton2triton/vllm/triton_apply_grammar_bitmask'
    harness=module_at(task/'scripts/task_runner.py',monkeypatch)
    checks=module_at(task/'_arena_checks.py',monkeypatch)
    args=grammar_inputs();expected=args[0].clone()
    expected[0].fill_(float('-inf'));expected[0,1]=args[0][0,1]
    expected[2].fill_(float('-inf'));expected[2,[0,31,32]]=args[0][2,[0,31,32]]
    assert torch.equal(checks.reference(harness,args,33),expected)
    independent_grammar(*args,33);assert torch.equal(args[0],expected)


@pytest.mark.parametrize('mode',['correct','ignore_mapping','drop_bit31','clobber_unselected','mutate_bits','nonfinite','wrong_unmasked'])
def test_grammar_correctness_checks_signed_words_mapping_and_all_rows(monkeypatch,mode):
    task=ROOT/'tasks/triton2triton/vllm/triton_apply_grammar_bitmask'
    harness=module_at(task/'scripts/task_runner.py',monkeypatch)
    checks=module_at(task/'_arena_checks.py',monkeypatch)
    def candidate(logits,indices,bits,vocab):
        index=indices if mode!='ignore_mapping' else torch.arange(indices.numel(),dtype=indices.dtype)
        mask=bits if mode!='drop_bit31' else bits.bitwise_and(2147483647)
        independent_grammar(logits,index,mask,vocab)
        if mode=='clobber_unselected':
            unselected=next(i for i in range(logits.shape[0]) if i not in indices.tolist());logits[unselected].zero_()
        elif mode=='mutate_bits':bits.zero_()
        elif mode=='nonfinite':logits[0,1]=float('nan')
        elif mode=='wrong_unmasked':logits[torch.isfinite(logits)]+=10
    mod=SimpleNamespace(apply_grammar_bitmask=candidate);load=lambda:mod;harness.load_module=load
    with checks.checked_modules(harness):
        call=lambda:harness.load_module().apply_grammar_bitmask(*grammar_inputs(),33)
        if mode=='correct':call()
        else:
            with pytest.raises(AssertionError):call()
    assert harness.load_module is load and mod.apply_grammar_bitmask is candidate


@pytest.mark.parametrize('mode',['correct','wrong_timed','stale','no_write','wrong_replay','mutate_bits','replay_raises'])
def test_grammar_actual_timed_in_place_output_and_restoration(monkeypatch,mode):
    task=ROOT/'tasks/triton2triton/vllm/triton_apply_grammar_bitmask'
    harness=module_at(task/'scripts/task_runner.py',monkeypatch)
    checks=module_at(task/'_arena_checks.py',monkeypatch);harness._TimedRun=SimpleNamespace
    logits,logits_indices,bitmask=grammar_inputs();logits_work=logits.clone();vocab_size=33
    inputs=(logits,logits_indices,bitmask,logits_work);pristine=tuple(value.clone() for value in inputs)
    def fn():independent_grammar(logits_work,logits_indices,bitmask,vocab_size)
    def prepare():logits_work.copy_(logits)
    options=[]
    def benchmark(measured,*,timed_run,**kwargs):
        options.append(kwargs);kwargs['prepare_fn']();output=measured();cached=output.clone()
        if mode=='wrong_timed':output.zero_()
        def replay():
            kwargs['prepare_fn']()
            if mode=='replay_raises':raise RuntimeError('injected replay failure')
            if mode=='stale':output.copy_(cached)
            elif mode!='no_write':
                measured()
                if mode=='wrong_replay':output.zero_()
                elif mode=='mutate_bits':bitmask.zero_()
            return output
        timed_run.outputs=output;timed_run.rerun=replay
        return 0.25,dict(benchmark_method='cuda_graph')
    call=lambda:checks.checked_benchmark(harness,benchmark,fn,warmup=10,repetition=100,target_ms=20.0,prepare_fn=prepare)
    if mode=='correct':
        ms,metadata=call();assert ms==0.25 and metadata['signed_mask_mapping_checked'] and metadata['perturbed_input_replay_checked']
    elif mode=='replay_raises':
        with pytest.raises(RuntimeError,match='injected replay failure'):call()
    else:
        with pytest.raises(AssertionError):call()
    assert options==[dict(warmup=10,repetition=100,target_ms=20.0,prepare_fn=prepare)]
    assert all(torch.equal(value,saved) for value,saved in zip(inputs,pristine))


def test_grammar_adapter_installs_correctness_and_timing_checks(monkeypatch):
    adapter=module_at(ROOT/'tasks/triton2triton/vllm/triton_apply_grammar_bitmask/_arena_eval.py',monkeypatch)
    harness=adapter.load_harness()
    assert harness.run_correctness.__module__==harness.run_performance.__module__=='_grammar_checks'


def _expert_counts_cpu(ids, experts):
    valid = ids[(ids >= 0) & (ids < experts)].to(torch.int64)
    return torch.bincount(valid, minlength=experts).to(torch.int32)


@pytest.mark.parametrize('mode', ['correct', 'wrong_values', 'dtype', 'shape', 'mutate_input', 'count_invalid_as_zero'])
def test_expert_count_reference_and_output_contract(monkeypatch, mode):
    task = ROOT/'tasks/triton2triton/vllm/triton_count_expert_tokens'
    h = module_at(task/'scripts/task_runner.py', monkeypatch)
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    ids = torch.tensor([[0, 1, 2], [2, -1, -1]], dtype=torch.int32)
    expected = torch.tensor([1, 1, 2, 0], dtype=torch.int32)
    torch.testing.assert_close(h.reference_count(ids, 4), expected, atol=0, rtol=0)
    def candidate(ids, experts):
        if mode == 'mutate_input': ids.zero_()
        value = _expert_counts_cpu(ids.clamp_min(0) if mode == 'count_invalid_as_zero' else ids, experts)
        if mode == 'wrong_values': value.zero_()
        if mode == 'dtype': value = value.float()
        if mode == 'shape': value = value[:-1]
        return value
    mod = SimpleNamespace(count_expert_num_tokens=candidate)
    h.load_module = lambda: mod
    with checks.checked_modules(h):
        checked = h.load_module().count_expert_num_tokens
        if mode == 'correct': torch.testing.assert_close(checked(ids, 4), expected)
        else:
            with pytest.raises(AssertionError): checked(ids, 4)
    assert mod.count_expert_num_tokens is candidate


@pytest.mark.parametrize('mode', ['correct', 'wrong_timed', 'stale', 'no_write', 'wrong_replay',
                                 'mutate_timed', 'mutate_replay', 'raise_replay'])
def test_expert_count_captured_replay_and_restoration(monkeypatch, mode):
    task = ROOT/'tasks/triton2triton/vllm/triton_count_expert_tokens'
    h = module_at(task/'scripts/task_runner.py', monkeypatch)
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    h._TimedRun = SimpleNamespace
    topk_ids = torch.tensor([[0, 1, 2], [2, 3, 1]], dtype=torch.int32)
    pristine = topk_ids.clone()
    num_experts = 4
    mod = SimpleNamespace(count_expert_num_tokens=_expert_counts_cpu)
    invocations = []
    def fn():
        invocations.append(True)
        mod.count_expert_num_tokens(topk_ids, num_experts)
    options = []
    def benchmark(measured, *, timed_run, **kwargs):
        options.append(kwargs)
        output = measured(); cached = output.clone()
        assert len(invocations) == 1
        if mode == 'wrong_timed': output.zero_()
        if mode == 'mutate_timed': topk_ids.zero_()
        def replay():
            if mode == 'raise_replay': raise RuntimeError('replay failed')
            if mode == 'stale': output.copy_(cached)
            elif mode != 'no_write': output.copy_(measured())
            if mode == 'wrong_replay': output.zero_()
            if mode == 'mutate_replay': topk_ids.zero_()
            return output
        timed_run.outputs, timed_run.rerun = output, replay
        return 0.125, {'benchmark_method': 'cuda_graph'}
    run = lambda: checks.checked_benchmark(h, benchmark, fn, warmup=10, repetition=100)
    if mode == 'correct':
        ms, metadata = run()
        assert ms == 0.125 and metadata['perturbed_input_replay_checked']
    else:
        with pytest.raises((AssertionError, RuntimeError)): run()
    assert options == [dict(warmup=10, repetition=100)]
    torch.testing.assert_close(topk_ids, pristine)
    assert mod.count_expert_num_tokens is _expert_counts_cpu


def test_expert_count_adapter_installs_checks(monkeypatch):
    adapter = module_at(ROOT/'tasks/triton2triton/vllm/triton_count_expert_tokens/_arena_eval.py', monkeypatch)
    h = adapter.load_harness()
    assert h.run_correctness.__module__ == h.run_performance.__module__ == '_expert_count_checks'


PADDED_EAGLE = ['triton_eagle_prepare_inputs_padded', 'triton_eagle_prepare_next_token_padded']


def _padded_eagle_cpu_inputs(name):
    if name == PADDED_EAGLE[0]:
        inputs = (torch.tensor([2, 2, 5], dtype=torch.int32),
                  torch.tensor([2, 1, 1], dtype=torch.int32),
                  torch.tensor([0, 4, 7, 12], dtype=torch.int32))
        scalars = ()
        expected = (torch.tensor([2, 6, 8], dtype=torch.int32), torch.tensor([1, 0, 3], dtype=torch.int32))
        def candidate(cu, vs, qsl):
            drafts = cu - torch.cat((cu.new_zeros(1), cu[:-1]))
            rejected = torch.where(drafts > 0, drafts+1-vs, 0)
            return qsl[1:]-1-rejected, rejected
    else:
        inputs = (torch.tensor([[1, -1, 3], [5, 6, 7], [-1, -1, -1]], dtype=torch.int32),
                  torch.tensor([False, True, False]), torch.tensor([9, 8, 4], dtype=torch.int32))
        scalars = (10,)
        expected = (torch.tensor([3, 8, 4], dtype=torch.int32), torch.tensor([2, 0, 0], dtype=torch.int32))
        def candidate(sampled, dm, backup, vs):
            valid = (sampled != -1) & (sampled < vs)
            count = valid.sum(1).to(torch.int32)
            indices = torch.where(valid, torch.arange(sampled.shape[1])[None, :], -1).max(1).values
            selected = sampled.gather(1, indices.clamp_min(0)[:, None]).squeeze(1)
            return torch.where((count > 0) & ~dm, selected, backup), torch.where(dm, 0, count)
    return inputs, scalars, expected, candidate


@pytest.mark.parametrize('name', PADDED_EAGLE)
@pytest.mark.parametrize('mode', ['correct', 'dtype', 'shape', 'missing_second', 'wrong_first', 'wrong_second', 'mutate_input'])
def test_padded_eagle_known_answers_output_contract_and_pristine_inputs(monkeypatch, name, mode):
    task = ROOT/'tasks/triton2triton/vllm'/name
    h = module_at(task/'scripts/task_runner.py', monkeypatch)
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    inputs, scalars, expected, correct = _padded_eagle_cpu_inputs(name)
    for actual, known in zip(h.reference(*inputs, *scalars), expected):
        torch.testing.assert_close(actual, known, atol=0, rtol=0)
    def candidate(*args):
        if mode == 'mutate_input': args[0].zero_()
        outputs = list(correct(*args))
        if mode == 'dtype': outputs[0] = outputs[0].float()
        if mode == 'shape': outputs[0] = outputs[0][:1]
        if mode == 'missing_second': outputs.pop()
        if mode == 'wrong_first': outputs[0].zero_()
        if mode == 'wrong_second': outputs[1].fill_(17)
        return tuple(outputs)
    mod = SimpleNamespace(**{checks.SYMBOL: candidate})
    h.load_module = lambda: mod
    with checks.checked_modules(h):
        call = getattr(h.load_module(), checks.SYMBOL)
        if mode == 'correct': checks.check_outputs(call(*inputs, *scalars), expected)
        else:
            with pytest.raises(AssertionError): call(*inputs, *scalars)
    assert getattr(mod, checks.SYMBOL) is candidate


@pytest.mark.parametrize('name', PADDED_EAGLE)
@pytest.mark.parametrize('mode', ['correct', 'wrong_timed', 'stale', 'no_write', 'omit_second',
                                 'wrong_replay', 'mutate_timed', 'mutate_replay', 'raise_replay'])
def test_padded_eagle_captured_outputs_replay_and_restoration(monkeypatch, name, mode):
    task = ROOT/'tasks/triton2triton/vllm'/name
    h = module_at(task/'scripts/task_runner.py', monkeypatch)
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    h._TimedRun = SimpleNamespace
    inputs, scalars, expected, correct = _padded_eagle_cpu_inputs(name)
    pristine = checks.snapshots(inputs)
    mod = SimpleNamespace(**{checks.SYMBOL: correct})
    if name == PADDED_EAGLE[0]:
        cu, vs, qsl = inputs
        def fn(): mod.eagle_prepare_inputs_padded(cu, vs, qsl)
    else:
        sampled, dm, backup = inputs
        vs, = scalars
        def fn(): mod.eagle_prepare_next_token_padded(sampled, dm, backup, vs)
    options = []
    def benchmark(measured, *, timed_run, **kwargs):
        options.append(kwargs)
        outputs = measured(); cached = tuple(out.clone() for out in outputs)
        if mode == 'wrong_timed': outputs[0].zero_()
        if mode == 'mutate_timed': inputs[0].zero_()
        def replay():
            if mode == 'raise_replay': raise RuntimeError('replay failed')
            if mode != 'no_write':
                computed = cached if mode == 'stale' else measured()
                for i, (out, value) in enumerate(zip(outputs, computed)):
                    if mode != 'omit_second' or i == 0: out.copy_(value)
            if mode == 'wrong_replay': outputs[1].fill_(17)
            if mode == 'mutate_replay': inputs[0].zero_()
            return outputs
        timed_run.outputs, timed_run.rerun = outputs, replay
        return 0.125, {'benchmark_method': 'cuda_graph'}
    call = lambda: checks.checked_benchmark(h, benchmark, fn, warmup=10, repetition=100)
    if mode == 'correct':
        ms, metadata = call()
        assert ms == 0.125 and metadata['perturbed_input_replay_checked']
    else:
        with pytest.raises((AssertionError, RuntimeError)): call()
    checks.unchanged(inputs, pristine)
    assert options == [dict(warmup=10, repetition=100)]
    assert getattr(mod, checks.SYMBOL) is correct


@pytest.mark.parametrize('name', PADDED_EAGLE)
def test_padded_eagle_adapter_installs_checks(monkeypatch, name):
    adapter = module_at(ROOT/'tasks/triton2triton/vllm'/name/'_arena_eval.py', monkeypatch)
    h = adapter.load_harness()
    assert h.run_correctness.__module__ == h.run_performance.__module__ == '_eagle_padded_checks'


CUMSUM_TASKS = ['triton_fla_cumsum_scalar', 'triton_fla_cumsum_vector']


def _chunk_cumsum_cpu(g, chunk_size, reverse=False):
    # Deliberately separate scalar recurrence from the harness's torch.cumsum.
    result = torch.empty_like(g, dtype=torch.float32)
    for start in range(0, g.shape[1], chunk_size):
        indices = list(range(start, min(start + chunk_size, g.shape[1])))
        if reverse:
            indices.reverse()
        total = torch.zeros_like(g[:, 0], dtype=torch.float32)
        for i in indices:
            total = total + g[:, i]
            result[:, i] = total
    return result


@pytest.mark.parametrize('name', CUMSUM_TASKS)
@pytest.mark.parametrize('reverse', [False, True])
def test_chunk_cumsum_reference_independent_known_answer(monkeypatch, name, reverse):
    h = module_at(ROOT/'tasks/triton2triton/vllm'/name/'scripts/task_runner.py', monkeypatch)
    g = torch.arange(1, 6, dtype=torch.float32).reshape(1, 5, 1)
    expected = torch.tensor([10, 9, 7, 4, 5] if reverse else [1, 3, 6, 10, 5]).reshape(1, 5, 1).float()
    if name.endswith('vector'):
        g = torch.stack((g, -2*g), dim=-1)
        expected = torch.stack((expected, -2*expected), dim=-1)
    torch.testing.assert_close(h.reference(g, 4, reverse), expected, atol=0, rtol=0)
    torch.testing.assert_close(_chunk_cumsum_cpu(g, 4, reverse), expected, atol=0, rtol=0)


@pytest.mark.parametrize('name', CUMSUM_TASKS)
@pytest.mark.parametrize('mode', ['correct', 'dtype', 'shape', 'nonfinite', 'mutate_input',
                                 'ignores_reverse', 'fixed_chunk', 'fixed_geometry'])
def test_chunk_cumsum_original_correctness_orchestration(monkeypatch, name, mode):
    task = ROOT/'tasks/triton2triton/vllm'/name
    h = module_at(task/'scripts/task_runner.py', monkeypatch)
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    original_inputs = h.gen_inputs
    # Keep all five original seeds and original tensor geometries, on CPU.
    h.gen_inputs = lambda seed, device: original_inputs(seed, 'cpu')
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    calls = []
    def candidate(g, chunk_size, reverse=False):
        calls.append((tuple(g.shape), chunk_size, reverse))
        if mode == 'mutate_input': g.zero_()
        value = _chunk_cumsum_cpu(g, 64 if mode == 'fixed_chunk' else chunk_size,
                                  False if mode == 'ignores_reverse' else reverse)
        if mode == 'dtype': value = value.double()
        if mode == 'shape': value = value[:, :1]
        if mode == 'nonfinite': value.fill_(float('inf'))
        if mode == 'fixed_geometry' and g.shape[0] != 2: value.zero_()
        return value
    mod = SimpleNamespace(**{checks.SYMBOL: candidate})
    h.load_module = lambda: mod
    checks.install(h)
    ok, reason = h.run_correctness()
    assert ok is (mode == 'correct'), reason
    if mode == 'correct':
        assert len(calls) == 10
        assert all(shape[:3] == (2, 128, 4) and chunk == 64 and not rev
                   for shape, chunk, rev in calls[1::2])
        assert all(shape[:3] == (1, 125, 2) and chunk == 32 and rev
                   for shape, chunk, rev in calls[::2])
        if name.endswith('vector'):
            assert all(shape[-1] == 17 for shape, _, _ in calls[::2])
    assert getattr(mod, checks.SYMBOL) is candidate


@pytest.mark.parametrize('name', CUMSUM_TASKS)
@pytest.mark.parametrize('mode', ['correct', 'wrong_timed', 'stale', 'no_write', 'wrong_replay',
                                 'mutate_timed', 'mutate_replay', 'raise_replay'])
def test_chunk_cumsum_original_performance_replay_and_restoration(monkeypatch, name, mode):
    task = ROOT/'tasks/triton2triton/vllm'/name
    h = module_at(task/'scripts/task_runner.py', monkeypatch)
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    h._TimedRun = SimpleNamespace
    original_inputs = h.gen_inputs
    inputs, originals, options = [], [], []
    def gen_inputs(seed, device):
        args, kwargs = original_inputs(seed, 'cpu')
        inputs.append(args[0]); originals.append(args[0].clone())
        return args, kwargs
    h.gen_inputs = gen_inputs
    mod = SimpleNamespace(**{checks.SYMBOL: _chunk_cumsum_cpu})
    h.load_module = lambda: mod
    def benchmark(measured, *, timed_run, **kwargs):
        options.append(kwargs)
        output = measured(); cached = output.clone()
        g = inputs[-1]
        if mode == 'wrong_timed': output.zero_()
        if mode == 'mutate_timed': g.zero_()
        def replay():
            if mode == 'raise_replay': raise RuntimeError('replay failed')
            if mode == 'stale': output.copy_(cached)
            elif mode != 'no_write': output.copy_(measured())
            if mode == 'wrong_replay': output.zero_()
            if mode == 'mutate_replay': g.zero_()
            return output
        timed_run.outputs, timed_run.rerun = output, replay
        return 0.125, {'benchmark_method': 'cuda_graph'}
    h._benchmark_cuda_graph_or_events = benchmark
    checks.install(h)
    cases = h.run_performance()
    assert len(cases) == 5
    assert options == [dict(warmup=10, repetition=100)] * 5
    assert all(case['execution_time_ms'] == (0.125 if mode == 'correct' else -1.0) for case in cases)
    if mode == 'correct':
        assert all(case['perturbed_input_replay_checked'] and case['source_buffers_unchanged'] for case in cases)
    for value, saved in zip(inputs, originals):
        torch.testing.assert_close(value, saved, atol=0, rtol=0)
    assert h._benchmark_cuda_graph_or_events is benchmark
    assert getattr(mod, checks.SYMBOL) is _chunk_cumsum_cpu


@pytest.mark.parametrize('name', CUMSUM_TASKS)
def test_chunk_cumsum_adapter_installs_checks(monkeypatch, name):
    adapter = module_at(ROOT/'tasks/triton2triton/vllm'/name/'_arena_eval.py', monkeypatch)
    h = adapter.load_harness()
    assert h.run_correctness.__module__ == h.run_performance.__module__ == '_cumsum_checks'


def _l2norm_cpu(x, eps=1e-6):
    # Independent norm operator, rather than the harness's sum(x*x).
    return x / torch.sqrt(torch.linalg.vector_norm(x, dim=-1, keepdim=True).square() + eps)


def test_fla_l2norm_reference_known_answer_and_epsilon(monkeypatch):
    h = module_at(ROOT/'tasks/triton2triton/vllm/triton_fla_l2norm/scripts/task_runner.py', monkeypatch)
    x = torch.tensor([[3., 4.], [0., 0.], [1e-4, 0.]])
    expected = torch.tensor([[3/26**0.5, 4/26**0.5], [0., 0.], [1e-4/(1+1e-8)**0.5, 0.]])
    torch.testing.assert_close(h.reference(x, 1.), expected, atol=1e-7, rtol=1e-7)
    torch.testing.assert_close(_l2norm_cpu(x, 1.), expected, atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize('mode', ['correct', 'dtype', 'shape', 'nonfinite', 'mutate_input',
                                 'ignores_eps', 'fixed_geometry'])
def test_fla_l2norm_actual_correctness_orchestration(monkeypatch, mode):
    task = ROOT/'tasks/triton2triton/vllm/triton_fla_l2norm'
    h = module_at(task/'scripts/task_runner.py', monkeypatch)
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    generator = h.gen_inputs
    h.gen_inputs = lambda seed, device: generator(seed, 'cpu')
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    calls = []
    def candidate(x, eps=1e-6):
        calls.append((tuple(x.shape), eps))
        if mode == 'mutate_input': x.zero_()
        value = _l2norm_cpu(x, 1e-6 if mode == 'ignores_eps' else eps)
        if mode == 'dtype': value = value.double()
        if mode == 'shape': value = value[..., :1]
        if mode == 'nonfinite': value.fill_(float('nan'))
        if mode == 'fixed_geometry' and x.shape != (512, 128): value.zero_()
        return value
    mod = SimpleNamespace(l2norm_fwd=candidate)
    h.load_module = lambda: mod
    checks.install(h)
    ok, reason = h.run_correctness()
    assert ok is (mode == 'correct'), reason
    if mode == 'correct':
        assert calls == [((2, 3, 17), 1e-3), ((512, 128), 1e-6)] * 5
    assert mod.l2norm_fwd is candidate


@pytest.mark.parametrize('mode', ['correct', 'wrong_timed', 'stale', 'no_write', 'wrong_replay',
                                 'mutate_timed', 'mutate_replay', 'raise_replay'])
def test_fla_l2norm_actual_performance_replay_restores_inputs(monkeypatch, mode):
    task = ROOT/'tasks/triton2triton/vllm/triton_fla_l2norm'
    h = module_at(task/'scripts/task_runner.py', monkeypatch)
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    h._TimedRun = SimpleNamespace
    generator = h.gen_inputs
    inputs, pristine, options = [], [], []
    def gen_inputs(seed, device):
        args, kwargs = generator(seed, 'cpu')
        inputs.append(args[0]); pristine.append(args[0].clone())
        return args, kwargs
    h.gen_inputs = gen_inputs
    mod = SimpleNamespace(l2norm_fwd=_l2norm_cpu)
    h.load_module = lambda: mod
    def benchmark(measured, *, timed_run, **kwargs):
        options.append(kwargs)
        output = measured(); cached = output.clone(); x = inputs[-1]
        if mode == 'wrong_timed': output.zero_()
        if mode == 'mutate_timed': x.zero_()
        def replay():
            if mode == 'raise_replay': raise RuntimeError('replay failed')
            if mode == 'stale': output.copy_(cached)
            elif mode != 'no_write': output.copy_(measured())
            if mode == 'wrong_replay': output.zero_()
            if mode == 'mutate_replay': x.zero_()
            return output
        timed_run.outputs, timed_run.rerun = output, replay
        return 0.125, {'benchmark_method': 'cuda_graph'}
    h._benchmark_cuda_graph_or_events = benchmark
    checks.install(h)
    rows = h.run_performance()
    assert len(rows) == 5
    assert options == [dict(warmup=10, repetition=100)] * 5
    assert all(row['execution_time_ms'] == (0.125 if mode == 'correct' else -1.) for row in rows)
    if mode == 'correct':
        assert all(row['timed_output_checked'] and row['perturbed_input_replay_checked'] for row in rows)
    for x, saved in zip(inputs, pristine):
        torch.testing.assert_close(x, saved, atol=0, rtol=0)
    assert mod.l2norm_fwd is _l2norm_cpu
    assert h._benchmark_cuda_graph_or_events is benchmark


def test_fla_l2norm_adapter_installs_checks(monkeypatch):
    adapter = module_at(ROOT/'tasks/triton2triton/vllm/triton_fla_l2norm/_arena_eval.py', monkeypatch)
    h = adapter.load_harness()
    assert h.run_correctness.__module__ == h.run_performance.__module__ == '_l2norm_checks'


def _gated_norm_cpu(x, g, weight=None, bias=None, activation='swish', eps=1e-5, is_rms_norm=True):
    if is_rms_norm:
        mean = None
        rstd = (torch.linalg.vector_norm(x, dim=-1).square()/x.shape[-1] + eps).rsqrt()
        y = x * rstd[:, None]
        if weight is not None: y = y * weight
        if bias is not None: y = y + bias
    else:
        var, mean = torch.var_mean(x, dim=-1, unbiased=False)
        rstd = (var + eps).rsqrt()
        y = torch.nn.functional.layer_norm(x, (x.shape[-1],), weight, bias, eps)
    if activation in ('silu', 'swish'): y = y * torch.nn.functional.silu(g)
    elif activation == 'sigmoid': y = y * torch.sigmoid(g)
    return y, mean, rstd


@pytest.mark.parametrize('is_rms', [False, True])
def test_gated_norm_reference_known_full_tuple(monkeypatch, is_rms):
    task = ROOT/'tasks/triton2triton/vllm/triton_fla_layernorm_gated'
    h = module_at(task/'scripts/task_runner.py', monkeypatch)
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    x, g = torch.tensor([[1., 3.]]), torch.zeros(1, 2)
    eps = 1e-5
    scale = (5+eps if is_rms else 1+eps)**-0.5
    y = torch.tensor([[.5, 1.5] if is_rms else [-.5, .5]]) * scale
    expected = y, None if is_rms else torch.tensor([2.]), torch.tensor([scale])
    options = dict(activation='sigmoid', is_rms_norm=is_rms, eps=eps)
    checks.check_outputs(checks.reference(h, (x, g, None, None), options), expected)
    checks.check_outputs(_gated_norm_cpu(x, g, **options), expected)


@pytest.mark.parametrize('mode', ['correct', 'dtype', 'stat_dtype', 'shape', 'nonfinite',
                                 'missing_stats', 'wrong_mean', 'wrong_rstd', 'mutate_input'])
def test_gated_norm_actual_correctness_full_outputs(monkeypatch, mode):
    task = ROOT/'tasks/triton2triton/vllm/triton_fla_layernorm_gated'
    h = module_at(task/'scripts/task_runner.py', monkeypatch)
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    generator = h.gen_inputs
    h.gen_inputs = lambda seed, case_idx, device: generator(seed, case_idx, 'cpu')
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    calls = []
    def candidate(x, g, **kwargs):
        calls.append((tuple(x.shape), kwargs['is_rms_norm']))
        if mode == 'mutate_input': g.zero_()
        outputs = list(_gated_norm_cpu(x, g, **kwargs))
        if mode == 'dtype': outputs[0] = outputs[0].double()
        if mode == 'stat_dtype': outputs[2] = outputs[2].double()
        if mode == 'shape': outputs[0] = outputs[0][:1]
        if mode == 'nonfinite': outputs[2].fill_(float('inf'))
        if mode == 'missing_stats': outputs = outputs[:1]
        if mode == 'wrong_mean': outputs[1] = torch.full_like(outputs[2], 17.)
        if mode == 'wrong_rstd': outputs[2].zero_()
        return tuple(outputs)
    mod = SimpleNamespace(layer_norm_gated_fwd=candidate)
    h.load_module = lambda: mod
    checks.install(h)
    ok, reason = h.run_correctness()
    assert ok is (mode == 'correct'), reason
    if mode == 'correct':
        assert calls == [((case[0], case[1]), case[3]) for case in h.TEST_CASES]
    assert mod.layer_norm_gated_fwd is candidate


@pytest.mark.parametrize('mode', ['correct', 'wrong_timed', 'stale', 'no_write', 'omit_mean',
                                 'omit_rstd', 'wrong_replay', 'mutate_timed', 'mutate_replay', 'raise_replay'])
def test_gated_norm_actual_performance_full_tuple_replay(monkeypatch, mode):
    task = ROOT/'tasks/triton2triton/vllm/triton_fla_layernorm_gated'
    h = module_at(task/'scripts/task_runner.py', monkeypatch)
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    h._TimedRun = SimpleNamespace
    generator = h.gen_inputs
    inputs, pristine, options = [], [], []
    def gen_inputs(seed, case_idx, device):
        args, kwargs = generator(seed, case_idx, 'cpu')
        values = (*args, kwargs.get('weight'), kwargs.get('bias'))
        inputs.append(values); pristine.append(checks.snapshots(values))
        return args, kwargs
    h.gen_inputs = gen_inputs
    mod = SimpleNamespace(layer_norm_gated_fwd=_gated_norm_cpu)
    h.load_module = lambda: mod
    def benchmark(measured, *, timed_run, **kwargs):
        options.append(kwargs)
        outputs = measured(); cached = checks.snapshots(outputs)
        if mode == 'wrong_timed': outputs[2].zero_()
        if mode == 'mutate_timed': inputs[-1][1].zero_()
        def replay():
            if mode == 'raise_replay': raise RuntimeError('replay failed')
            if mode != 'no_write':
                computed = cached if mode == 'stale' else measured()
                for i, (value, calculated) in enumerate(zip(outputs, computed)):
                    if value is not None and not (mode == 'omit_mean' and i == 1 or mode == 'omit_rstd' and i == 2):
                        value.copy_(calculated)
            if mode == 'wrong_replay': outputs[2].zero_()
            if mode == 'mutate_replay': inputs[-1][0].zero_()
            return outputs
        timed_run.outputs, timed_run.rerun = outputs, replay
        return 0.125, {'benchmark_method': 'cuda_graph'}
    h._benchmark_cuda_graph_or_events = benchmark
    checks.install(h)
    rows = h.run_performance()
    assert len(rows) == 5
    assert options == [dict(warmup=10, repetition=100)] * 5
    for case, row in zip(h.TEST_CASES, rows):
        # RMSNorm deliberately has no mean buffer to rewrite.
        success = mode == 'correct' or (mode == 'omit_mean' and case[3])
        assert row['execution_time_ms'] == (0.125 if success else -1.)
        if success: assert row['perturbed_input_replay_checked']
    for values, saved in zip(inputs, pristine): checks.unchanged(values, saved)
    assert mod.layer_norm_gated_fwd is _gated_norm_cpu
    assert h._benchmark_cuda_graph_or_events is benchmark


def test_gated_norm_adapter_installs_checks(monkeypatch):
    adapter = module_at(ROOT/'tasks/triton2triton/vllm/triton_fla_layernorm_gated/_arena_eval.py', monkeypatch)
    h = adapter.load_harness()
    assert h.run_correctness.__module__ == h.run_performance.__module__ == '_gatednorm_checks'


def _expand_cpu(x, cu, num_tokens, replace_from=0, replace_to=0):
    counts = cu - torch.cat((cu.new_zeros(1), cu[:-1]))
    values = torch.where(x == replace_from, replace_to, x)
    return values.repeat_interleave(counts)


@pytest.mark.parametrize('mode', ['correct', 'dtype', 'shape', 'mutate_source', 'mutate_counts',
                                 'uniform_only', 'ignores_replacement'])
def test_expand_ragged_replacement_known_answer_and_negative_controls(monkeypatch, mode):
    task = ROOT/'tasks/triton2triton/vllm/triton_expand'
    h = module_at(task/'scripts/task_runner.py', monkeypatch)
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    x = torch.tensor([7, 2, 7, 9], dtype=torch.int32)
    cu = torch.tensor([0, 1, 4, 6], dtype=torch.int64)
    known = torch.tensor([2, -3, -3, -3, 9, 9], dtype=torch.int32)
    torch.testing.assert_close(checks.reference(x, cu, 6, 7, -3), known, atol=0, rtol=0)
    torch.testing.assert_close(_expand_cpu(x, cu, 6, 7, -3), known, atol=0, rtol=0)
    def candidate(x, cu, num_tokens, replace_from=0, replace_to=0):
        if mode == 'mutate_source': x.zero_()
        if mode == 'mutate_counts': cu[0] += 1
        output = _expand_cpu(x, cu, num_tokens, replace_from,
                             replace_from if mode == 'ignores_replacement' else replace_to)
        if mode == 'uniform_only': output = x.repeat_interleave(max(1, num_tokens//len(x)))
        if mode == 'dtype': output = output.float()
        if mode == 'shape': output = output[:1]
        return output
    mod = SimpleNamespace(expand_batch_to_tokens=candidate)
    h.load_module = lambda: mod
    with checks.checked_modules(h):
        call = h.load_module().expand_batch_to_tokens
        if mode == 'correct': torch.testing.assert_close(call(x, cu, 6, 7, -3), known)
        else:
            with pytest.raises(AssertionError): call(x, cu, 6, 7, -3)
    assert mod.expand_batch_to_tokens is candidate


@pytest.mark.parametrize('mode', ['correct', 'wrong_timed', 'stale', 'no_write', 'wrong_replay',
                                 'mutate_timed', 'mutate_replay', 'raise_replay'])
def test_expand_original_performance_orchestration_and_restoration(monkeypatch, mode):
    import inspect
    task = ROOT/'tasks/triton2triton/vllm/triton_expand'
    h = module_at(task/'scripts/task_runner.py', monkeypatch)
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    h._TimedRun = SimpleNamespace
    for name in ('randint', 'full'):
        factory = getattr(torch, name)
        def cpu_factory(*args, _factory=factory, **kwargs):
            return _factory(*args, **{**kwargs, 'device': 'cpu'})
        monkeypatch.setattr(torch, name, cpu_factory)
    inputs, pristine, options = [], [], []
    mod = SimpleNamespace(expand_batch_to_tokens=_expand_cpu)
    h.load_module = lambda: mod
    def benchmark(measured, *, timed_run, **kwargs):
        options.append(kwargs)
        fn = inspect.getclosurevars(measured).nonlocals['fn']
        closed = inspect.getclosurevars(fn).nonlocals
        x, cu = closed['x'], closed['cu']
        inputs.append((x, cu)); pristine.append((x.clone(), cu.clone()))
        output = measured(); cached = output.clone()
        if mode == 'wrong_timed': output.fill_(-1)
        if mode == 'mutate_timed': x.zero_()
        def replay():
            if mode == 'raise_replay': raise RuntimeError('replay failed')
            if mode == 'stale': output.copy_(cached)
            elif mode != 'no_write': output.copy_(measured())
            if mode == 'wrong_replay': output.fill_(-1)
            if mode == 'mutate_replay': cu[0] += 1
            return output
        timed_run.outputs, timed_run.rerun = output, replay
        return .125, {'benchmark_method': 'cuda_graph'}
    h._benchmark_cuda_graph_or_events = benchmark
    checks.install(h)
    rows = h.run_performance()
    assert len(rows) == len(h.TEST_SHAPES) == 5
    assert options == [dict(warmup=10, repetition=100)] * 5
    for row, (batch, tpr) in zip(rows, h.TEST_SHAPES):
        assert row['params'] == dict(batch_size=batch, tokens_per_req=tpr)
        assert row['execution_time_ms'] == (.125 if mode == 'correct' else -1.)
        if mode == 'correct': assert row['perturbed_input_replay_checked']
    for values, saved in zip(inputs, pristine): checks.unchanged(values, saved)
    assert mod.expand_batch_to_tokens is _expand_cpu
    assert h._benchmark_cuda_graph_or_events is benchmark


def test_expand_adapter_installs_checks(monkeypatch):
    adapter = module_at(ROOT/'tasks/triton2triton/vllm/triton_expand/_arena_eval.py', monkeypatch)
    h = adapter.load_harness()
    assert h.run_correctness.__module__ == h.run_performance.__module__ == '_expand_checks'


def _num_nans_cpu(logits):
    return (logits != logits).sum(-1).to(torch.int32)


@pytest.mark.parametrize('mode', ['correct', 'dtype', 'shape', 'nonfinite_as_nan', 'mutate_source', 'change_nan_payload'])
def test_num_nans_known_answers_and_pristine_nan_payloads(monkeypatch, mode):
    task = ROOT/'tasks/triton2triton/vllm/triton_num_nans'
    h = module_at(task/'scripts/task_runner.py', monkeypatch)
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    x = torch.tensor([[float('nan'), float('inf'), -float('inf')], [0., 1., -1.]], dtype=torch.float16)
    known = torch.tensor([1, 0], dtype=torch.int32)
    torch.testing.assert_close(h.reference_num_nans(x), known, atol=0, rtol=0)
    torch.testing.assert_close(_num_nans_cpu(x), known, atol=0, rtol=0)
    checks.unchanged(x, x.clone())
    def candidate(logits):
        if mode == 'mutate_source': logits.nan_to_num_()
        if mode == 'change_nan_payload':
            raw = logits.view(torch.int16)
            raw[logits != logits] = 0x7e01
        out = (~torch.isfinite(logits)).sum(-1).int() if mode == 'nonfinite_as_nan' else _num_nans_cpu(logits)
        if mode == 'dtype': out = out.long()
        if mode == 'shape': out = out[:1]
        return out
    mod = SimpleNamespace(get_num_nans=candidate)
    h.load_module = lambda: mod
    with checks.checked_modules(h):
        checked = h.load_module().get_num_nans
        if mode == 'correct': torch.testing.assert_close(checked(x), known, atol=0, rtol=0)
        else:
            with pytest.raises(AssertionError): checked(x)
    assert mod.get_num_nans is candidate


@pytest.mark.parametrize('mode', ['correct', 'wrong_timed', 'stale', 'no_write', 'wrong_replay',
                                 'mutate_timed', 'mutate_replay', 'raise_replay'])
def test_num_nans_actual_finite_timing_and_nan_replay(monkeypatch, mode):
    import inspect
    task = ROOT/'tasks/triton2triton/vllm/triton_num_nans'
    h = module_at(task/'scripts/task_runner.py', monkeypatch)
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    h._TimedRun = SimpleNamespace
    factory = torch.randn
    monkeypatch.setattr(torch, 'randn', lambda *args, **kwargs: factory(*args, **{**kwargs, 'device': 'cpu'}))
    inputs, pristine, options = [], [], []
    mod = SimpleNamespace(get_num_nans=_num_nans_cpu)
    h.load_module = lambda: mod
    def benchmark(measured, *, timed_run, **kwargs):
        options.append(kwargs)
        fn = inspect.getclosurevars(measured).nonlocals['fn']
        logits = inspect.getclosurevars(fn).nonlocals['logits']
        assert torch.isfinite(logits).all(), 'Original finite scored inputs are retained'
        inputs.append(logits); pristine.append(logits.clone())
        output = measured(); cached = output.clone()
        if mode == 'wrong_timed': output.fill_(-1)
        if mode == 'mutate_timed': logits.zero_()
        def replay():
            if mode == 'raise_replay': raise RuntimeError('replay failed')
            if mode == 'stale': output.copy_(cached)
            elif mode != 'no_write': output.copy_(measured())
            if mode == 'wrong_replay': output.zero_()
            if mode == 'mutate_replay': logits.nan_to_num_()
            return output
        timed_run.outputs, timed_run.rerun = output, replay
        return .125, {'benchmark_method': 'cuda_graph'}
    h._benchmark_cuda_graph_or_events = benchmark
    checks.install(h)
    rows = h.run_performance()
    assert len(rows) == len(h.TEST_SHAPES) == 5
    assert options == [dict(warmup=10, repetition=100)] * 5
    for row, (reqs, vocab) in zip(rows, h.TEST_SHAPES):
        assert row['params'] == dict(num_reqs=reqs, vocab_size=vocab)
        assert row['execution_time_ms'] == (.125 if mode == 'correct' else -1.)
        if mode == 'correct': assert row['perturbed_input_replay_checked']
    for x, saved in zip(inputs, pristine): checks.unchanged(x, saved)
    assert mod.get_num_nans is _num_nans_cpu
    assert h._benchmark_cuda_graph_or_events is benchmark


def test_num_nans_adapter_installs_checks(monkeypatch):
    adapter = module_at(ROOT/'tasks/triton2triton/vllm/triton_num_nans/_arena_eval.py', monkeypatch)
    h = adapter.load_harness()
    assert h.run_correctness.__module__ == h.run_performance.__module__ == '_nancount_checks'


def _mean_cpu(x, dim, keepdim=False, dtype=None):
    dtype = dtype or (x.dtype if x.is_floating_point() else torch.float32)
    return (x.to(dtype).float().sum(dim=dim, keepdim=keepdim) / x.shape[dim]).to(dtype)


@pytest.mark.parametrize('mode', ['correct', 'dtype', 'shape', 'nonfinite', 'mutate_source',
                                 'ignores_keepdim', 'ignores_dtype', 'ignores_negative_dim'])
def test_mean_known_answer_optional_arguments_and_negative_controls(monkeypatch, mode):
    task = ROOT/'tasks/triton2triton/vllm/triton_mean'
    h = module_at(task/'scripts/task_runner.py', monkeypatch)
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    x = torch.arange(30, dtype=torch.float16).reshape(2, 3, 5)
    expected = torch.tensor([[[2.], [7.], [12.]], [[17.], [22.], [27.]]])
    torch.testing.assert_close(checks.reference(x, -1, True, torch.float32), expected, atol=0, rtol=0)
    torch.testing.assert_close(_mean_cpu(x, -1, True, torch.float32), expected, atol=0, rtol=0)
    def candidate(x, dim, keepdim=False, dtype=None):
        if mode == 'mutate_source': x.zero_()
        if mode == 'ignores_keepdim': keepdim = False
        if mode == 'ignores_dtype': dtype = None
        if mode == 'ignores_negative_dim' and dim < 0: dim = 0
        output = _mean_cpu(x, dim, keepdim, dtype)
        if mode == 'dtype': output = output.double()
        if mode == 'shape': output = output[:1]
        if mode == 'nonfinite': output.fill_(float('nan'))
        return output
    mod = SimpleNamespace(mean_dim=candidate)
    h.load_module = lambda: mod
    with checks.checked_modules(h):
        call = h.load_module().mean_dim
        if mode == 'correct': torch.testing.assert_close(call(x, -1, True, torch.float32), expected)
        else:
            with pytest.raises(AssertionError): call(x, -1, True, torch.float32)
    assert mod.mean_dim is candidate


@pytest.mark.parametrize('mode', ['correct', 'wrong_timed', 'stale', 'no_write', 'wrong_replay',
                                 'mutate_timed', 'mutate_replay', 'raise_replay'])
def test_mean_original_scored_performance_and_actual_replay(monkeypatch, mode):
    import inspect
    task = ROOT/'tasks/triton2triton/vllm/triton_mean'
    h = module_at(task/'scripts/task_runner.py', monkeypatch)
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    h._TimedRun = SimpleNamespace
    factory = torch.randn
    monkeypatch.setattr(torch, 'randn', lambda *args, **kwargs: factory(*args, **{**kwargs, 'device': 'cpu'}))
    inputs, pristine, options = [], [], []
    mod = SimpleNamespace(mean_dim=_mean_cpu)
    h.load_module = lambda: mod
    def benchmark(measured, *, timed_run, **kwargs):
        options.append(kwargs)
        fn = inspect.getclosurevars(measured).nonlocals['fn']
        x = inspect.getclosurevars(fn).nonlocals['x']
        inputs.append(x); pristine.append(x.clone())
        output = measured(); cached = output.clone()
        if mode == 'wrong_timed': output.fill_(100)
        if mode == 'mutate_timed': x.zero_()
        def replay():
            if mode == 'raise_replay': raise RuntimeError('replay failed')
            if mode == 'stale': output.copy_(cached)
            elif mode != 'no_write': output.copy_(measured())
            if mode == 'wrong_replay': output.fill_(100)
            if mode == 'mutate_replay': x.zero_()
            return output
        timed_run.outputs, timed_run.rerun = output, replay
        return .125, {'benchmark_method': 'cuda_graph'}
    h._benchmark_cuda_graph_or_events = benchmark
    checks.install(h)
    rows = h.run_performance()
    assert len(rows) == len(h.TEST_SHAPES) == 5
    assert options == [dict(warmup=10, repetition=100)] * 5
    for row, (shape, dim) in zip(rows, h.TEST_SHAPES):
        assert row['params'] == dict(shape=list(shape), dim=dim)
        assert row['execution_time_ms'] == (.125 if mode == 'correct' else -1.)
        if mode == 'correct': assert row['perturbed_input_replay_checked']
    for x, saved in zip(inputs, pristine): checks.unchanged(x, saved)
    assert mod.mean_dim is _mean_cpu
    assert h._benchmark_cuda_graph_or_events is benchmark


def test_mean_adapter_installs_checks(monkeypatch):
    adapter = module_at(ROOT/'tasks/triton2triton/vllm/triton_mean/_arena_eval.py', monkeypatch)
    h = adapter.load_harness()
    assert h.run_correctness.__module__ == h.run_performance.__module__ == '_mean_checks'


def _sampled_counts_cpu(num_sampled, seq, cu, mapping, prefill):
    chunked = seq < prefill[mapping.long()]
    samples = torch.where(chunked, 0, num_sampled)
    rejected = torch.where(chunked, 0, cu[1:] - cu[:-1] - samples)
    num_sampled.copy_(samples)
    return num_sampled, rejected


@pytest.mark.parametrize('mode', ['correct', 'dtype', 'shape', 'wrong_second', 'skip_in_place',
                                 'identity_mapping', 'inclusive_threshold', 'mutate_readonly'])
def test_sampled_count_ragged_mapping_and_in_place_known_answers(monkeypatch, mode):
    task = ROOT/'tasks/triton2triton/vllm/triton_get_num_sampled_and_rejected'
    h = module_at(task/'scripts/task_runner.py', monkeypatch)
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    inputs = checks.diagnostic_inputs('cpu')
    expected = torch.tensor([1, 0, 0, 3], dtype=torch.int32), torch.tensor([1, 0, 0, 1], dtype=torch.int32)
    for value, known in zip(checks.reference(h, inputs), expected):
        torch.testing.assert_close(value, known, atol=0, rtol=0)
    def candidate(ns, seq, cu, mapping, prefill):
        if mode == 'mutate_readonly': seq.zero_()
        if mode == 'identity_mapping': mapping = torch.arange(len(mapping)).int()
        if mode == 'inclusive_threshold': seq = seq-1
        result = list(_sampled_counts_cpu(ns.clone() if mode == 'skip_in_place' else ns, seq, cu, mapping, prefill))
        if mode == 'dtype': result[1] = result[1].long()
        if mode == 'shape': result[1] = result[1][:1]
        if mode == 'wrong_second': result[1].fill_(-1)
        return tuple(result)
    mod = SimpleNamespace(get_num_sampled_and_rejected=candidate)
    h.load_module = lambda: mod
    with checks.checked_modules(h):
        call = h.load_module().get_num_sampled_and_rejected
        if mode == 'correct': checks.check_outputs(call(*inputs), expected, inputs[0])
        else:
            with pytest.raises(AssertionError): call(*inputs)
    assert mod.get_num_sampled_and_rejected is candidate


@pytest.mark.parametrize('mode', ['correct', 'wrong_timed', 'stale', 'no_write', 'wrong_replay',
                                 'mutate_seed', 'mutate_replay', 'raise_replay'])
def test_sampled_count_original_prepared_timing_and_replay_restoration(monkeypatch, mode):
    import inspect
    task = ROOT/'tasks/triton2triton/vllm/triton_get_num_sampled_and_rejected'
    h = module_at(task/'scripts/task_runner.py', monkeypatch)
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    h._TimedRun = SimpleNamespace
    for name in ('arange', 'randint', 'full', 'zeros'):
        factory = getattr(torch, name)
        def cpu_factory(*args, _factory=factory, **kwargs):
            return _factory(*args, **{**kwargs, 'device': 'cpu'})
        monkeypatch.setattr(torch, name, cpu_factory)
    all_inputs, pristine, options = [], [], []
    mod = SimpleNamespace(get_num_sampled_and_rejected=_sampled_counts_cpu)
    h.load_module = lambda: mod
    def benchmark(fn, *, timed_run, **kwargs):
        options.append({key: val for key, val in kwargs.items() if key != 'prepare_fn'})
        closed = inspect.getclosurevars(fn).nonlocals
        prepare = kwargs['prepare_fn']
        prepared = inspect.getclosurevars(prepare).nonlocals
        assert prepared['num_sampled_work'] is closed['num_sampled_work']
        seed = prepared['num_sampled']
        inputs = (seed, closed['seq_lens'], closed['cu_num_logits'], closed['idx_mapping'],
                  closed['prefill_len'], closed['num_sampled_work'])
        all_inputs.append(inputs); pristine.append(checks.snapshots(inputs))
        prepare()
        outputs = fn(); cached = checks.snapshots(outputs)
        if mode == 'wrong_timed': outputs[1].fill_(-1)
        if mode == 'mutate_seed': seed.zero_()
        def replay():
            if mode == 'raise_replay': raise RuntimeError('replay failed')
            prepare()
            if mode == 'stale':
                for out, saved in zip(outputs, cached): out.copy_(saved)
            elif mode != 'no_write':
                computed = fn()
                for out, value in zip(outputs, computed): out.copy_(value)
            if mode == 'wrong_replay': outputs[1].fill_(-1)
            if mode == 'mutate_replay': closed['seq_lens'].zero_()
            return outputs
        timed_run.outputs, timed_run.rerun = outputs, replay
        return .125, {'benchmark_method': 'cuda_graph'}
    h._benchmark_cuda_graph_or_events = benchmark
    checks.install(h)
    rows = h.run_performance()
    assert len(rows) == len(h.TEST_SHAPES) == 5
    assert options == [dict(warmup=10, repetition=100, target_ms=20.)] * 5
    for row, (reqs, spec) in zip(rows, h.TEST_SHAPES):
        assert row['params'] == dict(num_reqs=reqs, num_spec_steps=spec)
        assert row['execution_time_ms'] == (.125 if mode == 'correct' else -1.)
        if mode == 'correct': assert row['in_place_sampled_state_checked'] and row['perturbed_input_replay_checked']
    for inputs, saved in zip(all_inputs, pristine): checks.unchanged(inputs, saved)
    assert mod.get_num_sampled_and_rejected is _sampled_counts_cpu
    assert h._benchmark_cuda_graph_or_events is benchmark


def test_sampled_count_adapter_installs_checks(monkeypatch):
    adapter = module_at(ROOT/'tasks/triton2triton/vllm/triton_get_num_sampled_and_rejected/_arena_eval.py', monkeypatch)
    h = adapter.load_harness()
    assert h.run_correctness.__module__ == h.run_performance.__module__ == '_sampled_checks'


def _log_softmax_cpu(x, dim=-1):
    values = x.float()
    return (values - torch.logsumexp(values, dim=dim, keepdim=True)).to(x.dtype)


def test_log_softmax_equal_and_extreme_known_answers(monkeypatch):
    import math
    checks = module_at(ROOT/'tasks/triton2triton/vllm/triton_log_softmax/_arena_checks.py', monkeypatch)
    x = torch.tensor([[0., 0., 0.], [1000., 0., -1000.]], dtype=torch.float16)
    expected = torch.tensor([[-math.log(3)]*3, [0., -1000., -2000.]], dtype=torch.float16)
    torch.testing.assert_close(checks.reference(x), expected, atol=0, rtol=0)
    torch.testing.assert_close(_log_softmax_cpu(x), expected, atol=0, rtol=0)


@pytest.mark.parametrize('mode', ['correct', 'dtype', 'shape', 'nonfinite', 'mutate_source', 'fixed_rank', 'no_stability'])
def test_log_softmax_actual_correctness_tail_and_stability(monkeypatch, mode):
    task = ROOT/'tasks/triton2triton/vllm/triton_log_softmax'
    h = module_at(task/'scripts/task_runner.py', monkeypatch)
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    factory = torch.randn
    monkeypatch.setattr(torch, 'randn', lambda *args, **kwargs: factory(*args, **{**kwargs, 'device': 'cpu'}))
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    calls = []
    def candidate(x, dim=-1):
        calls.append((tuple(x.shape), dim))
        if mode == 'mutate_source': x.zero_()
        out = _log_softmax_cpu(x, dim)
        if mode == 'no_stability': out = (x.float()-x.float().exp().sum(dim, keepdim=True).log()).to(x.dtype)
        if mode == 'dtype': out = out.float()
        if mode == 'shape': out = out[..., :1]
        if mode == 'nonfinite': out.fill_(float('nan'))
        if mode == 'fixed_rank' and x.ndim != 2: out.zero_()
        return out
    mod = SimpleNamespace(log_softmax=candidate)
    h.load_module = lambda: mod
    checks.install(h)
    ok, reason = h.run_correctness()
    assert ok is (mode == 'correct'), reason
    if mode == 'correct':
        assert calls[::2] == [((2, 3, 7), 2)] * 5
        assert calls[1::2] == [(tuple(shape), -1) for shape in h.TEST_SHAPES]
    assert mod.log_softmax is candidate


@pytest.mark.parametrize('mode', ['correct', 'wrong_timed', 'stale', 'no_write', 'wrong_replay',
                                 'mutate_timed', 'mutate_replay', 'raise_replay'])
def test_log_softmax_actual_scored_performance_and_replay(monkeypatch, mode):
    import inspect
    task = ROOT/'tasks/triton2triton/vllm/triton_log_softmax'
    h = module_at(task/'scripts/task_runner.py', monkeypatch)
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    h._TimedRun = SimpleNamespace
    factory = torch.randn
    monkeypatch.setattr(torch, 'randn', lambda *args, **kwargs: factory(*args, **{**kwargs, 'device': 'cpu'}))
    inputs, pristine, options = [], [], []
    mod = SimpleNamespace(log_softmax=_log_softmax_cpu)
    h.load_module = lambda: mod
    def benchmark(measured, *, timed_run, **kwargs):
        options.append(kwargs)
        fn = inspect.getclosurevars(measured).nonlocals['fn']
        x = inspect.getclosurevars(fn).nonlocals['x']
        inputs.append(x); pristine.append(x.clone())
        output = measured(); cached = output.clone()
        if mode == 'wrong_timed': output.zero_()
        if mode == 'mutate_timed': x.zero_()
        def replay():
            if mode == 'raise_replay': raise RuntimeError('replay failed')
            if mode == 'stale': output.copy_(cached)
            elif mode != 'no_write': output.copy_(measured())
            if mode == 'wrong_replay': output.zero_()
            if mode == 'mutate_replay': x.zero_()
            return output
        timed_run.outputs, timed_run.rerun = output, replay
        return .125, {'benchmark_method': 'cuda_graph'}
    h._benchmark_cuda_graph_or_events = benchmark
    checks.install(h)
    rows = h.run_performance()
    assert len(rows) == len(h.TEST_SHAPES) == 5
    assert options == [dict(warmup=10, repetition=100)] * 5
    for row, (rows_count, cols_count) in zip(rows, h.TEST_SHAPES):
        assert row['params'] == dict(rows=rows_count, cols=cols_count)
        assert row['execution_time_ms'] == (.125 if mode == 'correct' else -1.)
        if mode == 'correct': assert row['perturbed_input_replay_checked']
    for x, saved in zip(inputs, pristine): checks.unchanged(x, saved)
    assert mod.log_softmax is _log_softmax_cpu
    assert h._benchmark_cuda_graph_or_events is benchmark


def test_log_softmax_adapter_installs_checks(monkeypatch):
    adapter = module_at(ROOT/'tasks/triton2triton/vllm/triton_log_softmax/_arena_eval.py', monkeypatch)
    h = adapter.load_harness()
    assert h.run_correctness.__module__ == h.run_performance.__module__ == '_logsoftmax_checks'


def _bitmatrix_cpu(ids, experts):
    rows = []
    for assignments in ids.tolist():
        rows.append([sum(1 << bit for bit in {eid % 32 for eid in assignments if eid // 32 == col})
                     for col in range((experts+31)//32)])
    return torch.tensor(rows, dtype=torch.uint32, device=ids.device)


def _evaluate_pack_kernel_bit_expression(source, ids, experts):
    """Execute the actual packing expression with CPU Triton arithmetic stand-ins."""
    node = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'x' for t in n.targets))
    indices = torch.full((len(ids), 32), -1, dtype=torch.int64)
    indices[:, :ids.shape[1]] = ids.long()
    mask = torch.arange(32)[None, :].expand_as(indices) < ids.shape[1]
    class UInt32One:
        def __lshift__(self, shift):
            return torch.ones_like(shift) << (shift & 31)
    columns = []
    for col in range((experts+31)//32):
        scope = dict(tl=SimpleNamespace(where=torch.where), indices=indices, mask=mask,
                     div=torch.div(indices, 32, rounding_mode='trunc'), rem=torch.fmod(indices, 32),
                     one=UInt32One(), offs=torch.tensor([col]))
        values = eval(compile(ast.Expression(node.value), '<actual packing expression>', 'eval'), scope)
        packed = torch.zeros(len(ids), dtype=torch.int64)
        for k in range(32): packed |= values[:, k, 0]
        columns.append(packed)
    return torch.stack(columns, dim=1).to(torch.uint32)


@pytest.mark.parametrize('experts,assignments,expected', [
    (8, [[0, 1], [2, 7]], [[3], [132]]),
    (64, [[0, 1], [31, 32], [63, 63]], [[3, 0], [2147483648, 1], [0, 2147483648]]),
    (33, [[0, 32], [31, 31]], [[1, 1], [2147483648, 0]]),
])
def test_pack_bitmatrix_real_membership_known_answers_reject_padding_bit(monkeypatch, experts, assignments, expected):
    task = ROOT/'tasks/triton2triton/vllm/triton_pack_bitmatrix'
    h = module_at(task/'scripts/task_runner.py', monkeypatch)
    ids = torch.tensor(assignments, dtype=torch.int16)
    known = torch.tensor(expected, dtype=torch.uint32)
    current = (task/'source/triton_pack_bitmatrix.py').read_text()
    old_source = subprocess.check_output(['git', 'show', f'{BASE}:{(task/"source/triton_pack_bitmatrix.py").relative_to(ROOT)}'], cwd=ROOT, text=True)
    for output in (h.reference_pack_bitmatrix(ids, experts), _bitmatrix_cpu(ids, experts),
                   _evaluate_pack_kernel_bit_expression(current, ids, experts)):
        torch.testing.assert_close(output, known, atol=0, rtol=0)
    original_harness = subprocess.check_output(['git', 'show', f'{BASE}:{(task/"scripts/task_runner.py").relative_to(ROOT)}'], cwd=ROOT, text=True)
    reference_node = next(n for n in ast.parse(original_harness).body
                          if isinstance(n, ast.FunctionDef) and n.name == 'reference_pack_bitmatrix')
    scope = {}
    exec(compile(ast.Module(body=[reference_node], type_ignores=[]), '<old reference>', 'exec'), scope)
    old_result = scope['reference_pack_bitmatrix'](ids, experts)
    torch.testing.assert_close(_evaluate_pack_kernel_bit_expression(old_source, ids, experts), old_result, atol=0, rtol=0)
    assert not torch.equal(old_result, known), 'Old reference accepted a nonexistent expert assignment'


@pytest.mark.parametrize('mode', ['correct', 'forced_padding_bit', 'dtype', 'shape', 'mutate_source'])
def test_pack_bitmatrix_output_contract_and_old_bug_negative_control(monkeypatch, mode):
    task = ROOT/'tasks/triton2triton/vllm/triton_pack_bitmatrix'
    h = module_at(task/'scripts/task_runner.py', monkeypatch)
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    ids = torch.tensor([[0, 1], [2, 7]], dtype=torch.int16)
    expected = torch.tensor([[3], [132]], dtype=torch.uint32)
    def candidate(ids, experts):
        if mode == 'mutate_source': ids.zero_()
        out = _bitmatrix_cpu(ids, experts)
        if mode == 'forced_padding_bit':
            signed = out.view(torch.int32)
            signed[:, 0] |= -2147483648
        if mode == 'dtype': out = out.long()
        if mode == 'shape': out = out[:1]
        return out
    mod = SimpleNamespace(pack_topk_to_bitmatrix=candidate)
    h.load_module = lambda: mod
    with checks.checked_modules(h):
        call = h.load_module().pack_topk_to_bitmatrix
        if mode == 'correct': torch.testing.assert_close(call(ids, 8), expected, atol=0, rtol=0)
        else:
            with pytest.raises(AssertionError): call(ids, 8)
    assert mod.pack_topk_to_bitmatrix is candidate


@pytest.mark.parametrize('mode', ['correct', 'wrong_timed', 'stale', 'no_write', 'wrong_replay',
                                 'mutate_timed', 'mutate_replay', 'raise_replay'])
def test_pack_bitmatrix_original_performance_and_captured_replay(monkeypatch, mode):
    import inspect
    task = ROOT/'tasks/triton2triton/vllm/triton_pack_bitmatrix'
    h = module_at(task/'scripts/task_runner.py', monkeypatch)
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    h._TimedRun = SimpleNamespace
    factory = torch.randint
    monkeypatch.setattr(torch, 'randint', lambda *args, **kwargs: factory(*args, **{**kwargs, 'device': 'cpu'}))
    inputs, pristine, options = [], [], []
    mod = SimpleNamespace(pack_topk_to_bitmatrix=_bitmatrix_cpu)
    h.load_module = lambda: mod
    def benchmark(measured, *, timed_run, **kwargs):
        options.append(kwargs)
        fn = inspect.getclosurevars(measured).nonlocals['fn']
        ids = inspect.getclosurevars(fn).nonlocals['topk_ids']
        inputs.append(ids); pristine.append(ids.clone())
        output = measured(); cached = output.clone()
        if mode == 'wrong_timed': output.zero_()
        if mode == 'mutate_timed': ids.zero_()
        def replay():
            if mode == 'raise_replay': raise RuntimeError('replay failed')
            if mode == 'stale': output.copy_(cached)
            elif mode != 'no_write': output.copy_(measured())
            if mode == 'wrong_replay': output.zero_()
            if mode == 'mutate_replay': ids.zero_()
            return output
        timed_run.outputs, timed_run.rerun = output, replay
        return .125, {'benchmark_method': 'cuda_graph'}
    h._benchmark_cuda_graph_or_events = benchmark
    checks.install(h)
    rows = h.run_performance()
    assert len(rows) == len(h.TEST_SHAPES) == 5
    assert options == [dict(warmup=10, repetition=100)] * 5
    for row, (rows_count, experts, topk) in zip(rows, h.TEST_SHAPES):
        assert row['params'] == dict(n_rows=rows_count, num_experts=experts, topk=topk)
        assert row['execution_time_ms'] == (.125 if mode == 'correct' else -1.)
        if mode == 'correct': assert row['perturbed_input_replay_checked']
    for ids, saved in zip(inputs, pristine): checks.unchanged(ids, saved)
    assert mod.pack_topk_to_bitmatrix is _bitmatrix_cpu
    assert h._benchmark_cuda_graph_or_events is benchmark


def test_pack_bitmatrix_adapter_installs_checks(monkeypatch):
    adapter = module_at(ROOT/'tasks/triton2triton/vllm/triton_pack_bitmatrix/_arena_eval.py', monkeypatch)
    h = adapter.load_harness()
    assert h.run_correctness.__module__ == h.run_performance.__module__ == '_bitmatrix_checks'


def _fla_norm_cpu(x, weight, bias=None, eps=1e-5, z=None, norm_before_gate=True, is_rms_norm=False):
    value = x if z is None or norm_before_gate else x * torch.nn.functional.silu(z)
    if is_rms_norm:
        mean = None
        rstd = (torch.linalg.vector_norm(value, dim=-1).square()/value.shape[-1] + eps).rsqrt()
        y = value * rstd[:, None] * weight
        if bias is not None: y = y + bias
    else:
        variance, mean = torch.var_mean(value, dim=-1, unbiased=False)
        rstd = (variance + eps).rsqrt()
        y = torch.nn.functional.layer_norm(value, (value.shape[-1],), weight, bias, eps)
    if z is not None and norm_before_gate: y = y * torch.nn.functional.silu(z)
    return y, mean, rstd


@pytest.mark.parametrize('rms', [False, True])
@pytest.mark.parametrize('before_gate', [False, True])
def test_fla_norm_independent_full_tuple_known_answer(monkeypatch, rms, before_gate):
    import math
    task = ROOT/'tasks/triton2triton/vllm/triton_fla_layernorm'
    h = module_at(task/'scripts/task_runner.py', monkeypatch)
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    x, weight, bias, z = torch.tensor([[1., 3.]]), torch.tensor([2., -1.]), torch.tensor([.5, 2.]), torch.ones(1, 2)
    q = 1/(1+math.exp(-1))
    factor = 1 if before_gate else q
    scale = ((5 if rms else 1)*factor**2 + 1e-3)**-0.5
    normalized = torch.tensor([[1., 3.] if rms else [-1., 1.]]) * factor * scale
    y = normalized * weight + bias
    if before_gate: y *= q
    wanted = y, None if rms else torch.tensor([2*factor], dtype=torch.float32), torch.tensor([scale])
    options = dict(eps=1e-3, is_rms_norm=rms, norm_before_gate=before_gate)
    checks.check_outputs(checks.reference(h, (x, weight, bias, z), options), wanted)
    checks.check_outputs(_fla_norm_cpu(x, weight, bias, z=z, **options), wanted)


@pytest.mark.parametrize('mode', ['correct', 'dtype', 'stat_dtype', 'shape', 'nonfinite',
                                 'missing_stats', 'wrong_mean', 'wrong_rstd', 'mutate_input',
                                 'ignore_weight', 'ignore_bias', 'ignore_gate_order'])
def test_fla_norm_actual_correctness_affine_gate_and_outputs(monkeypatch, mode):
    task = ROOT/'tasks/triton2triton/vllm/triton_fla_layernorm'
    h = module_at(task/'scripts/task_runner.py', monkeypatch)
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    generator = h.gen_inputs
    h.gen_inputs = lambda seed, case_idx, device: generator(seed, case_idx, 'cpu')
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    calls = []
    def candidate(x, w, b=None, **kwargs):
        calls.append((tuple(x.shape), kwargs['is_rms_norm'], kwargs['norm_before_gate']))
        if mode == 'mutate_input': x.zero_()
        if mode == 'ignore_weight': w = torch.ones_like(w)
        if mode == 'ignore_bias': b = None
        if mode == 'ignore_gate_order': kwargs['norm_before_gate'] = True
        outputs = list(_fla_norm_cpu(x, w, b, **kwargs))
        if mode == 'dtype': outputs[0] = outputs[0].double()
        if mode == 'stat_dtype': outputs[2] = outputs[2].double()
        if mode == 'shape': outputs[0] = outputs[0][:1]
        if mode == 'nonfinite': outputs[2].fill_(float('inf'))
        if mode == 'missing_stats': outputs = outputs[:1]
        if mode == 'wrong_mean': outputs[1] = torch.full_like(outputs[2], 17.)
        if mode == 'wrong_rstd': outputs[2].zero_()
        return tuple(outputs)
    mod = SimpleNamespace(layer_norm_fwd=candidate)
    h.load_module = lambda: mod
    checks.install(h)
    ok, reason = h.run_correctness()
    assert ok is (mode == 'correct'), reason
    if mode == 'correct':
        assert [c for c in calls if c[0] != (2, 17)] == [((case[0], case[1]), case[5], case[4]) for case in h.TEST_CASES]
        assert [c[1:] for c in calls if c[0] == (2, 17)] == [(rms, before) for rms in (False, True) for before in (False, True)]
    assert mod.layer_norm_fwd is candidate


@pytest.mark.parametrize('mode', ['correct', 'wrong_timed', 'stale', 'no_write', 'omit_mean',
                                 'omit_rstd', 'wrong_replay', 'mutate_timed', 'mutate_replay', 'raise_replay'])
def test_fla_norm_actual_performance_affine_replay_restores_inputs(monkeypatch, mode):
    task = ROOT/'tasks/triton2triton/vllm/triton_fla_layernorm'
    h = module_at(task/'scripts/task_runner.py', monkeypatch)
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    h._TimedRun = SimpleNamespace
    generator = h.gen_inputs
    inputs, pristine, options = [], [], []
    def gen_inputs(seed, case_idx, device):
        args, kwargs = generator(seed, case_idx, 'cpu')
        values = (*args, kwargs.get('z'))
        assert torch.equal(args[1], torch.ones_like(args[1]))
        assert args[2] is None or torch.equal(args[2], torch.zeros_like(args[2]))
        inputs.append(values); pristine.append(checks.snapshots(values))
        return args, kwargs
    h.gen_inputs = gen_inputs
    mod = SimpleNamespace(layer_norm_fwd=_fla_norm_cpu)
    h.load_module = lambda: mod
    def benchmark(measured, *, timed_run, **kwargs):
        options.append(kwargs)
        outputs = measured(); cached = checks.snapshots(outputs)
        if mode == 'wrong_timed': outputs[2].zero_()
        if mode == 'mutate_timed': inputs[-1][1].zero_()
        def replay():
            if mode == 'raise_replay': raise RuntimeError('replay failed')
            if mode != 'no_write':
                computed = cached if mode == 'stale' else measured()
                for i, (value, calculated) in enumerate(zip(outputs, computed)):
                    if value is not None and not (mode == 'omit_mean' and i == 1 or mode == 'omit_rstd' and i == 2):
                        value.copy_(calculated)
            if mode == 'wrong_replay': outputs[2].zero_()
            if mode == 'mutate_replay': inputs[-1][0].zero_()
            return outputs
        timed_run.outputs, timed_run.rerun = outputs, replay
        return 0.125, {'benchmark_method': 'cuda_graph'}
    h._benchmark_cuda_graph_or_events = benchmark
    checks.install(h)
    rows = h.run_performance()
    assert len(rows) == 5
    assert options == [dict(warmup=10, repetition=100)] * 5
    for case, row in zip(h.TEST_CASES, rows):
        success = mode == 'correct' or (mode == 'omit_mean' and case[5])
        assert row['execution_time_ms'] == (0.125 if success else -1.)
        if success: assert row['perturbed_input_replay_checked']
    for values, saved in zip(inputs, pristine): checks.unchanged(values, saved)
    assert mod.layer_norm_fwd is _fla_norm_cpu
    assert h._benchmark_cuda_graph_or_events is benchmark


def test_fla_norm_adapter_installs_full_contract_checks(monkeypatch):
    adapter = module_at(ROOT/'tasks/triton2triton/vllm/triton_fla_layernorm/_arena_eval.py', monkeypatch)
    h = adapter.load_harness()
    assert h.run_correctness.__module__ == h.run_performance.__module__ == '_fla_layernorm_checks'


def _gdn_gate_cpu(A_log, a, b, dt_bias, beta=1., threshold=20.):
    x = a.double() + dt_bias.double()
    softplus = torch.where(beta*x > threshold, x, torch.logaddexp(torch.zeros_like(x), beta*x)/beta)
    g = (-A_log.double().exp() * softplus).float().unsqueeze(0)
    gate = (1/(1+(-b.double()).exp())).to(b.dtype).unsqueeze(0)
    return g, gate


def test_gdn_gate_independent_known_threshold_answer(monkeypatch):
    import math
    task = ROOT/'tasks/triton2triton/vllm/triton_fused_gdn_gating'
    h = module_at(task/'scripts/task_runner.py', monkeypatch)
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    inputs = torch.zeros(3), torch.tensor([[-4., 0., 6.]], dtype=torch.float16), torch.zeros(1, 3, dtype=torch.float16), torch.zeros(3)
    expected = (torch.tensor([[[-2*math.log1p(math.exp(-2)), -2*math.log(2), -6.]]]),
                torch.full((1, 1, 3), .5, dtype=torch.float16))
    options = dict(beta=.5, threshold=2.)
    checks.check_outputs(checks.reference(h, inputs, options), expected)
    checks.check_outputs(_gdn_gate_cpu(*inputs, **options), expected)


@pytest.mark.parametrize('mode', ['correct', 'g_dtype', 'beta_dtype', 'device', 'shape', 'nonfinite',
                                 'missing_beta', 'wrong_g', 'wrong_beta', 'mutate_input',
                                 'ignore_beta', 'ignore_threshold', 'ignore_tail'])
def test_gdn_gate_actual_correctness_complete_contract(monkeypatch, mode):
    task = ROOT/'tasks/triton2triton/vllm/triton_fused_gdn_gating'
    h = module_at(task/'scripts/task_runner.py', monkeypatch)
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    generator = h.make_inputs
    h.make_inputs = lambda batch, nh, device='cpu': generator(batch, nh, 'cpu')
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    calls = []
    def candidate(A, a, b, bias, beta=1., threshold=20.):
        calls.append((tuple(a.shape), beta, threshold))
        if mode == 'mutate_input': a.zero_()
        outputs = list(_gdn_gate_cpu(A, a, b, bias,
                       1. if mode == 'ignore_beta' else beta,
                       20. if mode == 'ignore_threshold' else threshold))
        if mode == 'g_dtype': outputs[0] = outputs[0].half()
        if mode == 'beta_dtype': outputs[1] = outputs[1].float()
        if mode == 'device': outputs[1] = torch.empty_like(outputs[1], device='meta')
        if mode == 'shape': outputs[0] = outputs[0].squeeze(0)
        if mode == 'nonfinite': outputs[1].fill_(float('inf'))
        if mode == 'missing_beta': outputs = outputs[:1]
        if mode == 'wrong_g': outputs[0].zero_()
        if mode == 'wrong_beta': outputs[1].zero_()
        if mode == 'ignore_tail' and a.shape[-1] % 8: outputs[0][..., -1] = 0
        return tuple(outputs)
    mod = SimpleNamespace(fused_gdn_gating=candidate)
    h.load_module = lambda: mod
    checks.install(h)
    ok, reason = h.run_correctness()
    assert ok is (mode == 'correct'), reason
    if mode == 'correct':
        assert [v for v in calls if v[0] != (2, 11)] == [(shape, 1., 20.) for shape in h.TEST_SHAPES]
        assert [v for v in calls if v[0] == (2, 11)] == [((2, 11), .5, 2.)]
    assert mod.fused_gdn_gating is candidate


@pytest.mark.parametrize('mode', ['correct', 'wrong_timed', 'stale', 'no_write', 'omit_g',
                                 'omit_beta', 'wrong_replay', 'mutate_timed', 'mutate_replay', 'raise_replay'])
def test_gdn_gate_actual_timed_outputs_replay_and_restore(monkeypatch, mode):
    task = ROOT/'tasks/triton2triton/vllm/triton_fused_gdn_gating'
    h = module_at(task/'scripts/task_runner.py', monkeypatch)
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    h._TimedRun = SimpleNamespace
    generator = h.make_inputs
    inputs, pristine, options = [], [], []
    def make_inputs(batch, nh, device='cpu'):
        values = generator(batch, nh, 'cpu')
        assert tuple(x.dtype for x in values) == (torch.float32, torch.float16, torch.float16, torch.float32)
        inputs.append(values); pristine.append(checks.snapshots(values))
        return values
    h.make_inputs = make_inputs
    mod = SimpleNamespace(fused_gdn_gating=_gdn_gate_cpu)
    h.load_module = lambda: mod
    def benchmark(measured, *, timed_run, **kwargs):
        options.append(kwargs)
        outputs = measured(); cached = checks.snapshots(outputs)
        if mode == 'wrong_timed': outputs[1].zero_()
        if mode == 'mutate_timed': inputs[-1][0].zero_()
        def replay():
            if mode == 'raise_replay': raise RuntimeError('replay failed')
            if mode != 'no_write':
                computed = cached if mode == 'stale' else measured()
                for i, (value, calculated) in enumerate(zip(outputs, computed)):
                    if not (mode == 'omit_g' and i == 0 or mode == 'omit_beta' and i == 1):
                        value.copy_(calculated)
            if mode == 'wrong_replay': outputs[1].zero_()
            if mode == 'mutate_replay': inputs[-1][2].zero_()
            return outputs
        timed_run.outputs, timed_run.rerun = outputs, replay
        return .125, {'benchmark_method': 'cuda_graph'}
    h._benchmark_cuda_graph_or_events = benchmark
    checks.install(h)
    rows = h.run_performance()
    assert len(rows) == 5
    assert options == [dict(warmup=10, repetition=100)] * 5
    for (batch, nh), row in zip(h.TEST_SHAPES, rows):
        assert row['params'] == dict(batch=batch, num_heads=nh)
        assert row['execution_time_ms'] == (.125 if mode == 'correct' else -1.)
        if mode == 'correct': assert row['perturbed_input_replay_checked']
    for values, saved in zip(inputs, pristine): checks.unchanged(values, saved)
    assert mod.fused_gdn_gating is _gdn_gate_cpu
    assert h._benchmark_cuda_graph_or_events is benchmark


def test_gdn_gate_adapter_installs_checks(monkeypatch):
    adapter = module_at(ROOT/'tasks/triton2triton/vllm/triton_fused_gdn_gating/_arena_eval.py', monkeypatch)
    h = adapter.load_harness()
    assert h.run_correctness.__module__ == h.run_performance.__module__ == '_gdn_gate_checks'


def _grouped_norm_cpu(x, weight, bias, eps, z=None, out=None, group_size=None,
                      norm_before_gate=True, is_rms_norm=False):
    width = group_size or x.shape[-1]
    ys, means, rstds = [], [], []
    for start in range(0, x.shape[-1], width):
        part = slice(start, start+width)
        y, mean, rstd = _fla_norm_cpu(x[:, part].float(), weight[part].float(),
            None if bias is None else bias[part].float(), eps,
            None if z is None else z[:, part].float(), norm_before_gate, is_rms_norm)
        ys.append(y)
        if mean is not None: means.append(mean)
        rstds.append(rstd)
    y = torch.cat(ys, dim=-1).to(x.dtype)
    if out is not None: out.copy_(y); y = out
    return y, None if is_rms_norm else torch.cat(means), torch.cat(rstds)


@pytest.mark.parametrize('rms', [False, True])
def test_grouped_norm_independent_group_order_known_answer(monkeypatch, rms):
    task = ROOT/'tasks/triton2triton/vllm/triton_layernorm_gated'
    h = module_at(task/'scripts/task_runner.py', monkeypatch)
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    x = torch.tensor([[1., 3., 2., 6.], [2., 4., 3., 5.]], dtype=torch.float16)
    weight, bias = torch.ones(4, dtype=torch.float16), torch.zeros(4, dtype=torch.float16)
    stats = torch.tensor([5., 10., 20., 17.] if rms else [1., 1., 4., 1.])
    rstd = (stats + 1e-3).rsqrt()
    mean = None if rms else torch.tensor([2., 3., 4., 4.])
    centered = x.float() if rms else torch.tensor([[-1., 1., -2., 2.], [-1., 1., -1., 1.]])
    scale = rstd.reshape(2, 2).T.repeat_interleave(2, dim=1)
    expected = (centered*scale).half(), mean, rstd
    options = dict(eps=1e-3, group_size=2, is_rms_norm=rms)
    checks.check_outputs(checks.reference(h, (x, weight, bias, None), options), expected)
    checks.check_outputs(_grouped_norm_cpu(x, weight, bias, **options), expected)


@pytest.mark.parametrize('mode', ['correct', 'dtype', 'stat_dtype', 'shape', 'nonfinite',
                                 'missing_stats', 'wrong_mean', 'wrong_rstd', 'mutate_input',
                                 'ignore_groups', 'ignore_out', 'ignore_gate_order'])
def test_grouped_norm_actual_correctness_optional_paths(monkeypatch, mode):
    task = ROOT/'tasks/triton2triton/vllm/triton_layernorm_gated'
    h = module_at(task/'scripts/task_runner.py', monkeypatch)
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    factory = torch.randn
    monkeypatch.setattr(torch, 'randn', lambda *args, **kwargs: factory(*args, **{**kwargs, 'device': 'cpu'}))
    calls = []
    def candidate(x, w, b, eps, **kwargs):
        calls.append((tuple(x.shape), kwargs['is_rms_norm'], kwargs['norm_before_gate'], kwargs['group_size']))
        if mode == 'mutate_input': x.zero_()
        if mode == 'ignore_groups': kwargs['group_size'] = None
        if mode == 'ignore_out': kwargs['out'] = None
        if mode == 'ignore_gate_order': kwargs['norm_before_gate'] = True
        outputs = list(_grouped_norm_cpu(x, w, b, eps, **kwargs))
        if mode == 'dtype': outputs[0] = outputs[0].float()
        if mode == 'stat_dtype': outputs[2] = outputs[2].half()
        if mode == 'shape': outputs[0] = outputs[0][:1]
        if mode == 'nonfinite': outputs[2].fill_(float('inf'))
        if mode == 'missing_stats': outputs = outputs[:1]
        if mode == 'wrong_mean': outputs[1] = torch.full_like(outputs[2], 17.)
        if mode == 'wrong_rstd': outputs[2].zero_()
        return tuple(outputs)
    mod = SimpleNamespace(layer_norm_fwd=candidate)
    h.load_module = lambda: mod
    checks.install(h)
    ok, reason = h.run_correctness()
    assert ok is (mode == 'correct'), reason
    if mode == 'correct':
        assert [c for c in calls if c[0] != (2, 34)] == [((v[0], v[1]), v[2], True, None) for v in h.TEST_SHAPES]
        assert [c[1:] for c in calls if c[0] == (2, 34)] == [(rms, before, 17) for rms in (False, True) for before in (False, True)]
    assert mod.layer_norm_fwd is candidate


def test_grouped_norm_explicit_in_place_output_is_preserved(monkeypatch):
    task = ROOT/'tasks/triton2triton/vllm/triton_layernorm_gated'
    h = module_at(task/'scripts/task_runner.py', monkeypatch)
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    x = torch.tensor([[1., 3., 2., 6.]], dtype=torch.float16)
    weight, bias = torch.ones(4, dtype=torch.float16), torch.zeros(4, dtype=torch.float16)
    expected = _grouped_norm_cpu(x.clone(), weight, bias, 1e-3, group_size=2)
    mod = SimpleNamespace(layer_norm_fwd=_grouped_norm_cpu)
    h.load_module = lambda: mod
    with checks.checked_modules(h):
        result = h.load_module().layer_norm_fwd(x, weight, bias, 1e-3, out=x, group_size=2)
        assert result[0] is x
        checks.check_outputs(result, expected)


@pytest.mark.parametrize('mode', ['correct', 'wrong_timed', 'stale', 'no_write', 'omit_mean',
                                 'omit_rstd', 'wrong_replay', 'mutate_timed', 'mutate_replay', 'raise_replay'])
def test_grouped_norm_original_timing_and_full_tuple_replay(monkeypatch, mode):
    import inspect
    task = ROOT/'tasks/triton2triton/vllm/triton_layernorm_gated'
    h = module_at(task/'scripts/task_runner.py', monkeypatch)
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    h._TimedRun = SimpleNamespace
    factory = torch.randn
    monkeypatch.setattr(torch, 'randn', lambda *args, **kwargs: factory(*args, **{**kwargs, 'device': 'cpu'}))
    inputs, pristine, options = [], [], []
    mod = SimpleNamespace(layer_norm_fwd=_grouped_norm_cpu)
    h.load_module = lambda: mod
    def benchmark(measured, *, timed_run, **kwargs):
        fn = inspect.getclosurevars(measured).nonlocals['fn']
        state = inspect.getclosurevars(fn).nonlocals
        values = tuple(state[k] for k in ('x', 'w', 'b', 'z'))
        inputs.append(values); pristine.append(checks.snapshots(values)); options.append(kwargs)
        outputs = measured(); cached = checks.snapshots(outputs)
        if mode == 'wrong_timed': outputs[2].zero_()
        if mode == 'mutate_timed': values[1].zero_()
        def replay():
            if mode == 'raise_replay': raise RuntimeError('replay failed')
            if mode != 'no_write':
                computed = cached if mode == 'stale' else measured()
                for i, (value, calculated) in enumerate(zip(outputs, computed)):
                    if value is not None and not (mode == 'omit_mean' and i == 1 or mode == 'omit_rstd' and i == 2):
                        value.copy_(calculated)
            if mode == 'wrong_replay': outputs[2].zero_()
            if mode == 'mutate_replay': values[0].zero_()
            return outputs
        timed_run.outputs, timed_run.rerun = outputs, replay
        return .125, {'benchmark_method': 'cuda_graph'}
    h._benchmark_cuda_graph_or_events = benchmark
    checks.install(h)
    rows = h.run_performance()
    assert len(rows) == 5
    assert options == [dict(warmup=10, repetition=100)] * 5
    for case, row in zip(h.TEST_SHAPES, rows):
        success = mode == 'correct' or (mode == 'omit_mean' and case[2])
        assert row['execution_time_ms'] == (.125 if success else -1.)
        if success: assert row['perturbed_input_replay_checked']
    for values, saved in zip(inputs, pristine): checks.unchanged(values, saved)
    assert mod.layer_norm_fwd is _grouped_norm_cpu
    assert h._benchmark_cuda_graph_or_events is benchmark


def test_grouped_norm_adapter_installs_checks(monkeypatch):
    adapter = module_at(ROOT/'tasks/triton2triton/vllm/triton_layernorm_gated/_arena_eval.py', monkeypatch)
    h = adapter.load_harness()
    assert h.run_correctness.__module__ == h.run_performance.__module__ == '_grouped_layernorm_checks'



def _bincount_cpu(mapping, tokens, prompt, prefill, masks, counts, *, reset=True):
    if reset:
        masks[mapping.long()] = 0
        counts[mapping.long()] = 0
    for request in mapping.tolist():
        p, n = int(prompt[request]), int(prefill[request])
        for token in tokens[request, :p].unique().tolist():
            masks[request, token // 32] |= 1 << (token % 32)
        output = tokens[request, p:n].long()
        counts[request].scatter_add_(0, output, torch.ones_like(output, dtype=counts.dtype))


def test_bincount_pristine_known_answer_signed_bits_and_inactive_rows(monkeypatch):
    checks = module_at(ROOT/'tasks/triton2triton/vllm/triton_bincount/_arena_checks.py', monkeypatch)
    mapping = torch.tensor([2, 0], dtype=torch.int32)
    tokens = torch.tensor([[31, 31, 64, 0, 0], [1, 2, 3, 4, 5], [32, 64, 31, 32, 32]], dtype=torch.int32)
    prompt, prefill = torch.tensor([2, 0, 3], dtype=torch.int32), torch.tensor([5, 0, 5], dtype=torch.int32)
    masks, counts = torch.full((3, 3), 17, dtype=torch.int32), torch.full((3, 65), 9, dtype=torch.int32)
    expected_mask = torch.tensor([[-2147483648, 0, 0], [17, 17, 17], [-2147483648, 1, 1]], dtype=torch.int32)
    expected_counts = torch.zeros_like(counts)
    expected_counts[0, 0] = 2; expected_counts[0, 64] = 1
    expected_counts[1].fill_(9); expected_counts[2, 32] = 2
    expected = expected_mask, expected_counts
    checks.check_outputs(checks.reference((mapping, tokens, prompt, prefill), (masks, counts)), expected)
    _bincount_cpu(mapping, tokens, prompt, prefill, masks, counts)
    checks.check_outputs((masks, counts), expected)


@pytest.mark.parametrize('mode', ['correct', 'mutate_tokens', 'mutate_lengths', 'mutate_mapping',
                                 'identity_mapping', 'first_block_only', 'ignore_prompt', 'ignore_prefill',
                                 'wrong_mask', 'wrong_count', 'dtype', 'shape', 'clear_inactive', 'skip_reset'])
def test_bincount_actual_correctness_partial_mapping_and_launch_tail(monkeypatch, mode):
    task = ROOT/'tasks/triton2triton/vllm/triton_bincount'
    h = module_at(task/'scripts/task_runner.py', monkeypatch)
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    for name in ('arange', 'randint', 'full', 'zeros', 'tensor'):
        factory = getattr(torch, name)
        monkeypatch.setattr(torch, name, lambda *a, _factory=factory, **kw: _factory(*a, **{**kw, 'device': 'cpu'}))
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    calls = []
    def candidate(mapping, tokens, prompt, prefill, masks, counts, maximum):
        calls.append((tuple(tokens.shape), counts.shape[1], maximum))
        if mode == 'mutate_tokens': tokens.zero_()
        if mode == 'mutate_lengths': prompt.zero_()
        if mode == 'mutate_mapping': mapping.zero_()
        if mode == 'identity_mapping': mapping = torch.arange(mapping.numel(), dtype=mapping.dtype)
        if mode == 'first_block_only': prompt, prefill = prompt.clamp_max(1024), prefill.clamp_max(1024)
        if mode == 'ignore_prompt': prompt = torch.zeros_like(prompt)
        if mode == 'ignore_prefill': prefill = torch.full_like(prefill, tokens.shape[1])
        if mode == 'clear_inactive': masks.zero_(); counts.zero_()
        _bincount_cpu(mapping, tokens, prompt, prefill, masks, counts, reset=mode != 'skip_reset')
        if mode == 'wrong_mask': masks.zero_()
        if mode == 'wrong_count': counts.zero_()
        if mode == 'dtype': masks.data = masks.long()
        if mode == 'shape': masks.resize_(1, 1)
    mod = SimpleNamespace(bincount=candidate)
    h.load_module = lambda: mod
    checks.install(h)
    ok, reason = h.run_correctness(case_index=None if mode == 'correct' else 0)
    assert ok is (mode == 'correct'), reason
    if mode == 'correct':
        assert [c for c in calls if c[0] != (4, 1031)] == [((b, n), v, n) for b, n, v in h.TEST_SHAPES]
        assert [c for c in calls if c[0] == (4, 1031)] == [((4, 1031), 65, 1031)]
    assert mod.bincount is candidate


@pytest.mark.parametrize('mode', ['correct', 'wrong_timed', 'stale', 'no_write', 'wrong_replay',
                                 'mutate_timed', 'mutate_replay', 'zero_tokens_and_outputs',
                                 'omit_reset', 'raise_replay'])
def test_bincount_original_atomic_timing_reset_replay_and_restoration(monkeypatch, mode):
    import inspect
    task = ROOT/'tasks/triton2triton/vllm/triton_bincount'
    h = module_at(task/'scripts/task_runner.py', monkeypatch)
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    h._TimedRun = module_at(ROOT/'src/tools/perf/aka_benchmark.py', monkeypatch).TimedRun
    for name in ('arange', 'randint', 'full', 'zeros'):
        factory = getattr(torch, name)
        monkeypatch.setattr(torch, name, lambda *a, _factory=factory, **kw: _factory(*a, **{**kw, 'device': 'cpu'}))
    launches, buffers, saved, options = [], [], [], []
    class Kernel:
        def __getitem__(self, grid):
            def launch(mapping, tokens, ts, prompt, prefill, masks, ms, counts, cs, *, BLOCK_SIZE):
                assert grid == (mapping.numel(), (tokens.shape[1] + 1023)//1024)
                assert (ts, ms, cs, BLOCK_SIZE) == (tokens.stride(0), masks.stride(0), counts.stride(0), 1024)
                launches.append(grid)
                _bincount_cpu(mapping, tokens, prompt, prefill, masks, counts, reset=False)
            return launch
    def forbidden_wrapper(*args): raise AssertionError('Original timing targets the JIT launch')
    mod = SimpleNamespace(_bincount_kernel=Kernel(), bincount=forbidden_wrapper)
    h.load_module = lambda: mod
    def benchmark(measured, *, timed_run, **kwargs):
        fn = inspect.getclosurevars(measured).nonlocals['fn']
        state = inspect.getclosurevars(fn).nonlocals
        inputs = tuple(state[k] for k in ('idx_mapping', 'all_token_ids', 'prompt_len', 'prefill_len'))
        outputs = tuple(state[k] for k in ('prompt_mask', 'output_counts'))
        assert torch.equal(inputs[0], torch.arange(inputs[0].numel(), dtype=torch.int32))
        assert (inputs[2] == inputs[1].shape[1]//2).all() and (inputs[3] == inputs[1].shape[1]).all()
        prepare = kwargs.pop('prepare_fn'); options.append(kwargs)
        assert prepare.__name__ == 'prepare_kernel'
        ps = inspect.getclosurevars(prepare).nonlocals
        assert ps['prompt_mask'] is outputs[0] and ps['output_counts'] is outputs[1]
        buffers.append(inputs+outputs); saved.append(checks.snapshots(inputs+outputs))
        outputs[0].fill_(17); outputs[1].fill_(9)
        prepare()
        assert all(torch.count_nonzero(v) == 0 for v in outputs)
        value = measured(); cache = checks.snapshots(outputs)
        if mode == 'wrong_timed': outputs[1].zero_()
        if mode == 'mutate_timed': inputs[1].zero_()
        def replay():
            if mode == 'raise_replay': raise RuntimeError('Replay failed')
            if mode != 'omit_reset': prepare()
            if mode == 'stale':
                for v, old in zip(outputs, cache): v.copy_(old)
            elif mode != 'no_write': measured()
            if mode == 'wrong_replay': outputs[1].zero_()
            if mode == 'mutate_replay': inputs[3].zero_()
            if mode == 'zero_tokens_and_outputs':
                inputs[1].zero_()
                for v in outputs: v.zero_()
            return outputs
        timed_run._bind(replay, value)
        return .125, {'benchmark_method': 'cuda_graph', 'effective_repeats': 1}
    h._benchmark_cuda_graph_or_events = benchmark
    checks.install(h)
    rows = h.run_performance()
    assert len(rows) == 5
    assert options == [dict(warmup=10, repetition=100, target_ms=20.)] * 5
    assert launches
    for (b, n, v), row in zip(h.TEST_SHAPES, rows):
        assert row['params'] == dict(batch=b, seq_len=n, vocab=v)
        assert row['execution_time_ms'] == (.125 if mode == 'correct' else -1.)
        if mode == 'correct': assert row['original_atomic_reset_replay_checked']
    for values, originals in zip(buffers, saved): checks.unchanged(values, originals)
    assert h._benchmark_cuda_graph_or_events is benchmark


def test_bincount_adapter_installs_checks(monkeypatch):
    adapter = module_at(ROOT/'tasks/triton2triton/vllm/triton_bincount/_arena_eval.py', monkeypatch)
    h = adapter.load_harness()
    assert h.run_correctness.__module__ == h.run_performance.__module__ == '_bincount_checks'



def _reduce_segments_cpu(partial, maxima, sums, output, lengths, starts, tile_size=16):
    weights = maxima.double().softmax(-1)
    numerator = (partial.double() * weights.unsqueeze(-1)).sum(2)
    denominator = (sums.double() * weights).sum(-1).unsqueeze(-1)
    output.copy_((numerator / denominator)[..., :output.shape[-1]])
    return output


def _reduce_segments_cpu_harness(monkeypatch):
    import sys
    task = ROOT/'tasks/triton2triton/vllm/triton_reduce_segments'
    h = module_at(task/'scripts/task_runner.py', monkeypatch)
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    monkeypatch.setitem(sys.modules, 'triton', SimpleNamespace(next_power_of_2=lambda n: 1 << (n-1).bit_length()))
    make = h.make_test_data
    h.make_test_data = lambda *a: make(*a[:-1], device='cpu')
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    return h, checks


def test_reduce_segments_independent_known_answer(monkeypatch):
    h, checks = _reduce_segments_cpu_harness(monkeypatch)
    partial = torch.tensor([[[[2., 4.], [6., 8.]]]])
    maxima = torch.tensor([[[0., float(torch.log(torch.tensor(3.)))]]])
    sums = torch.ones(1, 1, 2)
    output = torch.empty(1, 1, 2, dtype=torch.float16)
    lengths, starts = torch.tensor([32], dtype=torch.int32), torch.tensor([0, 1], dtype=torch.int32)
    expected = torch.tensor([[[5., 7.]]], dtype=torch.float16)
    checks.check_output(checks.reference(h, (partial, maxima, sums, lengths, starts), output), expected)
    checks.check_output(_reduce_segments_cpu(partial, maxima, sums, output, lengths, starts), expected)


@pytest.mark.parametrize('mode', ['correct', 'dtype', 'shape', 'nonfinite', 'wrong_value', 'return_copy',
                                 'omit_output', 'mutate_partial', 'mutate_maxima', 'mutate_sums', 'mutate_lengths', 'mutate_starts'])
def test_reduce_segments_actual_correctness_output_contract_and_pristine_inputs(monkeypatch, mode):
    h, checks = _reduce_segments_cpu_harness(monkeypatch)
    seen = []
    def candidate(partial, maxima, sums, output, lengths, starts, tile_size=16):
        seen.append((tuple(partial.shape), tuple(output.shape), tile_size))
        tensors = dict(mutate_partial=partial, mutate_maxima=maxima, mutate_sums=sums,
                       mutate_lengths=lengths, mutate_starts=starts)
        if mode in tensors: tensors[mode].zero_()
        if mode == 'dtype': output.data = output.float()
        if mode == 'shape': output.resize_(1, 1, 1)
        result = _reduce_segments_cpu(partial, maxima, sums, output, lengths, starts, tile_size)
        if mode == 'nonfinite': output.fill_(torch.inf)
        if mode == 'wrong_value': output.zero_()
        if mode == 'return_copy': result = result.clone()
        if mode == 'omit_output': result = None
        return result
    mod = SimpleNamespace(reduce_attention_segments=candidate)
    h.load_module = lambda: mod
    checks.install(h)
    ok, reason = h.run_correctness()
    assert ok is (mode == 'correct'), reason
    if mode == 'correct':
        assert seen == [((b, nh, ns, hs), (b, nh, hs), 16) for b, nh, hs, ns, sl in h.TEST_SHAPES]
    assert mod.reduce_attention_segments is candidate


@pytest.mark.parametrize('mode', ['correct', 'wrong_timed', 'stale', 'no_write', 'wrong_replay',
                                 'mutate_timed', 'mutate_replay', 'mutate_routing', 'zero_input_and_output', 'raise_replay'])
def test_reduce_segments_original_timing_and_replay_restore_all_buffers(monkeypatch, mode):
    import inspect
    h, checks = _reduce_segments_cpu_harness(monkeypatch)
    h._TimedRun = module_at(ROOT/'src/tools/perf/aka_benchmark.py', monkeypatch).TimedRun
    mod = SimpleNamespace(reduce_attention_segments=_reduce_segments_cpu)
    h.load_module = lambda: mod
    buffers, saved, options = [], [], []
    def benchmark(measured, *, timed_run, **kwargs):
        fn = inspect.getclosurevars(measured).nonlocals['fn']
        state = inspect.getclosurevars(fn).nonlocals
        inputs = tuple(state[k] for k in ('segm_output', 'segm_max_t', 'segm_expsum', 'seqused_k', 'cu_seqlens_q'))
        output = state['output']
        buffers.append(inputs+(output,)); saved.append(checks.snapshots(inputs+(output,))); options.append(kwargs)
        value = measured(); cache = output.clone()
        if mode == 'wrong_timed': output.zero_()
        if mode == 'mutate_timed': inputs[0].zero_()
        def replay():
            if mode == 'raise_replay': raise RuntimeError('Replay failed')
            if mode == 'stale': output.copy_(cache)
            elif mode != 'no_write': measured()
            if mode == 'wrong_replay': output.zero_()
            if mode == 'mutate_replay': inputs[1].zero_()
            if mode == 'mutate_routing': inputs[4].zero_()
            if mode == 'zero_input_and_output': inputs[0].zero_(); output.zero_()
            return output
        timed_run._bind(replay, value)
        return .125, {'benchmark_method': 'cuda_graph'}
    h._benchmark_cuda_graph_or_events = benchmark
    checks.install(h)
    rows = h.run_performance()
    assert len(rows) == 5
    assert options == [dict(warmup=10, repetition=100)] * 5
    for (b, nh, hs, ns, sl), row in zip(h.TEST_SHAPES, rows):
        assert row['params'] == dict(num_seqs=b, num_query_heads=nh, head_size=hs, num_segments=ns, seq_len_k=sl)
        assert row['execution_time_ms'] == (.125 if mode == 'correct' else -1.)
        if mode == 'correct': assert row['perturbed_input_replay_checked']
    for values, originals in zip(buffers, saved): checks.unchanged(values, originals)
    assert h._benchmark_cuda_graph_or_events is benchmark


def test_reduce_segments_adapter_installs_checks(monkeypatch):
    adapter = module_at(ROOT/'tasks/triton2triton/vllm/triton_reduce_segments/_arena_eval.py', monkeypatch)
    h = adapter.load_harness()
    assert h.run_correctness.__module__ == h.run_performance.__module__ == '_reduce_segments_checks'


def _gather_tables_cpu(mapping, source, destination, counts):
    for row, request in enumerate(mapping.tolist()):
        length = int(counts[request])
        destination[row, :length].copy_(source[request, :length])
    return destination[:mapping.numel()]


def test_gather_tables_independent_known_answer_retains_unwritten_regions(monkeypatch):
    task = ROOT/'tasks/triton2triton/vllm/triton_gather_block_tables'
    h = module_at(task/'scripts/task_runner.py', monkeypatch)
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    source = torch.arange(24, dtype=torch.int32).reshape(4, 6)
    mapping, counts = torch.tensor([3, 1, 3], dtype=torch.int32), torch.tensor([0, 0, 1, 4], dtype=torch.int32)
    destination = torch.full_like(source, -9)
    expected = torch.tensor([[18, 19, 20, 21, -9, -9], [-9]*6, [18, 19, 20, 21, -9, -9], [-9]*6], dtype=torch.int32)
    torch.testing.assert_close(checks.reference(h, (mapping, source, counts), destination), expected, atol=0, rtol=0)
    output = _gather_tables_cpu(mapping, source, destination, counts)
    checks.check_output(output, destination, expected, 3)


@pytest.mark.parametrize('mode', ['correct', 'dtype', 'device', 'shape', 'return_copy', 'mutate_mapping',
                                 'mutate_counts', 'mutate_source_and_destination', 'ignore_counts',
                                 'identity_mapping', 'first_tile_only', 'clear_unwritten', 'ignore_duplicate'])
def test_gather_tables_actual_correctness_mapping_tail_and_source_controls(monkeypatch, mode):
    task = ROOT/'tasks/triton2triton/vllm/triton_gather_block_tables'
    h = module_at(task/'scripts/task_runner.py', monkeypatch)
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    for name in ('randperm', 'randint', 'arange', 'full'):
        factory = getattr(torch, name)
        monkeypatch.setattr(torch, name, lambda *a, _factory=factory, **kw: _factory(*a, **{**kw, 'device': 'cpu'}))
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    calls = []
    def candidate(mapping, source, destination, counts):
        calls.append((mapping.numel(), tuple(source.shape)))
        if mode == 'mutate_mapping': mapping.zero_()
        if mode == 'mutate_counts': counts.zero_()
        if mode == 'mutate_source_and_destination': source.zero_(); destination.zero_()
        if mode == 'clear_unwritten': destination.zero_()
        seen = set()
        for row, request in enumerate(mapping.tolist()):
            if mode == 'ignore_duplicate' and request in seen: continue
            seen.add(request)
            if mode == 'identity_mapping': request = row
            length = source.shape[1] if mode == 'ignore_counts' else int(counts[request])
            if mode == 'first_tile_only': length = min(length, 1024)
            destination[row, :length].copy_(source[request, :length])
        output = destination[:mapping.numel()]
        if mode == 'dtype': output = output.long()
        if mode == 'device': output = torch.empty_like(output, device='meta')
        if mode == 'shape': output = output[:1]
        if mode == 'return_copy': output = output.clone()
        return output
    mod = SimpleNamespace(gather_block_tables=candidate)
    h.load_module = lambda: mod
    checks.install(h)
    ok, reason = h.run_correctness()
    assert ok is (mode == 'correct'), reason
    if mode == 'correct':
        assert [c for c in calls if c[1] != (5, 1031)] == [(n, (r, b)) for n, r, b in h.TEST_SHAPES]
        assert [c for c in calls if c[1] == (5, 1031)] == [(3, (5, 1031))]
    assert mod.gather_block_tables is candidate


@pytest.mark.parametrize('mode', ['correct', 'wrong_timed', 'stale', 'no_write', 'wrong_replay',
                                 'mutate_timed', 'mutate_replay', 'zero_source_and_destination',
                                 'overwrite_tail', 'overwrite_inactive', 'raise_replay'])
def test_gather_tables_original_full_row_timing_partial_replay_restores_buffers(monkeypatch, mode):
    import inspect
    task = ROOT/'tasks/triton2triton/vllm/triton_gather_block_tables'
    h = module_at(task/'scripts/task_runner.py', monkeypatch)
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    h._TimedRun = SimpleNamespace
    for name in ('randint', 'arange', 'full'):
        factory = getattr(torch, name)
        monkeypatch.setattr(torch, name, lambda *a, _factory=factory, **kw: _factory(*a, **{**kw, 'device': 'cpu'}))
    buffers, snapshots, options = [], [], []
    mod = SimpleNamespace(gather_block_tables=_gather_tables_cpu)
    h.load_module = lambda: mod
    def benchmark(measured, *, timed_run, **kwargs):
        fn = inspect.getclosurevars(measured).nonlocals['fn']
        c = inspect.getclosurevars(fn).nonlocals
        mapping, source, counts, destination = (c[k] for k in ('idx_mapping', 'src_block_table', 'num_blocks', 'dst_block_table'))
        assert torch.equal(mapping, torch.arange(mapping.numel(), dtype=mapping.dtype))
        assert (counts == source.shape[1]).all(), 'Original full-length scored copy is retained'
        values = (mapping, source, counts, destination)
        buffers.append(values); snapshots.append(tuple(v.clone() for v in values)); options.append(kwargs)
        output = measured(); cached = destination.clone()
        if mode == 'wrong_timed': destination[0, 0] = -1
        if mode == 'mutate_timed': source.zero_()
        def replay():
            if mode == 'raise_replay': raise RuntimeError('replay failed')
            if mode == 'stale': destination.copy_(cached)
            elif mode != 'no_write': measured()
            if mode == 'wrong_replay': destination[0, 0] = -1
            if mode == 'mutate_replay': counts.zero_()
            if mode == 'zero_source_and_destination': source.zero_(); destination.zero_()
            if mode == 'overwrite_tail':
                for row, request in enumerate(mapping.tolist()): destination[row].copy_(source[request])
            if mode == 'overwrite_inactive': destination[mapping.numel():].fill_(-1)
            return output
        timed_run.outputs, timed_run.rerun = output, replay
        return .125, {'benchmark_method': 'cuda_graph'}
    h._benchmark_cuda_graph_or_events = benchmark
    checks.install(h)
    rows = h.run_performance()
    assert len(rows) == 5
    assert options == [dict(warmup=10, repetition=100)] * 5
    for case, row in zip(h.TEST_SHAPES, rows):
        assert row['params'] == dict(num_reqs=case[0], max_num_reqs=case[1], max_num_blocks=case[2])
        assert row['execution_time_ms'] == (.125 if mode == 'correct' else -1.)
        if mode == 'correct': assert row['perturbed_input_replay_checked']
    for values, saved in zip(buffers, snapshots): checks.unchanged(values, saved)
    assert mod.gather_block_tables is _gather_tables_cpu
    assert h._benchmark_cuda_graph_or_events is benchmark


def test_gather_tables_adapter_installs_checks(monkeypatch):
    adapter = module_at(ROOT/'tasks/triton2triton/vllm/triton_gather_block_tables/_arena_eval.py', monkeypatch)
    h = adapter.load_harness()
    assert h.run_correctness.__module__ == h.run_performance.__module__ == '_gather_table_checks'



def _recovered_tokens_cpu(cu, ids, draft, target, q, maximum, vocab):
    sizes = torch.diff(cu, prepend=cu.new_zeros(1)).long()
    rows = torch.repeat_interleave(torch.arange(cu.numel()), sizes)
    scores = target.clone() if draft is None else (target-draft).clamp_min(0)
    if draft is None: scores[torch.arange(ids.numel()), ids.long()] = 0
    return (scores/q[rows]).argmax(-1).to(ids.dtype)


def _recovered_tokens_cpu_harness(monkeypatch):
    task = ROOT/'tasks/triton2triton/vllm/triton_sample_recovered_tokens'
    h = module_at(task/'scripts/task_runner.py', monkeypatch)
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    for name in ('randint', 'rand', 'empty'):
        factory = getattr(torch, name)
        monkeypatch.setattr(torch, name, lambda *a, _factory=factory, **kw: _factory(*a, **{**kw, 'device': 'cpu'}))
    original = torch.Tensor.to
    def to_cpu(value, *args, **kwargs):
        if args and isinstance(args[0], str) and args[0].startswith('cuda'): args = ('cpu', *args[1:])
        return original(value, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'to', to_cpu)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    return h, checks


@pytest.mark.parametrize('no_draft', [False, True])
def test_recovered_tokens_independent_known_answer_both_paths(monkeypatch, no_draft):
    h, checks = _recovered_tokens_cpu_harness(monkeypatch)
    cu, ids = torch.tensor([2, 3], dtype=torch.int32), torch.tensor([0, 1, 2], dtype=torch.int32)
    draft = None if no_draft else torch.tensor([[.8, .1, .1], [.1, .5, .4], [.2, .2, .6]])
    target = torch.tensor([[.4, .5, .1], [.3, .6, .1], [.1, .2, .7]])
    q = torch.tensor([[1., 1., .5], [.1, 2., 1.]])
    expected = torch.tensor([1, 0, 0 if no_draft else 2], dtype=torch.int32)
    inputs = (cu, ids, draft, target, q)
    checks.check_output(checks.reference(h, inputs, 3), expected)
    checks.check_output(_recovered_tokens_cpu(*inputs, 2, 3), expected)


@pytest.mark.parametrize('mode', ['correct', 'dtype', 'no_draft_dtype', 'shape', 'wrong', 'wrong_no_draft',
                                 'mutate_cu', 'mutate_ids', 'mutate_draft', 'mutate_target', 'mutate_q'])
def test_recovered_tokens_actual_correctness_both_paths_and_readonly_inputs(monkeypatch, mode):
    h, checks = _recovered_tokens_cpu_harness(monkeypatch)
    calls = []
    def candidate(cu, ids, draft, target, q, maximum, vocab):
        calls.append((cu.tolist(), maximum, vocab, draft is None))
        outputs = _recovered_tokens_cpu(cu, ids, draft, target, q, maximum, vocab)
        values = dict(mutate_cu=cu, mutate_ids=ids, mutate_draft=draft, mutate_target=target, mutate_q=q)
        if mode in values and values[mode] is not None: values[mode].zero_()
        if mode == 'dtype' or mode == 'no_draft_dtype' and draft is None: outputs = outputs.float()
        if mode == 'shape': outputs = outputs[:1]
        if mode == 'wrong' or mode == 'wrong_no_draft' and draft is None: outputs.fill_(-1)
        return outputs
    mod = SimpleNamespace(sample_recovered_tokens=candidate)
    h.load_module = lambda: mod
    checks.install(h)
    ok, reason = h.run_correctness()
    assert ok is (mode == 'correct'), reason
    if mode == 'correct':
        expected = []
        for b, m, v in h.TEST_SHAPES:
            cu = torch.tensor([m-(j%2) for j in range(b)]).cumsum(0).tolist()
            expected.extend([(cu, m, v, False), (cu, m, v, True)])
        assert calls == expected
    assert mod.sample_recovered_tokens is candidate


@pytest.mark.parametrize('mode', ['correct', 'wrong_timed', 'stale', 'no_write', 'wrong_replay',
                                 'mutate_timed', 'mutate_replay', 'mutate_routing', 'raise_replay'])
def test_recovered_tokens_original_full_request_timing_and_exact_replay(monkeypatch, mode):
    import inspect
    h, checks = _recovered_tokens_cpu_harness(monkeypatch)
    h._TimedRun = module_at(ROOT/'src/tools/perf/aka_benchmark.py', monkeypatch).TimedRun
    mod = SimpleNamespace(sample_recovered_tokens=_recovered_tokens_cpu)
    h.load_module = lambda: mod
    buffers, saved, options = [], [], []
    def benchmark(measured, *, timed_run, **kwargs):
        fn = inspect.getclosurevars(measured).nonlocals['fn']
        state = inspect.getclosurevars(fn).nonlocals
        inputs = tuple(state[k] for k in ('cu', 'draft_ids', 'draft_probs', 'target_probs', 'q'))
        assert inputs[2] is not None
        assert torch.equal(inputs[0], torch.arange(1, inputs[0].numel()+1)*state['max_draft'])
        buffers.append(inputs); saved.append(checks.snapshots(inputs)); options.append(kwargs)
        out = measured(); cache = out.clone()
        if mode == 'wrong_timed': out.fill_(-1)
        if mode == 'mutate_timed': inputs[3].zero_()
        def replay():
            if mode == 'raise_replay': raise RuntimeError('Replay failed')
            if mode == 'stale': out.copy_(cache)
            elif mode != 'no_write': out.copy_(measured())
            if mode == 'wrong_replay': out.fill_(-1)
            if mode == 'mutate_replay': inputs[4].fill_(1.)
            if mode == 'mutate_routing': inputs[0].zero_()
            return out
        timed_run._bind(replay, out)
        return .125, {'benchmark_method': 'cuda_graph'}
    h._benchmark_cuda_graph_or_events = benchmark
    checks.install(h)
    rows = h.run_performance()
    assert len(rows) == 5
    assert options == [dict(warmup=10, repetition=100)] * 5
    for (b, m, v), row in zip(h.TEST_SHAPES, rows):
        assert row['params'] == dict(batch=b, max_draft=m, vocab=v)
        assert row['execution_time_ms'] == (.125 if mode == 'correct' else -1.)
        if mode == 'correct': assert row['perturbed_input_replay_checked']
    for values, originals in zip(buffers, saved): checks.unchanged(values, originals)
    assert h._benchmark_cuda_graph_or_events is benchmark
    assert mod.sample_recovered_tokens is _recovered_tokens_cpu


def test_recovered_tokens_adapter_installs_checks(monkeypatch):
    adapter = module_at(ROOT/'tasks/triton2triton/vllm/triton_sample_recovered_tokens/_arena_eval.py', monkeypatch)
    h = adapter.load_harness()
    assert h.run_correctness.__module__ == h.run_performance.__module__ == '_recovered_tokens_checks'


_INT8_QUANT_TASKS = ['per_token_quant_int8', 'per_token_group_quant_int8']


def _int8_quant_cpu(x, group_size=None, eps=1e-10, dtype=None):
    width = group_size or x.shape[-1]
    groups = x.float().reshape(-1, width)
    maximum = groups.abs().amax(-1, keepdim=True).clamp_min(eps)
    scales = maximum/127
    codes = groups/scales
    codes = codes.trunc() if group_size else codes.round()
    quant = codes.clamp(-128, 127).to(dtype or torch.int8).reshape(x.shape)
    return quant, scales.reshape(*x.shape[:-1], x.shape[-1]//width)


def _int8_quant_cpu_harness(monkeypatch, symbol):
    task = ROOT/'tasks/triton2triton/vllm'/('triton_'+symbol)
    h = module_at(task/'scripts/task_runner.py', monkeypatch)
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    for name in ('randn', 'arange'):
        factory = getattr(torch, name)
        monkeypatch.setattr(torch, name, lambda *a, _factory=factory, **kw: _factory(*a, **{**kw, 'device': 'cpu'}))
    original = torch.Tensor.to
    def cpu_to(value, *args, **kwargs):
        if args and isinstance(args[0], str) and args[0].startswith('cuda'): args = ('cpu', *args[1:])
        return original(value, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'to', cpu_to)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    return h, checks


@pytest.mark.parametrize('symbol', _INT8_QUANT_TASKS)
def test_int8_quant_independent_known_codes_and_scale_contract(monkeypatch, symbol):
    h, checks = _int8_quant_cpu_harness(monkeypatch, symbol)
    x = torch.tensor([[0., 1., -2., 0.], [0., 0., 0., 0.]], dtype=torch.float16)
    grouped = 'group' in symbol
    options = dict(group_size=2) if grouped else {}
    expected_q = torch.tensor([[0, 127, -127, 0], [0, 0, 0, 0]] if grouped else
                              [[0, 64, -127, 0], [0, 0, 0, 0]], dtype=torch.int8)
    expected_s = torch.tensor([[1/127, 2/127], [1e-10/127, 1e-10/127]] if grouped else
                              [[2/127], [1e-10/127]], dtype=torch.float32)
    expected = expected_q, expected_s
    checks.check_outputs(checks.reference(h, x, options), expected)
    checks.check_outputs(_int8_quant_cpu(x, **options), expected)
    # Existing +1 code allowance is preserved; a two-code error still fails.
    allowed = expected_q.clone(); allowed[0, 0] = 1
    checks.check_outputs((allowed, expected_s), expected)
    allowed[0, 0] = 2
    with pytest.raises(AssertionError): checks.check_outputs((allowed, expected_s), expected)
    invalid_scales = expected_s.clone(); invalid_scales[1].zero_()
    with pytest.raises(AssertionError, match='positive'): checks.check_outputs((expected_q, invalid_scales), expected)


@pytest.mark.parametrize('symbol,mode', [
    (symbol, mode) for symbol in _INT8_QUANT_TASKS
    for mode in ['correct', 'q_dtype', 'scale_dtype', 'q_shape', 'scale_shape',
                 'nonfinite_scale', 'empty_tuple', 'missing_scale', 'wrong_q', 'wrong_scale',
                 'mutate_input', 'zero_case', 'ignore_eps', 'truncate_tail', 'flatten_output']
    if mode != 'ignore_eps' or 'group' in symbol
])
def test_int8_quant_actual_correctness_metadata_and_optional_inputs(monkeypatch, symbol, mode):
    h, checks = _int8_quant_cpu_harness(monkeypatch, symbol)
    calls = []
    def candidate(x, **kwargs):
        calls.append((tuple(x.shape), x.is_contiguous(), dict(kwargs)))
        if mode == 'mutate_input': x.zero_()
        if mode == 'ignore_eps': kwargs['eps'] = 1e-10
        quant, scales = _int8_quant_cpu(x, **kwargs)
        if mode == 'q_dtype': quant = quant.float()
        if mode == 'scale_dtype': scales = scales.half()
        if mode == 'q_shape': quant = quant.flatten()
        if mode == 'scale_shape': scales = scales.flatten()
        if mode == 'nonfinite_scale': scales.fill_(float('nan'))
        if mode == 'empty_tuple': return ()
        if mode == 'missing_scale': return (quant,)
        if mode == 'wrong_q': quant.fill_(0)
        if mode == 'wrong_scale': scales.mul_(2)
        if mode == 'zero_case':
            zero = x.reshape(-1, x.shape[-1]).abs().amax(-1) == 0
            scales.reshape(zero.numel(), -1)[zero] = 0
        if mode == 'truncate_tail':
            if x.shape[-1] & (x.shape[-1]-1): quant[..., -1].fill_(-128)
        if mode == 'flatten_output':
            quant, scales = quant.reshape(-1, x.shape[-1]), scales.reshape(-1, scales.shape[-1])
        return quant, scales
    # The protected group harness calls group_size positionally.
    def public(x, group_size=None, **kw):
        if group_size is not None: kw['group_size'] = group_size
        return candidate(x, **kw)
    mod = SimpleNamespace(**{symbol: public})
    h.load_module = lambda: mod
    checks.install(h)
    ok, reason = h.run_correctness()
    assert ok is (mode == 'correct'), reason
    if mode == 'correct':
        original = [c for c in calls if len(c[0]) == 2 and c[0] != (17, 6)]
        assert [c[0] for c in original] == [tuple(v[:2]) for v in h.TEST_SHAPES]
        if 'group' in symbol:
            assert [c[2]['group_size'] for c in original] == [v[2] for v in h.TEST_SHAPES]
            assert ((2, 3, 34), True, {'group_size':17, 'eps':.5}) in calls
        else:
            assert any(c[0] == (2, 3, 17) for c in calls)
            assert any(c[0] == (17, 6) and not c[1] for c in calls)
    assert getattr(mod, symbol) is public


@pytest.mark.parametrize('symbol', _INT8_QUANT_TASKS)
@pytest.mark.parametrize('mode', ['correct', 'wrong_timed', 'stale', 'no_write', 'omit_q', 'omit_scale',
                                 'wrong_replay', 'mutate_timed', 'mutate_replay', 'raise_replay'])
def test_int8_quant_original_timing_full_tuple_replay_and_restored_input(monkeypatch, symbol, mode):
    import inspect
    h, checks = _int8_quant_cpu_harness(monkeypatch, symbol)
    h._TimedRun = module_at(ROOT/'src/tools/perf/aka_benchmark.py', monkeypatch).TimedRun
    mod = SimpleNamespace(**{symbol: _int8_quant_cpu})
    h.load_module = lambda: mod
    inputs, pristine, options = [], [], []
    def benchmark(measured, *, timed_run, **kwargs):
        fn = inspect.getclosurevars(measured).nonlocals['fn']
        state = inspect.getclosurevars(fn).nonlocals
        x = state['x']; inputs.append(x); pristine.append(x.clone()); options.append(kwargs)
        outputs = measured(); cached = tuple(v.clone() for v in outputs)
        if mode == 'wrong_timed': outputs[0].zero_()
        if mode == 'mutate_timed': x.zero_()
        def replay():
            if mode == 'raise_replay': raise RuntimeError('Replay failed')
            if mode != 'no_write':
                computed = cached if mode == 'stale' else measured()
                for i, (out, value) in enumerate(zip(outputs, computed)):
                    if mode == 'omit_q' and i == 0 or mode == 'omit_scale' and i == 1: continue
                    out.copy_(value)
            if mode == 'wrong_replay': outputs[0].zero_()
            if mode == 'mutate_replay': x.zero_()
            return outputs
        timed_run._bind(replay, outputs)
        return .125, {'benchmark_method':'cuda_graph'}
    h._benchmark_cuda_graph_or_events = benchmark
    checks.install(h)
    rows = h.run_performance()
    assert len(rows) == 5
    assert options == [dict(warmup=10, repetition=100)] * 5
    for case, row in zip(h.TEST_SHAPES, rows):
        expected = dict(M=case[0], N=case[1])
        if 'group' in symbol: expected['group_size'] = case[2]
        assert row['params'] == expected
        assert row['execution_time_ms'] == (.125 if mode == 'correct' else -1.)
        if mode == 'correct': assert row['perturbed_input_replay_checked']
    for x, saved in zip(inputs, pristine): checks.unchanged(x, saved)
    assert getattr(mod, symbol) is _int8_quant_cpu
    assert h._benchmark_cuda_graph_or_events is benchmark


@pytest.mark.parametrize('symbol', _INT8_QUANT_TASKS)
def test_int8_quant_adapters_install_task_local_checks(monkeypatch, symbol):
    adapter = module_at(ROOT/'tasks/triton2triton/vllm'/('triton_'+symbol)/'_arena_eval.py', monkeypatch)
    h = adapter.load_harness()
    assert h.run_correctness.__module__ == h.run_performance.__module__ == '_int8_quant_checks'


_FP8_GROUP_TASKS = ['per_token_group_quant_fp8', 'per_token_group_quant_fp8_colmajor']


def _fp8_group_cpu(x, group_size, eps=1e-10, dtype=None, use_ue8m0=False, *, colmajor=False):
    dtype = dtype or torch.float8_e4m3fnuz
    limit = 240. if dtype == torch.float8_e4m3fnuz else torch.finfo(dtype).max
    grouped = x.float().reshape(x.shape[0], -1, group_size)
    scales = grouped.abs().amax(-1).clamp_min(eps)/limit
    if use_ue8m0: scales = torch.pow(2., scales.log2().ceil())
    quant = (grouped/scales.unsqueeze(-1)).clamp(-limit, limit).to(dtype).reshape(x.shape)
    if colmajor:
        transposed = torch.empty((scales.shape[1], scales.shape[0]), dtype=torch.float32)
        transposed.copy_(scales.t()); scales = transposed.t()
    return quant, scales


def _fp8_group_cpu_harness(monkeypatch, symbol):
    task = ROOT/'tasks/triton2triton/vllm'/('triton_'+symbol)
    h = module_at(task/'scripts/task_runner.py', monkeypatch)
    checks = module_at(task/'_arena_checks.py', monkeypatch)
    for name in ('randn', 'arange'):
        factory = getattr(torch, name)
        monkeypatch.setattr(torch, name, lambda *a, _factory=factory, **kw: _factory(*a, **{**kw, 'device':'cpu'}))
    original = torch.Tensor.to
    def cpu_to(value, *args, **kwargs):
        if args and isinstance(args[0], str) and args[0].startswith('cuda'): args = ('cpu', *args[1:])
        return original(value, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'to', cpu_to)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    return h, checks


@pytest.mark.parametrize('symbol', _FP8_GROUP_TASKS)
def test_fp8_group_independent_known_answers_and_original_tolerance(monkeypatch, symbol):
    h, checks = _fp8_group_cpu_harness(monkeypatch, symbol)
    x = torch.tensor([[0., .5, -1., 0.], [0., 0., 0., 0.]], dtype=torch.float16)
    expected_q = torch.tensor([[0., 240., -240., 0.], [0., 0., 0., 0.]]).to(torch.float8_e4m3fnuz)
    expected_s = torch.tensor([[.5/240, 1/240], [1e-10/240, 1e-10/240]])
    ref = checks.reference(h, x, dict(group_size=2))
    torch.testing.assert_close(ref[0].float(), expected_q.float(), atol=0, rtol=0)
    torch.testing.assert_close(ref[1], expected_s, atol=0, rtol=0)
    actual = _fp8_group_cpu(x, 2, colmajor=checks.COLUMN_MAJOR)
    checks.check_outputs(actual, (expected_q, expected_s), 2)
    # An FP8 rounding step remains allowed by the original dequantized threshold.
    changed = actual[0].float(); changed[0, 1] = 224
    checks.check_outputs((changed.to(actual[0].dtype), actual[1]), ref, 2)
    changed[0, 1] = 0
    with pytest.raises(AssertionError): checks.check_outputs((changed.to(actual[0].dtype), actual[1]), ref, 2)
    rounded = checks.reference(h, x, dict(group_size=2, eps=.5, use_ue8m0=True))
    torch.testing.assert_close(rounded[1], torch.tensor([[1/256, 1/128], [1/256, 1/256]]), atol=0, rtol=0)


@pytest.mark.parametrize('symbol,mode', [
    (symbol, mode) for symbol in _FP8_GROUP_TASKS
    for mode in ['correct', 'q_dtype', 'spoof_dtype_helper', 'scale_dtype', 'q_shape', 'scale_shape',
                 'nonfinite_q', 'nonfinite_scale', 'empty_tuple', 'missing_scale', 'wrong_q',
                 'wrong_scale', 'mutate_input', 'zero_scales', 'ignore_eps', 'ignore_ue8m0',
                 'truncate_tail', 'contiguous_scales']
    if mode != 'contiguous_scales' or symbol.endswith('colmajor')
])
def test_fp8_group_original_correctness_output_contract_and_public_modes(monkeypatch, symbol, mode):
    h, checks = _fp8_group_cpu_harness(monkeypatch, symbol)
    calls = []
    def public(x, group_size, **kwargs):
        calls.append((tuple(x.shape), x.stride(), group_size, dict(kwargs)))
        if mode == 'mutate_input': x.zero_()
        if mode == 'ignore_eps': kwargs['eps'] = 1e-10
        if mode == 'ignore_ue8m0': kwargs['use_ue8m0'] = False
        quant, scales = _fp8_group_cpu(x, group_size, **kwargs, colmajor=checks.COLUMN_MAJOR)
        if mode in ('q_dtype', 'spoof_dtype_helper'): quant = quant.half()
        if mode == 'scale_dtype': scales = scales.half()
        if mode == 'q_shape': quant = quant.flatten()
        if mode == 'scale_shape': scales = scales.flatten()
        if mode == 'nonfinite_q': quant.copy_(torch.full_like(x, float('nan')).to(quant.dtype))
        if mode == 'nonfinite_scale': scales.fill_(float('nan'))
        if mode == 'empty_tuple': return ()
        if mode == 'missing_scale': return (quant,)
        if mode == 'wrong_q': quant.copy_(torch.zeros_like(x).to(quant.dtype))
        if mode == 'wrong_scale': scales.mul_(2)
        if mode == 'zero_scales': scales[x.abs().amax(-1)==0] = 0
        if mode == 'truncate_tail' and group_size == 17:
            bad = quant.float(); bad[:, -1] = -240; quant.copy_(bad.to(quant.dtype))
        if mode == 'contiguous_scales': scales = scales.contiguous()
        return quant, scales
    mod = SimpleNamespace(**{symbol:public, '_get_fp8_dtype':lambda: torch.float16 if mode == 'spoof_dtype_helper' else torch.float8_e4m3fnuz})
    h.load_module = lambda: mod
    checks.install(h)
    ok, reason = h.run_correctness()
    assert ok is (mode == 'correct'), reason
    if mode == 'correct':
        original = [c for c in calls if c[0] != (3, 34)]
        assert [(c[0][0], c[0][1], c[2]) for c in original] == h.TEST_SHAPES
        diagnostic = [c for c in calls if c[0] == (3, 34)]
        assert len(diagnostic) == 2
        assert all(c[1] == (68, 1) and c[2] == 17 and c[3]['eps'] == .5 for c in diagnostic)
        assert diagnostic[-1][3]['use_ue8m0']
    assert getattr(mod, symbol) is public


@pytest.mark.parametrize('symbol', _FP8_GROUP_TASKS)
@pytest.mark.parametrize('mode', ['correct', 'wrong_timed', 'stale', 'no_write', 'omit_q', 'omit_scale',
                                 'wrong_replay', 'mutate_timed', 'mutate_replay', 'raise_replay'])
def test_fp8_group_original_timing_validates_both_outputs_and_restores_input(monkeypatch, symbol, mode):
    import inspect
    h, checks = _fp8_group_cpu_harness(monkeypatch, symbol)
    h._TimedRun = module_at(ROOT/'src/tools/perf/aka_benchmark.py', monkeypatch).TimedRun
    def public(x, group_size, **kwargs):
        return _fp8_group_cpu(x, group_size, **kwargs, colmajor=checks.COLUMN_MAJOR)
    mod = SimpleNamespace(**{symbol:public}); h.load_module = lambda: mod
    inputs, pristine, options = [], [], []
    def benchmark(measured, *, timed_run, **kwargs):
        fn = inspect.getclosurevars(measured).nonlocals['fn']
        state = inspect.getclosurevars(fn).nonlocals
        x = state['x']; inputs.append(x); pristine.append(x.clone()); options.append(kwargs)
        outputs = measured(); cached = tuple(v.clone() for v in outputs)
        if mode == 'wrong_timed': outputs[0].copy_(torch.zeros_like(x).to(outputs[0].dtype))
        if mode == 'mutate_timed': x.zero_()
        def replay():
            if mode == 'raise_replay': raise RuntimeError('Replay failed')
            if mode != 'no_write':
                computed = cached if mode == 'stale' else measured()
                for i, (out, value) in enumerate(zip(outputs, computed)):
                    if mode == 'omit_q' and i == 0 or mode == 'omit_scale' and i == 1: continue
                    out.copy_(value)
            if mode == 'wrong_replay': outputs[0].copy_(torch.zeros_like(x).to(outputs[0].dtype))
            if mode == 'mutate_replay': x.zero_()
            return outputs
        timed_run._bind(replay, outputs)
        return .125, {'benchmark_method':'cuda_graph'}
    h._benchmark_cuda_graph_or_events = benchmark
    checks.install(h)
    rows = h.run_performance()
    assert len(rows) == 5 and options == [dict(warmup=10, repetition=100)]*5
    for case, row in zip(h.TEST_SHAPES, rows):
        assert row['params'] == dict(M=case[0], N=case[1], group_size=case[2])
        assert row['execution_time_ms'] == (.125 if mode == 'correct' else -1.)
        if mode == 'correct': assert row['perturbed_input_replay_checked']
    for x, saved in zip(inputs, pristine): checks.unchanged(x, saved)
    assert getattr(mod, symbol) is public
    assert h._benchmark_cuda_graph_or_events is benchmark


@pytest.mark.parametrize('symbol', _FP8_GROUP_TASKS)
def test_fp8_group_adapters_install_task_local_checks(monkeypatch, symbol):
    adapter = module_at(ROOT/'tasks/triton2triton/vllm'/('triton_'+symbol)/'_arena_eval.py', monkeypatch)
    h = adapter.load_harness()
    assert h.run_correctness.__module__ == h.run_performance.__module__ == '_fp8_group_checks'


def _silu_fp8_cpu(x, output=None, use_ue8m0=False, eps=1e-10):
    gate, up = x.float().chunk(2, dim=-1)
    # Reproduce the public implementation's fp16 activation rounding;
    # the task reference remains independent fp32, with the original gate.
    values = (torch.nn.functional.silu(gate).to(x.dtype) * up.to(x.dtype)).float()
    quant, scales = _fp8_group_cpu(values, 128, eps=eps, use_ue8m0=use_ue8m0, colmajor=True)
    if output is not None: output.copy_(quant); quant = output
    return quant, scales


def _silu_fp8_cpu_harness(monkeypatch):
    h, checks = _fp8_group_cpu_harness(monkeypatch, 'silu_mul_quant_fp8')
    return h, checks


def test_silu_fp8_independent_constant_known_answer_and_power_two_scale(monkeypatch):
    import math
    h, checks = _silu_fp8_cpu_harness(monkeypatch)
    x = torch.ones(128, 256, dtype=torch.float16); x[0].zero_()
    expected_q = torch.full((128, 128), 240.); expected_q[0].zero_()
    expected_s = torch.full((128, 1), (1/(1+math.exp(-1)))/240); expected_s[0] = 1e-10/240
    ref = checks.reference(h, x, {})
    torch.testing.assert_close(ref[0].float(), expected_q, atol=0, rtol=0)
    torch.testing.assert_close(ref[1], expected_s, atol=1e-9, rtol=0)
    checks.check_outputs(_silu_fp8_cpu(x), ref)
    reference_ue = checks.reference(h, x, dict(eps=16, use_ue8m0=True))
    torch.testing.assert_close(reference_ue[1], torch.full((128, 1), .125), atol=0, rtol=0)
    actual = _silu_fp8_cpu(x, eps=16, use_ue8m0=True)
    checks.check_outputs(actual, reference_ue, use_ue8m0=True)
    actual[1].mul_(1.01)
    # This error remains within the old numerical tolerance but violates UE8M0.
    checks.check_outputs(actual, reference_ue)
    with pytest.raises(AssertionError, match='power-of-two'): checks.check_outputs(actual, reference_ue, use_ue8m0=True)


@pytest.mark.parametrize('mode', ['correct', 'q_dtype', 'scale_dtype', 'scale_layout', 'q_shape', 'scale_shape',
                                 'nonfinite_q', 'nonfinite_scale', 'missing_scale', 'wrong_q', 'wrong_scale',
                                 'mutate_input', 'zero_scales', 'ignore_eps', 'ignore_ue8m0', 'ignore_output'])
def test_silu_fp8_original_correctness_and_public_output_options(monkeypatch, mode):
    h, checks = _silu_fp8_cpu_harness(monkeypatch)
    calls = []
    def public(x, output=None, use_ue8m0=False, eps=1e-10):
        calls.append((tuple(x.shape), output is not None, use_ue8m0, eps))
        if mode == 'mutate_input': x.zero_()
        if mode == 'ignore_eps': eps = 1e-10
        if mode == 'ignore_ue8m0': use_ue8m0 = False
        if mode == 'ignore_output': output = None
        quant, scales = _silu_fp8_cpu(x, output, use_ue8m0, eps)
        if mode == 'q_dtype': quant = quant.half()
        if mode == 'scale_dtype': scales = scales.half()
        if mode == 'scale_layout': scales = scales.contiguous()
        if mode == 'q_shape': quant = quant.flatten()
        if mode == 'scale_shape': scales = scales.flatten()
        if mode == 'nonfinite_q': quant.copy_(torch.full_like(quant, float('nan'), dtype=torch.float32).to(quant.dtype))
        if mode == 'nonfinite_scale': scales.fill_(float('nan'))
        if mode == 'missing_scale': return (quant,)
        if mode == 'wrong_q': quant.copy_(torch.zeros_like(quant, dtype=torch.float32).to(quant.dtype))
        if mode == 'wrong_scale': scales.mul_(10)
        if mode == 'zero_scales': scales[x.abs().amax(-1)==0] = 0
        return quant, scales
    mod = SimpleNamespace(**{checks.SYMBOL:public, '_get_fp8_dtype':lambda:torch.float8_e4m3fnuz})
    h.load_module = lambda: mod
    checks.install(h)
    ok, reason = h.run_correctness()
    assert ok is (mode == 'correct'), reason
    if mode == 'correct':
        assert [c[0] for c in calls if not c[1]] == h.TEST_SHAPES
        assert [c for c in calls if c[1]] == [((128,512),True,False,16.),((128,512),True,True,16.)]
    assert getattr(mod, checks.SYMBOL) is public


@pytest.mark.parametrize('mode', ['correct', 'wrong_timed', 'stale', 'no_write', 'omit_q', 'omit_scale',
                                 'wrong_replay', 'mutate_timed', 'mutate_replay', 'raise_replay'])
def test_silu_fp8_original_measured_pair_and_replay_restores_input(monkeypatch, mode):
    import inspect
    h, checks = _silu_fp8_cpu_harness(monkeypatch)
    h._TimedRun = module_at(ROOT/'src/tools/perf/aka_benchmark.py', monkeypatch).TimedRun
    mod = SimpleNamespace(**{checks.SYMBOL:_silu_fp8_cpu}); h.load_module = lambda: mod
    inputs, pristine, options = [], [], []
    def benchmark(measured, *, timed_run, **kwargs):
        fn = inspect.getclosurevars(measured).nonlocals['fn']
        state = inspect.getclosurevars(fn).nonlocals
        x = state['x']; inputs.append(x); pristine.append(x.clone()); options.append(kwargs)
        outputs = measured(); cached = tuple(v.clone() for v in outputs)
        def zero_q(): outputs[0].copy_(torch.zeros_like(outputs[0], dtype=torch.float32).to(outputs[0].dtype))
        if mode == 'wrong_timed': zero_q()
        if mode == 'mutate_timed': x.zero_()
        def replay():
            if mode == 'raise_replay': raise RuntimeError('Replay failed')
            if mode != 'no_write':
                computed = cached if mode == 'stale' else measured()
                for i, (out, value) in enumerate(zip(outputs, computed)):
                    if mode == 'omit_q' and i == 0 or mode == 'omit_scale' and i == 1: continue
                    out.copy_(value)
            if mode == 'wrong_replay': zero_q()
            if mode == 'mutate_replay': x.zero_()
            return outputs
        timed_run._bind(replay, outputs)
        return .125, {'benchmark_method':'cuda_graph'}
    h._benchmark_cuda_graph_or_events = benchmark
    checks.install(h)
    rows = h.run_performance()
    assert len(rows) == 5 and options == [dict(warmup=10, repetition=100)]*5
    for case, row in zip(h.TEST_SHAPES, rows):
        assert row['params'] == dict(M=case[0],N=case[1])
        assert row['execution_time_ms'] == (.125 if mode == 'correct' else -1.)
    for x, saved in zip(inputs, pristine): checks.unchanged(x,saved)
    assert getattr(mod, checks.SYMBOL) is _silu_fp8_cpu
    assert h._benchmark_cuda_graph_or_events is benchmark


def test_silu_fp8_adapter_installs_task_local_checks(monkeypatch):
    h=module_at(ROOT/'tasks/triton2triton/vllm/triton_silu_mul_quant_fp8/_arena_eval.py',monkeypatch).load_harness()
    assert h.run_correctness.__module__ == h.run_performance.__module__ == '_silu_fp8_checks'


def _scaled_mm_cpu(a, b, sa, sb, out_dtype, bias=None, **options):
    # Independent accumulation/order: FP64 matrix product, then row/column scales.
    result = ((a.double()@b.double())*sa.double().reshape(-1,1)*sb.double().reshape(1,-1)).to(out_dtype)
    return result if bias is None else result+bias.to(out_dtype)


def _scaled_mm_cpu_harness(monkeypatch):
    h, checks = _fp8_group_cpu_harness(monkeypatch, 'scaled_mm')
    rand = torch.rand
    monkeypatch.setattr(torch, 'rand', lambda *a,**kw: rand(*a,**{**kw,'device':'cpu'}))
    return h, checks


def test_scaled_mm_independent_known_answer_and_unchanged_gate(monkeypatch):
    h, checks = _scaled_mm_cpu_harness(monkeypatch)
    values = [torch.tensor([[1.,2.],[3.,4.]]),torch.tensor([[5.,6.],[7.,8.]]),
              torch.tensor([2.,3.]),torch.tensor([.5,2.]),torch.tensor([1.,-1.])]
    expected = torch.tensor([[20.,87.],[65.5,299.]],dtype=torch.float16)
    checks.check_output(checks.reference(h,values,torch.float16),expected)
    checks.check_output(_scaled_mm_cpu(*values[:4],torch.float16,bias=values[4]),expected)
    allowed = expected.clone();allowed[0,0]+=.125
    checks.check_output(allowed,expected)
    allowed[0,0]+=1
    with pytest.raises(AssertionError): checks.check_output(allowed,expected)


@pytest.mark.parametrize('mode', ['correct','shape','dtype','nonfinite','wrong_values','ignore_bias',
                                 'ignore_scales','omit_partial_tiles','wrong_output_dtype',
                                 'mutate_a','mutate_b','mutate_sa','mutate_sb','mutate_bias'])
def test_scaled_mm_original_correctness_and_unscored_layouts(monkeypatch,mode):
    h, checks = _scaled_mm_cpu_harness(monkeypatch)
    calls = []
    def public(a,b,sa,sb,dtype,bias=None,**kwargs):
        calls.append((tuple(a.shape),tuple(b.shape),a.stride(),b.stride(),tuple(sa.shape),tuple(sb.shape),dtype,kwargs))
        if mode.startswith('mutate_'):
            target={'mutate_a':a,'mutate_b':b,'mutate_sa':sa,'mutate_sb':sb,'mutate_bias':bias}[mode]
            if target is not None: target.zero_()
        result=_scaled_mm_cpu(a,b,sa,sb,dtype,bias=bias)
        if mode=='shape':result=result.flatten()
        if mode=='dtype':result=result.double()
        if mode=='nonfinite':result.fill_(float('nan'))
        if mode=='wrong_values':result.zero_()
        if mode=='ignore_bias':result=_scaled_mm_cpu(a,b,sa,sb,dtype)
        if mode=='ignore_scales':result=(a.float()@b.float()).to(dtype)
        if mode=='omit_partial_tiles' and a.shape[0]==17:result[-1].add_(10)
        if mode=='wrong_output_dtype':result=result.half()
        return result
    mod=SimpleNamespace(triton_scaled_mm=public);h.load_module=lambda:mod
    checks.install(h)
    ok,reason=h.run_correctness()
    assert ok is (mode=='correct'),reason
    if mode=='correct':
        scored=[c for c in calls if c[0][0]!=17]
        assert [(c[0][0],c[0][1],c[1][1]) for c in scored]==[v[:3] for v in h.TEST_SHAPES]
        diagnostics=[c for c in calls if c[0][0]==17]
        assert len(diagnostics)==2
        assert all(c[0]==(17,35) and c[1]==(35,19) and c[2]==(70,1) and c[3]==(1,35) for c in diagnostics)
        assert diagnostics[0][4:6]==((1,),(19,)) and not diagnostics[0][7]['use_heuristic']
        assert diagnostics[1][6]==torch.float32
    assert mod.triton_scaled_mm is public


@pytest.mark.parametrize('mode',['correct','wrong_timed','stale','no_write','wrong_replay',
                                'mutate_timed','mutate_replay','raise_replay'])
def test_scaled_mm_original_timing_replay_and_all_readonly_buffers_restored(monkeypatch,mode):
    import inspect
    h,checks=_scaled_mm_cpu_harness(monkeypatch)
    h._TimedRun=module_at(ROOT/'src/tools/perf/aka_benchmark.py',monkeypatch).TimedRun
    mod=SimpleNamespace(triton_scaled_mm=_scaled_mm_cpu);h.load_module=lambda:mod
    values,snapshots,options=[],[],[]
    def benchmark(measured,*,timed_run,**kwargs):
        fn=inspect.getclosurevars(measured).nonlocals['fn'];state=inspect.getclosurevars(fn).nonlocals
        inputs=[state[k] for k in ('input_t','weight','scale_a','scale_b','bias')]
        values.append(inputs);snapshots.append(checks.snapshot(inputs));options.append(kwargs)
        output=measured();cached=output.clone()
        if mode=='wrong_timed':output.zero_()
        if mode=='mutate_timed':
            for value in inputs:
                if value is not None:value.zero_()
        def replay():
            if mode=='raise_replay':raise RuntimeError('Replay failed')
            if mode!='no_write':output.copy_(cached if mode=='stale' else measured())
            if mode=='wrong_replay':output.zero_()
            if mode=='mutate_replay':
                for value in inputs:
                    if value is not None:value.zero_()
            return output
        timed_run._bind(replay,output)
        return .125,{'benchmark_method':'cuda_graph'}
    h._benchmark_cuda_graph_or_events=benchmark;checks.install(h)
    rows=h.run_performance()
    assert len(rows)==5 and options==[dict(warmup=10,repetition=100)]*5
    for case,row in zip(h.TEST_SHAPES,rows):
        assert row['params']==dict(zip(('M','K','N','per_token_scale_a','per_channel_scale_b','has_bias'),case))
        assert row['execution_time_ms']==(.125 if mode=='correct' else -1.)
    for inputs,saved in zip(values,snapshots):checks.unchanged(inputs,saved)
    assert mod.triton_scaled_mm is _scaled_mm_cpu and h._benchmark_cuda_graph_or_events is benchmark


def test_scaled_mm_adapter_installs_task_local_checks(monkeypatch):
    h=module_at(ROOT/'tasks/triton2triton/vllm/triton_scaled_mm/_arena_eval.py',monkeypatch).load_harness()
    assert h.run_correctness.__module__==h.run_performance.__module__=='_scaled_mm_checks'


def _unpack_cpu(packed, lengths, **options):
    return torch.cat([packed[i,:int(n)] for i,n in enumerate(lengths.tolist())],dim=0)


def _unpack_cpu_harness(monkeypatch):
    p=ROOT/'tasks/triton2triton/vllm/triton_unpack_seq'
    h=module_at(p/'scripts/task_runner.py',monkeypatch);checks=module_at(p/'_arena_checks.py',monkeypatch)
    for name in ('randn','arange','empty','tensor'):
        factory=getattr(torch,name)
        monkeypatch.setattr(torch,name,lambda *a,_factory=factory,**kw:_factory(*a,**{**kw,'device':'cpu'}))
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    return h,checks


def test_unpack_seq_independent_known_answer_zero_prefix_and_rank(monkeypatch):
    h,checks=_unpack_cpu_harness(monkeypatch)
    packed=torch.arange(3*3*2,dtype=torch.float32).reshape(3,3,2)
    lens=torch.tensor([0,2,1],dtype=torch.int32)
    expected=torch.tensor([[6.,7.],[8.,9.],[12.,13.]])
    checks.check_output(checks.reference(h,packed,lens),expected)
    checks.check_output(_unpack_cpu(packed,lens),expected)
    ranked=packed.reshape(3,3,1,2)
    checks.check_output(checks.reference(h,ranked,lens),expected.reshape(3,1,2))
    empty=checks.reference(h,ranked,torch.zeros_like(lens))
    assert empty.shape==(0,1,2)
    checks.check_output(_unpack_cpu(ranked,torch.zeros_like(lens)),empty)


@pytest.mark.parametrize('mode',['correct','dtype','shape','nonfinite','wrong_values','mutate_packed',
                                'mutate_lengths','drop_second_time_block','drop_feature_tail','flatten_rank','reject_empty'])
def test_unpack_seq_original_correctness_and_unscored_partial_multidimensional_cases(monkeypatch,mode):
    h,checks=_unpack_cpu_harness(monkeypatch);calls=[]
    def public(packed,lengths,**kwargs):
        calls.append((tuple(packed.shape),lengths.tolist(),kwargs))
        if mode=='mutate_packed':packed.zero_()
        if mode=='mutate_lengths':lengths.fill_(1)
        result=_unpack_cpu(packed,lengths)
        if mode=='dtype':result=result.float()
        if mode=='shape':result=result.flatten()
        if mode=='nonfinite':result.fill_(float('nan'))
        if mode=='wrong_values':result.zero_()
        if mode=='drop_second_time_block' and packed.shape[1]>64:result[64].add_(10)
        if mode=='drop_feature_tail' and packed.ndim>3:result.reshape(result.shape[0],-1)[:,-1].add_(10)
        if mode=='flatten_rank':result=result.reshape(result.shape[0],-1)
        if mode=='reject_empty' and result.numel()==0:raise ValueError('Empty sequences unsupported')
        return result
    mod=SimpleNamespace(unpack_seq=public);h.load_module=lambda:mod;checks.install(h)
    ok,reason=h.run_correctness();assert ok is (mode=='correct'),reason
    if mode=='correct':
        scored=[c for c in calls if len(c[0])==3]
        assert [(c[0][0],c[1],c[0][2]) for c in scored]==h.TEST_SHAPES
        diag=[c for c in calls if len(c[0])==4]
        assert len(diag)==2 and diag[0][0]==(4,65,3,23) and diag[0][1]==[0,65,1,3]
        assert diag[0][2]==dict(block_t=32,block_d=32) and diag[1][1]==[0,0,0,0]
    assert mod.unpack_seq is public


@pytest.mark.parametrize('mode',['correct','wrong_timed','stale','no_write','wrong_replay',
                                'mutate_timed','mutate_replay','raise_replay'])
def test_unpack_seq_direct_jit_timing_grid_replay_and_full_buffer_restore(monkeypatch,mode):
    import inspect
    h,checks=_unpack_cpu_harness(monkeypatch)
    h._TimedRun=module_at(ROOT/'src/tools/perf/aka_benchmark.py',monkeypatch).TimedRun
    launches=[];values=[];saved=[];options=[]
    class Kernel:
        def __getitem__(self,grid):
            def launch(packed,out,lengths,B,Lmax,D,**kwargs):
                launches.append((grid,kwargs))
                out.copy_(_unpack_cpu(packed,lengths))
            return launch
    def forbidden_public(*args,**kwargs):raise AssertionError('Scored unit must remain original direct JIT launch')
    mod=SimpleNamespace(_unpack_seq_triton_kernel=Kernel(),unpack_seq=forbidden_public);h.load_module=lambda:mod
    def benchmark(measured,*,timed_run,**kwargs):
        fn=inspect.getclosurevars(measured).nonlocals['fn'];state=inspect.getclosurevars(fn).nonlocals
        packed,lengths,out=state['packed'],state['lengths'],state['out']
        values.append((packed,lengths,out));saved.append((packed.clone(),lengths.clone(),out.clone()));options.append(kwargs)
        actual=measured();cached=actual.clone()
        if mode=='wrong_timed':out.zero_()
        if mode=='mutate_timed':packed.zero_();lengths.fill_(1)
        def replay():
            if mode=='raise_replay':raise RuntimeError('Replay failed')
            if mode!='no_write':
                if mode=='stale':out.copy_(cached)
                else:measured()
            if mode=='wrong_replay':out.zero_()
            if mode=='mutate_replay':packed.zero_();lengths.fill_(1)
            return out
        timed_run._bind(replay,out)
        return .125,{'benchmark_method':'cuda_graph'}
    h._benchmark_cuda_graph_or_events=benchmark;checks.install(h)
    rows=h.run_performance()
    assert len(rows)==5 and options==[dict(warmup=10,repetition=100)]*5
    expected_grids=[]
    for case,row in zip(h.TEST_SHAPES,rows):
        B,lens,D=case
        assert row['params']==dict(B=B,D=D,lengths=lens)
        assert row['execution_time_ms']==(.125 if mode=='correct' else -1.)
        expected_grids.append((B,(max(lens)+63)//64,(D+63)//64))
    assert {grid for grid,kw in launches}==set(expected_grids)
    assert all(kw==dict(BLOCK_T=64,BLOCK_D=64,num_warps=4,num_stages=2) for grid,kw in launches)
    for (packed,lens,out),(old_packed,old_lens,old_out) in zip(values,saved):
        checks.unchanged(packed,lens,old_packed,old_lens)
        # Preallocated empty output may contain NaNs: compare storage bits.
        assert torch.equal(out.view(torch.int16),old_out.view(torch.int16))
    assert h._benchmark_cuda_graph_or_events is benchmark


def test_unpack_seq_adapter_installs_task_local_checks(monkeypatch):
    h=module_at(ROOT/'tasks/triton2triton/vllm/triton_unpack_seq/_arena_eval.py',monkeypatch).load_harness()
    assert h.run_correctness.__module__==h.run_performance.__module__=='_unpack_checks'


_CACHE_SCATTER_TASKS = ['reshape_and_cache_flash','reshape_and_cache_flash_diffkv']


def _cache_scatter_cpu(key,value,*args,kv_cache_dtype='auto',k_scale=None,v_scale=None,**options):
    different=len(args)==2
    caches=list(args[:-1]);slots=args[-1]
    head_major=not different and caches[0].ndim==5
    block_size=caches[0].shape[3] if head_major else caches[0].shape[1]
    selected=torch.nonzero(slots>=0).flatten();dest=slots[selected].long()
    k,v=key.float(),value.float()
    if kv_cache_dtype.startswith('fp8'):
        if not str(key.dtype).startswith('torch.float8_'):k=k/(1. if k_scale is None else k_scale)
        if not str(value.dtype).startswith('torch.float8_'):v=v/(1. if v_scale is None else v_scale)
    if different:
        data=[torch.cat([k,v],dim=-1)];canonical=[caches[0]]
    elif head_major:
        data=[k,v]
        canonical=[caches[0].permute(0,3,1,2,4).contiguous().reshape(caches[0].shape[0],block_size,key.shape[1],key.shape[2]),
                   caches[1].permute(0,3,1,2).contiguous()]
    else:data=[k,v];canonical=caches
    for target,source in zip(canonical,data):
        rows=target.reshape(-1,*target.shape[2:])
        converted=source[selected].to(target.dtype).contiguous()
        rows.view(torch.uint8).reshape(rows.shape[0],-1).index_copy_(0,dest,converted.view(torch.uint8).reshape(len(selected),-1))
    if head_major:
        caches[0].copy_(canonical[0].reshape(caches[0].shape[0],block_size,key.shape[1],key.shape[2]//caches[0].shape[-1],caches[0].shape[-1]).permute(0,2,3,1,4))
        caches[1].copy_(canonical[1].permute(0,2,3,1))


def _cache_scatter_cpu_harness(monkeypatch,symbol):
    p=ROOT/'tasks/triton2triton/vllm'/('triton_'+symbol)
    h=module_at(p/'scripts/task_runner.py',monkeypatch);checks=module_at(p/'_arena_checks.py',monkeypatch)
    for name in ('randn','arange','zeros','randperm','tensor'):
        factory=getattr(torch,name)
        monkeypatch.setattr(torch,name,lambda *a,_factory=factory,**kw:_factory(*a,**{**kw,'device':'cpu'}))
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    return h,checks


@pytest.mark.parametrize('symbol',_CACHE_SCATTER_TASKS)
def test_cache_scatter_independent_known_answer_preserves_untouched_slots(monkeypatch,symbol):
    h,checks=_cache_scatter_cpu_harness(monkeypatch,symbol)
    key=torch.tensor([[[1.,2.]],[[7.,8.]],[[3.,4.]]],dtype=torch.float16)
    value=-key;slots=torch.tensor([2,-1,0],dtype=torch.int64)
    caches=[torch.full((2,2,1,4),.25,dtype=torch.float16)] if checks.DIFFERENT_DIMS else [torch.full((2,2,1,2),.25,dtype=torch.float16),torch.full((2,2,1,2),-.25,dtype=torch.float16)]
    expected=checks.reference(h,key,value,caches,slots)
    _cache_scatter_cpu(key,value,*caches,slots)
    checks.check_caches(caches,expected)
    if checks.DIFFERENT_DIMS:
        assert caches[0][0,0,0].tolist()==[3.,4.,-3.,-4.]
        assert caches[0][1,0,0].tolist()==[1.,2.,-1.,-2.]
    else:
        assert caches[0][0,0,0].tolist()==[3.,4.]
        assert caches[1][1,0,0].tolist()==[-1.,-2.]
    assert (caches[0][0,1]==.25).all()
    # Scalar inputs and already-FP8 inputs participate in read-only checks.
    scalar=torch.tensor(.5);fp8=key.to(torch.float8_e4m3fnuz)
    checks.unchanged([scalar,fp8],[scalar.clone(),fp8.clone()])


@pytest.mark.parametrize('symbol,mode',[(s,m) for s in _CACHE_SCATTER_TASKS for m in
    ['correct','no_write','wrong_values','mutate_key','mutate_value','mutate_mapping','mutate_scale',
     'ignore_negative','wipe_untouched','ignore_fp8_scale','rescale_fp8_input','ignore_head_major']
    if m!='ignore_head_major' or not s.endswith('diffkv')])
def test_cache_scatter_original_correctness_and_public_layout_scale_paths(monkeypatch,symbol,mode):
    h,checks=_cache_scatter_cpu_harness(monkeypatch,symbol);calls=[]
    def public(key,value,*args,**options):
        caches=list(args[:-1]);slots=args[-1];calls.append((tuple(key.shape),[tuple(c.shape) for c in caches],key.dtype,dict(options)))
        if mode=='no_write':return
        if mode=='mutate_key':key.zero_()
        if mode=='mutate_value':value.zero_()
        if mode=='mutate_mapping':slots.zero_()
        if mode=='mutate_scale' and options.get('k_scale') is not None:options['k_scale'].mul_(2)
        if mode=='ignore_negative':slots=torch.where(slots<0,15,slots)
        if mode=='ignore_fp8_scale':options['kv_cache_dtype']='auto'
        if mode=='rescale_fp8_input' and str(key.dtype).startswith('torch.float8_'):key,value=key.float(),value.float()
        if mode=='ignore_head_major' and caches[0].ndim==5:return
        _cache_scatter_cpu(key,value,*caches,slots,**options)
        if mode=='wrong_values':
            for c in caches:c.zero_()
        if mode=='wipe_untouched' and key.shape[0]==7:
            for c in caches:
                floats=c.float();floats[floats==.25]=0;floats[floats==-.25]=0;c.copy_(floats)
    mod=SimpleNamespace(**{symbol:public});h.load_module=lambda:mod;checks.install(h)
    ok,reason=h.run_correctness();assert ok is (mode=='correct'),reason
    if mode=='correct':
        scored=[c for c in calls if c[0][0]!=7]
        assert [c[0] for c in scored]==[tuple(v[:3]) for v in h.TEST_SHAPES]
        diag=[c for c in calls if c[0][0]==7]
        assert len(diag)==(3 if checks.DIFFERENT_DIMS else 4)
        assert str(diag[-1][2]).startswith('torch.float8_')
        assert diag[-1][3]['kv_cache_dtype']=='fp8'
    assert getattr(mod,symbol) is public


@pytest.mark.parametrize('symbol',_CACHE_SCATTER_TASKS)
@pytest.mark.parametrize('mode',['correct','wrong_timed','stale','no_write','omit_key','omit_value','mutate_timed','mutate_replay','raise_replay'])
def test_cache_scatter_preserves_scored_zero_reset_and_poisons_only_actual_rerun(monkeypatch,symbol,mode):
    import inspect
    h,checks=_cache_scatter_cpu_harness(monkeypatch,symbol)
    h._TimedRun=module_at(ROOT/'src/tools/perf/aka_benchmark.py',monkeypatch).TimedRun
    replaying=False;current_caches=None;cached=None
    def public(key,value,*args,**options):
        caches=list(args[:-1]);slots=args[-1]
        if replaying and mode=='no_write':return
        if replaying and mode=='stale':
            for c,old in zip(caches,cached):c.copy_(old)
            return
        saved_outputs=checks.clone(caches)
        _cache_scatter_cpu(key,value,*caches,slots,**options)
        if replaying and mode in ('omit_key','omit_value'):
            if checks.DIFFERENT_DIMS:
                cut=key.shape[-1];sl=slice(None,cut) if mode=='omit_key' else slice(cut,None)
                caches[0][...,sl].copy_(saved_outputs[0][...,sl])
            else:
                index=0 if mode=='omit_key' else 1;caches[index].copy_(saved_outputs[index])
    mod=SimpleNamespace(**{symbol:public});h.load_module=lambda:mod
    records=[];options_seen=[]
    def benchmark(measured,*,timed_run,prepare_fn,**kwargs):
        nonlocal replaying,cached
        fn=inspect.getclosurevars(measured).nonlocals['fn'];state=inspect.getclosurevars(fn).nonlocals
        caches=[state['kv_cache']] if checks.DIFFERENT_DIMS else [state['key_cache'],state['value_cache']]
        readonly=[state[k] for k in ('key','value','slot_mapping','k_scale','v_scale')]
        records.append((readonly,checks.clone(readonly),caches,checks.clone(caches)));options_seen.append(kwargs)
        replaying=False
        prepare_fn();assert all((c==0).all() for c in caches) # No poisoning in the scored preparation.
        outputs=measured();cached=checks.clone(caches)
        if mode=='wrong_timed':
            for c in caches:c.zero_()
        if mode=='mutate_timed':readonly[0].zero_()
        def replay():
            nonlocal replaying
            replaying=True;prepare_fn()
            mapping=readonly[2];selected=mapping[mapping>=0]
            for c in caches:
                flat=c.view(-1,c.shape[-2]*c.shape[-1]);mask=torch.ones(flat.shape[0],dtype=torch.bool);mask[selected]=False
                assert torch.isnan(flat[selected]).all() and (flat[mask]==0).all()
            if mode=='raise_replay':raise RuntimeError('Replay failed')
            result=measured()
            if mode=='mutate_replay':readonly[1].zero_();readonly[3].zero_()
            return result
        timed_run._bind(replay,outputs)
        return .125,{'benchmark_method':'cuda_graph'}
    h._benchmark_cuda_graph_or_events=benchmark;checks.install(h);rows=h.run_performance()
    assert len(rows)==5 and options_seen==[dict(warmup=10,repetition=100)]*5
    keys=('num_tokens','num_heads','head_size_k','head_size_v','num_blocks','block_size') if checks.DIFFERENT_DIMS else ('num_tokens','num_heads','head_size','num_blocks','block_size')
    for case,row in zip(h.TEST_SHAPES,rows):
        assert row['params']==dict(zip(keys,case))
        assert row['execution_time_ms']==(.125 if mode=='correct' else -1.)
        if mode=='correct':assert row['scored_reset_unchanged'] and row['replay_poison_after_original_prepare']
    for values,old,caches,saved in records:checks.unchanged(values,old);checks.unchanged(caches,saved)
    assert h._benchmark_cuda_graph_or_events is benchmark


@pytest.mark.parametrize('symbol',_CACHE_SCATTER_TASKS)
def test_cache_scatter_adapter_installs_task_local_checks(monkeypatch,symbol):
    h=module_at(ROOT/'tasks/triton2triton/vllm'/('triton_'+symbol)/'_arena_eval.py',monkeypatch).load_harness()
    assert h.run_correctness.__module__==h.run_performance.__module__=='_cache_scatter_checks'
