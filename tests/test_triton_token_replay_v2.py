"""CPU contract/oracle and measured-replay negative controls for 20 token tasks.

These tests do not claim GPU compilation, graph timing or semantic validator PASS.
"""
import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import types

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / 'tasks/triton2triton/vllm'
NAMES = ('bad_words combine_sampled_and_draft_tokens logit_bias min_p pack_seq penalties '
         'post_update prepare_eagle_docode prepare_eagle_inputs prepare_mrope_positions '
         'prepare_pos_seq_lens prepare_prefill_inputs prompt_logprobs_token_ids ranks '
         'rejection_greedy_sample rejection_sample temperature topk_topp update_eagle_inputs '
         'write_zeros_to_output').split()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    cwd = os.getcwd()
    try:
        spec.loader.exec_module(module)
    finally:
        os.chdir(cwd)
    return module


def modules(name):
    task = TASKS / ('triton_' + name)
    return (load(task / '_arena_replay.py', 'replay_' + name),
            load(task / '_arena_contract.py', 'contract_' + name),
            load(task / 'scripts/task_runner.py', 'harness_' + name))


@pytest.mark.parametrize('name', NAMES)
def test_original_cases_and_benchmark_contract_preserved(name):
    task = TASKS / ('triton_' + name)
    manifest = json.loads((task / 'workloads.json').read_text())
    _, contract, harness = modules(name)
    assert manifest['input_table'] == json.loads(json.dumps(harness.TEST_SHAPES))
    controls = [row for row in manifest['cases'] if row['test_case_id'] == 'contract_controls']
    assert len(controls) == 1 and controls[0]['checks'] == ['correctness']
    assert controls[0]['params']['case_index'] == contract.CONTROL_INDEX
    assert harness.WARMUP_ITERATIONS == 10 and harness.BENCHMARK_ITERATIONS == 100
    tree = ast.parse((task / 'source' / ('triton_' + name + '.py')).read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == contract.FUNCTION)
    for args in contract.control_inputs(harness):
        assert len(args) == len(function.args.args)
    # Never depend on the Arena repository at task runtime.
    for path in (task / '_arena_contract.py', task / '_arena_replay.py'):
        tree = ast.parse(path.read_text())
        assert not any(isinstance(n, ast.ImportFrom) and (n.module or '').split('.')[0] in ('src', 'agents') for n in ast.walk(tree))


@pytest.mark.parametrize('name', NAMES)
def test_oracles_do_not_mutate_inputs_and_enforce_metadata(name):
    replay, contract, harness = modules(name)
    for args in contract.control_inputs(harness):
        saved = replay.clone(args)
        answer = replay.expected(harness, contract, args)
        replay.unchanged(args, saved, ())
        replay.check(harness, contract, replay.clone(answer), answer, args)
        bad = replay.clone(answer)
        if isinstance(bad, tuple):
            bad = (bad[0].double(), *bad[1:])
        else:
            bad = bad.double()
        with pytest.raises(AssertionError):
            replay.check(harness, contract, bad, answer, args)


def test_small_independent_known_answers():
    expected = {
        'temperature': [[2,-4,6,8],[2,1,-1,3],[2,3,-1,0],[2,3,4,5]],
        'min_p': [[-torch.inf,1,2,3],[0,1,2,3],[-torch.inf,-torch.inf,-torch.inf,3],
                  [-torch.inf,-torch.inf,-torch.inf,3],[-torch.inf,1,2,3]],
        'ranks': [3,4,4],
        'prompt_logprobs_token_ids': [35,3,4,5,22,23],
        'rejection_greedy_sample': [[80,-1,-1,-1],[-1,-1,-1,-1],[10,11,12,82],[20,99,-1,-1]],
        'prepare_mrope_positions': [[94,3,4,5,61,62,-9],[104,3,4,5,71,72,-9],[114,3,4,5,81,82,-9]],
    }
    for name, values in expected.items():
        _, contract, harness = modules(name)
        answer = contract.reference(harness, next(contract.control_inputs(harness)))
        assert torch.equal(answer, torch.tensor(values, dtype=answer.dtype))
    _, contract, harness = modules('combine_sampled_and_draft_tokens')
    out, indices = contract.reference(harness, next(contract.control_inputs(harness)))
    assert out.tolist() == [-9,-9,10,1,12,7,8] and indices.tolist() == [0,2,3,4,5,6]
    _, contract, harness = modules('prepare_pos_seq_lens')
    pos, seq = contract.reference(harness, next(contract.control_inputs(harness)))
    assert pos.tolist() == [4,2,3,4,7,8] and seq.tolist() == [5,5,9,0,0]
    _, contract, harness = modules('prepare_prefill_inputs')
    ids, next_tokens = contract.reference(harness, next(contract.control_inputs(harness)))
    assert ids.tolist() == [34,2,3,4,-9,-9] and next_tokens.tolist() == [81,82,83,35]


def test_bad_word_speculative_prefix_controls():
    _, contract, harness = modules('bad_words')
    answer = contract.reference(harness, next(contract.control_inputs(harness)))
    assert torch.isneginf(answer).nonzero().tolist() == [[1,10],[2,9],[3,8]]


def test_pack_padding_and_copy_known_answers():
    _, contract, harness = modules('pack_seq')
    for args in contract.control_inputs(harness):
        out = contract.reference(harness, args)
        assert out[0,0].tolist() == [0,1,2]
        assert out[1].tolist() == [[3,4,5],[6,7,8],[9,10,11]]
        assert out[2,0].tolist() == [12,13,14]
        assert torch.all(out[0,1:] == args[2]) and torch.all(out[2,1:] == args[2])


def test_post_update_zero_accept_and_routing_known_answer():
    _, contract, harness = modules('post_update')
    nct, last, hist, tokens, length = contract.reference(harness, next(contract.control_inputs(harness)))
    assert nct.tolist() == [4,4,7,6] and last.tolist() == [4,11,8,13]
    assert length.tolist() == [3,3,6,5]
    assert hist.nonzero().tolist() == [[0,4],[2,7],[2,8]]
    assert tokens[0,2] == 4 and tokens[2,4:6].tolist() == [7,8]
    assert torch.all(tokens[3] == -9)  # Zero accepted tokens retain history.


def test_eagle_clamps_copies_and_padding_known_answers():
    for name in ('prepare_eagle_docode', 'update_eagle_inputs'):
        _, contract, harness = modules(name)
        args = next(contract.control_inputs(harness))
        result = contract.reference(harness, args)
        if name == 'prepare_eagle_docode':
            pos, seq, qsl, ids, hs = result
            assert pos[:3].tolist() == [31,31,31] and seq.tolist() == [32,32,32,0,0]
            assert qsl.tolist() == [0,1,2,3,3,3]
            assert torch.equal(hs[:3], args[1][args[2].long()])
        else:
            ids, pos, hs, seq = result
            assert pos.tolist() == [15,15,15] and seq.tolist() == [16,16,16]
            assert torch.equal(hs, args[1])
        assert torch.equal(ids[:3], args[0])
    _, contract, harness = modules('prepare_eagle_inputs')
    args = next(contract.control_inputs(harness))
    last, ids, pos = contract.reference(harness, args)
    assert last.tolist() == [3,6,9]
    assert ids[3] == args[4][4] and ids[6] == args[3][0] and ids[9] == args[4][2]
    assert pos.tolist() == [0,1,2,3,0,1,2,0,0,1,0,0]


def test_penalty_speculative_history_and_disabled_branch():
    _, contract, harness = modules('penalties')
    args = next(contract.control_inputs(harness))
    out = contract.reference(harness,args)
    # Request 0 disables all penalties; request 2 adds speculative token 4.
    torch.testing.assert_close(out[2:4],args[0][2:4],atol=0,rtol=0)
    assert out[0,3].item() == pytest.approx((-1.85*0.8)+1+1)
    assert out[1,4].item() == pytest.approx((-1.0*0.8)+0.5+1)


def test_logit_bias_allows_masks_and_stop_minimum():
    _, contract, harness = modules('logit_bias')
    out = contract.reference(harness,next(contract.control_inputs(harness)))
    assert torch.isfinite(out[0]).nonzero().flatten().tolist() == [0,2]
    assert out[0,2].item() == pytest.approx(2.2)
    assert out[1,1].item() == pytest.approx(1.4)
    assert torch.isneginf(out[2,2])
    assert torch.isfinite(out[3]).nonzero().flatten().tolist() == [1,3]
    assert out[3,3].item() == pytest.approx(1.7)


def test_rejection_defined_prefix_only_and_wrong_count_rejected():
    replay, contract, harness = modules('rejection_sample')
    args = next(contract.control_inputs(harness))
    answer = contract.reference(harness,args)
    assert answer[1].tolist() == [4,1,2]
    assert answer[0][0].tolist() == [10,11,12,13]
    assert answer[0][1,0] == 20 and answer[0][2,:2].tolist() == [30,31]
    changed = replay.clone(answer)
    changed[0][1,1:] = -123
    replay.check(harness,contract,changed,answer,args)  # Undefined suffix is not data.
    changed[1][1] = 2
    with pytest.raises(AssertionError):
        replay.check(harness,contract,changed,answer,args)


@pytest.mark.parametrize('name', ['min_p','topk_topp','logit_bias','bad_words'])
def test_nan_cannot_masquerade_as_masked_logit(name):
    replay, contract, harness = modules(name)
    args = next(contract.control_inputs(harness))
    answer = contract.reference(harness,args)
    bad = answer.clone()
    index = torch.isneginf(bad).nonzero()[0]
    bad[tuple(index)] = float('nan')
    with pytest.raises(AssertionError):
        replay.check(harness,contract,bad,answer,args)


def test_zero_subnormal_is_rejected():
    replay, contract, harness = modules('write_zeros_to_output')
    args = next(contract.control_inputs(harness))
    answer = contract.reference(harness,args)
    bad = torch.zeros_like(answer)
    bad[0,0] = torch.nextafter(torch.tensor(0.),torch.tensor(1.))
    with pytest.raises(AssertionError):
        replay.check(harness,contract,bad,answer,args)


def test_topk_original_boundary_allowance_preserved_but_large_error_rejected():
    replay, contract, harness = modules('topk_topp')
    args = (torch.arange(256,dtype=torch.float32)[None,:],torch.tensor([8],dtype=torch.int32),None,-torch.inf)
    answer = contract.reference(harness,args)
    one_boundary = answer.clone(); one_boundary[0,247]=247
    replay.check(harness,contract,one_boundary,answer,args)
    with pytest.raises(AssertionError):
        replay.check(harness,contract,args[0],answer,args)


class Timed:
    def _bind(self, fn, output):
        self.fn, self.outputs = fn, output
    def rerun(self):
        self.outputs = self.fn()
        return self.outputs


def fake_benchmark(fn, timed_run, prepare_fn=None, **kwargs):
    def run():
        if prepare_fn:
            prepare_fn()
        return fn()
    output = run()
    timed_run._bind(run,output)
    return 1.0, {'benchmark_method':'cpu_test_only'}


def fake_function(replay, contract, harness, arity, *, cached=False, fail=False):
    state = {}
    def impl(args):
        if fail and 'called' in state:
            raise RuntimeError('injected timing failure')
        state['called'] = True
        answer = replay.expected(harness,contract,args)
        if cached:
            state.setdefault('answer',replay.clone(answer))
            answer = replay.clone(state['answer'])
        if contract.FUNCTION == 'combine_sampled_and_draft_tokens':
            args[0].copy_(answer[0]); return answer[1]
        if contract.MUTABLE:
            outputs = answer if len(contract.MUTABLE)>1 else (answer,)
            for index,value in zip(contract.MUTABLE,outputs):
                args[index].copy_(value)
            return args[0] if len(contract.MUTABLE)==1 else None
        return answer
    ns={'impl':impl}
    names=','.join('arg'+str(i) for i in range(arity))
    exec('def fn('+names+'):\n    return impl(('+names+',))',ns)
    return ns['fn']


@pytest.mark.parametrize('name', [n for n in NAMES if n not in ('pack_seq','topk_topp')])
def test_actual_replay_and_finally_restore_cpu_simulation(name):
    replay, contract, harness = modules(name)
    args = next(contract.control_inputs(harness))
    pristine = replay.clone(args)
    recorder = replay.Recorder(harness,contract)
    harness._TimedRun = Timed
    fn = recorder.wrap(fake_function(replay,contract,harness,len(args)))
    _, metadata = recorder.benchmark(fake_benchmark,lambda: fn(*args))
    assert metadata['timed_output_checked'] and metadata['input_state_restored']
    replay.unchanged(args,pristine,())


@pytest.mark.parametrize('name', ['temperature','ranks','post_update','prepare_eagle_inputs','rejection_sample'])
def test_cached_answer_rejected_and_state_restored(name):
    replay, contract, harness = modules(name)
    args = next(contract.control_inputs(harness)); pristine = replay.clone(args)
    recorder = replay.Recorder(harness,contract); harness._TimedRun=Timed
    fn = recorder.wrap(fake_function(replay,contract,harness,len(args),cached=True))
    with pytest.raises(AssertionError):
        recorder.benchmark(fake_benchmark,lambda: fn(*args))
    replay.unchanged(args,pristine,())


def test_exception_restores_buffers_and_readonly_mutation_fails():
    replay, contract, harness = modules('temperature')
    args = next(contract.control_inputs(harness)); pristine = replay.clone(args)
    recorder = replay.Recorder(harness,contract); harness._TimedRun=Timed
    fn=recorder.wrap(fake_function(replay,contract,harness,len(args),fail=True))
    with pytest.raises(RuntimeError,match='injected'):
        recorder.benchmark(fake_benchmark,lambda:fn(*args))
    replay.unchanged(args,pristine,())
    args[1][0]=0
    with pytest.raises(AssertionError,match='read-only'):
        replay.unchanged(args,pristine,contract.MUTABLE)


@pytest.mark.parametrize('name', ['pack_seq','topk_topp'])
@pytest.mark.parametrize('cached', [False,True])
def test_direct_preallocated_launch_is_observed_and_checked(name,cached):
    replay,contract,harness=modules(name)
    harness._TimedRun=Timed
    recorder=replay.Recorder(harness,contract)
    memory={}
    if name=='pack_seq':
        x=torch.arange(15,dtype=torch.float16).reshape(5,3)
        lengths=torch.tensor([1,3,1],dtype=torch.int32)
        out=torch.empty((3,3,3),dtype=x.dtype)
        def launch():
            value=harness.reference_pack_seq(x,lengths.tolist(),0.)
            memory.setdefault('answer',value.clone())
            out.copy_(memory['answer'] if cached else value)
        args=(x,lengths)
    else:
        logits=torch.arange(256,dtype=torch.float32)[None,:]
        k=torch.tensor([32],dtype=torch.int32)
        kernel_args=(logits,None,None,None,k,None)
        kernel_meta={'TOPK_ENABLED':True,'TOPP_ENABLED':False,'MASK_VALUE':-torch.inf}
        def launch():
            value=harness.reference_apply_top_k_top_p(kernel_args[0],kernel_args[4],None)
            assert kernel_meta['TOPK_ENABLED']
            memory.setdefault('answer',value.clone())
            kernel_args[0].copy_(memory['answer'] if cached else value)
        args=(logits,k)
    before=replay.clone(args)
    if cached:
        with pytest.raises(AssertionError):
            recorder.benchmark(fake_benchmark,launch)
    else:
        recorder.benchmark(fake_benchmark,launch)
    replay.unchanged(args,before,())


@pytest.mark.parametrize('name',NAMES)
def test_original_manifest_rows_and_generated_region_are_byte_preserved(name):
    task=TASKS/('triton_'+name)
    relative=task.relative_to(ROOT).as_posix()
    old=json.loads(subprocess.check_output(['git','show','4a056113:'+relative+'/workloads.json'],cwd=ROOT,text=True))
    current=json.loads((task/'workloads.json').read_text())
    assert current['cases'][:-1]==old['cases']
    assert {k:v for k,v in current.items() if k!='cases'}=={k:v for k,v in old.items() if k!='cases'}
    before=subprocess.check_output(['git','show','4a056113:'+relative+'/scripts/task_runner.py'],cwd=ROOT,text=True)
    after=(task/'scripts/task_runner.py').read_text()
    start='# >>> AKA-GENERATED:'; end='# <<< AKA-GENERATED <<<'
    assert before.split(start)[1].split(end)[0]==after.split(start)[1].split(end)[0]
    candidate=task/'source'/('triton_'+name+'.py')
    assert candidate.read_bytes()==subprocess.check_output(['git','show','4a056113:'+candidate.relative_to(ROOT).as_posix()],cwd=ROOT)


@pytest.mark.parametrize('is_prefill',[False,True])
def test_mrope_scored_setup_honors_declared_scenario(is_prefill):
    path=TASKS/'triton_prepare_mrope_positions/scripts/task_runner.py'
    fn=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='run_performance')
    assignments=[n for n in ast.walk(fn) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ('prefill_lens','num_computed_tokens') for t in n.targets)]
    assert len(assignments)==2
    ns=dict(torch=torch,max_num_reqs=8,max_model_len=128,device='cpu',is_prefill=is_prefill)
    exec(compile(ast.Module(body=assignments,type_ignores=[]),str(path),'exec'),ns)
    assert torch.all(ns['prefill_lens']==(128 if is_prefill else 10))
    assert torch.all(ns['num_computed_tokens']==(0 if is_prefill else 50))
