"""MXFP: encoding known answers, immutable inputs and the actual timer/replay adapter."""
import ast
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import types

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT/'tasks/triton2triton/rocmbench/hard/test_matmul_MXFP'


def load(path):
    spec = importlib.util.spec_from_file_location('_mxfp_' + path.stem, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module', autouse=True)
def cpu_budget():
    threads = torch.get_num_threads(); state = torch.random.get_rng_state()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads); torch.random.set_rng_state(state)


@pytest.fixture
def ref(monkeypatch):
    module = load(TASK/'_arena_reference.py')
    monkeypatch.setitem(sys.modules, '_arena_reference', module)
    return module


def protected_functions(names, namespace):
    tree = ast.parse((TASK/'test_matmul_MXFP.py').read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    for n in nodes: n.decorator_list = []
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<protected-mxfp>', 'exec'), namespace)
    return namespace


@pytest.mark.parametrize('encoding', ['e2m1', 'e4m3', 'e5m2'])
def test_decoder_independent_known_answers(encoding):
    namespace = protected_functions(['mxfp_to_bf16_torch'], {'torch': torch})
    # Hand-decoded signed values, deliberately independent of decoder bit shifts.
    codes = {'e2m1': [0x12, 0xab, 0x67, 0xfe], 'e4m3': [0x30, 0xb8, 0x40, 0xc4], 'e5m2': [0x38, 0xbc, 0x40, 0xc2]}[encoding]
    values = [.5, 1., -1., -1.5, 4., 6., -6., -4.] if encoding == 'e2m1' else [.5, -1., 2., -3.]
    packed = torch.tensor([codes]*3, dtype=torch.uint8)
    scales = torch.tensor([126, 127, 128], dtype=torch.uint8)
    expected = torch.tensor(values)[None, :] * torch.tensor([.5, 1., 2.])[:, None]
    actual = namespace['mxfp_to_bf16_torch'](packed, scales, encoding)
    assert torch.equal(actual.float(), expected)


def test_independent_scaled_matrix_known_answer():
    ns = protected_functions(['mxfp_to_bf16_torch', 'dot_scale_ref'], {'torch': torch})
    a = torch.full((2, 32), 0x22, dtype=torch.uint8)
    scale = torch.tensor([[127, 128], [126, 127]], dtype=torch.uint8)
    b = torch.full((64, 3), 0x3c, dtype=torch.uint8)
    expected = torch.tensor([[96., 96., 96.], [48., 48., 48.]], dtype=torch.bfloat16)
    assert torch.equal(ns['dot_scale_ref'](a, scale, b, 'e2m1', 'e5m2'), expected)


def inputs(dtype=torch.float32):
    a = torch.tensor([[2, -3, 4], [1, 5, -2]], dtype=dtype).T.contiguous().T
    b = torch.tensor([[1, 3], [2, -1], [-2, 4]], dtype=dtype)
    return a, b, torch.full((2, 2), -7, dtype=torch.float16)


@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
def test_fp32_loads_full_output_and_metadata(ref, dtype):
    a, b, out = inputs(dtype); check = ref.prepare(dict(is_scaled_mode=False, a_tensor=a, b_tensor=b, output_buffer=out), None)
    known = torch.tensor([[-12, 25], [15, -10]], dtype=torch.float16)
    assert torch.equal(check.expected, known)
    out.copy_(known); check(out)
    out[-1, -1] += 1
    with pytest.raises(ref.NumericalMismatch): check(out)
    with pytest.raises(ValueError, match='Unexpected'): check(out.float())
    out.copy_(known); a[0, 0] += 1
    with pytest.raises(ValueError, match='Read-only'): check(out)
    check.restore()
    a = torch.tensor([[1.0004, .123456]], dtype=torch.float32); b = torch.tensor([[1.0004], [10.]], dtype=torch.float32); out = torch.zeros((1, 1), dtype=torch.float16)
    check = ref.prepare(dict(is_scaled_mode=False, a_tensor=a, b_tensor=b, output_buffer=out), None)
    assert torch.equal(check.expected, (a @ b).half())
    assert not torch.equal(check.expected, a.half() @ b.half())


@pytest.mark.parametrize('mode', ['graph', 'events', 'cached', 'no_write', 'wrong_timed', 'wrong_replay', 'mutation', 'crash', 'bad_timing'])
def test_real_timer_replay_and_finally_restore(ref, monkeypatch, mode):
    adapter = load(TASK/'_arena_eval.py'); a, b, out = inputs()
    original = [x.clone() for x in (a, b, out)]; phase = ['initial']; cached = (a @ b).half(); timer_args = []
    plugin = types.SimpleNamespace(action='performance', current_row={'test_case_id': 'cpu'}, exercised=set())
    class Base:
        def __init__(self, fn):
            self.op_callable = fn; self.prepare_fn = None; self.use_cuda_graph = mode != 'events'
            self.fallback_reason = 'explicit events' if mode == 'events' else None
            self.config = types.SimpleNamespace(warm_up=10, repetition=100)
    class Timed:
        outputs = None
        def rerun(self):
            phase[0] = 'replay'
            if mode == 'crash': raise RuntimeError('injected replay crash')
            return op()
    def timer(fn, **kw):
        timer_args.append(kw); phase[0] = 'timed'; kw['timed_run'].outputs = fn()
        return ([float('nan')] if mode == 'bad_timing' else [1., 2.]), {'benchmark_method': 'cuda_event_fallback' if mode == 'events' else 'cuda_graph'}
    monkeypatch.setitem(sys.modules, '_aka_benchmark', types.SimpleNamespace(TimedRun=Timed, benchmark_cuda_graph_or_events_samples=timer))
    monkeypatch.setitem(sys.modules, 'performance_utils_pytest', types.SimpleNamespace(_compute_timing_stats=lambda ts, cfg: {'mean': sum(ts)/len(ts)}))
    def op():
        if not (mode == 'no_write' and phase[0] == 'replay'): out.copy_(cached if mode == 'cached' else a @ b)
        if mode == 'wrong_' + phase[0]: out[-1, -1] += 10
        if mode == 'mutation' and phase[0] == 'replay': a[0, 0] += 1
        return out
    bench = adapter.benchmark_type(Base, plugin, None)(op); bench.context = dict(is_scaled_mode=False, a_tensor=a, b_tensor=b, output_buffer=out)
    if mode in ['graph', 'events']:
        bench.run_benchmark(baseline_callable=lambda: pytest.fail('peer baseline'))
        assert all(bench_value for key, bench_value in plugin.current_row['metadata'].items() if key.endswith('_checked') or key.endswith('_restored'))
        helper = load(ROOT/'src/tools/perf/performance_utils_pytest.py'); previous = []
        monkeypatch.setattr(helper, 'benchmark_cuda_graph_or_events_samples', lambda fn, **kw: (previous.append(kw) or [1.], {}))
        helper._measure_times(op, bench.config, prepare_fn=None, use_cuda_graph=bench.use_cuda_graph, fallback_reason=bench.fallback_reason)
        sig = inspect.signature(load(ROOT/'src/tools/perf/aka_benchmark.py').benchmark_cuda_graph_or_events_samples)
        def effective(kwargs):
            bound = sig.bind_partial(None, **kwargs); bound.apply_defaults()
            return {k: v for k, v in bound.arguments.items() if k not in ['fn', 'timed_run']}
        assert effective(previous[0]) == effective(timer_args[0])
    else:
        with pytest.raises((ValueError, RuntimeError, AssertionError)): bench.run_benchmark()
        assert not plugin.exercised
    assert all(ref.equal_bytes(x, y) for x, y in zip((a, b, out), original))


class CPUTorch:
    def __getattr__(self, name):
        original = getattr(torch, name)
        if name not in ['empty', 'randn', 'randint', 'arange', 'tensor', 'full']: return original
        def call(*args, **kwargs):
            kwargs.pop('device', None)
            return original(*args, **kwargs)
        return call


@pytest.mark.parametrize('encoding', ['e2m1', 'e4m3', 'e5m2', 'original'])
@pytest.mark.parametrize('mode', ['good', 'cache', 'tail', 'input_mutation', 'no_replay_write'])
def test_actual_converter_bodies_reject_malicious_candidates(ref, encoding, mode):
    ns = {'torch': CPUTorch(), 'triton': types.SimpleNamespace(cdiv=lambda a,b: (a+b-1)//b), 'set_seed': lambda: None, 'result_gold': {}}
    protected_functions(['mxfp_to_bf16_torch', 'test_converter_encoding_control', 'test_mxfp_to_bf16_numerical_correctness'], ns)
    calls = []; before = []
    class Kernel:
        def __getitem__(self, grid):
            def launch(x, s, out, rows, e, m, block, **kw):
                if not before: before.extend([(z, z.clone()) for z in (x, s, out)])
                dtype = {(2,1): 'e2m1', (4,3): 'e4m3', (5,2): 'e5m2'}[(e,m)]
                answer = ns['mxfp_to_bf16_torch'](x, s, dtype).float()
                calls.append(answer.clone())
                if not (mode == 'no_replay_write' and len(calls) > 1): out.copy_(calls[0] if mode == 'cache' else answer)
                if mode == 'tail': out[-1, -1] += 10
                if mode == 'input_mutation': x[0,0] ^= 1
            return launch
    ns['mxfp_to_bf16_kernel'] = Kernel(); request = types.SimpleNamespace(node=types.SimpleNamespace(name='cpu', user_properties=[]))
    def run():
        if encoding == 'original': ns['test_mxfp_to_bf16_numerical_correctness'](request)
        else: ns['test_converter_encoding_control'](encoding, 2 if encoding=='e2m1' else 4 if encoding=='e4m3' else 5, 1 if encoding=='e2m1' else 3 if encoding=='e4m3' else 2, request)
    if mode == 'good':
        run(); assert len(calls)==2 and request.node.user_properties
    else:
        with pytest.raises((ValueError, AssertionError)): run()
        assert not request.node.user_properties
    assert all(ref.equal_bytes(x, snapshot) for x, snapshot in before)


@pytest.mark.parametrize('scale', [False, True])
@pytest.mark.parametrize('mode', ['good', 'cache', 'zero', 'tail', 'input_mutation'])
def test_actual_pipeline_pristine_oracle_and_scaled_known_answer(ref, scale, mode):
    ns = {'torch': CPUTorch(), 'triton': types.SimpleNamespace(cdiv=lambda a,b: (a+b-1)//b), 'set_seed': lambda: None, 'result_gold': {},
          'check_capabilities': lambda: None, 'is_hopper': lambda: False, 'is_hip_mi200': lambda: False}
    protected_functions(['mxfp_to_bf16_torch', 'dot_scale_ref', 'test_pipeline_matmul'], ns)
    calls=[]; before=[]
    class Kernel:
        def __getitem__(self, grid):
            def launch(a, s, b, out, *args, **kw):
                if not before: before.extend([(z, z.clone()) for z in [a,b,out]+([s] if s is not None else [])])
                answer = ns['dot_scale_ref'](a,s,b,kw['a_type'],kw['b_type']) if s is not None else a@b
                calls.append(answer.clone()); out.copy_(calls[0] if mode=='cache' else answer)
                if mode=='zero': out.zero_()
                if mode=='tail': out[-1,-1]+=10
                if mode=='input_mutation': a[0,0]+=1
            return launch
    ns['matmul_kernel']=Kernel(); request=types.SimpleNamespace(node=types.SimpleNamespace(name='cpu',user_properties=[]))
    if mode=='good':
        ns['test_pipeline_matmul'](scale,request,device='cpu'); assert len(calls)==(3 if scale else 2)
        assert request.node.user_properties
    else:
        with pytest.raises((ValueError,AssertionError)): ns['test_pipeline_matmul'](scale,request,device='cpu')
        assert not request.node.user_properties
    assert all(ref.equal_bytes(x,snapshot) for x,snapshot in before)


def test_original_kernel_scoring_manifest_and_numeric_gates(ref, monkeypatch):
    source=(TASK/'test_matmul_MXFP.py').read_text();nodes={n.name:n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}
    for name,sha in ORIGINAL['functions'].items(): assert hashlib.sha256(ast.get_source_segment(source,nodes[name]).encode()).hexdigest()==sha
    for name,sha in ORIGINAL['files'].items(): assert hashlib.sha256((TASK/name).read_bytes()).hexdigest()==sha
    rows=json.loads((TASK/'workloads.json').read_text())['cases']
    assert len(rows)==12 and sum('performance' in row['checks'] for row in rows)==6
    assert hashlib.sha256(json.dumps(rows[:9],sort_keys=True,separators=(',',':')).encode()).hexdigest()==ORIGINAL['rows']
    adapter=load(TASK/'_arena_eval.py')
    decorator=nodes['test_converter_encoding_control'].decorator_list[0]
    actual={adapter.identity('test_converter_encoding_control',dict(zip(['encoding','e_bits','m_bits'],row))) for row in ast.literal_eval(decorator.args[1])}
    assert actual=={r['test_case_id'] for r in rows[9:]}
    assert all(r['checks']==['correctness'] and not r['coverage']['scored'] for r in rows[9:])
    assert 'assert_close(output, reference, atol=0.0, rtol=0.0)' in source
    assert 'assert_close(ref_out, output, atol=atol, rtol=rtol, equal_nan=scale)' in source
    monkeypatch.setattr(pytest,'main',lambda *args,**kwargs:0)
    from src.task_protocol import parse_command_result
    result=adapter.evaluate('task','validate-task')
    assert parse_command_result('ARENA_EVAL_RESULT='+json.dumps(result),role='task',action='validate-task',returncode=0).status=='PASS'


# Stable checked-in expectations; CI does not depend on Git history.
ORIGINAL = {'files': {'config.yaml': '80c1baf0b084c027ad773ce8a30ddf8e05c77bb4560a65c2e95204251795e4fc',
           'performance_utils_pytest.py': 'e0dfef878da9ac042c39f1832e7940520fa1738ab29d9af2fcdbbc7e42d146c9'},
 'functions': {'calculate_mxfp_matmul_gbps': 'c811beae77e3f8310d4404b6153eb930b829b6a129d1068c1530c949c5ad527c',
               'calculate_mxfp_matmul_tflops': '4181c5f41b58e59f10bedd2de5123f0e5116dbb809ef3f45ef5e1451f7aebef5',
               'check_capabilities': 'b8d446058ae8847018dce41280eae54199e8665175885b95fe1c8f19510582a5',
               'dot_scale_ref': 'bfeab3b05129b13d0932539ba05f1a21c4e91e580350e8ea627db5cdfe78cad9',
               'get_torch_dtype_from_str': '904e9f35c23a1ac99a8ed171485983539c76fa5cbc3ed23f19d6908e1bc20825',
               'is_cuda': '344ddb015296f2ac8a4ec12629daceefbddc48714bc9071e21aa54eec0864218',
               'is_hip': '18a1484321fc6773db7fa5328ed8ad9af0261147caf1ed9d9ef3be4d42408c57',
               'is_hip_mi200': '17e776c836a079562178c79b40851caaa05cc828e674de2d9ec4c5f7108f870d',
               'is_hopper': '5f70b52fabccbf5ad3cb44ef21bf7f51245066f75e5beec5ce9d1b54ab06f48b',
               'matmul_kernel': 'b088e8aa74c8c8b677b4744aa4f8d0aad8499d2f503b7dbd141c81a0ec79a2cf',
               'matmul_mxfp_triton_wrapper': 'a585a1d46410eec496fb6e70145e7f14ad7960bbc379b5b42073d71d606e314e',
               'mxfp_to_bf16_kernel': '6ca98e97444ec3e4826e01b8433062158fba798c1593ca01db94fe3df8343a29',
               'mxfp_to_bf16_torch': '07f98bd00120d77e5de67a41d88674ae57a1987e67b9b01227fb147adce02228',
               'set_seed': 'cfdef0cf93153372206c26f3a11dfe9ecd53892109b325cb7dc55337e70be1ce',
               'test_performance': 'eab1d33d0cd6dc67e8e1a2f3b0aa9215d4270dcff2f7d8323062e0b7719da93b',
               'test_save_performance_results': 'e75049a8bfa70933554c63918336d2909e576a9a7940225ed7fa3c50aad1b8d9',
               'test_save_results': '557a528e777fd099f5b131e01721dfebf7caa45416f1f0864010681d99fa97b8'},
 'rows': '56c92100535e01b7adf1fee632b591f734210e730f02a970be6705583dee1270'}
