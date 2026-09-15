"""CPU oracle, replay, full-domain and original-contract regressions for cast GEMM."""
import ast
import hashlib
import importlib.util
import inspect
import itertools
import json
from pathlib import Path
import sys
import types

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
TASKS = ['instruction2triton/rocmbench/test_cast_matmul',
         'triton2triton/rocmbench/medium/test_cast_matmul']


def load(path):
    spec = importlib.util.spec_from_file_location('_cast_' + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module', autouse=True)
def cpu_budget():
    threads = torch.get_num_threads(); state = torch.random.get_rng_state()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads); torch.random.set_rng_state(state)


@pytest.fixture(params=TASKS)
def task(request, monkeypatch):
    path = ROOT / 'tasks' / request.param
    ref = load(path / '_arena_reference.py')
    monkeypatch.setitem(sys.modules, '_arena_reference', ref)
    return path, ref


def inputs(ad=torch.float32, bd=torch.float16, od=torch.float16, strided=False):
    a = torch.tensor([[2, -3, 4], [1, 5, -2]], dtype=ad)
    b = torch.tensor([[1, 3], [2, -1], [-2, 4]], dtype=bd)
    if strided:
        aa = torch.full((2, 6), 97, dtype=ad); aa[:, ::2] = a; a = aa[:, ::2]
        bb = torch.full((3, 4), 97, dtype=bd); bb[:, ::2] = b; b = bb[:, ::2]
    return a, b, torch.full((2, 2), -7, dtype=od)


@pytest.mark.parametrize('ad,bd,od', list(itertools.product(
    [torch.float16, torch.float32, torch.float64],
    [torch.float16, torch.float32, torch.float64], [torch.float16, torch.float32])))
@pytest.mark.parametrize('strided', [False, True])
def test_known_answer_entire_declared_dtype_domain(task, ad, bd, od, strided):
    _, ref = task; a, b, out = inputs(ad, bd, od, strided)
    check = ref.CastMatmulCheck(a, b, out)
    # Hand calculated two-by-two product, independent of torch.matmul.
    known = torch.tensor([[-12, 25], [15, -10]], dtype=od)
    assert torch.equal(check.expected, known)
    out.copy_(known); check(out)
    for bad in [torch.zeros_like(out), known.roll(1, 0), known + 10]:
        out.copy_(bad)
        with pytest.raises(ref.NumericalMismatch): check(out)
    out.copy_(known); out[-1, -1] += 2
    with pytest.raises(ref.NumericalMismatch): check(out)


def test_cast_before_product_not_after_and_original_tolerance(task):
    _, ref = task
    a = torch.tensor([[2048.75, -2048]], dtype=torch.float64)
    b = torch.ones(2, 1, dtype=torch.float32); out = torch.zeros(1, 1, dtype=torch.float16)
    check = ref.CastMatmulCheck(a, b, out)
    assert check.expected.item() == 0
    check(out)
    out.fill_(.25); check(out)
    out.copy_((a @ b.double()).half())
    assert out.item() == .75
    with pytest.raises(ref.NumericalMismatch): check(out)


@pytest.mark.parametrize('failure', ['cached', 'no_write', 'wrong_tail', 'mutation', 'exception'])
def test_changed_input_replay_and_finally_restore(task, failure):
    _, ref = task; a, b, out = inputs(strided=True)
    originals = [x.clone() for x in (a, b, out)]
    check = ref.CastMatmulCheck(a, b, out); out.copy_(check.expected); old = out.clone()
    check(out)
    try:
        check.fresh(out)
        assert torch.isnan(out).all() and not torch.equal(check.expected, old)
        if failure == 'cached': out.copy_(old)
        elif failure == 'wrong_tail': out.copy_(check.expected); out[-1, -1] += 10
        elif failure == 'mutation': out.copy_(check.expected); a[0, 0] += 1
        elif failure == 'exception': raise RuntimeError('injected replay exception')
        with pytest.raises((ref.NumericalMismatch, ValueError)): check(out)
    except RuntimeError as exc:
        assert failure == 'exception' and str(exc) == 'injected replay exception'
    finally: check.restore()
    assert all(torch.equal(x, y) for x, y in zip((a, b, out), originals))


@pytest.mark.parametrize('bad', ['nan', 'wrong_return_buffer', 'none', 'input_bytes'])
def test_output_and_pristine_input_contract(task, bad):
    _, ref = task; a, b, out = inputs(); a[0, 0] = 0
    check = ref.CastMatmulCheck(a, b, out); out.copy_(check.expected); returned = out
    if bad == 'nan': out[-1, -1] = float('nan')
    if bad == 'wrong_return_buffer': returned = out.clone()
    if bad == 'none': returned = None
    if bad == 'input_bytes': a[0, 0] = -0.
    with pytest.raises((ValueError, TypeError)): check(returned)
    check.restore()


@pytest.mark.parametrize('mode', ['graph', 'events', 'cached', 'no_replay_write', 'wrong_timed',
                                  'wrong_replay', 'mutate_timed', 'mutate_replay', 'crash', 'bad_timing', 'unbound'])
def test_actual_benchmark_adapter_replay_and_restore(task, monkeypatch, mode):
    path, ref = task; adapter = load(path / '_arena_eval.py')
    a, b, out = inputs(); original = [x.clone() for x in (a, b, out)]; phase = ['initial']
    cached = a.to(out.dtype) @ b.to(out.dtype)
    plugin = types.SimpleNamespace(action='performance', current_row={'test_case_id': 'cpu'}, exercised=set())
    class Base:
        def __init__(self, op):
            self.op_callable = op; self.prepare_fn = None; self.use_cuda_graph = mode != 'events'
            self.fallback_reason = 'explicit observable events' if mode == 'events' else None
            self.config = types.SimpleNamespace(warm_up=10, repetition=100)
    class Timed:
        outputs = None
        def rerun(self):
            phase[0] = 'replay'
            if mode == 'crash': raise RuntimeError('injected replay crash')
            return op()
    timer_args = []
    def timer(fn, **kw):
        timer_args.append(kw); phase[0] = 'timed'
        if mode != 'unbound': kw['timed_run'].outputs = fn()
        return ([float('nan')] if mode == 'bad_timing' else [1., 2.]), {
            'benchmark_method': 'cuda_event_fallback' if mode == 'events' else 'cuda_graph',
            'benchmark_fallback_reason': kw['fallback_reason']}
    monkeypatch.setitem(sys.modules, '_aka_benchmark', types.SimpleNamespace(TimedRun=Timed, benchmark_cuda_graph_or_events_samples=timer))
    monkeypatch.setitem(sys.modules, 'performance_utils_pytest', types.SimpleNamespace(_compute_timing_stats=lambda times, cfg: {'mean': sum(times)/len(times)}))
    def op():
        if not (mode == 'no_replay_write' and phase[0] == 'replay'):
            out.copy_(cached if mode == 'cached' else a.to(out.dtype) @ b.to(out.dtype))
        if mode == 'wrong_' + phase[0]: out[-1, -1] += 10
        if mode == 'mutate_' + phase[0]: a[0, 0] += 1
        return out
    bench = adapter.benchmark_type(Base, plugin, None)(op)
    bench.context = {'a': a, 'b': b, 'out_triton': out}
    if mode in ['graph', 'events']:
        bench.run_benchmark(baseline_callable=lambda: pytest.fail('peer timing'))
        assert plugin.exercised == {'cpu'} and plugin.current_row['execution_time_ms'] == 1.5
        evidence = plugin.current_row['metadata']
        for key in ['timed_output_checked', 'fresh_input_replay_checked', 'readonly_input_checked', 'input_state_restored', 'poisoned_output_restored']:
            assert evidence[key]
        helper = load(ROOT/'src/tools/perf/performance_utils_pytest.py'); prior = []
        monkeypatch.setattr(helper, 'benchmark_cuda_graph_or_events_samples', lambda fn, **kw: (prior.append(kw) or [1.], {}))
        helper._measure_times(op, bench.config, prepare_fn=None, use_cuda_graph=bench.use_cuda_graph, fallback_reason=bench.fallback_reason)
        sig = inspect.signature(load(ROOT/'src/tools/perf/aka_benchmark.py').benchmark_cuda_graph_or_events_samples)
        def effective(kwargs):
            bound = sig.bind_partial(None, **kwargs); bound.apply_defaults()
            return {k: v for k, v in bound.arguments.items() if k not in ['fn', 'timed_run']}
        assert effective(timer_args[0]) == effective(prior[0])
    else:
        with pytest.raises((ref.NumericalMismatch, ValueError, TypeError, RuntimeError)): bench.run_benchmark()
        assert not plugin.exercised and 'execution_time_ms' not in plugin.current_row
    assert all(torch.equal(x, y) for x, y in zip((a, b, out), original))


def test_entire_manifest_is_collected_without_skips(task):
    path, _ = task; source = (path/'test_cast_matmul.py').read_text(); tree = ast.parse(source)
    namespace = {'pytest': pytest}
    bodies = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ['input_dtypes', 'out_dtypes', 'CAST_MATMUL_SHAPES_FOR_PERF', 'CAST_MATMUL_DTYPES_FOR_PERF']:
            bodies.append(node)
        if isinstance(node, ast.FunctionDef) and node.name in ['test_cast_matmul', 'test_performance', 'test_cast_matmul_strided']:
            node.body = [ast.Pass()]; bodies.append(node)
    exec(compile(ast.fix_missing_locations(ast.Module(body=bodies, type_ignores=[])), '<collection>', 'exec'), namespace)
    adapter = load(path/'_arena_eval.py'); found = {}
    for name in ['test_cast_matmul', 'test_performance', 'test_cast_matmul_strided']:
        parameter_sets = [{}]
        for mark in namespace[name].pytestmark:
            keys = mark.args[0].replace(' ', '').split(',')
            parameter_sets = [dict(previous, **dict(zip(keys, getattr(values, 'values', values)))) for previous in parameter_sets for values in mark.args[1]]
        for params in parameter_sets: found[adapter.identity(name, params)] = {'function': name, 'arguments': params}
    data = json.loads((path/'workloads.json').read_text()); expected = {r['test_case_id']: r['params'] for r in data['cases']}
    assert found == expected and len(found) == (59 if 'triton2triton' in path.parts else 57)
    assert sum('performance' in r['checks'] for r in data['cases']) == 18
    assert 'pytest.skip(' not in source
    assert 'torch.testing.assert_close(out_torch, out_triton, atol=0.3, rtol=0.01)' in source


def test_validate_task_emits_complete_manifest_envelope(task, monkeypatch):
    path, _ = task
    adapter = load(path/'_arena_eval.py')
    # Collection identities are checked above. Here simulate successful pytest
    # collection to exercise the actual envelope path, including appended rows.
    monkeypatch.setattr(pytest, 'main', lambda *args, **kwargs: 0)
    result = adapter.evaluate('task', 'validate-task')
    assert result['status'] == 'PASS', result.get('reason')
    assert result['metadata']['candidate_state'] == 'implemented'
    assert len(result['cases']) == (59 if 'triton2triton' in path.parts else 57)
    assert all(row['status'] == 'PASS' for row in result['cases'])
    from src.task_protocol import parse_command_result
    # Verify the real framework parser accepts task-owned collection evidence.
    parsed = parse_command_result('ARENA_EVAL_RESULT=' + json.dumps(result, allow_nan=False),
                                  role='task', action='validate-task', returncode=0)
    assert parsed.status == 'PASS'


def test_original_kernel_config_manifest_and_launch_contract(task):
    path, _ = task; expected = ORIGINAL[path.relative_to(ROOT).as_posix()]
    source = (path/'test_cast_matmul.py').read_text(); nodes = {n.name: n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
    for name in ['config.yaml', 'performance_utils_pytest.py']:
        assert hashlib.sha256((path/name).read_bytes()).hexdigest() == expected[name]
    for name in ['matmul_kernel', 'cast_matmul_triton_wrapper', 'set_seed']:
        assert hashlib.sha256(ast.get_source_segment(source, nodes[name]).encode()).hexdigest() == expected[name]
    if 'triton2triton' in path.parts:
        param=next(d for d in nodes['test_cast_matmul'].decorator_list if isinstance(d,ast.Call))
        extra=param.args[1]
        assert isinstance(extra,ast.BinOp) and isinstance(extra.op,ast.Add)
        assert [tuple(ast.literal_eval(a) for a in n.args) for n in extra.right.elts] == [
            (31,48,48,'float16','float32','float16'), (65,48,80,'float64','float16','float32')]
        param.args[1]=extra.left
    for name, decorators in expected['decorators'].items():
        assert ast.dump(ast.Module(body=nodes[name].decorator_list, type_ignores=[])) == decorators
    rows = json.loads((path/'workloads.json').read_text())['cases']
    assert hashlib.sha256(json.dumps(rows[:54], sort_keys=True, separators=(',', ':')).encode()).hexdigest() == expected['original_rows_sha256']
    assert all(r['checks'] == ['correctness'] for r in rows[54:])
    assert 'do_bench_config(warm_up=10, repetition=100)' in source
    assert hashlib.sha256(ast.dump(normalized_source(source)).encode()).hexdigest() == expected['normalized_original_source_sha256']


def normalized_source(source):
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if not isinstance(n, ast.FunctionDef) or n.name not in ['test_cast_matmul', 'test_cast_matmul_strided']]
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name == 'test_performance':
            n.body = [x for x in n.body if not isinstance(x, ast.If) or not any(isinstance(c, ast.Call) and ast.unparse(c.func) == 'pytest.skip' for c in ast.walk(x))]
    return tree


@pytest.mark.parametrize('mode', ['good', 'mutated_input', 'padding', 'tail'])
def test_public_strided_control_rejects_invalid_implementations(task, mode):
    path, _ = task
    node = next(n for n in ast.parse((path/'test_cast_matmul.py').read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'test_cast_matmul_strided')
    node.decorator_list = []
    observed = []
    class CPUTorch:
        def __getattr__(self, name):
            if name not in ['randn', 'full']: return getattr(torch, name)
            def constructor(*args, **kwargs):
                kwargs['device'] = 'cpu'
                return getattr(torch, name)(*args, **kwargs)
            return constructor
    def wrapper(a, b, out, *args):
        out.copy_(a.to(out.dtype) @ b.to(out.dtype))
        observed.append((a, b, out, a.clone(), b.clone(), out._base.clone()))
        if mode == 'mutated_input': a[0, 0] += 1
        if mode == 'padding': out._base[0, 0] = 0
        if mode == 'tail': out[-1, -1] += 10
        return out
    namespace = {'torch': CPUTorch(), 'tl': types.SimpleNamespace(float16='float16'),
                 'set_seed': lambda: torch.manual_seed(42), 'cast_matmul_triton_wrapper': wrapper}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<original-control>', 'exec'), namespace)
    request = types.SimpleNamespace(node=types.SimpleNamespace(user_properties=[]))
    call = lambda: namespace['test_cast_matmul_strided'](3, 5, 7, 'float64', 'float16', 'float16', request)
    if mode == 'good':
        call()
        assert request.node.user_properties[0][1]['padding_checked']
    else:
        with pytest.raises((ValueError, AssertionError)): call()
        assert not request.node.user_properties
    a, b, out, pristine_a, pristine_b, _ = observed[0]
    assert torch.equal(a, pristine_a) and torch.equal(b, pristine_b)
    assert bool((out._base == -97).all())


# Checked-in original expectations; no Git history is required in CI.
ORIGINAL = {'tasks/instruction2triton/rocmbench/test_cast_matmul': {'cast_matmul_triton_wrapper': '619c07778bebba091098b8f70ea8028264c1a3730a742cb4a323cef152a66bfc',
                                                         'config.yaml': 'c4ce380bb4451376b0fcce71a5641a31edb0335313b4748052faa0cee5fe2cec',
                                                         'decorators': {'matmul_kernel': "Module(body=[Attribute(value=Name(id='triton', "
                                                                                         'ctx=Load()), '
                                                                                         "attr='jit', "
                                                                                         'ctx=Load())], '
                                                                                         'type_ignores=[])',
                                                                        'test_cast_matmul': "Module(body=[Call(func=Attribute(value=Attribute(value=Name(id='pytest', "
                                                                                            'ctx=Load()), '
                                                                                            "attr='mark', "
                                                                                            'ctx=Load()), '
                                                                                            "attr='parametrize', "
                                                                                            'ctx=Load()), '
                                                                                            "args=[Constant(value='M, "
                                                                                            'K, N, w_dtype, '
                                                                                            'x_dtype, '
                                                                                            "out_dtype'), "
                                                                                            "ListComp(elt=Tuple(elts=[Name(id='M', "
                                                                                            'ctx=Load()), '
                                                                                            "Name(id='K', "
                                                                                            'ctx=Load()), '
                                                                                            "Name(id='N', "
                                                                                            'ctx=Load()), '
                                                                                            "Name(id='w', "
                                                                                            'ctx=Load()), '
                                                                                            "Name(id='x', "
                                                                                            'ctx=Load()), '
                                                                                            "Name(id='o', "
                                                                                            'ctx=Load())], '
                                                                                            'ctx=Load()), '
                                                                                            "generators=[comprehension(target=Tuple(elts=[Name(id='M', "
                                                                                            'ctx=Store()), '
                                                                                            "Name(id='K', "
                                                                                            'ctx=Store()), '
                                                                                            "Name(id='N', "
                                                                                            'ctx=Store())], '
                                                                                            'ctx=Store()), '
                                                                                            'iter=List(elts=[Tuple(elts=[Constant(value=128), '
                                                                                            'Constant(value=128), '
                                                                                            'Constant(value=128)], '
                                                                                            'ctx=Load()), '
                                                                                            'Tuple(elts=[Constant(value=1280), '
                                                                                            'Constant(value=768), '
                                                                                            'Constant(value=1024)], '
                                                                                            'ctx=Load())], '
                                                                                            'ctx=Load()), '
                                                                                            'ifs=[], '
                                                                                            'is_async=0), '
                                                                                            "comprehension(target=Name(id='w', "
                                                                                            'ctx=Store()), '
                                                                                            "iter=Name(id='input_dtypes', "
                                                                                            'ctx=Load()), '
                                                                                            'ifs=[], '
                                                                                            'is_async=0), '
                                                                                            "comprehension(target=Name(id='x', "
                                                                                            'ctx=Store()), '
                                                                                            "iter=Name(id='input_dtypes', "
                                                                                            'ctx=Load()), '
                                                                                            'ifs=[], '
                                                                                            'is_async=0), '
                                                                                            "comprehension(target=Name(id='o', "
                                                                                            'ctx=Store()), '
                                                                                            "iter=Name(id='out_dtypes', "
                                                                                            'ctx=Load()), '
                                                                                            'ifs=[], '
                                                                                            'is_async=0)])], '
                                                                                            'keywords=[])], '
                                                                                            'type_ignores=[])',
                                                                        'test_performance': "Module(body=[Call(func=Attribute(value=Attribute(value=Name(id='pytest', "
                                                                                            'ctx=Load()), '
                                                                                            "attr='mark', "
                                                                                            'ctx=Load()), '
                                                                                            "attr='parametrize', "
                                                                                            'ctx=Load()), '
                                                                                            "args=[Constant(value='M, "
                                                                                            "K, N'), "
                                                                                            "Name(id='CAST_MATMUL_SHAPES_FOR_PERF', "
                                                                                            'ctx=Load())], '
                                                                                            'keywords=[]), '
                                                                                            "Call(func=Attribute(value=Attribute(value=Name(id='pytest', "
                                                                                            'ctx=Load()), '
                                                                                            "attr='mark', "
                                                                                            'ctx=Load()), '
                                                                                            "attr='parametrize', "
                                                                                            'ctx=Load()), '
                                                                                            "args=[Constant(value='a_dtype_str, "
                                                                                            'b_dtype_str, '
                                                                                            'c_dtype_str, '
                                                                                            "dot_acc_tl_dtype_str'), "
                                                                                            "Name(id='CAST_MATMUL_DTYPES_FOR_PERF', "
                                                                                            'ctx=Load())], '
                                                                                            'keywords=[])], '
                                                                                            'type_ignores=[])'},
                                                         'matmul_kernel': '9dcd0994c067cddf385b00e6f2b6bdc7561ec11d47f1526b35b517b47c0208f7',
                                                         'normalized_original_source_sha256': '5f22cb82b08b948fbfaf7d4ac43771b150f53e5e69d9360b05823c70bf1e9cc7',
                                                         'original_rows_sha256': '61f975a587a8fb64f5e36b223e07065ab8a77ff654f6d4cb15b93447546e3d0c',
                                                         'original_source_sha256': '279136ba9297f38f3dd9b47f532a08775c32255053bf22bd3128cbe67497b852',
                                                         'performance_utils_pytest.py': 'e0dfef878da9ac042c39f1832e7940520fa1738ab29d9af2fcdbbc7e42d146c9',
                                                         'set_seed': 'cfdef0cf93153372206c26f3a11dfe9ecd53892109b325cb7dc55337e70be1ce',
                                                         'workloads.json': '669cbdd597515c92db9012624e6e8ee4d1ede59018187a2a255a6c0d9b640e88'},
 'tasks/triton2triton/rocmbench/medium/test_cast_matmul': {'cast_matmul_triton_wrapper': '619c07778bebba091098b8f70ea8028264c1a3730a742cb4a323cef152a66bfc',
                                                           'config.yaml': 'c4ce380bb4451376b0fcce71a5641a31edb0335313b4748052faa0cee5fe2cec',
                                                           'decorators': {'matmul_kernel': "Module(body=[Attribute(value=Name(id='triton', "
                                                                                           'ctx=Load()), '
                                                                                           "attr='jit', "
                                                                                           'ctx=Load())], '
                                                                                           'type_ignores=[])',
                                                                          'test_cast_matmul': "Module(body=[Call(func=Attribute(value=Attribute(value=Name(id='pytest', "
                                                                                              'ctx=Load()), '
                                                                                              "attr='mark', "
                                                                                              'ctx=Load()), '
                                                                                              "attr='parametrize', "
                                                                                              'ctx=Load()), '
                                                                                              "args=[Constant(value='M, "
                                                                                              'K, N, '
                                                                                              'w_dtype, '
                                                                                              'x_dtype, '
                                                                                              "out_dtype'), "
                                                                                              "ListComp(elt=Tuple(elts=[Name(id='M', "
                                                                                              'ctx=Load()), '
                                                                                              "Name(id='K', "
                                                                                              'ctx=Load()), '
                                                                                              "Name(id='N', "
                                                                                              'ctx=Load()), '
                                                                                              "Name(id='w', "
                                                                                              'ctx=Load()), '
                                                                                              "Name(id='x', "
                                                                                              'ctx=Load()), '
                                                                                              "Name(id='o', "
                                                                                              'ctx=Load())], '
                                                                                              'ctx=Load()), '
                                                                                              "generators=[comprehension(target=Tuple(elts=[Name(id='M', "
                                                                                              'ctx=Store()), '
                                                                                              "Name(id='K', "
                                                                                              'ctx=Store()), '
                                                                                              "Name(id='N', "
                                                                                              'ctx=Store())], '
                                                                                              'ctx=Store()), '
                                                                                              'iter=List(elts=[Tuple(elts=[Constant(value=128), '
                                                                                              'Constant(value=128), '
                                                                                              'Constant(value=128)], '
                                                                                              'ctx=Load()), '
                                                                                              'Tuple(elts=[Constant(value=1280), '
                                                                                              'Constant(value=768), '
                                                                                              'Constant(value=1024)], '
                                                                                              'ctx=Load())], '
                                                                                              'ctx=Load()), '
                                                                                              'ifs=[], '
                                                                                              'is_async=0), '
                                                                                              "comprehension(target=Name(id='w', "
                                                                                              'ctx=Store()), '
                                                                                              "iter=Name(id='input_dtypes', "
                                                                                              'ctx=Load()), '
                                                                                              'ifs=[], '
                                                                                              'is_async=0), '
                                                                                              "comprehension(target=Name(id='x', "
                                                                                              'ctx=Store()), '
                                                                                              "iter=Name(id='input_dtypes', "
                                                                                              'ctx=Load()), '
                                                                                              'ifs=[], '
                                                                                              'is_async=0), '
                                                                                              "comprehension(target=Name(id='o', "
                                                                                              'ctx=Store()), '
                                                                                              "iter=Name(id='out_dtypes', "
                                                                                              'ctx=Load()), '
                                                                                              'ifs=[], '
                                                                                              'is_async=0)])], '
                                                                                              'keywords=[])], '
                                                                                              'type_ignores=[])',
                                                                          'test_performance': "Module(body=[Call(func=Attribute(value=Attribute(value=Name(id='pytest', "
                                                                                              'ctx=Load()), '
                                                                                              "attr='mark', "
                                                                                              'ctx=Load()), '
                                                                                              "attr='parametrize', "
                                                                                              'ctx=Load()), '
                                                                                              "args=[Constant(value='M, "
                                                                                              "K, N'), "
                                                                                              "Name(id='CAST_MATMUL_SHAPES_FOR_PERF', "
                                                                                              'ctx=Load())], '
                                                                                              'keywords=[]), '
                                                                                              "Call(func=Attribute(value=Attribute(value=Name(id='pytest', "
                                                                                              'ctx=Load()), '
                                                                                              "attr='mark', "
                                                                                              'ctx=Load()), '
                                                                                              "attr='parametrize', "
                                                                                              'ctx=Load()), '
                                                                                              "args=[Constant(value='a_dtype_str, "
                                                                                              'b_dtype_str, '
                                                                                              'c_dtype_str, '
                                                                                              "dot_acc_tl_dtype_str'), "
                                                                                              "Name(id='CAST_MATMUL_DTYPES_FOR_PERF', "
                                                                                              'ctx=Load())], '
                                                                                              'keywords=[])], '
                                                                                              'type_ignores=[])'},
                                                           'matmul_kernel': '9dcd0994c067cddf385b00e6f2b6bdc7561ec11d47f1526b35b517b47c0208f7',
                                                           'normalized_original_source_sha256': '05a9c40f4ae0941b9846df3dbb20e26456bc1abbcc81091d9baf2226911b907a',
                                                           'original_rows_sha256': '61f975a587a8fb64f5e36b223e07065ab8a77ff654f6d4cb15b93447546e3d0c',
                                                           'original_source_sha256': '5668a10d6b08bb2320e53bd12d81188f4552c7ceeb19d80d7cc9365ef85ef3f9',
                                                           'performance_utils_pytest.py': 'e0dfef878da9ac042c39f1832e7940520fa1738ab29d9af2fcdbbc7e42d146c9',
                                                           'set_seed': 'cfdef0cf93153372206c26f3a11dfe9ecd53892109b325cb7dc55337e70be1ce',
                                                           'workloads.json': '4b35e27ca83fe34937f0833fda83d5d48fac449dce15149cbdb29ad785691842'}}
