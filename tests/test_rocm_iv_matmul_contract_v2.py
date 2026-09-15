"""IV-dependent GEMM: original cases, real replay and partial-tile controls."""
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
TASKS = ['instruction2triton/rocmbench/test_iv_dependent_matmul',
         'triton2triton/rocmbench/medium/test_iv_dependent_matmul']


def load(path):
    spec = importlib.util.spec_from_file_location('_iv_' + path.stem, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module', autouse=True)
def cpu_budget():
    threads = torch.get_num_threads(); state = torch.random.get_rng_state()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads); torch.random.set_rng_state(state)


@pytest.fixture(params=TASKS)
def task(request, monkeypatch):
    path = ROOT/'tasks'/request.param; ref = load(path/'_arena_reference.py')
    monkeypatch.setitem(sys.modules, '_arena_reference', ref)
    return path, ref


def inputs(dtype=torch.float32, transposed=False):
    a = torch.tensor([[2, -3, 4], [1, 5, -2]], dtype=torch.float16)
    b = torch.tensor([[1, 3], [2, -1], [-2, 4]], dtype=torch.float16)
    if transposed: a = a.T.contiguous().T; b = b.T.contiguous().T
    return a, b, torch.full((2, 2), -7, dtype=dtype)


@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
@pytest.mark.parametrize('transposed', [False, True])
def test_independent_known_answer_and_full_tail(task, dtype, transposed):
    _, ref = task; a, b, out = inputs(dtype, transposed)
    check = ref.IVMatmulCheck(a, b, out)
    known = torch.tensor([[-12, 25], [15, -10]], dtype=torch.float32)
    assert torch.equal(check.expected, known.to(dtype)) and check.expected.dtype == dtype
    out.copy_(known); check(out)
    out[-1, -1] += 2
    with pytest.raises(ref.NumericalMismatch): check(out)
    out.copy_(known); a[0, 0] += 1
    with pytest.raises(ValueError, match='Read-only'): check(out)
    check.restore()


@pytest.mark.parametrize('mode', ['graph', 'events', 'cached', 'no_write', 'wrong_timed', 'wrong_replay', 'mutation', 'crash', 'bad_timing'])
def test_real_adapter_changed_input_replay_and_restore(task, monkeypatch, mode):
    path, ref = task; adapter = load(path/'_arena_eval.py'); a, b, out = inputs(transposed=True)
    original = [x.clone() for x in (a, b, out)]; phase = ['initial']; cached = a.float() @ b.float(); timer_args = []
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
        return ([float('nan')] if mode == 'bad_timing' else [1., 2.]), {'benchmark_method': 'cuda_event_fallback' if mode == 'events' else 'cuda_graph', 'benchmark_fallback_reason': kw['fallback_reason']}
    monkeypatch.setitem(sys.modules, '_aka_benchmark', types.SimpleNamespace(TimedRun=Timed, benchmark_cuda_graph_or_events_samples=timer))
    monkeypatch.setitem(sys.modules, 'performance_utils_pytest', types.SimpleNamespace(_compute_timing_stats=lambda ts, cfg: {'mean': sum(ts)/len(ts)}))
    def op():
        if not (mode == 'no_write' and phase[0] == 'replay'): out.copy_(cached if mode == 'cached' else a.float() @ b.float())
        if mode == 'wrong_' + phase[0]: out[-1, -1] += 10
        if mode == 'mutation' and phase[0] == 'replay': a[0, 0] += 1
        return out
    bench = adapter.benchmark_type(Base, plugin, None)(op); bench.context = {'a': a, 'b': b, 'triton_output_buffer': out}
    if mode in ['graph', 'events']:
        bench.run_benchmark(baseline_callable=lambda: pytest.fail('peer baseline'))
        evidence = plugin.current_row['metadata']
        assert all(evidence[k] for k in ['fresh_input_replay_checked', 'timed_output_checked', 'readonly_input_checked', 'input_state_restored', 'poisoned_output_restored'])
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


def test_all_original_and_added_cases_collect_independently(task, monkeypatch):
    path, _ = task; source = (path/'test_iv_dependent_matmul.py').read_text()
    tree = ast.parse(source); namespace = {}; found = {}; adapter = load(path/'_arena_eval.py')
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try: namespace[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError): pass
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef): continue
        parameter_sets = [{}]; marked = False
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call) or ast.unparse(decorator.func) != 'pytest.mark.parametrize': continue
            marked = True; keys = ast.literal_eval(decorator.args[0]).replace(' ', '').split(',')
            values = eval(compile(ast.Expression(body=decorator.args[1]), '<original-params>', 'eval'), namespace)
            if len(keys) == 1: values = [(v,) for v in values]
            parameter_sets = [dict(previous, **dict(zip(keys, row))) for previous in parameter_sets for row in values]
        if marked:
            for params in parameter_sets:
                found[adapter.identity(node.name, params)] = {'function': node.name, 'arguments': adapter.serial(params)}
    data = json.loads((path/'workloads.json').read_text()); expected = {r['test_case_id']: r['params'] for r in data['cases']}
    assert found == expected and len(found) == 2115
    assert sum('performance' in r['checks'] for r in data['cases']) == 2100
    assert 'pytest.skip(' not in source
    monkeypatch.setattr(pytest, 'main', lambda *args, **kwargs: 0)
    result = adapter.evaluate('task', 'validate-task')
    from src.task_protocol import parse_command_result
    assert parse_command_result('ARENA_EVAL_RESULT=' + json.dumps(result), role='task', action='validate-task', returncode=0).status == 'PASS'
    assert len(result['cases']) == 2115


@pytest.mark.parametrize('variant', ['pre_load', 'post_load', 'post_pre_mixed', 'post_load_two_iters', 'post_load_three_iters'])
@pytest.mark.parametrize('dtype', ['fp16', 'fp32'])
@pytest.mark.parametrize('mode', ['good', 'tail', 'input_mutation'])
def test_actual_partial_tile_controls_are_meaningful(task, variant, dtype, mode):
    path, ref = task
    node = next(n for n in ast.parse((path/'test_iv_dependent_matmul.py').read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'test_partial_tile_control')
    node.decorator_list = []; observed = []
    class CPUTorch:
        def __getattr__(self, name):
            if name not in ['randn', 'empty']: return getattr(torch, name)
            def constructor(*args, **kwargs):
                kwargs['device'] = 'cpu'; return getattr(torch, name)(*args, **kwargs)
            return constructor
    def wrapper(a, b, output, M, N, K, BM, BN, BK, kernel_type, stages, warps):
        assert (M, N, K, BM, BN, BK, kernel_type) == (17, 19, 13, 16, 16, 32, variant)
        observed.append((a, b, output, a.clone(), b.clone(), output.clone()))
        output.copy_((a @ b).half())
        if mode == 'tail': output[-1, -1] += 10
        if mode == 'input_mutation': a[0, 0] += 1
        return output
    namespace = {'torch': CPUTorch(), 'set_seed': lambda: torch.manual_seed(42), 'iv_dependent_matmul_triton_wrapper': wrapper}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<partial-control>', 'exec'), namespace)
    request = types.SimpleNamespace(node=types.SimpleNamespace(user_properties=[]))
    call = lambda: namespace['test_partial_tile_control'](17, 19, 13, variant, dtype, 16, 16, 32, 4 if variant == 'post_load_three_iters' else 3, 4, request)
    if mode == 'good':
        call(); assert request.node.user_properties[0][1]['unscored_partial_tile_control']
    else:
        with pytest.raises((ValueError, ref.NumericalMismatch)): call()
        assert not request.node.user_properties
    assert all(ref.equal_bytes(a, b) for a, b in zip(observed[0][:3], observed[0][3:]))


def normalized_performance(node):
    node.body = [n for n in node.body if not (
        isinstance(n, ast.If) and any(isinstance(c, ast.Call) and ast.unparse(c.func) == 'pytest.skip' for c in ast.walk(n))
        or isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'smem_elements_needed')]
    return ast.dump(node)


def test_original_kernel_launch_gate_and_all_scored_parameters_preserved(task):
    path, _ = task; original = ORIGINAL[path.relative_to(ROOT).as_posix()]
    source = (path/'test_iv_dependent_matmul.py').read_text(); nodes = {n.name:n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
    for name, digest in original['functions'].items():
        assert hashlib.sha256(ast.get_source_segment(source, nodes[name]).encode()).hexdigest() == digest
    for name, digest in original['files'].items(): assert hashlib.sha256((path/name).read_bytes()).hexdigest() == digest
    for name, digest in original['decorators'].items():
        assert hashlib.sha256(ast.dump(ast.Module(body=nodes[name].decorator_list, type_ignores=[])).encode()).hexdigest() == digest
    rows = json.loads((path/'workloads.json').read_text())['cases']
    assert hashlib.sha256(json.dumps(rows[:2105], sort_keys=True, separators=(',', ':')).encode()).hexdigest() == original['rows']
    assert all(r['checks'] == ['correctness'] for r in rows[2105:])
    assert hashlib.sha256(normalized_performance(nodes['test_performance']).encode()).hexdigest() == original['performance_except_heuristics']
    assert 'torch.testing.assert_close(torch_output, triton_output, rtol=1e-2, atol=1e-2)' in source


# Stable original expectations; no branch history dependency in CI.
ORIGINAL = {'tasks/instruction2triton/rocmbench/test_iv_dependent_matmul': {'decorators': {'calculate_gemm_gbps': '3543b4693a36a1098850b8bc928887694ed59a6deb7d3dfd0339de01f55a77b6',
                                                                                'calculate_gemm_tflops': '3543b4693a36a1098850b8bc928887694ed59a6deb7d3dfd0339de01f55a77b6',
                                                                                'iv_dependent_matmul': '6978ec5d24aebd8ab58b32e8b910203a142e1c6a3159d082fd6c7179b0c8dc04',
                                                                                'iv_dependent_matmul_triton_wrapper': '3543b4693a36a1098850b8bc928887694ed59a6deb7d3dfd0339de01f55a77b6',
                                                                                'set_seed': '3543b4693a36a1098850b8bc928887694ed59a6deb7d3dfd0339de01f55a77b6',
                                                                                'test_iv_dependent_matmul': 'b8ad88b1900033db2cd8e76a282653e12fecc0c4241f67764e835dd7de61e85c',
                                                                                'test_performance': '8a29b3de93290988d0838ad10e0acc0d4f7e55418cbda697c194a2fd4e75a880',
                                                                                'test_save_performance_results': '3543b4693a36a1098850b8bc928887694ed59a6deb7d3dfd0339de01f55a77b6',
                                                                                'test_save_results': '3543b4693a36a1098850b8bc928887694ed59a6deb7d3dfd0339de01f55a77b6'},
                                                                 'files': {'config.yaml': 'b1432729bbe70949c9e13c04ab3505285852d92a7a0a80e881481f7601551f74',
                                                                           'performance_utils_pytest.py': 'e0dfef878da9ac042c39f1832e7940520fa1738ab29d9af2fcdbbc7e42d146c9'},
                                                                 'functions': {'calculate_gemm_gbps': 'd164c02a9a91bcec2efd530f626e0d3d0aef191d0cd41efd8367d354a8c0777f',
                                                                               'calculate_gemm_tflops': '95c332258cd36e5e0e4f3487f8d759bcba4b36f1ba5851c7feedf7c22968128f',
                                                                               'iv_dependent_matmul': 'd94791643aec48dfc3199c29723f9f935d2e70f93ab2c134cb02e0e3315744a0',
                                                                               'iv_dependent_matmul_triton_wrapper': '355f4d4e5d40c4ffdabaf8cdf1a23db60c80effcb05805f75aa6111f5874af19',
                                                                               'set_seed': 'cfdef0cf93153372206c26f3a11dfe9ecd53892109b325cb7dc55337e70be1ce',
                                                                               'test_save_performance_results': 'e75049a8bfa70933554c63918336d2909e576a9a7940225ed7fa3c50aad1b8d9',
                                                                               'test_save_results': '557a528e777fd099f5b131e01721dfebf7caa45416f1f0864010681d99fa97b8'},
                                                                 'performance_except_heuristics': '5678d5858fd55a5f3f5ed37d7ed8ea1b29e9be284fb399b8ce808c84449b10d0',
                                                                 'rows': 'e41316baae3dd7312e0453743e525f021f66c80693d9c237f89069ce4902bdd8'},
 'tasks/triton2triton/rocmbench/medium/test_iv_dependent_matmul': {'decorators': {'calculate_gemm_gbps': '3543b4693a36a1098850b8bc928887694ed59a6deb7d3dfd0339de01f55a77b6',
                                                                                  'calculate_gemm_tflops': '3543b4693a36a1098850b8bc928887694ed59a6deb7d3dfd0339de01f55a77b6',
                                                                                  'iv_dependent_matmul': '6978ec5d24aebd8ab58b32e8b910203a142e1c6a3159d082fd6c7179b0c8dc04',
                                                                                  'iv_dependent_matmul_triton_wrapper': '3543b4693a36a1098850b8bc928887694ed59a6deb7d3dfd0339de01f55a77b6',
                                                                                  'set_seed': '3543b4693a36a1098850b8bc928887694ed59a6deb7d3dfd0339de01f55a77b6',
                                                                                  'test_iv_dependent_matmul': 'b8ad88b1900033db2cd8e76a282653e12fecc0c4241f67764e835dd7de61e85c',
                                                                                  'test_performance': '8a29b3de93290988d0838ad10e0acc0d4f7e55418cbda697c194a2fd4e75a880',
                                                                                  'test_save_performance_results': '3543b4693a36a1098850b8bc928887694ed59a6deb7d3dfd0339de01f55a77b6',
                                                                                  'test_save_results': '3543b4693a36a1098850b8bc928887694ed59a6deb7d3dfd0339de01f55a77b6'},
                                                                   'files': {'config.yaml': 'b1432729bbe70949c9e13c04ab3505285852d92a7a0a80e881481f7601551f74',
                                                                             'performance_utils_pytest.py': 'e0dfef878da9ac042c39f1832e7940520fa1738ab29d9af2fcdbbc7e42d146c9'},
                                                                   'functions': {'calculate_gemm_gbps': 'd164c02a9a91bcec2efd530f626e0d3d0aef191d0cd41efd8367d354a8c0777f',
                                                                                 'calculate_gemm_tflops': '95c332258cd36e5e0e4f3487f8d759bcba4b36f1ba5851c7feedf7c22968128f',
                                                                                 'iv_dependent_matmul': 'd94791643aec48dfc3199c29723f9f935d2e70f93ab2c134cb02e0e3315744a0',
                                                                                 'iv_dependent_matmul_triton_wrapper': '355f4d4e5d40c4ffdabaf8cdf1a23db60c80effcb05805f75aa6111f5874af19',
                                                                                 'set_seed': 'cfdef0cf93153372206c26f3a11dfe9ecd53892109b325cb7dc55337e70be1ce',
                                                                                 'test_save_performance_results': 'e75049a8bfa70933554c63918336d2909e576a9a7940225ed7fa3c50aad1b8d9',
                                                                                 'test_save_results': '557a528e777fd099f5b131e01721dfebf7caa45416f1f0864010681d99fa97b8'},
                                                                   'performance_except_heuristics': 'ee91b74526b9cd7b55dd84da0e1f92ee3743b32625716aa5bb762c2bd6304b08',
                                                                   'rows': 'e41316baae3dd7312e0453743e525f021f66c80693d9c237f89069ce4902bdd8'}}
