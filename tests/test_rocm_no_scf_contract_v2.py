"""No-SCF GEMM: independent oracle, real replay and precise unsupported-CTA tests."""
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
TASKS = ['instruction2triton/rocmbench/test_gemm_no_scf',
         'triton2triton/rocmbench/medium/test_gemm_no_scf']


def load(path):
    spec = importlib.util.spec_from_file_location('_no_scf_' + path.stem, path)
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
    check = ref.GemmCheck(a, b, out)
    known = torch.tensor([[-12, 25], [15, -10]], dtype=torch.float32)
    assert torch.equal(check.expected, known) and check.expected.dtype == torch.float32
    out.copy_(known); check(out)
    out[-1, -1] += 2
    with pytest.raises(ref.NumericalMismatch): check(out)
    out.copy_(known); a[0, 0] += 1
    with pytest.raises(ValueError, match='Read-only'): check(out)
    check.restore()


@pytest.mark.parametrize('mode', ['exact_error', 'wrong_error', 'wrong_type', 'success', 'output_mutation', 'input_mutation'])
def test_rejected_launch_must_be_specific_and_unchanged(task, mode):
    _, ref = task; a, b, out = inputs(); check = ref.GemmCheck(a, b, out); calls = []
    def launch():
        calls.append(True)
        if mode == 'success': return
        if mode == 'input_mutation': a[0, 0] += 1
        if mode == 'output_mutation': out[0, 0] = 8
        if mode == 'wrong_error': raise ValueError('unrelated shape error')
        if mode == 'wrong_type': raise RuntimeError('num_ctas > 1 not supported on gfx950')
        raise ValueError('num_ctas > 1 not supported on gfx950')
    try:
        if mode == 'exact_error': ref.check_cta_rejection(launch, check, 'gfx950')
        else:
            with pytest.raises((ValueError, RuntimeError, AssertionError)): ref.check_cta_rejection(launch, check, 'gfx950')
        assert calls == [True]
    finally: check.restore()
    assert torch.equal(out, torch.full_like(out, -7)) and a[0, 0] == 2


@pytest.mark.parametrize('backend,support,num_ctas', [('hip', False, 4), ('hip', True, 4), ('cuda', False, 4), ('hip', False, 1)])
def test_capability_policy_does_not_invent_unsupported_backends(task, monkeypatch, backend, support, num_ctas):
    _, ref = task
    for name in ['triton', 'triton.backends', 'triton.backends.amd']:
        module = types.ModuleType(name); module.__path__ = []; monkeypatch.setitem(sys.modules, name, module)
    target = types.SimpleNamespace(backend=backend, arch='gfx950')
    monkeypatch.setitem(sys.modules, 'triton.runtime', types.SimpleNamespace(driver=types.SimpleNamespace(active=types.SimpleNamespace(get_current_target=lambda: target))))
    monkeypatch.setitem(sys.modules, 'triton.backends.amd.compiler', types.SimpleNamespace(amd=types.SimpleNamespace(supports_multi_cta_launch=lambda arch: support)))
    assert ref.rejected_cta_arch(num_ctas) == ('gfx950' if backend == 'hip' and not support and num_ctas > 1 else None)


@pytest.mark.parametrize('mode', ['good', 'expected_rejection', 'bad_rejection', 'modified_input', 'wrong_tail'])
def test_actual_original_function_pristine_oracle_and_rejection(task, monkeypatch, mode):
    path, ref = task
    node = next(n for n in ast.parse((path/'test_gemm_no_scf.py').read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'test_gemm_no_scf')
    node.decorator_list = []; observed = []
    class CPUTorch:
        def __getattr__(self, name):
            if name != 'randn': return getattr(torch, name)
            def randn(*args, **kwargs):
                kwargs['device'] = 'cpu'; return torch.randn(*args, **kwargs)
            return randn
    class Kernel:
        def __getitem__(self, grid):
            def launch(**kw):
                a, b, c = kw['a_ptr'], kw['b_ptr'], kw['c_ptr']
                observed.append((a, b, c, a.clone(), b.clone(), c.clone()))
                if mode == 'expected_rejection': raise ValueError('num_ctas > 1 not supported on gfx950')
                if mode == 'bad_rejection': raise ValueError('wrong compiler error')
                c.copy_(a.float() @ b.float())
                if mode == 'modified_input': a[0, 0] += 1
                if mode == 'wrong_tail': c[-1, -1] += 10
            return launch
    monkeypatch.setattr(ref, 'rejected_cta_arch', lambda n: 'gfx950' if n == 4 else None)
    namespace = {'torch': CPUTorch(), 'set_seed': lambda: torch.manual_seed(42),
                 'matmul_no_scf_kernel': Kernel(), 'result_gold': {}, 'assert_close': torch.testing.assert_close}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<protected-original>', 'exec'), namespace)
    request = types.SimpleNamespace(node=types.SimpleNamespace(name='cpu', user_properties=[]))
    args = (2, 2, 4, 4 if 'rejection' in mode else 1, 4, True, False, 'float32', True, request)
    if mode in ['good', 'expected_rejection']:
        namespace['test_gemm_no_scf'](*args)
        evidence = request.node.user_properties[0][1]
        assert evidence['readonly_input_checked']
        assert evidence.get('expected_error_verified') if mode == 'expected_rejection' else evidence['full_output_checked']
    else:
        with pytest.raises((ValueError, AssertionError)): namespace['test_gemm_no_scf'](*args)
        assert not request.node.user_properties
    a, b, c, pristine_a, pristine_b, pristine_c = observed[0]
    assert ref.equal_bytes(a, pristine_a) and ref.equal_bytes(b, pristine_b) and ref.equal_bytes(c, pristine_c)


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
    bench = adapter.benchmark_type(Base, plugin, None)(op); bench.context = {'a_host': a, 'b_host': b, 'c_buffer': out}
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


def test_original_kernel_performance_cases_and_numeric_gate(task, monkeypatch):
    path, _ = task; original = ORIGINAL[path.relative_to(ROOT).as_posix()]
    source = (path/'test_gemm_no_scf.py').read_text(); nodes = {n.name: n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
    for name, sha in original['functions'].items():
        assert hashlib.sha256(ast.get_source_segment(source, nodes[name]).encode()).hexdigest() == sha
    for name, sha in original['files'].items(): assert hashlib.sha256((path/name).read_bytes()).hexdigest() == sha
    for name, sha in original['decorators'].items():
        assert hashlib.sha256(ast.dump(ast.Module(body=nodes[name].decorator_list, type_ignores=[])).encode()).hexdigest() == sha
    rows = json.loads((path/'workloads.json').read_text())['cases']
    assert len(rows) == 56 and sum('performance' in r['checks'] for r in rows) == 20
    assert hashlib.sha256(json.dumps(rows[:52], sort_keys=True, separators=(',', ':')).encode()).hexdigest() == original['rows']
    assert all(r['checks'] == ['correctness'] for r in rows[52:])
    assert 'assert_close(c, golden, rtol=1e-2, atol=1e-3, check_dtype=False)' in source
    adapter = load(path/'_arena_eval.py'); monkeypatch.setattr(pytest, 'main', lambda *args, **kwargs: 0)
    result = adapter.evaluate('task', 'validate-task')
    from src.task_protocol import parse_command_result
    assert parse_command_result('ARENA_EVAL_RESULT=' + json.dumps(result), role='task', action='validate-task', returncode=0).status == 'PASS'
    assert len(result['cases']) == 56


def test_independent_manifest_matches_every_real_parametrize(task):
    path, _ = task
    adapter = load(path/'_arena_eval.py')
    tree = ast.parse((path/'test_gemm_no_scf.py').read_text())
    found = {}
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef): continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call) or ast.unparse(decorator.func) != 'pytest.mark.parametrize': continue
            keys = ast.literal_eval(decorator.args[0]).replace(' ', '').split(',')
            values = eval(compile(ast.Expression(body=decorator.args[1]), '<original-params>', 'eval'), {'itertools': itertools})
            for row in values:
                params = dict(zip(keys, row))
                found[adapter.identity(node.name, params)] = {'function': node.name, 'arguments': params}
    expected = {r['test_case_id']: r['params'] for r in json.loads((path/'workloads.json').read_text())['cases']}
    assert found == expected and len(found) == 56


def test_fp32_oracle_is_not_rounded_to_half_before_comparison(task):
    _, ref = task
    a = torch.tensor([[.3333, .1]], dtype=torch.float16)
    b = torch.tensor([[3.], [.3]], dtype=torch.float16)
    out = torch.zeros((1, 1), dtype=torch.float16)
    check = ref.GemmCheck(a, b, out)
    scalar = sum(float(x) * float(y) for x, y in zip(a.flatten(), b.flatten()))
    assert check.expected.item() == torch.tensor(scalar, dtype=torch.float32).item()
    assert check.expected.item() != check.expected.half().float().item()
    out.copy_(check.expected); check(out)


# Stable expectations; no Git history needed in CI.
ORIGINAL = {'tasks/instruction2triton/rocmbench/test_gemm_no_scf': {'decorators': {'calculate_gemm_no_scf_gbps': '3543b4693a36a1098850b8bc928887694ed59a6deb7d3dfd0339de01f55a77b6',
                                                                        'calculate_gemm_no_scf_tflops': '3543b4693a36a1098850b8bc928887694ed59a6deb7d3dfd0339de01f55a77b6',
                                                                        'gemm_no_scf_triton_wrapper': '3543b4693a36a1098850b8bc928887694ed59a6deb7d3dfd0339de01f55a77b6',
                                                                        'is_hip': '3543b4693a36a1098850b8bc928887694ed59a6deb7d3dfd0339de01f55a77b6',
                                                                        'matmul_no_scf_kernel': '6978ec5d24aebd8ab58b32e8b910203a142e1c6a3159d082fd6c7179b0c8dc04',
                                                                        'set_seed': '3543b4693a36a1098850b8bc928887694ed59a6deb7d3dfd0339de01f55a77b6',
                                                                        'test_gemm_no_scf': 'd8f19f3a2223d6821c34ba693a265c486e716edcfafb1deb92d708deda454327',
                                                                        'test_performance': '8cd711a1611d993b3a80c766347378269ddc11824bfd4193881bfee12ee49b7b',
                                                                        'test_save_performance_results': '3543b4693a36a1098850b8bc928887694ed59a6deb7d3dfd0339de01f55a77b6',
                                                                        'test_save_results': '3543b4693a36a1098850b8bc928887694ed59a6deb7d3dfd0339de01f55a77b6'},
                                                         'files': {'config.yaml': '4021d5e191c8e5aa4ab7f03b2e163e62fae90d7134e9cf6a48bd2fc127ba00fe',
                                                                   'performance_utils_pytest.py': 'e0dfef878da9ac042c39f1832e7940520fa1738ab29d9af2fcdbbc7e42d146c9'},
                                                         'functions': {'calculate_gemm_no_scf_gbps': '46d8366a0efc2d55546c836a2ac7d4e0cd5fc28984aa0249d39472d9e90b39d8',
                                                                       'calculate_gemm_no_scf_tflops': 'e00fdd0e78c2c271389fd816e7f9de2cf2777aae30804f51fea99b82972e789b',
                                                                       'gemm_no_scf_triton_wrapper': 'a437ee2551e874aedd438934cb6a8446bfd3e3ecff97b4f9e562a76c348d8272',
                                                                       'is_hip': '18a1484321fc6773db7fa5328ed8ad9af0261147caf1ed9d9ef3be4d42408c57',
                                                                       'matmul_no_scf_kernel': '97cb9773c82c8f4bcb267f0bb024d94e795d0154e55283f6a01f5cb41a7313ff',
                                                                       'set_seed': 'cfdef0cf93153372206c26f3a11dfe9ecd53892109b325cb7dc55337e70be1ce',
                                                                       'test_performance': 'ed3bc5f30244aec77a3d79f53cab766816dab3e652410629ab1f0cd90767a774',
                                                                       'test_save_performance_results': 'e75049a8bfa70933554c63918336d2909e576a9a7940225ed7fa3c50aad1b8d9',
                                                                       'test_save_results': '557a528e777fd099f5b131e01721dfebf7caa45416f1f0864010681d99fa97b8'},
                                                         'rows': '490ac51a96dfb8b58482eb4430f3c6994d52eb7c89991e5841e2a63b68bb6bcf'},
 'tasks/triton2triton/rocmbench/medium/test_gemm_no_scf': {'decorators': {'calculate_gemm_no_scf_gbps': '3543b4693a36a1098850b8bc928887694ed59a6deb7d3dfd0339de01f55a77b6',
                                                                          'calculate_gemm_no_scf_tflops': '3543b4693a36a1098850b8bc928887694ed59a6deb7d3dfd0339de01f55a77b6',
                                                                          'gemm_no_scf_triton_wrapper': '3543b4693a36a1098850b8bc928887694ed59a6deb7d3dfd0339de01f55a77b6',
                                                                          'is_hip': '3543b4693a36a1098850b8bc928887694ed59a6deb7d3dfd0339de01f55a77b6',
                                                                          'matmul_no_scf_kernel': '6978ec5d24aebd8ab58b32e8b910203a142e1c6a3159d082fd6c7179b0c8dc04',
                                                                          'set_seed': '3543b4693a36a1098850b8bc928887694ed59a6deb7d3dfd0339de01f55a77b6',
                                                                          'test_gemm_no_scf': 'd8f19f3a2223d6821c34ba693a265c486e716edcfafb1deb92d708deda454327',
                                                                          'test_performance': '8cd711a1611d993b3a80c766347378269ddc11824bfd4193881bfee12ee49b7b',
                                                                          'test_save_performance_results': '3543b4693a36a1098850b8bc928887694ed59a6deb7d3dfd0339de01f55a77b6',
                                                                          'test_save_results': '3543b4693a36a1098850b8bc928887694ed59a6deb7d3dfd0339de01f55a77b6'},
                                                           'files': {'config.yaml': '4021d5e191c8e5aa4ab7f03b2e163e62fae90d7134e9cf6a48bd2fc127ba00fe',
                                                                     'performance_utils_pytest.py': 'e0dfef878da9ac042c39f1832e7940520fa1738ab29d9af2fcdbbc7e42d146c9'},
                                                           'functions': {'calculate_gemm_no_scf_gbps': '46d8366a0efc2d55546c836a2ac7d4e0cd5fc28984aa0249d39472d9e90b39d8',
                                                                         'calculate_gemm_no_scf_tflops': 'e00fdd0e78c2c271389fd816e7f9de2cf2777aae30804f51fea99b82972e789b',
                                                                         'gemm_no_scf_triton_wrapper': 'a437ee2551e874aedd438934cb6a8446bfd3e3ecff97b4f9e562a76c348d8272',
                                                                         'is_hip': '18a1484321fc6773db7fa5328ed8ad9af0261147caf1ed9d9ef3be4d42408c57',
                                                                         'matmul_no_scf_kernel': '97cb9773c82c8f4bcb267f0bb024d94e795d0154e55283f6a01f5cb41a7313ff',
                                                                         'set_seed': 'cfdef0cf93153372206c26f3a11dfe9ecd53892109b325cb7dc55337e70be1ce',
                                                                         'test_performance': 'b206752f784f38ae4864dbbcd3254fb19226a3937ec5fc4ffac2e14afec23fcd',
                                                                         'test_save_performance_results': 'e75049a8bfa70933554c63918336d2909e576a9a7940225ed7fa3c50aad1b8d9',
                                                                         'test_save_results': '557a528e777fd099f5b131e01721dfebf7caa45416f1f0864010681d99fa97b8'},
                                                           'rows': '490ac51a96dfb8b58482eb4430f3c6994d52eb7c89991e5841e2a63b68bb6bcf'}}
