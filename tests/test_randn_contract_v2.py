"""CPU RNG contract/adapter regressions; GPU qualification is recorded separately."""
import ast
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
TASKS = ['triton2triton/rocmbench/easy/test_randn', 'instruction2triton/rocmbench/test_randn']


def load(path):
    spec = importlib.util.spec_from_file_location('_rng_test_' + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module', autouse=True)
def cpu_thread_budget():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(params=TASKS, ids=['triton', 'instruction'])
def task(request, monkeypatch):
    path = ROOT / 'tasks' / request.param
    ref = load(path / '_arena_reference.py')
    monkeypatch.setitem(sys.modules, '_arena_reference', ref)
    return path, ref


def scalar_philox(counter, key):
    """Independent arbitrary-precision scalar implementation, checked against KATs."""
    a, b, c, d = counter
    k0, k1 = key
    mask = (1 << 32) - 1
    for _ in range(10):
        product_a = a * 0xD2511F53
        product_c = c * 0xCD9E8D57
        a, b, c, d = ((product_c >> 32) ^ b ^ k0, product_c & mask,
                      (product_a >> 32) ^ d ^ k1, product_a & mask)
        k0 = (k0 + 0x9E3779B9) & mask
        k1 = (k1 + 0xBB67AE85) & mask
    return a, b, c, d


# Random123 v1.14.0 tests/kat_vectors; all four lanes, not a task-generated answer.
@pytest.mark.parametrize('counter,key,expected', [
    ((0, 0, 0, 0), (0, 0), (0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8)),
    ((0xffffffff,) * 4, (0xffffffff,) * 2, (0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd)),
    ((0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344), (0xa4093822, 0x299f31d0),
     (0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1)),
])
def test_independent_scalar_random123_known_answers(counter, key, expected):
    assert scalar_philox(counter, key) == expected


@pytest.mark.parametrize('seed', [0, 42, 124, 54])
def test_task_philox_and_uniform_against_independent_scalar(task, seed):
    _, ref = task
    offsets = [0, 1, 2, 17, 1023, 1024, 99999]
    words = ref.philox32(seed, max(offsets) + 1)
    floats = ref.expected_uniform(seed, max(offsets) + 1, 'cpu')
    for offset in offsets:
        word = scalar_philox((offset, 0, 0, 0), (seed, 0))[0]
        assert int(words[offset]) == word
        magnitude = (~word & 0xffffffff) if word & (1 << 31) else word
        expected = np.float32(magnitude) * np.float32(4.6566127342e-10)
        assert floats[offset].item() == float(expected)
    assert floats.dtype == torch.float32 and floats.shape == (100000,)
    assert bool(torch.all((floats >= 0) & (floats < 1)))


def ks(values):
    x = np.sort(values.numpy())
    n = len(x)
    return max(np.max(np.arange(1, n + 1) / n - x), np.max(x - np.arange(0, n) / n))


@pytest.mark.parametrize('seed', [0, 42, 124, 54])
def test_statistically_identical_permutation_is_rejected(task, seed):
    _, ref = task
    good = ref.expected_uniform(seed, 100000, 'cpu')
    bad = good.roll(1)
    assert bool(torch.all((bad >= 0) & (bad <= 1)))
    assert ks(good) == ks(bad) < 0.01
    ref.check_seeded_output(good, seed, 100000)
    with pytest.raises(ref.NumericalMismatch):
        ref.check_seeded_output(bad, seed, 100000)


@pytest.mark.parametrize('seed,dtype,const_seed', [(s, d, c) for s in [0, 42, 124, 54]
                                                  for d in ['int32', 'int64'] for c in [True, False]])
def test_original16_statistical_functions_also_run_exact_oracle(task, seed, dtype, const_seed):
    path, ref = task
    calls = []
    class Kernel:
        def __init__(self, const): self.const = const
        def __getitem__(self, grid):
            def launch(x, n, *args, **kwargs):
                assert grid == (98,) and n == 100000 and kwargs['dtype'] == dtype
                assert (not args) == self.const
                actual_seed = kwargs['seed'] if self.const else args[0]
                assert actual_seed == seed
                calls.append(self.const)
                x.copy_(ref.expected_uniform(actual_seed, n, 'cpu'))
            return launch
    source = (path / 'test_randn.py').read_text()
    fn = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'test_rand')
    fn.decorator_list = []
    ns = dict(torch=torch, np=np, set_seed=lambda s: None, BLOCK=1024,
              triton=types.SimpleNamespace(cdiv=lambda n, b: (n + b - 1) // b),
              tl=types.SimpleNamespace(int32='int32', int64='int64'),
              randn_kernel_const_seed=Kernel(True), randn_kernel_runtime_seed=Kernel(False),
              result_gold={}, all=lambda x: bool(torch.all(x)))
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path/'test_randn.py'), 'exec'), ns)
    request = types.SimpleNamespace(node=types.SimpleNamespace(name='original_cpu_case', user_properties=[]))
    ns['test_rand'](100000, seed, dtype, const_seed, request, device='cpu')
    metrics = dict(request.node.user_properties)['rng_contract']
    assert calls == [const_seed]
    assert metrics['exact_seeded_sequence_checked'] and metrics['original_range_checked']
    assert metrics['original_ks_checked'] and metrics['ks_stat'] < metrics['ks_limit'] == 0.01
    # Exercise the actual harness hook, not only the checker in isolation.
    original = ref.check_seeded_output
    def corrupt_actual(output, seed, count):
        output.copy_(output.roll(1))
        original(output, seed, count)
    ref.check_seeded_output = corrupt_actual
    with pytest.raises(ref.NumericalMismatch):
        ns['test_rand'](100000, seed, dtype, const_seed, request, device='cpu')


@pytest.mark.parametrize('change', ['dtype', 'shape', 'nan', 'wrong_seed', 'tail', 'return_alias'])
def test_output_contract_and_exact_negative_controls(task, change):
    _, ref = task
    out = ref.expected_uniform(42, 16, 'cpu')
    c = dict(seed_val=42, N_elements=16, x_output_buffer=out)
    checker = ref.prepare(c, None)
    if change == 'return_alias':
        with pytest.raises(ValueError): checker(out.clone())
        return
    if change == 'dtype': out = out.double()
    if change == 'shape': out = out.reshape(2, 8)
    if change == 'nan': out[-1] = float('nan')
    if change == 'wrong_seed': out = ref.expected_uniform(124, 16, 'cpu')
    if change == 'tail': out[-1] = torch.nextafter(out[-1], torch.tensor(1.))
    with pytest.raises((ValueError, ref.NumericalMismatch)):
        ref.check_seeded_output(out, 42, 16)


@pytest.mark.parametrize('action', ['correctness', 'performance'])
@pytest.mark.parametrize('mode', ['good', 'events', 'unwritten_replay', 'corrupt_replay', 'wrong_timed',
                                 'wrong_alias', 'unobservable', 'crash', 'wrong_timing'])
def test_actual_adapter_bound_replay_restore_and_unchanged_timing(task, monkeypatch, action, mode):
    path, ref = task
    adapter = load(path / '_arena_eval.py')
    out = torch.full((16,), 19.)
    saved = out.clone()
    plugin = types.SimpleNamespace(action=action, current_row={'test_case_id': 'cpu'}, exercised=set())
    stage = ['initial']
    timer_options = []
    class Base:
        def __init__(self, op_callable):
            self.op_callable = op_callable
            self.prepare_fn = None
            self.use_cuda_graph = mode != 'events'
            self.fallback_reason = 'explicit event path' if mode == 'events' else None
            self.config = types.SimpleNamespace(warm_up=10, repetition=100)
    class Timed:
        def rerun(self):
            stage[0] = 'replay'
            return self.fn()
    def benchmark(fn, **kwargs):
        timer_options.append(kwargs)
        assert {k: v for k, v in kwargs.items() if k != 'timed_run'} == dict(
            warmup=10, repetition=100, prepare_fn=None, use_cuda_graph=mode != 'events',
            fallback_reason='explicit event path' if mode == 'events' else None)
        if mode in ['unobservable', 'crash']: raise RuntimeError(mode)
        stage[0] = 'timed'
        t = kwargs['timed_run']; t.fn = fn; t.outputs = fn()
        metadata = {'benchmark_method': 'cuda_event_fallback' if mode == 'events' else 'cuda_graph',
                    'benchmark_samples': 100, 'benchmark_warmup': 10}
        if mode == 'events': metadata['benchmark_fallback_reason'] = 'explicit event path'
        return ([float('nan')] if mode == 'wrong_timing' else [1., 2.]), metadata
    monkeypatch.setitem(sys.modules, '_aka_benchmark', types.SimpleNamespace(
        TimedRun=Timed, benchmark_cuda_graph_or_events_samples=benchmark))
    monkeypatch.setitem(sys.modules, 'performance_utils_pytest', types.SimpleNamespace(
        _compute_timing_stats=lambda ts, cfg: {'mean': sum(ts)/len(ts)}))
    def op():
        if not (stage[0] == 'replay' and mode == 'unwritten_replay'):
            out.copy_(ref.expected_uniform(42, 16, 'cpu'))
        if (stage[0] == 'replay' and mode == 'corrupt_replay') or (stage[0] == 'timed' and mode == 'wrong_timed'):
            out[-1] += 0.1
        return out.clone() if stage[0] == 'timed' and mode == 'wrong_alias' else out
    wrapped = adapter.benchmark_type(Base, plugin, None)(op)
    wrapped.context = dict(seed_val=42, N_elements=16, x_output_buffer=out)
    if action == 'correctness' or mode in ['good', 'events']:
        wrapped.run_benchmark(baseline_callable=lambda: pytest.fail('peer timing forbidden'))
        assert plugin.exercised == {'cpu'}
        if action == 'performance':
            meta = plugin.current_row['metadata']
            assert meta['poisoned_output_replay_checked'] and meta['replay_seed_unchanged']
            assert meta['timed_output_checked'] and meta['output_state_restored']
            assert 'fresh_input_replay_checked' not in meta
            assert plugin.current_row['execution_time_ms'] == 1.5
            assert meta['device_timing']['benchmark_samples'] == 100
            if mode == 'events':
                assert meta['device_timing']['benchmark_fallback_reason'] == 'explicit event path'
            # Compare effective defaults with the actual original helper path,
            # including graph batching/calibration defaults not stated by tasks.
            helper = load(ROOT/'src/tools/perf/performance_utils_pytest.py')
            original_options = []
            monkeypatch.setattr(helper, 'benchmark_cuda_graph_or_events_samples',
                lambda fn, **kw: (original_options.append(kw) or [1.], {}))
            helper._measure_times(op, wrapped.config, prepare_fn=wrapped.prepare_fn,
                                  use_cuda_graph=wrapped.use_cuda_graph,
                                  fallback_reason=wrapped.fallback_reason)
            canonical = load(ROOT/'src/tools/perf/aka_benchmark.py')
            signature = inspect.signature(canonical.benchmark_cuda_graph_or_events_samples)
            def effective(options):
                bound = signature.bind_partial(None, **options)
                bound.apply_defaults()
                return {k: v for k, v in bound.arguments.items() if k not in ('fn', 'timed_run')}
            assert effective(timer_options[0]) == effective(original_options[0])
    else:
        with pytest.raises((ref.NumericalMismatch, RuntimeError, ValueError)):
            wrapped.run_benchmark()
        assert not plugin.exercised and 'execution_time_ms' not in plugin.current_row
    assert torch.equal(out, saved)


def test_successful_report_carries_executed_statistical_evidence(task):
    path, _ = task
    adapter = load(path / '_arena_eval.py')
    manifest = json.loads((path/'workloads.json').read_text())
    plugin = adapter.ReportPlugin(manifest, 'correctness')
    key = next(k for k, r in plugin.rows.items() if r['params']['function'] == 'test_rand')
    plugin.node_rows['node'] = key
    report = types.SimpleNamespace(nodeid='node', when='call', failed=False, skipped=False,
        user_properties=[('rng_contract', {'exact_seeded_sequence_checked': True, 'ks_stat': 0.003})])
    plugin.pytest_runtest_logreport(report)
    assert plugin.rows[key]['status'] == 'PASS'
    assert plugin.rows[key]['metrics']['exact_seeded_sequence_checked']
    assert plugin.rows[key]['metrics']['ks_stat'] == 0.003


# Original immutable digests are checked in below; pytest never invokes git.
ORIGINAL = {'instruction2triton/rocmbench/test_randn': {'config.yaml': 'e852b6187590249439310ce333ae2fa9511eac1421037a8c9a24e60bb8f88108',
                                             'performance_utils_pytest.py': 'e0dfef878da9ac042c39f1832e7940520fa1738ab29d9af2fcdbbc7e42d146c9',
                                             'randn_kernel_const_seed': '5a8e8a1c142c14ea16672bab148824961da0ff8ba7a455f3829d3d50e13bca4b',
                                             'randn_kernel_const_seed_decorators': '6978ec5d24aebd8ab58b32e8b910203a142e1c6a3159d082fd6c7179b0c8dc04',
                                             'randn_kernel_runtime_seed': '730fb8425143b89a961ed66da3273ed2796ef8216471aebe3bb393ee3791f1eb',
                                             'randn_kernel_runtime_seed_decorators': '6978ec5d24aebd8ab58b32e8b910203a142e1c6a3159d082fd6c7179b0c8dc04',
                                             'randn_triton_wrapper': '5c7c5b29429f86349517f557e879e055c265674779a5f855b5bcdf79ca42e44c',
                                             'randn_triton_wrapper_decorators': '3543b4693a36a1098850b8bc928887694ed59a6deb7d3dfd0339de01f55a77b6',
                                             'set_seed': 'cfdef0cf93153372206c26f3a11dfe9ecd53892109b325cb7dc55337e70be1ce',
                                             'set_seed_decorators': '3543b4693a36a1098850b8bc928887694ed59a6deb7d3dfd0339de01f55a77b6',
                                             'test_performance': '5162e2d503f4265a66e2989f143bdebfd6aa797e30b35a7d34e846000545c9be',
                                             'test_performance_decorators': '044b23f2e1b831142f93045934a58b25cbcbd941a22a8c18e55189ada00aaff6',
                                             'workloads.json': '5b3a1fd031b5d290f445ab7f64ccc4b7d3b5a751a3f3d1ad76d75b2bafff7fb5'},
 'triton2triton/rocmbench/easy/test_randn': {'config.yaml': 'e852b6187590249439310ce333ae2fa9511eac1421037a8c9a24e60bb8f88108',
                                             'performance_utils_pytest.py': 'e0dfef878da9ac042c39f1832e7940520fa1738ab29d9af2fcdbbc7e42d146c9',
                                             'randn_kernel_const_seed': '5a8e8a1c142c14ea16672bab148824961da0ff8ba7a455f3829d3d50e13bca4b',
                                             'randn_kernel_const_seed_decorators': '6978ec5d24aebd8ab58b32e8b910203a142e1c6a3159d082fd6c7179b0c8dc04',
                                             'randn_kernel_runtime_seed': '730fb8425143b89a961ed66da3273ed2796ef8216471aebe3bb393ee3791f1eb',
                                             'randn_kernel_runtime_seed_decorators': '6978ec5d24aebd8ab58b32e8b910203a142e1c6a3159d082fd6c7179b0c8dc04',
                                             'randn_triton_wrapper': '5c7c5b29429f86349517f557e879e055c265674779a5f855b5bcdf79ca42e44c',
                                             'randn_triton_wrapper_decorators': '3543b4693a36a1098850b8bc928887694ed59a6deb7d3dfd0339de01f55a77b6',
                                             'set_seed': 'cfdef0cf93153372206c26f3a11dfe9ecd53892109b325cb7dc55337e70be1ce',
                                             'set_seed_decorators': '3543b4693a36a1098850b8bc928887694ed59a6deb7d3dfd0339de01f55a77b6',
                                             'test_performance': 'f60b6d3753b4b509da7a0b6aa5d527366b36e1b737684e8983776825c25f0e42',
                                             'test_performance_decorators': '044b23f2e1b831142f93045934a58b25cbcbd941a22a8c18e55189ada00aaff6',
                                             'workloads.json': 'e3908ae54dd32da497218fe5f40256987317c17d8d61e5ef3b5328361a4b9712'}}


def test_original_kernels_manifest_seeds_timing_and_generated_stub_are_preserved(task):
    path, _ = task
    expected = ORIGINAL[path.relative_to(ROOT/'tasks').as_posix()]
    source = (path/'test_randn.py').read_text()
    tree = ast.parse(source)
    for name in ['randn_kernel_runtime_seed', 'randn_kernel_const_seed', 'randn_triton_wrapper',
                 'set_seed', 'test_performance']:
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest() == expected[name]
        assert hashlib.sha256(ast.dump(ast.Module(body=node.decorator_list, type_ignores=[])).encode()).hexdigest() == expected[name+'_decorators']
    for name in ['config.yaml', 'performance_utils_pytest.py', 'workloads.json']:
        data=(path/name).read_bytes()
        if name=='workloads.json' and 'triton2triton' in path.parts:
            manifest=json.loads(data)
            assert len(manifest['cases'][120:])==9
            assert all(r['checks']==['correctness'] for r in manifest['cases'][120:])
            manifest['cases']=manifest['cases'][:120]
            data=(json.dumps(manifest,indent=2)+'\n').encode()
        assert hashlib.sha256(data).hexdigest() == expected[name]
    manifest = json.loads((path/'workloads.json').read_text())
    assert len(manifest['cases']) == (129 if 'triton2triton' in path.parts else 120)
    assert sum(r['params']['function'] == 'test_rand' for r in manifest['cases']) == 16
    assert sum('performance' in r['checks'] for r in manifest['cases']) == 104
    assert 'assert all((x >= 0) & (x <= 1))' in source
    assert 'ks_stat = max(np.max(np.arange(1, n + 1) / n - x_np), np.max(x_np - np.arange(0, n) / n))' in source
    assert 'assert ks_stat < 0.01' in source
    stats = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'test_rand')
    decorator = next(n for n in stats.decorator_list if isinstance(n, ast.Call))
    ns = {}
    rows = eval(compile(ast.Expression(decorator.args[1]), '<original-case-params>', 'eval'), ns)
    assert set(rows) == {(100000, s, d, c) for s in [0, 42, 124, 54]
                         for d in ['int32', 'int64'] for c in [True, False]}
