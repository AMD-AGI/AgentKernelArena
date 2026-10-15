"""Source and CPU acceptance regressions; simulated timing is not GPU validation."""
import ast
import functools
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from triton_signal_migration_helpers import historical_manifest, original_ssd_draw, SCALES


ROOT = Path(__file__).resolve().parents[1]
NAMES = tuple(SCALES)


def original_source(path):
    return subprocess.check_output(['git', 'show', 'HEAD:' + path.relative_to(ROOT).as_posix()], cwd=ROOT, text=True)


@pytest.fixture(params=NAMES)
def task(request, monkeypatch):
    path = ROOT / 'tasks/triton2triton/vllm' / request.param
    for key in list(sys.modules):
        if key == 'scripts' or key.startswith('scripts.'):
            monkeypatch.delitem(sys.modules, key)
    monkeypatch.syspath_prepend(str(path))
    monkeypatch.chdir(path)
    spec = importlib.util.spec_from_file_location('_signal_distribution', path / '_arena_eval.py')
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    h = adapter.load_harness()
    monkeypatch.setattr(adapter, 'load_harness', lambda: h)
    for name in ('randn', 'rand', 'arange', 'zeros', 'empty', 'full'):
        original = getattr(torch, name)
        def factory(*args, _original=original, **kwargs):
            if kwargs.get('device') == 'cuda':
                kwargs['device'] = 'cpu'
            return _original(*args, **kwargs)
        monkeypatch.setattr(torch, name, factory)
    original_to = torch.Tensor.to
    def cpu_to(self, *args, **kwargs):
        if args and args[0] == 'cuda':
            args = ('cpu', *args[1:])
        if kwargs.get('device') == 'cuda':
            kwargs['device'] = 'cpu'
        return original_to(self, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'to', cpu_to)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield path, h, adapter
    torch.set_num_threads(threads)


def draws(path, h):
    old = original_source(path / 'scripts/task_runner.py')
    if path.name == 'triton_ssd_chunk_scan':
        old_draw = original_ssd_draw(old)
        for index, shape in enumerate(h.TEST_SHAPES):
            for seed in (42 + index, 0):
                yield old_draw(shape, seed, 'cpu'), h.gen_inputs(shape, seed, 'cpu', historical=True), h.gen_inputs(shape, seed, 'cpu')
    else:
        tree = ast.parse(old)
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'gen_inputs')
        namespace = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<original generator>', 'exec'), namespace)
        for seed in h.SEEDS:
            old_args, _ = namespace['gen_inputs'](seed, 'cpu')
            historical, _ = h.gen_inputs(seed, 'cpu', historical=True)
            current, _ = h.gen_inputs(seed, 'cpu')
            names = ('q', 'k', 'v', 'g', 'beta', 'scale', 'h0')
            yield dict(zip(names, old_args)), dict(zip(names, historical)), dict(zip(names, current))


def test_all_old_draws_are_exact_and_only_declared_operands_scale(task):
    path, h, _ = task
    count = 0
    for old, historical, current in draws(path, h):
        count += 1
        assert old.keys() == historical.keys() == current.keys()
        for key, original in old.items():
            if isinstance(original, torch.Tensor):
                assert torch.equal(historical[key], original), key
                assert torch.equal(current[key], original * SCALES[path.name].get(key, 1)), key
                assert current[key].dtype == original.dtype and current[key].shape == original.shape
            else:
                assert current[key] == historical[key] == original
    assert count == (10 if path.name == 'triton_ssd_chunk_scan' else 5)
    assert h.WARMUP_ITERATIONS == 10 and h.BENCHMARK_ITERATIONS == 100


def test_original_kernels_references_helpers_and_manifest_are_exact_after_reviewed_additions(task):
    path, _, adapter = task
    source = path / 'source' / (path.name + '.py')
    assert source.read_text() == original_source(source)
    before, after = original_source(path / 'scripts/task_runner.py'), (path / 'scripts/task_runner.py').read_text()
    old = {n.name: n for n in ast.parse(before).body if isinstance(n, ast.FunctionDef)}
    new = {n.name: n for n in ast.parse(after).body if isinstance(n, ast.FunctionDef)}
    for name, node in old.items():
        if name.startswith('reference'):
            assert ast.dump(node) == ast.dump(new[name])
    region = lambda text: text.split('# >>> AKA-GENERATED:')[1].split('# <<< AKA-GENERATED <<<')[0]
    assert region(after) == region(before)
    assert historical_manifest(adapter.load_manifest(), path.name) == json.loads(original_source(path / 'workloads.json'))


def ssd_independent_terms(inputs, chunk):
    # FP64 causal matrix oracle, independent of the original token-loop FP32
    # additions. Subtracting inter from a rounded total loses small intra bits.
    cb, x, dt, decay, C, states = (inputs[key].double() for key in
                                  ('cb', 'x', 'dt', 'dA_cumsum', 'C', 'states'))
    length, heads, _ = x.shape
    inter, intra = torch.zeros_like(x), torch.zeros_like(x)
    for c in range(length // chunk):
        start, end = c * chunk, (c + 1) * chunk
        for h in range(heads):
            group = h // (heads // C.shape[1])
            a = decay[h, c]
            coefficients = torch.tril(cb[c, group] * (a[:, None] - a[None, :]).exp()
                                      * dt[h, c][None, :])
            intra[start:end, h] = coefficients @ x[start:end, h]
            if c > 0 and inputs['seq_idx'][c] == inputs['seq_idx'][c - 1]:
                inter[start:end, h] = (C[start:end, group] @ states[c - 1, h].T) * a.exp()[:, None]
    return inter.float(), intra.float()


def test_all_scored_draws_reject_the_relevant_omission_at_original_tolerance(task):
    path, h, _ = task
    for old, _, current in draws(path, h):
        if path.name == 'triton_fla_fused_recurrent':
            keys = ('q', 'k', 'v', 'g', 'beta', 'scale', 'h0')
            old_values = h.semantic_controls.reference_outputs(*(old[key] for key in keys))
            expected = h.semantic_controls.reference_outputs(*(current[key] for key in keys))
            h.semantic_controls.require_signal(expected)
            torch.testing.assert_close(expected[0], old_values[0] * 64, atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(expected[1], old_values[1] * 8, atol=1e-6, rtol=1e-6)
            assert torch.allclose(torch.zeros_like(old_values[0]), old_values[0], atol=.05, rtol=.05)
        else:
            keys = ('cb', 'x', 'dt', 'dA_cumsum', 'C', 'states', 'seq_idx')
            chunk = old['dt'].shape[-1]
            old_values = h.reference_chunk_scan(*(old[key] for key in keys), chunk)
            expected = h.reference_chunk_scan(*(current[key] for key in keys), chunk)
            inter = h.semantic_controls.inter_chunk_reference(old['x'], old['dA_cumsum'], old['C'], old['states'], old['seq_idx'], chunk)
            independent_inter, independent_intra = ssd_independent_terms(old, chunk)
            torch.testing.assert_close(inter, independent_inter, atol=2e-6, rtol=2e-5)
            torch.testing.assert_close(old_values, independent_inter + independent_intra, atol=2e-6, rtol=2e-5)
            assert torch.allclose(inter, old_values, atol=.05, rtol=.05)
            h.semantic_controls.require_intra_signal(expected, inter)
            torch.testing.assert_close(expected, independent_inter + independent_intra * 32, atol=2e-6, rtol=2e-5)


def module_for(path, h, *, fault=None, calls=None):
    if path.name == 'triton_fla_fused_recurrent':
        public, symbol = h.semantic_controls.reference_outputs, 'fused_recurrent_gated_delta_rule_fwd'
    else:
        def public(cb, x, dt, dA_cumsum, C, states, cu_chunk_seqlens, out, seq_idx):
            out.copy_(h.reference_chunk_scan(cb, x, dt, dA_cumsum, C, states, seq_idx, dt.shape[-1]))
        symbol = 'chunk_scan_fwd'
    @functools.wraps(public)
    def operation(*args, **kwargs):
        inputs = inspect.signature(public).bind(*args, **kwargs).arguments
        if calls is not None:
            calls.append({key: value.clone() if isinstance(value, torch.Tensor) else value
                          for key, value in inputs.items() if key != 'out'})
        if fault == 'input':
            next(value for value in inputs.values() if isinstance(value, torch.Tensor)).add_(1)
        result = public(*args, **kwargs)
        if fault == 'output':
            value = result[0] if isinstance(result, tuple) else inputs['out']
            value.add_(1)
        return result
    return SimpleNamespace(**{symbol: operation})


@pytest.mark.parametrize('fault', [None, 'output', 'input'])
def test_historical_case_executes_and_retains_output_and_readonly_guards(task, monkeypatch, fault):
    path, h, _ = task
    monkeypatch.setattr(h, 'load_module', lambda: module_for(path, h, fault=fault))
    ok, error = h.run_correctness(case_index=20000)
    assert ok is (fault is None), error
    ok, _ = h.run_correctness(case_index=29999)
    assert not ok


@pytest.mark.parametrize('role', ['baseline', 'candidate'])
def test_adapter_routes_every_declared_historical_case_for_both_roles(task, monkeypatch, role):
    _, h, adapter = task
    cases, signal = [], []
    monkeypatch.setattr(h, 'run_correctness', lambda *, case_index: (cases.append(case_index) is None, None))
    monkeypatch.setattr(h, 'run_signal_controls', lambda *, case_index: signal.append(case_index) or {'status': 'PASS'})
    monkeypatch.setattr(h, 'run_semantic_controls', lambda: [], raising=False)
    result = adapter.evaluate(role, 'correctness')
    assert result['status'] == 'PASS'
    assert cases == signal == [row['params']['case_index'] for row in result['cases']]
    assert sum(20000 <= index < 30000 for index in cases) == len(h.HISTORICAL_CASES)
    assert len([row for row in adapter.load_manifest()['cases'] if 'performance' in row['checks']]) == 5


def test_manifest_rejects_distribution_or_historical_row_drift(task, monkeypatch):
    _, _, adapter = task
    data = adapter.load_manifest()
    data['historical_cases'][0]['seed'] = -1
    monkeypatch.setattr(adapter, 'load_manifest', lambda: data)
    result = adapter.evaluate('candidate', 'correctness')
    assert result['status'] == 'FAIL' and 'distribution/history disagrees' in result['reason']


def test_actual_performance_runner_uses_new_inputs_and_unchanged_timing_policy(task, monkeypatch):
    path, h, _ = task
    from src.tools.perf.aka_benchmark import TimedRun
    calls, options = [], []
    monkeypatch.setattr(h, 'load_module', lambda: module_for(path, h, calls=calls))
    monkeypatch.setitem(sys.modules, '_aka_benchmark', SimpleNamespace(TimedRun=TimedRun))
    h._TimedRun = TimedRun
    def benchmark(fn, *, timed_run, **kwargs):
        options.append(kwargs)
        output = fn()
        timed_run._bind(fn, output)
        return .125, {'benchmark_method': 'cuda_graph', 'cpu_fixture': True}
    monkeypatch.setattr(h, '_benchmark_cuda_graph_or_events', benchmark)
    rows = h.run_performance()
    assert len(rows) == 5 and all(row['execution_time_ms'] == .125 for row in rows), rows
    assert options == [{'warmup': 10, 'repetition': 100}] * 5
    assert len(calls) == 10
    for index, measured in enumerate(calls[::2]):
        if path.name == 'triton_ssd_chunk_scan':
            expected = h.gen_inputs(h.TEST_SHAPES[index], 0, 'cpu')
            for key, value in measured.items():
                assert torch.equal(value, expected[key]), key
        else:
            args, _ = h.gen_inputs(h.SEEDS[index], 'cpu')
            for actual, expected in zip(measured.values(), args):
                if isinstance(expected, torch.Tensor):
                    assert torch.equal(actual, expected)
                else:
                    assert actual == expected
