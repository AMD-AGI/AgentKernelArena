"""Actual task measurement driver rejects skipped calls and cached answers."""
import importlib.util
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch

from src.tools.perf import aka_benchmark
from src.tools.perf.aka_benchmark import TimedRun


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=['triton_ssd_bmm', 'triton_paged_prefix_prefill_alibi'])
def samples(request):
    path = ROOT / 'tasks/triton2triton/vllm' / request.param / '_arena_samples.py'
    spec = importlib.util.spec_from_file_location('measured_' + request.param, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('mode', ['correct', 'cached', 'only_nan', 'skip_middle', 'mutate', 'multi_call'])
def test_one_call_stream_checks_samples_and_retains_unpoisoned_control(samples, mode):
    a = torch.arange(1, 9, dtype=torch.float32)
    b = torch.arange(11, 19, dtype=torch.float32)
    before_a, before_b = a.clone(), b.clone()
    out = torch.full_like(a, float('nan'))
    phase = {'sample': None, 'calls': 0}
    observed_inputs = []

    def call():
        phase['calls'] += 1
        if mode == 'cached' and phase['calls'] > 1:
            return out
        if mode == 'only_nan' and torch.isfinite(out).all():
            return out
        if mode == 'skip_middle' and phase['sample'] == 50:
            return out
        out.copy_(a * b + b)
        if mode == 'mutate' and phase['sample'] == 50:
            a.add_(1)
        return out

    def benchmark(fn, *, warmup, repetition, max_graph_repeats, prepare_fn, timed_run):
        assert warmup == 10 and repetition == 100 and max_graph_repeats == 1
        for _ in range(warmup):
            prepare_fn(); fn()
        # Model capture and prime without counting either as measured samples.
        prepare_fn(); captured = fn()
        prepare_fn(); fn()
        for i in range(repetition):
            phase['sample'] = i
            prepare_fn()
            observed_inputs.append(a.clone())
            fn()
            timed_run.after_sample(captured)
        phase['sample'] = None
        def rerun():
            prepare_fn()
            return fn()
        timed_run._bind(rerun, captured)
        return 1., {'benchmark_method': 'cuda_graph',
                    'benchmark_timed_run_kind': 'captured_graph',
                    'benchmark_effective_repeats': 16 if mode == 'multi_call' else 1}

    kwargs = dict(benchmark=benchmark, timed=TimedRun(), call=call,
                  inputs={'a': a, 'b': b}, varied=('a', 'b'),
                  reference=lambda: a * b + b,
                  compare=lambda x, y: torch.testing.assert_close(x, y, atol=.01, rtol=.01),
                  warmup=10, repetition=100, atol=.01, rtol=.01)
    if mode == 'correct':
        _, metadata = samples.measure(**kwargs)
        assert metadata['measured_samples_checked'] == 100
        assert metadata['changed_input_replay_unpoisoned']
        assert len({x.numpy().tobytes() for x in observed_inputs}) == 100
    else:
        with pytest.raises(AssertionError):
            samples.measure(**kwargs)
    assert torch.equal(a, before_a) and torch.equal(b, before_b)


def test_task_drivers_route_scored_measurements_through_one_call_helper():
    import ast
    root = ROOT / 'tasks/triton2triton/vllm'
    for task, file, function in [
        ('triton_ssd_bmm', 'scripts/task_runner.py', 'run_performance'),
        ('triton_paged_prefix_prefill_alibi', '_arena_checks.py', 'checked_benchmark'),
    ]:
        tree = ast.parse((root / task / file).read_text())
        owner = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == function)
        assert any(isinstance(x, ast.ImportFrom) and x.module == '_arena_samples'
                   for x in ast.walk(owner))
        calls = [x for x in ast.walk(owner) if isinstance(x, ast.Call)
                 and isinstance(x.func, ast.Name) and x.func.id == 'measure']
        assert len(calls) == 1
        assert {'warmup', 'repetition', 'atol', 'rtol'} <= {x.arg for x in calls[0].keywords}


@pytest.mark.parametrize('metadata_fault', [None, 'missing_reason', 'wrong_kind', 'input_mutation'])
def test_real_event_fallback_observes_fresh_output_and_restores_inputs(
    samples, monkeypatch, metadata_fault
):
    """Exercise the canonical eager Event loop with a CPU-only Event stub."""
    class Event:
        def __init__(self, **_):
            pass

        def record(self, *_):
            pass

        def synchronize(self):
            pass

        def elapsed_time(self, _):
            return 1.0

    class Stream:
        def wait_stream(self, _):
            pass

    monkeypatch.setattr(aka_benchmark, '_require_gpu_timing', lambda: None)
    monkeypatch.setattr(torch.cuda, 'Event', Event)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(torch.cuda, 'current_stream', Stream)
    monkeypatch.setattr(torch.cuda, 'stream', lambda _: nullcontext())
    a = torch.arange(1, 9, dtype=torch.float32)
    b = torch.arange(11, 19, dtype=torch.float32)
    pristine_a, pristine_b = a.clone(), b.clone()
    calls = [0]
    outputs = []

    def call():
        calls[0] += 1
        output = a * b + b  # A fresh output allocation for each eager call.
        outputs.append(output)
        if metadata_fault == 'input_mutation' and calls[0] == 60:
            a.add_(1)
        return output

    def driver(fn, **kwargs):
        result, metadata = aka_benchmark.benchmark_cuda_graph_or_events(
            fn, use_cuda_graph=False, fallback_reason='cpu_test_explicit_event', **kwargs)
        if metadata_fault == 'missing_reason':
            metadata.pop('benchmark_fallback_reason')
        elif metadata_fault == 'wrong_kind':
            metadata['benchmark_timed_run_kind'] = 'captured_graph'
        return result, metadata

    request = dict(benchmark=driver, timed=TimedRun(), call=call,
                   inputs={'a': a, 'b': b}, varied=('a', 'b'),
                   reference=lambda: a * b + b,
                   compare=lambda x, y: torch.testing.assert_close(x, y, atol=.01, rtol=.01),
                   warmup=10, repetition=100, atol=.01, rtol=.01)
    if metadata_fault is None:
        _, metadata = samples.measure(**request)
        assert metadata['benchmark_method'] == 'cuda_event_fallback'
        assert metadata['benchmark_timed_run_kind'] == 'eager_callable'
        assert metadata['measured_samples_checked'] == 100
        assert len({id(output) for output in outputs}) == len(outputs)
    else:
        with pytest.raises(AssertionError):
            samples.measure(**request)
    assert torch.equal(a, pristine_a) and torch.equal(b, pristine_b)
