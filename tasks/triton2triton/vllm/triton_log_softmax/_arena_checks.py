"""Check log-softmax output contracts, stable tail inputs and actual replay."""
from contextlib import contextmanager
import inspect

SYMBOL = 'log_softmax'
TOL = 0.01


def unchanged(g, pristine):
    import torch
    if not torch.equal(g, pristine):
        raise AssertionError('Log-softmax modified its read-only input')


def reference(x, dim=-1):
    import torch.nn.functional as F
    return F.log_softmax(x.float().cpu(), dim=dim).to(device=x.device, dtype=x.dtype)


def check_output(output, expected):
    import torch
    if not isinstance(output, torch.Tensor) or (output.shape != expected.shape or
            output.dtype != expected.dtype or output.device != expected.device):
        raise AssertionError('Log-softmax output shape/dtype/device is invalid')
    if not torch.isfinite(output).all():
        raise AssertionError('Log-softmax output must be finite')
    torch.testing.assert_close(output, expected, atol=TOL, rtol=TOL)


@contextmanager
def checked_modules(harness):
    load_original = harness.load_module
    patched = []

    def load():
        module = load_original()
        original = getattr(module, SYMBOL)
        patched.append((module, original))

        def checked(x, dim=-1):
            pristine = x.clone()
            expected = reference(pristine, dim)
            diagnostic = pristine[:6, :7].contiguous().clone().reshape(2, 3, 7)
            diagnostic[0, 0].zero_()
            diagnostic[0, 1].copy_(x.new_tensor([-1000, -50, 0, 1, 2, 50, 1000]))
            saved = diagnostic.clone()
            wanted = reference(saved, 2)
            check_output(original(diagnostic, dim=2), wanted)
            unchanged(diagnostic, saved)
            output = original(x, dim=dim)
            unchanged(x, pristine)
            check_output(output, expected)
            return output

        setattr(module, SYMBOL, checked)
        return module

    harness.load_module = load
    try:
        yield
    finally:
        harness.load_module = load_original
        for module, original in reversed(patched):
            setattr(module, SYMBOL, original)


def checked_benchmark(harness, benchmark, fn, **options):
    c = inspect.getclosurevars(fn).nonlocals
    module, g = (c[name] for name in ('mod', 'x'))
    dim = -1
    pristine = g.clone()
    expected = reference(pristine, dim)
    original = getattr(module, SYMBOL)
    captured = None

    def collect(*args, **kwargs):
        nonlocal captured
        captured = original(*args, **kwargs)
        return captured

    def measured():
        nonlocal captured
        captured = None
        fn()
        if captured is None:
            raise AssertionError('Benchmark did not invoke the log-softmax candidate')
        return captured

    setattr(module, SYMBOL, collect)
    try:
        timed = harness._TimedRun()
        ms, metadata = benchmark(measured, timed_run=timed, **options)
        unchanged(g, pristine)
        check_output(timed.outputs, expected)
        g.mul_(-1).add_(4)
        replay_pristine = g.clone()
        replay_expected = reference(replay_pristine, dim)
        timed.outputs.fill_(float('nan'))
        replayed = timed.rerun()
        unchanged(g, replay_pristine)
        check_output(replayed, replay_expected)
        return ms, {**metadata, 'timed_output_checked': True,
                    'perturbed_input_replay_checked': True, 'source_buffers_unchanged': True}
    finally:
        setattr(module, SYMBOL, original)
        g.copy_(pristine)


def install(harness):
    correctness_original = harness.run_correctness
    performance_original = harness.run_performance

    def correctness(*args, **kwargs):
        with checked_modules(harness):
            return correctness_original(*args, **kwargs)

    def performance():
        benchmark = harness._benchmark_cuda_graph_or_events
        harness._benchmark_cuda_graph_or_events = lambda fn, **kwargs: checked_benchmark(harness, benchmark, fn, **kwargs)
        try:
            return performance_original()
        finally:
            harness._benchmark_cuda_graph_or_events = benchmark

    harness.run_correctness = correctness
    harness.run_performance = performance
