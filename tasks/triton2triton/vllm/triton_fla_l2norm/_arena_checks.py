"""Check L2 output contracts, epsilon/tail diagnostics and actual measured replay."""
from contextlib import contextmanager
import inspect

SYMBOL = 'l2norm_fwd'
TOL = 0.0001


def unchanged(g, pristine):
    import torch
    if not torch.equal(g, pristine):
        raise AssertionError('L2 normalization modified its read-only input')


def reference(harness, x, eps):
    return harness.reference(x.cpu(), eps).to(device=x.device, dtype=x.dtype)


def check_output(output, expected):
    import torch
    if not isinstance(output, torch.Tensor) or (output.shape != expected.shape or
            output.dtype != expected.dtype or output.device != expected.device):
        raise AssertionError('L2 normalization output shape/dtype/device is invalid')
    if not torch.isfinite(output).all():
        raise AssertionError('L2 normalization output must be finite')
    torch.testing.assert_close(output, expected, atol=TOL, rtol=TOL)


def diagnostic_input(x):
    # Rank/row/feature tails plus a zero and a near-zero row make epsilon
    # observable. These extra checks do not replace the five scored inputs.
    result = x[:6, :min(17, x.shape[-1])].contiguous().clone()
    result[0].zero_()
    result[1].mul_(1e-4)
    return result.reshape(2, 3, -1)


@contextmanager
def checked_modules(harness):
    load_original = harness.load_module
    patched = []

    def load():
        module = load_original()
        original = getattr(module, SYMBOL)
        patched.append((module, original))

        def checked(x, eps=1e-6):
            pristine = x.clone()
            expected = reference(harness, pristine, eps)
            diagnostic = diagnostic_input(pristine)
            saved = diagnostic.clone()
            diagnostic_eps = 1e-3
            diagnostic_expected = reference(harness, saved, diagnostic_eps)
            check_output(original(diagnostic, eps=diagnostic_eps), diagnostic_expected)
            unchanged(diagnostic, saved)
            output = original(x, eps=eps)
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
    module, args, kwargs = (c[name] for name in ('mod', 'args', 'kwargs'))
    g, = args
    eps = kwargs.get('eps', 1e-6)
    pristine = g.clone()
    expected = reference(harness, pristine, eps)
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
            raise AssertionError('Benchmark did not invoke the L2 candidate')
        return captured

    setattr(module, SYMBOL, collect)
    try:
        timed = harness._TimedRun()
        ms, metadata = benchmark(measured, timed_run=timed, **options)
        unchanged(g, pristine)
        check_output(timed.outputs, expected)
        g.mul_(-1).add_(0.25)
        replay_pristine = g.clone()
        replay_expected = reference(harness, replay_pristine, eps)
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
