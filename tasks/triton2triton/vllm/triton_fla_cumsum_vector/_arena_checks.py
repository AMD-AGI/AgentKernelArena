"""Check the original cumsum gate, reverse/tail diagnostics and measured replay."""
from contextlib import contextmanager
import inspect

SYMBOL = 'chunk_local_cumsum_vector'
TOL = 0.001


def unchanged(g, pristine):
    import torch
    if not torch.equal(g, pristine):
        raise AssertionError('Chunk cumsum modified its read-only input')


def reference(harness, g, chunk_size, reverse):
    return harness.reference(g.cpu(), chunk_size, reverse).to(g.device)


def check_output(output, expected):
    import torch
    if not isinstance(output, torch.Tensor) or (output.shape != expected.shape or
            output.dtype != expected.dtype or output.device != expected.device):
        raise AssertionError('Chunk cumsum output shape/dtype/device is invalid')
    if not torch.isfinite(output).all():
        raise AssertionError('Chunk cumsum output must be finite')
    torch.testing.assert_close(output, expected, atol=TOL, rtol=TOL)


def diagnostic_input(g):
    # A small contiguous slice exercises a partial final chunk and a different
    # batch/head count without adding a scored case or changing timed tensors.
    result = g[:1, :max(1, g.shape[1]-3), :min(2, g.shape[2])]
    if g.ndim == 4:
        result = result[..., :min(17, g.shape[-1])]
    return result.contiguous().clone()


@contextmanager
def checked_modules(harness):
    load_original = harness.load_module
    patched = []

    def load():
        module = load_original()
        original = getattr(module, SYMBOL)
        patched.append((module, original))

        def checked(g, chunk_size, reverse=False):
            pristine = g.clone()
            expected = reference(harness, pristine, chunk_size, reverse)
            diagnostic = diagnostic_input(pristine)
            saved = diagnostic.clone()
            diagnostic_chunk = max(1, chunk_size // 2)
            diagnostic_expected = reference(harness, saved, diagnostic_chunk, not reverse)
            check_output(original(diagnostic, diagnostic_chunk, reverse=not reverse), diagnostic_expected)
            unchanged(diagnostic, saved)
            output = original(g, chunk_size, reverse=reverse)
            unchanged(g, pristine)
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
    g, chunk_size = args
    reverse = kwargs.get('reverse', False)
    pristine = g.clone()
    expected = reference(harness, pristine, chunk_size, reverse)
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
            raise AssertionError('Benchmark did not invoke the cumsum candidate')
        return captured

    setattr(module, SYMBOL, collect)
    try:
        timed = harness._TimedRun()
        ms, metadata = benchmark(measured, timed_run=timed, **options)
        unchanged(g, pristine)
        check_output(timed.outputs, expected)
        g.mul_(-1).add_(0.25)
        replay_pristine = g.clone()
        replay_expected = reference(harness, replay_pristine, chunk_size, reverse)
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
