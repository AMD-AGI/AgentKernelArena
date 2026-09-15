"""Validate NaN counting with pristine inputs and the actual measured replay."""
from contextlib import contextmanager
import inspect

SYMBOL = 'get_num_nans'


def unchanged(x, pristine):
    import torch
    # NaN != NaN: compare bytes so unchanged NaNs pass, while all mutations
    # (including NaN payloads and signed zeros) remain observable.
    if not torch.equal(x.contiguous().view(torch.uint8), pristine.contiguous().view(torch.uint8)):
        raise AssertionError('NaN counter modified read-only logits')


def reference(harness, x):
    return harness.reference_num_nans(x.cpu()).to(x.device)


def check_output(output, expected):
    import torch
    if not isinstance(output, torch.Tensor) or (output.shape != expected.shape or
            output.dtype != expected.dtype or output.device != expected.device):
        raise AssertionError('NaN counts must have the declared int32 shape and input device')
    if not torch.equal(output, expected):
        raise AssertionError('NaN counts differ from pristine-input reference')


@contextmanager
def checked_modules(harness):
    load_original = harness.load_module
    patched = []

    def load():
        module = load_original()
        original = getattr(module, SYMBOL)
        patched.append((module, original))

        def checked(logits):
            pristine = logits.clone()
            expected = reference(harness, pristine)
            nan, inf = float('nan'), float('inf')
            diagnostic = logits.new_tensor([[nan, nan, 1, -inf, inf, 0, -0.],
                                            [0, 1, 2, -1, inf, -inf, 0],
                                            [nan]*7])
            saved = diagnostic.clone()
            wanted = reference(harness, saved)
            check_output(original(diagnostic), wanted)
            unchanged(diagnostic, saved)
            output = original(logits)
            unchanged(logits, pristine)
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
    module, logits = (c[name] for name in ('mod', 'logits'))
    pristine = logits.clone()
    expected = reference(harness, pristine)
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
            raise AssertionError('Benchmark did not invoke NaN counting')
        return captured

    setattr(module, SYMBOL, collect)
    try:
        timed = harness._TimedRun()
        ms, metadata = benchmark(measured, timed_run=timed, **options)
        unchanged(logits, pristine)
        check_output(timed.outputs, expected)
        # Original timed tensors stay finite. Only the untimed replay gets
        # NaNs, distinguishing real counting from cached/all-zero output.
        logits[:, :2] = float('nan')
        logits[0, :] = float('nan')
        logits[1, 2] = float('nan')
        replay_pristine = logits.clone()
        replay_expected = reference(harness, replay_pristine)
        timed.outputs.fill_(-1)
        replayed = timed.rerun()
        unchanged(logits, replay_pristine)
        check_output(replayed, replay_expected)
        return ms, {**metadata, 'timed_output_checked': True,
                    'perturbed_input_replay_checked': True, 'source_buffers_unchanged': True}
    finally:
        setattr(module, SYMBOL, original)
        logits.copy_(pristine)


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
