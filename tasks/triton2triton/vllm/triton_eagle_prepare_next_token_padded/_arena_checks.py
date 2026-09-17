"""Check both padded EAGLE outputs using pristine inputs and captured replay."""
from contextlib import contextmanager
import inspect

SYMBOL = 'eagle_prepare_next_token_padded'
INPUT_NAMES = ('sampled', 'dm', 'backup')
SCALAR_NAMES = ('vs',)


def snapshots(inputs):
    return tuple(value.clone() for value in inputs)


def unchanged(inputs, pristine):
    import torch
    if any(not torch.equal(value, saved) for value, saved in zip(inputs, pristine)):
        raise AssertionError('Padded EAGLE modified read-only inputs')


def reference(harness, inputs, scalars):
    return tuple(value.to(inputs[0].device) for value in
                 harness.reference(*(v.cpu() for v in inputs), *scalars))


def check_outputs(outputs, expected):
    import torch
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 2:
        raise AssertionError('Padded EAGLE must return both outputs')
    for actual, ref in zip(outputs, expected):
        if not isinstance(actual, torch.Tensor) or (actual.shape != ref.shape or
                actual.dtype != ref.dtype or actual.device != ref.device):
            raise AssertionError('Padded EAGLE output shape/dtype/device is invalid')
        if not torch.equal(actual, ref):
            raise AssertionError('Padded EAGLE output differs from the independent reference')


def perturb(inputs, scalars):
    import torch
    sampled, discard, backup = inputs
    vocab_size, = scalars
    sampled.copy_(torch.where(sampled == -1, sampled, (sampled + 1).remainder(vocab_size)))
    sampled.copy_(sampled.flip(1))
    discard.logical_not_()
    backup.copy_((backup + 1).remainder(vocab_size))
    sampled[0] = -1  # Retain explicit all-rejected fallback coverage.


@contextmanager
def checked_modules(harness):
    load_original = harness.load_module
    patched = []

    def load():
        module = load_original()
        original = getattr(module, SYMBOL)
        patched.append((module, original))

        def checked(*args):
            inputs, scalars = args[:len(INPUT_NAMES)], args[len(INPUT_NAMES):]
            pristine = snapshots(inputs)
            expected = reference(harness, pristine, scalars)
            outputs = original(*args)
            unchanged(inputs, pristine)
            check_outputs(outputs, expected)
            return outputs

        setattr(module, SYMBOL, checked)
        return module

    harness.load_module = load
    try:
        yield
    finally:
        harness.load_module = load_original
        for module, original in reversed(patched):
            setattr(module, SYMBOL, original)


def checked_benchmark(harness, benchmark, fn, **kwargs):
    c = inspect.getclosurevars(fn).nonlocals
    module = c['mod']
    inputs = tuple(c[name] for name in INPUT_NAMES)
    scalars = tuple(c[name] for name in SCALAR_NAMES)
    pristine = snapshots(inputs)
    expected = reference(harness, pristine, scalars)
    original = getattr(module, SYMBOL)
    captured = None

    def collect(*args):
        nonlocal captured
        captured = original(*args)
        return captured

    def measured():
        nonlocal captured
        captured = None
        fn()  # Preserve the original public invocation/allocation boundary.
        if captured is None:
            raise AssertionError('Benchmark did not invoke the padded EAGLE candidate')
        return captured

    setattr(module, SYMBOL, collect)
    try:
        timed = harness._TimedRun()
        ms, metadata = benchmark(measured, timed_run=timed, **kwargs)
        unchanged(inputs, pristine)
        check_outputs(timed.outputs, expected)
        perturb(inputs, scalars)
        replay_pristine = snapshots(inputs)
        replay_expected = reference(harness, replay_pristine, scalars)
        for output in timed.outputs:
            output.fill_(-2)
        replayed = timed.rerun()
        unchanged(inputs, replay_pristine)
        check_outputs(replayed, replay_expected)
        return ms, {**metadata, 'timed_output_checked': True,
                    'perturbed_input_replay_checked': True, 'source_buffers_unchanged': True}
    finally:
        setattr(module, SYMBOL, original)
        for value, saved in zip(inputs, pristine):
            value.copy_(saved)


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
