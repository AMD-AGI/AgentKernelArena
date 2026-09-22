"""Check packed-mask routing, untouched rows and the actual timed in-place output."""
from contextlib import contextmanager
import inspect


def snapshots(values):
    return tuple(value.clone() for value in values)


def unchanged(values, pristine):
    import torch
    if any(not torch.equal(value, saved) for value, saved in zip(values, pristine)):
        raise AssertionError('Grammar masking modified read-only inputs')


def reference(harness, inputs, vocab_size):
    return harness.reference_apply_grammar_bitmask(
        *(value.cpu() for value in inputs), vocab_size).to(inputs[0].device)


def check_output(output, expected):
    import torch
    if not isinstance(output, torch.Tensor) or (output.shape != expected.shape or
            output.dtype != expected.dtype or output.device != expected.device):
        raise AssertionError('Grammar masking output shape/dtype/device is invalid')
    masked = torch.isneginf(expected)
    if not torch.equal(torch.isneginf(output), masked):
        raise AssertionError('Grammar masking changed the required negative-infinity pattern')
    # Same original torch.allclose defaults, now covering unselected rows too.
    if not torch.allclose(output[~masked], expected[~masked], atol=1e-8, rtol=1e-5):
        raise AssertionError('Grammar masking changed unmasked values or unselected rows')


def perturb(inputs):
    logits, indices, bitmask = inputs
    logits.neg_()
    indices.copy_((indices.flip(0) + 1).remainder(logits.shape[0]))
    bitmask.bitwise_not_()  # Includes bit31, which the original positive RNG omits.


@contextmanager
def checked_modules(harness):
    original_load = harness.load_module
    modules = []

    def load():
        module = original_load()
        original = module.apply_grammar_bitmask
        modules.append((module, original))

        def checked(logits, indices, bitmask, vocab_size):
            pristine = snapshots((logits, indices, bitmask))
            expected = reference(harness, pristine, vocab_size)
            diagnostic = snapshots(pristine)
            perturb(diagnostic)
            diagnostic_pristine = snapshots(diagnostic)
            diagnostic_expected = reference(harness, diagnostic_pristine, vocab_size)
            original(*diagnostic, vocab_size)
            unchanged(diagnostic[1:], diagnostic_pristine[1:])
            check_output(diagnostic[0], diagnostic_expected)
            result = original(logits, indices, bitmask, vocab_size)
            unchanged((indices, bitmask), pristine[1:])
            check_output(logits, expected)
            return result

        module.apply_grammar_bitmask = checked
        return module

    harness.load_module = load
    try:
        yield
    finally:
        harness.load_module = original_load
        for module, original in reversed(modules):
            module.apply_grammar_bitmask = original


def checked_benchmark(harness, benchmark, fn, **kwargs):
    c = inspect.getclosurevars(fn).nonlocals
    working, indices, bitmask, vocab_size = (
        c[key] for key in ('logits_work', 'logits_indices', 'bitmask', 'vocab_size'))
    initial = inspect.getclosurevars(kwargs['prepare_fn']).nonlocals['logits']
    inputs = (initial, indices, bitmask)
    pristine = snapshots(inputs)
    working_saved = working.clone()
    expected = reference(harness, pristine, vocab_size)

    def measured():
        fn()
        return working

    try:
        timed = harness._TimedRun()
        ms, metadata = benchmark(measured, timed_run=timed, **kwargs)
        if timed.outputs is not working:
            raise AssertionError('Benchmark did not retain the actual grammar output buffer')
        unchanged(inputs, pristine)
        check_output(timed.outputs, expected)
        perturb(inputs)
        replay_pristine = snapshots(inputs)
        replay_expected = reference(harness, replay_pristine, vocab_size)
        working.fill_(float('nan'))
        replayed = timed.rerun()  # Same original prepare_fn restores working logits.
        unchanged(inputs, replay_pristine)
        if replayed is not working:
            raise AssertionError('Replay returned a different grammar output buffer')
        check_output(replayed, replay_expected)
        return ms, {**metadata, 'timed_output_checked': True,
                    'perturbed_input_replay_checked': True, 'signed_mask_mapping_checked': True,
                    'source_buffers_unchanged': True}
    finally:
        for value, saved in zip(inputs, pristine):
            value.copy_(saved)
        working.copy_(working_saved)


def install(harness):
    original_correctness = harness.run_correctness
    original_performance = harness.run_performance

    def correctness(*args, **kwargs):
        with checked_modules(harness):
            return original_correctness(*args, **kwargs)

    def performance():
        benchmark = harness._benchmark_cuda_graph_or_events
        harness._benchmark_cuda_graph_or_events = lambda fn, **kwargs: checked_benchmark(
            harness, benchmark, fn, **kwargs)
        try:
            return original_performance()
        finally:
            harness._benchmark_cuda_graph_or_events = benchmark

    harness.run_correctness = correctness
    harness.run_performance = performance
