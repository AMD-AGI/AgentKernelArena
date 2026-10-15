"""Read-only input, output and captured-replay checks over the RMS harness.

All reference work and input perturbation happen outside device timing. The
original case stream, random seeds, warmups, samples and tolerance stay intact.
"""
from contextlib import contextmanager
import inspect


def check_output(output, x, weight, eps, reference, *, expected=None):
    import torch

    if not isinstance(output, torch.Tensor):
        raise AssertionError('RMSNorm must return a tensor')
    if output.shape != x.shape or output.dtype != x.dtype or output.device != x.device:
        raise AssertionError('RMSNorm output shape, dtype and device must match the input')
    if not torch.isfinite(output).all():
        raise AssertionError('RMSNorm output contains nonfinite values')
    if expected is None:
        expected = reference(x, weight, eps)
    if not torch.allclose(output, expected, atol=1e-2, rtol=1e-2):
        raise AssertionError('RMSNorm output differs from the original reference tolerance')


def snapshot_inputs(x, weight):
    return x.clone(), weight.clone()


def check_inputs(x, weight, saved):
    import torch

    for name, value, original in zip(('input', 'weight'), (x, weight), saved):
        if (value.shape != original.shape or value.dtype != original.dtype
                or value.device != original.device or not torch.equal(value, original)):
            raise AssertionError(f'RMSNorm {name} is read-only; candidate modified it')


def restore_inputs(x, weight, saved):
    x.copy_(saved[0])
    weight.copy_(saved[1])


@contextmanager
def checked_modules(harness):
    original_load = harness.load_module
    patched = []

    def load():
        module = original_load()
        original = module.rms_norm

        def checked(x, weight, eps=1e-6):
            saved = snapshot_inputs(x, weight)
            try:
                expected = harness.reference_rms_norm(*saved, eps)
                output = original(x, weight, eps=eps)
                check_inputs(x, weight, saved)
                check_output(output, *saved, eps, harness.reference_rms_norm, expected=expected)
                return output
            finally:
                restore_inputs(x, weight, saved)

        patched.append((module, original))
        module.rms_norm = checked
        return module

    harness.load_module = load
    try:
        yield
    finally:
        harness.load_module = original_load
        for module, original in reversed(patched):
            module.rms_norm = original


def checked_benchmark(harness, benchmark, fn, **kwargs):
    # These are the original protected _bench_fn's explicit closure inputs.
    inputs = inspect.getclosurevars(fn).nonlocals
    module, x, weight, eps = (inputs[key] for key in ('mod', 'x', 'weight', 'eps'))
    original = module.rms_norm
    saved = snapshot_inputs(x, weight)
    observed = []

    def observe(*args, **call_kwargs):
        output = original(*args, **call_kwargs)
        observed[:] = [output]
        return output

    def measured():
        fn()
        return observed[0]

    module.rms_norm = observe
    try:
        # Snapshot and oracle work stay outside the timed callback/capture.
        expected = harness.reference_rms_norm(*saved, eps)
        timed = harness._TimedRun()
        ms, metadata = benchmark(measured, timed_run=timed, **kwargs)
        check_inputs(x, weight, saved)
        check_output(timed.outputs, *saved, eps, harness.reference_rms_norm, expected=expected)
        # Change the graph's actual inputs and poison its output buffer. A
        # cached answer, no-op replay or merely changing wrong answer must fail.
        x.neg_()
        weight.mul_(0.5)
        replay_inputs = snapshot_inputs(x, weight)
        replay_expected = harness.reference_rms_norm(*replay_inputs, eps)
        timed.outputs.fill_(float('nan'))
        replayed = timed.rerun()
        check_inputs(x, weight, replay_inputs)
        check_output(replayed, *replay_inputs, eps, harness.reference_rms_norm,
                     expected=replay_expected)
        return ms, {**metadata, 'timed_output_checked': True,
                    'perturbed_input_replay_checked': True, 'readonly_inputs_checked': True}
    finally:
        module.rms_norm = original
        restore_inputs(x, weight, saved)


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
