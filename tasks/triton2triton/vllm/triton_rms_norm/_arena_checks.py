"""Additional output and captured-replay checks over the original RMS harness.

All reference work and input perturbation happen outside device timing. The
original case stream, random seeds, warmups, samples and tolerance stay intact.
"""
from contextlib import contextmanager
import inspect


def check_output(output, x, weight, eps, reference):
    import torch

    if not isinstance(output, torch.Tensor):
        raise AssertionError('RMSNorm must return a tensor')
    if output.shape != x.shape or output.dtype != x.dtype or output.device != x.device:
        raise AssertionError('RMSNorm output shape, dtype and device must match the input')
    if not torch.isfinite(output).all():
        raise AssertionError('RMSNorm output contains nonfinite values')
    expected = reference(x, weight, eps)
    if not torch.allclose(output, expected, atol=1e-2, rtol=1e-2):
        raise AssertionError('RMSNorm output differs from the original reference tolerance')


@contextmanager
def checked_modules(harness):
    original_load = harness.load_module

    def load():
        module = original_load()
        original = module.rms_norm

        def checked(x, weight, eps=1e-6):
            output = original(x, weight, eps=eps)
            check_output(output, x, weight, eps, harness.reference_rms_norm)
            return output

        module.rms_norm = checked
        return module

    harness.load_module = load
    try:
        yield
    finally:
        harness.load_module = original_load


def checked_benchmark(harness, benchmark, fn, **kwargs):
    # These are the original protected _bench_fn's explicit closure inputs.
    inputs = inspect.getclosurevars(fn).nonlocals
    module, x, weight, eps = (inputs[key] for key in ('mod', 'x', 'weight', 'eps'))
    original = module.rms_norm
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
        timed = harness._TimedRun()
        ms, metadata = benchmark(measured, timed_run=timed, **kwargs)
        check_output(timed.outputs, x, weight, eps, harness.reference_rms_norm)
        # Change the graph's actual inputs and poison its output buffer. A
        # cached answer, no-op replay or merely changing wrong answer must fail.
        x.neg_()
        weight.mul_(0.5)
        timed.outputs.fill_(float('nan'))
        replayed = timed.rerun()
        check_output(replayed, x, weight, eps, harness.reference_rms_norm)
        return ms, {**metadata, 'timed_output_checked': True,
                    'perturbed_input_replay_checked': True}
    finally:
        module.rms_norm = original


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
