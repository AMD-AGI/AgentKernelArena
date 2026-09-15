"""Validate the exact GEMM timing graph outside the measured interval."""
import inspect


def check_finite(output):
    import torch

    if not isinstance(output, torch.Tensor) or not torch.isfinite(output).all():
        raise AssertionError('GEMM output must be a finite tensor')


def checked_correctness(harness, indices):
    original = harness.gemm_a16w16

    def checked(*args, **kwargs):
        output = original(*args, **kwargs)
        check_finite(output)
        return output

    harness.gemm_a16w16 = checked
    try:
        return harness.run_correctness(indices)
    finally:
        harness.gemm_a16w16 = original


def check_output(output, x, w, bias):
    import torch
    import torch.nn.functional as F

    check_finite(output)
    # The original task reference and tolerance also enforce shape/dtype/device.
    torch.testing.assert_close(output, F.linear(x, w, bias=bias), atol=1e-1, rtol=1e-1)


def checked_benchmark(benchmark, fn, **kwargs):
    from _aka_benchmark import TimedRun

    inputs = inspect.getclosurevars(fn).nonlocals
    x, w, bias = (inputs[key] for key in ('x', 'w', 'bias'))
    timed = TimedRun()
    ms, metadata = benchmark(fn, timed_run=timed, **kwargs)
    check_output(timed.outputs, x, w, bias)
    # Perturb only after timing. The graph must overwrite its poisoned output
    # with the correct result for these same input buffers' new contents.
    x.neg_()
    bias.mul_(0.5)
    timed.outputs.fill_(float('nan'))
    check_output(timed.rerun(), x, w, bias)
    metadata.update(timed_output_checked=True, perturbed_input_replay_checked=True)
    return ms, metadata
