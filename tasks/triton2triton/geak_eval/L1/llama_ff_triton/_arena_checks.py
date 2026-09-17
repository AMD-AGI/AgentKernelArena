"""Check the measured fused feed-forward invocation with its original oracle."""
import inspect


def check_output(output, x, w1, w3, rms_w, reference):
    import torch

    if not isinstance(output, torch.Tensor) or not torch.isfinite(output).all():
        raise AssertionError('Feed-forward output must be a finite tensor')
    torch.testing.assert_close(output, reference(x, w1, w3, rms_w), atol=0.25, rtol=0.15)


def checked_benchmark(harness, benchmark, fn, **kwargs):
    from _aka_benchmark import TimedRun

    inputs = inspect.getclosurevars(fn).nonlocals
    # The original harness times the candidate followed by a diagnostic peer.
    # The peer is never substituted for the Arena baseline or candidate result.
    if 'ref_fn' in inputs:
        return benchmark(fn, **kwargs)
    x, w1, w3, rms_w = (inputs[key] for key in ('x', 'w1', 'w3', 'rms_w'))
    timed = TimedRun()
    ms, metadata = benchmark(fn, timed_run=timed, **kwargs)
    check_output(timed.outputs, x, w1, w3, rms_w, harness.reference_ff)
    original_x, original_rms_w = x.clone(), rms_w.clone()
    try:
        # These buffers are read directly by the graph. Leave the original
        # weight-preparation cache policy and measured invocation intact.
        x.neg_()
        rms_w.mul_(0.5)
        timed.outputs.fill_(float('nan'))
        check_output(timed.rerun(), x, w1, w3, rms_w, harness.reference_ff)
    finally:
        # The original harness next times its diagnostic reference on these
        # same tensors; it must receive the original seeded input values.
        x.copy_(original_x)
        rms_w.copy_(original_rms_w)
    metadata.update(timed_output_checked=True, perturbed_input_replay_checked=True)
    return ms, metadata
