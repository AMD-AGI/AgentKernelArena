"""Keep the FP4 GEMM oracle independent of mutable candidate input pointers."""
from contextlib import contextmanager
import inspect


def snapshots(inputs):
    return tuple(value.clone() for value in inputs)


def unchanged(inputs, pristine):
    import torch
    if any(not torch.equal(value, saved) for value, saved in zip(inputs, pristine)):
        raise AssertionError('FP4 GEMM modified read-only inputs')


def check_output(harness, output, expected):
    import torch
    if not isinstance(output, torch.Tensor) or (output.shape != expected.shape or
            output.dtype != expected.dtype or output.device != expected.device):
        raise AssertionError('FP4 GEMM output shape/dtype/device is invalid')
    if not torch.isfinite(output).all():
        raise AssertionError('FP4 GEMM output must be finite')
    torch.testing.assert_close(output, expected, atol=harness.ATOL, rtol=harness.RTOL)


@contextmanager
def checked_correctness(harness):
    original = harness.gemm_a16wfp4

    def checked(x, w, scales, **kwargs):
        inputs = (x, w, scales)
        pristine = snapshots(inputs)
        expected = harness.run_torch_reference(*pristine, kwargs['dtype'])
        output = original(x, w, scales, **kwargs)
        unchanged(inputs, pristine)
        check_output(harness, output, expected)
        return output

    harness.gemm_a16wfp4 = checked
    try:
        yield
    finally:
        harness.gemm_a16wfp4 = original


def checked_benchmark(harness, benchmark, fn, **kwargs):
    from _aka_benchmark import TimedRun
    closure = inspect.getclosurevars(fn).nonlocals
    inputs = tuple(closure[key] for key in ('x', 'w_kernel', 'w_scales_kernel'))
    pristine = snapshots(inputs)
    expected = harness.run_torch_reference(*pristine, harness.DTYPE)
    try:
        timed = TimedRun()
        ms, metadata = benchmark(fn, timed_run=timed, **kwargs)
        unchanged(inputs, pristine)
        check_output(harness, timed.outputs, expected)
        x, w, scales = inputs
        x.mul_(-2)
        w.copy_(w.roll(1, 0))
        scales.copy_(scales.roll(1, 0))
        replay_pristine = snapshots(inputs)
        replay_expected = harness.run_torch_reference(*replay_pristine, harness.DTYPE)
        timed.outputs.fill_(float('nan'))
        replayed = timed.rerun()
        unchanged(inputs, replay_pristine)
        check_output(harness, replayed, replay_expected)
        metadata.update(timed_output_checked=True, perturbed_input_replay_checked=True,
                        source_buffers_unchanged=True)
        return ms, metadata
    finally:
        for value, saved in zip(inputs, pristine):
            value.copy_(saved)
