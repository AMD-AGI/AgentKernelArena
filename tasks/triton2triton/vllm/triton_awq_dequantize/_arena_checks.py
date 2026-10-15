"""Pristine-input, read-only and actual replay checks outside device timing."""
from contextlib import contextmanager
import inspect


def check_output(output, qweight, scales, zeros, reference, *, expected=None):
    import torch

    shape = (qweight.shape[0], qweight.shape[1] * 8)
    if not isinstance(output, torch.Tensor):
        raise AssertionError('AWQ dequantization must return a tensor')
    if output.shape != shape or output.dtype != scales.dtype or output.device != scales.device:
        raise AssertionError('AWQ output shape/dtype/device violates the contract')
    if not torch.isfinite(output).all():
        raise AssertionError('AWQ output contains nonfinite values')
    if expected is None:
        expected = reference_output(qweight, scales, zeros, reference)
    if not torch.allclose(output, expected, atol=1e-2, rtol=1e-2):
        raise AssertionError('AWQ output differs from the original reference tolerance')


def reference_output(qweight, scales, zeros, reference):
    group_size = qweight.shape[0] // scales.shape[0]
    return reference(qweight, scales, zeros, group_size).to(scales.device)


def snapshot_inputs(qweight, scales, zeros):
    return tuple(value.detach().clone() for value in (qweight, scales, zeros))


def check_inputs(qweight, scales, zeros, saved):
    import torch

    for name, value, initial in zip(('qweight', 'scales', 'zeros'), (qweight, scales, zeros), saved):
        if (value.shape != initial.shape or value.dtype != initial.dtype
                or value.device != initial.device or value.stride() != initial.stride()
                or not torch.equal(value.contiguous().reshape(-1).view(torch.uint8),
                                   initial.contiguous().reshape(-1).view(torch.uint8))):
            raise AssertionError(f'AWQ {name} is read-only; candidate modified it')


def restore_inputs(qweight, scales, zeros, saved):
    import torch

    with torch.no_grad():
        for value, initial in zip((qweight, scales, zeros), saved):
            value.copy_(initial)


@contextmanager
def checked_modules(harness):
    original_load = harness.load_module
    patched = []

    def load():
        module = original_load()
        original = module.awq_dequantize_triton

        def checked(qweight, scales, zeros, *args, **kwargs):
            saved = snapshot_inputs(qweight, scales, zeros)
            try:
                expected = reference_output(*saved, harness.reference_awq_dequantize)
                output = original(qweight, scales, zeros, *args, **kwargs)
                check_inputs(qweight, scales, zeros, saved)
                check_output(output, *saved, harness.reference_awq_dequantize, expected=expected)
                return output
            finally:
                restore_inputs(qweight, scales, zeros, saved)

        patched.append((module, original))
        module.awq_dequantize_triton = checked
        return module

    harness.load_module = load
    try:
        yield
    finally:
        harness.load_module = original_load
        for module, original in reversed(patched):
            module.awq_dequantize_triton = original


def checked_benchmark(harness, benchmark, fn, **kwargs):
    inputs = inspect.getclosurevars(fn).nonlocals
    module, qweight, scales, zeros = (inputs[key] for key in ('mod', 'qweight', 'scales', 'zeros'))
    original = module.awq_dequantize_triton
    saved = snapshot_inputs(qweight, scales, zeros)
    observed = []

    def observe(*args, **call_kwargs):
        output = original(*args, **call_kwargs)
        observed[:] = [output]
        return output

    def measured():
        fn()
        return observed[0]

    module.awq_dequantize_triton = observe
    try:
        expected = reference_output(*saved, harness.reference_awq_dequantize)
        timed = harness._TimedRun()
        ms, metadata = benchmark(measured, timed_run=timed, **kwargs)
        check_inputs(qweight, scales, zeros, saved)
        check_output(timed.outputs, *saved, harness.reference_awq_dequantize, expected=expected)
        # Include signed packed words/high nibbles in the replay control, while
        # preserving the original seed and values for the timed workload.
        qweight.bitwise_xor_(-1)
        zeros.bitwise_xor_(-1431655766)  # int32 representation of 0xaaaaaaaa
        scales.mul_(0.5)
        replay_inputs = snapshot_inputs(qweight, scales, zeros)
        replay_expected = reference_output(*replay_inputs, harness.reference_awq_dequantize)
        timed.outputs.fill_(float('nan'))
        replayed = timed.rerun()
        check_inputs(qweight, scales, zeros, replay_inputs)
        check_output(replayed, *replay_inputs, harness.reference_awq_dequantize,
                     expected=replay_expected)
        return ms, {**metadata, 'timed_output_checked': True,
                    'perturbed_input_replay_checked': True, 'readonly_inputs_checked': True}
    finally:
        module.awq_dequantize_triton = original
        restore_inputs(qweight, scales, zeros, saved)


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
