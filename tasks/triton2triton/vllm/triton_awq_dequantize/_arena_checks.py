"""Check the actual AWQ dequantization graph using the original unpacking reference."""
import inspect


def check_output(output, qweight, scales, zeros, reference):
    import torch

    shape = (qweight.shape[0], qweight.shape[1] * 8)
    if not isinstance(output, torch.Tensor):
        raise AssertionError('AWQ dequantization must return a tensor')
    if output.shape != shape or output.dtype != scales.dtype or output.device != scales.device:
        raise AssertionError('AWQ output shape/dtype/device violates the contract')
    if not torch.isfinite(output).all():
        raise AssertionError('AWQ output contains nonfinite values')
    group_size = qweight.shape[0] // scales.shape[0]
    expected = reference(qweight, scales, zeros, group_size).to(scales.device)
    if not torch.allclose(output, expected, atol=1e-2, rtol=1e-2):
        raise AssertionError('AWQ output differs from the original reference tolerance')


def checked_benchmark(harness, benchmark, fn, **kwargs):
    inputs = inspect.getclosurevars(fn).nonlocals
    module, qweight, scales, zeros = (inputs[key] for key in ('mod', 'qweight', 'scales', 'zeros'))
    original = module.awq_dequantize_triton
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
        timed = harness._TimedRun()
        ms, metadata = benchmark(measured, timed_run=timed, **kwargs)
        check_output(timed.outputs, qweight, scales, zeros, harness.reference_awq_dequantize)
        # Include signed packed words/high nibbles in the replay control, while
        # preserving the original seed and values for the timed workload.
        qweight.bitwise_xor_(-1)
        zeros.bitwise_xor_(-1431655766)  # int32 representation of 0xaaaaaaaa
        scales.mul_(0.5)
        timed.outputs.fill_(float('nan'))
        check_output(timed.rerun(), qweight, scales, zeros, harness.reference_awq_dequantize)
        return ms, {**metadata, 'timed_output_checked': True,
                    'perturbed_input_replay_checked': True}
    finally:
        module.awq_dequantize_triton = original


def install(harness):
    original_performance = harness.run_performance

    def performance():
        benchmark = harness._benchmark_cuda_graph_or_events
        harness._benchmark_cuda_graph_or_events = lambda fn, **kwargs: checked_benchmark(
            harness, benchmark, fn, **kwargs)
        try:
            return original_performance()
        finally:
            harness._benchmark_cuda_graph_or_events = benchmark

    harness.run_performance = performance
