# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Observe the measured graph, including integer/tuple outputs, outside timing."""
import torch


def tensors(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        return [tensor for item in value for tensor in tensors(item)]
    raise ValueError('Timed native invocation must expose its actual output tensors')


def measure(benchmark, invoke, inputs, check, **policy):
    from _aka_benchmark import TimedRun
    pristine = [value.detach().clone() for value in inputs]
    observed = TimedRun()
    elapsed, metadata = benchmark(invoke, timed_run=observed, **policy)

    def validate(output):
        torch.cuda.synchronize()
        check(output)
        for value, original in zip(inputs, pristine):
            torch.testing.assert_close(value, original, rtol=0, atol=0)
        for output_tensor in tensors(output):
            if any(output_tensor.untyped_storage().data_ptr() == value.untyped_storage().data_ptr()
                   for value in inputs):
                raise ValueError('Native output aliases a caller-owned input')

    validate(observed.outputs)
    with torch.no_grad():
        for value in tensors(observed.outputs):
            if value.is_floating_point():
                value.fill_(float('nan'))
            else:
                value.bitwise_not_()
    validate(observed.rerun())
    return elapsed, {**metadata, 'replay_validation_valid': True,
                     'replay_validation': 'full_reference_output_and_unchanged_inputs'}
