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
    grad_state = [(value.grad, None if value.grad is None else value.grad.detach().clone())
                  for value in inputs]
    try:
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
                # These tasks return floating distances and/or signed integer indices.
                # NaN is not representable in an integer output; -1 is an invalid index.
                value.fill_(float('nan') if value.is_floating_point() else -1)
        validate(observed.rerun())
        return elapsed, {**metadata, 'replay_validation_valid': True,
                         'replay_validation': 'full_reference_output_and_unchanged_inputs'}
    finally:
        with torch.no_grad():
            for value, original, (grad_buffer, grad_values) in zip(inputs, pristine, grad_state):
                value.copy_(original)
                value.grad = grad_buffer
                if grad_buffer is not None:
                    grad_buffer.copy_(grad_values)
        torch.cuda.synchronize()
