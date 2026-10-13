# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Observe the measured graph, including integer/tuple outputs, outside timing."""
import torch


def tensors(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        return [tensor for item in value for tensor in tensors(item)]
    raise ValueError('Timed native invocation must expose its actual output tensors')


def measure(benchmark, invoke, inputs, check, change_inputs, **policy):
    from _aka_benchmark import TimedRun
    pristine = [value.detach().clone() for value in inputs]
    observed = TimedRun()

    def validate(output, checker=check, input_snapshot=pristine):
        torch.cuda.synchronize()
        checker(output)
        for value, original in zip(inputs, input_snapshot):
            torch.testing.assert_close(value, original, rtol=0, atol=0)
        for output_tensor in tensors(output):
            if any(output_tensor.untyped_storage().data_ptr() == value.untyped_storage().data_ptr()
                   for value in inputs):
                raise ValueError('Native output aliases a caller-owned input')

    checked_samples = 0

    def check_sample(output):
        nonlocal checked_samples
        # The sample observer only reads the completed output buffers.
        check(output)
        checked_samples += 1

    observed.after_sample = check_sample
    elapsed, metadata = benchmark(invoke, timed_run=observed, max_graph_repeats=1, **policy)
    if checked_samples != metadata.get('benchmark_samples') or checked_samples != policy.get('repetition', 100):
        raise ValueError('Reported benchmark samples were not all validated')

    validate(observed.outputs)
    with torch.no_grad():
        for value in tensors(observed.outputs):
            # These tasks return floating distances and/or signed integer indices.
            # NaN is not representable in an integer output; -1 is an invalid index.
            value.fill_(float('nan') if value.is_floating_point() else -1)
    validate(observed.rerun())
    changed_count = 0
    try:
        for change_input in change_inputs:
            changed_check = change_input()
            if not callable(changed_check):
                raise ValueError('Changed-input reference check is missing')
            changed_inputs = [value.detach().clone() for value in inputs]
            with torch.no_grad():
                for value in tensors(observed.outputs):
                    if value.is_floating_point():
                        value.fill_(float('nan'))
                    else:
                        value.bitwise_not_()
            validate(observed.rerun(), changed_check, changed_inputs)
            changed_count += 1
            with torch.no_grad():
                for value, original in zip(inputs, pristine):
                    value.copy_(original)
        if not changed_count:
            raise ValueError('No changed-input replay was validated')
    finally:
        with torch.no_grad():
            for value, original in zip(inputs, pristine):
                value.copy_(original)
    return elapsed, {**metadata, 'replay_validation_valid': True,
                     'timed_output_checked': True,
                     'changed_input_replay_valid': True,
                     'changed_input_replay_count': changed_count,
                     'validated_sample_count': checked_samples,
                     'replay_validation': 'full_reference_output_and_unchanged_inputs'}
