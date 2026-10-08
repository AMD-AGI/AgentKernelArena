# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Validate returned values and caller state from the actual timed invocation.

Observe either the captured graph or the last actual explicit Event sample.
All numerical checks run outside the reported samples. The canonical helper
owns capture, warmups, repetitions, stream ordering and device timing.
"""
import copy
import inspect

import torch


def unchanged_inputs(before, after):
    for expected, actual in zip(before, after):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        elif actual != expected:
            raise ValueError("operator changed a caller-owned input")


def unchanged_model_state(before, module):
    if not before:
        return
    after = module.state_dict()
    if before.keys() != after.keys():
        raise ValueError('operator changed model state entries')
    for name, expected in before.items():
        torch.testing.assert_close(after[name], expected, rtol=0, atol=0, equal_nan=True)


def check_result(actual, expected, inputs, output_contract, compare, rtol, atol):
    output_contract(expected, actual)
    if not compare(expected, actual, rtol=rtol, atol=atol):
        raise ValueError("Timed operator output disagrees with the protected reference")
    separate_output(actual, inputs)


def separate_output(actual, inputs):
    # The task reference returns an independent output: preserving values alone cannot establish its
    # caller-visible storage contract.
    for value in inputs:
        if isinstance(value, torch.Tensor) and actual.untyped_storage().data_ptr() == value.untyped_storage().data_ptr():
            raise ValueError("operator output aliases a caller-owned input")


def changed_input_reference(inputs, index, expected, reference, compare, rtol, atol):
    """Change one independent operand while preserving the other operands."""
    value = inputs[index]
    if not isinstance(value, torch.Tensor) or not value.is_floating_point() or not value.numel():
        raise ValueError(f'Operand {index} has no floating data to replay')
    for flat_index in dict.fromkeys((0, value.numel() // 2, value.numel() - 1)):
        remainder = flat_index
        coordinates = []
        for size in reversed(value.shape):
            remainder, coordinate = divmod(remainder, size)
            coordinates.append(coordinate)
        coordinates = tuple(reversed(coordinates))
        original = value[coordinates].detach().clone()
        for trial in (original + 2, original - 2, torch.zeros_like(original)):
            with torch.no_grad():
                value[coordinates].copy_(trial)
                changed = reference(*copy.deepcopy(inputs))
            if not compare(expected, changed, rtol=rtol, atol=atol):
                return changed, copy.deepcopy(inputs)
            with torch.no_grad():
                value[coordinates].copy_(original)
    raise ValueError(f'Operand {index} did not produce a distinguishable reference output')


def install(perf, output_contract):
    from _aka_benchmark import TimedRun
    signature = inspect.signature(perf.cal_kernel_perf).parameters
    rtol, atol = signature['rtol'].default, signature['atol'].default

    def latency(module, inputs, hip_fn=None, n_iter=100, n_warmup=10,
                use_cuda_graph=True, fallback_reason=None, prepare_fn=None):
        pristine = copy.deepcopy(inputs)
        state = {name: value.detach().clone() for name, value in module.state_dict().items()} if hasattr(module, "state_dict") else {}
        try:
            with torch.no_grad():
                # The functional module's default is the protected PyTorch operator,
                # never the supplied HIP function. The PyTorch baseline is checked
                # eagerly here and against the functional reference by correctness.
                expected = module(*copy.deepcopy(inputs))
            observed = TimedRun()
            checked_samples = 0

            def check_sample(output):
                nonlocal checked_samples
                # The canonical observer may only read completed sample outputs.
                output_contract(expected, output)
                if not perf._compare_results(expected, output, rtol=rtol, atol=atol):
                    raise ValueError('Timed operator output disagrees with the protected reference')
                checked_samples += 1

            observed.after_sample = check_sample
            invoke = (lambda: module(*inputs)) if hip_fn is None else (lambda: module(*inputs, fn=hip_fn))
            elapsed, metadata = perf.benchmark_cuda_graph_or_events(
                invoke, warmup=n_warmup, repetition=n_iter,
                use_cuda_graph=use_cuda_graph, fallback_reason=fallback_reason,
                prepare_fn=prepare_fn, timed_run=observed, max_graph_repeats=1)
            kind = metadata.get('benchmark_timed_run_kind')
            expected_method = {'captured_graph': 'cuda_graph', 'eager_callable': 'cuda_event_fallback'}.get(kind)
            if expected_method is None or metadata.get('benchmark_method') != expected_method:
                raise ValueError('Benchmark did not identify its actual observed invocation')
            if checked_samples != metadata.get('benchmark_samples') or checked_samples != n_iter:
                raise ValueError('Reported benchmark samples were not all validated')
            torch.cuda.synchronize()
            check_result(observed.outputs, expected, inputs, output_contract, perf._compare_results, rtol, atol)
            unchanged_inputs(pristine, inputs)
            unchanged_model_state(state, module)
            # Check the actual measured output first. Then poison it and re-invoke
            # the measured unit: the same graph, or the same eager callable for
            # explicit Event timing (which can allocate a new output).
            with torch.no_grad():
                observed.outputs.fill_(float('nan'))
            actual = observed.rerun()
            check_result(actual, expected, inputs, output_contract, perf._compare_results, rtol, atol)
            unchanged_inputs(pristine, inputs)
            unchanged_model_state(state, module)
            changed_count = 0
            for index, value in enumerate(inputs):
                if not isinstance(value, torch.Tensor) or not value.is_floating_point() or not value.numel():
                    continue
                changed_expected, changed_inputs = changed_input_reference(
                    inputs, index, expected, module, perf._compare_results, rtol, atol)
                with torch.no_grad():
                    observed.outputs.fill_(float('nan'))
                changed_actual = observed.rerun()
                check_result(changed_actual, changed_expected, inputs, output_contract,
                             perf._compare_results, rtol, atol)
                unchanged_inputs(changed_inputs, inputs)
                unchanged_model_state(state, module)
                changed_count += 1
                with torch.no_grad():
                    for original, current in zip(pristine, inputs):
                        if isinstance(current, torch.Tensor):
                            current.copy_(original)
            if not changed_count:
                raise ValueError('No changed-input replay was validated')
            # The original scored generator is nonnegative. Check the same
            # operator and shape on valid negative inputs outside measurement;
            # a candidate that clamps x below -3 must not pass the positive cases.
            with torch.no_grad():
                inputs[0].copy_(-4 - pristine[0])
                negative_inputs = copy.deepcopy(inputs)
                negative_expected = module(*copy.deepcopy(inputs))
            if perf._compare_results(expected, negative_expected, rtol=rtol, atol=atol):
                raise ValueError('Negative input did not distinguish the reference output')
            with torch.no_grad():
                observed.outputs.fill_(float('nan'))
            negative_actual = observed.rerun()
            check_result(negative_actual, negative_expected, inputs, output_contract,
                         perf._compare_results, rtol, atol)
            unchanged_inputs(negative_inputs, inputs)
            unchanged_model_state(state, module)
            changed_count += 1
            with torch.no_grad():
                inputs[0].copy_(pristine[0])
            return elapsed, {**metadata, 'replay_validation_valid': True,
                             'validated_sample_count': checked_samples,
                             'timed_output_checked': True,
                             'changed_input_replay_valid': True,
                             'changed_input_replay_count': changed_count,
                             'negative_domain_replay_valid': True,
                             'input_state_restored': True,
                             'model_state_validation_valid': True,
                             'model_state_tensor_count': len(state),
                             'validated_invocation_kind': kind,
                             'replay_validation': 'full_reference_output_and_unchanged_inputs'}
        finally:
            with torch.no_grad():
                for original, value in zip(pristine, inputs):
                    if isinstance(value, torch.Tensor):
                        value.copy_(original)
                if state:
                    current = module.state_dict()
                    for name, original in state.items():
                        current[name].copy_(original)
            torch.cuda.synchronize()


    perf.cal_hip_latency = latency
    if hasattr(perf, 'cal_modu_latency'):
        perf.cal_modu_latency = lambda module, inputs, **kwargs: latency(module, inputs, **kwargs)
