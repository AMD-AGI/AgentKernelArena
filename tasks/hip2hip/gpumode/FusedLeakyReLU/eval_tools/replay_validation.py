# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Validate the operator's returned values and caller-visible state on the timed graph.

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


def install(perf, output_contract):
    from _aka_benchmark import TimedRun
    signature = inspect.signature(perf.cal_kernel_perf).parameters
    rtol, atol = signature['rtol'].default, signature['atol'].default

    def latency(module, inputs, hip_fn=None, n_iter=100, n_warmup=10,
                use_cuda_graph=True, fallback_reason=None, prepare_fn=None):
        pristine = copy.deepcopy(inputs)
        with torch.no_grad():
            # The functional module's default is the protected PyTorch operator,
            # never the supplied HIP function. The PyTorch baseline is checked
            # eagerly here and against the functional reference by correctness.
            expected = module(*copy.deepcopy(inputs))
        observed = TimedRun()
        invoke = (lambda: module(*inputs)) if hip_fn is None else (lambda: module(*inputs, fn=hip_fn))
        elapsed, metadata = perf.benchmark_cuda_graph_or_events(
            invoke, warmup=n_warmup, repetition=n_iter,
            use_cuda_graph=use_cuda_graph, fallback_reason=fallback_reason,
            prepare_fn=prepare_fn, timed_run=observed)
        torch.cuda.synchronize()
        check_result(observed.outputs, expected, inputs, output_contract, perf._compare_results, rtol, atol)
        unchanged_inputs(pristine, inputs)
        # Poison the output and replay the very graph that was timed. A later
        # ordinary Python invocation is not evidence about that graph.
        with torch.no_grad():
            observed.outputs.fill_(float('nan'))
        actual = observed.rerun()
        check_result(actual, expected, inputs, output_contract, perf._compare_results, rtol, atol)
        unchanged_inputs(pristine, inputs)
        # A correct cached answer also survives same-input poisoning. Change the
        # captured input storage, independently recompute the oracle, and replay
        # that same graph again. These controls run after all measured samples.
        try:
            changed = False
            with torch.no_grad():
                for original, value in zip(pristine, inputs):
                    if isinstance(value, torch.Tensor) and value.is_floating_point():
                        value.copy_(original).neg_().add_(0.5)
                        if not torch.isfinite(value).all():
                            raise ValueError('fresh replay input must remain finite')
                        changed |= not torch.equal(value, original)
                if not changed:
                    raise ValueError('fresh replay requires changed floating input')
                fresh_inputs = copy.deepcopy(inputs)
                fresh_expected = module(*copy.deepcopy(inputs))
                if perf._compare_results(expected, fresh_expected, rtol=rtol, atol=atol):
                    raise ValueError('fresh replay oracle must distinguish the original answer')
                observed.outputs.fill_(float('nan'))
            actual = observed.rerun()
            check_result(actual, fresh_expected, inputs, output_contract,
                         perf._compare_results, rtol, atol)
            unchanged_inputs(fresh_inputs, inputs)
        finally:
            # Preserve the original scored workload for the other role, even if
            # the oracle, replay, or numerical check fails. Never perturb model
            # bias/slope/scale or consume the input generator's RNG here.
            with torch.no_grad():
                for original, value in zip(pristine, inputs):
                    if isinstance(value, torch.Tensor):
                        value.copy_(original)
            torch.cuda.synchronize()
        return elapsed, {**metadata, 'replay_validation_valid': True,
                         'replay_validation': 'full_reference_output_and_unchanged_inputs',
                         'changed_input_validation_valid': True,
                         'changed_input_transform': 'x -> 0.5 - x; floating tensors only',
                         'changed_input_restore': 'finally; original scored inputs'}

    perf.cal_hip_latency = latency
    if hasattr(perf, 'cal_modu_latency'):
        perf.cal_modu_latency = lambda module, inputs, **kwargs: latency(module, inputs, **kwargs)
