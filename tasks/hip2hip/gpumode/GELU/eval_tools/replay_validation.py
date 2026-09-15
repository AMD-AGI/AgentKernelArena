# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Validate GELU's returned values and caller-visible state on the timed graph.

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
            raise ValueError("GELU changed a caller-owned input")


def check_result(actual, expected, inputs, output_contract, compare, rtol, atol):
    output_contract(expected, actual)
    if not compare(expected, actual, rtol=rtol, atol=atol):
        raise ValueError("Timed GELU output disagrees with the protected reference")
    separate_output(actual, inputs)


def separate_output(actual, inputs):
    # F.gelu is out-of-place: preserving values alone cannot establish its
    # caller-visible storage contract.
    for value in inputs:
        if isinstance(value, torch.Tensor) and actual.untyped_storage().data_ptr() == value.untyped_storage().data_ptr():
            raise ValueError("GELU output aliases a caller-owned input")


def install(perf, output_contract):
    from _aka_benchmark import TimedRun
    signature = inspect.signature(perf.cal_kernel_perf).parameters
    rtol, atol = signature['rtol'].default, signature['atol'].default

    def latency(module, inputs, hip_fn=None, n_iter=100, n_warmup=10,
                use_cuda_graph=True, fallback_reason=None, prepare_fn=None):
        pristine = copy.deepcopy(inputs)
        with torch.no_grad():
            # The functional module's default is the protected PyTorch GELU,
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
        return elapsed, {**metadata, 'replay_validation_valid': True,
                         'replay_validation': 'full_reference_output_and_unchanged_inputs'}

    perf.cal_hip_latency = latency
    if hasattr(perf, 'cal_modu_latency'):
        perf.cal_modu_latency = lambda module, inputs, **kwargs: latency(module, inputs, **kwargs)
