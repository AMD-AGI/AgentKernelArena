# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Validate the operator's returned values and caller-visible state on the timed graph.

All numerical checks run outside the reported samples. The canonical helper
owns capture, warmups, repetitions, stream ordering and device timing.
"""
import copy
import inspect

import torch


def softmax_reference(x, axis=-1, chunk_elements=1048576):
    """FP64 exp/sum definition with bounded scratch; never calls softmax."""
    if chunk_elements <= 0:
        raise ValueError("Positive reference scratch bound required")
    values = x.detach().movedim(axis, -1)
    width = values.shape[-1]
    rows = values.reshape(-1, width)
    output = torch.empty_like(rows)
    step = max(1, chunk_elements // width)
    for start in range(0, rows.shape[0], step):
        part = rows[start:start + step].double()
        weights = torch.exp(part - part.amax(dim=-1, keepdim=True))
        output[start:start + step] = (weights / weights.sum(dim=-1, keepdim=True)).to(x.dtype)
    return output.reshape(values.shape).movedim(-1, axis)


def reference_self_test(module, functional):
    """Known answers cross-check both actual production paths independently."""
    import math
    axis = getattr(module, 'axis', -1)
    values = [[0., math.log(2), math.log(3), math.log(4)],
              [1000., 1000., 999., 998.], [-1000., -1000., -1000., -1000.]]
    normalizer = 2 + math.exp(-1) + math.exp(-2)
    expected = [[.1, .2, .3, .4],
                [1/normalizer, 1/normalizer, math.exp(-1)/normalizer, math.exp(-2)/normalizer],
                [.25, .25, .25, .25]]
    x = torch.tensor(values, dtype=torch.float32)
    answer = torch.tensor(expected, dtype=torch.float32)
    if axis == 2:
        x, answer = x.unsqueeze(0), answer.unsqueeze(0)
    torch.testing.assert_close(softmax_reference(x, axis, chunk_elements=4), answer, rtol=1e-4, atol=1e-5)
    for implementation in (module, functional):
        with torch.no_grad():
            actual = implementation(x.clone())
        if actual.shape != x.shape or actual.dtype != x.dtype or actual.device != x.device:
            raise ValueError("Softmax reference output contract mismatch")
        torch.testing.assert_close(actual, answer, rtol=1e-4, atol=1e-5)
    wrong_axis = 1 if axis == 2 else 0
    if torch.allclose(softmax_reference(x, wrong_axis), answer, rtol=1e-4, atol=1e-5):
        raise ValueError("Softmax control cannot detect wrong-axis normalization")


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
                # Independent exp/sum oracle for the actual timed inputs;
                # both baseline and candidate are checked outside timing.
                expected = softmax_reference(inputs[0], getattr(module, 'axis', -1))
            observed = TimedRun()
            invoke = (lambda: module(*inputs)) if hip_fn is None else (lambda: module(*inputs, fn=hip_fn))
            elapsed, metadata = perf.benchmark_cuda_graph_or_events(
                invoke, warmup=n_warmup, repetition=n_iter,
                use_cuda_graph=use_cuda_graph, fallback_reason=fallback_reason,
                prepare_fn=prepare_fn, timed_run=observed)
            kind = metadata.get('benchmark_timed_run_kind')
            expected_method = {'captured_graph': 'cuda_graph', 'eager_callable': 'cuda_event_fallback'}.get(kind)
            if expected_method is None or metadata.get('benchmark_method') != expected_method:
                raise ValueError('Benchmark did not identify its actual observed invocation')
            torch.cuda.synchronize()
            check_result(observed.outputs, expected, inputs, output_contract, perf._compare_results, rtol, atol)
            unchanged_inputs(pristine, inputs)
            unchanged_model_state(state, module)
            # Poison the output and replay the very graph that was timed. A later
            # ordinary Python invocation is not evidence about that graph.
            with torch.no_grad():
                observed.outputs.fill_(float('nan'))
            actual = observed.rerun()
            check_result(actual, expected, inputs, output_contract, perf._compare_results, rtol, atol)
            unchanged_inputs(pristine, inputs)
            unchanged_model_state(state, module)
            return elapsed, {**metadata, 'replay_validation_valid': True,
                             'input_state_restored': True,
                             'model_state_validation_valid': True,
                             'model_state_tensor_count': len(state),
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
