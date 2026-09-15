# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Validate GELU's returned values and caller-visible state on the timed graph.

All numerical checks run outside the reported samples. The canonical helper
owns capture, warmups, repetitions, stream ordering and device timing.
"""
import copy
import inspect

import torch



def gelu_reference(x, chunk_size=1048576):
    """Independent erf definition, bounded FP64 scratch; never calls F.gelu."""
    import math
    flat = x.detach().reshape(-1)
    output = torch.empty_like(flat)
    for start in range(0, flat.numel(), chunk_size):
        part = flat[start:start + chunk_size].to(torch.float64)
        output[start:start + chunk_size] = (0.5 * part * (1.0 + torch.erf(part / math.sqrt(2.0)))).to(x.dtype)
    return output.reshape(x.shape)


def reference_self_test(module, functional):
    """Analytic controls of the actual baseline paths; no candidate execution."""
    import math
    values = [-6., -3., -2.5, -1., 0., .5, 1., 2.5, 3., 6.]
    x = torch.tensor(values, dtype=torch.float32)
    expected = torch.tensor([0.5 * v * (1.0 + math.erf(v / math.sqrt(2.0))) for v in values])
    torch.testing.assert_close(gelu_reference(x, chunk_size=3), expected, rtol=1e-4, atol=1e-5)
    with torch.no_grad():
        for implementation in (module, functional):
            result = implementation(x.clone())
            if result.shape != x.shape or result.dtype != x.dtype or result.device != x.device:
                raise ValueError("GELU reference output contract mismatch")
            torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-5)


def unchanged_inputs(before, after):
    for expected, actual in zip(before, after):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        elif actual != expected:
            raise ValueError("GELU changed a caller-owned input")


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
        state = {name: value.detach().clone() for name, value in module.state_dict().items()} if hasattr(module, "state_dict") else {}
        try:
            with torch.no_grad():
                # Independent mathematical oracle for the actual timed inputs,
                # including PyTorch baseline actions; never F.gelu vs F.gelu.
                expected = gelu_reference(inputs[0])
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
