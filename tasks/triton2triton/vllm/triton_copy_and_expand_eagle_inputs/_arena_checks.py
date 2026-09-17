"""Validate all six EAGLE outputs, including the exact captured invocation."""
from contextlib import contextmanager
import inspect


def snapshots(inputs):
    return tuple(value.clone() for value in inputs)


def unchanged(inputs, pristine):
    import torch
    if any(not torch.equal(value, saved) for value, saved in zip(inputs, pristine)):
        raise AssertionError('EAGLE expansion modified read-only inputs')


def reference(harness, inputs, padding, drafting, slots, shift):
    outputs = harness.reference_copy_and_expand(
        *(value.cpu() for value in inputs), padding, drafting, slots, shift)
    return tuple(value.to(inputs[0].device) for value in outputs)


def check_outputs(outputs, expected):
    import torch
    if not isinstance(outputs, tuple) or len(outputs) != 6:
        raise AssertionError('EAGLE expansion must return exactly six tensors')
    for index, (output, gold) in enumerate(zip(outputs, expected)):
        if not isinstance(output, torch.Tensor) or (output.shape != gold.shape or
                output.dtype != gold.dtype or output.device != gold.device):
            raise AssertionError(f'EAGLE output {index} shape/dtype/device is invalid')
        if not torch.equal(output, gold):
            raise AssertionError(f'EAGLE output {index} violates the exact reference')


@contextmanager
def checked_modules(harness):
    original_load = harness.load_module
    modules = []

    def load():
        module = original_load()
        original = module.copy_and_expand_eagle_inputs
        modules.append((module, original))

        def checked(tt, tp, nt, qsl, qel, padding_token_id,
                    parallel_drafting_token_id, num_padding_slots_per_request,
                    shift_input_ids, max_output_tokens_per_req):
            inputs = (tt, tp, nt, qsl, qel)
            pristine = snapshots(inputs)
            expected = reference(harness, pristine, padding_token_id,
                                 parallel_drafting_token_id,
                                 num_padding_slots_per_request, shift_input_ids)
            output = original(*inputs, padding_token_id, parallel_drafting_token_id,
                              num_padding_slots_per_request, shift_input_ids,
                              max_output_tokens_per_req)
            unchanged(inputs, pristine)
            check_outputs(output, expected)
            return output

        module.copy_and_expand_eagle_inputs = checked
        return module

    harness.load_module = load
    try:
        yield
    finally:
        harness.load_module = original_load
        for module, original in reversed(modules):
            module.copy_and_expand_eagle_inputs = original


def checked_benchmark(harness, benchmark, fn, **kwargs):
    import torch
    c = inspect.getclosurevars(fn).nonlocals
    module = c['mod']
    inputs = tuple(c[name] for name in ('tt', 'tp', 'nt', 'qsl', 'qel'))
    pristine = snapshots(inputs)
    expected = reference(harness, pristine, -1, -2, c['nps'], False)
    original = module.copy_and_expand_eagle_inputs
    observed = []

    def observe(*args, **call_kwargs):
        output = original(*args, **call_kwargs)
        observed[:] = [output]
        return output

    def measured():
        fn()
        return observed[0]

    module.copy_and_expand_eagle_inputs = observe
    try:
        timed = harness._TimedRun()
        ms, metadata = benchmark(measured, timed_run=timed, **kwargs)
        unchanged(inputs, pristine)
        check_outputs(timed.outputs, expected)
        tt, tp, nt, qsl, qel = inputs
        tt.add_(17).remainder_(32000)
        tp.add_(11)
        nt.add_(23).remainder_(32000)
        # Alternate zero-rejection and maximal-rejection requests; tensor sizes
        # and launch bounds remain identical to the originally timed workload.
        ends = qsl[1:] - 1
        qel.copy_(torch.where(qel == ends, qsl[:-1], ends))
        replay_pristine = snapshots(inputs)
        replay_expected = reference(harness, replay_pristine, -1, -2, c['nps'], False)
        for output, gold in zip(timed.outputs, replay_expected):
            if output.dtype == torch.bool:
                output.copy_(~gold)
            else:
                output.fill_(-2147483648)
        replayed = timed.rerun()
        unchanged(inputs, replay_pristine)
        check_outputs(replayed, replay_expected)
        return ms, {**metadata, 'timed_output_checked': True,
                    'all_six_outputs_checked': True, 'perturbed_input_replay_checked': True,
                    'source_buffers_unchanged': True}
    finally:
        module.copy_and_expand_eagle_inputs = original
        for value, saved in zip(inputs, pristine):
            value.copy_(saved)


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
