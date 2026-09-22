"""Exact request-index conversion checks, including counted output and replay."""
from contextlib import contextmanager
import inspect


def snapshots(values):
    return tuple(value.clone() for value in values)


def unchanged(values, pristine):
    import torch
    if any(not torch.equal(value, saved) for value, saved in zip(values, pristine)):
        raise AssertionError('Index conversion modified read-only input buffers')


def reference(harness, inputs, block_size):
    # The unchanged scalar reference uses item(); CPU snapshots avoid repeated
    # device synchronizations while preserving the original exact integer rule.
    result = harness.reference_convert(*(value.cpu() for value in inputs), block_size)
    return result.to(inputs[2].device)


def check_result(result, expected, return_valid_counts):
    import torch
    if return_valid_counts:
        if not isinstance(result, tuple) or len(result) != 2:
            raise AssertionError('Counted conversion must return both indices and counts')
        outputs = result
        references = (expected, (expected != -1).sum(dim=1).to(torch.int32))
    else:
        outputs, references = (result,), (expected,)
    for output, gold in zip(outputs, references):
        if not isinstance(output, torch.Tensor) or (output.shape != gold.shape or
                output.dtype != gold.dtype or output.device != gold.device):
            raise AssertionError('Index conversion output shape/dtype/device is invalid')
        if not torch.equal(output, gold):
            raise AssertionError('Index conversion output or valid counts differ from the exact reference')


def perturb(inputs, block_size):
    req_id, block_table, token_indices = inputs
    req_id.add_(1).remainder_(block_table.shape[0])
    block_table.add_(7)
    limit = block_table.shape[1] * block_size
    token_indices.clamp_min_(0).add_(1).remainder_(limit)
    # Keep original tensor shapes while sensitizing negative, exactly-OOB,
    # farther-OOB and valid boundary behavior in each diagnostic replay.
    token_indices[:, :4] = token_indices.new_tensor([-1, limit, limit + block_size, 0])


@contextmanager
def checked_modules(harness):
    original_load = harness.load_module
    modules = []

    def load():
        module = original_load()
        original = module.convert_req_to_global_index
        modules.append((module, original))

        def checked(req_id, block_table, token_indices, **kwargs):
            inputs = (req_id, block_table, token_indices)
            pristine = snapshots(inputs)
            block_size = kwargs.get('BLOCK_SIZE', 64)
            counted = kwargs.get('return_valid_counts', False)
            expected = reference(harness, pristine, block_size)
            diagnostic = snapshots(pristine)
            perturb(diagnostic, block_size)
            diagnostic_pristine = snapshots(diagnostic)
            diagnostic_expected = reference(harness, diagnostic_pristine, block_size)
            check_result(original(*diagnostic, **kwargs), diagnostic_expected, counted)
            unchanged(diagnostic, diagnostic_pristine)
            result = original(*inputs, **kwargs)
            unchanged(inputs, pristine)
            check_result(result, expected, counted)
            return result

        module.convert_req_to_global_index = checked
        return module

    harness.load_module = load
    try:
        yield
    finally:
        harness.load_module = original_load
        for module, original in reversed(modules):
            module.convert_req_to_global_index = original


def checked_benchmark(harness, benchmark, fn, **kwargs):
    c = inspect.getclosurevars(fn).nonlocals
    module, req_id, block_table, token_indices, block_size = (
        c[key] for key in ('mod', 'req_id', 'block_table', 'token_indices', 'bs'))
    original = module.convert_req_to_global_index
    inputs = (req_id, block_table, token_indices)
    pristine = snapshots(inputs)
    expected = reference(harness, pristine, block_size)
    observed = []

    def observe(*args, **call_kwargs):
        output = original(*args, **call_kwargs)
        observed[:] = [output]
        return output

    def measured():
        fn()
        return observed[0]

    module.convert_req_to_global_index = observe
    try:
        timed = harness._TimedRun()
        ms, metadata = benchmark(measured, timed_run=timed, **kwargs)
        unchanged(inputs, pristine)
        check_result(timed.outputs, expected, False)
        perturb(inputs, block_size)
        replay_pristine = snapshots(inputs)
        replay_expected = reference(harness, replay_pristine, block_size)
        timed.outputs.fill_(-2147483648)
        replayed = timed.rerun()
        unchanged(inputs, replay_pristine)
        check_result(replayed, replay_expected, False)
        return ms, {**metadata, 'timed_output_checked': True,
                    'perturbed_input_replay_checked': True, 'out_of_bounds_checked': True,
                    'source_buffers_unchanged': True}
    finally:
        module.convert_req_to_global_index = original
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
