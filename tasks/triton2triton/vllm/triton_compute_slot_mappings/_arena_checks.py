"""Exact slot-mapping checks with ragged, permuted diagnostic inputs and replay."""
from contextlib import contextmanager
import inspect


def snapshots(inputs):
    return tuple(value.clone() for value in inputs)


def unchanged(inputs, pristine):
    import torch
    if any(not torch.equal(value, saved) for value, saved in zip(inputs, pristine)):
        raise AssertionError('Slot mapping modified read-only inputs')


def reference(harness, inputs, block_size):
    return harness.reference_compute_slot_mappings(
        *(value.cpu() for value in inputs), block_size).to(inputs[2].device)


def check_output(output, expected):
    import torch
    if not isinstance(output, torch.Tensor) or (output.shape != expected.shape or
            output.dtype != expected.dtype or output.device != expected.device):
        raise AssertionError('Slot mapping output shape/dtype/device is invalid')
    if not torch.equal(output, expected):
        raise AssertionError('Slot mapping differs from the exact reference')


def perturb(inputs, block_size):
    mapping, starts, positions, table = inputs
    # Keep request/token counts and the timed launch bound unchanged.
    mapping.copy_(mapping.flip(0))
    if mapping.numel() > 1:
        # Shift one boundary while retaining a valid cumulative partition.
        if int((starts[2] - starts[1]).item()) > 1:
            starts[1].add_(1)
        else:
            starts[1].sub_(1)
    positions.mul_(3).add_(block_size - 1).remainder_(table.shape[1] * block_size)
    table.add_(11)


@contextmanager
def checked_modules(harness):
    original_load = harness.load_module
    modules = []

    def load():
        module = original_load()
        original = module.compute_slot_mappings
        modules.append((module, original))

        def checked(mapping, starts, positions, table, block_size, max_num_tokens):
            inputs = (mapping, starts, positions, table)
            pristine = snapshots(inputs)
            expected = reference(harness, pristine, block_size)
            diagnostic = snapshots(pristine)
            perturb(diagnostic, block_size)
            diagnostic_pristine = snapshots(diagnostic)
            diagnostic_expected = reference(harness, diagnostic_pristine, block_size)
            check_output(original(*diagnostic, block_size, max_num_tokens), diagnostic_expected)
            unchanged(diagnostic, diagnostic_pristine)
            output = original(*inputs, block_size, max_num_tokens)
            unchanged(inputs, pristine)
            check_output(output, expected)
            return output

        module.compute_slot_mappings = checked
        return module

    harness.load_module = load
    try:
        yield
    finally:
        harness.load_module = original_load
        for module, original in reversed(modules):
            module.compute_slot_mappings = original


def checked_benchmark(harness, benchmark, fn, **kwargs):
    c = inspect.getclosurevars(fn).nonlocals
    module, block_size = c['mod'], c['block_size']
    inputs = tuple(c[key] for key in ('idx_mapping', 'query_start_loc', 'positions', 'block_table'))
    pristine = snapshots(inputs)
    expected = reference(harness, pristine, block_size)
    original = module.compute_slot_mappings
    observed = []

    def observe(*args, **call_kwargs):
        output = original(*args, **call_kwargs)
        observed[:] = [output]
        return output

    def measured():
        fn()
        return observed[0]

    module.compute_slot_mappings = observe
    try:
        timed = harness._TimedRun()
        ms, metadata = benchmark(measured, timed_run=timed, **kwargs)
        unchanged(inputs, pristine)
        check_output(timed.outputs, expected)
        perturb(inputs, block_size)
        replay_pristine = snapshots(inputs)
        replay_expected = reference(harness, replay_pristine, block_size)
        timed.outputs.fill_(-2)
        replayed = timed.rerun()
        unchanged(inputs, replay_pristine)
        check_output(replayed, replay_expected)
        return ms, {**metadata, 'timed_output_checked': True,
                    'perturbed_input_replay_checked': True, 'ragged_permuted_mapping_checked': True,
                    'source_buffers_unchanged': True}
    finally:
        module.compute_slot_mappings = original
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
