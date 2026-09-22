"""Validate staged writes from the same graph whose device time is reported."""
import inspect


def check_output(output, initial, indices, starts, contents, cumulative, reference):
    import torch

    if not isinstance(output, torch.Tensor):
        raise AssertionError('Staged writes must produce a tensor')
    if output.shape != initial.shape or output.dtype != initial.dtype or output.device != initial.device:
        raise AssertionError('Staged write output shape/dtype/device violates the contract')
    expected = reference(initial.cpu(), indices.cpu(), starts.cpu(), contents.cpu(), cumulative.cpu())
    if not torch.equal(output.cpu(), expected):
        raise AssertionError('Staged writes differ from the exact reference, including untouched cells')


def checked_benchmark(harness, benchmark, fn, **kwargs):
    import torch

    inputs = inspect.getclosurevars(fn).nonlocals
    output, indices, starts, contents, cumulative = (inputs[key] for key in (
        'output_work', 'write_indices', 'write_starts', 'write_contents', 'cu_lens'))
    initial = inspect.getclosurevars(kwargs['prepare_fn']).nonlocals['output']

    def measured():
        fn()
        return output

    timed = harness._TimedRun()
    ms, metadata = benchmark(measured, timed_run=timed, **kwargs)
    check_output(timed.outputs, initial, indices, starts, contents, cumulative,
                 harness.reference_apply_write)

    # Preserve the original workload and seed during timing. Afterwards vary
    # mapping, offsets, lengths and contents at the same captured addresses.
    # Pairwise length changes preserve the total content buffer size.
    lengths = torch.diff(cumulative, prepend=torch.zeros_like(cumulative[:1]))
    pairs = (lengths.numel() // 2) * 2
    lengths[:pairs:2].sub_(1)
    lengths[1:pairs:2].add_(1)
    if (lengths <= 0).any():
        raise AssertionError('Replay control requires positive staged-write lengths')
    cumulative.copy_(lengths.cumsum(0))
    indices.copy_(indices.flip(0))
    starts.copy_(1 + torch.arange(starts.numel(), device=starts.device, dtype=starts.dtype) % 3)
    if (starts + lengths > output.shape[1]).any():
        raise AssertionError('Replay control would exceed its declared row size')
    contents.neg_()
    initial.fill_(-23)
    output.fill_(-99)
    # TimedRun retains prepare_fn: each replay restores the modified initial
    # state outside timing, then executes exactly the measured graph.
    check_output(timed.rerun(), initial, indices, starts, contents, cumulative,
                 harness.reference_apply_write)
    return ms, {**metadata, 'timed_output_checked': True,
                'perturbed_input_replay_checked': True}


def install(harness):
    original_performance = harness.run_performance

    def performance():
        benchmark = harness._benchmark_cuda_graph_or_events
        harness._benchmark_cuda_graph_or_events = lambda fn, **kwargs: checked_benchmark(
            harness, benchmark, fn, **kwargs)
        try:
            return original_performance()
        finally:
            harness._benchmark_cuda_graph_or_events = benchmark

    harness.run_performance = performance
