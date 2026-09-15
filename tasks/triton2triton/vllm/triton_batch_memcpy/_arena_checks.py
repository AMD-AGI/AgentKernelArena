"""Check every copied byte in the exact timed batch-memcpy graph."""


def check_outputs(outputs, sources):
    import torch

    if not isinstance(outputs, (list, tuple)) or len(outputs) != len(sources):
        raise AssertionError('Batch memcpy must expose every destination buffer')
    for output, source in zip(outputs, sources):
        if not isinstance(output, torch.Tensor):
            raise AssertionError('Batch memcpy destination must be a tensor')
        if output.shape != source.shape or output.dtype != source.dtype or output.device != source.device:
            raise AssertionError('Batch memcpy output shape/dtype/device violates the contract')
        if not torch.equal(output, source):
            raise AssertionError('Batch memcpy differs from the exact source bytes')


def checked_benchmark(harness, benchmark, fn, sources, destinations, **kwargs):
    def measured():
        fn()
        return tuple(destinations)

    timed = harness._TimedRun()
    ms, metadata = benchmark(measured, timed_run=timed, **kwargs)
    check_outputs(timed.outputs, sources)
    # Change bytes in place, retaining all original pointer tables, lengths,
    # buffer allocations and timed workloads. Check every variable-length copy.
    for source, destination in zip(sources, destinations):
        source.bitwise_xor_(255)
        destination.zero_()
    check_outputs(timed.rerun(), sources)
    return ms, {**metadata, 'timed_output_checked': True,
                'perturbed_input_replay_checked': True}


def install(harness):
    original_performance = harness.run_performance

    def performance():
        benchmark = harness._benchmark_cuda_graph_or_events
        make_inputs = harness.make_inputs
        current = []

        def retained_inputs(*args, **kwargs):
            inputs = make_inputs(*args, **kwargs)
            current[:] = inputs[:2]
            return inputs

        def measured(fn, **kwargs):
            return checked_benchmark(harness, benchmark, fn, *current, **kwargs)

        harness.make_inputs = retained_inputs
        harness._benchmark_cuda_graph_or_events = measured
        try:
            return original_performance()
        finally:
            harness.make_inputs = make_inputs
            harness._benchmark_cuda_graph_or_events = benchmark

    harness.run_performance = performance
