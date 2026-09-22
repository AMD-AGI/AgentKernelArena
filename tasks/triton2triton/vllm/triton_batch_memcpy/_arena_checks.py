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


def check_sources(sources, pristine):
    try:
        check_outputs(sources, pristine)
    except AssertionError as exc:
        raise AssertionError('Batch memcpy modified caller-owned source buffers') from exc


def checked_benchmark(harness, benchmark, fn, sources, destinations, **kwargs):
    def measured():
        fn()
        return tuple(destinations)

    pristine = tuple(source.clone() for source in sources)
    timed = harness._TimedRun()
    ms, metadata = benchmark(measured, timed_run=timed, **kwargs)
    check_sources(sources, pristine)
    check_outputs(timed.outputs, pristine)
    # Change bytes in place, retaining all original pointer tables, lengths,
    # buffer allocations and timed workloads. Check every variable-length copy.
    for source, destination in zip(sources, destinations):
        source.bitwise_xor_(255)
        destination.zero_()
    replay_pristine = tuple(source.clone() for source in sources)
    outputs = timed.rerun()
    check_sources(sources, replay_pristine)
    check_outputs(outputs, replay_pristine)
    return ms, {**metadata, 'timed_output_checked': True,
                'perturbed_input_replay_checked': True,
                'source_buffers_unchanged': True}


def install(harness):
    original_correctness = harness.run_correctness
    original_performance = harness.run_performance

    def correctness(*args, **kwargs):
        make_inputs = harness.make_inputs
        retained = []

        def retained_inputs(*input_args, **input_kwargs):
            inputs = make_inputs(*input_args, **input_kwargs)
            sources, destinations = inputs[:2]
            retained.append((sources, destinations, tuple(source.clone() for source in sources)))
            return inputs

        harness.make_inputs = retained_inputs
        try:
            ok, error = original_correctness(*args, **kwargs)
            if not ok:
                return ok, error
            for sources, destinations, pristine in retained:
                check_sources(sources, pristine)
                check_outputs(destinations, pristine)
            return True, None
        except Exception as exc:
            return False, f'{type(exc).__name__}: {exc}'
        finally:
            harness.make_inputs = make_inputs

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

    harness.run_correctness = correctness
    harness.run_performance = performance
