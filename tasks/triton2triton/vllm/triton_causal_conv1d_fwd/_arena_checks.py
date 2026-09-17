"""Validate causal-convolution outputs, state transitions and actual timed replay."""
from contextlib import contextmanager
import inspect


def snapshots(values):
    return tuple(None if value is None else value.clone() for value in values)


def unchanged(values, pristine):
    import torch
    if any(value is not None and not torch.equal(value, saved)
           for value, saved in zip(values, pristine)):
        raise AssertionError('Causal convolution modified read-only input buffers')


def restore(values, pristine):
    for value, saved in zip(values, pristine):
        if value is not None:
            value.copy_(saved)


def expected_state(x, initial, starts, cache_indices):
    """Original workload: each sequence supplies at least width-1 new tokens."""
    result = initial.clone()
    state_len = initial.shape[-1]
    boundaries, slots = starts.cpu().tolist(), cache_indices.cpu().tolist()
    for index, slot in enumerate(slots):
        start, end = boundaries[index:index + 2]
        if end - start < state_len or slot < 0:
            raise ValueError('State reference requires the declared non-padding workload')
        result[slot].copy_(x[:, end - state_len:end])
    return result


def references(harness, x, weight, bias, initial, starts, cache_indices, has_init, activation):
    # Calculate before invoking the candidate; candidate mutations must never
    # rewrite either the numerical oracle or the expected state transition.
    output = harness.reference_conv1d(
        x, weight, bias, initial, starts.cpu(), has_init.cpu(), activation).to(x.device)
    return output.to(x.dtype), expected_state(x, initial, starts, cache_indices)


def check_outputs(output, state, expected):
    import torch
    for value, reference in zip((output, state), expected):
        if not isinstance(value, torch.Tensor) or (value.shape != reference.shape or
                value.dtype != reference.dtype or value.device != reference.device):
            raise AssertionError('Causal convolution output/state shape, dtype or device is invalid')
        if not torch.isfinite(value).all():
            raise AssertionError('Causal convolution output/state is nonfinite')
    if not torch.allclose(output, expected[0], atol=1e-1, rtol=1e-1):
        raise AssertionError('Causal convolution violates the original output tolerance')
    # The cache contains copied input tokens, with no arithmetic or rounding.
    if not torch.equal(state, expected[1]):
        raise AssertionError('Causal convolution cached state differs from the input tail')


@contextmanager
def checked_modules(harness):
    original_load = harness.load_module
    modules = []

    def load():
        module = original_load()
        original = module.causal_conv1d_fwd
        modules.append((module, original))

        def checked(x, weight, bias, state, starts, cache_indices, has_init, *, activation=None):
            inputs = (x, weight, bias, starts, cache_indices, has_init)
            pristine = snapshots(inputs)
            expected = references(harness, *pristine[:3], state.clone(),
                                  *pristine[3:], activation)
            output = original(x, weight, bias, state, starts, cache_indices,
                              has_init, activation=activation)
            unchanged(inputs, pristine)
            check_outputs(output, state, expected)
            return output

        module.causal_conv1d_fwd = checked
        return module

    harness.load_module = load
    try:
        yield
    finally:
        harness.load_module = original_load
        for module, original in reversed(modules):
            module.causal_conv1d_fwd = original


def checked_benchmark(harness, benchmark, fn, **kwargs):
    c = inspect.getclosurevars(fn).nonlocals
    x, weight, bias, state, starts, indices, has_init, out = (
        c[key] for key in ('x', 'weight', 'bias_t', 'conv_states', 'query_start_loc',
                          'cache_indices', 'has_init', 'out'))
    initial = inspect.getclosurevars(kwargs['prepare_fn']).nonlocals['initial_conv_states']
    inputs = (x, weight, bias, starts, indices, has_init,
              c['batch_ptr'], c['token_chunk_offset_ptr'], initial)
    pristine = snapshots(inputs)
    original_state = state.clone()
    activation = 'silu' if c['activation'] in ('silu', 'swish') else None
    expected = references(harness, x.clone(), weight.clone(),
                          None if bias is None else bias.clone(), initial.clone(),
                          starts.clone(), indices.clone(), has_init.clone(), activation)

    def measured():
        fn()
        return out, state

    try:
        timed = harness._TimedRun()
        ms, metadata = benchmark(measured, timed_run=timed, **kwargs)
        if timed.outputs[0] is not out or timed.outputs[1] is not state:
            raise AssertionError('Benchmark did not retain actual output and state buffers')
        unchanged(inputs, pristine)
        check_outputs(*timed.outputs, expected)
        x.neg_()
        weight.mul_(0.5)
        if bias is not None:
            bias.neg_()
        initial.mul_(0.25)
        replay_pristine = snapshots(inputs)
        replay_expected = references(harness, x.clone(), weight.clone(),
                                     None if bias is None else bias.clone(), initial.clone(),
                                     starts.clone(), indices.clone(), has_init.clone(), activation)
        out.fill_(float('nan'))
        state.fill_(float('nan'))
        replayed = timed.rerun()  # Original prepare_fn resets state outside timing.
        unchanged(inputs, replay_pristine)
        if replayed[0] is not out or replayed[1] is not state:
            raise AssertionError('Replay returned different output/state buffers')
        check_outputs(*replayed, replay_expected)
        return ms, {**metadata, 'timed_output_checked': True,
                    'cached_state_checked': True, 'perturbed_input_replay_checked': True,
                    'source_buffers_unchanged': True}
    finally:
        restore(inputs, pristine)
        state.copy_(original_state)


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
