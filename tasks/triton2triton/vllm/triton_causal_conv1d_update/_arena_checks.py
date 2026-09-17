"""Check the in-place output and convolution-cache update, including timed replay."""
from contextlib import contextmanager
import inspect


def snapshots(values):
    return tuple(None if value is None else value.clone() for value in values)


def restore(values, saved):
    for value, pristine in zip(values, saved):
        if value is not None:
            value.copy_(pristine)


def unchanged(values, saved):
    import torch
    if any(value is not None and not torch.equal(value, pristine)
           for value, pristine in zip(values, saved)):
        raise AssertionError('Convolution update modified read-only input buffers')


def references(harness, x, state, weight, bias, activation, indices):
    import torch
    slots = indices.to(dtype=torch.long)
    output = harness.reference_conv1d_update(
        x, state.index_select(0, slots), weight, bias, activation).to(x.device, x.dtype)
    expected_state = state.clone()
    for sequence, slot in enumerate(slots.cpu().tolist()):
        if slot < 0:
            raise ValueError('State reference requires the declared non-padding workload')
        history = torch.cat((state[slot], x[sequence]), dim=-1)
        expected_state[slot].copy_(history[:, -state.shape[-1]:])
    return output, expected_state


def check_outputs(output, x, state, expected):
    import torch
    for value, reference in zip((output, x, state), (expected[0], expected[0], expected[1])):
        if not isinstance(value, torch.Tensor) or (value.shape != reference.shape or
                value.dtype != reference.dtype or value.device != reference.device):
            raise AssertionError('Convolution update output/state shape, dtype or device is invalid')
        if not torch.isfinite(value).all():
            raise AssertionError('Convolution update output/state is nonfinite')
    # For the declared equal-dtype inputs, the existing API writes x in place.
    if output.data_ptr() != x.data_ptr():
        raise AssertionError('Convolution update must preserve the in-place output contract')
    if (not torch.allclose(output, expected[0], atol=1e-1, rtol=1e-1) or
            not torch.allclose(x, expected[0], atol=1e-1, rtol=1e-1)):
        raise AssertionError('Convolution update violates the original output tolerance')
    if not torch.equal(state, expected[1]):
        raise AssertionError('Convolution update cached state differs from the shifted input history')


@contextmanager
def checked_modules(harness):
    original_load = harness.load_module
    modules = []

    def load():
        module = original_load()
        original = module.causal_conv1d_update
        modules.append((module, original))

        def checked(x, state, weight, *, bias=None, activation=None, conv_state_indices=None):
            inputs = (weight, bias, conv_state_indices)
            pristine = snapshots(inputs)
            expected = references(harness, x.clone(), state.clone(), *pristine[:2],
                                  activation, pristine[2])
            output = original(x, state, weight, bias=bias, activation=activation,
                              conv_state_indices=conv_state_indices)
            unchanged(inputs, pristine)
            check_outputs(output, x, state, expected)
            return output

        module.causal_conv1d_update = checked
        return module

    harness.load_module = load
    try:
        yield
    finally:
        harness.load_module = original_load
        for module, original in reversed(modules):
            module.causal_conv1d_update = original


def checked_benchmark(harness, benchmark, fn, **kwargs):
    c = inspect.getclosurevars(fn).nonlocals
    x_work, state_work, weight, bias, indices, activation = (
        c[key] for key in ('x_work', 'conv_state_work', 'weight', 'bias_t',
                          'conv_state_indices', 'activation'))
    preparation = inspect.getclosurevars(kwargs['prepare_fn']).nonlocals
    x, state = preparation['x'], preparation['conv_state']
    inputs = (x, state, weight, bias, indices)
    pristine = snapshots(inputs)
    working = (x_work, state_work)
    working_saved = snapshots(working)
    expected = references(harness, *pristine[:4], activation, pristine[4])

    def measured():
        return fn(), state_work

    try:
        timed = harness._TimedRun()
        ms, metadata = benchmark(measured, timed_run=timed, **kwargs)
        unchanged(inputs, pristine)
        if timed.outputs[1] is not state_work:
            raise AssertionError('Benchmark did not retain the actual updated state buffer')
        check_outputs(timed.outputs[0], x_work, timed.outputs[1], expected)
        x.neg_()
        state.mul_(0.25)
        weight.mul_(0.5)
        if bias is not None:
            bias.neg_()
        replay_pristine = snapshots(inputs)
        replay_expected = references(harness, *replay_pristine[:4], activation, replay_pristine[4])
        x_work.fill_(float('nan'))
        state_work.fill_(float('nan'))
        replayed = timed.rerun()  # Same prepare_fn restores the in-place inputs.
        unchanged(inputs, replay_pristine)
        if replayed[1] is not state_work:
            raise AssertionError('Replay returned a different state buffer')
        check_outputs(replayed[0], x_work, replayed[1], replay_expected)
        return ms, {**metadata, 'timed_output_checked': True,
                    'cached_state_checked': True, 'perturbed_input_replay_checked': True,
                    'source_buffers_unchanged': True}
    finally:
        restore(inputs, pristine)
        restore(working, working_saved)


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
