"""Check selected-token log probabilities and exact timed output replay."""
from contextlib import contextmanager
import inspect

SYMBOL = 'compute_token_logprobs'


def snapshots(values):
    return tuple(value.clone() for value in values)


def unchanged(values, saved):
    import torch
    for value, original in zip(values, saved):
        if not torch.equal(value, original):
            raise AssertionError('Token log-probability modified a read-only input')


def reference(inputs):
    import torch
    logits, token_ids = inputs
    return torch.log_softmax(logits.float(), dim=-1).gather(1, token_ids.long())


def check_output(value, expected):
    import torch
    if not isinstance(value, torch.Tensor) or (value.shape != expected.shape or
            value.dtype != expected.dtype or value.device != expected.device):
        raise AssertionError('Token log-probability output shape/dtype/device is invalid')
    if not torch.isfinite(value).all():
        raise AssertionError('Token log-probability output must be finite')
    torch.testing.assert_close(value, expected, atol=1e-2, rtol=1e-2)


@contextmanager
def checked_modules(harness):
    loader = harness.load_module
    patched = []

    def load():
        module = loader()
        original = getattr(module, SYMBOL)
        patched.append((module, original))
        diagnosed = False

        def verify(logits, token_ids):
            inputs = (logits, token_ids)
            pristine = snapshots(inputs)
            expected = reference(pristine)
            try:
                result = original(logits, token_ids)
                unchanged(inputs, pristine)
                check_output(result, expected)
                return result
            finally:
                for value, saved in zip(inputs, pristine):
                    value.copy_(saved)

        def checked(logits, token_ids):
            nonlocal diagnosed
            import torch
            result = verify(logits, token_ids)
            if not diagnosed:
                # Unscored: a partial second 1024-token block, repeated and edge
                # token IDs, a non-power-of-two gather width, and large logits.
                index = torch.arange(3*1031, device=logits.device).reshape(3,1031)
                data = ((index%17)-8).float()
                data[0, 0], data[0, -1] = 1000., -1000.
                data[1, 0], data[1, -1] = -1000., 1000.
                data[2].zero_()
                ids = torch.tensor([[0,1030,0,1024,7,1029,1]], device=logits.device).expand(3,-1).clone()
                verify(data, ids)
                diagnosed = True
            return result

        setattr(module, SYMBOL, checked)
        return module

    harness.load_module = load
    try:
        yield
    finally:
        harness.load_module = loader
        for module, original in reversed(patched):
            setattr(module, SYMBOL, original)


def checked_benchmark(harness, benchmark, fn, **options):
    state = inspect.getclosurevars(fn).nonlocals
    module = state['mod']
    inputs = tuple(state[name] for name in ('logits', 'token_ids'))
    pristine = snapshots(inputs)
    expected = reference(pristine)
    original = getattr(module, SYMBOL)
    captured = None

    def collect(*args, **kwargs):
        nonlocal captured
        captured = original(*args, **kwargs)
        return captured

    def measured():
        nonlocal captured
        captured = None
        fn()
        if captured is None:
            raise AssertionError('Timed token log-probability did not return an output')
        return captured

    setattr(module, SYMBOL, collect)
    try:
        timed = harness._TimedRun()
        ms, metadata = benchmark(measured, timed_run=timed, **options)
        unchanged(inputs, pristine)
        check_output(timed.outputs, expected)
        inputs[0].mul_(-.75)
        inputs[1].add_(17).remainder_(inputs[0].shape[1])
        replay_inputs = snapshots(inputs)
        replay_expected = reference(replay_inputs)
        timed.outputs.fill_(float('nan'))
        replayed = timed.rerun()
        unchanged(inputs, replay_inputs)
        check_output(replayed, replay_expected)
        return ms, {**metadata, 'timed_output_checked': True,
                    'perturbed_input_replay_checked': True, 'source_buffers_unchanged': True}
    finally:
        setattr(module, SYMBOL, original)
        for value, saved in zip(inputs, pristine):
            value.copy_(saved)


def install(harness):
    correctness, performance = harness.run_correctness, harness.run_performance

    def checked_correctness(*args, **kwargs):
        with checked_modules(harness):
            return correctness(*args, **kwargs)

    def checked_performance():
        benchmark = harness._benchmark_cuda_graph_or_events
        harness._benchmark_cuda_graph_or_events = lambda fn, **kwargs: checked_benchmark(harness, benchmark, fn, **kwargs)
        try:
            return performance()
        finally:
            harness._benchmark_cuda_graph_or_events = benchmark

    harness.run_correctness, harness.run_performance = checked_correctness, checked_performance
