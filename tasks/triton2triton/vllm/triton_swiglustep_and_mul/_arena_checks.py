"""Validate both clamp branches and the actual measured SwiGLU output."""
from contextlib import contextmanager
import inspect


def unchanged(value, original):
    import torch
    if not torch.equal(value, original):
        raise AssertionError('SwiGLU-step modified its read-only input')


def reference(harness, data, limit):
    return harness.reference_swiglustep_and_mul(data, limit)


def check_output(value, expected):
    import torch
    if not isinstance(value, torch.Tensor) or (value.shape != expected.shape or
            value.dtype != expected.dtype or value.device != expected.device):
        raise AssertionError('SwiGLU-step output shape/dtype/device is invalid')
    if not torch.isfinite(value).all():
        raise AssertionError('SwiGLU-step output must be finite')
    torch.testing.assert_close(value, expected, atol=1e-2, rtol=1e-2)


def clamp_controls(device, dtype):
    """Representative legal widths around lane and tile boundaries."""
    import torch
    pattern = torch.tensor([-20., -8., -1., 0., 1., 8., 20.], device=device, dtype=dtype)
    for width in (1, 3, 31, 33, 63, 65, 127, 129, 257, 511, 513,
                  1023, 1025, 1031, 2047, 2049):
        index = torch.arange(6 * width, device=device).reshape(6, width) % 7
        data = torch.cat((pattern[index], pattern[(index + 3) % 7]), dim=1)[::2]
        for limit in (7.0, .1):
            yield data, limit


@contextmanager
def checked_modules(harness):
    loader = harness.load_module
    patched = []

    def load():
        module = loader()
        original = module.swiglustep_and_mul
        patched.append((module, original))
        diagnosed = False

        def verify(data, limit):
            pristine = data.clone()
            expected = reference(harness, pristine, limit)
            result = original(data, limit=limit)
            unchanged(data, pristine)
            check_output(result, expected)
            return result

        def checked(input, limit=7.0):
            nonlocal diagnosed
            import torch
            result = verify(input, limit)
            if not diagnosed:
                # Keep the original 1031-column probe and cover smaller legal
                # widths and neighbors of lane/tile boundaries independently.
                for data, control_limit in clamp_controls(input.device, input.dtype):
                    verify(data, control_limit)
                diagnosed = True
            return result

        module.swiglustep_and_mul = checked
        return module

    harness.load_module = load
    try:
        yield
    finally:
        harness.load_module = loader
        for module, original in reversed(patched):
            module.swiglustep_and_mul = original


def checked_benchmark(harness, benchmark, fn, **options):
    state = inspect.getclosurevars(fn).nonlocals
    module, data, limit = state['mod'], state['x'], state['limit']
    pristine = data.clone()
    expected = reference(harness, pristine, limit)
    original = module.swiglustep_and_mul
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
            raise AssertionError('Timed SwiGLU-step did not return an output')
        return captured

    module.swiglustep_and_mul = collect
    try:
        timed = harness._TimedRun()
        checked_samples = [0]
        def check_sample(output):
            check_output(output, expected)
            checked_samples[0] += 1
        timed.after_sample = check_sample
        # The public wrapper allocates its output on every call. Replay of a
        # captured graph omits that stage and can count unchecked calls.
        options = {**options, 'use_cuda_graph': False,
                   'fallback_reason': 'full_public_invocation'}
        ms, metadata = benchmark(measured, timed_run=timed, **options)
        if not timed.bound or checked_samples[0] != options['repetition']:
            raise AssertionError('Reported sample outputs were not all checked')
        unchanged(data, pristine)
        check_output(timed.outputs, expected)
        data.mul_(-3.)
        replay_input = data.clone()
        expected_replay = reference(harness, replay_input, limit)
        timed.outputs.fill_(float('nan'))
        replayed = timed.rerun()
        unchanged(data, replay_input)
        check_output(replayed, expected_replay)
        return ms, {**metadata, 'timed_output_checked': True,
                    'perturbed_input_replay_checked': True, 'source_buffers_unchanged': True,
                    'measured_samples_checked': checked_samples[0], }
    finally:
        module.swiglustep_and_mul = original
        data.copy_(pristine)


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
