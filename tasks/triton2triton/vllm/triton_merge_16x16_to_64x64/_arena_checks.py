"""Check 64-column merged triangular inverses and exact measured replay."""
from contextlib import contextmanager
import inspect


def unchanged(value, original):
    import torch
    if not torch.equal(value, original):
        raise AssertionError('Merged triangular inverse modified its read-only input')


def reference(harness, data):
    return harness.reference(data).to(data.device)


def check_output(value, expected):
    import torch
    if not isinstance(value, torch.Tensor) or (value.shape != expected.shape or
            value.dtype != expected.dtype or value.device != expected.device):
        raise AssertionError('Merged triangular inverse output shape/dtype/device is invalid')
    if not torch.isfinite(value).all():
        raise AssertionError('Merged triangular inverse output must be finite')
    torch.testing.assert_close(value, expected, atol=1e-2, rtol=1e-2)


@contextmanager
def checked_modules(harness):
    loader = harness.load_module
    patched = []

    def load():
        module = loader()
        original = module.merge_16x16_to_64x64
        patched.append((module, original))
        diagnosed = False

        def verify(data):
            pristine = data.clone()
            expected = reference(harness, pristine)
            try:
                result = original(data)
                unchanged(data, pristine)
                check_output(result, expected)
                return result
            finally:
                data.copy_(pristine)

        def checked(A):
            nonlocal diagnosed
            import torch
            result = verify(A)
            if not diagnosed:
                data = ((torch.arange(67*3*64, device=A.device).reshape(1,67,3,64)%7)-3).to(A.dtype)*.05
                rows = torch.arange(67, device=A.device)%64
                cols = torch.arange(64, device=A.device)
                data *= (rows[:,None] > cols[None,:])[None,:,None,:]
                # Full first tile and three-row final tile, plus identity blocks.
                verify(data)
                verify(torch.zeros_like(data))
                diagnosed = True
            return result

        module.merge_16x16_to_64x64 = checked
        return module

    harness.load_module = load
    try:
        yield
    finally:
        harness.load_module = loader
        for module, original in reversed(patched):
            module.merge_16x16_to_64x64 = original


def checked_benchmark(harness, benchmark, fn, **options):
    state = inspect.getclosurevars(fn).nonlocals
    module, data = state['mod'], state['args'][0]
    pristine = data.clone()
    expected = reference(harness, pristine)
    original = module.merge_16x16_to_64x64
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
            raise AssertionError('Timed merged triangular inverse did not return an output')
        return captured

    module.merge_16x16_to_64x64 = collect
    try:
        timed = harness._TimedRun()
        ms, metadata = benchmark(measured, timed_run=timed, **options)
        unchanged(data, pristine)
        check_output(timed.outputs, expected)
        data.mul_(-.5)
        replay_input = data.clone()
        expected_replay = reference(harness, replay_input)
        timed.outputs.fill_(float('nan'))
        replayed = timed.rerun()
        unchanged(data, replay_input)
        check_output(replayed, expected_replay)
        return ms, {**metadata, 'timed_output_checked': True,
                    'perturbed_input_replay_checked': True, 'source_buffers_unchanged': True}
    finally:
        module.merge_16x16_to_64x64 = original
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
