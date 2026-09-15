"""Byte-exact layout validation, including output from the measured replay."""
from contextlib import contextmanager
import inspect


def bytes_of(value):
    import torch
    return value.contiguous().view(torch.uint8)


def unchanged(value, original):
    import torch
    if not torch.equal(bytes_of(value), bytes_of(original)):
        raise AssertionError('Scale swizzle modified its read-only input')


def reference(harness, data):
    import torch
    rows, cols = data.shape
    padded = torch.zeros(((rows+127)//128*128, (cols+3)//4*4), dtype=torch.uint8)
    padded[:rows, :cols] = bytes_of(data).cpu()
    return harness.reference_scale_swizzle(padded).to(data.device).view(data.dtype)


def check_output(value, expected):
    import torch
    if not isinstance(value, torch.Tensor) or (value.shape != expected.shape or
            value.dtype != expected.dtype or value.device != expected.device):
        raise AssertionError('Scale swizzle output shape/dtype/device is invalid')
    # This is a byte permutation; FP8 encodings, including NaN, remain bytes.
    if not torch.equal(bytes_of(value), bytes_of(expected)):
        raise AssertionError('Scale swizzle output differs from the byte-exact oracle')


@contextmanager
def checked_modules(harness):
    loader = harness.load_module
    patched = []

    def load():
        module = loader()
        original = module.triton_mx_block_rearrange
        patched.append((module, original))
        diagnosed = False

        def verify(data):
            pristine = data.clone()
            expected = reference(harness, pristine)
            result = original(data)
            unchanged(data, pristine)
            check_output(result, expected)
            return result

        def checked(scale_tensor):
            nonlocal diagnosed
            import torch
            result = verify(scale_tensor)
            if not diagnosed:
                # Cover zero padding in both dimensions and every byte value.
                raw = (torch.arange(129*5, device=scale_tensor.device)%256).to(torch.uint8).reshape(129, 5)
                verify(raw)
                verify(raw.view(torch.int8))
                if hasattr(torch, 'float8_e4m3fnuz'):
                    verify(raw.view(torch.float8_e4m3fnuz))
                diagnosed = True
            return result

        module.triton_mx_block_rearrange = checked
        return module

    harness.load_module = load
    try:
        yield
    finally:
        harness.load_module = loader
        for module, original in reversed(patched):
            module.triton_mx_block_rearrange = original


def checked_benchmark(harness, benchmark, fn, **options):
    state = inspect.getclosurevars(fn).nonlocals
    module, data = state['mod'], state['data']
    pristine = data.clone()
    expected = reference(harness, pristine)
    original = module.triton_mx_block_rearrange
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
            raise AssertionError('Timed scale swizzle did not return an output')
        return captured

    module.triton_mx_block_rearrange = collect
    try:
        timed = harness._TimedRun()
        ms, metadata = benchmark(measured, timed_run=timed, **options)
        unchanged(data, pristine)
        check_output(timed.outputs, expected)
        bytes_of(data).bitwise_xor_(85)
        replay_input = data.clone()
        expected_replay = reference(harness, replay_input)
        # Every output byte starts wrong, including any padded output region.
        timed.outputs.copy_(bytes_of(expected_replay).bitwise_xor(255).view(expected_replay.dtype))
        replayed = timed.rerun()
        unchanged(data, replay_input)
        check_output(replayed, expected_replay)
        return ms, {**metadata, 'timed_output_checked': True,
                    'perturbed_input_replay_checked': True, 'source_buffers_unchanged': True}
    finally:
        module.triton_mx_block_rearrange = original
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
