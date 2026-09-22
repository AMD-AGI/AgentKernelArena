"""Preserve scaled GEMM gates while checking output metadata and measured replay."""
from contextlib import contextmanager
import inspect


def snapshot(values):
    return [v.clone() if v is not None else None for v in values]


def unchanged(values, originals):
    import torch
    for value, original in zip(values, originals):
        if value is not None and not torch.equal(value, original):
            raise AssertionError('Scaled GEMM modified a read-only operand, scale or bias')


def reference(harness, values, dtype):
    a, b, sa, sb, bias = values
    return harness.reference_scaled_mm(a, b, sa.reshape(-1, 1), sb.reshape(-1, 1), dtype, bias=bias)


def check_output(value, expected):
    import torch
    if not isinstance(value, torch.Tensor) or (value.shape != expected.shape or
            value.dtype != expected.dtype or value.device != expected.device):
        raise AssertionError('Scaled GEMM output shape/dtype/device is invalid')
    if not torch.isfinite(value).all():
        raise AssertionError('Scaled GEMM output must be finite')
    torch.testing.assert_close(value, expected, atol=1e-2, rtol=1e-2)


@contextmanager
def checked_modules(harness):
    loader = harness.load_module
    patched = []

    def load():
        module = loader()
        original = module.triton_scaled_mm
        patched.append((module, original))
        diagnosed = False

        def verify(a, b, sa, sb, dtype, bias=None, **options):
            inputs = [a, b, sa, sb, bias]
            pristine = snapshot(inputs)
            expected = reference(harness, pristine, dtype)
            result = original(a, b, sa, sb, dtype, bias=bias, **options)
            unchanged(inputs, pristine)
            check_output(result, expected)
            return result

        def checked(input, weight, scale_a, scale_b, out_dtype, bias=None,
                    block_size_m=32, block_size_n=32, block_size_k=32, use_heuristic=True):
            nonlocal diagnosed
            import torch
            result = verify(input, weight, scale_a, scale_b, out_dtype, bias,
                            block_size_m=block_size_m, block_size_n=block_size_n,
                            block_size_k=block_size_k, use_heuristic=use_heuristic)
            if not diagnosed:
                # Legal weak-contiguous layouts; all M/N/K have partial blocks.
                a = ((torch.arange(34*35, device=input.device).reshape(34, 35)%7-3)/8).to(input.dtype)[::2]
                b = ((torch.arange(19*35, device=input.device).reshape(19, 35)%11-5)/8).to(input.dtype).t()
                sa = torch.tensor([.75], dtype=torch.float32, device=input.device)
                sb = torch.linspace(.5, 1.5, 19, dtype=torch.float32, device=input.device)
                bias = torch.linspace(-.25, .25, 19, dtype=input.dtype, device=input.device)
                # Scalar A / per-channel B is absent from the original table;
                # it stays an unscored check, including explicit tile selection.
                verify(a, b, sa, sb, out_dtype, bias, use_heuristic=False)
                verify(a, b, sa, sb, torch.float32, bias)
                diagnosed = True
            return result

        module.triton_scaled_mm = checked
        return module

    harness.load_module = load
    try:
        yield
    finally:
        harness.load_module = loader
        for module, original in reversed(patched):
            module.triton_scaled_mm = original


def checked_benchmark(harness, benchmark, fn, **options):
    state = inspect.getclosurevars(fn).nonlocals
    module = state['mod']
    inputs = [state[k] for k in ('input_t', 'weight', 'scale_a', 'scale_b', 'bias')]
    pristine = snapshot(inputs)
    expected = reference(harness, pristine, state['dtype'])
    original = module.triton_scaled_mm
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
            raise AssertionError('Timed scaled GEMM did not return an output')
        return captured

    module.triton_scaled_mm = collect
    try:
        timed = harness._TimedRun()
        ms, metadata = benchmark(measured, timed_run=timed, **options)
        unchanged(inputs, pristine)
        check_output(timed.outputs, expected)
        inputs[0].mul_(-.5)
        inputs[1].mul_(.75)
        inputs[2].mul_(1.25)
        inputs[3].mul_(.5)
        if inputs[4] is not None: inputs[4].add_(.375)
        replay_inputs = snapshot(inputs)
        expected_replay = reference(harness, replay_inputs, state['dtype'])
        timed.outputs.fill_(float('nan'))
        replayed = timed.rerun()
        unchanged(inputs, replay_inputs)
        check_output(replayed, expected_replay)
        return ms, {**metadata, 'timed_output_checked': True,
                    'perturbed_input_replay_checked': True, 'source_buffers_unchanged': True}
    finally:
        module.triton_scaled_mm = original
        for value, saved in zip(inputs, pristine):
            if value is not None: value.copy_(saved)


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
