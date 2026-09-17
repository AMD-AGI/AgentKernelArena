"""Protected AWQ output, input preservation and measured-invocation checks."""
from contextlib import contextmanager
import inspect


def snapshot(values):
    return tuple(value.clone() for value in values)


def unchanged(values, originals):
    import torch
    for value, original in zip(values, originals):
        if not torch.equal(value, original):
            raise AssertionError('AWQ GEMM modified a read-only input')


def reference(harness, values):
    a, weight, scales, zeros = values
    group_size = weight.shape[0] // zeros.shape[0]
    return harness.reference_awq_gemm(a, weight, scales, zeros, group_size).to(a.device)


def check_output(output, expected):
    import torch
    if not isinstance(output, torch.Tensor) or (output.shape != expected.shape or
            output.dtype != expected.dtype or output.device != expected.device):
        raise AssertionError('AWQ GEMM output shape/dtype/device is invalid')
    if not torch.isfinite(output).all():
        raise AssertionError('AWQ GEMM output must be finite')
    # Preserve the actual original harness gate, including on timed replay.
    torch.testing.assert_close(output, expected, atol=1e-1, rtol=1e-1)


@contextmanager
def checked_modules(harness):
    loader = harness.load_module
    patched = []

    def load():
        module = loader()
        original = module.awq_gemm_triton
        patched.append((module, original))
        diagnosed = False

        def verify(a, weight, scales, zeros, split, **kwargs):
            values = (a, weight, scales, zeros)
            originals = snapshot(values)
            expected = reference(harness, originals)
            try:
                output = original(*values, split, **kwargs)
                unchanged(values, originals)
                check_output(output, expected)
                return output
            finally:
                for value, saved in zip(values, originals):
                    value.copy_(saved)

        def checked(input, qweight, scales, qzeros, split_k_iters, **kwargs):
            nonlocal diagnosed
            import torch
            output = verify(input, qweight, scales, qzeros, split_k_iters, **kwargs)
            if not diagnosed:
                # Unscored supported row/column tails and actual split-K paths.
                # Deterministic construction does not change the original RNG.
                a = ((torch.arange(35*64, device=input.device)%13)-6).reshape(35,64).to(input.dtype)*.125
                weight = ((torch.arange(64*3, device=input.device)*0x12345+0x13579B)&0x7fffffff).reshape(64,3).to(qweight.dtype)
                scale = torch.full((2,24), .0625, device=scales.device, dtype=scales.dtype)
                zero = torch.full((2,3), 0x22222222, device=qzeros.device, dtype=qzeros.dtype)
                for split in (1, 2, 4):
                    verify(a, weight, scale, zero, split)
                diagnosed = True
            return output

        module.awq_gemm_triton = checked
        return module

    harness.load_module = load
    try:
        yield
    finally:
        harness.load_module = loader
        for module, original in reversed(patched):
            module.awq_gemm_triton = original


def checked_benchmark(harness, benchmark, fn, **options):
    state = inspect.getclosurevars(fn).nonlocals
    module = state['mod']
    values = tuple(state[key] for key in ('input_tensor', 'qweight', 'scales', 'qzeros'))
    originals = snapshot(values)
    expected = reference(harness, originals)
    original = module.awq_gemm_triton
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
            raise AssertionError('Timed AWQ GEMM did not return an output')
        return captured

    module.awq_gemm_triton = collect
    try:
        timed = harness._TimedRun()
        ms, metadata = benchmark(measured, timed_run=timed, **options)
        unchanged(values, originals)
        check_output(timed.outputs, expected)
        a, weight, scales, zeros = values
        a.mul_(-.5)
        weight.bitwise_xor_(0x11111111)
        scales.mul_(.5).add_(.015625)
        zeros.bitwise_xor_(0x22222222)
        replay_inputs = snapshot(values)
        expected_replay = reference(harness, replay_inputs)
        timed.outputs.fill_(float('nan'))
        replayed = timed.rerun()
        unchanged(values, replay_inputs)
        check_output(replayed, expected_replay)
        return ms, {**metadata, 'timed_output_checked': True,
                    'perturbed_input_replay_checked': True,
                    'source_buffers_unchanged': True}
    finally:
        module.awq_gemm_triton = original
        for value, saved in zip(values, originals):
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
