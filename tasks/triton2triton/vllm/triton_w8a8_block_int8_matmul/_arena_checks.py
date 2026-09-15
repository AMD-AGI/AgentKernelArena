"""Full block-quantized GEMM outputs and the exact measured graph replay."""
from contextlib import contextmanager
import inspect

SYMBOL = 'w8a8_block_int8_matmul'
REFERENCE = 'reference_w8a8_block_int8_matmul'
FP8 = False


def snapshot(values):
    return [v.clone() for v in values]


def unchanged(values, originals):
    import torch
    for value, original in zip(values, originals):
        # FP8 equality kernels are unavailable on some supported runtimes.
        if not torch.equal(value.contiguous().reshape(-1).view(torch.uint8),
                           original.contiguous().reshape(-1).view(torch.uint8)):
            raise AssertionError('Block GEMM modified a read-only operand or scale')


def check_input_dtypes(values):
    import torch
    a, b, sa, sb = values
    dtype = torch.int8
    if FP8:
        dtype = torch.float8_e4m3fn
        if hasattr(torch, 'float8_e4m3fnuz'):
            try:
                torch.zeros(1, dtype=torch.float8_e4m3fnuz, device=a.device)
                dtype = torch.float8_e4m3fnuz
            except (RuntimeError, TypeError):
                pass
    if a.dtype != dtype or b.dtype != dtype or sa.dtype != torch.float32 or sb.dtype != torch.float32:
        raise AssertionError('Block GEMM input dtypes must follow the protected platform contract')


def reference(harness, values, blocks, dtype):
    a, b, sa, sb = values
    # The original oracle dequantizes blocks independently on CPU. Flatten only
    # leading rows for its 2-D interface; restore the public output shape.
    expected = getattr(harness, REFERENCE)(a.reshape(-1, a.shape[-1]), b,
        sa.reshape(-1, sa.shape[-1]), sb, blocks, dtype)
    return expected.reshape(*a.shape[:-1], b.shape[0]).to(a.device)


def check_output(value, expected):
    import torch
    if not isinstance(value, torch.Tensor) or (value.shape != expected.shape or
            value.dtype != expected.dtype or value.device != expected.device):
        raise AssertionError('Block GEMM output shape/dtype/device is invalid')
    if not torch.isfinite(value).all():
        raise AssertionError('Block GEMM output must be finite')
    # Preserve the original tolerance, including for the original, larger
    # performance operands/scales. A real overflow or mismatch must fail.
    torch.testing.assert_close(value, expected, atol=1e-1, rtol=1e-1)


@contextmanager
def checked_modules(harness):
    loader = harness.load_module
    patched = []

    def load():
        module = loader()
        original = getattr(module, SYMBOL)
        patched.append((module, original))
        diagnosed = False

        def verify(a, b, sa, sb, blocks, dtype):
            inputs = [a, b, sa, sb]
            check_input_dtypes(inputs)
            pristine = snapshot(inputs)
            expected = reference(harness, pristine, blocks, dtype)
            result = original(a, b, sa, sb, blocks, output_dtype=dtype)
            unchanged(inputs, pristine)
            check_output(result, expected)
            return result

        def checked(A, B, As, Bs, block_size, output_dtype=None):
            nonlocal diagnosed
            import torch
            dtype = torch.float16 if output_dtype is None else output_dtype
            result = verify(A, B, As, Bs, block_size, dtype)
            if not diagnosed:
                # Contiguous multi-dimensional A, two partial scale blocks in
                # K/N, and supported non-default output dtype/block size.
                a = ((torch.arange(3*67, device=A.device).reshape(1, 3, 67)%7)-3).to(A.dtype)
                b = ((torch.arange(65*67, device=B.device).reshape(65, 67)%5)-2).to(B.dtype)
                if FP8:
                    # B permits strides, unlike the INT8 wrapper's contiguous B.
                    b = b.t().contiguous().t()
                sa = torch.tensor([[[.5, 1.25], [.75, .25], [1.5, .5]]], device=A.device)
                sb = torch.tensor([[.75, 1.5], [.25, .5]], device=A.device)
                verify(a, b, sa, sb, [64, 64], torch.float32)
                verify(a, b, sa, sb, [64, 64], torch.bfloat16)
                verify(torch.zeros_like(a), b, sa, sb, [64, 64], dtype)
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
    import torch
    state = inspect.getclosurevars(fn).nonlocals
    module = state['mod']
    inputs = [state[name] for name in ('A', 'B', 'As', 'Bs')]
    check_input_dtypes(inputs)
    blocks = [state['block_n'], state['block_k']]
    pristine = snapshot(inputs)
    expected = reference(harness, pristine, blocks, torch.float16)
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
            raise AssertionError('Timed block GEMM did not return an output')
        return captured

    setattr(module, SYMBOL, collect)
    try:
        timed = harness._TimedRun()
        ms, metadata = benchmark(measured, timed_run=timed, **options)
        unchanged(inputs, pristine)
        check_output(timed.outputs, expected)
        # Change all four operands/scales outside timing. Use representable
        # values and smaller scales; retain the scored input distribution.
        inputs[0].copy_((-pristine[0].float()*.5).to(inputs[0].dtype))
        inputs[1].copy_((pristine[1].float()*.5).to(inputs[1].dtype))
        inputs[2].mul_(.5)
        inputs[3].mul_(.75)
        replay_inputs = snapshot(inputs)
        expected_replay = reference(harness, replay_inputs, blocks, torch.float16)
        timed.outputs.fill_(float('nan'))
        replayed = timed.rerun()
        unchanged(inputs, replay_inputs)
        check_output(replayed, expected_replay)
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
