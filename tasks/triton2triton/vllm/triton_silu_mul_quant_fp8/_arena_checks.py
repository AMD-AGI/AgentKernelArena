"""Check FP8 representation, scale layout, and actual quantization replay."""
from contextlib import contextmanager
import inspect
import math

SYMBOL = 'silu_mul_per_token_group_quant_fp8_colmajor'
COLUMN_MAJOR = True


def platform_dtype(device):
    """Match the original platform selection without trusting candidate helpers."""
    import torch
    if hasattr(torch, 'float8_e4m3fnuz'):
        try:
            torch.zeros(1, device=device, dtype=torch.float8_e4m3fnuz)
            return torch.float8_e4m3fnuz
        except (RuntimeError, TypeError):
            pass
    return torch.float8_e4m3fn


def unchanged(value, original):
    import torch
    if not torch.equal(value, original):
        raise AssertionError('FP8 quantization modified its read-only input')


def reference(harness, x, options):
    import torch
    dtype = platform_dtype(x.device)
    eps = options.get('eps', 1e-10)
    if eps == 1e-10 and not options.get('use_ue8m0', False):
        # Preserve the existing FP32 SiLU/mul oracle for all scored cases.
        quant, scales = harness.reference_silu_mul_quant_fp8(x, dtype)
    else:
        data = x.cpu().float()
        gate, up = data.chunk(2, dim=-1)
        values = gate/(1.+torch.exp(-gate))*up
        maximum = 240. if dtype == torch.float8_e4m3fnuz else torch.finfo(dtype).max
        quant = torch.empty_like(values)
        scales = torch.empty(values.shape[0], values.shape[1]//128, dtype=torch.float32)
        for row in range(values.shape[0]):
            for group in range(scales.shape[1]):
                part = values[row, group*128:(group+1)*128]
                scale = max(part.abs().max().item(), eps)/maximum
                if options.get('use_ue8m0', False):
                    scale = math.ldexp(1., math.ceil(math.log2(scale)))
                scales[row, group] = scale
                quant[row, group*128:(group+1)*128] = (part/scale).clamp(-maximum, maximum)
        quant = quant.to(dtype)
    return quant.to(x.device), scales.to(x.device)


def check_outputs(result, expected, *, use_ue8m0=False):
    import torch
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise AssertionError('FP8 quantization must return quantized values and scales')
    for actual, answer in zip(result, expected):
        if not isinstance(actual, torch.Tensor) or (actual.shape != answer.shape or
                actual.dtype != answer.dtype or actual.device != answer.device):
            raise AssertionError('FP8 output shape/dtype/device is invalid')
    quant, scales = result
    if COLUMN_MAJOR and scales.stride() != (1, scales.shape[0]):
        raise AssertionError('Scale output must use the declared column-major layout')
    if not torch.isfinite(quant.float()).all() or not torch.isfinite(scales).all() or not (scales > 0).all():
        raise AssertionError('FP8 values must be finite and scales must be positive finite')
    if use_ue8m0 and not (torch.frexp(scales)[0] == .5).all():
        raise AssertionError("UE8M0 mode requires power-of-two scales")
    # Preserve both original numerical gates; no direct exact-FP8 comparison.
    torch.testing.assert_close(scales, expected[1], atol=1e-2, rtol=1e-1)
    dequant = quant.float() * scales.repeat_interleave(128, dim=-1)
    answer = expected[0].float() * expected[1].repeat_interleave(128, dim=-1)
    torch.testing.assert_close(dequant, answer, atol=5e-1, rtol=1e-1)


@contextmanager
def checked_modules(harness):
    loader = harness.load_module
    patched = []

    def load():
        module = loader()
        original = getattr(module, SYMBOL)
        patched.append((module, original))
        diagnosed = False

        def verify(x, options):
            pristine = x.clone()
            expected = reference(harness, pristine, options)
            result = original(x, **options)
            unchanged(x, pristine)
            check_outputs(result, expected, use_ue8m0=options.get('use_ue8m0', False))
            if options.get('output') is not None and (result[0].data_ptr() != options['output'].data_ptr() or
                    result[0].stride() != options['output'].stride()):
                raise AssertionError('Supplied output must be populated and returned')
            return result

        def checked(input, output=None, use_ue8m0=False, eps=1e-10):
            nonlocal diagnosed
            import torch
            result = verify(input, dict(output=output, use_ue8m0=use_ue8m0, eps=eps))
            if not diagnosed:
                # Keep the public M%128 / N%256 constraints. Diagnostics add no timing rows.
                data = ((torch.arange(128*512, device=input.device).reshape(128, 512) % 11 - 5)/4).to(input.dtype)
                data[0].zero_()
                data[1, :128].zero_()
                target = torch.full((128, 256), float('nan'), device=input.device).to(platform_dtype(input.device))
                verify(data, dict(output=target, eps=16.))
                target.copy_(torch.full_like(target, float('nan'), dtype=torch.float32).to(target.dtype))
                verify(data, dict(output=target, eps=16., use_ue8m0=True))
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
    x, module = state['x'], state['mod']
    arguments = {}
    pristine = x.clone()
    expected = reference(harness, pristine, arguments)
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
            raise AssertionError('Benchmark did not produce quantized values and scales')
        return captured

    setattr(module, SYMBOL, collect)
    try:
        timed = harness._TimedRun()
        ms, metadata = benchmark(measured, timed_run=timed, **options)
        unchanged(x, pristine)
        check_outputs(timed.outputs, expected)
        x.mul_(-0.5)
        replay_input = x.clone()
        replay_expected = reference(harness, replay_input, arguments)
        # FP8 CPU/GPU fill support differs: copy converted poison without altering timing.
        import torch
        timed.outputs[0].copy_(torch.full_like(timed.outputs[0], float('nan'), dtype=torch.float32).to(timed.outputs[0].dtype))
        timed.outputs[1].fill_(float('nan'))
        replayed = timed.rerun()
        unchanged(x, replay_input)
        check_outputs(replayed, replay_expected)
        return ms, {**metadata, 'timed_output_checked': True,
                    'perturbed_input_replay_checked': True, 'source_buffers_unchanged': True}
    finally:
        setattr(module, SYMBOL, original)
        x.copy_(pristine)


def install(harness):
    correctness = harness.run_correctness
    performance = harness.run_performance

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

    harness.run_correctness = checked_correctness
    harness.run_performance = checked_performance
