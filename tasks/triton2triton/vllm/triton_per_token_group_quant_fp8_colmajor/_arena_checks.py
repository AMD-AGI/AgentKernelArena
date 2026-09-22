"""Check FP8 representation, scale layout, and actual quantization replay."""
from contextlib import contextmanager
import inspect
import math

SYMBOL = 'per_token_group_quant_fp8_colmajor'
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
    group_size, eps = options['group_size'], options.get('eps', 1e-10)
    if not options.get('use_ue8m0', False):
        # Keep the original independent CPU oracle for every scored case.
        quant, scales = getattr(harness, 'reference_'+SYMBOL)(x, group_size, dtype, eps)
    else:
        # Public optional power-of-two scale path; unscored diagnostic only.
        data = x.cpu().float()
        maximum = 240. if dtype == torch.float8_e4m3fnuz else torch.finfo(dtype).max
        quant = torch.empty_like(data)
        scales = torch.empty(data.shape[0], data.shape[1]//group_size, dtype=torch.float32)
        for row in range(data.shape[0]):
            for group in range(scales.shape[1]):
                values = data[row, group*group_size:(group+1)*group_size]
                raw_scale = max(values.abs().max().item(), eps)/maximum
                scale = math.ldexp(1., math.ceil(math.log2(raw_scale)))
                scales[row, group] = scale
                quant[row, group*group_size:(group+1)*group_size] = (values/scale).clamp(-maximum, maximum)
        quant = quant.to(dtype)
    return quant.to(x.device), scales.to(x.device)


def check_outputs(result, expected, group_size):
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
    # Preserve both original numerical gates; no direct exact-FP8 comparison.
    torch.testing.assert_close(scales, expected[1], atol=1e-5, rtol=1e-3)
    dequant = quant.float() * scales.repeat_interleave(group_size, dim=-1)
    answer = expected[0].float() * expected[1].repeat_interleave(group_size, dim=-1)
    torch.testing.assert_close(dequant, answer, atol=1e-1, rtol=1e-1)


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
            check_outputs(result, expected, options['group_size'])
            return result

        def checked(x, group_size, eps=1e-10, dtype=None, use_ue8m0=False):
            nonlocal diagnosed
            import torch
            options = dict(group_size=group_size, eps=eps, use_ue8m0=use_ue8m0)
            if dtype is not None:
                options['dtype'] = dtype
            result = verify(x, options)
            if not diagnosed:
                # Both public wrappers accept padded 2-D rows with unit last stride.
                data = ((torch.arange(6*34, device=x.device).reshape(6, 34) % 11 - 5) / 4).to(x.dtype)[::2]
                data[0].zero_()
                data[1, :17].zero_()
                verify(data, dict(group_size=17, eps=0.5))
                verify(data, dict(group_size=17, eps=0.5, use_ue8m0=True))
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
    arguments = dict(group_size=state['group_size'])
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
        check_outputs(timed.outputs, expected, arguments['group_size'])
        x.mul_(-0.5)
        replay_input = x.clone()
        replay_expected = reference(harness, replay_input, arguments)
        # FP8 CPU/GPU fill support differs: copy converted poison without altering timing.
        import torch
        timed.outputs[0].copy_(torch.full_like(x, float('nan')).to(timed.outputs[0].dtype))
        timed.outputs[1].fill_(float('nan'))
        replayed = timed.rerun()
        unchanged(x, replay_input)
        check_outputs(replayed, replay_expected, arguments['group_size'])
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
