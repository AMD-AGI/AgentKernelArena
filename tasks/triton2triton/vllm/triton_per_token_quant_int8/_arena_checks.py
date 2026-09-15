"""Verify both quantization outputs and the actual timed invocation."""
from contextlib import contextmanager
import inspect

SYMBOL = 'per_token_quant_int8'


def unchanged(value, original):
    import torch
    if not torch.equal(value, original):
        raise AssertionError('INT8 quantization modified its read-only input')


def reference(harness, x, options):
    flat = x.reshape(-1, x.shape[-1])
    quant, scales = harness.reference_per_token_quant_int8(flat)
    quant = quant.reshape(x.shape).to(x.device)
    scales = scales.reshape(*x.shape[:-1], 1).to(x.device)
    return quant, scales


def check_outputs(result, expected):
    import torch
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise AssertionError('Quantization must return quantized values and scales')
    for actual, answer in zip(result, expected):
        if not isinstance(actual, torch.Tensor) or (actual.shape != answer.shape or
                actual.dtype != answer.dtype or actual.device != answer.device):
            raise AssertionError('Quantization output shape/dtype/device is invalid')
    quant, scales = result
    if not torch.isfinite(scales).all() or not (scales > 0).all():
        raise AssertionError('Quantization scales must be positive and finite')
    # Retain the original one-code rounding allowance and scale thresholds.
    torch.testing.assert_close(quant.float(), expected[0].float(), atol=1., rtol=0.)
    torch.testing.assert_close(scales, expected[1], atol=1e-4, rtol=1e-3)


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
            check_outputs(result, expected)
            return result

        def checked(x):
            nonlocal diagnosed
            import torch
            result = verify(x, {})
            if not diagnosed:
                # Unscored higher-rank input, non-power-of-two width, zero row,
                # and a strided 2-D input accepted by the public wrapper.
                data = ((torch.arange(2*3*17, device=x.device).reshape(2, 3, 17) % 11 - 5) / 4).to(x.dtype)
                data[0, 0].zero_()
                verify(data, {})
                strided = data.reshape(6, 17).t()
                verify(strided, {})
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
        timed.outputs[0].fill_(-128)
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
