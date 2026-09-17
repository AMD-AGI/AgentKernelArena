"""Validate all gated-normalization outputs and the actual timed invocation."""
from contextlib import contextmanager
import inspect

SYMBOL = 'layer_norm_gated_fwd'


def snapshots(inputs):
    return tuple(x.clone() if x is not None else None for x in inputs)


def unchanged(inputs, pristine):
    import torch
    for x, saved in zip(inputs, pristine):
        if x is not None and not torch.equal(x, saved):
            raise AssertionError('Gated normalization modified read-only input')


def reference(harness, inputs, options):
    import torch
    x, g, weight, bias = inputs
    y = harness.reference(x, g, weight=weight, bias=bias, **options).to(device=x.device, dtype=x.dtype)
    cpu = x.float().cpu()
    is_rms = options.get('is_rms_norm', True)
    mean = None if is_rms else cpu.mean(-1)
    centered = cpu if is_rms else cpu - mean[:, None]
    rstd = ((centered * centered).mean(-1) + options.get('eps', 1e-5)).rsqrt()
    return y, None if mean is None else mean.to(x.device), rstd.to(x.device)


def check_outputs(outputs, expected):
    import torch
    if not isinstance(outputs, tuple) or len(outputs) != 3:
        raise AssertionError('Gated normalization must return (y, mean, rstd)')
    for name, actual, wanted in zip(('y', 'mean', 'rstd'), outputs, expected):
        if wanted is None:
            if actual is not None:
                raise AssertionError('RMS normalization mean must be None')
            continue
        if not isinstance(actual, torch.Tensor) or (actual.shape != wanted.shape or
                actual.dtype != wanted.dtype or actual.device != wanted.device):
            raise AssertionError(f'Gated normalization {name} shape/dtype/device is invalid')
        if not torch.isfinite(actual).all():
            raise AssertionError(f'Gated normalization {name} must be finite')
        torch.testing.assert_close(actual, wanted, atol=1e-3, rtol=1e-3)


@contextmanager
def checked_modules(harness):
    load_original = harness.load_module
    patched = []

    def load():
        module = load_original()
        original = getattr(module, SYMBOL)
        patched.append((module, original))

        def checked(x, g, weight=None, bias=None, activation='swish', eps=1e-5, is_rms_norm=True):
            inputs = (x, g, weight, bias)
            pristine = snapshots(inputs)
            options = dict(activation=activation, eps=eps, is_rms_norm=is_rms_norm)
            expected = reference(harness, pristine, options)
            outputs = original(x, g, weight=weight, bias=bias, **options)
            unchanged(inputs, pristine)
            check_outputs(outputs, expected)
            return outputs

        setattr(module, SYMBOL, checked)
        return module

    harness.load_module = load
    try:
        yield
    finally:
        harness.load_module = load_original
        for module, original in reversed(patched):
            setattr(module, SYMBOL, original)


def checked_benchmark(harness, benchmark, fn, **options):
    c = inspect.getclosurevars(fn).nonlocals
    module, args, kwargs = (c[name] for name in ('mod', 'args', 'kwargs'))
    inputs = (*args, kwargs.get('weight'), kwargs.get('bias'))
    pristine = snapshots(inputs)
    ref_options = {k: v for k, v in kwargs.items() if k not in ('weight', 'bias')}
    expected = reference(harness, pristine, ref_options)
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
            raise AssertionError('Benchmark did not invoke gated normalization')
        return captured

    setattr(module, SYMBOL, collect)
    try:
        timed = harness._TimedRun()
        ms, metadata = benchmark(measured, timed_run=timed, **options)
        unchanged(inputs, pristine)
        check_outputs(timed.outputs, expected)
        for i, value in enumerate(inputs):
            if value is not None:
                value.mul_(-1.5).add_(0.25 * (i+1))
        replay_pristine = snapshots(inputs)
        replay_expected = reference(harness, replay_pristine, ref_options)
        for value in timed.outputs:
            if value is not None:
                value.fill_(float('nan'))
        replayed = timed.rerun()
        unchanged(inputs, replay_pristine)
        check_outputs(replayed, replay_expected)
        return ms, {**metadata, 'timed_output_checked': True,
                    'perturbed_input_replay_checked': True, 'source_buffers_unchanged': True}
    finally:
        setattr(module, SYMBOL, original)
        for value, saved in zip(inputs, pristine):
            if value is not None:
                value.copy_(saved)


def install(harness):
    correctness_original = harness.run_correctness
    performance_original = harness.run_performance

    def correctness(*args, **kwargs):
        with checked_modules(harness):
            return correctness_original(*args, **kwargs)

    def performance():
        benchmark = harness._benchmark_cuda_graph_or_events
        harness._benchmark_cuda_graph_or_events = lambda fn, **kwargs: checked_benchmark(harness, benchmark, fn, **kwargs)
        try:
            return performance_original()
        finally:
            harness._benchmark_cuda_graph_or_events = benchmark

    harness.run_correctness = correctness
    harness.run_performance = performance
