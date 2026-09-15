"""Validate both GDN gate outputs and the actual timed invocation."""
from contextlib import contextmanager
import inspect

SYMBOL = 'fused_gdn_gating'


def snapshots(inputs):
    return tuple(x.clone() if x is not None else None for x in inputs)


def unchanged(inputs, pristine):
    import torch
    for x, saved in zip(inputs, pristine):
        if x is not None and not torch.equal(x, saved):
            raise AssertionError('GDN gating modified read-only input')


def reference(harness, inputs, options):
    A_log, a, b, dt_bias = inputs
    cpu = tuple(x.cpu() for x in inputs)
    g, beta_output = harness.reference(*cpu, **options)
    return g.to(a.device), beta_output.to(b.device)


def check_outputs(outputs, expected):
    import torch
    if not isinstance(outputs, tuple) or len(outputs) != 2:
        raise AssertionError('GDN gating must return (g, beta_output)')
    for name, actual, wanted in zip(('g', 'beta_output'), outputs, expected):
        if not isinstance(actual, torch.Tensor) or (actual.shape != wanted.shape or
                actual.dtype != wanted.dtype or actual.device != wanted.device):
            raise AssertionError(f'GDN gating {name} shape/dtype/device is invalid')
        if not torch.isfinite(actual).all():
            raise AssertionError(f'GDN gating {name} must be finite')
        torch.testing.assert_close(actual, wanted, atol=1e-3, rtol=1e-3)


@contextmanager
def checked_modules(harness):
    load_original = harness.load_module
    patched = []

    def load():
        module = load_original()
        original = getattr(module, SYMBOL)
        patched.append((module, original))

        diagnosed = False

        def verify(inputs, options):
            pristine = snapshots(inputs)
            expected = reference(harness, pristine, options)
            outputs = original(*inputs, **options)
            unchanged(inputs, pristine)
            check_outputs(outputs, expected)
            return outputs

        def checked(A_log, a, b, dt_bias, beta=1.0, threshold=20.0):
            nonlocal diagnosed
            import torch
            outputs = verify((A_log, a, b, dt_bias), dict(beta=beta, threshold=threshold))
            if not diagnosed:
                # Unscored masked-tail/non-default softplus diagnostic. Original
                # scored shapes, FP16 inputs, defaults and seed remain unchanged.
                values = torch.tensor([-100., -4., 0., 6., 100., -2., 2., -20., 20., .5, -.5],
                                      device=a.device, dtype=a.dtype)
                da = values.repeat(2, 1)
                db = values.flip(0).repeat(2, 1).to(b.dtype)
                dA = torch.linspace(-.5, .5, 11, device=A_log.device, dtype=A_log.dtype)
                d_bias = torch.linspace(-.1, .2, 11, device=dt_bias.device, dtype=dt_bias.dtype)
                verify((dA, da, db, d_bias), dict(beta=.5, threshold=2.0))
                diagnosed = True
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
    module, inputs = (c[name] for name in ('mod', 'inputs'))
    pristine = snapshots(inputs)
    ref_options = {}
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
            raise AssertionError('Benchmark did not invoke GDN gating')
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
