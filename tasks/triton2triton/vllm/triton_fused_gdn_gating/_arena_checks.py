"""Validate both GDN gate outputs and the actual timed invocation."""
from contextlib import contextmanager
import inspect

SYMBOL = 'fused_gdn_gating'


def snapshots(inputs):
    return tuple(x.clone() if x is not None else None for x in inputs)


def _storage_bytes(x):
    import torch
    # Include padding and storage outside a sliced view, not just logical values.
    length = x.untyped_storage().nbytes() // x.element_size()
    return torch.as_strided(x, (length,), (1,), storage_offset=0).view(torch.uint8)


def readonly_state(inputs):
    return tuple(None if x is None else (
        tuple(x.shape), tuple(x.stride()), x.storage_offset(), x.data_ptr(),
        x.dtype, x.device, x.untyped_storage().nbytes(), _storage_bytes(x).clone(),
        x.untyped_storage()
    ) for x in inputs)


def unchanged(inputs, pristine, state=None):
    import torch
    for index, (x, saved) in enumerate(zip(inputs, pristine)):
        if x is None:
            continue
        if not torch.equal(x, saved):
            raise AssertionError('GDN gating modified read-only input')
        if state is not None:
            shape, strides, offset, pointer, dtype, device, nbytes, original_bytes, _ = state[index]
            if (tuple(x.shape), tuple(x.stride()), x.storage_offset(), x.data_ptr(),
                    x.dtype, x.device, x.untyped_storage().nbytes()) != (
                    shape, strides, offset, pointer, dtype, device, nbytes
            ) or not torch.equal(_storage_bytes(x), original_bytes):
                raise AssertionError('GDN gating modified read-only input storage or layout')


def restore_storage(inputs, state):
    for x, saved in zip(inputs, state):
        if x is None:
            continue
        shape, strides, offset, pointer, dtype, device, nbytes, original_bytes, storage = saved
        if (tuple(x.shape), tuple(x.stride()), x.storage_offset(), x.data_ptr()) != (
                shape, strides, offset, pointer):
            x.set_(storage, offset, shape, strides)
        _storage_bytes(x).copy_(original_bytes)


def strided_view(x):
    import torch
    padded = torch.empty((*x.shape, 2), device=x.device, dtype=x.dtype)
    padded[..., 0].copy_(x)
    padded[..., 1].fill_(7)
    view = padded[..., 0]
    if view.is_contiguous():
        raise AssertionError('GDN stride control was not strided')
    return view


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
            state = readonly_state(inputs)
            expected = reference(harness, pristine, options)
            try:
                outputs = original(*inputs, **options)
                unchanged(inputs, pristine, state)
                check_outputs(outputs, expected)
                return outputs
            finally:
                restore_storage(inputs, state)

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
                diagnostic = (dA, da, db, d_bias)
                options = dict(beta=.5, threshold=2.0)
                verify(diagnostic, options)
                # Independently stride each public operand. A flat-storage
                # reader can be wrong for any one while passing scored cases.
                for operand in range(len(diagnostic)):
                    inputs = list(diagnostic)
                    inputs[operand] = strided_view(inputs[operand])
                    verify(tuple(inputs), options)
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
    initial_state = readonly_state(inputs)
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

    prior_prepare = options.pop('prepare_fn', None)
    def prepare_sample():
        if prior_prepare is not None:
            prior_prepare()
        # Preparation limits each graph to the device work captured from one
        # public wrapper invocation. Once its output buffers are
        # captured, invalidate both before each sample's start event so a
        # replay that omits either device write cannot pass after_sample.
        if captured is not None:
            for value in captured:
                value.fill_(float('nan'))

    setattr(module, SYMBOL, collect)
    try:
        timed = harness._TimedRun()
        checked_samples = [0]
        def check_sample(output):
            unchanged(inputs, pristine, initial_state)
            check_outputs(output, expected)
            checked_samples[0] += 1
        timed.after_sample = check_sample
        ms, metadata = benchmark(measured, timed_run=timed,
                                 prepare_fn=prepare_sample, **options)
        if not timed.bound or checked_samples[0] != options['repetition']:
            raise AssertionError('Reported sample outputs were not all checked')
        if (metadata.get('benchmark_method') == 'cuda_graph' and
                metadata.get('benchmark_effective_repeats') != 1):
            raise AssertionError('Graph timing included unobserved wrapper calls')
        unchanged(inputs, pristine, initial_state)
        check_outputs(timed.outputs, expected)
        for i, value in enumerate(inputs):
            if value is not None:
                value.mul_(-1.5).add_(0.25 * (i+1))
        replay_pristine = snapshots(inputs)
        replay_state = readonly_state(inputs)
        replay_expected = reference(harness, replay_pristine, ref_options)
        for value in timed.outputs:
            if value is not None:
                value.fill_(float('nan'))
        replayed = timed.rerun()
        unchanged(inputs, replay_pristine, replay_state)
        check_outputs(replayed, replay_expected)
        return ms, {**metadata, 'timed_output_checked': True,
                    'perturbed_input_replay_checked': True, 'source_buffers_unchanged': True,
                    'measured_samples_checked': checked_samples[0], }
    finally:
        setattr(module, SYMBOL, original)
        restore_storage(inputs, initial_state)


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
