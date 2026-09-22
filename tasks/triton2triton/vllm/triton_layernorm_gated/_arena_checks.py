"""Validate grouped gated normalization outputs and the actual timed invocation."""
from contextlib import contextmanager
import inspect

SYMBOL = 'layer_norm_fwd'


def snapshots(inputs):
    return tuple(x.clone() if x is not None else None for x in inputs)


def unchanged(inputs, pristine):
    import torch
    for x, saved in zip(inputs, pristine):
        if x is not None and not torch.equal(x, saved):
            raise AssertionError('LayerNorm modified read-only input')


def reference(harness, inputs, options):
    import torch
    x, weight, bias, z = inputs
    width = options.get('group_size') or x.shape[-1]
    eps = options.get('eps', 1e-5)
    rms = options.get('is_rms_norm', False)
    before_gate = options.get('norm_before_gate', True)
    ys, means, rstds = [], [], []
    for start in range(0, x.shape[-1], width):
        part = slice(start, start + width)
        xc = x[:, part].float().cpu()
        wc = weight[part].float().cpu()
        bc = None if bias is None else bias[part].float().cpu()
        zc = None if z is None else z[:, part].float().cpu()
        ys.append(harness.reference(xc, wc, bc, eps, zc, rms, before_gate))
        if zc is not None and not before_gate:
            xc = xc * zc * torch.sigmoid(zc)
        mean = None if rms else xc.mean(-1)
        centered = xc if rms else xc - mean[:, None]
        if mean is not None: means.append(mean)
        rstds.append(((centered * centered).mean(-1) + eps).rsqrt())
    return (torch.cat(ys, dim=-1).to(device=x.device, dtype=x.dtype),
            None if rms else torch.cat(means).to(x.device),
            torch.cat(rstds).to(x.device))


def check_outputs(outputs, expected):
    import torch
    if not isinstance(outputs, tuple) or len(outputs) != 3:
        raise AssertionError('LayerNorm must return (y, mean, rstd)')
    for name, actual, wanted in zip(('y', 'mean', 'rstd'), outputs, expected):
        if wanted is None:
            if actual is not None:
                raise AssertionError('RMS normalization mean must be None')
            continue
        if not isinstance(actual, torch.Tensor) or (actual.shape != wanted.shape or
                actual.dtype != wanted.dtype or actual.device != wanted.device):
            raise AssertionError(f'LayerNorm {name} shape/dtype/device is invalid')
        if not torch.isfinite(actual).all():
            raise AssertionError(f'LayerNorm {name} must be finite')
        torch.testing.assert_close(actual, wanted, atol=1e-2, rtol=1e-2)


@contextmanager
def checked_modules(harness):
    load_original = harness.load_module
    patched = []

    def load():
        module = load_original()
        original = getattr(module, SYMBOL)
        patched.append((module, original))

        diagnosed = False

        def verify(inputs, options, out=None):
            x, weight, bias, z = inputs
            pristine = snapshots(inputs)
            expected = reference(harness, pristine, options)
            outputs = original(x, weight, bias, z=z, out=out, **options)
            # An explicitly supplied out=x requests in-place output. Preserve
            # that existing API while checking every other input as read-only.
            for value, saved in zip(inputs, pristine):
                requested_output = (value is not None and out is not None and
                    value.data_ptr() == out.data_ptr() and value.shape == out.shape and
                    value.dtype == out.dtype and value.stride() == out.stride())
                if not requested_output:
                    unchanged((value,), (saved,))
            check_outputs(outputs, expected)
            if out is not None:
                if outputs[0].data_ptr() != out.data_ptr():
                    raise AssertionError('Provided output buffer must receive and return the result')
                check_outputs((out, outputs[1], outputs[2]), expected)
            return outputs

        def checked(x, weight, bias, eps, z=None, out=None, group_size=None,
                    norm_before_gate=True, is_rms_norm=False):
            nonlocal diagnosed
            import torch
            options = dict(eps=eps, group_size=group_size, norm_before_gate=norm_before_gate,
                           is_rms_norm=is_rms_norm)
            outputs = verify((x, weight, bias, z), options, out)
            if not diagnosed:
                # Existing public grouped/out-buffer API, both advertised gate
                # orders, and a masked tail; all extra checks remain unscored.
                dx = (torch.arange(68, device=x.device, dtype=torch.float32).reshape(2, 34) / 13 - 2).to(x.dtype)
                dw = torch.linspace(-2.25, 1.25, 34, device=x.device, dtype=weight.dtype)
                db = torch.linspace(.2, 1., 34, device=x.device, dtype=x.dtype)
                dz = (torch.arange(68, device=x.device, dtype=torch.float32).reshape(2, 34) / 23 - .8).to(x.dtype)
                for rms in (False, True):
                    for before_gate in (False, True):
                        supplied = torch.full_like(dx, float('nan'))
                        verify((dx, dw, db, dz), dict(eps=1e-3, group_size=17,
                               is_rms_norm=rms, norm_before_gate=before_gate), supplied)
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
    module = c['mod']
    inputs = tuple(c[name] for name in ('x', 'w', 'b', 'z'))
    pristine = snapshots(inputs)
    ref_options = dict(eps=1e-5, is_rms_norm=c['is_rms'])
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
            raise AssertionError('Benchmark did not invoke LayerNorm')
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
