"""Check persistent matmul and exact timed output replay."""
from contextlib import contextmanager
import inspect

SYMBOL = 'matmul_persistent'


def snapshots(values):
    return tuple(value.clone() for value in values)


def unchanged(values, saved):
    import torch
    for value, original in zip(values, saved):
        if not torch.equal(value, original):
            raise AssertionError('Persistent matmul modified a read-only input')


def reference(inputs):
    import torch
    a, b, *bias = inputs
    result = a.float() @ b.float()
    if bias:
        result = result + bias[0].float()
    return result.to(a.dtype)


def check_output(value, expected):
    import torch
    if not isinstance(value, torch.Tensor) or (value.shape != expected.shape or
            value.dtype != expected.dtype or value.device != expected.device):
        raise AssertionError('Persistent matmul output shape/dtype/device is invalid')
    if not torch.isfinite(value).all():
        raise AssertionError('Persistent matmul output must be finite')
    torch.testing.assert_close(value, expected, atol=1e-2, rtol=1e-2)


@contextmanager
def checked_modules(harness):
    loader = harness.load_module
    patched = []

    def load():
        module = loader()
        original = getattr(module, SYMBOL)
        patched.append((module, original))
        diagnosed = False

        def verify(a, b, bias=None):
            inputs = (a, b) if bias is None else (a, b, bias)
            pristine = snapshots(inputs)
            expected = reference(pristine)
            try:
                result = original(a, b, bias=bias)
                unchanged(inputs, pristine)
                check_output(result, expected)
                return result
            finally:
                for value, saved in zip(inputs, pristine):
                    value.copy_(saved)

        def checked(a, b, bias=None):
            nonlocal diagnosed
            import torch
            result = verify(a, b, bias)
            if not diagnosed:
                # Unscored partial M/N/K tiles, strided operands, optional bias.
                ai = torch.arange(258*134, device=a.device).reshape(258,134)
                bi = torch.arange(518*67, device=b.device).reshape(518,67)
                da = (.25+(ai%5)/16).to(a.dtype)[::2,::2]
                db = (.125+(bi%7)/16).to(b.dtype)[::2].t()
                dbias = (2+(torch.arange(259,device=a.device)%11)/8).to(a.dtype)
                verify(da, db, dbias)
                # 340 original FP16 tiles cover partial grouped scheduling and
                # more than one persistent wave on the qualified gfx950 GPU.
                ai = torch.arange(1153*16,device=a.device).reshape(1153,16)
                bi = torch.arange(16*8449,device=b.device).reshape(16,8449)
                verify((.25+(ai%5)/16).to(a.dtype),
                       (.125+(bi%7)/16).to(b.dtype))
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
    module = state['mod']
    inputs = tuple(state[name] for name in ('a', 'b'))
    pristine = snapshots(inputs)
    expected = reference(pristine)
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
            raise AssertionError('Timed persistent matmul did not return an output')
        return captured

    setattr(module, SYMBOL, collect)
    try:
        timed = harness._TimedRun()
        ms, metadata = benchmark(measured, timed_run=timed, **options)
        unchanged(inputs, pristine)
        check_output(timed.outputs, expected)
        inputs[0].mul_(-.5).add_(.5)
        inputs[1].mul_(.5).add_(.25)
        replay_inputs = snapshots(inputs)
        replay_expected = reference(replay_inputs)
        timed.outputs.fill_(float('nan'))
        replayed = timed.rerun()
        unchanged(inputs, replay_inputs)
        check_output(replayed, replay_expected)
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
