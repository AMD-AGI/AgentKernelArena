"""Check both recomputed tensors against pristine inputs and exact timed replay."""
from contextlib import contextmanager
import inspect

SYMBOL = 'kda_recompute_wu'
ARGUMENTS = ('k', 'v', 'beta', 'A', 'gk')


def snapshots(values):
    return tuple(value.clone() for value in values)


def unchanged(values, saved):
    import torch
    for value, original in zip(values, saved):
        if not torch.equal(value, original):
            raise AssertionError('Recompute modified a read-only input')


def reference(harness, inputs):
    # Preserve both original FP32 references and their original 0.05/0.05 gate.
    return tuple(value.to(inputs[0].device) for value in harness.reference(*inputs))


def check_outputs(values, expected, inputs):
    import torch
    if not isinstance(values, tuple) or len(values) != 2:
        raise AssertionError('Recompute must return exactly the (w, u) tuple')
    for value, wanted, prototype in zip(values, expected, inputs[:2]):
        if not isinstance(value, torch.Tensor) or (value.shape != wanted.shape or
                value.dtype != prototype.dtype or value.device != wanted.device):
            raise AssertionError('Recompute output shape/dtype/device is invalid')
        if not torch.isfinite(value).all():
            raise AssertionError('Recompute outputs must be finite')
        torch.testing.assert_close(value.float(), wanted, atol=5e-2, rtol=5e-2)


def diagnostic_inputs(inputs):
    import torch
    k, v, beta = inputs[:3]
    device = k.device
    ik = torch.arange(2*35*3*65, device=device).reshape(2,35,3,65)
    iv = torch.arange(2*35*3*70, device=device).reshape(2,35,3,70)
    ib = torch.arange(2*35*3, device=device).reshape(2,35,3)
    ia = torch.arange(2*35*3*32, device=device).reshape(2,35,3,32)
    data = {'k': (.25+(ik%5)/16).to(k.dtype),
            'v': (.125+(iv%7)/16).to(v.dtype),
            'beta': (.25+(ib%3)/8).to(beta.dtype),
            'A': (.5+(ia%7)/32).to(inputs[ARGUMENTS.index('A')].dtype)}
    gate = ARGUMENTS[-1] if ARGUMENTS[-1] != 'A' else ARGUMENTS[-2]
    indices = ik if gate == 'gk' else ib
    data[gate] = ((indices%7-3)/10).to(inputs[ARGUMENTS.index(gate)].dtype)
    return tuple(data[name] for name in ARGUMENTS)


@contextmanager
def checked_modules(harness):
    loader = harness.load_module
    patched = []

    def load():
        module = loader()
        original = getattr(module, SYMBOL)
        patched.append((module, original))
        diagnosed = False

        def verify(inputs):
            pristine = snapshots(inputs)
            expected = reference(harness, pristine)
            try:
                result = original(*inputs)
                unchanged(inputs, pristine)
                check_outputs(result, expected, inputs)
                return result
            finally:
                for value, saved in zip(inputs, pristine):
                    value.copy_(saved)

        def checked(*args, **kwargs):
            nonlocal diagnosed
            bound = inspect.signature(original).bind(*args, **kwargs)
            inputs = tuple(bound.arguments[name] for name in ARGUMENTS)
            result = verify(inputs)
            if not diagnosed:
                # Unscored second BT block, partial row and K/V tiles, unequal
                # widths, multiple batches/heads, and nontrivial positive output.
                verify(diagnostic_inputs(inputs))
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
    original = getattr(module, SYMBOL)
    bound = inspect.signature(original).bind(*state['args'], **state['kwargs'])
    inputs = tuple(bound.arguments[name] for name in ARGUMENTS)
    pristine = snapshots(inputs)
    expected = reference(harness, pristine)
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
            raise AssertionError('Timed recompute did not return its outputs')
        return captured

    setattr(module, SYMBOL, collect)
    try:
        timed = harness._TimedRun()
        ms, metadata = benchmark(measured, timed_run=timed, **options)
        unchanged(inputs, pristine)
        check_outputs(timed.outputs, expected, inputs)
        # Deterministic nontrivial replay: change all inputs, including the gate
        # and A. Original small random operands can hide zero output at 0.05.
        for name, value in zip(ARGUMENTS, inputs):
            if name == 'k': value.mul_(-.5).add_(.5)
            elif name == 'v': value.mul_(.5).add_(.75)
            elif name == 'beta': value.mul_(.5).add_(.25)
            elif name == 'A': value.mul_(.5).add_(.125)
            else: value.mul_(.5).add_(.2)
        replay_inputs = snapshots(inputs)
        replay_expected = reference(harness, replay_inputs)
        for value in timed.outputs:
            value.fill_(float('nan'))
        replayed = timed.rerun()
        unchanged(inputs, replay_inputs)
        check_outputs(replayed, replay_expected, inputs)
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
