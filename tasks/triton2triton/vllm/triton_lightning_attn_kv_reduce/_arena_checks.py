"""Check the KV recurrence and both in-place outputs of the prepared timed run."""
from contextlib import contextmanager
import inspect

SYMBOL = 'lightning_attn_kv_reduce_forward'


def snapshot(values):
    return tuple(value.clone() for value in values)


def unchanged(values, saved):
    import torch
    for value, original in zip(values, saved):
        if not torch.equal(value, original):
            raise AssertionError('Lightning KV reduce modified a read-only source')


def check_outputs(result, outputs, expected):
    import torch
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise AssertionError('Lightning KV reduce must return both in-place outputs')
    for returned, output, reference in zip(result, outputs, expected):
        if not isinstance(returned, torch.Tensor) or (returned.shape != reference.shape or
                returned.dtype != reference.dtype or returned.device != reference.device):
            raise AssertionError('Lightning KV reduce output shape/dtype/device is invalid')
        if returned.data_ptr() != output.data_ptr() or returned.stride() != output.stride():
            raise AssertionError('Lightning KV reduce must return the supplied in-place buffers')
        torch.testing.assert_close(output, reference, atol=1e-2, rtol=1e-2)


def reference(harness, slope, kv, history, n, block):
    return harness.reference_kv_reduce(slope.reshape(-1), kv, history, n, block)


@contextmanager
def checked_modules(harness):
    loader = harness.load_module
    patched = []

    def load():
        module = loader()
        original = getattr(module, SYMBOL)
        patched.append((module, original))
        diagnosed = False

        def verify(s, kv, history, n, block):
            saved_s = s.clone()
            expected = reference(harness, saved_s, kv.clone(), history.clone(), n, block)
            try:
                result = original(s, kv, history, n, block)
                unchanged((s,), (saved_s,))
                check_outputs(result, (kv, history), expected)
                return result
            finally:
                s.copy_(saved_s)

        def checked(s, kv, kv_history, n, BLOCK=256):
            nonlocal diagnosed
            import torch
            result = verify(s, kv, kv_history, n, BLOCK)
            if not diagnosed:
                # 256+17 tokens: exercise the carry and partial-block decay.
                # Supported power-of-two D/E are retained; slope can be 4-D.
                slope = torch.tensor([0., .03], dtype=s.dtype, device=s.device).reshape(1,2,1,1)
                blocks = ((torch.arange(2*2*16*32, device=kv.device)%11)-5).reshape(1,2,2,16,32).to(kv.dtype)*.125
                history = torch.full((1,2,16,32), .25, dtype=kv_history.dtype, device=kv_history.device)
                verify(slope, blocks, history, 273, 256)
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
    prep_state = inspect.getclosurevars(options['prepare_fn']).nonlocals
    module, slope, kv_work, history_work = (state[k] for k in ('mod','s','kv_work','history_work'))
    kv, history = prep_state['kv'], prep_state['kv_history']
    readonly = (slope, kv, history)
    pristine = snapshot(readonly)
    work_before = snapshot((kv_work, history_work))
    n, block = state['N'], state['BLOCK']
    expected = reference(harness, *pristine, n, block)
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
            raise AssertionError('Timed KV reduce did not return both outputs')
        return captured

    setattr(module, SYMBOL, collect)
    try:
        timed = harness._TimedRun()
        ms, metadata = benchmark(measured, timed_run=timed, **options)
        unchanged(readonly, pristine)
        check_outputs(timed.outputs, (kv_work, history_work), expected)
        slope.mul_(.5)
        kv.mul_(-.5).add_(.25)
        history.fill_(.75)
        replay_inputs = snapshot(readonly)
        replay_expected = reference(harness, *replay_inputs, n, block)
        # Both outputs are also input state: NaN poisoning would change the
        # recurrence itself. The original prepare_fn installs changed source
        # values before the exact replay; compare both resulting state buffers.
        replayed = timed.rerun()
        unchanged(readonly, replay_inputs)
        check_outputs(replayed, (kv_work, history_work), replay_expected)
        return ms, {**metadata, 'timed_output_checked': True,
                    'perturbed_input_replay_checked': True,
                    'prepared_inplace_replay_checked': True,
                    'source_buffers_unchanged': True}
    finally:
        setattr(module, SYMBOL, original)
        for value, saved in zip((*readonly, kv_work, history_work), (*pristine, *work_before)):
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
