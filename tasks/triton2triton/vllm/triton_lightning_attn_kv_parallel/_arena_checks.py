"""Check all KV blocks and the output of the exact timed invocation."""
from contextlib import contextmanager
import inspect

SYMBOL = "lightning_attn_kv_parallel_forward"


def snapshots(inputs):
    return tuple(value.clone() for value in inputs)


def unchanged(inputs, saved):
    import torch
    for value, original in zip(inputs, saved):
        if not torch.equal(value, original):
            raise AssertionError("KV attention modified a read-only input")


def reference(harness, inputs, block=256, cblock=64):
    k, v, s = inputs
    expected = harness.reference_kv_parallel(k.cpu(), v.cpu(), s.reshape(-1).cpu(), block, cblock)
    return expected.to(k.device)


def check_output(value, expected):
    import torch
    if not isinstance(value, torch.Tensor) or (value.shape != expected.shape or
            value.dtype != expected.dtype or value.device != expected.device):
        raise AssertionError("KV attention output shape/dtype/device is invalid")
    if not torch.isfinite(value).all():
        raise AssertionError("KV attention output must be finite")
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

        def verify(k, v, s, n, block, cblock):
            inputs = (k, v, s)
            pristine = snapshots(inputs)
            expected = reference(harness, pristine, block, cblock)
            result = original(k, v, s, n, block, cblock)
            unchanged(inputs, pristine)
            check_output(result, expected)
            return result

        def checked(k, v, s, n, BLOCK=256, CBLOCK=64):
            nonlocal diagnosed
            import torch
            result = verify(k, v, s, n, BLOCK, CBLOCK)
            if not diagnosed:
                # Unscored second BLOCK and partial CBLOCK. Positive operands
                # make the final token's contribution substantial, not masked
                # by an accidental zero or cancellation in the diagnostic.
                indices = torch.arange(2 * 273 * 32, device=k.device).reshape(1, 2, 273, 32)
                dk = (0.125 + (indices % 7) / 8).to(k.dtype)
                dv = (0.25 + (indices % 5) / 4).to(v.dtype)
                ds = torch.tensor([0.01, 0.03], device=k.device).reshape(1, 2, 1, 1)
                verify(dk, dv, ds, 273, 256, 64)
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
    inputs = tuple(state[name] for name in ("k", "v", "s"))
    module = state["mod"]
    pristine = snapshots(inputs)
    expected = reference(harness, pristine)
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
            raise AssertionError("Benchmark did not produce KV attention output")
        return captured

    setattr(module, SYMBOL, collect)
    try:
        timed = harness._TimedRun()
        ms, metadata = benchmark(measured, timed_run=timed, **options)
        unchanged(inputs, pristine)
        check_output(timed.outputs, expected)
        inputs[0].neg_()
        inputs[1].mul_(0.5)
        inputs[2].mul_(0.75)
        replay_inputs = snapshots(inputs)
        replay_expected = reference(harness, replay_inputs)
        timed.outputs.fill_(float("nan"))
        replayed = timed.rerun()
        unchanged(inputs, replay_inputs)
        check_output(replayed, replay_expected)
        return ms, {**metadata, "timed_output_checked": True,
                    "perturbed_input_replay_checked": True, "source_buffers_unchanged": True}
    finally:
        setattr(module, SYMBOL, original)
        for value, saved in zip(inputs, pristine):
            value.copy_(saved)


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
