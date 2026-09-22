"""Check exact token dtype/device and the output of the measured invocation."""
from contextlib import contextmanager
import inspect

SYMBOL = "sample_recovered_tokens"


def snapshots(values):
    return tuple(None if value is None else value.clone() for value in values)


def unchanged(values, saved):
    import torch
    for value, original in zip(values, saved):
        if value is not None and not torch.equal(value, original):
            raise AssertionError("Recovered-token sampling modified a read-only input")


def reference(harness, inputs, vocab):
    cpu = tuple(None if value is None else value.cpu() for value in inputs)
    return harness.reference_sample_recovered_tokens(*cpu, vocab).to(inputs[1].device)


def check_output(value, expected):
    import torch
    if not isinstance(value, torch.Tensor) or (value.shape != expected.shape or
            value.dtype != expected.dtype or value.device != expected.device):
        raise AssertionError("Recovered-token output shape/dtype/device is invalid")
    if not torch.equal(value, expected):
        raise AssertionError("Recovered tokens differ from the pristine integer reference")


@contextmanager
def checked_modules(harness):
    loader = harness.load_module
    patched = []

    def load():
        module = loader()
        original = getattr(module, SYMBOL)
        patched.append((module, original))

        def checked(cu, ids, draft, target, q, maximum, vocab):
            inputs = (cu, ids, draft, target, q)
            pristine = snapshots(inputs)
            expected = reference(harness, pristine, vocab)
            result = original(cu, ids, draft, target, q, maximum, vocab)
            unchanged(inputs, pristine)
            check_output(result, expected)
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
    inputs = tuple(state[k] for k in ("cu", "draft_ids", "draft_probs", "target_probs", "q"))
    module, vocab = state["mod"], state["vocab_size"]
    pristine = snapshots(inputs)
    expected = reference(harness, pristine, vocab)
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
            raise AssertionError("Benchmark did not produce recovered tokens")
        return captured

    setattr(module, SYMBOL, collect)
    try:
        timed = harness._TimedRun()
        ms, metadata = benchmark(measured, timed_run=timed, **options)
        unchanged(inputs, pristine)
        check_output(timed.outputs, expected)
        inputs[1].add_(1).remainder_(vocab)
        # Permute the vocabulary consistently, retaining valid distributions
        # and positive exponential-race inputs. Routing/lengths are unchanged.
        for value in inputs[2:]:
            if value is not None:
                value.copy_(value.roll(1, dims=-1))
        replay_inputs = snapshots(inputs)
        replay_expected = reference(harness, replay_inputs, vocab)
        timed.outputs.fill_(-1)
        replayed = timed.rerun()
        unchanged(inputs, replay_inputs)
        check_output(replayed, replay_expected)
        return ms, {**metadata, "timed_output_checked": True,
                    "perturbed_input_replay_checked": True, "source_buffers_unchanged": True}
    finally:
        setattr(module, SYMBOL, original)
        for value, saved in zip(inputs, pristine):
            if value is not None:
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
