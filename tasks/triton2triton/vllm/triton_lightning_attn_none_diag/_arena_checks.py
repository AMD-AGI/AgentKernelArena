"""Validate the in-place cross-block update and its prepared timed replay."""
from contextlib import contextmanager
import inspect

SYMBOL = "lightning_attn_none_diag_forward"


def snapshots(inputs):
    return tuple(value.clone() for value in inputs)


def unchanged(inputs, saved):
    import torch
    for value, original in zip(inputs, saved):
        if not torch.equal(value, original):
            raise AssertionError("Lightning attention modified a read-only input")


def reference(harness, q, o, s, kv, block, cblock):
    # Run the original independent block formula on CPU and pristine inputs.
    value = harness.reference_none_diag(q.cpu(), o.cpu(), s.reshape(-1).cpu(), kv.cpu(), block, cblock)
    return value.to(device=o.device, dtype=o.dtype)


def check_output(value, expected):
    import torch
    if not isinstance(value, torch.Tensor) or (value.shape != expected.shape or
            value.dtype != expected.dtype or value.device != expected.device):
        raise AssertionError("Lightning attention output shape/dtype/device is invalid")
    if not torch.isfinite(value).all():
        raise AssertionError("Lightning attention output must be finite")
    torch.testing.assert_close(value, expected, atol=1e-2, rtol=1e-2)


def same_buffer(actual, output):
    if actual.data_ptr() != output.data_ptr() or actual.stride() != output.stride():
        raise AssertionError("Lightning attention must update and return the supplied output")


@contextmanager
def checked_modules(harness):
    loader = harness.load_module
    patched = []

    def load():
        module = loader()
        original = getattr(module, SYMBOL)
        patched.append((module, original))
        diagnosed = False

        def verify(q, out, s, kv, block, cblock):
            inputs = (q, s, kv)
            pristine = snapshots(inputs)
            expected = reference(harness, pristine[0], out.clone(), pristine[1], pristine[2], block, cblock)
            result = original(q, out, s, kv, block, cblock)
            unchanged(inputs, pristine)
            check_output(out, expected)
            check_output(result, expected)
            same_buffer(result, out)
            return result

        def checked(q, out, s, kv, BLOCK=256, CBLOCK=64):
            nonlocal diagnosed
            import torch
            result = verify(q, out, s, kv, BLOCK, CBLOCK)
            if not diagnosed:
                # Preserve every scored input; add a second main block, partial
                # sub-block, and the documented four-dimensional slope form.
                dq = ((torch.arange(2 * 273 * 32, device=q.device).reshape(1, 2, 273, 32) % 19 - 9) / 16).to(q.dtype)
                do = torch.full_like(dq, 0.25)
                ds = torch.tensor([0.01, 0.03], device=q.device).reshape(1, 2, 1, 1)
                dkv = (torch.arange(2 * 2 * 32 * 32, device=q.device).reshape(1, 2, 2, 32, 32) % 23 - 11).float() / 100
                verify(dq, do, ds, dkv, 256, 64)
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
    q, output, s, kv = (state[k] for k in ("q", "output_work", "s", "kv"))
    prepare = options["prepare_fn"]
    diagonal = inspect.getclosurevars(prepare).nonlocals["o"]
    block, cblock = state["BLOCK"], state["CBLOCK"]
    inputs = (q, s, kv, diagonal)
    pristine = snapshots(inputs)
    output_before = output.clone()
    expected = reference(harness, pristine[0], pristine[3], pristine[1], pristine[2], block, cblock)

    def measured():
        fn()
        return output

    try:
        timed = harness._TimedRun()
        ms, metadata = benchmark(measured, timed_run=timed, **options)
        unchanged(inputs, pristine)
        check_output(timed.outputs, expected)
        q.neg_()
        s.mul_(0.75)
        kv.mul_(0.5)
        diagonal.neg_()
        replay_inputs = snapshots(inputs)
        replay_expected = reference(harness, q, diagonal, s, kv, block, cblock)
        output.fill_(float("nan"))
        # TimedRun reruns the original preparation before the captured/eager
        # callable; the in-place add must start from the changed diagonal.
        replayed = timed.rerun()
        unchanged(inputs, replay_inputs)
        check_output(replayed, replay_expected)
        return ms, {**metadata, "timed_output_checked": True,
                    "perturbed_input_replay_checked": True, "source_buffers_unchanged": True,
                    "prepared_in_place_replay_checked": True}
    finally:
        for value, original in zip(inputs, pristine):
            value.copy_(original)
        output.copy_(output_before)


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
