"""Observe the paged-attention output without changing its lock-reset timing."""
import inspect
from _aka_benchmark import TimedRun


def readonly(case):
    tensors = [case[name] for name in ("q", "k", "v", "kv_block_tables", "batch_num_block_n")]
    tensors.extend(indices for head in case["ref_indices"] for indices in head)
    return tuple(tensors)


def snapshots(tensors):
    return tuple(value.clone() for value in tensors)


def unchanged(tensors, saved):
    import torch
    for value, original in zip(tensors, saved):
        if not torch.equal(value, original):
            raise AssertionError("Paged attention modified a read-only input or mapping")


def reference(harness, case, cfg):
    return harness.torch_op(case["q"], case["k"], case["v"],
                            case["ref_indices"], cfg[2], case["sm_scale"])


def check_output(harness, value, expected):
    import torch
    if not isinstance(value, torch.Tensor) or (value.shape != expected.shape or
            value.dtype != expected.dtype or value.device != expected.device):
        raise AssertionError("Paged-attention output shape/dtype/device is invalid")
    if not torch.isfinite(value).all():
        raise AssertionError("Paged-attention output must be finite")
    torch.testing.assert_close(value, expected, atol=harness.ATOL, rtol=harness.RTOL)


def checked_benchmark(harness, benchmark, fn, **options):
    state = inspect.getclosurevars(fn).nonlocals
    case, cfg = state["case"], state["cfg"]
    tensors = readonly(case)
    pristine = snapshots(tensors)
    scratch = tuple(case[name] for name in ("Mp", "Lp", "Op", "locks"))
    scratch_before = snapshots(scratch)
    expected = reference(harness, case, cfg)
    unchanged(tensors, pristine)
    try:
        timed = TimedRun()
        ms, metadata = benchmark(fn, timed_run=timed, **options)
        unchanged(tensors, pristine)
        check_output(harness, timed.outputs, expected)
        case["q"].neg_()
        case["k"].mul_(0.75)
        case["v"].neg_()
        replay_inputs = snapshots(tensors)
        replay_expected = reference(harness, case, cfg)
        # Scratch has no input value contract; the operator must write each
        # intermediate it reads. Keep this diagnostic outside measured work.
        for name in ("Mp", "Lp", "Op"):
            case[name].fill_(float("nan"))
        timed.outputs.fill_(float("nan"))
        # Preserve the original prepare_fn=locks.zero_; TimedRun owns replay
        # preparation and the exact graph/eager invocation that was measured.
        replayed = timed.rerun()
        unchanged(tensors, replay_inputs)
        check_output(harness, replayed, replay_expected)
        return ms, {**metadata, "timed_output_checked": True,
                    "perturbed_input_replay_checked": True, "source_buffers_unchanged": True,
                    "prepared_lock_replay_checked": True, "scratch_reinitialized_checked": True}
    finally:
        for value, saved in zip(tensors, pristine):
            value.copy_(saved)
        for value, saved in zip(scratch, scratch_before):
            value.copy_(saved)


def install(harness):
    if getattr(harness, "_arena_lean_checks_installed", False):
        return
    correctness = harness.run_correctness
    benchmark = harness.benchmark_cuda_graph_or_events

    def checked_correctness(*args, **kwargs):
        call = harness._call_triton

        def checked_call(case, cfg):
            tensors = readonly(case)
            pristine = snapshots(tensors)
            expected = reference(harness, case, cfg)
            actual = call(case, cfg)
            unchanged(tensors, pristine)
            check_output(harness, actual, expected)
            return actual

        harness._call_triton = checked_call
        try:
            return correctness(*args, **kwargs)
        finally:
            harness._call_triton = call

    harness.run_correctness = checked_correctness
    harness.benchmark_cuda_graph_or_events = lambda fn, **kwargs: checked_benchmark(harness, benchmark, fn, **kwargs)
    harness._arena_lean_checks_installed = True
