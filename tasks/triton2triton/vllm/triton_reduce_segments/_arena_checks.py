"""Preserve the in-place output contract and check actual timed reductions."""
from contextlib import contextmanager
from functools import lru_cache
import importlib.util
import inspect
from pathlib import Path


def snapshots(values):
    return tuple(value.clone() for value in values)


def unchanged(values, saved):
    import torch
    for value, original in zip(values, saved):
        if not torch.equal(value, original):
            raise AssertionError("Segment reduction modified a read-only input")


@lru_cache(maxsize=1)
def reference_module():
    # Bind this task's protected reference by path, without a generic module
    # name that could resolve to another task's controls in the same process.
    spec = importlib.util.spec_from_file_location(
        '_reduce_segments_reference', Path(__file__).with_name('_upstream_controls.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference(harness, inputs, output, tile_size=16):
    partial, maxima, exp_sums, lengths, starts = (v.cpu() for v in inputs)
    # Respect packed query routing, active segments and zero denominators.
    # Original fully active FP32-input/FP16-output cases retain the same values.
    return reference_module().reference_reduce(
        partial, maxima, exp_sums, output.shape[-1], lengths, starts, tile_size).to(
        device=output.device, dtype=output.dtype)


def check_output(value, expected):
    import torch
    if not isinstance(value, torch.Tensor) or (value.shape != expected.shape or
            value.dtype != expected.dtype or value.device != expected.device):
        raise AssertionError("Segment reduction output shape/dtype/device is invalid")
    if not torch.isfinite(value).all():
        raise AssertionError("Segment reduction output must be finite")
    torch.testing.assert_close(value, expected, atol=1e-2, rtol=1e-2)


@contextmanager
def checked_modules(harness):
    loader = harness.load_module
    patched = []

    def load():
        module = loader()
        original = module.reduce_attention_segments
        patched.append((module, original))

        def checked(partial, maxima, sums, output, lengths, starts, tile_size=16):
            inputs = (partial, maxima, sums, lengths, starts)
            pristine = snapshots(inputs)
            expected = reference(harness, pristine, output, tile_size)
            result = original(partial, maxima, sums, output, lengths, starts, tile_size)
            unchanged(inputs, pristine)
            check_output(output, expected)
            check_output(result, expected)
            if result.data_ptr() != output.data_ptr() or result.stride() != output.stride():
                raise AssertionError("Segment reduction must return the supplied output buffer")
            return result

        module.reduce_attention_segments = checked
        return module

    harness.load_module = load
    try:
        yield
    finally:
        harness.load_module = loader
        for module, original in reversed(patched):
            module.reduce_attention_segments = original


def checked_benchmark(harness, benchmark, fn, **options):
    state = inspect.getclosurevars(fn).nonlocals
    inputs = tuple(state[k] for k in ("segm_output", "segm_max_t", "segm_expsum", "seqused_k", "cu_seqlens_q"))
    output = state["output"]
    pristine, output_before = snapshots(inputs), output.clone()
    expected = reference(harness, pristine, output)

    def measured():
        fn()
        return output

    try:
        timed = harness._TimedRun()
        ms, metadata = benchmark(measured, timed_run=timed, **options)
        unchanged(inputs, pristine)
        check_output(timed.outputs, expected)
        inputs[0].neg_()
        inputs[1].copy_(inputs[1].roll(1, dims=-1))
        inputs[2].mul_(0.75)
        replay_inputs = snapshots(inputs)
        replay_expected = reference(harness, replay_inputs, output)
        output.fill_(float("nan"))
        replayed = timed.rerun()
        unchanged(inputs, replay_inputs)
        check_output(replayed, replay_expected)
        return ms, {**metadata, "timed_output_checked": True,
                    "perturbed_input_replay_checked": True, "source_buffers_unchanged": True}
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
