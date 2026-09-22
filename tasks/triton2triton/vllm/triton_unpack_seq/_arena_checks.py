"""Validate unpacked tensor contract and the exact directly timed kernel."""
from contextlib import contextmanager
import inspect


def unchanged(packed, lengths, saved_packed, saved_lengths):
    import torch
    if not torch.equal(packed, saved_packed) or not torch.equal(lengths, saved_lengths):
        raise AssertionError('unpack_seq modified packed data or sequence lengths')


def reference(harness, packed, lengths):
    flat = packed.reshape(packed.shape[0], packed.shape[1], -1)
    expected = harness.reference_unpack_seq(flat, lengths.cpu().tolist())
    return expected.reshape(expected.shape[0], *packed.shape[2:])


def check_output(value, expected):
    import torch
    if not isinstance(value, torch.Tensor) or (value.shape != expected.shape or
            value.dtype != expected.dtype or value.device != expected.device):
        raise AssertionError('unpack_seq output shape/dtype/device is invalid')
    if not torch.isfinite(value).all():
        raise AssertionError('unpack_seq output must be finite')
    torch.testing.assert_close(value, expected, atol=1e-3, rtol=1e-3)


@contextmanager
def checked_modules(harness):
    loader = harness.load_module
    patched = []

    def load():
        module = loader()
        original = module.unpack_seq
        patched.append((module, original))
        diagnosed = False

        def verify(packed, lengths, **options):
            saved_packed, saved_lengths = packed.clone(), lengths.clone()
            expected = reference(harness, saved_packed, saved_lengths)
            result = original(packed, lengths, **options)
            unchanged(packed, lengths, saved_packed, saved_lengths)
            check_output(result, expected)
            return result

        def checked(packed_tensor, lengths, block_t=64, block_d=64):
            nonlocal diagnosed
            import torch
            result = verify(packed_tensor, lengths, block_t=block_t, block_d=block_d)
            if not diagnosed:
                # Higher-rank features, zero-length prefix, second time block and
                # partial time/feature tiles. No added performance measurements.
                data = ((torch.arange(4*65*3*23, device=packed_tensor.device).reshape(4,65,3,23)%37-18)/4).to(packed_tensor.dtype)
                lens = torch.tensor([0,65,1,3], device=lengths.device, dtype=lengths.dtype)
                verify(data, lens, block_t=32, block_d=32)
                verify(data, torch.zeros_like(lens))
                diagnosed = True
            return result

        module.unpack_seq = checked
        return module

    harness.load_module = load
    try:
        yield
    finally:
        harness.load_module = loader
        for module, original in reversed(patched):
            module.unpack_seq = original


def checked_benchmark(harness, benchmark, fn, **options):
    state = inspect.getclosurevars(fn).nonlocals
    packed, lengths, out = state['packed'], state['lengths'], state['out']
    saved_packed, saved_lengths, saved_out = packed.clone(), lengths.clone(), out.clone()
    expected = reference(harness, saved_packed, saved_lengths)

    def measured():
        # Keep the existing direct JIT launch, grid, reusable output and timing unit.
        fn()
        return out

    try:
        timed = harness._TimedRun()
        ms, metadata = benchmark(measured, timed_run=timed, **options)
        unchanged(packed, lengths, saved_packed, saved_lengths)
        check_output(timed.outputs, expected)
        packed.mul_(-.5).add_(.125)
        # Same total output size and launch grid; routing and prefix sums change.
        lengths.copy_(saved_lengths.flip(0))
        replay_packed, replay_lengths = packed.clone(), lengths.clone()
        replay_expected = reference(harness, replay_packed, replay_lengths)
        out.fill_(float('nan'))
        replayed = timed.rerun()
        unchanged(packed, lengths, replay_packed, replay_lengths)
        check_output(replayed, replay_expected)
        return ms, {**metadata, 'timed_output_checked': True,
                    'perturbed_input_replay_checked': True, 'source_buffers_unchanged': True}
    finally:
        packed.copy_(saved_packed)
        lengths.copy_(saved_lengths)
        out.copy_(saved_out)


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
