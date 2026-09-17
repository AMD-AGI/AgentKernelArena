"""Exact expert offsets/indices with pristine counts and timed replay checks."""
from contextlib import contextmanager
import inspect


def snapshot(values):
    return tuple(v.clone() for v in values)


def unchanged(value, original):
    import torch
    if not torch.equal(value, original):
        raise AssertionError('EP scatter 1 modified read-only token counts')


def reference(harness, counts, device):
    return tuple(v.to(device) for v in harness.reference_scatter_1(counts.cpu()))


def check_outputs(actual, expected):
    import torch
    if not isinstance(actual, tuple) or len(actual) != 2:
        raise AssertionError('EP scatter 1 requires both in-place outputs')
    for value, wanted in zip(actual, expected):
        if not isinstance(value, torch.Tensor) or (value.shape != wanted.shape or
                value.dtype != wanted.dtype or value.device != wanted.device):
            raise AssertionError('EP scatter 1 output shape/dtype/device is invalid')
        if not torch.equal(value, wanted):
            raise AssertionError('EP scatter 1 exact offsets/indices mismatch')


@contextmanager
def checked_modules(harness):
    loader = harness.load_module
    patched = []

    def load():
        module = loader()
        original = module.ep_scatter_1
        patched.append((module, original))
        diagnosed = False

        def verify(counts, starts, indices):
            saved = counts.clone()
            wanted = reference(harness, saved, counts.device)
            try:
                result = original(counts, starts, indices)
                unchanged(counts, saved)
                check_outputs((starts, indices), wanted)
                return result
            finally:
                counts.copy_(saved)

        def checked(counts, starts, indices):
            nonlocal diagnosed
            import torch
            result = verify(counts, starts, indices)
            if not diagnosed:
                diagnostic = torch.tensor([0,1,127,128,129,257,0],device=counts.device,dtype=counts.dtype)
                expected = reference(harness, diagnostic, counts.device)
                verify(diagnostic, torch.empty_like(expected[0]), torch.full_like(expected[1],-1))
                diagnosed = True
            return result

        module.ep_scatter_1 = checked
        return module

    harness.load_module = load
    try:
        yield
    finally:
        harness.load_module = loader
        for module, original in reversed(patched):
            module.ep_scatter_1 = original


def checked_benchmark(harness, benchmark, fn, **options):
    state = inspect.getclosurevars(fn).nonlocals
    counts = state['tokens_per_expert']
    outputs = (state['expert_start_loc'], state['m_indices'])
    buffers = (counts, *outputs)
    pristine = snapshot(buffers)
    wanted = reference(harness, pristine[0], counts.device)

    def measured():
        fn()
        return outputs

    try:
        timed = harness._TimedRun()
        ms, metadata = benchmark(measured, timed_run=timed, **options)
        unchanged(counts, pristine[0])
        check_outputs(timed.outputs, wanted)
        # Original scored performance counts are positive and <=128. Move the
        # first aligned block to the last expert, preserving output capacity;
        # change counts, prefix starts and the >128 fill-loop branch together.
        counts.copy_(129-pristine[0])
        counts[0] = 0
        counts[-1] += 128
        replay_counts = counts.clone()
        replay_wanted = reference(harness, replay_counts, counts.device)
        if replay_wanted[1].shape != outputs[1].shape:
            raise AssertionError('Perturbed counts changed allocated capacity')
        outputs[0].fill_(-777)
        outputs[1].fill_(-1)
        outputs[1][replay_wanted[1] >= 0] = -777
        replayed = timed.rerun()
        unchanged(counts, replay_counts)
        check_outputs(replayed, replay_wanted)
        return ms, {**metadata, 'timed_output_checked': True,
                    'perturbed_input_replay_checked': True, 'source_buffers_unchanged': True}
    finally:
        for value, saved in zip(buffers, pristine):
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
