"""Check integer count contracts and the exact measured expert-count replay."""
from contextlib import contextmanager
import inspect


def check_output(output, expected):
    import torch
    if not isinstance(output, torch.Tensor) or (output.shape != expected.shape or
            output.dtype != expected.dtype or output.device != expected.device):
        raise AssertionError('Expert counts have invalid shape/dtype/device')
    if not torch.equal(output, expected):
        raise AssertionError('Expert counts differ from the independent reference')


def unchanged(ids, pristine):
    import torch
    if not torch.equal(ids, pristine):
        raise AssertionError('Expert counting modified read-only input IDs')


def reference(harness, ids, experts):
    return harness.reference_count(ids.cpu(), experts).to(ids.device)


def perturb(ids, experts):
    ids.copy_((ids + 1).remainder(experts))
    # The public signed-ID interface reserves -1 for invalid assignments.
    ids.view(-1)[:2] = -1


@contextmanager
def checked_modules(harness):
    load_original = harness.load_module
    patched = []

    def load():
        module = load_original()
        original = module.count_expert_num_tokens
        patched.append((module, original))

        def checked(ids, experts):
            pristine = ids.clone()
            expected = reference(harness, pristine, experts)
            diagnostic = pristine.clone()
            perturb(diagnostic, experts)
            diagnostic_pristine = diagnostic.clone()
            diagnostic_expected = reference(harness, diagnostic_pristine, experts)
            check_output(original(diagnostic, experts), diagnostic_expected)
            unchanged(diagnostic, diagnostic_pristine)
            output = original(ids, experts)
            unchanged(ids, pristine)
            check_output(output, expected)
            return output

        module.count_expert_num_tokens = checked
        return module

    harness.load_module = load
    try:
        yield
    finally:
        harness.load_module = load_original
        for module, original in reversed(patched):
            module.count_expert_num_tokens = original


def checked_benchmark(harness, benchmark, fn, **kwargs):
    c = inspect.getclosurevars(fn).nonlocals
    module, ids, experts = (c[key] for key in ('mod', 'topk_ids', 'num_experts'))
    pristine = ids.clone()
    expected = reference(harness, pristine, experts)
    original = module.count_expert_num_tokens
    captured = None

    def collect(*args):
        nonlocal captured
        captured = original(*args)
        return captured

    def measured():
        nonlocal captured
        captured = None
        fn()
        if captured is None:
            raise AssertionError('Benchmark did not invoke the expert-count candidate')
        return captured

    module.count_expert_num_tokens = collect
    try:
        timed = harness._TimedRun()
        ms, metadata = benchmark(measured, timed_run=timed, **kwargs)
        unchanged(ids, pristine)
        check_output(timed.outputs, expected)
        perturb(ids, experts)
        replay_pristine = ids.clone()
        replay_expected = reference(harness, replay_pristine, experts)
        timed.outputs.fill_(-2)
        replayed = timed.rerun()
        unchanged(ids, replay_pristine)
        check_output(replayed, replay_expected)
        return ms, {**metadata, 'timed_output_checked': True,
                    'perturbed_input_replay_checked': True, 'source_buffers_unchanged': True}
    finally:
        module.count_expert_num_tokens = original
        ids.copy_(pristine)


def install(harness):
    correctness_original = harness.run_correctness
    performance_original = harness.run_performance

    def correctness(*args, **kwargs):
        with checked_modules(harness):
            return correctness_original(*args, **kwargs)

    def performance():
        benchmark = harness._benchmark_cuda_graph_or_events
        harness._benchmark_cuda_graph_or_events = lambda fn, **kwargs: checked_benchmark(harness, benchmark, fn, **kwargs)
        try:
            return performance_original()
        finally:
            harness._benchmark_cuda_graph_or_events = benchmark

    harness.run_correctness = correctness
    harness.run_performance = performance
