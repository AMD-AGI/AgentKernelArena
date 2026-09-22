"""Check actual Identity scaling timed outputs against pristine caller inputs."""
from contextlib import contextmanager
import inspect


def reference(hidden_states, expert_scales, top_k):
    # Same independent equation and accumulation dtype as the original oracle.
    scale_sum = expert_scales.sum(dim=-1, keepdim=True)
    return (hidden_states.float() * scale_sum.float()).to(hidden_states.dtype)


def check_output(output, expected):
    import torch
    if not isinstance(output, torch.Tensor):
        raise AssertionError('Identity scaling must return a tensor')
    if (output.shape != expected.shape or output.dtype != expected.dtype or
            output.device != expected.device):
        raise AssertionError('Identity scaling output shape/dtype/device violates the contract')
    if not torch.isfinite(output).all():
        raise AssertionError('Identity scaling output contains nonfinite values')
    if not torch.allclose(output, expected, atol=1e-2, rtol=1e-2):
        raise AssertionError('Identity scaling differs from the original reference tolerance')


def check_inputs(a, b, pristine):
    import torch
    if any(not torch.equal(value, saved) for value, saved in zip((a, b), pristine)):
        raise AssertionError('Identity scaling modified caller-owned inputs')


@contextmanager
def checked_modules(harness):
    original_load = harness.load_module
    modules = []

    def load():
        module = original_load()
        original = module.compute_identity
        modules.append((module, original))

        def checked(a, b, top_k):
            pristine = (a.clone(), b.clone())
            expected = reference(*pristine, top_k)
            output = original(a, b, top_k)
            check_inputs(a, b, pristine)
            check_output(output, expected)
            return output

        module.compute_identity = checked
        return module

    harness.load_module = load
    try:
        yield
    finally:
        harness.load_module = original_load
        for module, original in reversed(modules):
            module.compute_identity = original


def checked_benchmark(harness, benchmark, fn, **kwargs):
    inputs = inspect.getclosurevars(fn).nonlocals
    module, a, b, top_k = (inputs[key] for key in ('mod', 'hidden_states', 'expert_scales', 'top_k'))
    original = module.compute_identity
    pristine = (a.clone(), b.clone())
    expected = reference(*pristine, top_k)
    observed = []

    def observe(*args, **call_kwargs):
        output = original(*args, **call_kwargs)
        observed[:] = [output]
        return output

    def measured():
        fn()
        return observed[0]

    module.compute_identity = observe
    try:
        timed = harness._TimedRun()
        ms, metadata = benchmark(measured, timed_run=timed, **kwargs)
        check_inputs(a, b, pristine)
        check_output(timed.outputs, expected)
        a.neg_()
        b.mul_(0.5)
        replay_pristine = (a.clone(), b.clone())
        replay_expected = reference(*replay_pristine, top_k)
        timed.outputs.fill_(float('nan'))
        output = timed.rerun()
        check_inputs(a, b, replay_pristine)
        check_output(output, replay_expected)
        return ms, {**metadata, 'timed_output_checked': True,
                    'perturbed_input_replay_checked': True,
                    'source_buffers_unchanged': True}
    finally:
        module.compute_identity = original
        a.copy_(pristine[0])
        b.copy_(pristine[1])


def install(harness):
    original_correctness = harness.run_correctness
    original_performance = harness.run_performance

    def correctness(*args, **kwargs):
        with checked_modules(harness):
            return original_correctness(*args, **kwargs)

    def performance():
        benchmark = harness._benchmark_cuda_graph_or_events
        harness._benchmark_cuda_graph_or_events = lambda fn, **kwargs: checked_benchmark(
            harness, benchmark, fn, **kwargs)
        try:
            return original_performance()
        finally:
            harness._benchmark_cuda_graph_or_events = benchmark

    harness.run_correctness = correctness
    harness.run_performance = performance
