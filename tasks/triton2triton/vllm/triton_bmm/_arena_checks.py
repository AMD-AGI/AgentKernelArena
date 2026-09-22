"""Check actual BMM timed outputs against pristine caller inputs."""
from contextlib import contextmanager
import inspect


def reference(a, b):
    import torch
    return torch.bmm(a.float(), b.float()).to(a.dtype)


def check_output(output, expected):
    import torch
    if not isinstance(output, torch.Tensor):
        raise AssertionError('BMM must return a tensor')
    if (output.shape != expected.shape or output.dtype != expected.dtype or
            output.device != expected.device):
        raise AssertionError('BMM output shape/dtype/device violates the contract')
    if not torch.isfinite(output).all():
        raise AssertionError('BMM output contains nonfinite values')
    if not torch.allclose(output, expected, atol=1e-2, rtol=1e-2):
        raise AssertionError('BMM differs from the original FP32 reference tolerance')


def check_inputs(a, b, pristine):
    import torch
    if any(not torch.equal(value, saved) for value, saved in zip((a, b), pristine)):
        raise AssertionError('BMM modified caller-owned inputs')


@contextmanager
def checked_modules(harness):
    original_load = harness.load_module
    modules = []

    def load():
        module = original_load()
        original = module.bmm_triton
        modules.append((module, original))

        def checked(a, b):
            pristine = (a.clone(), b.clone())
            expected = reference(*pristine)
            output = original(a, b)
            check_inputs(a, b, pristine)
            check_output(output, expected)
            return output

        module.bmm_triton = checked
        return module

    harness.load_module = load
    try:
        yield
    finally:
        harness.load_module = original_load
        for module, original in reversed(modules):
            module.bmm_triton = original


def checked_benchmark(harness, benchmark, fn, **kwargs):
    inputs = inspect.getclosurevars(fn).nonlocals
    module, a, b = (inputs[key] for key in ('mod', 'a', 'b'))
    original = module.bmm_triton
    pristine = (a.clone(), b.clone())
    expected = reference(*pristine)
    observed = []

    def observe(*args, **call_kwargs):
        output = original(*args, **call_kwargs)
        observed[:] = [output]
        return output

    def measured():
        fn()
        return observed[0]

    module.bmm_triton = observe
    try:
        timed = harness._TimedRun()
        ms, metadata = benchmark(measured, timed_run=timed, **kwargs)
        check_inputs(a, b, pristine)
        check_output(timed.outputs, expected)
        a.neg_()
        b.mul_(0.5)
        replay_pristine = (a.clone(), b.clone())
        replay_expected = reference(*replay_pristine)
        timed.outputs.fill_(float('nan'))
        output = timed.rerun()
        check_inputs(a, b, replay_pristine)
        check_output(output, replay_expected)
        return ms, {**metadata, 'timed_output_checked': True,
                    'perturbed_input_replay_checked': True,
                    'source_buffers_unchanged': True}
    finally:
        module.bmm_triton = original


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
