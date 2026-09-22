"""Protected expansion oracle, ragged/replacement control and captured replay."""
from contextlib import contextmanager
import inspect

SYMBOL = 'expand_batch_to_tokens'


def unchanged(inputs, pristine):
    import torch
    if any(not torch.equal(x, saved) for x, saved in zip(inputs, pristine)):
        raise AssertionError('Expansion modified read-only source/count inputs')


def reference(x, cu, num_tokens, replace_from=0, replace_to=0):
    import torch
    values, ends = x.cpu().tolist(), cu.cpu().tolist()
    result, start = [], 0
    for value, end in zip(values, ends):
        if end < start:
            raise AssertionError('Cumulative counts must be nondecreasing')
        result.extend([replace_to if value == replace_from else value] * (end - start))
        start = end
    if len(result) != num_tokens:
        raise AssertionError('Cumulative counts disagree with declared token count')
    return torch.tensor(result, dtype=x.dtype, device=x.device)


def check_output(output, expected):
    import torch
    if not isinstance(output, torch.Tensor) or (output.shape != expected.shape or
            output.dtype != expected.dtype or output.device != expected.device):
        raise AssertionError('Expansion output shape/dtype/device is invalid')
    if not torch.equal(output, expected):
        raise AssertionError('Expansion output differs from pristine-input reference')


@contextmanager
def checked_modules(harness):
    load_original = harness.load_module
    patched = []

    def load():
        module = load_original()
        original = getattr(module, SYMBOL)
        patched.append((module, original))

        def checked(x, cu, num_tokens, replace_from=0, replace_to=0):
            pristine = (x.clone(), cu.clone())
            expected = reference(*pristine, num_tokens, replace_from, replace_to)
            # An unscored legal ragged case: empty request, unequal lengths,
            # and explicit replacement; does not replace the original cases.
            dx = x.new_tensor([7, 2, 7, 9])
            dc = cu.new_tensor([0, 1, 4, 6])
            saved = (dx.clone(), dc.clone())
            wanted = reference(*saved, 6, 7, -3)
            check_output(original(dx, dc, 6, 7, -3), wanted)
            unchanged((dx, dc), saved)
            result = original(x, cu, num_tokens, replace_from, replace_to)
            unchanged((x, cu), pristine)
            check_output(result, expected)
            return result

        setattr(module, SYMBOL, checked)
        return module

    harness.load_module = load
    try:
        yield
    finally:
        harness.load_module = load_original
        for module, original in reversed(patched):
            setattr(module, SYMBOL, original)


def checked_benchmark(harness, benchmark, fn, **options):
    c = inspect.getclosurevars(fn).nonlocals
    module, x, cu, num_tokens = (c[name] for name in ('mod', 'x', 'cu', 'num_tokens'))
    inputs = x, cu
    pristine = x.clone(), cu.clone()
    expected = reference(*pristine, num_tokens)
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
            raise AssertionError('Benchmark did not invoke expansion')
        return captured

    setattr(module, SYMBOL, collect)
    try:
        timed = harness._TimedRun()
        ms, metadata = benchmark(measured, timed_run=timed, **options)
        unchanged(inputs, pristine)
        check_output(timed.outputs, expected)
        x.add_(1000)
        # Transfer one token from request 1 to request 0: same total/shape,
        # positive original counts, and still within the MAX_SPEC_LEN bound.
        cu[0].add_(1)
        replay_pristine = x.clone(), cu.clone()
        replay_expected = reference(*replay_pristine, num_tokens)
        timed.outputs.fill_(-2147483648)
        replayed = timed.rerun()
        unchanged(inputs, replay_pristine)
        check_output(replayed, replay_expected)
        return ms, {**metadata, 'timed_output_checked': True,
                    'perturbed_input_replay_checked': True, 'source_buffers_unchanged': True}
    finally:
        setattr(module, SYMBOL, original)
        for value, saved in zip(inputs, pristine):
            value.copy_(saved)


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
