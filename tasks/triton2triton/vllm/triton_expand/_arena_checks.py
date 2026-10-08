"""Protected expansion oracle, dtype controls and measured-output checks."""
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


def dtype_controls(device):
    """Unscored fractional, wide-integer and 128-token boundary cases."""
    import torch
    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64,
                  torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        floating = dtype.is_floating_point
        values = [1.25, -2.5, 3.75] if floating else [7, 2, 9]
        old, new = (-2.5, 6.125) if floating else (2, 13)
        if dtype == torch.int64:
            values, old, new = [2**40 + 7, 2**40 + 2, -(2**40) + 9], 2**40 + 2, 2**41 + 13
        for count_dtype in (torch.int32, torch.int64):
            yield (torch.tensor(values, dtype=dtype, device=device),
                   torch.tensor([128, 129, 132], dtype=count_dtype, device=device),
                   old, new)
    # Bool is a legal source dtype too. Check both mixed-value expansion and
    # replacement; the latter necessarily collapses one of the two values.
    for count_dtype in (torch.int32, torch.int64):
        values = torch.tensor([True, False, True], dtype=torch.bool, device=device)
        counts = torch.tensor([128, 129, 132], dtype=count_dtype, device=device)
        yield values, counts, False, False
        yield values, counts, False, True


def stride_controls(device):
    """Legal 1-D views that expose independent source and count strides."""
    import torch
    for dtype, values in ((torch.int32, [7, 99, 3, 99]),
                          (torch.float32, [1.25, 99.0, -2.5, 99.0]),
                          (torch.bool, [True, True, False, True])):
        source = torch.tensor(values, dtype=dtype, device=device)
        counts = torch.tensor([2, 4], dtype=torch.int32, device=device)
        yield source[::2], counts, 4, (source, counts)
    for count_dtype in (torch.int32, torch.int64):
        source = torch.tensor([7, 3, 9], dtype=torch.int32, device=device)
        # Logical [1, 4, 4] has an empty last request. Reading this view as
        # unit-stride instead gives [1, 2, 4], a fully written wrong answer.
        counts = torch.tensor([1, 2, 4, 4, 4, 4], dtype=count_dtype, device=device)
        yield source, counts[::2], 4, (source, counts)


@contextmanager
def checked_modules(harness):
    load_original = harness.load_module
    patched = []

    def load():
        module = load_original()
        original = getattr(module, SYMBOL)
        patched.append((module, original))

        def checked(x, cu, num_tokens, replace_from=0, replace_to=0):
            import torch
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
            # Exercise the public dtype and per-request upper bound separately
            # from the historical integer workload. Distinct fractional values
            # and replacement expose integer-only or truncated implementations.
            for fx, fc, old, new in dtype_controls(x.device):
                fsaved = (fx.clone(), fc.clone())
                fwanted = reference(*fsaved, 132, old, new)
                check_output(original(fx, fc, 132, old, new), fwanted)
                unchanged((fx, fc), fsaved)
            for sx, sc, total, backing in stride_controls(x.device):
                saved = tuple(value.clone() for value in backing)
                wanted = reference(sx.clone(), sc.clone(), total)
                check_output(original(sx, sc, total), wanted)
                unchanged(backing, saved)
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
        checked_samples = [0]
        def check_sample(output):
            check_output(output, expected)
            checked_samples[0] += 1
        timed.after_sample = check_sample
        # Score one complete public invocation per sample. Graph batching can
        # count capture-time calls whose outputs are never independently seen.
        options = {**options, 'use_cuda_graph': False,
                   'fallback_reason': 'validate_each_public_invocation'}
        ms, metadata = benchmark(measured, timed_run=timed, **options)
        if not timed.bound or checked_samples[0] != options['repetition']:
            raise AssertionError('Reported sample outputs were not all checked')
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
                    'perturbed_input_replay_checked': True, 'source_buffers_unchanged': True,
                    'measured_samples_checked': checked_samples[0], }
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
