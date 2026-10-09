"""Check true expert membership and the exact measured bitmatrix replay."""
from contextlib import contextmanager
import inspect
from itertools import product


def check_output(output, expected):
    import torch
    if not isinstance(output, torch.Tensor) or (output.shape != expected.shape or
            output.dtype != expected.dtype or output.device != expected.device):
        raise AssertionError('Packed expert bits have invalid shape/dtype/device')
    if not torch.equal(output, expected):
        raise AssertionError('Packed expert bits differ from the independent reference')


def unchanged(ids, pristine):
    import torch
    if not torch.equal(ids, pristine):
        raise AssertionError('Bitmatrix packing modified read-only input IDs')


def reference(harness, ids, experts):
    return harness.reference_pack_bitmatrix(ids.cpu(), experts).to(ids.device)


def perturb(ids, experts):
    ids.copy_((ids + 1).remainder(experts))


def output_span(output):
    """Conservative byte span touched by a tensor, including a strided view."""
    base = output.untyped_storage().data_ptr()
    first = last = output.storage_offset()
    for size, stride in zip(output.shape, output.stride()):
        extent = (size - 1) * stride
        first += min(0, extent)
        last += max(0, extent)
    element_size = output.element_size()
    return base + first * element_size, base + (last + 1) * element_size


def check_disjoint_outputs(outputs):
    # Most captured outputs have non-overlapping bounding spans. Only groups
    # whose spans intersect need the exact, stride-aware byte check below.
    spans = sorted(((*output_span(output), index) for index, output in enumerate(outputs)))

    def check_group(group):
        if len(group) < 2:
            return
        owners = {}
        for _, _, index in group:
            output = outputs[index]
            base = output.untyped_storage().data_ptr()
            element_size = output.element_size()
            for position in product(*(range(size) for size in output.shape)):
                element = output.storage_offset() + sum(
                    coordinate * stride for coordinate, stride in zip(position, output.stride()))
                first_byte = base + element * element_size
                for address in range(first_byte, first_byte + element_size):
                    previous = owners.setdefault(address, index)
                    if previous != index:
                        raise AssertionError(
                            f'Captured bitmatrix outputs {previous} and {index} overlap')

    group = []
    group_end = 0
    for span in spans:
        if group and span[0] >= group_end:
            check_group(group)
            group = []
        group.append(span)
        group_end = max(group_end, span[1])
    check_group(group)


@contextmanager
def checked_modules(harness):
    load_original = harness.load_module
    patched = []

    def load():
        module = load_original()
        original = module.pack_topk_to_bitmatrix
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

        module.pack_topk_to_bitmatrix = checked
        return module

    harness.load_module = load
    try:
        yield
    finally:
        harness.load_module = load_original
        for module, original in reversed(patched):
            module.pack_topk_to_bitmatrix = original


def checked_benchmark(harness, benchmark, fn, **kwargs):
    import torch
    c = inspect.getclosurevars(fn).nonlocals
    module, ids, experts = (c[key] for key in ('mod', 'topk_ids', 'num_experts'))
    pristine = ids.clone()
    expected = reference(harness, pristine, experts)
    original = module.pack_topk_to_bitmatrix
    captured = None
    returned = []

    def collect(*args):
        nonlocal captured
        captured = original(*args)
        returned.append(captured)
        return captured

    def measured():
        nonlocal captured
        captured = None
        fn()
        if captured is None:
            raise AssertionError('Benchmark did not invoke the bitmatrix candidate')
        return captured

    module.pack_topk_to_bitmatrix = collect
    try:
        timed = harness._TimedRun()
        ms, metadata = benchmark(measured, timed_run=timed, **kwargs)
        method = metadata.get('benchmark_method')
        kind = metadata.get('benchmark_timed_run_kind')
        repeats = metadata.get('benchmark_effective_repeats')
        if type(repeats) is not int or repeats < 1:
            raise AssertionError('Benchmark omitted a valid effective repeat count')
        if method == 'cuda_graph':
            if kind != 'captured_graph' or len(returned) < repeats:
                raise AssertionError('Final captured graph invocations are not observable')
            outputs = returned[-repeats:]
        elif method == 'cuda_event_fallback':
            if kind != 'eager_callable' or repeats != 1 or not returned:
                raise AssertionError('Event timing invocation is not observable')
            outputs = returned[-1:]
        else:
            raise AssertionError('Unrecognized device benchmark method')
        if outputs[-1] is not timed.outputs:
            raise AssertionError('Measured output is not the final captured invocation')
        unchanged(ids, pristine)
        for output in outputs:
            check_output(output, expected)
        check_disjoint_outputs(outputs)
        perturb(ids, experts)
        replay_pristine = ids.clone()
        replay_expected = reference(harness, replay_pristine, experts)
        for output in outputs:
            output.fill_(0xFFFFFFFF)
        replayed = timed.rerun()
        unchanged(ids, replay_pristine)
        if method == 'cuda_graph' and replayed is not outputs[-1]:
            raise AssertionError('Replay did not return the final captured output')
        check_output(replayed, replay_expected)
        if method == 'cuda_graph':
            for output in outputs:
                check_output(output, replay_expected)
        return ms, {**metadata, 'timed_output_checked': True,
                    'perturbed_input_replay_checked': True, 'source_buffers_unchanged': True,
                    'timed_invocation_outputs_checked': len(outputs),
                    'timed_invocation_outputs_disjoint': True}
    finally:
        module.pack_topk_to_bitmatrix = original
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
