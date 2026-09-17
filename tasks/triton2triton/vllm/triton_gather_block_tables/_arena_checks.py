"""Check gather's caller-owned buffers and the actual captured copy operation."""
from contextlib import contextmanager
import inspect

SYMBOL = 'gather_block_tables'


def snapshots(inputs):
    return tuple(value.clone() for value in inputs)


def unchanged(inputs, pristine):
    import torch
    if any(not torch.equal(value, saved) for value, saved in zip(inputs, pristine)):
        raise AssertionError('Gather modified read-only mapping, source or counts')


def reference(harness, inputs, destination):
    import torch
    mapping, source, counts = inputs
    copied = harness.reference_gather(mapping.cpu(), source.cpu(), counts.cpu())
    expected = destination.clone()
    valid = torch.arange(source.shape[1])[None, :] < counts.cpu()[mapping.cpu().long(), None]
    active = expected[:mapping.numel()]
    active.copy_(torch.where(valid.to(active.device), copied.to(active.device), active))
    return expected


def check_output(output, destination, expected, num_reqs):
    import torch
    if not isinstance(output, torch.Tensor) or (output.shape != (num_reqs, destination.shape[1]) or
            output.dtype != destination.dtype or output.device != destination.device):
        raise AssertionError('Gather output shape/dtype/device is invalid')
    if output.data_ptr() != destination.data_ptr() or output.stride() != destination.stride():
        raise AssertionError('Gather must return the requested destination view')
    if not torch.equal(destination, expected) or not torch.equal(output, expected[:num_reqs]):
        raise AssertionError('Gather output or untouched destination region is incorrect')


@contextmanager
def checked_modules(harness):
    load_original = harness.load_module
    patched = []

    def load():
        module = load_original()
        original = getattr(module, SYMBOL)
        patched.append((module, original))
        diagnosed = False

        def verify(mapping, source, destination, counts):
            inputs = (mapping, source, counts)
            pristine = snapshots(inputs)
            expected = reference(harness, pristine, destination.clone())
            output = original(mapping, source, destination, counts)
            unchanged(inputs, pristine)
            check_output(output, destination, expected, mapping.numel())
            return output

        def checked(mapping, source, destination, counts):
            nonlocal diagnosed
            import torch
            result = verify(mapping, source, destination, counts)
            if not diagnosed:
                # Unscored: two tiles, a masked tail, duplicate/nonidentity
                # request routing, and a zero-length row retaining its sentinel.
                src = torch.arange(5*1031, dtype=source.dtype, device=source.device).reshape(5, 1031)
                dst = torch.full_like(src, -123456)
                idx = torch.tensor([4, 1, 4], dtype=mapping.dtype, device=mapping.device)
                lengths = torch.tensor([1, 0, 3, 1024, 1031], dtype=counts.dtype, device=counts.device)
                verify(idx, src, dst, lengths)
                diagnosed = True
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
    import torch
    c = inspect.getclosurevars(fn).nonlocals
    module = c['mod']
    inputs = tuple(c[key] for key in ('idx_mapping', 'src_block_table', 'num_blocks'))
    mapping, source, counts = inputs
    destination = c['dst_block_table']
    pristine = snapshots(inputs)
    initial_destination = destination.clone()
    expected = reference(harness, pristine, initial_destination)
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
            raise AssertionError('Benchmark did not invoke gather')
        return captured

    setattr(module, SYMBOL, collect)
    try:
        timed = harness._TimedRun()
        ms, metadata = benchmark(measured, timed_run=timed, **options)
        unchanged(inputs, pristine)
        check_output(timed.outputs, destination, expected, mapping.numel())
        source.add_(123456)
        mapping.add_(1).remainder_(source.shape[0])
        counts.copy_(torch.arange(counts.numel(), dtype=counts.dtype, device=counts.device) % (source.shape[1]+1))
        replay_pristine = snapshots(inputs)
        # Poison only declared writes. Unwritten rows/tails retain their actual
        # pre-replay contents and must remain untouched by the candidate.
        for row, req in enumerate(mapping.cpu().tolist()):
            destination[row, :int(counts[req].item())].fill_(torch.iinfo(destination.dtype).min)
        replay_expected = reference(harness, replay_pristine, destination.clone())
        output = timed.rerun()
        unchanged(inputs, replay_pristine)
        check_output(output, destination, replay_expected, mapping.numel())
        return ms, {**metadata, 'timed_output_checked': True,
                    'perturbed_input_replay_checked': True, 'source_buffers_unchanged': True}
    finally:
        setattr(module, SYMBOL, original)
        for value, saved in zip(inputs, pristine):
            value.copy_(saved)
        destination.copy_(initial_destination)


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
