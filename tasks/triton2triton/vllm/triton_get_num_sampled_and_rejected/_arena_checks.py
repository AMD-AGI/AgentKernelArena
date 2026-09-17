"""Check sampled/rejected outputs, explicit input state and prepared replay."""
from contextlib import contextmanager
import inspect

SYMBOL = 'get_num_sampled_and_rejected'


def snapshots(inputs):
    return tuple(value.clone() for value in inputs)


def unchanged(inputs, pristine):
    import torch
    if any(not torch.equal(value, saved) for value, saved in zip(inputs, pristine)):
        raise AssertionError('Sample/rejection counter modified read-only inputs')


def reference(harness, inputs):
    outputs = harness.reference_get_num_sampled_and_rejected(*(value.cpu() for value in inputs))
    return tuple(value.to(inputs[0].device) for value in outputs)


def check_outputs(outputs, expected, num_sampled):
    import torch
    if not isinstance(outputs, tuple) or len(outputs) != 2:
        raise AssertionError('Counter must return both sampled and rejected counts')
    for value, wanted in zip(outputs, expected):
        if not isinstance(value, torch.Tensor) or (value.shape != wanted.shape or
                value.dtype != wanted.dtype or value.device != wanted.device):
            raise AssertionError('Count output shape/dtype/device is invalid')
        if not torch.equal(value, wanted):
            raise AssertionError('Count output differs from pristine-input reference')
    if not torch.equal(num_sampled, expected[0]):
        raise AssertionError('The in-place sampled-count update is incorrect')


def diagnostic_inputs(device):
    import torch
    rows = ([1, 2, 1, 3], [10, 20, 30, 40], [0, 2, 5, 6, 10],
            [2, 0, 3, 1], [21, 39, 10, 31])
    return tuple(torch.tensor(row, dtype=torch.int32, device=device) for row in rows)


@contextmanager
def checked_modules(harness):
    load_original = harness.load_module
    patched = []

    def load():
        module = load_original()
        original = getattr(module, SYMBOL)
        patched.append((module, original))

        def checked(num_sampled, seq_lens, cu_num_logits, idx_mapping, prefill_len):
            inputs = (num_sampled, seq_lens, cu_num_logits, idx_mapping, prefill_len)
            pristine = snapshots(inputs)
            expected = reference(harness, pristine)
            diagnostic = diagnostic_inputs(num_sampled.device)
            saved = snapshots(diagnostic)
            wanted = reference(harness, saved)
            check_outputs(original(*diagnostic), wanted, diagnostic[0])
            unchanged(diagnostic[1:], saved[1:])
            outputs = original(*inputs)
            unchanged(inputs[1:], pristine[1:])
            check_outputs(outputs, expected, num_sampled)
            return outputs

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
    work = c['num_sampled_work']
    prepare = options['prepare_fn']
    seed = inspect.getclosurevars(prepare).nonlocals['num_sampled']
    inputs = (seed, c['seq_lens'], c['cu_num_logits'], c['idx_mapping'], c['prefill_len'])
    pristine = snapshots(inputs)
    work_initial = work.clone()
    expected = reference(harness, pristine)
    try:
        timed = harness._TimedRun()
        # This original lambda already returns both buffers; preserve it and
        # its original prepare_fn/target_ms/warmups/sample settings unchanged.
        ms, metadata = benchmark(fn, timed_run=timed, **options)
        unchanged(inputs, pristine)
        check_outputs(timed.outputs, expected, work)
        seed, seq, cu, mapping, prefill = inputs
        counts = cu[1:] - cu[:-1]
        seed.copy_(seed.remainder(counts) + 1)
        mapping.copy_(torch.arange(mapping.numel(), device=mapping.device, dtype=mapping.dtype).flip(0))
        thresholds = seq + torch.where(torch.arange(seq.numel(), device=seq.device) % 2 == 0, 1, -1)
        prefill[mapping.long()] = thresholds.to(prefill.dtype)
        replay_pristine = snapshots(inputs)
        replay_expected = reference(harness, replay_pristine)
        for output in timed.outputs:
            output.fill_(-2)
        # TimedRun reruns the same preparation before the captured invocation.
        replayed = timed.rerun()
        unchanged(inputs, replay_pristine)
        check_outputs(replayed, replay_expected, work)
        return ms, {**metadata, 'timed_output_checked': True,
                    'perturbed_input_replay_checked': True, 'source_buffers_unchanged': True,
                    'in_place_sampled_state_checked': True}
    finally:
        for value, saved in zip(inputs, pristine):
            value.copy_(saved)
        work.copy_(work_initial)


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
