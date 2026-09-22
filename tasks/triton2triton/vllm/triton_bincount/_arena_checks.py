"""Validate both atomic outputs against pristine tokens, including timed replay."""
from contextlib import contextmanager
import inspect


def snapshots(values):
    return tuple(value.clone() for value in values)


def unchanged(values, saved):
    import torch
    for value, original in zip(values, saved):
        if not torch.equal(value, original):
            raise AssertionError("Bincount modified a read-only input")


def reference(inputs, initial):
    """Independent CPU histograms and bit packing; preserve inactive rows."""
    import torch
    mapping, tokens, prompt_lengths, prefill_lengths = (v.cpu() for v in inputs)
    masks, counts = (v.cpu().clone() for v in initial)
    vocab = counts.shape[1]
    masks[mapping.long()] = 0
    counts[mapping.long()] = 0
    weights = 1 << torch.arange(32, dtype=torch.int64)
    for request in mapping.tolist():
        prompt, prefill = int(prompt_lengths[request]), int(prefill_lengths[request])
        presence = torch.bincount(tokens[request, :prompt].long(), minlength=vocab) > 0
        padded = torch.zeros(masks.shape[1] * 32, dtype=torch.int64)
        padded[:vocab] = presence
        packed = (padded.reshape(-1, 32) * weights).sum(-1).to(torch.int32)
        masks[request] |= packed
        counts[request] += torch.bincount(tokens[request, prompt:prefill].long(), minlength=vocab).to(torch.int32)
    return tuple(value.to(device=before.device) for value, before in zip((masks, counts), initial))


def check_outputs(outputs, expected):
    import torch
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 2:
        raise AssertionError("Both bincount output buffers must be checked")
    for output, answer in zip(outputs, expected):
        if not isinstance(output, torch.Tensor) or (output.shape != answer.shape or
                output.dtype != answer.dtype or output.device != answer.device):
            raise AssertionError("Bincount output shape/dtype/device is invalid")
        if not torch.equal(output, answer):
            raise AssertionError("Bincount output differs from the pristine integer reference")


@contextmanager
def checked_modules(harness):
    loader = harness.load_module
    patched = []

    def load():
        module = loader()
        original = module.bincount
        patched.append((module, original))
        diagnosed = False

        def verify(mapping, tokens, prompt, prefill, mask, counts, maximum):
            inputs = (mapping, tokens, prompt, prefill)
            pristine = snapshots(inputs)
            expected = reference(pristine, (mask, counts))
            result = original(mapping, tokens, prompt, prefill, mask, counts, maximum)
            unchanged(inputs, pristine)
            check_outputs((mask, counts), expected)
            return result

        def checked(mapping, tokens, prompt, prefill, mask, counts, maximum):
            nonlocal diagnosed
            import torch
            result = verify(mapping, tokens, prompt, prefill, mask, counts, maximum)
            if not diagnosed:
                # Unscored partial request mapping, empty prompt/prefill, bit31,
                # bit32/64, and a prompt crossing the 1024-token launch block.
                device = tokens.device
                dt = (torch.arange(4 * 1031, device=device).reshape(4, 1031) % 65).to(tokens.dtype)
                dm = torch.tensor([3, 1, 0], dtype=mapping.dtype, device=device)
                dp = torch.tensor([0, 0, 7, 1025], dtype=prompt.dtype, device=device)
                df = torch.tensor([17, 0, 19, 1030], dtype=prefill.dtype, device=device)
                masks = torch.full((4, 3), 12345, dtype=mask.dtype, device=device)
                histograms = torch.full((4, 65), 17, dtype=counts.dtype, device=device)
                verify(dm, dt, dp, df, masks, histograms, 1031)
                diagnosed = True
            return result

        module.bincount = checked
        return module

    harness.load_module = load
    try:
        yield
    finally:
        harness.load_module = loader
        for module, original in reversed(patched):
            module.bincount = original


def checked_benchmark(harness, benchmark, fn, **options):
    state = inspect.getclosurevars(fn).nonlocals
    inputs = tuple(state[k] for k in ("idx_mapping", "all_token_ids", "prompt_len", "prefill_len"))
    outputs = tuple(state[k] for k in ("prompt_mask", "output_counts"))
    pristine, output_before = snapshots(inputs), snapshots(outputs)
    empty = tuple(value.new_zeros(value.shape) for value in outputs)
    expected = reference(pristine, empty)

    def measured():
        fn()
        return outputs

    try:
        timed = harness._TimedRun()
        ms, metadata = benchmark(measured, timed_run=timed, **options)
        unchanged(inputs, pristine)
        check_outputs(timed.outputs, expected)
        mapping, tokens, prompt, prefill = inputs
        tokens.add_(1).remainder_(outputs[1].shape[1])
        mapping.add_(1).remainder_(tokens.shape[0])
        lengths = prompt.new_tensor(range(prompt.numel())) % (tokens.shape[1] + 1)
        prompt.copy_(lengths // 2)
        prefill.copy_(lengths)
        replay_inputs = snapshots(inputs)
        replay_expected = reference(replay_inputs, empty)
        # Atomic OR/add require zero initial state. TimedRun reuses the exact
        # original prepare_fn before replay; do not poison after that reset.
        replayed = timed.rerun()
        unchanged(inputs, replay_inputs)
        check_outputs(replayed, replay_expected)
        return ms, {**metadata, "timed_output_checked": True,
                    "perturbed_input_replay_checked": True, "source_buffers_unchanged": True,
                    "original_atomic_reset_replay_checked": True}
    finally:
        for value, saved in zip(inputs + outputs, pristine + output_before):
            value.copy_(saved)


def install(harness):
    correctness = harness.run_correctness
    performance = harness.run_performance

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

    harness.run_correctness = checked_correctness
    harness.run_performance = checked_performance
