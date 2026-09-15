"""Validate measured output and exact replay outside unchanged timing boundaries."""

def require_unchanged(inputs, originals):
    import torch

    if len(inputs) != len(originals) or any(
        a.shape != b.shape or a.dtype != b.dtype or a.device != b.device
        or not torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
        for a, b in zip(inputs, originals)
    ):
        raise AssertionError("Operator modified a read-only input")

def verify_timed_run(timed, *, inputs, originals, expected, perturb, reference, compare):
    """Check last measured output, then perturb and replay the measured unit.

    A fresh ordinary correctness invocation cannot substitute for either check.
    Event support must expose the last actual measured output through TimedRun;
    an unsupported collector is an error, never a skipped validation.
    """
    import torch

    if not timed.bound:
        raise RuntimeError("Benchmark did not expose its measured invocation")
    require_unchanged(inputs, originals)
    compare(timed.outputs, expected)
    try:
        perturb()
        changed = tuple(x.detach().clone() for x in inputs)
        expected_replay = reference()
        if not isinstance(timed.outputs, torch.Tensor):
            raise AssertionError("Measured output must be a Tensor")
        timed.outputs.fill_(float("nan"))
        actual_replay = timed.rerun()
        require_unchanged(inputs, changed)
        compare(actual_replay, expected_replay)
    finally:
        # Subsequent diagnostic timings see the original declared input, too.
        for value, original in zip(inputs, originals):
            value.copy_(original)
    return {"timed_output_correctness": "PASS", "replay_correctness": "PASS",
            "replay_inputs_perturbed": True, "replay_output_poisoned": True}


def compare_output(actual, expected, tolerance):
    """Original maximum absolute-error gate, with explicit finite checks."""
    import torch

    if (not isinstance(actual, torch.Tensor) or actual.shape != expected.shape
            or actual.dtype != expected.dtype or actual.device != expected.device
            or not torch.isfinite(actual).all() or not torch.isfinite(expected).all()):
        raise AssertionError("Invalid paged attention output")
    error = (actual.float() - expected.float()).abs().max().item()
    if error > tolerance:
        raise AssertionError(f"Paged attention max error {error} exceeds {tolerance}")
