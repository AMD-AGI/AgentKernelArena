"""Task-owned checks on measured outputs and the exact timed invocation.

Numerical policies are supplied by the operator harness. All checking, oracle
work, input perturbation and restoration occur after timing, on both roles.
The canonical benchmark owns graph/Event collection and sample boundaries.
"""


def require_tensor_contract(actual, expected, *, dtype=None):
    import torch

    if not isinstance(actual, torch.Tensor):
        raise AssertionError("Operator must return a Tensor")
    if actual.shape != expected.shape:
        raise AssertionError(f"Output shape {tuple(actual.shape)} != {tuple(expected.shape)}")
    if actual.dtype != (expected.dtype if dtype is None else dtype):
        raise AssertionError(f"Output dtype {actual.dtype} violates the operator contract")
    if actual.device != expected.device:
        raise AssertionError(f"Output device {actual.device} != {expected.device}")


def allclose_output(actual, expected, *, atol, rtol, dtype=None):
    import torch

    require_tensor_contract(actual, expected, dtype=dtype)
    if not bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()):
        raise AssertionError("Non-finite operator/reference output")
    # An explicit output dtype can differ from the oracle's accumulation dtype.
    compared = actual.to(expected.dtype)
    if not torch.allclose(compared, expected, atol=atol, rtol=rtol):
        error = (compared - expected).abs().max().item()
        raise AssertionError(f"Numerical mismatch: max_abs_error={error}, atol={atol}, rtol={rtol}")


def normalized_output(actual, expected, *, tolerance):
    import torch

    require_tensor_contract(actual, expected)
    if not bool(torch.isfinite(actual).all() and torch.isfinite(expected).all()):
        raise AssertionError("Non-finite operator/reference output")
    error = (actual.float() - expected.float()).abs().max().item()
    scale = expected.float().abs().max().item() + 1e-9
    if error / scale > tolerance:
        raise AssertionError(f"Numerical mismatch: normalized_max_error={error / scale}, tolerance={tolerance}")


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
    try:
        require_unchanged(inputs, originals)
        compare(timed.outputs, expected)
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
