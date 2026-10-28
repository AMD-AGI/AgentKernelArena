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


def verify_timed_run(timed, *, inputs, originals, expected, perturb, reference, compare, single_invocations=0):
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
    perturbations = (perturb,) if callable(perturb) else tuple(perturb)
    if not perturbations or not all(callable(change) for change in perturbations):
        raise ValueError("Replay requires callable input probes")
    perturbations += (nonuniform_input_probes(*inputs) if single_invocations else ())
    for change in perturbations:
        try:
            change()
            changed = tuple(x.detach().clone() for x in inputs)
            expected_replay = reference()
            if not isinstance(timed.outputs, torch.Tensor):
                raise AssertionError("Measured output must be a Tensor")
            check_single_invocations(timed, count=single_invocations, inputs=inputs,
                                     expected=expected_replay, compare=compare,
                                     poison=lambda output: output.fill_(float("nan")))
            if single_invocations:
                # Return to original inputs through isolated calls, then change
                # them again before the scored graph. Fresh captures cannot prime
                # the graph with the probed operands; grouped replay cannot hide
                # a stale first isolated call either. All controls remain unscored.
                for value, original in zip(inputs, originals):
                    value.copy_(original)
                check_single_invocations(timed, count=single_invocations, inputs=inputs,
                                         expected=reference(), compare=compare,
                                         poison=lambda output: output.fill_(float("nan")))
                change()
            timed.outputs.fill_(float("nan"))
            actual_replay = timed.rerun()
            require_unchanged(inputs, changed)
            compare(actual_replay, expected_replay)
        finally:
            # Subsequent diagnostic timings see the original declared input, too.
            for value, original in zip(inputs, originals):
                value.copy_(original)
    return {"timed_output_correctness": "PASS", "replay_correctness": "PASS",
            "single_invocations_per_probe": single_invocations,
            "restored_input_single_invocations_per_probe": single_invocations,
            "replay_inputs_perturbed": True, "replay_output_poisoned": True}


def check_single_invocations(timed, *, count, inputs, expected, compare, poison):
    """Supplement exact scored replay checks with complete one-call outputs.

    Two independent one-call captures expose alternating host-call omission;
    poisoned outputs expose partial writers. This control is never scored.
    Callers must also validate the actual measured/replayed unit separately.
    """
    import torch
    if type(count) is not int or count < 0 or count == 1:
        raise ValueError("Use zero or at least two isolated invocation checks")
    if not count:
        return 0
    invoke = getattr(timed, "single_invocation", None)
    if not callable(invoke):
        raise RuntimeError("Collector lacks isolated logical-call support")
    for _ in range(count):
        snapshots = tuple(value.detach().clone() for value in inputs)
        poison(timed.outputs)
        actual = invoke(before_replay=poison)
        if any(value.shape != original.shape or value.dtype != original.dtype
               or value.device != original.device
               or not torch.equal(value.contiguous().view(torch.uint8), original.contiguous().view(torch.uint8))
               for value, original in zip(inputs, snapshots)):
            raise AssertionError("Operator modified a read-only input during an isolated call")
        compare(actual, expected)
    return count


def nonuniform_input_probes(*tensors):
    """Change one floating operand's contents without just rescaling the whole.

    Alternating columns retain their original magnitude while the other columns
    are halved. This changes quantized code patterns, not just blockwise scales.
    No RNG, layout, shape, dtype or scored workload is changed.
    """
    def change(value):
        # A broadcast read-only scale can have zero strides. Change each
        # physical element once without changing its declared view/layout.
        target = value
        for dim in range(value.ndim - 1, -1, -1):
            if target.shape[dim] > 1 and target.stride(dim) == 0:
                target = target.select(dim, 0)
        values = target.float()
        if target.ndim and target.shape[-1] > 1:
            values[..., ::2] *= 0.5
        else:
            values *= 0.5
        target.copy_(values.to(target.dtype))
    return tuple(lambda value=value: change(value) for value in tensors
                 if value.is_floating_point() and bool(value.float().ne(0).any()))
