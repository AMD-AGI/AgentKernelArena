"""Check all routing outputs and the captured replay outside device timing."""


def require_topk_output(outputs, gating, topk, dtype_str, renormalize, reference):
    import torch

    if not isinstance(outputs, tuple) or len(outputs) != 3:
        raise AssertionError("Routing must expose weights, indices and token/expert indices")
    weights, indices, token_indices = outputs
    shape = (gating.shape[0], topk)
    for value, dtype in zip(outputs, (torch.float32, torch.int32, torch.int32)):
        if not isinstance(value, torch.Tensor) or value.shape != shape:
            raise AssertionError("Routing output shape mismatch")
        if value.dtype != dtype or value.device != gating.device:
            raise AssertionError("Routing output dtype/device mismatch")
    probs, ref_weights, _, ref_token_indices = reference(gating.float(), topk, renormalize)
    if not bool(torch.isfinite(weights).all() and torch.isfinite(ref_weights).all()):
        raise AssertionError("Nonfinite routing weights")
    if bool(((indices < 0) | (indices >= gating.shape[1])).any()):
        raise AssertionError("Routing expert index outside input")
    sorted_indices = indices.sort(dim=1).values
    if topk > 1 and bool((sorted_indices[:, 1:] == sorted_indices[:, :-1]).any()):
        raise AssertionError("Duplicate routing expert")
    selected = probs.gather(1, indices.long())
    # Preserve the original low-precision tie policy and numerical tolerances.
    prob_tol = 1e-3 if dtype_str in ("bf16", "f16") else 1e-6
    atol_weight = 2e-2 if dtype_str in ("bf16", "f16") else 1e-5
    threshold = probs.topk(topk, dim=1).values[:, -1:]
    if bool((selected < threshold - prob_tol).any()):
        raise AssertionError("Selected expert is below the top-K threshold")
    if (weights.sort(dim=1).values - ref_weights.sort(dim=1).values).abs().max().item() > atol_weight:
        raise AssertionError("Routing weight mismatch")
    # Check association as well as the original unordered weight comparison.
    associated = selected / selected.sum(dim=1, keepdim=True).clamp(min=1e-20) if renormalize else selected
    if (weights - associated).abs().max().item() > atol_weight:
        raise AssertionError("Routing weights do not match selected experts")
    if not torch.equal(token_indices, ref_token_indices):
        raise AssertionError("Token/expert index mismatch")
    if renormalize and (weights.sum(dim=1) - 1.0).abs().max().item() > atol_weight:
        raise AssertionError("Routing weights are not normalized")


def require_unchanged(actual, original):
    import torch

    if not torch.equal(actual.contiguous().view(torch.uint8), original.contiguous().view(torch.uint8)):
        raise AssertionError("Routing modified its read-only input")


def verify_topk_timed_run(timed, gating, original, topk, dtype_str, renormalize, reference, *, single_invocations=0):
    if not timed.bound:
        raise RuntimeError("Benchmark did not expose its measured routing invocation")
    require_unchanged(gating, original)
    require_topk_output(timed.outputs, gating, topk, dtype_str, renormalize, reference)
    for change in (lambda: gating.neg_(), lambda: gating.add_(1000.0)) + (nonuniform_input_probes(gating) if single_invocations else ()):
        try:
            change()
            changed = gating.clone()
            check_single_invocations(timed, count=single_invocations, inputs=(gating,), expected=None,
                compare=lambda actual, _: require_topk_output(actual, gating, topk, dtype_str, renormalize, reference),
                poison=lambda output: (output[0].fill_(float("nan")), output[1].fill_(-1), output[2].fill_(-1)))
            if single_invocations:
                # Return to original inputs through isolated calls, then change
                # them again before the scored graph. Fresh captures cannot prime
                # the graph with the probed operands; grouped replay cannot hide
                # a stale first isolated call either. All controls remain unscored.
                gating.copy_(original)
                check_single_invocations(timed, count=single_invocations, inputs=(gating,), expected=None,
                    compare=lambda actual, _: require_topk_output(actual, gating, topk, dtype_str, renormalize, reference),
                    poison=lambda output: (output[0].fill_(float("nan")), output[1].fill_(-1), output[2].fill_(-1)))
                change()
            timed.outputs[0].fill_(float("nan"))
            timed.outputs[1].fill_(-1)
            timed.outputs[2].fill_(-1)
            replayed = timed.rerun()
            require_unchanged(gating, changed)
            require_topk_output(replayed, gating, topk, dtype_str, renormalize, reference)
            # Snapshot the independent reference's current numerical expectation.
            # The task comparator recomputes it from immutable current input bytes.
        finally:
            gating.copy_(original)
    return {"timed_output_correctness": "PASS", "replay_correctness": "PASS", "single_invocations_per_probe": single_invocations,
            "restored_input_single_invocations_per_probe": single_invocations,
            "replay_inputs_perturbed": True, "replay_outputs_poisoned": True}


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
