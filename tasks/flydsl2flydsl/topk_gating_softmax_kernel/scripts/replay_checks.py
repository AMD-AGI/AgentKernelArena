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


def verify_topk_timed_run(timed, gating, original, topk, dtype_str, renormalize, reference):
    if not timed.bound:
        raise RuntimeError("Benchmark did not expose its measured routing invocation")
    require_unchanged(gating, original)
    require_topk_output(timed.outputs, gating, topk, dtype_str, renormalize, reference)
    try:
        gating.neg_()
        changed = gating.clone()
        timed.outputs[0].fill_(float("nan"))
        timed.outputs[1].fill_(-1)
        timed.outputs[2].fill_(-1)
        replayed = timed.rerun()
        require_unchanged(gating, changed)
        require_topk_output(replayed, gating, topk, dtype_str, renormalize, reference)
    finally:
        gating.copy_(original)
    return {"timed_output_correctness": "PASS", "replay_correctness": "PASS",
            "replay_inputs_perturbed": True, "replay_outputs_poisoned": True}
