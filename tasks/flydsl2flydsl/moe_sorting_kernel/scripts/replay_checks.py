"""Check all defined MoE sorting outputs, including payload association and zeroed state."""

def compare_outputs(outputs, expected, *, token_count, topk, unit_size, model_dim=4096):
    import torch
    if not isinstance(outputs, tuple) or len(outputs) != 5:
        raise AssertionError('All five MoE outputs are required')
    ids, weights, experts, valid, buffer = outputs
    ref_ids, ref_weights, ref_experts, ref_valid = expected
    for value, reference in zip(outputs[:4], expected):
        if value.shape != reference.shape or value.dtype != reference.dtype or value.device != reference.device:
            raise AssertionError('Malformed sorting output')
    if not torch.equal(valid, ref_valid):
        raise AssertionError('Invalid padded/token counts')
    if buffer.shape != (token_count, model_dim) or buffer.dtype != torch.bfloat16 or buffer.device != ids.device:
        raise AssertionError('Malformed MoE buffer')
    if not (buffer.contiguous().view(torch.uint8) == 0).all():
        raise AssertionError('MoE buffer was not zeroed')
    padded = int(ref_valid[0])
    if padded < 0 or padded > ids.numel() or padded % unit_size:
        raise AssertionError('Invalid reference padding')
    blocks = padded // unit_size
    if not torch.equal(experts[:blocks], ref_experts[:blocks]):
        raise AssertionError('Invalid expert block assignment')
    sentinel = (topk << 24) | token_count
    # Scatter order may differ within each expert. Compare the complete packed
    # ID multiset and its associated weights, not only independent aggregates.
    for expert in ref_experts[:blocks].unique():
        block_mask = ref_experts[:blocks] == expert
        selected = block_mask.repeat_interleave(unit_size)
        actual_ids, expected_ids = ids[:padded][selected], ref_ids[:padded][selected]
        actual_weights, expected_weights = weights[:padded][selected], ref_weights[:padded][selected]
        keep_actual, keep_expected = actual_ids != sentinel, expected_ids != sentinel
        actual_ids, order_actual = actual_ids[keep_actual].sort()
        expected_ids, order_expected = expected_ids[keep_expected].sort()
        if not torch.equal(actual_ids, expected_ids):
            raise AssertionError('Missing, repeated, or misplaced token route')
        actual_weights = actual_weights[keep_actual][order_actual]
        expected_weights = expected_weights[keep_expected][order_expected]
        if not torch.isfinite(actual_weights).all():
            raise AssertionError('Nonfinite routed weight')
        if actual_weights.numel() and not (actual_weights - expected_weights).abs().max().item() < 1e-5:
            raise AssertionError('Weight does not match its routed token')


def prepare_check(ids, weights, experts, unit_size, reference):
    inputs = (ids, weights)
    originals = tuple(value.clone() for value in inputs)
    def expected():
        return reference(ids, weights, experts, unit_size)
    def compare(outputs, ref):
        compare_outputs(outputs, ref, token_count=ids.shape[0], topk=ids.shape[1], unit_size=unit_size)
    def perturb():
        ids.copy_((ids + 1) % experts)
        weights.mul_(0.5)
    def poison(outputs):
        import torch
        for output in outputs:
            output.fill_(float('nan') if output.is_floating_point() else -1)
    def concentrated():
        import torch
        ids.copy_(torch.arange(ids.shape[1], device=ids.device, dtype=ids.dtype).expand_as(ids))
    return dict(inputs=inputs, originals=originals, expected=expected(), reference=expected,
                compare=compare, perturb=(perturb, concentrated), poison=poison)


def require_unchanged(inputs, originals):
    import torch
    for actual, original in zip(inputs, originals):
        if not torch.equal(actual.contiguous().view(torch.uint8), original.contiguous().view(torch.uint8)):
            raise AssertionError('Operator modified a read-only input')


def verify_timed_run(timed, *, inputs, originals, expected, perturb, reference, compare,
                     extra_perturb=None, poison=None, single_invocations=0):
    """Validate measured output and independent probes on the measured unit.

    Every probe starts from original input bytes and recomputes the oracle before
    a poisoned-output replay. Probes and oracle work never enter scored timing.
    """
    import torch

    if not timed.bound:
        raise RuntimeError("Benchmark did not expose its measured invocation")
    require_unchanged(inputs, originals)
    compare(timed.outputs, expected)
    perturbations = (perturb,) if callable(perturb) else tuple(perturb)
    if extra_perturb is not None:
        perturbations += (extra_perturb,)
    if not perturbations or not all(callable(change) for change in perturbations):
        raise ValueError("Replay requires at least one callable input probe")
    if single_invocations:
        perturbations += nonuniform_input_probes(*inputs)
    for change in perturbations:
        try:
            change()
            changed = tuple(x.detach().clone() for x in inputs)
            if single_invocations and all(torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
                   for a, b in zip(inputs, originals)):
                raise AssertionError("Replay input probe did not change any input")
            expected_replay = reference()
            if poison is None:
                raise ValueError("Structured outputs require their task-owned poison callback")
            check_single_invocations(timed, count=single_invocations, inputs=inputs,
                                     expected=expected_replay, compare=compare,
                                     poison=poison)
            if single_invocations:
                # Return to original inputs through isolated calls, then change
                # them again before the scored graph. Fresh captures cannot prime
                # the graph with the probed operands; grouped replay cannot hide
                # a stale first isolated call either. All controls remain unscored.
                for value, original in zip(inputs, originals):
                    value.copy_(original)
                check_single_invocations(timed, count=single_invocations, inputs=inputs,
                                         expected=reference(), compare=compare,
                                         poison=poison)
                change()
            poison(timed.outputs)
            actual_replay = timed.rerun()
            require_unchanged(inputs, changed)
            compare(actual_replay, expected_replay)
        finally:
            for value, original in zip(inputs, originals):
                value.copy_(original)
    return {"timed_output_correctness": "PASS", "replay_correctness": "PASS",
            "single_invocations_per_probe": single_invocations,
            "restored_input_single_invocations_per_probe": single_invocations,
            "replay_inputs_perturbed": True, "replay_output_poisoned": True,
            "replay_input_draws": len(perturbations)}


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
