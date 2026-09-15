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
        weights.neg_()
    def poison(outputs):
        import torch
        for output in outputs:
            output.fill_(float('nan') if output.is_floating_point() else -1)
    return dict(inputs=inputs, originals=originals, expected=expected(), reference=expected,
                compare=compare, perturb=perturb, poison=poison)


def require_unchanged(inputs, originals):
    import torch
    for actual, original in zip(inputs, originals):
        if not torch.equal(actual.contiguous().view(torch.uint8), original.contiguous().view(torch.uint8)):
            raise AssertionError('Operator modified a read-only input')


def verify_timed_run(timed, *, inputs, originals, expected, reference, compare, perturb, poison):
    if not timed.bound:
        raise RuntimeError('Benchmark did not expose its measured invocation')
    require_unchanged(inputs, originals)
    compare(timed.outputs, expected)
    try:
        perturb()
        changed = tuple(value.clone() for value in inputs)
        ref = reference()
        poison(timed.outputs)
        outputs = timed.rerun()
        require_unchanged(inputs, changed)
        compare(outputs, ref)
    finally:
        for value, original in zip(inputs, originals):
            value.copy_(original)
    return {'timed_output_correctness': 'PASS', 'replay_correctness': 'PASS',
            'replay_inputs_perturbed': True, 'replay_output_poisoned': True}
