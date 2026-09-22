"""Unscored, all-valid MXFP4 routing checks using the protected reference codec.

Payload rows stay in input order; block scales belong to sorted positions.
All input changes, reference work, poisoning and verification are untimed.
"""


def input_rows_in_sorted_order(inputs):
    """Decode the public packed IDs, independently of the candidate launcher."""
    import torch

    rows, tokens = inputs['rows'], inputs['token_num']
    topk = rows // tokens
    packed = inputs['sorted_ids']
    if (rows != tokens * topk or packed.dtype != torch.int32
            or packed.shape != (rows,) or inputs['num_sorted_rows'] != rows
            or inputs['num_valid_ids'].numel() != 1
            or int(inputs['num_valid_ids'].item()) != rows):
        raise AssertionError('Routing control requires all-valid packed IDs')
    token, slot = packed.long() & 0xFFFFFF, packed.long() >> 24
    order = token * topk + slot
    if (bool((token >= tokens).any()) or bool(((slot < 0) | (slot >= topk)).any())
            or not torch.equal(order.sort().values, torch.arange(rows, device=order.device))):
        raise AssertionError('Routing control requires a bijection of valid token/slot IDs')
    return order


def set_routing_control_(inputs):
    """Replace values and IDs in place at the smallest original scored shape.

    33-row rotation crosses both slot and 32-row scale-tile boundaries. Distinct
    BF16-representable row amplitudes and varying signed payload patterns prevent
    input-order scale stores or sorted-order payload stores from passing.
    """
    import torch

    rows, width, tokens = inputs['rows'], inputs['inter_dim'], inputs['token_num']
    if (tokens, rows, width, inputs['quant_mode']) != (64, 128, 1024, 'fp4'):
        raise AssertionError('Routing control uses the existing 64-token FP4 shape')
    device = inputs['x'].device
    row = torch.arange(rows, device=device, dtype=torch.long)[:, None]
    col = torch.arange(width, device=device, dtype=torch.long)[None, :]
    amplitude = torch.pow(2.0, (row % 8 - 4).float()) * (1.0 + (row // 8).float() / 16.0)
    block_amplitude = torch.pow(2.0, ((col // 32) % 4 - 2).float())
    pattern = torch.tensor([1., -.9375, .6875, -.625, .3125, -.1875, .0625, 0.], device=device)
    up = amplitude * block_amplitude * pattern[(col + row + col // 32) % len(pattern)]
    inputs['x'][:, :width].fill_(1.0)
    inputs['x'][:, width:].copy_(up)
    order = (torch.arange(rows, device=device, dtype=torch.long) + 33) % rows
    topk = rows // tokens
    packed = (order // topk) | ((order % topk) << 24)
    inputs['sorted_ids'].copy_(packed)
    # num_valid_ids, allocation addresses, scalar launch arguments and the
    # unspecified scale padding are unchanged; no partial-valid policy is added.


def routing_reference(inputs, reference_silu, reference_mxfp4):
    """Expected values at sorted positions, with input-order payload semantics."""
    order = input_rows_in_sorted_order(inputs)
    real = reference_silu(inputs['x'], inputs['inter_dim'])
    dequantized, input_scales = reference_mxfp4(real, inputs['scale_cols'])
    sorted_scales = input_scales[order.cpu().numpy()]
    return order, real.index_select(0, order), dequantized.index_select(0, order), sorted_scales


def check_routing_call(inputs, outputs, invoke, reference_silu, reference_mxfp4, decode):
    """Execute the selected implementation, then apply the unchanged MXFP4 rule."""
    import torch
    from scripts.replay_checks import require_unchanged

    values = tuple(value for value in inputs.values() if isinstance(value, torch.Tensor))
    originals = tuple(value.clone() for value in values)
    shapes = tuple(output.shape for output in outputs)
    try:
        set_routing_control_(inputs)
        changed = tuple(value.clone() for value in values)
        order, real, ref_deq, ref_scales = routing_reference(inputs, reference_silu, reference_mxfp4)
        # Assert the control itself distinguishes the two coordinate systems.
        _, input_scales = reference_mxfp4(reference_silu(inputs['x'], inputs['inter_dim']), inputs['scale_cols'])
        if not (input_scales != ref_scales).any():
            raise AssertionError('Routing control did not distinguish sorted and input scales')
        for output in outputs:
            output.bitwise_not_()
        actual = invoke()
        require_unchanged(values, changed)
        if not isinstance(actual, tuple) or len(actual) != 2:
            raise AssertionError('Routing control requires both payload and scales')
        for output, shape in zip(actual, shapes):
            if (not isinstance(output, torch.Tensor) or output.shape != shape
                    or output.dtype != torch.uint8 or output.device != inputs['x'].device):
                raise AssertionError('Invalid routing-control output contract')
        # Only gather payload rows: scales are already in sorted-position order.
        # Gathering both tensors here would silently accept input-order scales.
        payload, sorted_scales = actual
        deq, scales = decode(payload.index_select(0, order), sorted_scales,
                             inputs['rows'], inputs['inter_dim'])
        if not torch.isfinite(deq).all():
            raise AssertionError('Routing payload decoded to nonfinite values')
        if (scales != ref_scales).any():
            raise AssertionError('Routing E8M0 scales differ from sorted-position reference')
        mismatch = deq != ref_deq
        fraction = float(mismatch.float().mean())
        residual = (deq[mismatch] - real[mismatch]).abs()
        ref_residual = (ref_deq[mismatch] - real[mismatch]).abs()
        if fraction > 0.01 or (residual > ref_residual + 1e-6).any():
            raise AssertionError('Routing MXFP4 grid/tie rule failed')
    finally:
        for value, original in zip(values, originals):
            value.copy_(original)
    return {'routing_correctness': 'PASS', 'routing_control': 'all_valid_rotate33',
            'routing_payload_order': 'input', 'routing_scale_order': 'sorted',
            'routing_input_restored': True, 'routing_output_poisoned': True}


def verify_routing_replay(timed, inputs, reference_silu, reference_mxfp4, decode):
    if not timed.bound:
        raise RuntimeError('Routing check requires the actual measured invocation')
    result = check_routing_call(inputs, timed.outputs, timed.rerun,
                                reference_silu, reference_mxfp4, decode)
    return {**result, 'routing_timed_replay': 'PASS'}
