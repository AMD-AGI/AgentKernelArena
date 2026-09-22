"""Check both MXFP4 payload and scales against the unchanged independent codec."""

def prepare_check(inputs, out_buf, out_scale, reference_silu, reference_mxfp4, decode):
    import torch
    values = tuple(value for value in inputs.values() if isinstance(value, torch.Tensor))
    originals = tuple(value.clone() for value in values)
    rows, width = inputs['rows'], inputs['inter_dim']
    if inputs['quant_mode'] != 'fp4':
        raise AssertionError('Unsupported quantization mode for the declared reference')
    output_shapes = (out_buf.shape, out_scale.shape)

    def reference():
        real = reference_silu(inputs['x'], width)
        dequantized, scales = reference_mxfp4(real, width // 32)
        return real, dequantized, scales

    def compare(actual, expected):
        if not isinstance(actual, tuple) or len(actual) != 2:
            raise AssertionError('Measured payload and scales are both required')
        for output, shape in zip(actual, output_shapes):
            if output.shape != shape or output.dtype != torch.uint8 or output.device != inputs['x'].device:
                raise AssertionError('Invalid measured quantized output')
        real, ref_deq, ref_scales = expected
        deq, scales = decode(*actual, rows, width)
        if not torch.isfinite(deq).all():
            raise AssertionError('Quantized output decoded to nonfinite values')
        if (scales != ref_scales).any():
            raise AssertionError('E8M0 block scales differ from the independent reference')
        mismatch = deq != ref_deq
        fraction = float(mismatch.float().mean())
        residual = (deq[mismatch] - real[mismatch]).abs()
        ref_residual = (ref_deq[mismatch] - real[mismatch]).abs()
        if fraction > 0.01 or (residual > ref_residual + 1e-6).any():
            raise AssertionError('MXFP4 grid/tie rule failed')

    def perturb():
        inputs['x'][:, width:].neg_()

    def poison(outputs):
        for output in outputs:
            output.bitwise_not_()

    return dict(inputs=values, originals=originals, expected=reference(),
                reference=reference, compare=compare, perturb=perturb, poison=poison)


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
        expected_replay = reference()
        poison(timed.outputs)
        output = timed.rerun()
        require_unchanged(inputs, changed)
        compare(output, expected_replay)
    finally:
        for value, original in zip(inputs, originals):
            value.copy_(original)
    return {'timed_output_correctness': 'PASS', 'replay_correctness': 'PASS',
            'replay_inputs_perturbed': True, 'replay_output_poisoned': True}
