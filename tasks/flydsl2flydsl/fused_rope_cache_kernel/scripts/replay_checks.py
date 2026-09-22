"""Independent full-output checks around the unchanged RoPE/cache timed launch."""

def require_unchanged(inputs, originals):
    import torch
    for actual, original in zip(inputs, originals):
        if not torch.equal(actual.contiguous().view(torch.uint8), original.contiguous().view(torch.uint8)):
            raise AssertionError("Operator modified a read-only input")


def prepare_check(inp, reference_rope, atol, rtol):
    import torch
    names = ('Q', 'K', 'V', 'positions', 'cos_cache', 'sin_cache', 'slot_mapping', 'k_scale', 'v_scale')
    inputs = tuple(inp[name] for name in names)
    originals = tuple(value.clone() for value in inputs)
    initial_cache = (inp['key_cache'].clone(), inp['value_cache'].clone())
    valid = inp['slot_mapping'] >= 0
    slots = inp['slot_mapping'][valid].long()
    blocks, offsets = slots // inp['BS'], slots % inp['BS']

    def reference():
        q = reference_rope(inp['Q'], inp['cos_cache'], inp['sin_cache'], inp['positions'])
        k = reference_rope(inp['K'], inp['cos_cache'], inp['sin_cache'], inp['positions'])
        kc, vc = (value.clone() for value in initial_cache)
        kc[blocks, offsets] = k[valid]
        vc[blocks, offsets] = inp['V'][valid]
        return q, k, kc, vc

    def compare(actual, expected):
        if not isinstance(actual, tuple) or len(actual) != 4:
            raise AssertionError("All four measured outputs are required")
        for value, ref in zip(actual, expected):
            torch.testing.assert_close(value, ref, atol=atol, rtol=rtol)

    def perturb():
        for name in ('Q', 'K', 'V'):
            inp[name].neg_()

    def poison(outputs):
        outputs[0].fill_(float('nan'))
        outputs[1].fill_(float('nan'))
        # The operator only writes mapped slots; the rest is preserved state.
        outputs[2][blocks, offsets] = float('nan')
        outputs[3][blocks, offsets] = float('nan')

    return dict(inputs=inputs, originals=originals, expected=reference(),
                reference=reference, compare=compare, perturb=perturb, poison=poison)


def verify_timed_run(timed, *, inputs, originals, expected, reference, compare, perturb, poison):
    if not timed.bound:
        raise RuntimeError("Benchmark did not expose its measured invocation")
    require_unchanged(inputs, originals)
    compare(timed.outputs, expected)
    try:
        perturb()
        changed = tuple(value.clone() for value in inputs)
        expected_replay = reference()
        poison(timed.outputs)
        outputs = timed.rerun()
        require_unchanged(inputs, changed)
        compare(outputs, expected_replay)
    finally:
        for value, original in zip(inputs, originals):
            value.copy_(original)
    return {'timed_output_correctness': 'PASS', 'replay_correctness': 'PASS',
            'replay_inputs_perturbed': True, 'replay_output_poisoned': True}
