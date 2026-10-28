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
    original_cache = tuple(value.clone() for value in initial_cache)
    def mapping():
        valid = inp['slot_mapping'] >= 0
        slots = inp['slot_mapping'][valid].long()
        return valid, slots // inp['BS'], slots % inp['BS']

    def reference():
        valid, blocks, offsets = mapping()
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

    def mapped_probe():
        slots = inp['slot_mapping']
        valid = slots >= 0
        capacity = inp['key_cache'].shape[0] * inp['BS']
        slots[valid] = (slots[valid] + 1) % capacity
        # Nonzero untouched cache state must survive the replay unchanged.
        for cache in initial_cache:
            cache.fill_(0.375)

    def phase_probe():
        inp['positions'].add_(17).remainder_(inp['cos_cache'].shape[0])

    def permuted_slots():
        slots = inp['slot_mapping']
        valid = slots >= 0
        slots[valid] = slots[valid].flip(0)

    def skipped_slots():
        inp["slot_mapping"][::2] = -1
        for cache in initial_cache:
            cache.fill_(0.375)

    def poison(outputs):
        _, blocks, offsets = mapping()
        for output, original in zip(outputs[2:], initial_cache):
            output.copy_(original)
        outputs[0].fill_(float('nan'))
        outputs[1].fill_(float('nan'))
        # The operator only writes mapped slots; the rest is preserved state.
        outputs[2][blocks, offsets] = float('nan')
        outputs[3][blocks, offsets] = float('nan')

    def restore():
        for reference_cache, output, original in zip(initial_cache,
                (inp['key_cache'], inp['value_cache']), original_cache):
            reference_cache.copy_(original)
            output.copy_(original)

    return dict(inputs=inputs, originals=originals, expected=reference(),
                reference=reference, compare=compare, perturb=(perturb, mapped_probe, phase_probe, permuted_slots, skipped_slots), poison=poison, restore=restore)


def verify_timed_run(timed, *, inputs, originals, expected, perturb, reference, compare,
                     extra_perturb=None, poison=None, restore=None, single_invocations=0):
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
            if restore is not None:
                restore()
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
