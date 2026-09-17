"""Supplement original random cases with deterministic clamp-boundary probes."""

# Representable BF16 values on both sides of gate=+7 and linear=+/-7.
SATURATION_GATE = (8., 16., 2., 2., -8., 7., 7.0625, 6.9375, -7., -7.0625)
SATURATION_LINEAR = (2., 0., 8., -8., 2., -8., 7.0625, -7.0625, 2., 2.)


def saturation_input_(inp):
    """Keep the original shape/layout; change inputs only outside timing."""
    import torch
    inp.neg_()
    width = inp.shape[-1] // 2
    count = min(width, len(SATURATION_GATE))
    inp[:, :count] = torch.tensor(SATURATION_GATE[:count], dtype=inp.dtype, device=inp.device)
    inp[:, width:width + count] = torch.tensor(SATURATION_LINEAR[:count], dtype=inp.dtype, device=inp.device)


def check_saturation(h):
    """Run baseline/reference/candidate controls on every original case shape."""
    import torch
    mmod = h._load_module(h._KERNEL_DIR, h.MODEL_FILE, "saturation_model")
    model = mmod.Model(*mmod.get_init_inputs())
    candidate = None
    if not h.ARENA_PROVIDED_BASELINE:
        kmod = h._load_module(h._KERNEL_DIR, h.KERNEL_FILE, "saturation_candidate")
        candidate = getattr(kmod, h.KERNEL_ENTRY)
    for shape in h.SHAPES:
        inp = h._make_inputs(shape)
        saturation_input_(inp)
        original = inp.clone()
        with torch.no_grad():
            expected = h._checked_activation_output(h._aiter_op(inp), inp)
            actual = h._checked_activation_output(model(inp), inp)
            h.require_unchanged((inp,), (original,))
            h.normalized_output(actual, expected, tolerance=h.REL_TOL)
            if candidate is not None:
                result = h._checked_activation_output(candidate(inp), inp)
                h.require_unchanged((inp,), (original,))
                # Preserve the original candidate-vs-model normalized-error rule.
                h.normalized_output(result, actual, tolerance=h.REL_TOL)
        print(f"  saturation control PASS: {shape['name']} (gate upper / linear lower+upper)")
