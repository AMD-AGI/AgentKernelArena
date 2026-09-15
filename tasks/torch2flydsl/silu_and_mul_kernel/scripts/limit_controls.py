"""Additional positive-limit contract probes; formal timing keeps LIMIT=0."""
LIMIT_PROBES = (2.0, 2.03, 7.0)
GATE_VALUES = (8., -8., 3., 1., 0., 16., 2., -2.)
UP_VALUES = (8., -8., -16., 3., 2., 1., 0., -3.)


def check_positive_limits(h):
    import torch
    mmod = h._load_module(h._KERNEL_DIR, h.MODEL_FILE, "limit_model")
    target = None
    if not h.ARENA_PROVIDED_BASELINE:
        kmod = h._load_module(h._KERNEL_DIR, h.KERNEL_FILE, "limit_candidate")
        target = getattr(kmod, h.KERNEL_ENTRY)
    original_limit = h.LIMIT
    try:
        for limit in LIMIT_PROBES:
            h.LIMIT = limit
            model = mmod.Model(limit)
            for shape in h.SHAPES:
                inp = h._make_inputs(shape)
                d = inp.shape[1] // 2
                count = min(d, len(GATE_VALUES))
                inp[:, :count] = torch.tensor(GATE_VALUES[:count], dtype=inp.dtype, device=inp.device)
                inp[:, d:d+count] = torch.tensor(UP_VALUES[:count], dtype=inp.dtype, device=inp.device)
                original = inp.clone()
                with torch.no_grad():
                    ref = h._checked_silu_result(model(inp), inp)
                    truth = h._checked_silu_result(h._aiter_op(inp), inp)
                    h.require_unchanged((inp,), (original,))
                    h.normalized_output(ref, truth, tolerance=h.REL_TOL)
                    if target is not None:
                        out = h._checked_silu_result(target(inp, limit), inp)
                        h.require_unchanged((inp,), (original,))
                        h.normalized_output(out, ref, tolerance=h.REL_TOL)
                print(f"  limit control PASS: {shape['name']}, limit={limit}")
    finally:
        h.LIMIT = original_limit
