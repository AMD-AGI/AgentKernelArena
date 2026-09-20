"""IMMUTABLE - task-specific case builders for the bpreshuffle fp8 activation-scale materialization.

Op under test
-------------
`sglang.srt.layers.quantization.fp8_utils:materialize_bpreshuffle_fp8_scale(scale) -> scale'`

A per-1x128 fp8 activation scale leaves the producing quant kernel as a ROW-MAJOR `[M, G]` fp32
tensor; the gfx950 CK `gemm_a8w8_blockscale_bpreshuffle` GEMM reads those bytes COLUMN-MAJOR, so the
seam materializes a transposed-contiguous copy. That copy is the profiled
`elementwise_kernel_manual_unroll<128,4, ... direct_copy_kernel_cuda ...>` device kernel
(~26k launches over the profiling window).

Run VERBATIM by BOTH measurement legs (`leg_runner.py` under `baseline_overlay/` and under
`_cand_overlay/`), so the legs can never diverge in WHAT they compute - only in which code
`meta.target_callable` resolves to. Nothing here imports a backend by name and nothing imports from
`kernel_src/`.

CORRECTNESS IS TWO-PART FOR THIS OP (see `call` below): values AND physical strides. A candidate that
returns the right numbers in row-major order would pass a naive allclose and then silently feed the CK
GEMM mis-strided bytes end-to-end, so `call` enforces the layout contract at the seam.
"""
import importlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
_C = {}



def _load_frozen_capture(torch, path, **kwargs):
    import importlib.util
    from pathlib import Path
    helper = Path(__file__).resolve().with_name("task_contract.py")
    spec = importlib.util.spec_from_file_location("_frozen_capture_loader", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verified_torch_load(torch, path, **kwargs)


def _torch():
    import torch
    return torch


def _device():
    return "cuda" if _torch().cuda.is_available() else "cpu"


def _meta():
    if "meta" not in _C:
        with open(os.path.join(HERE, "meta.json")) as fh:
            _C["meta"] = json.load(fh)
    return _C["meta"]


def _blob():
    if "blob" not in _C:
        raise RuntimeError("generated GLM inputs must be initialized by the protected generated worker")
    return _C["blob"]


def _unsnap(x, device=None):
    """Rebuild a live tensor from a capture_shapes snapshot dict (or pass a plain value through)."""
    torch = _torch()
    dev = device or _device()
    if isinstance(x, dict) and x.get("__tensor__"):
        import harness_lib
        return harness_lib.capture_contract().restore_tensor(torch, x, dev)
    if isinstance(x, (list, tuple)):
        return type(x)(_unsnap(v, dev) for v in x)
    if isinstance(x, dict):
        return {k: _unsnap(v, dev) for k, v in x.items()}
    return x


def _resolve(dotted):
    mod_name, _, attr = dotted.partition(":")
    obj = importlib.import_module(mod_name)
    for part in attr.split("."):
        if part:
            obj = getattr(obj, part)
    return obj


# --------------------------------------------------------------------------- the op entry point
def call(args):
    """Invoke the CURRENT seam on one arg set and return its output WHOLE.

    Which implementation this resolves to is decided ONLY by the overlay on PYTHONPATH.
    Also enforces the op's PHYSICAL-LAYOUT contract, which is part of its correctness and is not
    expressible as a value comparison: for a 2-D `[M, G]` input the result must be logically `[M, G]`
    with the SAME values but transposed-contiguous storage (`out.t().is_contiguous()`, i.e. stride
    `(1, M)` whenever that is distinguishable). Raising here surfaces
    as a clean FAIL in the unittest driver.
    """
    if "fn" not in _C:
        _C["fn"] = _resolve(_meta()["target_callable"])
    scale = args["scale"]
    out = _C["fn"](scale)
    if hasattr(scale, "dim") and scale.dim() == 2:
        m, g = int(scale.shape[0]), int(scale.shape[1])
        if tuple(out.shape) != (m, g):
            raise ValueError(f"layout contract: expected logical shape {(m, g)}, got {tuple(out.shape)}")
        # Column-major-equivalent storage. Expressed as "the transpose is contiguous" rather than a
        # literal stride==(1, m) test, because when m or g is 1 the two layouts share the same bytes
        # and torch legitimately reports the degenerate stride (the live baseline itself does this for
        # the m == 1 warmup shapes); a hard stride equality would fail the unmodified baseline.
        if not out.t().is_contiguous():
            raise ValueError(
                f"layout contract: the bpreshuffle GEMM reads the scale COLUMN-MAJOR, so the seam must "
                f"return transposed-contiguous storage (stride (1, {m}) for a [{m}, {g}] scale); got "
                f"stride {tuple(out.stride())}. A value-correct but row-major return silently "
                f"mis-strides the CK GEMM end-to-end.")
    return out


# --------------------------------------------------------------------------- shape helpers
def _served(h, meta):
    try:
        return set(h.served_regimes(meta) or [])
    except Exception:
        return {str(r).lower() for r in (meta.get("served_regimes") or [])}


def _dims_of(c, fallback_g):
    """(M, G) for one case entry, tolerating both the meta `cases[]` and the attribute_weights
    `workload.cases[]` spellings (`dims` is a LIST of operand shapes there)."""
    d = c.get("dims")
    if isinstance(d, (list, tuple)) and d:
        first = d[0]
        if isinstance(first, (list, tuple)) and len(first) >= 2:
            return int(first[0]), int(first[1])
        if isinstance(first, int) and len(d) >= 2:
            return int(d[0]), int(d[1])
    if isinstance(d, dict):
        m = d.get("m") or d.get("M")
        g = d.get("g") or d.get("G")
        if m and g:
            return int(m), int(g)
    m = c.get("m") or c.get("M")
    if m:
        return int(m), int(fallback_g)
    return 0, 0


def _fallback_g(meta):
    for c in (meta.get("cases") or []):
        m, g = _dims_of(c, 0)
        if g:
            return g
    return 0


def _shape_specs(meta, source):
    """[(M, G, regime)] for the requested case source, de-duplicated, online-aligned dims only."""
    fg = _fallback_g(meta)
    raw = ((meta.get("workload") or {}).get("cases") or []) if source == "workload" else []
    if not raw:
        raw = meta.get("cases") or []
    out, seen = [], set()
    for c in raw:
        m, g = _dims_of(c, fg)
        if not m or not g:
            continue
        key = (m, g)
        if key in seen:
            continue
        seen.add(key)
        out.append((m, g, (c.get("regime") or "").lower()))
    return out


def _synth(m, g, gen=None, seed=0):
    """A FRESH in-regime operand: row-major [M, G] fp32 per-1x128 activation scale, exactly as the
    producing quant kernel emits it. Values are irrelevant to a pure layout copy (perf is
    value-independent) but are drawn in the real magnitude range so parity draws are meaningful."""
    torch = _torch()
    dev = _device()
    if gen is None:
        gen = torch.Generator(device=dev).manual_seed(int(seed))
    x = torch.rand((int(m), int(g)), generator=gen, device=dev, dtype=torch.float32)
    return {"scale": (x * 0.05 + 1e-3).contiguous()}


# --------------------------------------------------------------------------- (1) frozen oracle cases
def eager_cases(h, meta):
    dev = _device()
    served = _served(h, meta)
    out = []
    for i, r in enumerate(_blob()["records"]):
        reg = (r.get("regime") or "").lower()
        if served and reg and reg not in served:
            continue
        a = _unsnap(r["args"], dev)
        kw = _unsnap(r.get("kwargs") or {}, dev)
        scale = kw.get("scale") if "scale" in kw else a[0]
        out.append({"sig": f"oracle{i}_m{int(scale.shape[0])}_{reg or 'na'}",
                    "args": {"scale": scale},
                    "ref": _unsnap(r["output"], dev)})
    return out


# --------------------------------------------------------------------------- (2) weighted timing set
def timing_cases(h, meta):
    served = _served(h, meta)
    out = []
    for m, g, reg in _shape_specs(meta, "workload"):
        if served and reg and reg not in served:
            continue
        out.append({"sig": f"scale_m{m}x{g}_{reg or 'na'}", "regime": reg, "m": m,
                    "args": _synth(m, g, seed=1000 + m)})
    return out


# --------------------------------------------------------------------------- (3) random value parity
def random_shapes(h, meta):
    """FIXED online dims, FRESH values per draw."""
    served = _served(h, meta)
    specs = {}
    for src in ("meta", "workload"):
        for m, g, reg in _shape_specs(meta, src):
            if served and reg and reg not in served:
                continue
            specs.setdefault((m, g), reg)
    out = []
    for (m, g), reg in specs.items():
        def mk(rng, m=m, g=g):
            return _synth(m, g, gen=rng)
        out.append({"sig": f"rand_m{m}x{g}_{reg or 'na'}", "make_inputs": mk})
    return out
