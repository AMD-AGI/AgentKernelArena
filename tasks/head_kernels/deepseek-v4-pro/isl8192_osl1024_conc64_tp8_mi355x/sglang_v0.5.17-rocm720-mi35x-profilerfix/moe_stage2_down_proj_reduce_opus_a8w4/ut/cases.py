"""Cases for the aiter OPUS MoE STAGE-2 SEGMENT (DeepSeek-V4-Pro a8w4: mxfp8 acts x mxfp4 experts).

Seam: `meta.target_callable` = aiter.ops.opus.moe_stage2_a8w4_fused_adapter:opus_a8w4_stage2_wrapper.

WHY THIS SEAM AND NOT A DEEPER ONE (this is the load-bearing decision of this task):
the profile attributes the head to TWO device kernels that together form ONE segment --
`opus_moe_stage2_a8w4_decode_kernel_gfx950` (the down-projection grouped GEMM over the moe_sorting
block map) and `opus_moe_stage2_reduce_token_slot_route_output_kernel_gfx950` (the per-token top-k
reduction of the routed per-slot partials). The two deeper python launchers each own exactly ONE of
them (`opus_moe_stage2_a8w4_decode_fwd` -> GEMM only; `opus_moe_stage2_reduce_token_slot_route_output_fwd`
-> reduce only; see selection_validation_deeper_probe.json, where each reports
`device_kernel_not_under_target` for the other kernel). `opus_a8w4_stage2_wrapper` is the DEEPEST single
python callable under which BOTH profiled kernels launch, so it is the only seam whose isolated timing
denominator equals the profiled stage-2 SEGMENT. Anything shallower (`fused_moe_2stages`) also owns
stage 1, which is a different head.

TWO LIVE PATHS ARE CAPTURED, BOTH MUST KEEP WORKING:
  * decode (M=1/64/256): `route_out=False`, kernel_id 2005 (`opus_moe2_afp8_wfp4_atomic_...sbm32...`).
    The GEMM ATOMICALLY ACCUMULATES the topk partials straight into `out`, so `out` MUST arrive ZEROED;
    the reduce kernel does not run.
  * prefill (M=2048/32768): `route_out=True`, kernel_id 2003 (`opus_moe2_afp8_wfp4_fp8_...rbn3072`).
    The GEMM writes per-slot partials, then the reduce kernel sums them into `out`.

Run VERBATIM by BOTH legs (baseline_overlay / _cand_overlay) through leg_runner.py, so it must reach
the op ONLY through `meta.target_callable` and must never import kernel_src/ or a backend by name.

Everything here is REPLAYED FROM THE REAL CAPTURE (reference_io.pt): the fp8 stage-1 intermediate
activations, the mxfp4 expert weights, the e8m0 block scales, and -- critically -- the REAL routing
(the moe_sorting products sorted_token_ids / sorted_expert_ids / num_valid_ids / sorted_weights). MoE
routing is NOT value-independent, so nothing routing-shaped is ever synthesized here (no randperm, no
uniform expert assignment): the only values ever redrawn (random-parity leg) are the stage-2 INPUT
ACTIVATION values, at the fixed captured dims.

CAPTURE FACTS THIS FILE HAS TO UNDO:
  * the seam is called ALL-POSITIONALLY for its first 8 operands
    (inter_states, w1, w2, sorted_token_ids, sorted_expert_ids, num_valid_ids, out, topk), so `out` is
    POSITIONAL index 6 (`meta.inplace_out_arg`), not the `out=` keyword.
  * `out` is written IN PLACE and the capture snapshots arguments AFTER the call, so the recorded
    args[6] is the FILLED result. `_invoke` therefore always substitutes a fresh ZEROED buffer -- both
    required for the atomic decode path and correct for the route+reduce prefill path.
  * fp4 (`float4_e2m1fn_x2`) / e8m0 tensors have no CPU copy kernel, so capture stored them bitcast to
    uint8 (`bitcast: true`); they are viewed back to the recorded dtype here.
  * non-tensor args (the `kernelName` string) were stored as `repr` strings and are rehydrated by
    literal-eval / `meta.enum_modules` lookup.
  * the capture blob is storage-DEDUPED (the 1.6GB expert-weight set is shared by all 5 records), so the
    host->device upload is memoized per storage; the GPU copy is shared too.

RETURN-VALUE CONTRACT: the seam returns the single `out` tensor it was handed (both paths tail-return
it). Because `_invoke` hands it a FRESH zeroed buffer on every call, the returned tensor is fresh
storage per call, which keeps `harness_lib.assert_independent_outputs` meaningful; the guard in `call()`
rejects a candidate that instead returns a persistent/static buffer.
"""
import ast
import importlib
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "meta.json")) as _fh:
    META = json.load(_fh)

_ORACLE = None
_DEV_CACHE = {}
_LIVE_OUTS = []          # keeps the last few RAW outputs alive so the alias guard below is valid



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


def _resolve(dotted):
    mod_name, _, attr = dotted.partition(":")
    obj = importlib.import_module(mod_name)
    for part in attr.split("."):
        if part:
            obj = getattr(obj, part)
    return obj


_ENUM_RE = re.compile(r"^<?([A-Za-z_]\w*)\.([A-Za-z_]\w*)\b")


def _rehydrate_scalar(rep):
    """Rebuild a non-tensor argument captured only as `repr` (str, torch dtypes, aiter enums)."""
    torch = _torch()
    rep = rep.strip()
    if rep.startswith("torch."):
        obj = torch
        for part in rep.split(".")[1:]:
            obj = getattr(obj, part)
        return obj
    try:
        return ast.literal_eval(rep)          # plain str/int/float/tuple reprs (e.g. the kernelName)
    except (ValueError, SyntaxError):
        pass
    m = _ENUM_RE.match(rep)
    if m:
        cls_name, member = m.group(1), m.group(2)
        for mod_name in META.get("enum_modules", []):
            try:
                mod = importlib.import_module(mod_name)
            except Exception:
                continue
            cls = getattr(mod, cls_name, None)
            if cls is not None and hasattr(cls, member):
                return getattr(cls, member)
    raise RuntimeError(f"cases.py cannot rehydrate captured argument {rep!r}; add its module to "
                       f"meta.enum_modules")


def _load(node, device):
    """Rebuild one captured node (tensor snapshot / container / scalar / repr placeholder)."""
    if isinstance(node, dict) and node.get("__tensor__"):
        # Include the recorded view and target device in the cache identity.
        cpu = node["data"]
        key = (str(device), cpu.data_ptr(), tuple(cpu.shape), tuple(cpu.stride()),
               str(cpu.dtype), json.dumps({k: v for k, v in node.items() if k != "data"},
                                         sort_keys=True, default=str))
        t = _DEV_CACHE.get(key)
        if t is None:
            import harness_lib
            t = harness_lib.capture_contract().restore_tensor(_torch(), node, device)
            _DEV_CACHE[key] = t
        return t
    if isinstance(node, dict) and "__repr__" in node:
        return _rehydrate_scalar(node["__repr__"])
    if isinstance(node, (list, tuple)):
        return type(node)(_load(v, device) for v in node)
    if isinstance(node, dict):
        return {k: _load(v, device) for k, v in node.items()}
    return node


def _apply_tensor_attrs(args, kwargs):
    """Re-attach python attributes the launcher reads off tensors (e.g. w2 `is_shuffled`)."""
    import harness_lib
    for key, attrs in (META.get("tensor_attrs") or {}).items():
        try:
            target = args[int(key)] if key.lstrip("-").isdigit() else kwargs[key]
        except (IndexError, KeyError, ValueError) as exc:
            raise RuntimeError(f"captured tensor attributes name absent operand {key!r}") from exc
        if target is None:
            raise RuntimeError(f"captured tensor attributes name null operand {key!r}")
        harness_lib.capture_contract().tensor_attrs(target, attrs)


def _oracle():
    global _ORACLE
    if _ORACLE is None:
        torch = _torch()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        blob = _load_frozen_capture(torch, os.path.join(HERE, META.get("reference_io", "reference_io.pt")),
                          map_location="cpu", weights_only=False)
        recs = []
        for r in blob["records"]:
            args = list(_load(r["args"], device))
            kwargs = dict(_load(r["kwargs"], device))
            _apply_tensor_attrs(args, kwargs)
            ref_full = _load(r["output"], device)
            ref = ref_full[0] if isinstance(ref_full, (tuple, list)) else ref_full
            recs.append({
                "sig": r["sig"],
                "regime": r.get("regime", ""),
                "m": int(_primary({"args": args, "kwargs": kwargs}).shape[0]),
                "args": {"args": args, "kwargs": kwargs},
                "ref": ref,
                "ref_full": ref_full,
            })
        if len(recs) != int(META["num_cases"]):
            raise RuntimeError(f"capture case count {len(recs)} != declared {META['num_cases']}")
        if len({r["sig"] for r in recs}) != len(recs):
            raise RuntimeError("capture contains duplicate case signatures")
        _ORACLE = recs
    return _ORACLE


def oracle_records(h=None, meta=None):
    """Full frozen records — used by unittest.py for the graph-replay bundle and the recorded-order
    sequence check. Never used to pick which code path runs."""
    return _oracle()


# ---------------------------------------------------------------- the seam both legs go through
def _primary(args):
    """The stage-2 input activation operand (`inter_states`, [token, topk, inter_dim]); its leading dim
    is the case's M (token count)."""
    key = META.get("primary_input_key", "inter_states")
    if args["args"]:
        return args["args"][0]
    return args["kwargs"][key]


def _invoke(args):
    torch = _torch()
    fn = _resolve(META["target_callable"])
    pos = list(args["args"])
    kw = dict(args["kwargs"])
    out_key = META.get("out_key", "out")
    # NEVER replay a FILLED in-place output buffer. The decode kernel ATOMICALLY ACCUMULATES into
    # `out`, so replaying the captured (already-filled) buffer would double the result; and the
    # capture snapshots args AFTER the call, so the recorded buffer IS the previous result. Hand over
    # a fresh zeroed buffer instead -- which is exactly what the live dispatcher does (moe_out is a
    # freshly zeroed allocation per forward).
    if torch.is_tensor(kw.get(out_key)):
        kw[out_key] = torch.zeros_like(kw[out_key])
    idx = META.get("inplace_out_arg")
    if idx is not None and 0 <= int(idx) < len(pos) and torch.is_tensor(pos[int(idx)]):
        pos[int(idx)] = torch.zeros_like(pos[int(idx)])
    return fn(*pos, **kw)


def call_full(args):
    """args -> the WHOLE seam return (a single [token, model_dim] bf16 tensor for this seam)."""
    return _invoke(args)


def call(args):
    """args -> FRESH output tensor (the reduced bf16 MoE output, [token, model_dim]).

    `args` is {"args": [...positional...], "kwargs": {...}}. `_invoke` allocates the `out` buffer fresh
    per call and the seam tail-returns it, so the returned tensor is the call's own storage — exactly
    what the harness's output-independence check must see. The guard below keeps that anti-cheat honest:
    a candidate that hands back a persistent/static buffer is rejected here."""
    torch = _torch()
    out = _invoke(args)
    primary = out[0] if isinstance(out, (tuple, list)) else out
    if not torch.is_tensor(primary):
        raise TypeError("candidate did not return the captured tensor output")
    for prev in _LIVE_OUTS:
        if torch.is_tensor(prev) and prev.untyped_storage().data_ptr() == primary.untyped_storage().data_ptr():
            raise RuntimeError(
                "shared_output_buffer: this call returned the SAME storage as a still-live earlier "
                "output — the launcher contract is fn(args) -> FRESH out; a persistent/static return "
                "buffer is a tight-loop cheat that is wrong for any real (batched) caller.")
    _LIVE_OUTS.append(primary)
    del _LIVE_OUTS[:-4]
    return primary


def _bucket_sigs():
    """The timing buckets: meta.workload cases when the weighting step produced them, else the oracle."""
    wl = (META.get("workload") or {}).get("cases") or []
    return [c.get("sig") for c in wl if c.get("sig")]


def timing_cases(h, meta):
    served = set(meta.get("served_regimes") or []) or None
    recs = _oracle()
    by_sig = {r["sig"]: r for r in recs}
    declared = _bucket_sigs()
    missing = set(declared) - set(by_sig)
    if missing:
        raise RuntimeError(f"declared benchmark cases missing from capture: {sorted(missing)}")
    order = declared or [r["sig"] for r in recs]
    out = []
    for sig in order:
        r = by_sig[sig]
        if served and r["regime"] not in served:
            continue
        out.append({"sig": sig, "regime": r["regime"], "m": r["m"], "args": r["args"]})
    return out


def eager_cases(h, meta):
    served = set(meta.get("served_regimes") or []) or None
    return [{"args": r["args"], "ref": r["ref"], "sig": r["sig"]}
            for r in _oracle() if not served or r["regime"] in served]


def random_shapes(h, meta):
    """FRESH random stage-2 ACTIVATION values at the FIXED captured dims (dims are never randomized).

    Only `inter_states` is redrawn. The expert weights/scales stay as captured because random e8m0
    exponents / fp4 bit patterns are not an in-regime distribution, and every routing tensor
    (sorted_token_ids / sorted_expert_ids / num_valid_ids / sorted_weights) stays as captured because
    MoE routing is value-dependent and must never be fabricated."""
    torch = _torch()
    served = set(meta.get("served_regimes") or []) or None
    shapes = []
    for r in _oracle():
        if served and r["regime"] not in served:
            continue

        def make_inputs(rng, r=r):
            base = r["args"]
            x0 = _primary(base)
            x = torch.randn(tuple(x0.shape), generator=rng, device=x0.device, dtype=torch.float32)
            values = (x * float(meta.get("act_scale", 1.0))).to(x0.dtype)
            x = torch.empty_strided(tuple(x0.shape), tuple(x0.stride()),
                                    dtype=x0.dtype, device=x0.device)
            x.copy_(values)
            h.capture_contract().tensor_attrs(x, getattr(x0, "__dict__", {}))
            pos, kw = list(base["args"]), dict(base["kwargs"])
            if pos:
                pos[0] = x
            else:
                kw[META.get("primary_input_key", "inter_states")] = x
            return {"args": pos, "kwargs": kw}

        shapes.append({"sig": r["sig"], "make_inputs": make_inputs})
    return shapes
