"""IMMUTABLE — task-specific case builders for the fused-MoE op (grouped per-expert GEMM + routing).

Run VERBATIM by BOTH measurement legs (`leg_runner.py` under `baseline_overlay/` and under
`_cand_overlay/`), so the two legs can never diverge in what they compute — only in which module the
`meta.target_callable` resolves to.

Everything here reaches the op ONLY through `meta["target_callable"]`; nothing imports a backend by
name and nothing imports from `kernel_src/`.

Data provenance (see meta.notes):
  * `reference_io.pt` holds the REAL production operands captured from the live TP=8 sglang server:
    the fp8 block-scale expert weights (`w1`,`w2`) + their `[128,128]` block scales, a real bf16
    activation buffer, and the REAL router output (`topk_ids`/`topk_weights`).
  * Routing is NEVER synthesized uniformly (`randperm` is forbidden for MoE — it flattens the real
    expert skew that drives `moe_align_block_size` padding and per-expert effective-M). Timing and
    random-parity cases at online-aligned M buckets bootstrap their per-token routing rows WITH
    REPLACEMENT from the captured real routing pool, which preserves the empirical expert-popularity
    distribution.
"""
import importlib
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_C = {}


def _bootstrap():
    """Publish the deployment's sglang runtime config + a world-size-1 TP group in THIS process.

    Both legs need it and neither may import it off PYTHONPATH (HERE holds a file named unittest.py
    that shadows stdlib ), so load it by absolute file path. See sglang_bootstrap.py."""
    if _C.get("boot"):
        return
    spec = importlib.util.spec_from_file_location(
        "sglang_bootstrap", os.path.join(HERE, "sglang_bootstrap.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sglang_bootstrap"] = mod
    spec.loader.exec_module(mod)
    mod.ensure()
    _C["boot"] = True


def _torch():
    import torch
    return torch


def _device():
    torch = _torch()
    return "cuda" if torch.cuda.is_available() else "cpu"


def _blob():
    if "blob" not in _C:
        _C["blob"] = _torch().load(os.path.join(HERE, "reference_io.pt"),
                                   map_location="cpu", weights_only=False)
    return _C["blob"]


def _meta():
    if "meta" not in _C:
        with open(os.path.join(HERE, "meta.json")) as fh:
            _C["meta"] = json.load(fh)
    return _C["meta"]


def _shared():
    """Real captured expert weights + block scales, resident on device (loaded once per process)."""
    if "shared" not in _C:
        dev = _device()
        _C["shared"] = {k: v.to(dev).contiguous() for k, v in _blob()["shared"].items()}
    return _C["shared"]


def _act_std():
    if "act_std" not in _C:
        hs = _blob()["cases"][0]["hidden_states"]
        _C["act_std"] = float(hs.float().std().clamp_min(1e-3))
    return _C["act_std"]


def _resolve(dotted):
    mod_name, _, attr = dotted.partition(":")
    obj = importlib.import_module(mod_name)
    for part in attr.split("."):
        if part:
            obj = getattr(obj, part)
    return obj


# --------------------------------------------------------------------------- the op entry point
def call(args):
    """Invoke the CURRENT fused-MoE dispatcher on one arg set and return its output WHOLE.

    Which implementation this resolves to is decided ONLY by the overlay on PYTHONPATH."""
    if "fn" not in _C:
        _bootstrap()
        _C["fn"] = _resolve(_meta()["target_callable"])
    return _C["fn"](**args)


# --------------------------------------------------------------------------- arg assembly
def _build_args(hidden_states, topk_weights, topk_ids):
    a = dict(_shared())
    a.update(_blob()["static"])          # inplace=False, use_fp8_w8a8=True, block_shape=[128,128], ...
    a["hidden_states"] = hidden_states
    a["topk_weights"] = topk_weights
    a["topk_ids"] = topk_ids
    return a


def _routing_pool(regime):
    key = f"pool:{regime}"
    if key not in _C:
        torch = _torch()
        pool = _blob().get("routing_pool") or {}
        ent = pool.get(regime) or [x for v in pool.values() for x in v]
        _C[key] = (torch.cat([e["topk_ids"] for e in ent], 0).to(_device()),
                   torch.cat([e["topk_weights"] for e in ent], 0).to(_device()))
    return _C[key]


def _synth_inputs(m, regime, gen=None, seed=0):
    """Online-aligned dims + REAL routing rows (bootstrapped) + fresh in-regime activation values."""
    torch = _torch()
    dev = _device()
    if gen is None:
        gen = torch.Generator(device=dev).manual_seed(int(seed))
    ids, wts = _routing_pool(regime)
    idx = torch.randint(0, int(ids.shape[0]), (int(m),), generator=gen, device=dev)
    ti = ids.index_select(0, idx).contiguous()
    tw = wts.index_select(0, idx).contiguous()
    ref = _blob()["cases"][0]["hidden_states"]
    k = int(ref.shape[1])
    hs = (torch.randn(int(m), k, generator=gen, device=dev, dtype=torch.float32) * _act_std()
          ).to(ref.dtype).contiguous()
    return hs, tw, ti


def _case_m(c):
    """M for one attribute_weights workload case.  is a LIST of operand shapes
    ([[M,K],[N,K]] for the moe/gemm attributor), so read the explicit  first and only fall back
    to dims[0][0]; treating  as a dict silently yields zero cases."""
    m = c.get("m") or c.get("M")
    if not m:
        dims = c.get("dims")
        if isinstance(dims, dict):
            m = dims.get("m") or dims.get("M")
        elif isinstance(dims, (list, tuple)) and dims and isinstance(dims[0], (list, tuple)) and dims[0]:
            m = dims[0][0]
    return int(m or 0)


def _served(h, meta):
    try:
        return set(h.served_regimes(meta) or [])
    except Exception:
        return set(meta.get("served_regimes") or [])


# --------------------------------------------------------------------------- (1) frozen oracle cases
def eager_cases(h, meta):
    dev = _device()
    served = _served(h, meta)
    out = []
    for c in _blob()["cases"]:
        r = c.get("regime") or ""
        if served and r and r not in served:
            continue
        out.append({
            "sig": f"oracle_m{c['m']}_{r or 'na'}",
            "args": _build_args(c["hidden_states"].to(dev),
                                c["topk_weights"].to(dev),
                                c["topk_ids"].to(dev)),
            "ref": c["output"].to(dev),
        })
    return out


# --------------------------------------------------------------------------- (2) weighted timing set
def timing_cases(h, meta):
    served = _served(h, meta)
    specs = []
    for c in ((meta.get("workload") or {}).get("cases") or []):
        m = _case_m(c)
        if m:
            specs.append((m, c.get("regime") or ""))
    if not specs:
        specs = [(int(c["m"]), c.get("regime") or "") for c in _blob()["cases"]]
    out, seen = [], set()
    for m, r in specs:
        if served and r and r not in served:
            continue
        sig = f"moe_m{m}_{r or 'na'}"
        if sig in seen:
            continue
        seen.add(sig)
        hs, tw, ti = _synth_inputs(m, r, seed=1000 + m)
        out.append({"sig": sig, "regime": r, "m": m, "args": _build_args(hs, tw, ti)})
    return out


# --------------------------------------------------------------------------- (3) random value parity
def random_shapes(h, meta):
    """FIXED online dims, FRESH values per draw (activation values + a fresh real-routing bootstrap)."""
    served = _served(h, meta)
    out, seen = [], set()
    specs = [(int(c["m"]), c.get("regime") or "") for c in _blob()["cases"]]
    for c in ((meta.get("workload") or {}).get("cases") or []):
        m = _case_m(c)
        if m:
            specs.append((m, c.get("regime") or ""))
    for m, r in specs:
        if served and r and r not in served:
            continue
        sig = f"rand_m{m}_{r or 'na'}"
        if sig in seen:
            continue
        seen.add(sig)

        def mk(rng, m=m, r=r):
            hs, tw, ti = _synth_inputs(m, r, gen=rng)
            return _build_args(hs, tw, ti)

        out.append({"sig": sig, "make_inputs": mk})
    return out
