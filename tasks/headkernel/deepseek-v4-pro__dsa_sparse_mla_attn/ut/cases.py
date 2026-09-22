"""IMMUTABLE task cases for the DeepSeek-V4 (dsv4) fused sparse-MLA attention seam.

Run VERBATIM by BOTH legs (baseline_overlay / _cand_overlay) through leg_runner.py, so the two legs
can never diverge. Reaches the op ONLY through meta.target_callable — never through kernel_src/.

Seam: `sglang.kernels.ops.attention.dsa.tilelang_kernel:dpsk_v4_fp8_attention_fwd`, the TileLang
partial+combine pair (both device kernels are named `main_kernel`) that the HIP dsv4 backend reaches
through `hip_flash_mla.flash_mla_with_kvcache_entrypoint(backend="tilelang")`.

Oracle layout (reference_io.pt, written by the dsv4-aware capture shim):
  records[i] = {sig, regime, args, kwargs:{q, k_cache, indices, topk_length, attn_sink,
                extra_k_cache, extra_indices_in_kvcache, extra_topk_length, head_dim_v,
                softmax_scale, ...}, output:(o, lse)}
`k_cache` / `extra_k_cache` are stored PACKED ({"__dsa_kv__":True, shape, dtype, stride0, rows_idx,
rows}) because the live arguments are the layer's WHOLE paged KV pools (GBs). We rebuild a FULL-SIZE
buffer of the recorded shape AND the recorded dim-0 stride (the fp8 blocks carry scales/padding past
the visible extent, and `_build_fp8_combined_view` derives `block_pad_u32` from that stride, so the
stride is part of the op's contract) and scatter the recorded blocks back at their original ids, so
the gather sees the real addresses and the real values; unreferenced blocks are zero (never read).
"""
import importlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "meta.json")) as _fh:
    META = json.load(_fh)

_CACHE = {"records": None}
_KV_FULL_MAX_GB = float(os.environ.get("GEAK_KV_FULL_MAX_GB", "32"))

# tensors whose dim 0 is the token/batch axis: an `m` sub-batch slices all of them together
_ROW_ARGS = ("q", "indices", "topk_length", "extra_indices_in_kvcache", "extra_topk_length")
# (paged pool, the index tensor that names its blocks)
_KV_PAIRS = (("k_cache", "indices"), ("extra_k_cache", "extra_indices_in_kvcache"))


# --------------------------------------------------------------------------- seam
def _resolve(dotted):
    mod_name, _, attr = dotted.partition(":")
    obj = importlib.import_module(mod_name)
    for part in attr.split("."):
        if part:
            obj = getattr(obj, part)
    return obj


_UNDEF_KEY = "__undef__"


def call(args):
    """args -> a FRESH attention-output tensor. THE seam both legs go through.

    Returns the attention output `o` ONLY. The seam physically returns `(o, lse)`, but the harness's
    shared-buffer/independence probe requires a single tensor return, and `lse` is produced by the very
    same two device launches (`main_kernel` partial + combine), so nothing about the op escapes the gate
    by dropping it from the comparison surface.

    `args[_UNDEF_KEY]` (when present) is a boolean mask of the positions the LIVE op itself leaves
    UNDEFINED for this recorded call, and they are zeroed on the way out. Two real, verified sources of
    undefined rows in this seam's captured I/O:
      * decode x1024, rows 56..63 — the server pads the decode batch to the next bucket and the padding
        slots' `q` is UNINITIALISED (the frozen oracle's own `q` carries NaN on exactly those rows).
      * prefill x1024, rows 1792..6047 — every one of the row's 1024 `extra_indices_in_kvcache` entries
        is -1 (empty extra context), so the row's softmax denominator is empty and the live op returns
        NaN there.
    In both cases the frozen oracle's OUTPUT carries NaN at exactly the positions our rebuild does, so
    the values agree bit-for-bit and only the comparator (`NaN != NaN`) disagrees. The mask is derived
    from the ORACLE, never from the candidate's own output, and it is applied identically in both legs,
    so a candidate that invents a NaN anywhere the oracle is DEFINED still fails.
    """
    fn = _resolve(META["target_callable"])
    undef = args.get(_UNDEF_KEY) if isinstance(args, dict) else None
    kw = {k: v for k, v in args.items() if k != _UNDEF_KEY} if undef is not None else args
    out = fn(**kw)
    o = out[0] if isinstance(out, (list, tuple)) else out
    if undef is not None:
        return o.masked_fill(undef, 0)      # masked_fill (not _) => still a FRESH tensor per call
    return o


# --------------------------------------------------------------------------- oracle plumbing
def _torch():
    import torch
    return torch


def _dt(name, torch):
    return getattr(torch, str(name).split(".")[-1])


def _prod(xs):
    n = 1
    for v in xs:
        n *= int(v)
    return n


def _records():
    """The frozen oracle, memory-MAPPED.

    The blob is multi-GB (two dsv4 layer families x prefill/decode, each carrying its referenced KV
    pages) and `measure_legs` re-loads it in a FRESH subprocess for every bucket x leg x rep — a
    materializing `torch.load` would spend minutes per unittest run reading records the bucket never
    touches. mmap pages in only the tensors a case actually rebuilds; `timing_cases` is lazy for the
    same reason.
    """
    if _CACHE["records"] is None:
        torch = _torch()
        path = os.path.join(HERE, META.get("reference_io", "reference_io.pt"))
        try:
            blob = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        except (TypeError, RuntimeError):        # older torch / non-zipfile blob
            blob = torch.load(path, map_location="cpu", weights_only=False)
        _CACHE["records"] = blob["records"]
    return _CACHE["records"]


def _rec_by_sig(sig):
    for r in _records():
        if r["sig"] == sig:
            return r
    raise KeyError(f"no oracle record for sig {sig!r}")


def _is_tensor_snap(v):
    return isinstance(v, dict) and v.get("__tensor__")


def _is_kv_snap(v):
    return isinstance(v, dict) and v.get("__dsa_kv__")


def _t(snap, torch, device):
    """Rebuild a plain captured tensor."""
    return snap["data"].to(device=device)


def _kv_bytes(packed, torch):
    n = int(packed["shape"][0]) * int(packed.get("stride0") or _prod(packed["shape"][1:]))
    return n * torch.empty(0, dtype=_dt(packed["dtype"], torch)).element_size()


def _build_kv(packed, indices, torch, device, cache_key=""):
    """(kv, indices) rebuilt in-regime.

    Full-size when the recorded pool fits the budget (real block ids, real dim-0 stride, real gather
    locality); otherwise compacted to the referenced blocks with the indices remapped 1:1 (same
    values, denser addresses) so the task still runs on a small card.

    Values are ALWAYS the recorded ones, including for the random-parity draws: this pool is fp8 with
    per-block scales interleaved in the padding past the visible extent, so random bytes there
    denormalise the dequant and can produce NaN/Inf on BOTH legs (a NaN never compares equal, so the
    value-parity gate would fail on garbage rather than on a real divergence). The parity draws
    randomise `q` instead, which exercises the same arithmetic with well-formed operands.
    """
    dt = _dt(packed["dtype"], torch)
    esz = torch.empty(0, dtype=dt).element_size()
    shape = tuple(int(d) for d in packed["shape"])
    stride0 = int(packed.get("stride0") or _prod(shape[1:]))
    rows_idx = packed["rows_idx"].to(device=device, dtype=torch.int64)
    rows = packed["rows"].to(device=device)
    row_elems = _prod(shape[1:])

    if _kv_bytes(packed, torch) <= _KV_FULL_MAX_GB * (1 << 30):
        # One buffer per (cache_key, shape, dtype, device) is REUSED across builds: the live server
        # also has exactly one KV pool per layer, the op never writes it, and rebuilding a multi-GB
        # buffer per case would not fit. Blocks are (re)written on every build.
        key = (str(cache_key), shape, stride0, str(dt), str(device))
        kv = _CACHE.setdefault("kv", {}).get(key)
        if kv is None:
            base = torch.zeros(shape[0] * stride0, dtype=dt, device=device)
            strides = []
            acc = 1
            for d in reversed(shape[1:]):
                strides.append(acc)
                acc *= int(d)
            strides = [stride0] + list(reversed(strides))
            kv = torch.as_strided(base, shape, tuple(strides))
            _CACHE["kv"][key] = (kv, base)
        else:
            kv, base = kv
        # scatter through a byte view of the flat storage: fp8 dtypes have no index_put_ kernel
        flat = base.view(torch.uint8).view(shape[0], stride0 * esz)
        flat[rows_idx, : row_elems * esz] = rows.reshape(rows.shape[0], -1).view(torch.uint8)
        return kv, indices

    # compact fallback: kv := referenced blocks (dense stride), indices := their positions
    flat_idx = indices.reshape(-1).to(torch.int64).clamp_(min=0)
    new = torch.searchsorted(rows_idx, flat_idx).to(indices.dtype).reshape(indices.shape)
    return rows.contiguous(), new


def _build_args(sig, torch, device, m=None, rng=None, index_mode="recorded", undef=True):
    """Rebuild one call's kwargs from the frozen oracle.

    m           : keep only the first `m` token rows (a real sub-batch of the recorded batch; the op
                  is row-independent, so the golden output slices the same way).
    rng         : fresh random in-regime VALUES for `q` at the SAME dims (shapes never randomized).
    index_mode  : 'recorded'  -> the real captured sparse block ids
                  'shortctx'  -> every row reads the SAME block ids (extreme locality / duplicate-
                                 index boundary case for the graph-replay leg)
    """
    rec = _rec_by_sig(sig)
    kw = rec["kwargs"]
    out = {}
    # 1. plain tensors + scalars, with the row axis sliced to `m`
    for k, v in kw.items():
        if _is_kv_snap(v):
            continue
        if _is_tensor_snap(v):
            t = _t(v, torch, device)
            if m is not None and k in _ROW_ARGS and t.shape[0] > int(m):
                t = t[: int(m)].contiguous()
            out[k] = t
        elif isinstance(v, dict) and "__repr__" in v:
            # an opaque object (tile_scheduler_metadata) — the tilelang path ignores it
            out[k] = None
        else:
            out[k] = v
    if rng is not None and torch.is_tensor(out.get("q")):
        q = out["q"]
        out["q"] = torch.randn(q.shape, generator=rng, device=device, dtype=torch.float32).to(q.dtype)
    # 2. index rewrite for the replay boundary case
    if index_mode == "shortctx":
        for _, idx_name in _KV_PAIRS:
            idx = out.get(idx_name)
            if torch.is_tensor(idx) and idx.numel():
                out[idx_name] = idx[:1].expand(idx.shape).contiguous()
    # 3. paged pools, rebuilt against their (possibly sliced/rewritten) index tensors
    for kv_name, idx_name in _KV_PAIRS:
        packed = kw.get(kv_name)
        if packed is None:
            out[kv_name] = None
            continue
        if not _is_kv_snap(packed):
            out[kv_name] = _t(packed, torch, device) if _is_tensor_snap(packed) else packed
            continue
        idx = out.get(idx_name)
        kv, idx2 = _build_kv(packed, idx, torch, device, cache_key=sig + "|" + kv_name)
        out[kv_name] = kv
        if torch.is_tensor(idx2):
            out[idx_name] = idx2
    # 4. undefined-position mask (correctness paths only; timing_cases passes undef=False so the
    #    measured leg issues the op launch and NOTHING else).
    if undef:
        mask = _undef_mask(sig, torch, device, m=m)
        if mask is not None:
            out[_UNDEF_KEY] = mask
    return out


def _golden_o(sig, torch, device, m=None):
    """The frozen attention output `o`, row-sliced to `m` (the seam records `(o, lse)`; see call())."""
    rec = _rec_by_sig(sig)
    out = rec["output"]
    snap = out[0] if isinstance(out, (list, tuple)) else out
    t = snap["data"].to(device=device) if _is_tensor_snap(snap) else snap
    return t[: int(m)] if (m is not None and t.shape[0] > int(m)) else t


def _undef_mask(sig, torch, device, m=None):
    """Positions the LIVE op leaves undefined for this recorded call (NaN in the frozen output), or
    None when the record is fully defined. Cached per (sig, m, device) — the prefill mask is 8192 x 64
    x 512 bools and every case rebuild would otherwise re-scan the mmapped oracle."""
    key = (sig, None if m is None else int(m), str(device))
    cache = _CACHE.setdefault("undef", {})
    if key not in cache:
        g = _golden_o(sig, torch, device, m=m)
        mask = torch.isnan(g.float())
        cache[key] = mask if bool(mask.any()) else None
    return cache[key]


def _golden(sig, torch, device, m=None):
    """The comparison golden: the frozen `o` with the undefined positions zeroed, matching call()."""
    g = _golden_o(sig, torch, device, m=m)
    mask = _undef_mask(sig, torch, device, m=m)
    return g if mask is None else g.masked_fill(mask, 0)


def _specs(meta):
    """[{name, source_sig, m, regime}] — the online case set (from meta.case_specs)."""
    return list(meta.get("case_specs") or [])


def _spec_by_name(meta, name):
    for s in _specs(meta):
        if s["name"] == name:
            return s
    return None


class _LazyCase(dict):
    """A timing case whose `args` are built only when the key is actually read.

    `leg_runner --mode list` reads just `sig`, and `--mode time` runs ONE bucket per process, so
    eagerly rebuilding every case's multi-GB KV pool in every leg subprocess would cost far more
    than the measurement itself."""

    def __init__(self, base, build):
        super().__init__(base)
        self._build = build

    def __getitem__(self, key):
        if key == "args" and "args" not in self:
            super().__setitem__("args", self._build())
        return super().__getitem__(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default


# --------------------------------------------------------------------------- required entry points
def timing_cases(h, meta):
    torch = _torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wl = (meta.get("workload") or {}).get("cases")
    names = [c.get("name") for c in wl] if wl else [s["name"] for s in _specs(meta)]
    out = []
    for name in names:
        s = _spec_by_name(meta, name)
        if s is None:
            continue
        out.append(_LazyCase(
            {"sig": s["name"], "regime": s.get("regime", ""), "m": int(s.get("m") or 0)},
            (lambda s=s: _build_args(s["source_sig"], torch, device, m=s.get("m"), undef=False))))
    return out


def random_shapes(h, meta):
    """FRESH in-regime VALUE draws at the FIXED online dims (dims never randomized).

    The trailing deterministic entries are the graph-replay boundary inputs: they ignore `rng`, so the
    BASELINE leg's recorded output for them is the golden the replay leg compares against.
    """
    torch = _torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    shapes = []
    for s in _specs(meta):
        # `rand_m` caps the ROW count of the value-parity draws (prefill defaults to 1024 rows of the
        # recorded 8192). Parity is a VALUE-dependence check, not a shape check — the rows are
        # independent, so a row slice runs the identical code path, while the full prefill shape would
        # make the baseline leg write a multi-GB `_baseline_random.pt` on every unittest run.
        rm = int(s.get("rand_m") or min(int(s.get("m") or 0) or 1, 1024))
        shapes.append({
            "sig": s["name"],
            "make_inputs": (lambda rng, s=s, rm=rm: _build_args(s["source_sig"], torch, device,
                                                                m=rm, rng=rng)),
        })
    for s in _specs(meta):
        # only the spec the unittest captures its graph on (meta.case_specs[].replay) needs a
        # deterministic shortctx twin — that twin's BASELINE-leg output is the replay golden.
        if not s.get("replay"):
            continue
        shapes.append({
            "sig": "replay_shortctx:" + s["name"],
            "make_inputs": (lambda rng, s=s: _build_args(s["source_sig"], torch, device,
                                                         m=s.get("m"), index_mode="shortctx")),
        })
    return shapes


def eager_cases(h, meta):
    torch = _torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = []
    for s in _specs(meta):
        out.append({"sig": s["name"],
                    "args": _build_args(s["source_sig"], torch, device, m=s.get("m")),
                    "ref": _golden(s["source_sig"], torch, device, m=s.get("m"))})
    return out
