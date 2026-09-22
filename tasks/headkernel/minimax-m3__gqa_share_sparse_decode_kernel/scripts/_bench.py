#!/usr/bin/env python3
"""Native arena-methodology timing for a head-kernel task.

Replays the argument records frozen in ``ut/reference_io.pt`` -- the same live
captures the correctness oracle is built from -- against the op's production
callable, and times each one with 10 warmup + 100 measured iterations, reported
as the mean of per-iteration cuda-event device times.

Run by ``scripts/task_runner.py performance`` in a subprocess whose PYTHONPATH
carries the GEAK candidate overlay, so the callable resolved here is the code in
``source/``. Exits 4 when the oracle carries no replayable records, which tells
the runner to fall back to the GEAK harness legs.
"""
from __future__ import annotations

import argparse
import ast
import importlib
import json
import os
import sys

EXIT_NO_RECORDS = 4
MAX_CASES = 24          # captures can hold hundreds of near-identical records


def resolve(dotted):
    mod_name, _, attr = dotted.partition(":")
    obj = importlib.import_module(mod_name)
    for part in attr.split("."):
        if part:
            obj = getattr(obj, part)
    # The candidate overlay loads source/ as a TOP-LEVEL module (candidate_bind
    # names e.g. impl_module "geak_flydsl_moe_kernels"), so a kernel that does a
    # lazy relative import inside a function -- the DeepSeek MoE stages do seven
    # of them, `from .kernels.mixed_moe_gemm_2stage import ...` -- raises
    # "attempted relative import with no known parent package" as soon as that
    # branch is taken. Python resolves a relative import against the __package__
    # of the module the CODE lives in, so the fix belongs on the function's own
    # globals, not on the seam module. Point it at the seam's package so the
    # relative form resolves exactly as it does in the unshadowed module.
    # Which package the relative form is relative TO: if the seam module is itself
    # a package (has __path__, e.g. aiter.ops.flydsl/) then `.kernels` means
    # aiter.ops.flydsl.kernels and __package__ is the module's own dotted name;
    # if it is a plain module inside a package, __package__ is its parent.
    seam = sys.modules.get(mod_name)
    pkg = mod_name if hasattr(seam, "__path__") else mod_name.rpartition(".")[0]
    g = getattr(obj, "__globals__", None)
    if pkg and isinstance(g, dict) and not g.get("__package__"):
        g["__package__"] = pkg
    return obj


def _dtype(torch, spec):
    """torch dtype from the several spellings the captures use.

    Returns None for a missing or empty value -- ``packed_view_dtype`` is
    present but EMPTY on most records, and feeding "" to getattr(torch, ...)
    raises AttributeError rather than meaning "no re-view".
    """
    if spec is None or spec == "":
        return None
    if isinstance(spec, str):
        return getattr(torch, spec.replace("torch.", ""), None)
    return spec


def _literal(spec):
    """Rebuild a non-tensor argument the capture stored by repr.

    Enum members repr as ``<QuantType.per_1x32: 3>``, which is not a literal, so
    fall back to resolving the dotted name against its module.
    """
    text = spec.strip()
    try:
        return ast.literal_eval(text)
    except Exception:
        pass
    if text.startswith("<") and ":" in text:                # <QuantType.per_1x32: 3>
        dotted = text[1:text.index(":")].strip()
        cls, _, member = dotted.rpartition(".")
        for mod in ("aiter", "aiter.utility.fp4_utils", "aiter.jit.core"):
            try:
                obj = importlib.import_module(mod)
                return getattr(getattr(obj, cls.rpartition(".")[2]), member)
            except Exception:
                continue
    raise ValueError(f"cannot rebuild {spec!r}")


def unsnap(x, device, torch, shared=None):
    """Rebuild live arguments from the snapshots an oracle stores.

    Captures across the GEAK generations in this suite use several encodings.
    Handling only ``__tensor__`` (which is all this script used to do) silently
    skipped every record of any oracle that uses the others, which read as "the
    oracle carries no replayable records" and dropped the task onto the GEAK
    fallback path.
    """
    shared = shared or {}
    if isinstance(x, dict):
        # A back-reference into the blob's deduplicated pool. Big operands - MoE
        # expert weights, the gated-delta recurrent state - are stored once.
        if "__shared__" in x and len(x) == 1:
            name = x["__shared__"]
            if name not in shared:
                raise KeyError(f"oracle references shared operand {name!r} that is not in "
                               f"blob['shared'] (have: {sorted(shared)[:8]})")
            return unsnap(shared[name], device, torch, shared)
        if x.get("__tensor__"):
            t = x["data"]
            t = t.to(device) if hasattr(t, "to") else t
            # Sub-byte dtypes are stored as uint8 and re-viewed on the way out.
            # Two generations spell that differently: the newer one names the
            # target dtype in `packed_view_dtype`, the older one sets `bitcast`
            # to the STORAGE dtype and keeps the target in `dtype`. Both are
            # frequently present but empty, which is not a dtype.
            view = _dtype(torch, x.get("packed_view_dtype"))
            if view is None and x.get("bitcast"):
                view = _dtype(torch, x.get("dtype"))
            if view is not None and hasattr(t, "view") and getattr(t, "dtype", None) != view:
                t = t.view(view)
            stride, shape = x.get("stride"), x.get("shape")
            if stride is not None and shape is not None and hasattr(t, "as_strided"):
                # An explicit stride is part of the contract for these ops; do not
                # flatten it by calling .contiguous().
                return t.as_strided(tuple(shape), tuple(stride))
            return t.contiguous() if x.get("contiguous", True) and hasattr(t, "contiguous") else t
        # An output or workspace buffer the callable writes into: recreate it
        # empty rather than carrying its bytes in the oracle.
        if x.get("__tensor_factory__"):
            dtype = _dtype(torch, x.get("dtype"))
            shape, stride = tuple(x["shape"]), x.get("stride")
            if stride:
                return torch.empty_strided(shape, tuple(stride), dtype=dtype, device=device)
            return torch.empty(shape, dtype=dtype, device=device)
        if "__torch_dtype__" in x:
            return _dtype(torch, x["__torch_dtype__"])
        if "__repr__" in x:
            return _literal(x["__repr__"])
        if "__enum__" in x:
            # {"__enum__": "<module>:<Qualname>", "name": "<member>"} -- the member
            # name is in "name", not in the "__enum__" path.
            mod_name, _, qual = str(x["__enum__"]).partition(":")
            obj = importlib.import_module(mod_name)
            for part in qual.split("."):
                obj = getattr(obj, part)
            return obj[x["name"]] if "name" in x else obj(x["value"])
        return {k: unsnap(v, device, torch, shared) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(unsnap(v, device, torch, shared) for v in x)
    if hasattr(x, "to") and hasattr(x, "device"):
        return x.to(device)
    return x


def record_args(rec, device, torch, shared=None):
    """(args, kwargs) for one record, tolerating the capture spellings."""
    raw = rec.get("args")
    if raw is None:
        raw = rec.get("pos")
    kw = rec.get("kwargs")
    if kw is None:
        kw = rec.get("kw")                   # geak-attn-oracle-v1 spelling
    if raw is None and kw is None:
        return None, None
    args = unsnap(list(raw or ()), device, torch, shared)
    kwargs = unsnap(dict(kw or {}), device, torch, shared)
    return args, kwargs


def time_call(call, torch, warmup, iters):
    """Mean / median / min per-call device milliseconds."""
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        call()
        ends[i].record()
    torch.cuda.synchronize()

    samples = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    n = len(samples)
    median = samples[n // 2] if n % 2 else 0.5 * (samples[n // 2 - 1] + samples[n // 2])
    return {"mean_ms": sum(samples) / n, "median_ms": median, "min_ms": samples[0]}


def _load_module(name, path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def native_cases(ut_dir, meta, torch):
    """Time the op through the package's own case builder.

    Some ops are value-independent, so their UT regenerates a deterministic
    baseline at run time instead of freezing hundreds of MB of tensors. Those
    packages have no ``reference_io.pt`` to replay -- but they do expose the
    live geometries through ``cases.py``, which is how their own unit test
    builds them. Using that is strictly better than the alternative of reporting
    no measurement at all, and it keeps the arena's 10/100 methodology.

    Returns a list of ``(sig, regime, callable)`` or [] when the package has no
    such entry point.
    """
    cases_py = os.path.join(ut_dir, "cases.py")
    harness = os.path.join(ut_dir, "harness_lib.py")
    if not (os.path.isfile(cases_py) and os.path.isfile(harness)):
        return []

    # The UT sets this at module scope, before aiter is imported, so the
    # dispatcher picks the solutions that were actually observed live. Without
    # it the replay would time aiter's stock config and measure a different
    # kernel than the one this task is defined by.
    if meta.get("dispatch_config"):
        os.environ["AITER_CONFIG_GEMM_BF16"] = os.path.join(ut_dir, meta["dispatch_config"])

    sys.path.insert(0, ut_dir)
    try:
        h = _load_module("harness_lib", harness)
        sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != ut_dir]
        sys.modules.pop("unittest", None)
        cases = _load_module("_hk_cases", cases_py)
    except Exception as exc:
        print(f"[native] cannot load ut/cases.py: {type(exc).__name__}: {exc}", file=sys.stderr)
        return []

    out = []
    # Spelling A: per-case builder keyed off the meta case map (dense GEMM).
    if hasattr(cases, "timing_case") and hasattr(cases, "candidate_call"):
        try:
            selected = cases.selected_cases(meta, meta.get("ledger_ids") or [])
        except Exception:
            selected = list((cases.case_map(meta) or {}).values())
        for case in selected:
            t = cases.timing_case(case)
            out.append((t["sig"], t.get("regime", ""),
                        (lambda a=t["args"]: cases.candidate_call(a))))
    # Spelling B: whole-workload builder (add+RMSNorm).
    elif hasattr(cases, "timing_cases") and hasattr(cases, "call"):
        for t in cases.timing_cases(h, meta):
            out.append((t["sig"], t.get("regime", ""),
                        (lambda a=t["args"]: cases.call(a))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ut", required=True, help="path to the task's ut/ directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    oracle = os.path.join(args.ut, "reference_io.pt")
    meta_path = os.path.join(args.ut, "meta.json")
    meta_early = json.load(open(meta_path)) if os.path.isfile(meta_path) else {}

    if not os.path.isfile(oracle):
        # No frozen blob. Before giving up, try the package's own case builder.
        try:
            import torch
        except Exception as exc:
            print(f"cannot import torch: {exc}", file=sys.stderr)
            return 2
        if not torch.cuda.is_available():
            print("no CUDA/HIP device visible", file=sys.stderr)
            return 2
        native = native_cases(args.ut, meta_early, torch)
        if not native:
            print("no ut/reference_io.pt and no usable ut/cases.py entry point",
                  file=sys.stderr)
            return EXIT_NO_RECORDS
        cases = []
        for sig, regime, call in native:
            try:
                timing = time_call(call, torch, args.warmup, args.iters)
            except Exception as exc:
                print(f"[skip] {sig}: {type(exc).__name__}: {exc}", file=sys.stderr)
                continue
            cases.append({"sig": f"{sig}|{regime}" if regime else sig,
                          "params": {"regime": regime, "source": "ut/cases.py live geometry"},
                          **timing})
            torch.cuda.empty_cache()
        if not cases:
            return EXIT_NO_RECORDS
        with open(args.out, "w") as fh:
            json.dump({"target": meta_early.get("target_callable"), "timer": "cuda_event",
                       "warmup": args.warmup, "iters": args.iters,
                       "case_source": "ut/cases.py live geometries (no frozen oracle)",
                       "cases": cases}, fh, indent=2)
        for c in cases:
            print(f"[time] {c['sig']}: mean {c['mean_ms']:.6f} ms "
                  f"(median {c['median_ms']:.6f}, min {c['min_ms']:.6f})")
        return 0

    import torch
    if not torch.cuda.is_available():
        print("no CUDA/HIP device visible", file=sys.stderr)
        return 2
    device = "cuda"

    blob = torch.load(oracle, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict) or not blob.get("records"):
        print("oracle carries no 'records' - synthetic-spec layout, cannot replay",
              file=sys.stderr)
        return EXIT_NO_RECORDS

    # Prefer meta.json's target_callable over the blob's. They differ for the
    # paged-attention capture, where the blob names the sglang alias
    # (sglang...aiter_backend:paged_attention_ragged) while candidate_bind
    # rebinds aiter.ops.attention:paged_attention_ragged. Timing the alias can
    # resolve to the unmodified production function, i.e. measure the wrong code.
    meta = json.load(open(meta_path)) if os.path.isfile(meta_path) else {}
    target = meta.get("target_callable") or blob.get("target")
    if not target:
        print("no target callable recorded", file=sys.stderr)
        return EXIT_NO_RECORDS
    fn = resolve(target)
    shared = blob.get("shared") or {}

    cases, seen = [], set()
    for i, rec in enumerate(blob["records"]):
        if len(cases) >= MAX_CASES:
            break
        sig = rec.get("sig") or f"record{i}"
        regime = (rec.get("regime") or "").lower()
        key = (sig, regime)
        if key in seen:
            continue
        seen.add(key)

        # Rebuilding the arguments is as failure-prone as calling the kernel --
        # an unhandled encoding used to abort the whole process here rather than
        # skip one record, which read downstream as "this oracle has no
        # replayable records at all".
        try:
            pos, kw = record_args(rec, device, torch, shared)
        except Exception as exc:
            print(f"[skip] {sig}: cannot rebuild arguments: {type(exc).__name__}: {exc}",
                  file=sys.stderr)
            continue
        if pos is None:
            continue
        try:
            timing = time_call(lambda: fn(*pos, **kw), torch, args.warmup, args.iters)
        except Exception as exc:                       # one bad record must not sink the sweep
            print(f"[skip] {sig}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        operands = list(pos) + [kw[k] for k in sorted(kw)]   # some ops are kwargs-only
        shapes = [list(t.shape) for t in operands if hasattr(t, "shape")][:4]
        cases.append({"sig": f"{sig[:60]}|{regime}" if regime else sig[:60],
                      "params": {"regime": regime, "operand_shapes": shapes},
                      **timing})
        del pos, kw
        torch.cuda.empty_cache()

    if not cases:
        # The oracle has records but none of them survived a direct call. That
        # happens when the op needs setup the generic replay path knows nothing
        # about -- the DeepSeek MoE stages, for instance, reject a raw kwargs
        # replay with "tile_n=256 does not divide inter_dim=384" while their own
        # cases.py builds a valid configuration. Fall back to the package's own
        # case builder before declaring the oracle unreplayable.
        print("no record could be replayed directly; trying ut/cases.py",
              file=sys.stderr)
        native = native_cases(args.ut, meta, torch)
        for sig, regime, call in native:
            try:
                timing = time_call(call, torch, args.warmup, args.iters)
            except Exception as exc:
                print(f"[skip] {sig}: {type(exc).__name__}: {exc}", file=sys.stderr)
                continue
            cases.append({"sig": f"{sig}|{regime}" if regime else sig,
                          "params": {"regime": regime,
                                     "source": "ut/cases.py live geometry"},
                          **timing})
            torch.cuda.empty_cache()
    if not cases:
        print("no record could be replayed", file=sys.stderr)
        return EXIT_NO_RECORDS

    with open(args.out, "w") as fh:
        json.dump({"target": target, "timer": "cuda_event",
                   "warmup": args.warmup, "iters": args.iters,
                   "cases": cases}, fh, indent=2)
    for c in cases:
        print(f"[time] {c['sig']}: mean {c['mean_ms']:.6f} ms "
              f"(median {c['median_ms']:.6f}, min {c['min_ms']:.6f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
