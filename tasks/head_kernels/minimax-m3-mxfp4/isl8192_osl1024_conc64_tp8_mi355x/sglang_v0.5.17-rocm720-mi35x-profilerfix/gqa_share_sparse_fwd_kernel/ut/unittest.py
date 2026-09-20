#!/usr/bin/env python3
"""IMMUTABLE unittest for the MiniMax-M3 block-sparse GQA PREFILL seam
(`_gqa_share_sparse_fwd_kernel`, via `flash_prefill_with_gqa_share_sparse`).

Optimizers edit ONLY kernel_src/. Editing this file, meta.json, reference_io.pt or timing_geometry.pt
invalidates the result — their sha256s are recorded in meta.json and re-checked downstream.

What it proves, in order:
  1. CORRECTNESS vs the RECORDED live oracle, streamed lazily (each record of this op is ~2.4 GiB
     because the paged KV pool is an input) — h.iter_eager_cases_from_oracle + h.check_correct_multi_lazy.
  2. VALUE PARITY vs the frozen live baseline on fresh random draws at the same online geometry, and
     the FAIL-CLOSED CUDA-graph replay gate (regime.cuda_graph is true for this deployment) — both via
     the single h.run_correctness entrypoint.
  3. SPEEDUP from two SEPARATE subprocess legs (baseline_overlay vs _cand_overlay), device-event timed,
     folded by the serving weight model in meta.workload.

Exit codes: 0 pass | 1 correctness/perf-contract fail | 2 environment error | 3 UT harness incomplete.
"""
import json
import os
import sys
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
# This file is named unittest.py and sits next to the modules we import; torch imports stdlib
# `unittest`, so the task dir must NOT be on sys.path when torch loads. Load harness_lib by path
# first, then keep sys.path clean.
import importlib.util as _ilu


def _load(name, path):
    existing = sys.modules.get(name)
    if existing is not None:
        if os.path.realpath(getattr(existing, "__file__", "")) != os.path.realpath(path):
            raise RuntimeError(f"trusted helper alias names a different file: {name}")
        return existing
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
h = _load("harness_lib", os.path.join(_HERE, "harness_lib.py"))
cases = _load("cases", os.path.join(_HERE, "cases.py"))
import torch  # noqa: E402


def _build_replay(meta, baseline_outputs):
    """Static-buffer capture-once / replay-many bundle.

    Both boundary variants share one buffer set (same shapes, same launch grid, different ragged
    split), so `fill` is a pure copy_ into the captured storage — never a realloc. `run` issues one
    launch and copies its result into the static output. Refs come from the BASELINE leg's recorded
    outputs for the same sig/seed, so the candidate is never its own golden.
    """
    shapes = cases.replay_shapes(h, meta)
    built = []
    for s in shapes:
        ref = baseline_outputs.get(f"{s['sig']}|0")
        if ref is None:
            continue
        rng = torch.Generator(device="cuda").manual_seed(0)
        # The baseline leg records to CPU; check_graph_replay compares the ref as-is, and a
        # cross-device compare inside correct() degrades to (False, inf) rather than raising, so the
        # move has to happen here or every replay case reports a phantom failure.
        built.append({"sig": s["sig"], "args": s["make_inputs"](rng),
                      "ref": h.to_device_like(ref, "cuda")})
    if len(built) < 2:
        return None

    static = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in built[0]["args"].items()}
    out_static = cases.call(static)          # one eager launch to size the static output
    torch.cuda.synchronize()

    def fill(case):
        for k, v in case["args"].items():
            if torch.is_tensor(v):
                static[k].copy_(v)
            else:
                # host scalars steer the grid and are held IDENTICAL across the variants by
                # construction; assert rather than silently re-capture with a different grid.
                if static[k] != v:
                    raise RuntimeError(f"replay variant changed host arg {k!r} "
                                       f"({static[k]!r} -> {v!r}); grid would differ")

    def run():
        out_static.copy_(cases.call(static))

    def read_out():
        return out_static

    return {"fill": fill, "run": run, "read_out": read_out, "cases": built, "capture_idx": 0}


def main():
    with open(os.path.join(_HERE, "meta.json")) as fh:
        meta = json.load(fh)
    regime = meta.get("regime", {})
    tol = float(meta.get("tol", 2e-2))
    draws = int(meta.get("random_draws", 3))
    report = {"kernel": meta.get("short_name"), "op_kind": meta.get("op_kind"),
              "served_regimes": meta.get("served_regimes")}

    if not torch.cuda.is_available():
        print("ENV ERROR: no GPU visible")
        return 2

    # --- 1. recorded-oracle correctness, streamed (multi-GiB records) -------------------------
    ok_eager, per_eager = h.check_correct_multi_lazy(
        cases.call, cases.eager_cases(h, meta, device="cuda"), tol, max_keep_live=2)
    report["eager"] = per_eager
    if not per_eager:
        print(f"{h.UT_HARNESS_INCOMPLETE_SENTINEL}: oracle produced no cases")
        print(json.dumps(report, indent=1))
        return 3

    # --- 2. value parity vs the frozen live baseline + fail-closed graph replay ----------------
    baseline_outputs = h.baseline_random_outputs(_HERE, meta, seed=0, draws=draws)
    replay = _build_replay(meta, baseline_outputs)
    try:
        # eager_cases=[] here on purpose: leg 1 above ALREADY ran the oracle comparison lazily (and
        # the output-independence check with it). Passing the materialized list a second time would
        # need every ~2.4 GiB record resident at once. The fail-closed graph gate is untouched.
        ok_rest, rest = h.run_correctness(
            regime, eager_cases=[], current_call=cases.call,
            random_shapes=cases.random_shapes(h, meta), tol=tol,
            baseline_outputs=baseline_outputs, replay=replay, draws=draws)
    except h.HarnessIncompleteError:
        report.update({"correct": False, "reason": "harness_incomplete"})
        print(json.dumps(report, indent=1))
        return 3
    report.update({k: v for k, v in rest.items() if k != "eager"})
    correct = bool(ok_eager and ok_rest)
    report["correct"] = correct

    # --- 3. two-leg, device-event timed, serving-weighted speedup ------------------------------
    per_case = h.measure_legs(_HERE, meta)
    report["per_case"] = per_case
    weighted = h.serving_weighted_speedup(per_case, meta)
    report["speedup"] = weighted

    print(json.dumps(report, indent=1))
    if not correct:
        print("CORRECTNESS: FAIL")
        return 1
    print("CORRECTNESS: PASS")
    sp = (weighted or {}).get("primary")
    print(f"SPEEDUP: {sp}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except h.HarnessIncompleteError:
        sys.exit(3)
    except Exception:
        traceback.print_exc()
        sys.exit(2)
