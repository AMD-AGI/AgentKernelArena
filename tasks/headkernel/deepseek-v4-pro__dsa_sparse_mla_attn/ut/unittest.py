#!/usr/bin/env python3
"""IMMUTABLE correctness + workload-weighted timing harness for the DSA fused sparse-MLA seam.

Judges `meta.target_callable` as the LIVE stack resolves it. The two legs are the SAME code
(cases.py + leg_runner.py) under two PYTHONPATHs; nothing here binds a baseline callable.

Exit codes: 0 pass | 1 correctness/measurement FAIL | 2 env error | 3 UT is incomplete (regenerate).
"""
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# harness_lib FIRST, before HERE leaves sys.path / torch is imported.
h = _load("harness_lib", os.path.join(HERE, "harness_lib.py"))
# this file is named unittest.py — drop its dir so it cannot shadow stdlib `unittest` for torch.
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != HERE]
cases = _load("cases", os.path.join(HERE, "cases.py"))

with open(os.path.join(HERE, "meta.json")) as _fh:
    META = json.load(_fh)

TOL = float(META.get("tol", 2e-2))


def _replay_bundle(base_out):
    """Capture-once / replay-many bundle over >=2 BOUNDARY cases at the deployed decode shape.

    Under deployment this op runs inside the server's decode CUDA graph, where M is PADDED to the
    captured batch size — so M cannot vary between replays of one captured graph, and the boundary
    axis that IS reused across replays is the sparse INDEX CONTENT (page ids) feeding the gather:
      case 0  recorded  : the real scattered page ids from the oracle (capture case)
      case 1  shortctx  : every row reads the SAME 2048 pages (extreme locality / duplicate-index
                          edge) — the shape that trips an index-dependent workspace or an OOB write
                          sized to the first replay's access pattern.
    `ref` for case 0 is the frozen oracle; for case 1 it is the BASELINE leg's own output for the
    matching deterministic `random_shapes` entry (recorded in the baseline process, same inputs).
    """
    torch = h._torch()
    if not torch.cuda.is_available():
        return None
    dev = "cuda"
    # the spec the graph is captured on: meta.case_specs[].replay (the deployed decode batch of the
    # HEAVIER of the two dsv4 layer families — the static buffers are one shape, and that family is
    # the one whose gather can run off the end of a buffer sized by the first replay).
    dec = [s for s in (META.get("case_specs") or []) if s.get("replay")]
    if not dec:
        return None
    s = max(dec, key=lambda x: int(x.get("m") or 0))
    key = "replay_shortctx:" + s["name"] + "|0"
    if key not in (base_out or {}):
        return None
    a_rec = cases._build_args(s["source_sig"], torch, dev, m=s.get("m"))
    a_alt = cases._build_args(s["source_sig"], torch, dev, m=s.get("m"), index_mode="shortctx")
    ref_rec = cases._golden(s["source_sig"], torch, dev, m=s.get("m"))
    ref_alt = h.to_device_like(base_out[key], torch.device(dev))

    # Static input storage, allocated ONCE at the capture case's shape. The paged KV pools are the
    # (single, shared) full-size buffers both cases index into, exactly as the layer has one pool per
    # cache; only the per-token inputs that the replay actually varies get private storage.
    varying = ("q",) + tuple(n for _, n in cases._KV_PAIRS)
    static = dict(a_rec)
    for name in varying:
        t = static.get(name)
        if torch.is_tensor(t):
            static[name] = t.clone()
    st = {"out": None}

    def fill(c):
        for name in varying:
            t = static.get(name)
            if torch.is_tensor(t) and torch.is_tensor(c["args"].get(name)):
                t.copy_(c["args"][name])

    def run():
        st["out"] = cases.call(static)

    def read_out():
        return st["out"]

    return {"fill": fill, "run": run, "read_out": read_out, "capture_idx": 0,
            "cases": [{"sig": "replay_recorded_idx:" + s["name"], "args": a_rec, "ref": ref_rec},
                      {"sig": "replay_shortctx_idx:" + s["name"], "args": a_alt, "ref": ref_alt}]}


def _ordered_cases():
    """meta.call_sequence -> the real interleave, for the cross-call stale-state check."""
    seq = META.get("call_sequence") or []
    if not seq:
        return []
    torch = h._torch()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = []
    for name in seq:
        s = cases._spec_by_name(META, name)
        if s is None:
            continue
        out.append({"sig": s["name"],
                    "args": cases._build_args(s["source_sig"], torch, dev, m=s.get("m")),
                    "ref": cases._golden(s["source_sig"], torch, dev, m=s.get("m"))})
    return out


def main():
    regime = META.get("regime", {})
    draws = int(META.get("random_draws", 3))
    ok = True

    eager = cases.eager_cases(h, META)
    rshapes = cases.random_shapes(h, META)
    base_out = h.baseline_random_outputs(HERE, META, draws=draws)

    try:
        c_ok, report = h.run_correctness(
            regime, eager_cases=eager, current_call=cases.call, random_shapes=rshapes,
            tol=TOL, baseline_outputs=base_out, draws=draws, replay=_replay_bundle(base_out))
    except h.HarnessIncompleteError:
        sys.exit(3)                      # sentinel already printed by run_correctness
    ok = ok and c_ok
    print(json.dumps({"correctness": report}, indent=2, default=str))

    ordered = _ordered_cases()
    if ordered:
        s_ok, s_rep = h.check_correct_sequence(cases.call, ordered, TOL)
        ok = ok and s_ok
        print(json.dumps({"call_sequence": s_rep}, indent=2, default=str))

    print("CORRECTNESS: " + ("PASS" if ok else "FAIL"))

    per_case = h.measure_legs(HERE, META)
    res = h.serving_weighted_speedup(per_case, META)
    print(json.dumps({"per_case": res["per_case"], "geomean": res["geomean"],
                      "included": res["included"], "dropped_unserved": res["dropped_unserved"],
                      "suspect_identity": res["suspect_identity"], "reason": res["reason"]},
                     indent=2, default=str))
    w = res.get("weighted")
    if w is None:
        print("GEAK_WEIGHTED_SPEEDUP: UNTRUSTED")
        ok = False
    else:
        print("GEAK_WEIGHTED_SPEEDUP: %.4f" % w)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:                       # env / setup failure, not a kernel verdict
        import traceback
        traceback.print_exc()
        print(f"ENV_ERROR: {type(exc).__name__}: {exc}")
        sys.exit(2)
