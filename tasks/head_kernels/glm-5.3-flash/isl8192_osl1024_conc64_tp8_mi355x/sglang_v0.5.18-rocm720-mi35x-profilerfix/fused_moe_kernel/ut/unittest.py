#!/usr/bin/env python3
"""IMMUTABLE driver for the fused-MoE (grouped per-expert GEMM + routing) kernel task.

Correctness: h.run_correctness (frozen oracle + random-value parity vs the live baseline leg).
Timing:      h.measure_legs  (baseline_overlay vs _cand_overlay, interleaved fresh subprocesses).
Metric:      h.serving_weighted_speedup (self-weighted: measured baseline ms x analytic serving calls).

Exit codes: 0 pass | 1 correctness FAIL | 2 environment | 3 regenerate UT (harness incomplete).
"""
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# load the vendored libs BEFORE dropping HERE from sys.path / importing torch (this file is named
# unittest.py and would otherwise shadow the stdlib module).
_hs = importlib.util.spec_from_file_location("harness_lib", os.path.join(HERE, "harness_lib.py"))
h = importlib.util.module_from_spec(_hs)
sys.modules["harness_lib"] = h
_hs.loader.exec_module(h)
_cs = importlib.util.spec_from_file_location("cases", os.path.join(HERE, "cases.py"))
cases = importlib.util.module_from_spec(_cs)
sys.modules["cases"] = cases
_cs.loader.exec_module(cases)

sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != HERE]

with open(os.path.join(HERE, "meta.json")) as _fh:
    META = json.load(_fh)
TOL = float(META.get("tol", 2e-2))
DRAWS = int(META.get("random_draws", 3))


def main():
    regime = META.get("regime", {}) or {}
    try:
        eager = cases.eager_cases(h, META)
    except Exception as exc:
        print(f"FAIL: could not build oracle cases: {exc!r}")
        return 2
    if not eager:
        print("FAIL: no oracle cases for the served regimes")
        return 1

    base_out = h.baseline_random_outputs(HERE, META, draws=DRAWS)

    try:
        ok, report = h.run_correctness(
            regime,
            eager_cases=eager,
            current_call=cases.call,
            random_shapes=cases.random_shapes(h, META),
            tol=TOL,
            baseline_outputs=base_out,
            draws=DRAWS,
            replay=None,          # deployment is enforce-eager (--disable-cuda-graph): no graph replay
        )
    except h.HarnessIncompleteError:
        return 3                  # sentinel already printed by run_correctness

    for leg, entries in report.items():
        for e in entries:
            print(f"[correct:{leg}] {json.dumps(e, default=str)}")

    per_case = h.measure_legs(HERE, META)
    for c in per_case:
        print(f"[time] {json.dumps(c, default=str)}")

    w = h.serving_weighted_speedup(per_case, META)
    print(f"GEAK_PER_CASE={json.dumps(per_case, default=str)}")
    print(f"GEAK_WEIGHTED_DETAIL={json.dumps(w, default=str)}")
    print(f"GEAK_GEOMEAN_SPEEDUP={w.get('geomean')}")
    print(f"GEAK_WEIGHTED_SPEEDUP={w.get('weighted')}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except h.HarnessIncompleteError:
        sys.exit(3)
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(2)
