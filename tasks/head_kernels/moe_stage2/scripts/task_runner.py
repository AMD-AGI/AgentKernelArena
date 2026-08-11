#!/usr/bin/env python3
"""AgentKernelArena task runner for hip2hip/moe_stage2.

Format-only adapter: wraps the existing GEAK harness
(harness_test_moe_stage2_runtime.py) as a subprocess. The CK MoE GEMM is real
C++ JIT-compiled by aiter, so `compile` here invalidates the cached module (the
cache-busting logic from rebuild_and_test.sh) and the next correctness/perf run
recompiles the edited .cuh (~80s). Kernel and harness are untouched.

Modes:
  compile     — delete cached module_moe_ck2stages_*silu_per_1x128* (.so + build).
  correctness — `harness --correctness`; PASS iff "ALL CORRECTNESS CHECKS PASSED"
                (this triggers the ~80s recompile).
  performance — `harness --benchmark`; parse one ms entry per token config.
"""
import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "hip2hip/moe_stage2"
HARNESS = os.path.join(TASK_DIR, "harness_test_moe_stage2_runtime.py")
BUILD_DIR = os.path.join(TASK_DIR, "build")
JIT = os.path.join(TASK_DIR, "aiter_local", "jit")

CFG_LINE = re.compile(r"^\s*token=(\d+)\b.*?([0-9.]+)ms\s*$")
GEO_LINE = re.compile(r"GEAK_RESULT_LATENCY_MS=([0-9.eE+-]+)")


def _run_harness(flag, timeout=1800):
    r = subprocess.run(
        [sys.executable, HARNESS, flag], cwd=TASK_DIR,
        capture_output=True, text=True, timeout=timeout,
    )
    return r.returncode, (r.stdout or ""), (r.stderr or "")


def run_compile():
    """Invalidate the cached block-scale MoE GEMM modules so the next kernel call
    recompiles the edited CK source (matches rebuild_and_test.sh)."""
    try:
        removed = 0
        for so in glob.glob(os.path.join(JIT, "module_moe_ck2stages_*silu_per_1x128*.so")):
            os.remove(so); removed += 1
        for d in glob.glob(os.path.join(JIT, "build", "module_moe_ck2stages_*silu_per_1x128*")):
            shutil.rmtree(d, ignore_errors=True); removed += 1
        note = (f"invalidated {removed} cached artifact(s)" if removed
                else "no cached module found (will compile on first run)")
        print(f"[compile] {note}")
        return True, None
    except Exception as e:
        return False, str(e)


def run_correctness():
    try:
        rc, out, err = _run_harness("--correctness")
    except Exception as e:
        return False, str(e)
    sys.stdout.write(out)
    if "ALL CORRECTNESS CHECKS PASSED" in out:
        return True, None
    for line in out.splitlines():
        if "FAIL" in line or "ERROR" in line:
            return False, line.strip()[:240]
    return False, (err.strip()[-240:] or "correctness did not report PASS")


def run_performance():
    try:
        rc, out, err = _run_harness("--benchmark")
    except Exception as e:
        return [{"test_case_id": "run_fail", "execution_time_ms": -1.0,
                 "params": {"error": str(e)[:240]}}]
    sys.stdout.write(out)
    cases = []
    for line in out.splitlines():
        m = CFG_LINE.match(line)
        if m:
            tok, ms = m.groups()
            tok_i = int(tok)
            # Arena regime: token-parallel MoE, M = B*1024 -> id c{B}.
            cid = f"c{tok_i // 1024}" if tok_i % 1024 == 0 else f"token{tok_i}"
            cases.append({"test_case_id": cid, "execution_time_ms": float(ms),
                          "params": {"token": tok_i, "concurrency": tok_i // 1024,
                                     "seqlen": 1024}})
    g = GEO_LINE.search(out)
    if g:
        cases.append({"test_case_id": "geomean", "execution_time_ms": float(g.group(1)),
                      "params": {"aggregate": "geomean"}})
    if not cases:
        cases.append({"test_case_id": "no_result", "execution_time_ms": -1.0,
                      "params": {"stderr": err.strip()[-240:]}})
    return cases


def main():
    p = argparse.ArgumentParser(description=f"Task runner for {TASK_NAME}")
    p.add_argument("mode", choices=["compile", "correctness", "performance"])
    args = p.parse_args()
    os.makedirs(BUILD_DIR, exist_ok=True)

    if args.mode == "compile":
        ok, err = run_compile()
        json.dump({"status": "ok" if ok else "fail", "error": err},
                  open(os.path.join(BUILD_DIR, "compile_report.json"), "w"), indent=2)
        print(f"Compilation: {'PASS' if ok else 'FAIL'} (cache invalidated)")
        if err: print(f"Note: {err}")
        sys.exit(0 if ok else 1)
    elif args.mode == "correctness":
        ok, err = run_correctness()
        json.dump({"status": "ok" if ok else "fail", "error": err},
                  open(os.path.join(BUILD_DIR, "correctness_report.json"), "w"), indent=2)
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if err: print(f"Error: {err}")
        sys.exit(0 if ok else 1)
    elif args.mode == "performance":
        cases = run_performance()
        # Arena contract: {"test_cases":[{"test_case_id","execution_time_ms","params"}]}.
        # Keep per-concurrency cases (drop the geomean aggregate from the contract list).
        perf_cases = [c for c in cases if c["test_case_id"] != "geomean"]
        json.dump({"test_cases": perf_cases},
                  open(os.path.join(BUILD_DIR, "performance_report.json"), "w"), indent=2)
        for c in perf_cases:
            if c["execution_time_ms"] > 0:
                print(f"Performance: {c['execution_time_ms']:.4f} ms ({c['test_case_id']})")
        sys.exit(0)


if __name__ == "__main__":
    main()
