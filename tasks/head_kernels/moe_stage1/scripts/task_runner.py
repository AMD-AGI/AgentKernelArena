#!/usr/bin/env python3
"""AgentKernelArena task runner for hip2hip/moe_stage1.

Format-only adapter: wraps the existing GEAK harness
(harness_test_moe_stage1_runtime.py) as a subprocess. The CK MoE GEMM is real
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

TASK_NAME = "hip2hip/moe_stage1"
HARNESS = os.path.join(TASK_DIR, "harness_test_moe_stage1_runtime.py")
BUILD_DIR = os.path.join(TASK_DIR, "build")
JIT = os.path.join(TASK_DIR, "aiter_local", "jit")

CFG_LINE = re.compile(r"^\s*token=(\d+)\b.*?([0-9.]+)ms\s*$")
GEO_LINE = re.compile(r"GEAK_RESULT_LATENCY_MS=([0-9.eE+-]+)")

# Fixed workload regime: seqlen=1024, concurrency B in {2,32,64}.
# moe_stage1 is a token-parallel MoE GEMM -> num_tokens M = B*1024.
REGIME = [(2, 2048), (32, 32768), (64, 65536)]  # (B, token=B*1024)
TOKEN_TO_ID = {tok: f"c{b}" for b, tok in REGIME}


def _run_harness(flag, timeout=1800, extra_env=None):
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    r = subprocess.run(
        [sys.executable, HARNESS, flag], cwd=TASK_DIR,
        capture_output=True, text=True, timeout=timeout, env=env,
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
    # Pin the harness token sweep to the fixed regime (B*1024 tokens) and map
    # each token to its concurrency id c2/c32/c64.
    tokens = ",".join(str(tok) for _, tok in REGIME)
    try:
        rc, out, err = _run_harness("--benchmark", extra_env={"GEAK_TOKENS": tokens})
    except Exception as e:
        return [{"test_case_id": "run_fail", "execution_time_ms": -1.0,
                 "params": {"error": str(e)[:240]}}]
    sys.stdout.write(out)
    cases = []
    for line in out.splitlines():
        m = CFG_LINE.match(line)
        if m:
            tok, ms = m.groups()
            tok = int(tok)
            cid = TOKEN_TO_ID.get(tok, f"token{tok}")
            b = next((bb for bb, tt in REGIME if tt == tok), None)
            params = {"token": tok, "seqlen": 1024}
            if b is not None:
                params["concurrency"] = b
            cases.append({"test_case_id": cid, "execution_time_ms": float(ms),
                          "params": params})
    # Print one canonical Performance line per regime case.
    for c in cases:
        print(f"Performance: {c['execution_time_ms']} ms ({c['test_case_id']})")
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
        json.dump({"test_cases": cases},
                  open(os.path.join(BUILD_DIR, "performance_report.json"), "w"), indent=2)
        ok = [c for c in cases if c["execution_time_ms"] > 0]
        print(f"Performance: measured {len(ok)}/{len(cases)} case(s)")
        sys.exit(0)


if __name__ == "__main__":
    main()
