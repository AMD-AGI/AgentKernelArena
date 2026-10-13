#!/usr/bin/env python3
"""AgentKernelArena task runner for triton2triton/fused_moe_int4_w4a16.

Format-only adapter: wraps the kernel's existing harness (test_harness.py, which
uses _bench_common.py) as a subprocess and translates its output into the arena
contract (build/*_report.json + standard stdout). The kernel implementation
(kernel_jit.py) and harness are untouched.

Modes:
  compile     — syntax-check + import kernel_jit.py (the editable Triton source).
  correctness — `python test_harness.py --correctness`; PASS iff CORRECTNESS_OVERALL: PASS.
  performance — `python test_harness.py --full-benchmark`; parse one
                `CASE=<name> GEAK_RESULT_LATENCY_MS=<ms>` line per test case.
"""
import argparse
import json
import os
import re
import subprocess
import sys

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/fused_moe_int4_w4a16"
HARNESS = os.path.join(TASK_DIR, "test_harness.py")
SOURCE_FILE = os.path.join(TASK_DIR, "kernel_jit.py")
BUILD_DIR = os.path.join(TASK_DIR, "build")


def _run_harness(flag, timeout=1800):
    r = subprocess.run(
        [sys.executable, HARNESS, flag], cwd=TASK_DIR,
        capture_output=True, text=True, timeout=timeout,
    )
    return r.returncode, (r.stdout or ""), (r.stderr or "")


def run_compile():
    try:
        import ast
        with open(SOURCE_FILE) as f:
            ast.parse(f.read())
    except Exception as e:
        return False, f"syntax error in kernel_jit.py: {e}"
    # Import the editable module to surface @triton.jit decode/build errors early.
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("kernel_jit", SOURCE_FILE)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "fused_moe_kernel_gptq_awq"), "missing fused_moe_kernel_gptq_awq"
    except Exception as e:
        return False, f"import kernel_jit failed: {str(e)[:240]}"
    return True, None


def run_correctness():
    try:
        rc, out, err = _run_harness("--correctness")
    except Exception as e:
        return False, str(e)
    sys.stdout.write(out)
    if "CORRECTNESS_OVERALL: PASS" in out:
        return True, None
    # surface first FAIL line / stderr tail
    for line in out.splitlines():
        if "FAIL" in line:
            return False, line.strip()[:240]
    return False, (err.strip()[-240:] or "correctness did not report PASS")


def run_performance():
    try:
        rc, out, err = _run_harness("--full-benchmark")
    except Exception as e:
        return [{"test_case_id": "run_fail", "execution_time_ms": -1.0,
                 "params": {"error": str(e)[:240]}}]
    sys.stdout.write(out)
    cases = []
    # regime params per concurrency id (token-parallel: M = B*seqlen, seqlen=1024,
    # gemm1 gate_up shard N=1024 K=7168 group_size=32).
    REGIME = {
        "c2":  {"B": 2,  "seqlen": 1024, "M": 2048,  "N": 1024, "K": 7168, "group_size": 32},
        "c32": {"B": 32, "seqlen": 1024, "M": 32768, "N": 1024, "K": 7168, "group_size": 32},
        "c64": {"B": 64, "seqlen": 1024, "M": 65536, "N": 1024, "K": 7168, "group_size": 32},
    }
    pat = re.compile(r"CASE=(\S+)\s+GEAK_RESULT_LATENCY_MS=([0-9.eE+-]+)")
    for m in pat.finditer(out):
        cid = m.group(1)
        cases.append({
            "test_case_id": cid,
            "execution_time_ms": float(m.group(2)),
            "params": REGIME.get(cid, {}),
        })
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
        print(f"Compilation: {'PASS' if ok else 'FAIL'}")
        if err: print(f"Error: {err}")
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
        for c in cases:
            print(f"Performance: {c['execution_time_ms']} ms ({c['test_case_id']})")
        sys.exit(0)


if __name__ == "__main__":
    main()
