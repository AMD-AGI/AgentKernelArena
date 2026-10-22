#!/usr/bin/env python3
"""AgentKernelArena task runner for triton2triton/gemm_a8w8_blockscale.

Drives the Triton block-scale FP8 GEMM (target_kernel_functions: gemm_a8w8_blockscale,
the editable optimization entry in the installed aiter tree) via scripts/harness_run.py,
and translates output into the arena contract (build/*_report.json + standard stdout).

Workload regime: token-parallel GEMM, M = B*1024, concurrency B in {2,32,64}
-> 3 cases c2/c32/c64, model dims N=4096 K=3072 (qkv_proj). See harness_run.py.

Modes:
  compile     -> import aiter + the Triton kernel module (build check).
  correctness -> harness_run --correctness; golden-vs-edited; PASS iff
                 "ALL CORRECTNESS CHECKS PASSED".
  performance -> harness_run --benchmark; real CUDA-event timing, one ms entry per
                 concurrency case (c2/c32/c64).
"""
import argparse
import json
import os
import re
import subprocess
import sys

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "triton2triton/gemm_a8w8_blockscale"
HARNESS = os.path.join(TASK_DIR, "scripts", "harness_run.py")
BUILD_DIR = os.path.join(TASK_DIR, "build")

# "  c2 M=2048 N=4096 K=3072 fp8->bf16  0.2373ms"
CFG_LINE = re.compile(r"^\s*(c\d+)\s+M=(\d+)\s+N=(\d+)\s+K=(\d+)\s+\S+\s+([0-9.]+)ms\s*$")


def _run_harness(flag, timeout=1800):
    r = subprocess.run(
        [sys.executable, HARNESS, flag], cwd=TASK_DIR,
        capture_output=True, text=True, timeout=timeout,
    )
    return r.returncode, (r.stdout or ""), (r.stderr or "")


def run_compile():
    code = (
        "import aiter; "
        "from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import gemm_a8w8_blockscale; "
        "assert callable(gemm_a8w8_blockscale)"
    )
    try:
        subprocess.run([sys.executable, "-c", code], check=True,
                       capture_output=True, text=True, timeout=300)
        return True, None
    except Exception as e:
        tail = getattr(e, "stderr", "") or str(e)
        return False, f"import kernel failed: {tail.strip()[-300:]}"


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
            return False, line.strip()[:300]
    return False, (err.strip()[-300:] or "correctness did not report PASS")


def run_performance():
    try:
        rc, out, err = _run_harness("--benchmark")
    except Exception as e:
        return [{"test_case_id": "run_fail", "execution_time_ms": -1.0,
                 "params": {"error": str(e)[:300]}}]
    sys.stdout.write(out)
    cases = []
    for line in out.splitlines():
        m = CFG_LINE.match(line)
        if m:
            cid, M, N, K, ms = m.groups()
            cases.append({
                "test_case_id": cid,
                "execution_time_ms": float(ms),
                "params": {"M": int(M), "N": int(N), "K": int(K)},
            })
    if not cases:
        cases.append({"test_case_id": "no_result", "execution_time_ms": -1.0,
                      "params": {"stderr": err.strip()[-300:]}})
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
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)
    elif args.mode == "correctness":
        ok, err = run_correctness()
        json.dump({"status": "ok" if ok else "fail", "error": err},
                  open(os.path.join(BUILD_DIR, "correctness_report.json"), "w"), indent=2)
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)
    elif args.mode == "performance":
        cases = run_performance()
        report = {"test_cases": [
            {"test_case_id": c["test_case_id"],
             "execution_time_ms": c["execution_time_ms"],
             "params": c["params"]}
            for c in cases
        ]}
        json.dump(report, open(os.path.join(BUILD_DIR, "performance_report.json"), "w"), indent=2)
        for c in cases:
            print(f"Performance: {c['execution_time_ms']} ms ({c['test_case_id']})")
        ok = [c for c in cases if c["execution_time_ms"] > 0]
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
