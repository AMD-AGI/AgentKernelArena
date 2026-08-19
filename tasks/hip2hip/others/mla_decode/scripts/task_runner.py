#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""Task runner for hip2hip/mla_decode."""
import argparse
import json
import os
import subprocess
import sys

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "hip2hip/mla_decode"
BINARY = os.path.join(TASK_DIR, "applications_mla_decode")
BENCH_BINARY = os.path.join(TASK_DIR, "build", "native_graph_benchmark")
BENCH_SOURCE = os.path.join(TASK_DIR, "scripts", "native", "benchmark_driver.hip")

# 5 representative shapes covering the decode regime. The kernel is
# hardcoded to NHEAD=128 / LK=576 / LV=512, so the only free axes are
# batch and ctx. We deliberately keep batch * ctx bounded so the
# correctness check (an OpenMP fp32 host reference) and the full perf
# sweep are tractable on the naive baseline; the optimization headroom
# remains 100x+.
TEST_SHAPES = [
    (1,    512),
    (4,   1024),
    (16,  2048),
    (1,   4096),
    (1,   8192),
]


def run_compile():
    try:
        subprocess.run(
            ["make", "-C", TASK_DIR, "clean"],
            capture_output=True, text=True, timeout=30,
        )
        result = subprocess.run(
            ["make", "-C", TASK_DIR],
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            return False, f"make failed:\n{result.stderr}\n{result.stdout}"
        if not os.path.isfile(BINARY):
            return False, f"Binary {BINARY} not found after make"
        os.makedirs(os.path.dirname(BENCH_BINARY), exist_ok=True)
        result = subprocess.run(
            [
                os.environ.get("HIPCXX", "hipcc"),
                "-O3", "-ffast-math",
                "--offload-arch=gfx950", "--offload-arch=gfx942",
                "-munsafe-fp-atomics", "-std=c++17",
                BENCH_SOURCE, "-o", BENCH_BINARY,
            ],
            cwd=TASK_DIR, capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            return False, f"native benchmark compile failed:\n{result.stderr}\n{result.stdout}"
        if not os.path.isfile(BENCH_BINARY):
            return False, f"Native benchmark {BENCH_BINARY} not found after compile"
        return True, None
    except Exception as e:
        return False, str(e)


def run_correctness():
    if not os.path.isfile(BINARY):
        return False, "Binary not found. Run compile first."

    for i, (batch, ctx) in enumerate(TEST_SHAPES):
        try:
            result = subprocess.run(
                [BINARY, "--batch", str(batch), "--ctx", str(ctx), "--mode", "check"],
                capture_output=True, text=True, timeout=900,
            )
            output = result.stdout + result.stderr
            if "FAIL" in output:
                return False, f"Shape {i+1} (batch={batch}, ctx={ctx}): FAIL\n{output}"
            if "PASS" not in output:
                return False, f"Shape {i+1} (batch={batch}, ctx={ctx}): no PASS/FAIL in output\n{output}"
            if result.returncode != 0:
                return False, f"Shape {i+1} (batch={batch}, ctx={ctx}): non-zero exit code {result.returncode}"
        except subprocess.TimeoutExpired:
            return False, f"Shape {i+1} (batch={batch}, ctx={ctx}): timeout"
        except Exception as e:
            return False, f"Shape {i+1} (batch={batch}, ctx={ctx}): {e}"

    return True, None


def _parse_native_result(output):
    prefix = "AKA_BENCHMARK_RESULT "
    for line in output.splitlines():
        if not line.startswith(prefix):
            continue
        result = json.loads(line[len(prefix):])
        if result.get("benchmark_method") not in {"cuda_graph", "cuda_event_fallback"}:
            raise ValueError("native benchmark returned an invalid benchmark_method")
        elapsed = float(result["execution_time_ms"])
        if elapsed <= 0:
            raise ValueError("native benchmark returned a non-positive execution time")
        return result
    raise ValueError("native benchmark did not emit AKA_BENCHMARK_RESULT")


def run_performance():
    if not os.path.isfile(BENCH_BINARY):
        return [], "Native benchmark binary not found. Run compile first."

    test_cases = []
    for shape_idx, (batch, ctx) in enumerate(TEST_SHAPES):
        try:
            result = subprocess.run(
                [
                    BENCH_BINARY,
                    "--batch", str(batch),
                    "--ctx", str(ctx),
                    "--samples", "100",
                ],
                capture_output=True, text=True, timeout=300,
            )
            output = result.stdout + result.stderr
            if result.returncode != 0:
                return [], f"Shape {shape_idx} native benchmark failed:\n{output}"
            parsed = _parse_native_result(output)
            parsed["test_case_id"] = f"shape_{shape_idx}"
            parsed["params"] = {"batch": batch, "ctx": ctx}
            test_cases.append(parsed)
        except Exception as error:
            return [], f"Shape {shape_idx} native benchmark failed: {error}"

    return test_cases, None


def main():
    parser = argparse.ArgumentParser(description=f"Task runner for {TASK_NAME}")
    parser.add_argument("mode", choices=["compile", "correctness", "performance"])
    args = parser.parse_args()

    build_dir = os.path.join(TASK_DIR, "build")
    os.makedirs(build_dir, exist_ok=True)

    if args.mode == "compile":
        ok, err = run_compile()
        with open(os.path.join(build_dir, "compile_report.json"), "w") as f:
            json.dump({"status": "ok" if ok else "fail", "error": err}, f, indent=2)
        print(f"Compilation: {'PASS' if ok else 'FAIL'}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)

    elif args.mode == "correctness":
        ok, err = run_correctness()
        with open(os.path.join(build_dir, "correctness_report.json"), "w") as f:
            json.dump({"status": "ok" if ok else "fail", "error": err,
                       "num_shapes": len(TEST_SHAPES)}, f, indent=2)
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)

    elif args.mode == "performance":
        test_cases, err = run_performance()
        with open(os.path.join(build_dir, "performance_report.json"), "w") as f:
            json.dump({"test_cases": test_cases, "error": err}, f, indent=2)
        for case in test_cases:
            print(f"Performance: {case['execution_time_ms']:.4f} ms ({case['test_case_id']})")
        if err:
            print(f"Performance: FAIL\nError: {err}")
        sys.exit(0 if test_cases and not err else 1)


if __name__ == "__main__":
    main()
