#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Task runner for hip2hip/matrix_multiplication"""
import sys
import os
import json
import argparse
import subprocess
import time

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "hip2hip/matrix_multiplication"
BINARY = os.path.join(TASK_DIR, "hip_matrix_multiplication")

# 5 test shapes: (A_rows, A_cols, B_cols) - must be multiples of 16 (block_size)
TEST_SHAPES = [
    (256, 256, 256),
    (512, 256, 512),
    (1024, 512, 1024),
    (2048, 1024, 1024),
    (1024, 1024, 2048),
]
PERF_SHAPE_IDX = 3


def run_compile():
    try:
        result = subprocess.run(
            ["make", "-C", TASK_DIR, "clean"],
            capture_output=True, text=True, timeout=30)
        result = subprocess.run(
            ["make", "-C", TASK_DIR],
            capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            return False, f"make failed:\n{result.stderr}\n{result.stdout}"
        if not os.path.isfile(BINARY):
            return False, f"Binary {BINARY} not found after make"
        return True, None
    except Exception as e:
        return False, str(e)


def run_correctness():
    if not os.path.isfile(BINARY):
        return False, "Binary not found. Run compile first."

    for i, (rows, cols, bcols) in enumerate(TEST_SHAPES):
        try:
            result = subprocess.run(
                [BINARY, "--A_rows", str(rows), "--A_cols", str(cols), "--B_cols", str(bcols)],
                capture_output=True, text=True, timeout=60)
            output = result.stdout + result.stderr
            if "Validation failed" in output or result.returncode != 0:
                return False, f"Shape {i+1} ({rows}x{cols}x{bcols}): validation failed\n{output}"
            if "Validation passed" not in output:
                return False, f"Shape {i+1} ({rows}x{cols}x{bcols}): no validation result\n{output}"
        except subprocess.TimeoutExpired:
            return False, f"Shape {i+1} ({rows}x{cols}x{bcols}): timeout"
        except Exception as e:
            return False, f"Shape {i+1} ({rows}x{cols}x{bcols}): {str(e)}"

    return True, None


def run_performance():
    if not os.path.isfile(BINARY):
        return -1.0

    rows, cols, bcols = TEST_SHAPES[PERF_SHAPE_IDX]
    # Time multiple runs
    n_warmup = 3
    n_iter = 10
    for _ in range(n_warmup):
        subprocess.run(
            [BINARY, "--A_rows", str(rows), "--A_cols", str(cols), "--B_cols", str(bcols)],
            capture_output=True, timeout=60)

    total_ms = 0
    for _ in range(n_iter):
        t0 = time.perf_counter()
        subprocess.run(
            [BINARY, "--A_rows", str(rows), "--A_cols", str(cols), "--B_cols", str(bcols)],
            capture_output=True, timeout=60)
        t1 = time.perf_counter()
        total_ms += (t1 - t0) * 1000

    return total_ms / n_iter


def main():
    parser = argparse.ArgumentParser(description=f"Task runner for {TASK_NAME}")
    parser.add_argument("mode", choices=["compile", "correctness", "performance"])
    args = parser.parse_args()

    build_dir = os.path.join(TASK_DIR, "build")
    os.makedirs(build_dir, exist_ok=True)

    if args.mode == "compile":
        ok, err = run_compile()
        report = {"status": "ok" if ok else "fail", "error": err}
        with open(os.path.join(build_dir, "compile_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Compilation: {'PASS' if ok else 'FAIL'}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)

    elif args.mode == "correctness":
        ok, err = run_correctness()
        report = {"status": "ok" if ok else "fail", "error": err, "num_shapes": len(TEST_SHAPES)}
        with open(os.path.join(build_dir, "correctness_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if err:
            print(f"Error: {err}")
        sys.exit(0 if ok else 1)

    elif args.mode == "performance":
        elapsed_ms = run_performance()
        report = {"execution_time_ms": elapsed_ms, "shape": list(TEST_SHAPES[PERF_SHAPE_IDX])}
        with open(os.path.join(build_dir, "performance_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        print(f"Performance: {elapsed_ms:.4f} ms")
        sys.exit(0)


if __name__ == "__main__":
    main()
