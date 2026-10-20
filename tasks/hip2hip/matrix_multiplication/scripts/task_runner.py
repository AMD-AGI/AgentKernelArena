#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Task runner for hip2hip/matrix_multiplication"""
import sys
import os
import json
import argparse
import subprocess

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "hip2hip/matrix_multiplication"
BINARY = os.path.join(TASK_DIR, "hip_matrix_multiplication")
BENCH_BINARY = os.path.join(TASK_DIR, "build", "native_graph_benchmark")
BENCH_SOURCE = os.path.join(TASK_DIR, "scripts", "native", "benchmark_driver.hip")

# 5 test shapes: (A_rows, A_cols, B_cols) - must be multiples of 16 (block_size)
TEST_SHAPES = [
    (256, 256, 256),
    (512, 256, 512),
    (1024, 512, 1024),
    (2048, 1024, 1024),
    (1024, 1024, 2048),
]


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
        os.makedirs(os.path.dirname(BENCH_BINARY), exist_ok=True)
        result = subprocess.run(
            [
                os.environ.get("HIPCXX", "/opt/rocm/bin/hipcc"),
                "-std=c++17", "-Wall", "-Wextra",
                "-I", os.path.join(TASK_DIR, "Common"),
                BENCH_SOURCE, "-o", BENCH_BINARY,
            ],
            cwd=TASK_DIR, capture_output=True, text=True, timeout=300,
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
    for shape_idx, (rows, cols, bcols) in enumerate(TEST_SHAPES):
        try:
            completed = subprocess.run(
                [
                    BENCH_BINARY,
                    "--A_rows", str(rows),
                    "--A_cols", str(cols),
                    "--B_cols", str(bcols),
                    "--samples", "100",
                ],
                capture_output=True, text=True, timeout=300,
            )
            output = completed.stdout + completed.stderr
            if completed.returncode != 0:
                return [], f"Shape {shape_idx} native benchmark failed:\n{output}"
            result = _parse_native_result(output)
            result["test_case_id"] = f"shape_{shape_idx}"
            result["params"] = {
                "A_rows": rows,
                "A_cols": cols,
                "B_cols": bcols,
            }
            test_cases.append(result)
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
        test_cases, err = run_performance()
        report = {"test_cases": test_cases, "error": err}
        with open(os.path.join(build_dir, "performance_report.json"), "w") as f:
            json.dump(report, f, indent=2)
        for case in test_cases:
            print(f"Performance: {case['execution_time_ms']:.4f} ms ({case['test_case_id']})")
        if err:
            print(f"Performance: FAIL\nError: {err}")
        sys.exit(0 if test_cases and not err else 1)


if __name__ == "__main__":
    main()
