#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Task runner for hip2hip/silu"""
import sys
import os
import json
import argparse
import subprocess
import re

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(TASK_DIR)

TASK_NAME = "hip2hip/silu"
BINARY = os.path.join(TASK_DIR, "applications_silu")

# 5 test shapes: (B, H)
TEST_SHAPES = [
    (256, 1024),
    (1024, 2048),
    (2048, 4096),
    (4096, 6400),
    (512, 8192),
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

    for i, (B, H) in enumerate(TEST_SHAPES):
        try:
            result = subprocess.run(
                [BINARY, "--B", str(B), "--H", str(H)],
                capture_output=True, text=True, timeout=60)
            output = result.stdout + result.stderr
            if "FAIL" in output:
                return False, f"Shape {i+1} (B={B},H={H}): FAIL\n{output}"
            if "PASS" not in output:
                return False, f"Shape {i+1} (B={B},H={H}): no PASS/FAIL in output\n{output}"
            if result.returncode != 0:
                return False, f"Shape {i+1} (B={B},H={H}): non-zero exit code {result.returncode}"
        except subprocess.TimeoutExpired:
            return False, f"Shape {i+1} (B={B},H={H}): timeout"
        except Exception as e:
            return False, f"Shape {i+1} (B={B},H={H}): {str(e)}"

    return True, None


def run_performance():
    if not os.path.isfile(BINARY):
        return -1.0

    B, H = TEST_SHAPES[PERF_SHAPE_IDX]
    try:
        result = subprocess.run(
            [BINARY, "--B", str(B), "--H", str(H)],
            capture_output=True, text=True, timeout=60)
        output = result.stdout + result.stderr
        # Parse "Perf: X.XXX us/launch"
        match = re.search(r'Perf:\s+([\d.]+)\s+us/launch', output)
        if match:
            us = float(match.group(1))
            return us / 1000.0  # convert to ms
        # Fallback: just return a measured time
        return -1.0
    except Exception:
        return -1.0


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
