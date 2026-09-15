# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Retain the constant-input host gate and additionally check nonuniform inputs.

The protected CPU product validates every output on every existing shape. The
same independent reference also checks the exact timed graph in the driver.
"""
import subprocess


def self_test(harness):
    if not harness.TEST_SHAPES or any(any(type(x) is not int or x <= 0 for x in shape)
                                      for shape in harness.TEST_SHAPES):
        raise ValueError("Native host workload must contain positive integer dimensions")


def check_additional_paths(harness):
    for rows, inner, cols in harness.TEST_SHAPES:
        command = [harness.BENCH_BINARY, "--A_rows", str(rows), "--A_cols", str(inner),
                   "--B_cols", str(cols), "--check-only", "1"]
        completed = subprocess.run(command, capture_output=True, text=True, timeout=300)
        if completed.returncode or "Full reference validation passed" not in completed.stdout:
            raise ValueError(f"Full nonuniform product check failed for {(rows, inner, cols)}: "
                             f"{completed.stdout}\n{completed.stderr}")
