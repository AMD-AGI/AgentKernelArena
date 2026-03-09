#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""
Task runner for repository/rocprim/device_search_n.

This script provides a stable interface for AgentKernelArena's evaluator:
  - `compile`      : configure & build rocPRIM benchmark/test targets
  - `correctness`  : run `test_device_search_n`
  - `performance`  : run `benchmark_device_search_n` and emit `build/performance_report.json`

All reports are written under `<workspace>/build/` so the centralized evaluator can parse them.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple


TASK_NAME = "repository/rocprim/device_search_n"
BENCH_TARGET = "benchmark_device_search_n"
TEST_TARGET = "test_device_search_n"
REPO_SUBDIR = "rocPRIM"  # Cloned repo lives under workspace/rocPRIM/


def _workspace_root() -> Path:
    # scripts/task_runner.py -> scripts/ -> workspace root
    return Path(__file__).resolve().parents[1]


def _source_root(workspace: Path) -> Path:
    """CMake source directory (cloned rocPRIM repo)."""
    return workspace / REPO_SUBDIR


def _cmake_build_root(workspace: Path) -> Path:
    """CMake build root (workspace/rocPRIM/)."""
    return workspace / REPO_SUBDIR


def _cmake_build_dir(workspace: Path) -> Path:
    return _cmake_build_root(workspace) / "build"


def _report_root(workspace: Path) -> Path:
    """Report directory for evaluator (workspace/build_reports/)."""
    return workspace / "build_reports"


def _detect_arch() -> Optional[str]:
    # Main framework sets PYTORCH_ROCM_ARCH from target_gpu_model; reuse it for rocPRIM CMake.
    arch = os.environ.get("AMDGPU_TARGETS") or os.environ.get("PYTORCH_ROCM_ARCH")
    if not arch:
        return None
    return arch.strip() or None


def _run(cmd: list[str], cwd: Path, timeout_s: int, env: dict[str, str]) -> Tuple[bool, str]:
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode == 0, out
    except subprocess.TimeoutExpired as e:
        out = (getattr(e, "stdout", "") or "") + (getattr(e, "stderr", "") or "")
        return False, f"TIMEOUT after {timeout_s}s\n{out}"
    except Exception as e:
        return False, str(e)


def _ensure_configured(source_dir: Path, build_dir: Path) -> Tuple[bool, Optional[str]]:
    """
    Run CMake configure.

    Args:
        source_dir: CMake source directory (cloned repo, e.g. workspace/rocPRIM/)
        build_dir: CMake build directory (e.g. workspace/build/Release/)
    """
    build_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("ROCM_PATH", "/opt/rocm")
    env.setdefault("CXX", "hipcc")

    cmake_args = [
        "cmake",
        "-S",
        str(source_dir),
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
        "-DBUILD_BENCHMARK=ON",
        "-DBUILD_TEST=ON",
    ]

    arch = _detect_arch()
    if arch:
        cmake_args.append(f"-DAMDGPU_TARGETS={arch}")

    ok, out = _run(cmake_args, cwd=source_dir, timeout_s=600, env=env)
    if not ok:
        return False, f"CMake configure failed.\nCommand: {' '.join(cmake_args)}\nOutput:\n{out}"
    return True, None


def _cmake_build(source_dir: Path, build_dir: Path, target: str) -> Tuple[bool, Optional[str]]:
    """
    Run CMake build.

    Args:
        source_dir: CMake source directory (for cwd)
        build_dir: CMake build directory
        target: Build target name
    """
    env = os.environ.copy()
    env.setdefault("ROCM_PATH", "/opt/rocm")
    env.setdefault("CXX", "hipcc")

    cmd = ["cmake", "--build", str(build_dir), "--target", target, "-j"]
    ok, out = _run(cmd, cwd=source_dir, timeout_s=1800, env=env)
    if not ok:
        return False, f"Build failed for target '{target}'.\nCommand: {' '.join(cmd)}\nOutput:\n{out}"
    return True, None


def _maybe_build_target(
    source_dir: Path,
    build_dir: Path,
    target: str,
    binary_path: Path,
) -> Tuple[bool, Optional[str]]:
    """
    Avoid redundant builds when the binary already exists.

    Arena runs compile -> correctness -> performance sequentially, so correctness/perf
    should not rebuild unless the required binary is missing.
    """
    if binary_path.is_file():
        return True, None

    ok, err = _ensure_configured(source_dir, build_dir)
    if not ok:
        return False, err
    return _cmake_build(source_dir, build_dir, target)


def _test_binary_path(build_dir: Path) -> Path:
    return build_dir / "test" / "rocprim" / TEST_TARGET


def _bench_binary_path(build_dir: Path) -> Path:
    return build_dir / "benchmark" / BENCH_TARGET




def _parse_benchmark_results(output: str) -> list[dict]:
    """
    Parse rocPRIM benchmark output for all test cases.
    Returns list of dicts with test_case_id and bytes_per_second_gs for each test case.
    """
    pattern = re.compile(
        r"^(?P<name>.+?)/manual_time\s+"
        r"(?P<time>[\d\.]+)\s*(?P<time_unit>ns|us|ms|s)\s+"
        r"(?P<cpu>[\d\.]+)\s*(?P<cpu_unit>ns|us|ms|s)\s+"
        r"(?P<iterations>\d+)\s+"
        r"bytes_per_second=(?P<bps>[\d\.]+)(?P<bps_unit>[GT])/s",
        re.MULTILINE,
    )

    results = []
    for m in pattern.finditer(output):
        name = m.group("name").strip()
        bps = float(m.group("bps"))
        bps_unit = m.group("bps_unit")
        # Convert T/s to G/s
        if bps_unit == "T":
            bps *= 1024.0
        results.append({
            "test_case_id": name,
            "bytes_per_second_gs": bps,
        })

    return results


def run_compile(workspace: Path) -> Tuple[bool, Optional[str]]:
    build_dir = _cmake_build_dir(workspace)
    source_dir = _source_root(workspace)

    if not source_dir.is_dir():
        return False, f"Source directory not found: {source_dir}. Repo may not have been cloned."

    ok, err = _ensure_configured(source_dir, build_dir)
    if not ok:
        return False, err

    # Build both correctness and benchmark targets during compile phase.
    ok, err = _cmake_build(source_dir, build_dir, TEST_TARGET)
    if not ok:
        return False, err
    ok, err = _cmake_build(source_dir, build_dir, BENCH_TARGET)
    if not ok:
        return False, err

    # Sanity-check binaries exist.
    test_bin = _test_binary_path(build_dir)
    bench_bin = _bench_binary_path(build_dir)
    if not test_bin.is_file():
        return False, f"Test binary not found: {test_bin}"
    if not bench_bin.is_file():
        return False, f"Benchmark binary not found: {bench_bin}"

    return True, None


def run_correctness(workspace: Path) -> Tuple[bool, Optional[str]]:
    build_dir = _cmake_build_dir(workspace)
    source_dir = _source_root(workspace)

    test_bin = _test_binary_path(build_dir)
    ok, err = _maybe_build_target(source_dir, build_dir, TEST_TARGET, test_bin)
    if not ok:
        return False, err
    if not test_bin.is_file():
        return False, f"Test binary not found after build attempt: {test_bin}"

    env = os.environ.copy()
    ok, out = _run([str(test_bin)], cwd=workspace, timeout_s=1800, env=env)
    if not ok:
        return False, f"Correctness test failed.\nCommand: {test_bin}\nOutput:\n{out}"
    return True, None


def run_performance(workspace: Path, trials: int) -> Tuple[list[dict], str]:
    """
    Run benchmark and return (list_of_test_results, error_message).
    Each test result contains test_case_id and bytes_per_second_gs.
    Returns empty list on failure.
    """
    build_dir = _cmake_build_dir(workspace)
    source_dir = _source_root(workspace)
    report_root = _report_root(workspace)
    report_root.mkdir(parents=True, exist_ok=True)

    bench_bin = _bench_binary_path(build_dir)
    ok, err = _maybe_build_target(source_dir, build_dir, BENCH_TARGET, bench_bin)
    if not ok:
        return [], err or "build failed"
    if not bench_bin.is_file():
        return [], f"Benchmark binary not found after build attempt: {bench_bin}"

    env = os.environ.copy()
    cmd = [str(bench_bin), "--trials", str(trials)]
    ok, out = _run(cmd, cwd=workspace, timeout_s=3600, env=env)

    # Save benchmark log
    log_path = report_root / f"{BENCH_TARGET}.log"
    log_path.write_text(out, encoding="utf-8")
    print(f"Benchmark log saved to: {log_path}")

    if not ok:
        return [], f"Benchmark failed.\nCommand: {' '.join(cmd)}\nOutput:\n{out}"

    results = _parse_benchmark_results(out)
    if results:
        return results, ""

    return [], f"Failed to parse benchmark results from output.\nOutput:\n{out}"


def main() -> None:
    workspace = _workspace_root()
    os.chdir(workspace)
    report_root = _report_root(workspace)
    report_root.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description=f"Task runner for {TASK_NAME}")
    parser.add_argument("mode", choices=["compile", "correctness", "performance"])
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()

    if args.mode == "compile":
        ok, err = run_compile(workspace)
        report = {
            "status": "ok" if ok else "fail",
            "error": err,
            "arch": _detect_arch(),
            "source_dir": str(_source_root(workspace)),
            "build_dir": str(_cmake_build_dir(workspace)),
        }
        (report_root / "compile_report.json").write_text(json.dumps(report, indent=2))
        print(f"Compilation: {'PASS' if ok else 'FAIL'}")
        if err:
            print(err)
        sys.exit(0 if ok else 1)

    if args.mode == "correctness":
        ok, err = run_correctness(workspace)
        report = {
            "status": "ok" if ok else "fail",
            "error": err,
        }
        (report_root / "correctness_report.json").write_text(json.dumps(report, indent=2))
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if err:
            print(err)
        sys.exit(0 if ok else 1)

    if args.mode == "performance":
        results, err = run_performance(workspace, trials=args.trials)
        report = results if results else []
        (report_root / "performance_report.json").write_text(json.dumps(report, indent=2))

        if results:
            avg_bps = sum(r["bytes_per_second_gs"] for r in results) / len(results)
            print(f"Performance: {len(results)} test cases, avg {avg_bps:.4f} G/s")
            for r in results:
                print(f"  {r['test_case_id']}: {r['bytes_per_second_gs']:.4f} G/s")
        else:
            print("Performance: FAILED")
        if err:
            print(err)
        sys.exit(0 if results else 1)


if __name__ == "__main__":
    main()