#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""
Task runner for repository/rocprim/block_histogram.

This script provides a stable interface for AgentKernelArena's evaluator:
  - `compile`      : configure & build rocPRIM benchmark/test targets
  - `correctness`  : run `test_block_histogram`
  - `performance`  : run `benchmark_block_histogram` and emit `build/performance_report.json`

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


TASK_NAME = "repository/rocprim/block_histogram"
BENCH_TARGET = "benchmark_block_histogram"
TEST_TARGET = "test_block_histogram"
REPO_SUBDIR = "rocPRIM"  # Cloned repo lives under workspace/rocPRIM/


def _workspace_root() -> Path:
    # scripts/task_runner.py -> scripts/ -> workspace root
    return Path(__file__).resolve().parents[1]


def _source_root(workspace: Path) -> Path:
    """CMake source directory (cloned rocPRIM repo)."""
    return workspace / REPO_SUBDIR


def _cmake_build_root(workspace: Path) -> Path:
    """CMake build root inside the cloned repo (workspace/rocPRIM/build/)."""
    return _source_root(workspace) / "build"


def _cmake_build_dir(workspace: Path) -> Path:
    """CMake build directory (workspace/rocPRIM/build/Release/)."""
    return _cmake_build_root(workspace) / "Release"


def _report_root(workspace: Path) -> Path:
    """Report directory for evaluator (workspace/build/). Separate from CMake build."""
    return workspace / "build"


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




def _parse_time_ms(output: str) -> Optional[float]:
    # Try to find a reasonable "average/mean" latency number in common units.
    patterns = [
        r"avg(?:erage)?(?:\s+time)?\s*[:=]\s*([\d.]+)\s*(ns|us|ms|s)\b",
        r"mean(?:\s+time)?\s*[:=]\s*([\d.]+)\s*(ns|us|ms|s)\b",
        r"median(?:\s+time)?\s*[:=]\s*([\d.]+)\s*(ns|us|ms|s)\b",
        r"Perf(?:ormance)?\s*[:=]\s*([\d.]+)\s*(ns|us|ms|s)\b",
        r"([\d.]+)\s*(ns|us|ms|s)\s*/\s*(?:trial|iter(?:ation)?|launch)\b",
    ]
    unit_mul = {"ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1000.0}
    for pat in patterns:
        m = re.search(pat, output, re.IGNORECASE)
        if not m:
            continue
        val = float(m.group(1))
        unit = m.group(2).lower()
        if unit in unit_mul:
            return val * unit_mul[unit]
    return None


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


def run_performance(workspace: Path, trials: int) -> Tuple[float, str]:
    build_dir = _cmake_build_dir(workspace)
    source_dir = _source_root(workspace)

    bench_bin = _bench_binary_path(build_dir)
    ok, err = _maybe_build_target(source_dir, build_dir, BENCH_TARGET, bench_bin)
    if not ok:
        return -1.0, err or "build failed"
    if not bench_bin.is_file():
        return -1.0, f"Benchmark binary not found after build attempt: {bench_bin}"

    env = os.environ.copy()
    cmd = [str(bench_bin), "--trials", str(trials)]
    t0 = time.perf_counter()
    ok, out = _run(cmd, cwd=workspace, timeout_s=3600, env=env)
    elapsed_ms_total = (time.perf_counter() - t0) * 1000.0

    if not ok:
        return -1.0, f"Benchmark failed.\nCommand: {' '.join(cmd)}\nOutput:\n{out}"

    parsed_ms = _parse_time_ms(out)
    if parsed_ms is not None and parsed_ms > 0:
        return float(parsed_ms), ""

    # Fallback: approximate average per trial from wall-clock runtime.
    if trials > 0:
        return float(elapsed_ms_total / trials), ""
    return float(elapsed_ms_total), ""


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
        exec_ms, err = run_performance(workspace, trials=args.trials)
        report = [
            {
                "test_case_id": "test_case_0",
                "execution_time_ms": exec_ms,
                "params": {"trials": args.trials},
            }
        ]
        (report_root / "performance_report.json").write_text(json.dumps(report, indent=2))
        # Also print a recognizable line for stdout parsing fallback.
        print(f"Performance: {exec_ms:.4f} ms")
        if err:
            print(err)
        sys.exit(0 if exec_ms != -1.0 else 1)


if __name__ == "__main__":
    main()