#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""
Task runner for repository/rocprim/device_merge_sort.

This script provides a stable interface for AgentKernelArena's evaluator:
  - `compile`      : configure & build rocPRIM benchmark/test targets
  - `correctness`  : run `test_device_merge_sort`
  - `performance`  : run `benchmark_device_merge_sort` and emit `build/performance_report.json`
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

TASK_NAME = "repository/rocprim/device_merge_sort"
BENCH_TARGET = "benchmark_device_merge_sort"
TEST_TARGET = "test_device_merge_sort"
REPO_SUBDIR = "rocPRIM"

# Path helpers
def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]

def _source_root(ws: Path) -> Path:
    return ws / REPO_SUBDIR

def _build_dir(ws: Path) -> Path:
    return ws / REPO_SUBDIR / "build"

def _report_root(ws: Path) -> Path:
    return ws / "build"

def _test_bin(ws: Path) -> Path:
    return _build_dir(ws) / "test" / "rocprim" / TEST_TARGET

def _bench_bin(ws: Path) -> Path:
    return _build_dir(ws) / "benchmark" / BENCH_TARGET

def _get_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("ROCM_PATH", "/opt/rocm")
    env.setdefault("CXX", "hipcc")
    return env

def _detect_arch() -> Optional[str]:
    arch = os.environ.get("AMDGPU_TARGETS") or os.environ.get("PYTORCH_ROCM_ARCH")
    return arch.strip() if arch else None

def _print_phase(name: str, end: bool = False, status: str = ""):
    if end:
        print("=" * 60)
        if status:
            print(f"{name}: {status}")
    else:
        print("\n" + "=" * 60)
        print(name)
        print("=" * 60)


def _run(cmd: list[str], cwd: Path, timeout_s: int) -> Tuple[bool, str]:
    """Run command with real-time output streaming."""
    print(f"[RUN] {' '.join(cmd)}")
    print(f"[CWD] {cwd}")
    sys.stdout.flush()

    try:
        proc = subprocess.Popen(
            cmd, cwd=str(cwd), env=_get_env(),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        output_lines = []
        start_time = time.time()
        
        try:
            while True:
                if proc.poll() is not None:
                    remaining = proc.stdout.read()
                    if remaining:
                        print(remaining, end="", flush=True)
                        output_lines.append(remaining)
                    break
                
                if time.time() - start_time > timeout_s:
                    proc.kill()
                    proc.wait()
                    return False, f"TIMEOUT after {timeout_s}s\n{''.join(output_lines)}"
                
                line = proc.stdout.readline()
                if line:
                    print(line, end="", flush=True)
                    output_lines.append(line)
        finally:
            proc.stdout.close()
        
        return proc.returncode == 0, "".join(output_lines)
    except Exception as e:
        return False, str(e)


def _clean_stale_cmake_cache(source_dir: Path, build_dir: Path) -> None:
    """Remove stale CMake caches if generated for a different source directory."""
    
    def _check_cache_file(cache_file: Path) -> bool:
        """Check if cache file is stale. Returns True if stale."""
        if not cache_file.is_file():
            return False
        try:
            for line in cache_file.read_text(errors="ignore").splitlines():
                if line.startswith("CMAKE_HOME_DIRECTORY:"):
                    cached = line.split("=", 1)[1].strip() if "=" in line else ""
                    if cached and Path(cached).resolve() != source_dir.resolve():
                        return True
                    break
        except Exception:
            pass
        return False
    
    try:
        is_stale = False
        
        # Check top-level cache
        if _check_cache_file(build_dir / "CMakeCache.txt"):
            is_stale = True
        
        # Also check _deps subdirectories for stale caches (e.g., googletest-subbuild)
        deps_dir = build_dir / "_deps"
        if deps_dir.is_dir():
            for subdir in deps_dir.iterdir():
                if subdir.is_dir() and _check_cache_file(subdir / "CMakeCache.txt"):
                    is_stale = True
                    break
        
        if is_stale:
            print(f"Stale CMake cache detected, cleaning build directory...")
            for item in ["CMakeCache.txt", "CMakeFiles", "_deps"]:
                path = build_dir / item
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
            print(f"Cleaned: CMakeCache.txt, CMakeFiles, _deps")
    except Exception as e:
        print(f"Warning: Failed to check CMake cache: {e}")


def _cmake_configure(source_dir: Path, build_dir: Path) -> Tuple[bool, Optional[str]]:
    _print_phase("CMAKE CONFIGURE")
    build_dir.mkdir(parents=True, exist_ok=True)
    _clean_stale_cmake_cache(source_dir, build_dir)

    cmake_args = [
        "cmake", "-S", str(source_dir), "-B", str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
        "-DBUILD_BENCHMARK=ON", "-DBUILD_TEST=ON",
    ]
    if arch := _detect_arch():
        cmake_args.append(f"-DAMDGPU_TARGETS={arch}")

    ok, out = _run(cmake_args, cwd=source_dir, timeout_s=3600)
    _print_phase("CMAKE CONFIGURE", end=True, status="SUCCESS" if ok else "FAILED")
    return (True, None) if ok else (False, f"CMake configure failed.\n{out}")


def _cmake_build(source_dir: Path, build_dir: Path, target: str) -> Tuple[bool, Optional[str]]:
    _print_phase(f"CMAKE BUILD: {target}")
    cmd = ["cmake", "--build", str(build_dir), "--target", target, "-j"]
    ok, out = _run(cmd, cwd=source_dir, timeout_s=3600)
    _print_phase(f"CMAKE BUILD {target}", end=True, status="SUCCESS" if ok else "FAILED")
    return (True, None) if ok else (False, f"Build failed for '{target}'.\n{out}")


def _parse_benchmark_results(output: str) -> list[dict]:
    pattern = re.compile(
        r"^(?P<name>.+?)/manual_time\s+[\d\.]+\s*(?:ns|us|ms|s)\s+"
        r"[\d\.]+\s*(?:ns|us|ms|s)\s+\d+\s+"
        r"bytes_per_second=(?P<bps>[\d\.]+)(?P<unit>[GT])/s",
        re.MULTILINE,
    )
    results = []
    for m in pattern.finditer(output):
        bps = float(m.group("bps"))
        if m.group("unit") == "T":
            bps *= 1024.0
        results.append({"test_case_id": m.group("name").strip(), "bytes_per_second_gs": bps})
    return results


def run_compile(ws: Path) -> Tuple[bool, Optional[str]]:
    source_dir, build_dir = _source_root(ws), _build_dir(ws)
    if not source_dir.is_dir():
        return False, f"Source directory not found: {source_dir}"

    ok, err = _cmake_configure(source_dir, build_dir)
    if not ok:
        return False, err

    for target in [TEST_TARGET, BENCH_TARGET]:
        ok, err = _cmake_build(source_dir, build_dir, target)
        if not ok:
            return False, err

    for name, path in [("Test", _test_bin(ws)), ("Benchmark", _bench_bin(ws))]:
        if not path.is_file():
            return False, f"{name} binary not found: {path}"
    return True, None


def run_correctness(ws: Path) -> Tuple[bool, Optional[str]]:
    test_bin = _test_bin(ws)
    if not test_bin.is_file():
        ok, err = _cmake_configure(_source_root(ws), _build_dir(ws))
        if not ok:
            return False, err
        ok, err = _cmake_build(_source_root(ws), _build_dir(ws), TEST_TARGET)
        if not ok:
            return False, err

    _print_phase("CORRECTNESS TEST")
    ok, out = _run([str(test_bin)], cwd=ws, timeout_s=3600)
    _print_phase("CORRECTNESS TEST", end=True, status="PASSED" if ok else "FAILED")
    return (True, None) if ok else (False, f"Correctness test failed.\n{out}")


def run_performance(ws: Path, trials: int) -> Tuple[list[dict], str]:
    bench_bin = _bench_bin(ws)
    if not bench_bin.is_file():
        ok, err = _cmake_configure(_source_root(ws), _build_dir(ws))
        if not ok:
            return [], err
        ok, err = _cmake_build(_source_root(ws), _build_dir(ws), BENCH_TARGET)
        if not ok:
            return [], err

    _print_phase(f"PERFORMANCE BENCHMARK (trials={trials})")
    ok, out = _run([str(bench_bin), "--trials", str(trials)], cwd=ws, timeout_s=3600)
    _print_phase("PERFORMANCE BENCHMARK", end=True, status="SUCCESS" if ok else "FAILED")

    report_root = _report_root(ws)
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / f"{BENCH_TARGET}.log").write_text(out)

    if not ok:
        return [], f"Benchmark failed.\n{out}"
    
    results = _parse_benchmark_results(out)
    return (results, "") if results else ([], f"Failed to parse results.\n{out}")


def main() -> None:
    ws = _workspace_root()
    os.chdir(ws)
    report_root = _report_root(ws)
    report_root.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser(description=f"Task runner for {TASK_NAME}")
    parser.add_argument("mode", choices=["compile", "correctness", "performance"])
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()

    if args.mode == "compile":
        ok, err = run_compile(ws)
        report = {"status": "ok" if ok else "fail", "error": err,
                  "arch": _detect_arch(), "source_dir": str(_source_root(ws)),
                  "build_dir": str(_build_dir(ws))}
        (report_root / "compile_report.json").write_text(json.dumps(report, indent=2))
        print(f"Compilation: {'PASS' if ok else 'FAIL'}")
        if err:
            print(err)
        sys.exit(0 if ok else 1)

    elif args.mode == "correctness":
        ok, err = run_correctness(ws)
        report = {"status": "ok" if ok else "fail", "error": err}
        (report_root / "correctness_report.json").write_text(json.dumps(report, indent=2))
        print(f"Correctness: {'PASS' if ok else 'FAIL'}")
        if err:
            print(err)
        sys.exit(0 if ok else 1)

    elif args.mode == "performance":
        results, err = run_performance(ws, trials=args.trials)
        (report_root / "performance_report.json").write_text(json.dumps(results or [], indent=2))
        if results:
            avg = sum(r["bytes_per_second_gs"] for r in results) / len(results)
            print(f"Performance: {len(results)} test cases, avg {avg:.4f} G/s")
            for r in results:
                print(f"  {r['test_case_id']}: {r['bytes_per_second_gs']:.4f} G/s")
        else:
            print("Performance: FAILED")
            if err:
                print(err)
        sys.exit(0 if results else 1)


if __name__ == "__main__":
    main()
