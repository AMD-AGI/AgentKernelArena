#!/usr/bin/env python3
"""Task runner for hip2hip/aiter/level1/fused_split_gdr_update."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

TASK_NAME = "hip2hip/aiter/level1/fused_split_gdr_update"
MODULE_NAME = "module_fused_split_gdr_update"
SOURCE_FILE = "fused_split_gdr_update.cu"

import aiter as _aiter_pkg
_aiter_pkg_dir = Path(_aiter_pkg.__file__).resolve().parent
AITER_JIT_DIR = _aiter_pkg_dir / "jit"

import aiter_meta as _aiter_meta_pkg
_aiter_meta_dir = Path(_aiter_meta_pkg.__file__).resolve().parent
KERNEL_INSTALL_DIR = _aiter_meta_dir / "csrc" / "kernels"

_ws_aiter = Path("/workspace/aiter-src")
TEST_SCRIPT = _ws_aiter / "op_tests" / "test_split_gdr_update.py" if (_ws_aiter / "op_tests" / "test_split_gdr_update.py").exists() else _aiter_meta_dir.parent / "op_tests" / "test_split_gdr_update.py"
TEST_CWD = _ws_aiter if _ws_aiter.exists() else _aiter_meta_dir.parent


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]

def _report_root() -> Path:
    d = _workspace_root() / "build"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _src_dir() -> Path:
    return _workspace_root() / "src"

def _run_cmd(cmd, cwd=None, timeout_s=600):
    print(f"[RUN] {' '.join(cmd)}")
    sys.stdout.flush()
    try:
        proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True, timeout=timeout_s)
        output = proc.stdout + proc.stderr
        print(output, end="", flush=True)
        return proc.returncode == 0, output
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT after {timeout_s}s"
    except Exception as e:
        return False, str(e)

def _copy_source_to_install():
    src = _src_dir() / SOURCE_FILE
    dst = KERNEL_INSTALL_DIR / SOURCE_FILE
    shutil.copy2(str(src), str(dst))
    print(f"Copied: {src} -> {dst}")

def _delete_jit_cache():
    so_path = AITER_JIT_DIR / f"{MODULE_NAME}.so"
    build_dir = AITER_JIT_DIR / "build" / MODULE_NAME
    if so_path.exists():
        so_path.unlink()
        print(f"Deleted: {so_path}")
    if build_dir.exists():
        shutil.rmtree(build_dir)
        print(f"Deleted: {build_dir}")


def run_compile():
    _copy_source_to_install()
    _delete_jit_cache()
    ok, out = _run_cmd(["python3", str(TEST_SCRIPT)], cwd=TEST_CWD, timeout_s=600)
    report = {"status": "ok" if ok else "fail", "output": out[-2000:]}
    (_report_root() / "compile_report.json").write_text(json.dumps(report, indent=2))
    if not ok:
        print("Compilation: FAIL"); sys.exit(1)
    print("Compilation: PASS")


def run_correctness():
    ok, out = _run_cmd(["python3", str(TEST_SCRIPT)], cwd=TEST_CWD, timeout_s=600)
    passed = len(re.findall(r"all_close.*True|passed~", out, re.IGNORECASE))
    failed = len(re.findall(r"all_close.*False|failed!", out, re.IGNORECASE))
    # Also check for "True" in the table
    if passed == 0 and "True" in out:
        passed = out.count("True")
    report = {"status": "ok" if (ok and failed == 0 and passed > 0) else "fail", "passed": passed, "failed": failed}
    (_report_root() / "correctness_report.json").write_text(json.dumps(report, indent=2))
    print(f"Correctness: {passed} passed, {failed} failed")
    if failed > 0 or not ok:
        print("Correctness: FAIL"); sys.exit(1)
    print("Correctness: PASS")


def run_performance():
    ok, out = _run_cmd(["python3", str(TEST_SCRIPT)], cwd=TEST_CWD, timeout_s=600)
    if not ok:
        (_report_root() / "performance_report.json").write_text("[]")
        print("Performance: FAIL"); sys.exit(1)

    # Parse: hip_us from markdown table
    pattern = re.compile(r"hip_us\b")
    # Parse table rows for hip_us values
    row_pattern = re.compile(
        r"\|\s*(\d+)\s*\|\s*(\d+)\s*\|.*?\|\s*([\d.]+)\s*\|$", re.MULTILINE
    )
    results = []
    for m in row_pattern.finditer(out):
        batch = int(m.group(1))
        seqlen = int(m.group(2))
        hip_us = float(m.group(3))
        results.append({
            "test_case_id": f"split_gdr_update_b{batch}_s{seqlen}",
            "shape": [batch, seqlen],
            "execution_time_ms": hip_us / 1000.0,
            "metadata": {"batch_size": batch, "seqlen": seqlen,
                         "kernel": "fused_split_gdr_update_kernel", "unit_original": "us", "value_us": hip_us},
        })

    (_report_root() / "performance_report.json").write_text(json.dumps(results, indent=2))
    (_report_root() / "performance.log").write_text(out)

    if results:
        avg_us = sum(r["metadata"]["value_us"] for r in results) / len(results)
        print(f"Performance: {len(results)} test cases, avg {avg_us:.2f} us")
    else:
        print("Performance: no results parsed"); sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description=f"Task runner for {TASK_NAME}")
    parser.add_argument("mode", choices=["compile", "correctness", "performance"])
    args = parser.parse_args()
    {"compile": run_compile, "correctness": run_correctness, "performance": run_performance}[args.mode]()

if __name__ == "__main__":
    main()
