#!/usr/bin/env python3
"""Task runner for repository/aiter/rmsnorm_kernels.

Adapts the aiter JIT-compiled HIP kernel workflow to the AgentKernelArena
evaluator interface (compile / correctness / performance).

Key paths (aiter pip-installed environment):
  - Kernel source : /opt/venv/lib/python3.12/site-packages/aiter_meta/csrc/kernels/rmsnorm_kernels.cu
  - JIT module    : module_rmsnorm
  - JIT .so cache : /opt/venv/lib/python3.12/site-packages/aiter/jit/module_rmsnorm.so
  - JIT build dir : /opt/venv/lib/python3.12/site-packages/aiter/jit/build/module_rmsnorm/
  - Test file     : /workspace/aiter-src/op_tests/test_rmsnorm2d.py
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

TASK_NAME = "repository/aiter/rmsnorm_kernels"
MODULE_NAME = "module_rmsnorm"

# aiter pip-installed paths
AITER_JIT_DIR = Path("/opt/venv/lib/python3.12/site-packages/aiter/jit")
AITER_SRC_DIR = Path("/workspace/aiter-src")
TEST_SCRIPT = AITER_SRC_DIR / "op_tests" / "test_rmsnorm2d.py"


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _report_root() -> Path:
    d = _workspace_root() / "build"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _run_cmd(cmd: list[str], cwd: Path | None = None, timeout_s: int = 600) -> tuple[bool, str]:
    """Run a command and return (success, combined_output)."""
    print(f"[RUN] {' '.join(cmd)}")
    sys.stdout.flush()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        output = proc.stdout + proc.stderr
        print(output, end="", flush=True)
        return proc.returncode == 0, output
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT after {timeout_s}s"
    except Exception as e:
        return False, str(e)


def _delete_jit_cache() -> None:
    """Remove JIT-compiled .so and build directory to force recompilation."""
    so_path = AITER_JIT_DIR / f"{MODULE_NAME}.so"
    build_dir = AITER_JIT_DIR / "build" / MODULE_NAME

    if so_path.exists():
        so_path.unlink()
        print(f"Deleted: {so_path}")
    if build_dir.exists():
        import shutil
        shutil.rmtree(build_dir)
        print(f"Deleted: {build_dir}")


# ── compile ──────────────────────────────────────────────────────────────────

def run_compile() -> None:
    """Force JIT recompilation by deleting cache and running a small test."""
    _delete_jit_cache()

    ok, out = _run_cmd(
        ["python3", str(TEST_SCRIPT), "--m", "4", "--n", "4096", "-d", "bf16"],
        cwd=AITER_SRC_DIR,
        timeout_s=600,
    )

    report = {"status": "ok" if ok else "fail", "output": out[-2000:]}
    (_report_root() / "compile_report.json").write_text(json.dumps(report, indent=2))

    if not ok:
        print(f"Compilation: FAIL")
        sys.exit(1)
    print(f"Compilation: PASS")


# ── correctness ──────────────────────────────────────────────────────────────

def run_correctness() -> None:
    """Run correctness tests across multiple shapes and dtypes."""
    ok, out = _run_cmd(
        ["python3", str(TEST_SCRIPT), "-d", "bf16"],
        cwd=AITER_SRC_DIR,
        timeout_s=600,
    )

    # Parse passed/failed counts
    passed = len(re.findall(r"passed~", out))
    failed = len(re.findall(r"(?:failed!|FAIL)", out))

    report = {
        "status": "ok" if (ok and failed == 0) else "fail",
        "passed": passed,
        "failed": failed,
    }
    (_report_root() / "correctness_report.json").write_text(json.dumps(report, indent=2))

    print(f"Correctness: {passed} passed, {failed} failed")
    if failed > 0 or not ok:
        print("Correctness: FAIL")
        sys.exit(1)
    print("Correctness: PASS")


# ── performance ──────────────────────────────────────────────────────────────

def run_performance() -> None:
    """Run performance benchmarks and write performance_report.json.

    Output format from test_rmsnorm2d.py:
      [perf] dim: (4, 4096)  , dtype: torch.bfloat16, torch avg: 36.57  us, ck avg: 4.57  us, cu avg: 5.46  us,uplift: 700.0%
    """
    ok, out = _run_cmd(
        ["python3", str(TEST_SCRIPT), "-d", "bf16"],
        cwd=AITER_SRC_DIR,
        timeout_s=600,
    )

    if not ok:
        (_report_root() / "performance_report.json").write_text("[]")
        print("Performance: FAIL")
        sys.exit(1)

    # Parse performance lines
    pattern = re.compile(
        r"\[perf\] dim: \((?P<m>\d+),\s*(?P<n>\d+)\)\s*,"
        r"\s*dtype: (?P<dtype>\S+),"
        r".*?cu avg:\s*(?P<cu>[\d.]+)\s*us"
    )

    results = []
    seen = set()
    for m in pattern.finditer(out):
        shape_m, shape_n = int(m.group("m")), int(m.group("n"))
        cu_us = float(m.group("cu"))
        key = (shape_m, shape_n, m.group("dtype"))
        if key in seen:
            continue
        seen.add(key)
        results.append({
            "test_case_id": f"rmsnorm_m{shape_m}_n{shape_n}_{m.group('dtype')}",
            "shape": [shape_m, shape_n],
            "execution_time_ms": cu_us / 1000.0,
            "metadata": {
                "m": shape_m,
                "n": shape_n,
                "dtype": m.group("dtype"),
                "kernel": "cu",
                "unit_original": "us",
                "value_us": cu_us,
            },
        })

    (_report_root() / "performance_report.json").write_text(json.dumps(results, indent=2))

    if results:
        avg_us = sum(r["metadata"]["value_us"] for r in results) / len(results)
        print(f"Performance: {len(results)} test cases, avg {avg_us:.2f} us")
        for r in results:
            print(f"  {r['test_case_id']}: {r['metadata']['value_us']:.2f} us")
    else:
        print("Performance: no results parsed")
        sys.exit(1)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=f"Task runner for {TASK_NAME}")
    parser.add_argument("mode", choices=["compile", "correctness", "performance"])
    args = parser.parse_args()

    if args.mode == "compile":
        run_compile()
    elif args.mode == "correctness":
        run_correctness()
    else:
        run_performance()


if __name__ == "__main__":
    main()
