#!/usr/bin/env python3
"""Task runner for repository/aiter/layernorm2d.

Adapts the aiter JIT-compiled HIP kernel workflow to the AgentKernelArena
evaluator interface (compile / correctness / performance).

Key paths (aiter pip-installed environment):
  - Kernel source : /opt/venv/lib/python3.12/site-packages/aiter_meta/csrc/py_itfs_ck/norm_kernels.cu
  - JIT module    : module_norm
  - JIT .so cache : /opt/venv/lib/python3.12/site-packages/aiter/jit/module_norm.so
  - JIT build dir : /opt/venv/lib/python3.12/site-packages/aiter/jit/build/module_norm/
  - Test file     : /workspace/aiter-src/op_tests/test_layernorm2d.py

Notes:
  - Only bf16 is supported.
  - The test runs test_layernorm2d_fuseAdd (fused LayerNorm + residual add).
  - module_norm is a composite module (includes both layernorm2d and rmsnorm).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

TASK_NAME = "repository/aiter/layernorm2d"
MODULE_NAME = "module_norm"

# aiter pip-installed paths
AITER_JIT_DIR = Path("/opt/venv/lib/python3.12/site-packages/aiter/jit")
AITER_SRC_DIR = Path("/workspace/aiter-src")
TEST_SCRIPT = AITER_SRC_DIR / "op_tests" / "test_layernorm2d.py"

# Shapes to test
M_VALUES = [1, 4, 16, 64, 128, 256]
N_VALUES = [4096, 8192, 16384, 32768, 65536]


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
        ["python3", str(TEST_SCRIPT), "-m", "128", "-n", "8192"],
        cwd=AITER_SRC_DIR,
        timeout_s=600,
    )

    report = {"status": "ok" if ok else "fail", "output": out[-2000:]}
    (_report_root() / "compile_report.json").write_text(json.dumps(report, indent=2))

    if not ok:
        print("Compilation: FAIL")
        sys.exit(1)
    print("Compilation: PASS")


# ── correctness ──────────────────────────────────────────────────────────────

def run_correctness() -> None:
    """Run correctness tests across multiple shapes (bf16 only).

    Each run of test_layernorm2d.py outputs two lines:
      1. [perf] line containing passed~ (layernorm output check, atol=0.03, rtol=0.01)
      2. res check line containing passed~ (residual output check, atol=0.01, rtol=0.01)
    Both must show passed~ for success.
    """
    total_passed = 0
    total_failed = 0
    shape_results = []

    for m in M_VALUES:
        for n in N_VALUES:
            ok, out = _run_cmd(
                ["python3", str(TEST_SCRIPT), "-m", str(m), "-n", str(n)],
                cwd=AITER_SRC_DIR,
                timeout_s=120,
            )

            # Check [perf] line for passed~
            perf_passed = bool(re.search(r"\[perf\].*passed~", out))
            # Check res check line for passed~
            res_passed = bool(re.search(r"res check.*passed~", out))

            shape_ok = ok and perf_passed and res_passed
            shape_id = f"layernorm2d_m{m}_n{n}_bf16"

            if shape_ok:
                total_passed += 1
                shape_results.append({"shape": shape_id, "status": "pass"})
                print(f"  {shape_id}: PASS")
            else:
                total_failed += 1
                shape_results.append({
                    "shape": shape_id,
                    "status": "fail",
                    "perf_passed": perf_passed,
                    "res_passed": res_passed,
                    "returncode_ok": ok,
                })
                print(f"  {shape_id}: FAIL (perf_passed={perf_passed}, res_passed={res_passed})")

    report = {
        "status": "ok" if total_failed == 0 else "fail",
        "passed": total_passed,
        "failed": total_failed,
        "shapes": shape_results,
    }
    (_report_root() / "correctness_report.json").write_text(json.dumps(report, indent=2))

    print(f"Correctness: {total_passed} passed, {total_failed} failed")
    if total_failed > 0:
        print("Correctness: FAIL")
        sys.exit(1)
    print("Correctness: PASS")


# ── performance ──────────────────────────────────────────────────────────────

def run_performance() -> None:
    """Run performance benchmarks across multiple shapes (bf16 only).

    Output format from test_layernorm2d.py:
      [aiter] [perf] dim: (128, 8192)  , dtype: torch.bfloat16, torch avg: 21.18  us, ck avg: 12.01  us, uplift: 76.4%[checkAllclose ...]
    Parses ck avg value as execution_time_ms.
    """
    results = []

    for m in M_VALUES:
        for n in N_VALUES:
            ok, out = _run_cmd(
                ["python3", str(TEST_SCRIPT), "-m", str(m), "-n", str(n)],
                cwd=AITER_SRC_DIR,
                timeout_s=120,
            )

            if not ok:
                print(f"  layernorm2d_m{m}_n{n}_bf16: run FAILED")
                continue

            # Parse ck avg from [perf] line
            ck_match = re.search(r"\[perf\].*ck avg:\s*([\d.]+)\s*us", out)
            if ck_match:
                ck_us = float(ck_match.group(1))
                test_case_id = f"layernorm2d_m{m}_n{n}_bf16"
                results.append({
                    "test_case_id": test_case_id,
                    "shape": [m, n],
                    "execution_time_ms": ck_us / 1000.0,
                    "metadata": {
                        "m": m,
                        "n": n,
                        "dtype": "bf16",
                        "kernel": "ck",
                        "unit_original": "us",
                        "value_us": ck_us,
                    },
                })
                print(f"  {test_case_id}: {ck_us:.2f} us")
            else:
                print(f"  layernorm2d_m{m}_n{n}_bf16: could not parse ck avg")

    (_report_root() / "performance_report.json").write_text(json.dumps(results, indent=2))

    if results:
        avg_us = sum(r["metadata"]["value_us"] for r in results) / len(results)
        print(f"Performance: {len(results)} test cases, avg {avg_us:.2f} us")
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
