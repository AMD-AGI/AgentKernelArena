#!/usr/bin/env python3
"""Task runner for repository/aiter/sample_kernels.

Adapts the aiter JIT-compiled HIP kernel workflow to the AgentKernelArena
evaluator interface (compile / correctness / performance).

Key paths (dynamically resolved via `import aiter`):
  - JIT .so cache : <aiter_package>/jit/
  - Test script   : <aiter_repo>/op_tests/test_sample.py

Output format (Pandas markdown table, default run):
  Columns: M, N, dtype, eps, origin_us, exp_out_aiter_us, exp_out_aiter_err,
           exp_in_aiter_us, exp_in_aiter_err
  Default params: bf16, N=151936, M in [1, 8, 16, 32, 64, 128, 192, 256, 512]
  Correctness: atol=0, rtol=0 (strict exact match), "passed~" = pass, "failed!" / "FAIL" = fail
  Primary performance metric: exp_in_aiter_us
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

TASK_NAME = "repository/aiter/sample_kernels"
MODULE_NAME = "module_sample"

# Dynamically resolve aiter paths
import aiter as _aiter_pkg_ref
_aiter_pkg_dir = Path(_aiter_pkg_ref.__file__).parent

AITER_JIT_DIR = _aiter_pkg_dir / "jit"
AITER_SRC_DIR = _aiter_pkg_dir.parent
TEST_SCRIPT = AITER_SRC_DIR / "op_tests" / "test_sample.py"


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
    """Force JIT recompilation by deleting cache and running the test script."""
    _delete_jit_cache()

    ok, out = _run_cmd(
        ["python3", str(TEST_SCRIPT)],
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
    """Run correctness tests across sample types (greedy, random, mixed).

    The test uses atol=0, rtol=0 (strict exact match).
    Success indicator: "passed~"
    Failure indicators: "failed!" or "FAIL"
    """
    ok, out = _run_cmd(
        ["python3", str(TEST_SCRIPT)],
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

    Output format: Pandas markdown table printed via aiter.logger.info.
    Columns include: M, N, dtype, origin_us, exp_out_aiter_us, exp_in_aiter_us, etc.
    Primary metric: exp_in_aiter_us (inner exponential aiter implementation).
    Test case ID: sample_M{m}_N{n}

    Example output line:
      |   1 | 151936 | torch.bfloat16 | 1e-06 | 239.778 | 438.04 | 0 | 153.594 | 0 |
    """
    ok, out = _run_cmd(
        ["python3", str(TEST_SCRIPT)],
        cwd=AITER_SRC_DIR,
        timeout_s=600,
    )

    if not ok:
        (_report_root() / "performance_report.json").write_text("[]")
        print("Performance: FAIL")
        sys.exit(1)

    results = []
    seen = set()

    # Find all markdown tables in the output.
    # The table header for random/mixed sample looks like:
    #   |   M |      N | dtype          |   eps |   origin_us |   exp_out_aiter_us | ...
    # We split around headers and parse data rows.
    # Strategy: find header line, then parse subsequent data rows until blank/non-table line.

    lines = out.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        # Detect a markdown table header containing the expected performance columns
        if "|" in line and "exp_in_aiter_us" in line:
            # Parse this table: header is lines[i], separator is lines[i+1], data from lines[i+2]
            header_line = line
            headers = [h.strip() for h in header_line.strip().strip("|").split("|")]

            # Find column indices
            try:
                idx_m = headers.index("M")
                idx_n = headers.index("N")
                idx_origin = headers.index("origin_us")
                idx_exp_out = headers.index("exp_out_aiter_us")
                idx_exp_in = headers.index("exp_in_aiter_us")
            except ValueError:
                i += 1
                continue

            # Skip separator line
            i += 2
            while i < len(lines):
                row_line = lines[i]
                if "|" not in row_line:
                    break
                cols = [c.strip() for c in row_line.strip().strip("|").split("|")]
                if len(cols) <= max(idx_m, idx_n, idx_exp_in):
                    i += 1
                    continue
                try:
                    m = int(float(cols[idx_m]))
                    n = int(float(cols[idx_n]))
                    origin_us = float(cols[idx_origin])
                    exp_out_us = float(cols[idx_exp_out])
                    exp_in_us = float(cols[idx_exp_in])
                except (ValueError, IndexError):
                    i += 1
                    continue

                key = (m, n)
                if key not in seen:
                    seen.add(key)
                    results.append({
                        "test_case_id": f"sample_M{m}_N{n}",
                        "shape": [m, n],
                        "execution_time_ms": exp_in_us / 1000.0,
                        "metadata": {
                            "m": m,
                            "n": n,
                            "origin_us": origin_us,
                            "exp_out_aiter_us": exp_out_us,
                            "exp_in_aiter_us": exp_in_us,
                            "unit_original": "us",
                        },
                    })
                i += 1
            continue
        i += 1

    (_report_root() / "performance_report.json").write_text(json.dumps(results, indent=2))
    (_report_root() / "performance.log").write_text(out)

    if results:
        avg_us = sum(r["metadata"]["exp_in_aiter_us"] for r in results) / len(results)
        print(f"Performance: {len(results)} test cases, avg exp_in_aiter {avg_us:.2f} us")
        for r in results:
            m = r["metadata"]
            print(
                f"  {r['test_case_id']}: origin={m['origin_us']:.2f} us, "
                f"exp_out={m['exp_out_aiter_us']:.2f} us, "
                f"exp_in={m['exp_in_aiter_us']:.2f} us"
            )
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
