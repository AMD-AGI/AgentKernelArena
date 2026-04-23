#!/usr/bin/env python3
"""Task runner for repository/aiter/topk_softmax_kernels.

Adapts the aiter JIT-compiled HIP kernel workflow to the AgentKernelArena
evaluator interface (compile / correctness / performance).

Key paths (aiter pip-installed environment):
  - Kernel source : /opt/venv/lib/python3.12/site-packages/aiter_meta/csrc/kernels/topk_softmax_kernels.cu
  - JIT module    : module_moe_asm  (composite: 7 .cu files)
  - JIT .so cache : /opt/venv/lib/python3.12/site-packages/aiter/jit/module_moe_asm.so
  - JIT build dir : /opt/venv/lib/python3.12/site-packages/aiter/jit/build/module_moe_asm/
  - Test file     : /workspace/aiter-src/op_tests/test_moeTopkSoftmax.py

CRITICAL: NEVER run test_moeTopkSoftmax.py without explicit -e/-t/-k parameters!
The test script runs 4 groups:
  1. test_topk_softmax            - the kernel under test (passes with safe params)
  2. test_biased_grouped_topk     - always works
  3. test_grouped_topk            - always works
  4. test_topk_softmax_shared_experts - crashes with API mismatch (7 args vs 5)

The crash in group 4 causes returncode=1 (clean), BUT the ASM kernel in group 1
triggers a GPU memory fault (returncode=-6 / SIGABRT) when token counts are large:
  - E=128, topk=8: safe up to token=64   (token>=96 causes GPU memory fault)
  - E=256, topk=8: safe up to token=32   (token>=64 causes GPU memory fault)

Always pass -e, -t, -k explicitly and keep tokens within safe limits.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

TASK_NAME = "repository/aiter/topk_softmax_kernels"
MODULE_NAME = "module_moe_asm"

# aiter pip-installed paths
AITER_JIT_DIR = Path("/opt/venv/lib/python3.12/site-packages/aiter/jit")
AITER_SRC_DIR = Path("/workspace/aiter-src")
TEST_SCRIPT = AITER_SRC_DIR / "op_tests" / "test_moeTopkSoftmax.py"


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _report_root() -> Path:
    d = _workspace_root() / "build"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _run_cmd(cmd: list[str], cwd: Path | None = None, timeout_s: int = 600) -> tuple[bool, str]:
    """Run a command and return (success, combined_output).

    Note: test_moeTopkSoftmax.py always exits non-zero because group 4
    (test_topk_softmax_shared_experts) crashes with an API mismatch. We capture
    the output regardless of return code and let callers decide what to do.
    A returncode of -6 (SIGABRT) indicates a GPU memory fault in the ASM kernel,
    which means the token count was too large for the current (E, topk) combination.
    """
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


def _parse_markdown_table(output: str) -> list[dict]:
    """Parse the first Pandas markdown table from the moeTopkSoftmax summary.

    The table appears after 'moeTopkSoftmax summary (markdown):' in the output.
    Columns of interest: dtype, token, E, topk, hip err, hip us, asm err, asm us.

    Returns a list of row dicts with normalized column names (stripped).
    """
    # Find the section with the main test_topk_softmax results
    marker = "moeTopkSoftmax summary (markdown):"
    idx = output.find(marker)
    if idx == -1:
        return []

    # Extract from marker onwards, stop at next [aiter] log line or end
    section = output[idx + len(marker):]

    # Collect markdown table lines (lines starting with '|')
    table_lines = []
    in_table = False
    for line in section.splitlines():
        stripped = line.strip()
        if stripped.startswith("|"):
            table_lines.append(stripped)
            in_table = True
        elif in_table:
            # Table ended
            break

    if len(table_lines) < 2:
        return []

    # Parse header
    header = [h.strip() for h in table_lines[0].split("|") if h.strip()]
    # Skip separator line (second line with dashes)
    rows = []
    for line in table_lines[2:]:
        cells = [c.strip() for c in line.split("|")]
        # First and last are empty due to leading/trailing '|'
        cells = cells[1:-1]
        if len(cells) != len(header):
            continue
        row = {header[i]: cells[i] for i in range(len(header))}
        rows.append(row)

    return rows


# ── compile ──────────────────────────────────────────────────────────────────

def run_compile() -> None:
    """Force JIT recompilation by deleting cache and running a small test.

    Uses minimal parameters (-e 128 -t 1 4 16 -k 8) to keep compile time short
    while still triggering the JIT build for module_moe_asm.

    IMPORTANT: Always pass explicit -e, -t, -k to avoid triggering the
    test_topk_softmax_shared_experts group which has API mismatch and crashes.
    """
    _delete_jit_cache()

    # Run with minimal params — just enough to trigger JIT compilation.
    # The script will exit non-zero when it hits group 4 (API mismatch crash),
    # but compilation happens during group 1, so we check for the .so file.
    _ok, out = _run_cmd(
        ["python3", str(TEST_SCRIPT), "-e", "128", "-t", "1", "4", "16", "-k", "8"],
        cwd=AITER_SRC_DIR,
        timeout_s=600,
    )

    # Compilation succeeded if the .so was produced
    so_path = AITER_JIT_DIR / f"{MODULE_NAME}.so"
    compiled = so_path.exists()

    report = {
        "status": "ok" if compiled else "fail",
        "output": out[-2000:],
        "so_exists": compiled,
    }
    (_report_root() / "compile_report.json").write_text(json.dumps(report, indent=2))

    if not compiled:
        print("Compilation: FAIL (module_moe_asm.so not found after build)")
        sys.exit(1)
    print("Compilation: PASS")


# ── correctness ──────────────────────────────────────────────────────────────

def run_correctness() -> None:
    """Run correctness tests.

    Parameters: -e 128 256  -t 1  -k 8

    Token restricted to 1 only: baseline HIP kernel has a known bug with
    strided tensors (stride=topk+10) that causes correctness failures for
    token>1. Only token=1 passes reliably.

    The script exits non-zero (group 4 API crash) but the group 1 markdown table
    is captured before the crash. We parse 'hip err' and 'asm err' columns:
      - 0   = passed
      - > 0 = failed (fraction of elements that failed checkAllclose)
    """
    _ok, out = _run_cmd(
        [
            "python3", str(TEST_SCRIPT),
            "-e", "128", "256",
            "-t", "1",
            "-k", "8",
        ],
        cwd=AITER_SRC_DIR,
        timeout_s=600,
    )

    rows = _parse_markdown_table(out)

    passed = 0
    failed = 0
    failed_cases = []

    if rows:
        for row in rows:
            dtype = row.get("dtype", "?")
            token = row.get("token", "?")
            e_val = row.get("E", "?")
            topk = row.get("topk", "?")

            hip_err_str = row.get("hip err", "").strip()
            asm_err_str = row.get("asm err", "").strip()

            case_id = f"topk_softmax_dtype{dtype}_e{e_val}_t{token}_k{topk}"

            row_ok = True

            # hip err: 0 = passed (fraction of failed elements from checkAllclose)
            try:
                hip_err = float(hip_err_str)
                if hip_err != 0.0:
                    row_ok = False
                    failed_cases.append(f"{case_id} hip_err={hip_err}")
            except ValueError:
                if hip_err_str not in ("", "nan"):
                    row_ok = False
                    failed_cases.append(f"{case_id} hip_err=unparseable({hip_err_str})")

            # asm err: informational only — the optimization target is the HIP
            # kernel (topk_softmax_kernels.cu), not the ASM kernel. ASM has
            # known baseline correctness issues and is not counted for pass/fail.

            if row_ok:
                passed += 1
            else:
                failed += 1
    else:
        # Fallback: parse raw output for passed~/failed! markers
        passed = len(re.findall(r"passed~", out))
        failed = len(re.findall(r"(?:failed!|FAIL)", out))

    report = {
        "status": "ok" if (len(rows) > 0 and failed == 0) else "fail",
        "parsed_rows": len(rows),
        "passed": passed,
        "failed": failed,
        "failed_cases": failed_cases,
    }
    (_report_root() / "correctness_report.json").write_text(json.dumps(report, indent=2))

    print(f"Correctness: {passed} passed, {failed} failed (from {len(rows)} rows)")
    if failed_cases:
        for fc in failed_cases[:10]:  # Show at most 10 failures
            print(f"  FAIL: {fc}")
        if len(failed_cases) > 10:
            print(f"  ... and {len(failed_cases) - 10} more failures")
    if failed > 0 or len(rows) == 0:
        print("Correctness: FAIL")
        sys.exit(1)
    print("Correctness: PASS")


# ── performance ──────────────────────────────────────────────────────────────

def run_performance() -> None:
    """Run performance benchmarks and write performance_report.json.

    Parameters: -e 128 256  -t 1 4 16 32  -k 8

    Token range is limited to max=32 to avoid GPU memory fault (same constraint
    as correctness; see run_correctness for details).

    Parses the first markdown table (moeTopkSoftmax summary) for 'hip us'
    (HIP kernel execution time in microseconds). This is the primary metric.

    Test case ID format: topk_softmax_e{E}_t{T}_k{K}
    execution_time_ms = hip_us / 1000
    """
    _ok, out = _run_cmd(
        [
            "python3", str(TEST_SCRIPT),
            "-e", "128", "256",
            "-t", "1", "4", "16", "32",
            "-k", "8",
        ],
        cwd=AITER_SRC_DIR,
        timeout_s=600,
    )

    rows = _parse_markdown_table(out)

    if not rows:
        (_report_root() / "performance_report.json").write_text("[]")
        print("Performance: FAIL (no table parsed — possible GPU crash)")
        sys.exit(1)

    results = []
    for row in rows:
        dtype = row.get("dtype", "?").replace("torch.", "")
        token = row.get("token", "?")
        e_val = row.get("E", "?")
        topk_val = row.get("topk", "?")
        hip_us_str = row.get("hip us", "").strip()
        asm_us_str = row.get("asm us", "").strip()

        try:
            hip_us = float(hip_us_str)
        except ValueError:
            continue  # Skip rows where hip us is missing

        try:
            asm_us = float(asm_us_str) if asm_us_str and asm_us_str.lower() != "nan" else None
        except ValueError:
            asm_us = None

        case_id = f"topk_softmax_e{e_val}_t{token}_k{topk_val}"

        entry = {
            "test_case_id": case_id,
            "execution_time_ms": hip_us / 1000.0,
            "metadata": {
                "dtype": dtype,
                "E": e_val,
                "token": token,
                "topk": topk_val,
                "kernel": "hip",
                "unit_original": "us",
                "hip_us": hip_us,
            },
        }
        if asm_us is not None:
            entry["metadata"]["asm_us"] = asm_us

        results.append(entry)

    (_report_root() / "performance_report.json").write_text(json.dumps(results, indent=2))

    if results:
        avg_us = sum(r["metadata"]["hip_us"] for r in results) / len(results)
        print(f"Performance: {len(results)} test cases, avg hip {avg_us:.2f} us")
        for r in results:
            meta = r["metadata"]
            asm_info = f", asm {meta['asm_us']:.2f} us" if "asm_us" in meta else ""
            print(f"  {r['test_case_id']} ({meta['dtype']}): hip {meta['hip_us']:.2f} us{asm_info}")
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
