#!/usr/bin/env python3
"""Task runner for repository/aiter/quant_kernels.

Adapts the aiter JIT-compiled HIP kernel workflow to the AgentKernelArena
evaluator interface (compile / correctness / performance).

Key paths (aiter pip-installed environment):
  - Kernel source : /opt/venv/lib/python3.12/site-packages/aiter_meta/csrc/kernels/quant_kernels.cu
  - JIT module    : module_quant
  - JIT .so cache : /opt/venv/lib/python3.12/site-packages/aiter/jit/module_quant.so
  - JIT build dir : /opt/venv/lib/python3.12/site-packages/aiter/jit/build/module_quant/
  - Test file     : /workspace/aiter-src/op_tests/test_quant.py

Output format (Pandas markdown table, one table per quant-type/dtype combination):
  Columns always present : m, n, q_type, q_dtype, h_dtype, triton dq, triton dq err, hip dq, hip dq err
  Extra cols (per_Tensor): triton sq, triton sq err, hip sq, hip sq err

Correctness: hip dq err == 0 or very small (< 0.01) is a pass.
  Quantization operations have inherent precision loss; small non-zero errors are normal.
Performance: hip dq (us) is used as the primary execution-time metric.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

TASK_NAME = "repository/aiter/quant_kernels"
MODULE_NAME = "module_quant"

# aiter pip-installed paths
AITER_JIT_DIR = Path("/opt/venv/lib/python3.12/site-packages/aiter/jit")
AITER_SRC_DIR = Path("/workspace/aiter-src")
TEST_SCRIPT = AITER_SRC_DIR / "op_tests" / "test_quant.py"


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
    """Force JIT recompilation by deleting cache and running a minimal test."""
    _delete_jit_cache()

    # Use a single small shape and one quant type to trigger JIT compilation
    # without OOM risk from large shapes (m=163840).
    ok, out = _run_cmd(
        ["python3", str(TEST_SCRIPT), "-m", "1", "-n", "4096", "-q", "fp8_token"],
        cwd=AITER_SRC_DIR,
        timeout_s=600,
    )

    report = {"status": "ok" if ok else "fail", "output": out[-2000:]}
    (_report_root() / "compile_report.json").write_text(json.dumps(report, indent=2))

    if not ok:
        print("Compilation: FAIL")
        sys.exit(1)
    print("Compilation: PASS")


# ── helpers for table parsing ─────────────────────────────────────────────────

def _parse_markdown_tables(text: str) -> list[list[dict]]:
    """Extract all Pandas markdown tables from output.

    Each table is returned as a list of row-dicts.
    A markdown table looks like:
      | col1 | col2 | ...
      |-----:|-----:|...
      | val  | val  | ...
    """
    tables = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        # Look for a header line: starts with '|' and contains at least two '|'
        line = lines[i].strip()
        if line.startswith("|") and line.count("|") >= 3:
            # Collect header
            header_line = line
            # Next line should be separator (contains '---')
            if i + 1 < len(lines) and re.match(r"\|[-:| ]+\|", lines[i + 1].strip()):
                sep_line = lines[i + 1]
                # Parse headers
                headers = [h.strip() for h in header_line.strip("|").split("|")]
                # Collect data rows
                rows = []
                j = i + 2
                while j < len(lines) and lines[j].strip().startswith("|"):
                    vals = [v.strip() for v in lines[j].strip("|").split("|")]
                    if len(vals) == len(headers):
                        rows.append(dict(zip(headers, vals)))
                    j += 1
                if rows:
                    tables.append(rows)
                i = j
                continue
        i += 1
    return tables


def _safe_float(s: str) -> float | None:
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


# ── correctness ──────────────────────────────────────────────────────────────

def run_correctness() -> None:
    """Run correctness tests across all quant modes and dtypes.

    Parses Pandas markdown tables. For each row, checks hip dq err (and
    hip sq err if present). A row passes if all hip *err values are < 0.01
    (quantization has inherent precision loss so small non-zero errors are
    acceptable).

    Uses a limited set of m values to avoid OOM on GPU (m=163840 requires ~5 GiB).
    """
    ok, out = _run_cmd(
        ["python3", str(TEST_SCRIPT),
         "-m", "1", "16", "128", "1024", "16384"],
        cwd=AITER_SRC_DIR,
        timeout_s=600,
    )

    tables = _parse_markdown_tables(out)

    passed = 0
    failed = 0
    fail_details = []

    for table in tables:
        for row in table:
            # Identify hip error columns: hip dq err, hip sq err
            hip_err_cols = [c for c in row if re.match(r"hip\s+\w+\s+err", c) or c == "hip dq err" or c == "hip sq err"]
            if not hip_err_cols:
                # Fallback: any column containing 'hip' and 'err'
                hip_err_cols = [c for c in row if "hip" in c and "err" in c]

            row_pass = True
            for col in hip_err_cols:
                val = _safe_float(row.get(col, ""))
                if val is None:
                    continue
                if val >= 0.01:
                    row_pass = False
                    fail_details.append(f"  row {row}: {col}={val} >= 0.01")

            if row_pass:
                passed += 1
            else:
                failed += 1

    # Also count explicit pass/fail markers in output
    explicit_passed = len(re.findall(r"passed~", out))
    explicit_failed = len(re.findall(r"(?:failed!|FAIL(?!ED))", out))

    report = {
        "status": "ok" if (ok and failed == 0 and explicit_failed == 0) else "fail",
        "table_rows_passed": passed,
        "table_rows_failed": failed,
        "explicit_passed": explicit_passed,
        "explicit_failed": explicit_failed,
        "fail_details": fail_details,
    }
    (_report_root() / "correctness_report.json").write_text(json.dumps(report, indent=2))

    print(f"Correctness: {passed} rows passed, {failed} rows failed (table parse)")
    print(f"Correctness: {explicit_passed} explicit passed~, {explicit_failed} explicit failed!")
    if failed > 0 or explicit_failed > 0 or not ok:
        print("Correctness: FAIL")
        for d in fail_details:
            print(d)
        sys.exit(1)
    print("Correctness: PASS")


# ── performance ──────────────────────────────────────────────────────────────

def run_performance() -> None:
    """Run performance benchmarks and write performance_report.json.

    Parses Pandas markdown tables. The primary metric is 'hip dq' (us) —
    the HIP kernel execution time for dynamic quantization.
    If 'hip sq' is also present (per_Tensor), it is also recorded.

    test_case_id format: quant_{q_type}_{q_dtype}_{h_dtype}_m{m}_n{n}
    execution_time_ms = hip_dq_us / 1000
    """
    ok, out = _run_cmd(
        ["python3", str(TEST_SCRIPT),
         "-m", "1", "16", "128", "1024", "16384"],
        cwd=AITER_SRC_DIR,
        timeout_s=600,
    )

    if not ok:
        (_report_root() / "performance_report.json").write_text("[]")
        print("Performance: FAIL")
        sys.exit(1)

    tables = _parse_markdown_tables(out)

    results = []
    seen = set()

    for table in tables:
        for row in table:
            # Extract identifying fields
            m_str = row.get("m", "")
            n_str = row.get("n", "")
            q_type_str = row.get("q_type", "")
            q_dtype_str = row.get("q_dtype", "")
            h_dtype_str = row.get("h_dtype", "")

            m_val = _safe_float(m_str)
            n_val = _safe_float(n_str)
            if m_val is None or n_val is None:
                continue

            # Build a sanitized test case suffix
            def _sanitize(s: str) -> str:
                return re.sub(r"[^a-zA-Z0-9_]", "_", s).strip("_")

            q_dtype_tag = _sanitize(q_dtype_str.split(".")[-1])
            h_dtype_tag = _sanitize(h_dtype_str.split(".")[-1])

            # hip dq (dynamic quant)
            hip_dq_str = row.get("hip dq", "")
            hip_dq_us = _safe_float(hip_dq_str)
            if hip_dq_us is not None:
                key = ("dq", int(m_val), int(n_val), q_type_str, q_dtype_tag, h_dtype_tag)
                if key not in seen:
                    seen.add(key)
                    test_id = f"quant_dq_{q_dtype_tag}_{h_dtype_tag}_m{int(m_val)}_n{int(n_val)}"
                    results.append({
                        "test_case_id": test_id,
                        "shape": [int(m_val), int(n_val)],
                        "execution_time_ms": hip_dq_us / 1000.0,
                        "metadata": {
                            "m": int(m_val),
                            "n": int(n_val),
                            "q_type": q_type_str,
                            "q_dtype": q_dtype_str,
                            "h_dtype": h_dtype_str,
                            "kernel": "hip_dq",
                            "unit_original": "us",
                            "value_us": hip_dq_us,
                        },
                    })

            # hip sq (static quant, only for per_Tensor)
            hip_sq_str = row.get("hip sq", "")
            hip_sq_us = _safe_float(hip_sq_str)
            if hip_sq_us is not None:
                key = ("sq", int(m_val), int(n_val), q_type_str, q_dtype_tag, h_dtype_tag)
                if key not in seen:
                    seen.add(key)
                    test_id = f"quant_sq_{q_dtype_tag}_{h_dtype_tag}_m{int(m_val)}_n{int(n_val)}"
                    results.append({
                        "test_case_id": test_id,
                        "shape": [int(m_val), int(n_val)],
                        "execution_time_ms": hip_sq_us / 1000.0,
                        "metadata": {
                            "m": int(m_val),
                            "n": int(n_val),
                            "q_type": q_type_str,
                            "q_dtype": q_dtype_str,
                            "h_dtype": h_dtype_str,
                            "kernel": "hip_sq",
                            "unit_original": "us",
                            "value_us": hip_sq_us,
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
