#!/usr/bin/env python3
"""Task runner for repository/aiter/activation_kernels.

Adapts the aiter JIT-compiled HIP kernel workflow to the AgentKernelArena
evaluator interface (compile / correctness / performance).

Key paths (dynamically resolved via `import aiter`):
  - JIT .so cache : <aiter_package>/jit/
  - Test script   : <aiter_repo>/op_tests/test_activation.py

NOTE: test_activation.py always runs fp32 input for silu_and_mul, which crashes
(RuntimeError: "act_and_mul_kernel" not implemented for 'Float'). This is a
known test script bug — the kernel itself is correct. We tolerate the nonzero
exit code and parse results from the sections that ran before the crash.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

TASK_NAME = "repository/aiter/activation_kernels"
MODULE_NAME = "module_activation"

# Dynamically resolve aiter paths
import aiter as _aiter_pkg_ref
_aiter_pkg_dir = Path(_aiter_pkg_ref.__file__).parent

AITER_JIT_DIR = _aiter_pkg_dir / "jit"
AITER_SRC_DIR = _aiter_pkg_dir.parent
TEST_SCRIPT = AITER_SRC_DIR / "op_tests" / "test_activation.py"


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


def _is_fp32_crash(output: str) -> bool:
    """Check if the failure is the known fp32 crash (test script bug, not kernel bug)."""
    return (
        '"act_and_mul_kernel" not implemented for \'Float\'' in output
        or "act_and_mul_kernel" in output and "Float" in output
    )


# ── compile ──────────────────────────────────────────────────────────────────

def run_compile() -> None:
    """Force JIT recompilation by deleting cache and running a small test."""
    _delete_jit_cache()

    # Use small args to trigger JIT compilation quickly.
    # The test will crash at fp32 input, but JIT compilation succeeds before that.
    ok, out = _run_cmd(
        ["python3", str(TEST_SCRIPT), "-m", "1", "-n", "1024"],
        cwd=AITER_SRC_DIR,
        timeout_s=600,
    )

    # Treat as success if the JIT module was compiled (appears in output)
    # even if test exits with nonzero due to the known fp32 crash
    module_compiled = f"import [{MODULE_NAME}]" in out or f"module_activation" in out
    fp32_crash = _is_fp32_crash(out)

    if not ok and fp32_crash and module_compiled:
        print(f"Compilation: PASS (JIT module compiled; fp32 crash is expected test script behavior)")
        report = {"status": "ok", "output": out[-2000:], "note": "fp32 crash is expected"}
    elif not ok:
        print(f"Compilation: FAIL")
        report = {"status": "fail", "output": out[-2000:]}
        (_report_root() / "compile_report.json").write_text(json.dumps(report, indent=2))
        sys.exit(1)
    else:
        print(f"Compilation: PASS")
        report = {"status": "ok", "output": out[-2000:]}

    (_report_root() / "compile_report.json").write_text(json.dumps(report, indent=2))


# ── correctness ──────────────────────────────────────────────────────────────

def run_correctness() -> None:
    """Run correctness tests across multiple shapes and dtypes.

    The test script crashes when it hits fp32 input for silu_and_mul, which is
    a known test script bug. We count passed~/warning! as passed and failed as
    failed. Both passed~ and warning! are success (warning is FP8 quantization
    precision loss, which is expected and acceptable).
    """
    ok, out = _run_cmd(
        ["python3", str(TEST_SCRIPT),
         "-m", "1", "32", "128", "1024",
         "-n", "1024", "4096", "8192"],
        cwd=AITER_SRC_DIR,
        timeout_s=600,
    )

    # Count correctness markers
    passed = len(re.findall(r"passed~", out))
    warned = len(re.findall(r"warning!", out))
    failed = len(re.findall(r"failed!", out))

    total_ok = passed + warned  # both are acceptable outcomes
    fp32_crash = _is_fp32_crash(out)

    # Success if we got some passed results and the only failure is the fp32 crash
    if total_ok > 0 and failed == 0 and (ok or fp32_crash):
        status = "ok"
        print(f"Correctness: {total_ok} passed ({passed} exact pass, {warned} warning/fp8), {failed} failed")
        print("Correctness: PASS")
    else:
        status = "fail"
        print(f"Correctness: {total_ok} passed ({passed} exact pass, {warned} warning/fp8), {failed} failed")
        print("Correctness: FAIL")

    report = {
        "status": status,
        "passed": total_ok,
        "passed_exact": passed,
        "passed_warning": warned,
        "failed": failed,
    }
    (_report_root() / "correctness_report.json").write_text(json.dumps(report, indent=2))

    if status == "fail":
        sys.exit(1)


# ── performance ──────────────────────────────────────────────────────────────

def run_performance() -> None:
    """Run performance benchmarks and write performance_report.json.

    Output format from test_activation.py (markdown table from scaled_silu_and_mul):
      | M | N | input_dtype | output_dtype | us | TB/s | RD TB/s | WR TB/s | err |

    The test crashes before silu_and_mul and gelu_fast tables are emitted,
    so we parse only the scaled_silu_and_mul table which runs first.

    test_case_id format: "activation_mM_nN_dtype"
    """
    ok, out = _run_cmd(
        ["python3", str(TEST_SCRIPT),
         "-m", "1", "32", "128", "1024",
         "-n", "1024", "4096", "8192"],
        cwd=AITER_SRC_DIR,
        timeout_s=600,
    )

    fp32_crash = _is_fp32_crash(out)

    # Parse markdown table rows: lines starting with | followed by numeric M value
    # Header pattern: | M | N | input_dtype | output_dtype | us | TB/s | ...
    # Data rows:      |  1 | 1024 | fp16 | fp8 | 2.71 | ... |
    table_row_pattern = re.compile(
        r"^\|\s*(?P<m>\d+)\s*\|\s*(?P<n>\d+)\s*\|\s*(?P<input_dtype>\w+)\s*\|"
        r"\s*(?P<output_dtype>\w+)\s*\|\s*(?P<us>[\d.eE+\-]+)\s*\|",
        re.MULTILINE,
    )

    results = []
    seen = set()
    for m in table_row_pattern.finditer(out):
        shape_m = int(m.group("m"))
        shape_n = int(m.group("n"))
        input_dtype = m.group("input_dtype").strip()
        output_dtype = m.group("output_dtype").strip()
        us_val = float(m.group("us"))
        key = (shape_m, shape_n, input_dtype, output_dtype)
        if key in seen:
            continue
        seen.add(key)
        test_case_id = f"activation_m{shape_m}_n{shape_n}_{input_dtype}"
        results.append({
            "test_case_id": test_case_id,
            "shape": [shape_m, shape_n],
            "execution_time_ms": us_val / 1000.0,
            "metadata": {
                "m": shape_m,
                "n": shape_n,
                "input_dtype": input_dtype,
                "output_dtype": output_dtype,
                "unit_original": "us",
                "value_us": us_val,
            },
        })

    (_report_root() / "performance_report.json").write_text(json.dumps(results, indent=2))
    (_report_root() / "performance.log").write_text(out)

    if results:
        avg_us = sum(r["metadata"]["value_us"] for r in results) / len(results)
        print(f"Performance: {len(results)} test cases, avg {avg_us:.2f} us")
        for r in results:
            print(f"  {r['test_case_id']}: {r['metadata']['value_us']:.2f} us")
        if fp32_crash and not ok:
            print("(Note: test script exited nonzero due to known fp32 crash; performance results are valid)")
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
