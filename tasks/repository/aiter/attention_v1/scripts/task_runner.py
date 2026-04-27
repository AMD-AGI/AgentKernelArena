#!/usr/bin/env python3
"""Task runner for repository/aiter/attention_v1.

Adapts the aiter JIT-compiled HIP kernel workflow to the AgentKernelArena
evaluator interface (compile / correctness / performance).

Key paths (dynamically resolved via `import aiter`):
  - JIT .so cache : <aiter_package>/jit/
  - Test script   : <aiter_repo>/op_tests/test_pa_v1.py

Test output format (when run as __main__):
  - The script runs with fixed params: ctx_len=2048, num_seqs=8, heads=(8,1), head_size=128
  - Correctness + timing line:
      [aiter] golden vs aiter:{time_us}[checkAllclose atol=0.01 rtol=0.01 passed~]
  - "passed~" indicates success; "FAIL" or "failed" indicates failure
  - time_us is the kernel execution time in microseconds from perftest
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

TASK_NAME = "repository/aiter/attention_v1"
MODULE_NAME = "module_pa_v1"

# Dynamically resolve aiter paths
import aiter as _aiter_pkg_ref
_aiter_pkg_dir = Path(_aiter_pkg_ref.__file__).parent

AITER_JIT_DIR = _aiter_pkg_dir / "jit"
AITER_SRC_DIR = _aiter_pkg_dir.parent
TEST_SCRIPT = AITER_SRC_DIR / "op_tests" / "test_pa_v1.py"


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
    """Run correctness tests and parse passed/failed counts."""
    ok, out = _run_cmd(
        ["python3", str(TEST_SCRIPT)],
        cwd=AITER_SRC_DIR,
        timeout_s=600,
    )

    # Parse passed/failed counts
    # Success: "passed~" (from checkAllclose with ANSI codes stripped)
    # Failure: "FAIL" or "failed"
    passed = len(re.findall(r"passed~", out))
    failed = len(re.findall(r"(?:FAIL|failed)", out))

    report = {
        "status": "ok" if (ok and passed > 0 and failed == 0) else "fail",
        "passed": passed,
        "failed": failed,
    }
    (_report_root() / "correctness_report.json").write_text(json.dumps(report, indent=2))

    print(f"Correctness: {passed} passed, {failed} failed")
    if failed > 0 or not ok or passed == 0:
        print("Correctness: FAIL")
        sys.exit(1)
    print("Correctness: PASS")


# ── performance ──────────────────────────────────────────────────────────────

def run_performance() -> None:
    """Run performance benchmarks and write performance_report.json.

    Output format from test_pa_v1.py (run as __main__):
      [aiter] golden vs aiter:{time_us}[checkAllclose atol=0.01 rtol=0.01 passed~]

    The __main__ block runs with fixed params:
      ctx_len=2048, num_seqs=8, num_heads=(8, 1), head_size=128
    The timing value is in microseconds from the perftest decorator.
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

    # Parse timing from the checkAllclose output line:
    # "golden vs aiter:{time_us}[checkAllclose atol=0.01 rtol=0.01 passed~]"
    # ANSI escape codes may be present around "passed~"
    pattern = re.compile(
        r"golden vs aiter:([\d.]+)\[checkAllclose"
    )

    results = []
    for m in pattern.finditer(out):
        time_us = float(m.group(1))
        # Fixed params used by __main__: ctx_lens=2048, num_seqs=8, heads=(8,1), head_size=128
        results.append({
            "test_case_id": "pa_v1_ctx2048_seqs8_heads8x1_hd128",
            "execution_time_ms": time_us / 1000.0,
            "metadata": {
                "ctx_lens": 2048,
                "num_seqs": 8,
                "num_query_heads": 8,
                "num_kv_heads": 1,
                "head_size": 128,
                "dtype": "fp16",
                "kv_cache_layout": "NHD",
                "unit_original": "us",
                "value_us": time_us,
            },
        })

    (_report_root() / "performance_report.json").write_text(json.dumps(results, indent=2))
    (_report_root() / "performance.log").write_text(out)

    if results:
        avg_us = sum(r["metadata"]["value_us"] for r in results) / len(results)
        print(f"Performance: {len(results)} test case(s), avg {avg_us:.2f} us")
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
