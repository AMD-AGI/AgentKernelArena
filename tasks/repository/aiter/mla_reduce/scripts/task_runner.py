#!/usr/bin/env python3
"""Task runner for repository/aiter/mla_reduce.

Adapts the aiter JIT-compiled HIP kernel workflow to the AgentKernelArena
evaluator interface (compile / correctness / performance).

Uses a standalone test script (test_mla_bench.py) that directly calls
aiter.mla.mla_decode_fwd() without depending on aiter's test_mla.py.
This avoids runtime patching of upstream test scripts.

Key paths (dynamically resolved via `import aiter`):
  - JIT .so cache : <aiter_package>/jit/
  - Test script   : scripts/test_mla_bench.py (self-contained, shipped with this case)
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

TASK_NAME = "repository/aiter/mla_reduce"
MODULE_NAME = "module_mla_reduce"

# Dynamically resolve aiter paths
import aiter as _aiter_pkg_ref
_aiter_pkg_dir = Path(_aiter_pkg_ref.__file__).parent

AITER_JIT_DIR = _aiter_pkg_dir / "jit"

# Standalone test script (in the same scripts/ directory)
TEST_SCRIPT = Path(__file__).resolve().parent / "test_mla_bench.py"


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
    """Compile the MLA module by running a small test.

    If the JIT .so already exists, reuse it (avoids lengthy recompilation
    that requires the full JIT toolchain).  Otherwise, trigger a build.
    """
    so_path = AITER_JIT_DIR / f"{MODULE_NAME}.so"
    if so_path.exists():
        print(f"JIT module already exists: {so_path}")
        report = {"status": "ok", "output": f"JIT module already cached: {so_path}"}
        (_report_root() / "compile_report.json").write_text(json.dumps(report, indent=2))
        print("Compilation: PASS")
        return

    ok, out = _run_cmd(
        [
            "python3", str(TEST_SCRIPT),
            "--mode", "correctness",
            "--batch", "1",
            "--ctx", "64",
            "--nhead", "16",
        ],
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
    """Run correctness tests across multiple shapes.

    Parses [RESULT] lines from test_mla_bench.py output.
    status=PASS means passed, status=FAIL means failed, status=SKIP is ignored.
    """
    ok, out = _run_cmd(
        [
            "python3", str(TEST_SCRIPT),
            "--mode", "correctness",
            "--batch", "1", "16", "64", "128",
            "--ctx", "64", "256", "1200", "3200",
            "--nhead", "16", "128",
        ],
        timeout_s=600,
    )

    passed = len(re.findall(r"status=PASS", out))
    failed = len(re.findall(r"status=FAIL", out))

    report = {
        "status": "ok" if (ok and failed == 0 and passed > 0) else "fail",
        "passed": passed,
        "failed": failed,
    }
    (_report_root() / "correctness_report.json").write_text(json.dumps(report, indent=2))

    print(f"Correctness: {passed} passed, {failed} failed")
    if failed > 0 or passed == 0:
        print("Correctness: FAIL")
        sys.exit(1)
    print("Correctness: PASS")


# ── performance ──────────────────────────────────────────────────────────────

def run_performance() -> None:
    """Run performance benchmarks and write performance_report.json.

    Parses [RESULT] lines with us=<value> from test_mla_bench.py output.
    """
    ok, out = _run_cmd(
        [
            "python3", str(TEST_SCRIPT),
            "--mode", "performance",
            "--batch", "1", "16", "64", "128",
            "--ctx", "64", "256", "1200", "3200",
            "--nhead", "16", "128",
        ],
        timeout_s=600,
    )

    if not ok:
        (_report_root() / "performance_report.json").write_text("[]")
        print("Performance: FAIL")
        sys.exit(1)

    pattern = re.compile(
        r"\[RESULT\] batch=(\d+) ctx=(\d+) nhead=(\d+) us=([\d.]+)"
    )

    results = []
    for m in pattern.finditer(out):
        batch, ctx, nhead = m.group(1), m.group(2), m.group(3)
        us = float(m.group(4))
        results.append({
            "test_case_id": f"mla_reduce_b{batch}_c{ctx}_n{nhead}",
            "execution_time_ms": us / 1000.0,
            "metadata": {
                "batch_size": batch,
                "ctx_len": ctx,
                "nhead": nhead,
                "unit_original": "us",
                "value_us": us,
            },
        })

    (_report_root() / "performance_report.json").write_text(json.dumps(results, indent=2))
    (_report_root() / "performance.log").write_text(out)

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
