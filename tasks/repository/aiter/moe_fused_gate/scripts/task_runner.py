#!/usr/bin/env python3
"""Task runner for repository/aiter/moe_fused_gate.

Adapts the aiter JIT-compiled HIP kernel workflow to the AgentKernelArena
evaluator interface (compile / correctness / performance).

Uses a standalone test script (test_moe_fused_gate_bench.py) that directly
calls aiter.moe_fused_gate() without depending on aiter's test_moeTopkSoftmax.py.
This avoids interference with the topk_softmax_kernels case which shares
the same JIT module (module_moe_asm).

Key paths (dynamically resolved via `import aiter`):
  - JIT .so cache : <aiter_package>/jit/
  - Test script   : scripts/test_moe_fused_gate_bench.py (self-contained)
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

TASK_NAME = "repository/aiter/moe_fused_gate"
MODULE_NAME = "module_moe_asm"

# Dynamically resolve aiter paths
import aiter as _aiter_pkg_ref
_aiter_pkg_dir = Path(_aiter_pkg_ref.__file__).parent

AITER_JIT_DIR = _aiter_pkg_dir / "jit"

# Standalone test script (in the same scripts/ directory)
TEST_SCRIPT = Path(__file__).resolve().parent / "test_moe_fused_gate_bench.py"


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
    """Remove JIT-compiled .so and build directory to force recompilation.

    NOTE: module_moe_asm is a composite module shared with topk_softmax_kernels.
    Deleting it forces recompilation of all kernels in the module, which takes
    longer (~120s) than single-file modules.
    """
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
    """Compile the module by running a small correctness test.

    If the JIT .so already exists, reuse it to avoid lengthy recompilation.
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
            "--token", "1",
            "--expert", "128",
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

    Parses [RESULT] lines from test_moe_fused_gate_bench.py output.
    status=PASS means passed, status=FAIL means failed.
    """
    ok, out = _run_cmd(
        [
            "python3", str(TEST_SCRIPT),
            "--mode", "correctness",
            "--token", "1", "4", "16", "64",
            "--expert", "128", "256",
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

    Parses [RESULT] lines with us=<value> from test_moe_fused_gate_bench.py.
    """
    ok, out = _run_cmd(
        [
            "python3", str(TEST_SCRIPT),
            "--mode", "performance",
            "--token", "1", "4", "16", "64", "256",
            "--expert", "128", "256",
        ],
        timeout_s=600,
    )

    if not ok:
        (_report_root() / "performance_report.json").write_text("[]")
        print("Performance: FAIL")
        sys.exit(1)

    pattern = re.compile(
        r"\[RESULT\] token=(\d+) expert=(\d+) group=(\d+) topk=(\d+) us=([\d.]+)"
    )

    results = []
    for m in pattern.finditer(out):
        token, expert, group, topk = m.group(1), m.group(2), m.group(3), m.group(4)
        us = float(m.group(5))
        results.append({
            "test_case_id": f"moe_fused_gate_t{token}_e{expert}_g{group}_k{topk}",
            "execution_time_ms": us / 1000.0,
            "metadata": {
                "token": int(token),
                "expert": int(expert),
                "group": int(group),
                "topk": int(topk),
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
