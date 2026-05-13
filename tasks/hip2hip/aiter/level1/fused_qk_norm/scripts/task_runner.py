#!/usr/bin/env python3
"""Task runner for hip2hip/aiter/level1/fused_qk_norm.

Adapts the aiter JIT-compiled HIP kernel workflow to the AgentKernelArena
evaluator interface (compile / correctness / performance).

L1/L2 key difference from L3 (repository) cases:
  - Source code lives in src/ directory (self-contained)
  - task_runner copies modified .cu to aiter install path before JIT recompilation
  - No repo clone needed

Key paths (dynamically resolved via `import aiter`):
  - JIT .so cache      : <aiter_package>/jit/
  - Kernel install path: <aiter_meta>/csrc/kernels/
  - Test script        : <aiter_repo>/op_tests/test_fused_qk_norm.py
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

TASK_NAME = "hip2hip/aiter/level1/fused_qk_norm"
MODULE_NAME = "module_fused_qk_norm_rope_cache_quant_shuffle"
SOURCE_FILE = "fused_qk_norm.cu"

# ── Path resolution ──────────────────────────────────────────────────────────

import aiter as _aiter_pkg
_aiter_pkg_dir = Path(_aiter_pkg.__file__).resolve().parent

AITER_JIT_DIR = _aiter_pkg_dir / "jit"

# aiter_meta contains the C++ source / headers used by JIT
import aiter_meta as _aiter_meta_pkg
_aiter_meta_dir = Path(_aiter_meta_pkg.__file__).resolve().parent
KERNEL_INSTALL_DIR = _aiter_meta_dir / "csrc" / "kernels"

# Test script location: try workspace aiter-src first, fall back to installed
_workspace_aiter_src = Path("/workspace/aiter-src")
if (_workspace_aiter_src / "op_tests" / "test_fused_qk_norm.py").exists():
    TEST_SCRIPT = _workspace_aiter_src / "op_tests" / "test_fused_qk_norm.py"
    TEST_CWD = _workspace_aiter_src
else:
    # Fallback: look relative to aiter_meta
    TEST_SCRIPT = _aiter_meta_dir.parent / "op_tests" / "test_fused_qk_norm.py"
    TEST_CWD = _aiter_meta_dir.parent


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _report_root() -> Path:
    d = _workspace_root() / "build"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _src_dir() -> Path:
    return _workspace_root() / "src"


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


def _copy_source_to_install() -> None:
    """Copy modified .cu from src/ to aiter install path (overwrite)."""
    src = _src_dir() / SOURCE_FILE
    dst = KERNEL_INSTALL_DIR / SOURCE_FILE
    if not src.exists():
        print(f"ERROR: source file not found: {src}")
        sys.exit(1)
    shutil.copy2(str(src), str(dst))
    print(f"Copied: {src} -> {dst}")


def _delete_jit_cache() -> None:
    """Remove JIT-compiled .so and build directory to force recompilation."""
    so_path = AITER_JIT_DIR / f"{MODULE_NAME}.so"
    build_dir = AITER_JIT_DIR / "build" / MODULE_NAME

    if so_path.exists():
        so_path.unlink()
        print(f"Deleted: {so_path}")
    if build_dir.exists():
        shutil.rmtree(build_dir)
        print(f"Deleted: {build_dir}")


# ── compile ──────────────────────────────────────────────────────────────────

def run_compile() -> None:
    """Copy source, delete JIT cache, trigger recompilation via a small test."""
    _copy_source_to_install()
    _delete_jit_cache()

    ok, out = _run_cmd(
        ["python3", str(TEST_SCRIPT), "-d", "bf16", "-m", "1", "-n1", "1024", "-n2", "512"],
        cwd=TEST_CWD,
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
    """Run correctness tests across multiple shapes."""
    ok, out = _run_cmd(
        ["python3", str(TEST_SCRIPT), "-d", "bf16",
         "-m", "1", "64", "1024", "8192",
         "-n1", "1024", "1536",
         "-n2", "512", "1024"],
        cwd=TEST_CWD,
        timeout_s=600,
    )

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

    Output format from test_fused_qk_norm.py:
      [perf] === dtype:torch.bfloat16, M:1, N1:1024, N2:512 ===
             split_kernel avg: 8.37  us, fused_kernel avg: 6.45  us, uplift: 29.9%
    """
    ok, out = _run_cmd(
        ["python3", str(TEST_SCRIPT), "-d", "bf16",
         "-m", "1", "64", "1024", "8192",
         "-n1", "1024", "1536",
         "-n2", "512", "1024"],
        cwd=TEST_CWD,
        timeout_s=600,
    )

    if not ok:
        (_report_root() / "performance_report.json").write_text("[]")
        print("Performance: FAIL")
        sys.exit(1)

    # Parse: fused_kernel avg: X us
    pattern = re.compile(
        r"\[perf\].*?M:(?P<m>\d+),\s*N1:(?P<n1>\d+),\s*N2:(?P<n2>\d+).*?"
        r"fused_kernel avg:\s*(?P<fused>[\d.]+)\s*us"
    )

    results = []
    seen = set()
    for match in pattern.finditer(out):
        m_val = int(match.group("m"))
        n1_val = int(match.group("n1"))
        n2_val = int(match.group("n2"))
        fused_us = float(match.group("fused"))
        key = (m_val, n1_val, n2_val)
        if key in seen:
            continue
        seen.add(key)
        results.append({
            "test_case_id": f"fused_qk_norm_m{m_val}_n1_{n1_val}_n2_{n2_val}",
            "shape": [m_val, n1_val, n2_val],
            "execution_time_ms": fused_us / 1000.0,
            "metadata": {
                "m": m_val,
                "n1": n1_val,
                "n2": n2_val,
                "kernel": "fused_qk_rmsnorm_kernel",
                "unit_original": "us",
                "value_us": fused_us,
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
