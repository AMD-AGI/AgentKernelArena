#!/usr/bin/env python3
#
"""Task runner for hip2hip/aiter/level2/topk_per_row_kernels."""
from __future__ import annotations
import argparse, json, re, shutil, subprocess, sys
from pathlib import Path

TASK_NAME = "hip2hip/aiter/level2/topk_per_row_kernels"
# This kernel is referenced by TWO modules - need to clear both
MODULE_NAMES = ["module_top_k_per_row", "module_topk_plain"]
SOURCE_FILE = "topk_per_row_kernels.cu"

import aiter as _aiter_pkg
_aiter_pkg_dir = Path(_aiter_pkg.__file__).resolve().parent
AITER_JIT_DIR = _aiter_pkg_dir / "jit"
import aiter_meta as _m; _meta_dir = Path(_m.__file__).resolve().parent
KERNEL_INSTALL_DIR = _meta_dir / "csrc" / "kernels"
_ws = Path("/workspace/aiter-src")
TEST_SCRIPT = _ws / "op_tests" / "test_topk_per_row.py" if (_ws / "op_tests" / "test_topk_per_row.py").exists() else _meta_dir.parent / "op_tests" / "test_topk_per_row.py"
TEST_CWD = _ws if _ws.exists() else _meta_dir.parent

def _workspace_root(): return Path(__file__).resolve().parents[1]
def _report_root():
    d = _workspace_root() / "build"; d.mkdir(parents=True, exist_ok=True); return d
def _src_dir(): return _workspace_root() / "src"

def _run_cmd(cmd, cwd=None, timeout_s=600):
    print(f"[RUN] {' '.join(cmd)}"); sys.stdout.flush()
    try:
        proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True, timeout=timeout_s)
        output = proc.stdout + proc.stderr; print(output, end="", flush=True)
        return proc.returncode == 0, output
    except subprocess.TimeoutExpired: return False, f"TIMEOUT after {timeout_s}s"
    except Exception as e: return False, str(e)

def _copy_source_to_install():
    src = _src_dir() / SOURCE_FILE; dst = KERNEL_INSTALL_DIR / SOURCE_FILE
    shutil.copy2(str(src), str(dst)); print(f"Copied: {src} -> {dst}")

def _delete_jit_cache():
    for mod in MODULE_NAMES:
        so = AITER_JIT_DIR / f"{mod}.so"; bd = AITER_JIT_DIR / "build" / mod
        if so.exists(): so.unlink(); print(f"Deleted: {so}")
        if bd.exists(): shutil.rmtree(bd); print(f"Deleted: {bd}")

def run_compile():
    _copy_source_to_install(); _delete_jit_cache()
    ok, out = _run_cmd(["python3", str(TEST_SCRIPT)], cwd=TEST_CWD, timeout_s=600)
    (_report_root() / "compile_report.json").write_text(json.dumps({"status": "ok" if ok else "fail", "output": out[-2000:]}, indent=2))
    if not ok: print("Compilation: FAIL"); sys.exit(1)
    print("Compilation: PASS")

def run_correctness():
    ok, out = _run_cmd(["python3", str(TEST_SCRIPT)], cwd=TEST_CWD, timeout_s=600)
    passed = len(re.findall(r"passed~|True", out)); failed = len(re.findall(r"(?:failed!)", out))
    if passed == 0 and ok: passed = 1
    (_report_root() / "correctness_report.json").write_text(json.dumps({"status": "ok" if (ok and failed == 0) else "fail", "passed": passed, "failed": failed}, indent=2))
    print(f"Correctness: {passed} passed, {failed} failed")
    if failed > 0 or not ok: print("Correctness: FAIL"); sys.exit(1)
    print("Correctness: PASS")

def run_performance():
    ok, out = _run_cmd(["python3", str(TEST_SCRIPT)], cwd=TEST_CWD, timeout_s=600)
    if not ok: (_report_root() / "performance_report.json").write_text("[]"); print("Performance: FAIL"); sys.exit(1)
    # Parse table rows with hip_us
    pattern = re.compile(r"\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|.*?\|\s*([\d.]+)\s*\|$", re.MULTILINE)
    results = []; seen = set()
    for m in pattern.finditer(out):
        batch, context, topk, hip_us = int(m.group(1)), int(m.group(2)), int(m.group(3)), float(m.group(4))
        key = (batch, context, topk)
        if key in seen: continue
        seen.add(key)
        results.append({"test_case_id": f"topk_per_row_b{batch}_c{context}_k{topk}", "shape": [batch, context, topk],
            "execution_time_ms": hip_us / 1000.0, "metadata": {"batch": batch, "context_len": context, "topk": topk,
            "kernel": "topk_per_row", "unit_original": "us", "value_us": hip_us}})
    (_report_root() / "performance_report.json").write_text(json.dumps(results, indent=2))
    (_report_root() / "performance.log").write_text(out)
    if results:
        avg_us = sum(r["metadata"]["value_us"] for r in results) / len(results)
        print(f"Performance: {len(results)} test cases, avg {avg_us:.2f} us")
    else: print("Performance: no results parsed"); sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description=f"Task runner for {TASK_NAME}")
    parser.add_argument("mode", choices=["compile", "correctness", "performance"])
    args = parser.parse_args()
    {"compile": run_compile, "correctness": run_correctness, "performance": run_performance}[args.mode]()

if __name__ == "__main__": main()
