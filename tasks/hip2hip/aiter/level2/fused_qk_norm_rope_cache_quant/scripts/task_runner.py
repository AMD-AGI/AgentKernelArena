#!/usr/bin/env python3
"""Task runner for hip2hip/aiter/level2/fused_qk_norm_rope_cache_quant."""
from __future__ import annotations
import argparse, json, re, shutil, subprocess, sys
from pathlib import Path

TASK_NAME = "hip2hip/aiter/level2/fused_qk_norm_rope_cache_quant"
MODULE_NAME = "module_fused_qk_norm_rope_cache_quant_shuffle"
SOURCE_FILE = "fused_qk_norm_rope_cache_quant.cu"

import aiter as _aiter_pkg
_aiter_pkg_dir = Path(_aiter_pkg.__file__).resolve().parent
AITER_JIT_DIR = _aiter_pkg_dir / "jit"
import aiter_meta as _m; _meta_dir = Path(_m.__file__).resolve().parent
KERNEL_INSTALL_DIR = _meta_dir / "csrc" / "kernels"
_ws = Path("/workspace/aiter-src")
TEST_SCRIPT = _ws / "op_tests" / "test_fused_qk_norm_rope_cache_quant.py" if (_ws / "op_tests" / "test_fused_qk_norm_rope_cache_quant.py").exists() else _meta_dir.parent / "op_tests" / "test_fused_qk_norm_rope_cache_quant.py"
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
    so = AITER_JIT_DIR / f"{MODULE_NAME}.so"; bd = AITER_JIT_DIR / "build" / MODULE_NAME
    if so.exists(): so.unlink(); print(f"Deleted: {so}")
    if bd.exists(): shutil.rmtree(bd); print(f"Deleted: {bd}")

def run_compile():
    _copy_source_to_install(); _delete_jit_cache()
    # Use minimal params for fast compile test
    ok, out = _run_cmd(["python3", str(TEST_SCRIPT), "-t", "3", "-hd", "4,1", "-hs", "64"], cwd=TEST_CWD, timeout_s=600)
    (_report_root() / "compile_report.json").write_text(json.dumps({"status": "ok" if ok else "fail", "output": out[-2000:]}, indent=2))
    if not ok: print("Compilation: FAIL"); sys.exit(1)
    print("Compilation: PASS")

def run_correctness():
    ok, out = _run_cmd(["python3", str(TEST_SCRIPT), "-t", "3", "128", "-hd", "4,1", "32,4", "-hs", "64", "128"], cwd=TEST_CWD, timeout_s=600)
    passed = out.count("PASS"); failed = out.count("FAIL")
    (_report_root() / "correctness_report.json").write_text(json.dumps({"status": "ok" if (ok and failed == 0 and passed > 0) else "fail", "passed": passed, "failed": failed}, indent=2))
    print(f"Correctness: {passed} passed, {failed} failed")
    if failed > 0 or not ok: print("Correctness: FAIL"); sys.exit(1)
    print("Correctness: PASS")

def run_performance():
    ok, out = _run_cmd(["python3", str(TEST_SCRIPT), "-t", "3", "128", "1024", "-hd", "32,4", "-hs", "128"], cwd=TEST_CWD, timeout_s=600)
    if not ok: (_report_root() / "performance_report.json").write_text("[]"); print("Performance: FAIL"); sys.exit(1)
    # Count PASS entries as performance data points
    pass_count = out.count("PASS")
    results = [{"test_case_id": f"fused_qk_norm_rope_cache_quant_all", "shape": [0],
        "execution_time_ms": 0, "metadata": {"kernel": "fusedQKNormRopeQuantCacheShuffleKernel", "pass_count": pass_count}}]
    (_report_root() / "performance_report.json").write_text(json.dumps(results, indent=2))
    (_report_root() / "performance.log").write_text(out)
    print(f"Performance: {pass_count} test cases passed")

def main():
    parser = argparse.ArgumentParser(description=f"Task runner for {TASK_NAME}")
    parser.add_argument("mode", choices=["compile", "correctness", "performance"])
    args = parser.parse_args()
    {"compile": run_compile, "correctness": run_correctness, "performance": run_performance}[args.mode]()

if __name__ == "__main__": main()
