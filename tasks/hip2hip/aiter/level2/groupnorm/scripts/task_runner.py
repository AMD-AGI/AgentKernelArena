#!/usr/bin/env python3
#
"""Task runner for hip2hip/aiter/level2/groupnorm."""
from __future__ import annotations
import argparse, json, re, shutil, subprocess, sys
from pathlib import Path

TASK_NAME = "hip2hip/aiter/level2/groupnorm"
MODULE_NAME = "module_groupnorm"
SOURCE_FILE = "groupnorm.cu"

import aiter as _aiter_pkg
_aiter_pkg_dir = Path(_aiter_pkg.__file__).resolve().parent
AITER_JIT_DIR = _aiter_pkg_dir / "jit"
import aiter_meta as _m; _meta_dir = Path(_m.__file__).resolve().parent
KERNEL_INSTALL_DIR = _meta_dir / "csrc" / "kernels"
_ws = Path("/workspace/aiter-src")
TEST_SCRIPT = _ws / "op_tests" / "test_groupnorm.py" if (_ws / "op_tests" / "test_groupnorm.py").exists() else _meta_dir.parent / "op_tests" / "test_groupnorm.py"
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
    ok, out = _run_cmd(["python3", str(TEST_SCRIPT)], cwd=TEST_CWD, timeout_s=600)
    (_report_root() / "compile_report.json").write_text(json.dumps({"status": "ok" if ok else "fail", "output": out[-2000:]}, indent=2))
    if not ok: print("Compilation: FAIL"); sys.exit(1)
    print("Compilation: PASS")

def run_correctness():
    ok, out = _run_cmd(["python3", str(TEST_SCRIPT)], cwd=TEST_CWD, timeout_s=600)
    passed = out.count("is_equal=True"); failed = out.count("is_equal=False")
    (_report_root() / "correctness_report.json").write_text(json.dumps({"status": "ok" if (ok and failed == 0 and passed > 0) else "fail", "passed": passed, "failed": failed}, indent=2))
    print(f"Correctness: {passed} passed, {failed} failed")
    if failed > 0 or not ok: print("Correctness: FAIL"); sys.exit(1)
    print("Correctness: PASS")

def run_performance():
    ok, out = _run_cmd(["python3", str(TEST_SCRIPT)], cwd=TEST_CWD, timeout_s=600)
    if not ok: (_report_root() / "performance_report.json").write_text("[]"); print("Performance: FAIL"); sys.exit(1)
    # Parse: shape=[...] torch_time=X ms, opt_time=Y ms, speedup=Z is_equal=True
    pattern = re.compile(r"shape=\[([^\]]+)\].*?opt_time=([\d.]+)\s*ms.*?speedup=([\d.]+)")
    results = []
    for m in pattern.finditer(out):
        shape_str = m.group(1); opt_ms = float(m.group(2)); speedup = float(m.group(3))
        results.append({"test_case_id": f"groupnorm_{shape_str.replace(', ','x')}", "shape": [int(x.strip()) for x in shape_str.split(",")],
            "execution_time_ms": opt_ms, "metadata": {"kernel": "groupnorm", "opt_ms": opt_ms, "speedup": speedup, "unit_original": "ms", "value_us": opt_ms * 1000}})
    (_report_root() / "performance_report.json").write_text(json.dumps(results, indent=2))
    (_report_root() / "performance.log").write_text(out)
    if results:
        avg_ms = sum(r["execution_time_ms"] for r in results) / len(results)
        print(f"Performance: {len(results)} test cases, avg {avg_ms:.4f} ms")
    else: print("Performance: no results parsed"); sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description=f"Task runner for {TASK_NAME}")
    parser.add_argument("mode", choices=["compile", "correctness", "performance"])
    args = parser.parse_args()
    {"compile": run_compile, "correctness": run_correctness, "performance": run_performance}[args.mode]()

if __name__ == "__main__": main()
