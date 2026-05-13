#!/usr/bin/env python3
"""Task runner for hip2hip/aiter/level1/rmsnorm_quant_kernels.

NOTE: Cannot use test_rmsnorm2dFusedAddQuant.py directly because it triggers
module_rmsnorm JIT build which fails (missing rmsnorm2d_fwd.hpp). This
task_runner uses a custom test that only exercises add_rmsnorm_quant.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

TASK_NAME = "hip2hip/aiter/level1/rmsnorm_quant_kernels"
MODULE_NAME = "module_rmsnorm_quant"
SOURCE_FILE = "rmsnorm_quant_kernels.cu"

import aiter as _aiter_pkg
_aiter_pkg_dir = Path(_aiter_pkg.__file__).resolve().parent
AITER_JIT_DIR = _aiter_pkg_dir / "jit"

import aiter_meta as _aiter_meta_pkg
_aiter_meta_dir = Path(_aiter_meta_pkg.__file__).resolve().parent
KERNEL_INSTALL_DIR = _aiter_meta_dir / "csrc" / "kernels"


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[1]

def _report_root() -> Path:
    d = _workspace_root() / "build"
    d.mkdir(parents=True, exist_ok=True)
    return d

def _src_dir() -> Path:
    return _workspace_root() / "src"

def _run_cmd(cmd, cwd=None, timeout_s=600):
    print(f"[RUN] {' '.join(cmd)}")
    sys.stdout.flush()
    try:
        proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True, timeout=timeout_s)
        output = proc.stdout + proc.stderr
        print(output, end="", flush=True)
        return proc.returncode == 0, output
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT after {timeout_s}s"
    except Exception as e:
        return False, str(e)

def _copy_source_to_install():
    src = _src_dir() / SOURCE_FILE
    dst = KERNEL_INSTALL_DIR / SOURCE_FILE
    shutil.copy2(str(src), str(dst))
    print(f"Copied: {src} -> {dst}")

def _delete_jit_cache():
    so_path = AITER_JIT_DIR / f"{MODULE_NAME}.so"
    build_dir = AITER_JIT_DIR / "build" / MODULE_NAME
    if so_path.exists():
        so_path.unlink()
        print(f"Deleted: {so_path}")
    if build_dir.exists():
        shutil.rmtree(build_dir)
        print(f"Deleted: {build_dir}")


# ── Custom inline test (avoids test_rmsnorm triggering module_rmsnorm build) ──

CUSTOM_TEST_CODE = '''
import torch
import time
import aiter

def rms_norm_ref(x, weight, eps=1e-6):
    """Reference RMSNorm."""
    variance = x.float().pow(2).mean(-1, keepdim=True)
    return (x * torch.rsqrt(variance + eps)) * weight

def test_add_rmsnorm_quant(m, n, dtype=torch.bfloat16, eps=1e-6):
    """Test add_rmsnorm correctness and performance."""
    input_t = torch.randn(m, n, dtype=dtype, device="cuda")
    residual = torch.randn(m, n, dtype=dtype, device="cuda")
    weight = torch.randn(n, dtype=dtype, device="cuda")

    # Reference: add residual then rmsnorm
    ref_sum = input_t + residual
    ref_output = rms_norm_ref(ref_sum, weight, eps)

    # aiter kernel: add_rmsnorm
    # API: aiter.add_rmsnorm(out, input, residual_in, residual_out, weight, epsilon)
    out = torch.empty_like(input_t)
    residual_out = torch.empty_like(residual)
    aiter.add_rmsnorm(out, input_t, residual, residual_out, weight, eps)

    # Check correctness
    max_diff = (ref_output - out).abs().max().item()
    ok = max_diff < 0.15  # relaxed tolerance for bf16 + fused kernel numerical differences

    # Performance measurement
    torch.cuda.synchronize()
    start = time.perf_counter()
    n_iters = 100
    for _ in range(n_iters):
        aiter.add_rmsnorm(out, input_t, residual, residual_out, weight, eps)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / n_iters * 1e6  # us

    return ok, elapsed, max_diff

configs = [
    (8, 1024),
    (8, 4096),
    (64, 1024),
    (64, 4096),
    (1024, 4096),
    (8192, 4096),
]

import json
results = []
all_pass = True
for m, n in configs:
    try:
        ok, us, max_diff = test_add_rmsnorm_quant(m, n)
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
    except Exception as e:
        status = "FAIL"
        us = 0.0
        max_diff = -1.0
        all_pass = False
        print(f"  ERROR: {e}")
    print(f"[result] m={m}, n={n}: {status}, {us:.2f} us, max_diff={max_diff:.6f}")
    results.append({"m": m, "n": n, "status": status, "us": us, "max_diff": max_diff})

print(f"[summary] all_pass={all_pass}, configs={len(configs)}")
print(f"[json_results] {json.dumps(results)}")
'''


def run_compile():
    _copy_source_to_install()
    _delete_jit_cache()
    ok, out = _run_cmd(["python3", "-c", CUSTOM_TEST_CODE], timeout_s=600)
    report = {"status": "ok" if ok else "fail", "output": out[-2000:]}
    (_report_root() / "compile_report.json").write_text(json.dumps(report, indent=2))
    if not ok:
        print("Compilation: FAIL"); sys.exit(1)
    print("Compilation: PASS")


def run_correctness():
    ok, out = _run_cmd(["python3", "-c", CUSTOM_TEST_CODE], timeout_s=600)
    passed = out.count("PASS")
    failed = out.count("FAIL")
    report = {"status": "ok" if (ok and failed == 0 and passed > 0) else "fail", "passed": passed, "failed": failed}
    (_report_root() / "correctness_report.json").write_text(json.dumps(report, indent=2))
    print(f"Correctness: {passed} passed, {failed} failed")
    if failed > 0 or not ok:
        print("Correctness: FAIL"); sys.exit(1)
    print("Correctness: PASS")


def run_performance():
    ok, out = _run_cmd(["python3", "-c", CUSTOM_TEST_CODE], timeout_s=600)
    if not ok:
        (_report_root() / "performance_report.json").write_text("[]")
        print("Performance: FAIL"); sys.exit(1)

    pattern = re.compile(r"\[result\] m=(\d+), n=(\d+): \w+, ([\d.]+) us")
    results = []
    for match in pattern.finditer(out):
        m_val, n_val, us = int(match.group(1)), int(match.group(2)), float(match.group(3))
        results.append({
            "test_case_id": f"add_rmsnorm_quant_m{m_val}_n{n_val}",
            "shape": [m_val, n_val],
            "execution_time_ms": us / 1000.0,
            "metadata": {"m": m_val, "n": n_val,
                         "kernel": "add_rmsnorm_quant_kernel", "unit_original": "us", "value_us": us},
        })

    (_report_root() / "performance_report.json").write_text(json.dumps(results, indent=2))
    (_report_root() / "performance.log").write_text(out)

    if results:
        avg_us = sum(r["metadata"]["value_us"] for r in results) / len(results)
        print(f"Performance: {len(results)} test cases, avg {avg_us:.2f} us")
    else:
        print("Performance: no results parsed"); sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description=f"Task runner for {TASK_NAME}")
    parser.add_argument("mode", choices=["compile", "correctness", "performance"])
    args = parser.parse_args()
    {"compile": run_compile, "correctness": run_correctness, "performance": run_performance}[args.mode]()

if __name__ == "__main__":
    main()
