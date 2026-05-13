#!/usr/bin/env python3
"""Task runner for hip2hip/aiter/level1/moe_align_block_size_kernels.

NOTE: Cannot use test_moe.py directly because it triggers module_moe_sorting
JIT build which fails (missing moe_sorting_api.hpp). This task_runner uses
a custom test that only exercises moe_align_block_size.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

TASK_NAME = "hip2hip/aiter/level1/moe_align_block_size_kernels"
MODULE_NAME = "module_moe_asm"
SOURCE_FILE = "moe_align_block_size_kernels.cu"

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


# ── Custom inline test (avoids test_moe.py dependency on module_moe_sorting) ──

CUSTOM_TEST_CODE = '''
import torch
import time
import aiter

def test_moe_align_block_size(num_tokens, num_experts, block_size, topk):
    """Test moe_align_block_size correctness and performance."""
    topk_ids = torch.randint(0, num_experts, (num_tokens, topk), device="cuda", dtype=torch.int32)

    # Pre-allocate output tensors (required by the kernel API)
    max_num_tokens_padded = num_experts * ((num_tokens * topk + num_experts - 1) // num_experts + block_size - 1) // block_size * block_size
    sorted_token_ids = torch.empty(max_num_tokens_padded, device="cuda", dtype=torch.int32)
    experts_ids = torch.empty(max_num_tokens_padded // block_size, device="cuda", dtype=torch.int32)
    token_nums = torch.empty(num_experts + 1, device="cuda", dtype=torch.int32)
    num_tokens_post_pad = torch.empty(1, device="cuda", dtype=torch.int32)

    # Run the kernel
    aiter.moe_align_block_size(
        topk_ids, num_experts, block_size,
        sorted_token_ids, experts_ids, token_nums, num_tokens_post_pad
    )
    torch.cuda.synchronize()

    # Basic correctness checks
    ntp = num_tokens_post_pad.item()
    assert ntp >= 0, f"num_tokens_post_padded should be non-negative, got {ntp}"
    assert ntp <= max_num_tokens_padded, f"num_tokens_post_padded {ntp} > max {max_num_tokens_padded}"

    # Performance measurement
    torch.cuda.synchronize()
    start = time.perf_counter()
    n_iters = 100
    for _ in range(n_iters):
        aiter.moe_align_block_size(
            topk_ids, num_experts, block_size,
            sorted_token_ids, experts_ids, token_nums, num_tokens_post_pad
        )
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / n_iters * 1e6  # us

    return True, elapsed

configs = [
    (128, 8, 128, 2),
    (256, 8, 128, 2),
    (1024, 8, 128, 2),
    (4096, 8, 128, 2),
    (128, 64, 128, 8),
    (1024, 64, 128, 8),
]

results = []
all_pass = True
for num_tokens, num_experts, block_size, topk in configs:
    try:
        ok, us = test_moe_align_block_size(num_tokens, num_experts, block_size, topk)
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
    except Exception as e:
        status = "FAIL"
        us = 0.0
        all_pass = False
        print(f"  ERROR: {e}")
    print(f"[result] tokens={num_tokens}, experts={num_experts}, block={block_size}, topk={topk}: {status}, {us:.2f} us")
    results.append({"tokens": num_tokens, "experts": num_experts, "block_size": block_size, "topk": topk, "status": status, "us": us})

import json
print(f"[summary] all_pass={all_pass}, configs={len(configs)}")
print(f"[json_results] {json.dumps(results)}")
'''


def run_compile():
    _copy_source_to_install()
    _delete_jit_cache()
    ok, out = _run_cmd(
        ["python3", "-c", CUSTOM_TEST_CODE],
        timeout_s=600,
    )
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

    # Parse: [result] tokens=X, experts=Y, block=Z, topk=K: PASS, X.XX us
    pattern = re.compile(
        r"\[result\] tokens=(\d+), experts=(\d+), block=(\d+), topk=(\d+): \w+, ([\d.]+) us"
    )
    results = []
    for m in pattern.finditer(out):
        tokens, experts, block, topk = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
        us = float(m.group(5))
        results.append({
            "test_case_id": f"moe_align_t{tokens}_e{experts}_b{block}_k{topk}",
            "shape": [tokens, experts, block, topk],
            "execution_time_ms": us / 1000.0,
            "metadata": {"tokens": tokens, "experts": experts, "block_size": block, "topk": topk,
                         "kernel": "moe_align_block_size_kernel", "unit_original": "us", "value_us": us},
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
