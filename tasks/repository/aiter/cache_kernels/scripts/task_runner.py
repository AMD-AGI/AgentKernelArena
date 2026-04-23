#!/usr/bin/env python3
"""Task runner for repository/aiter/cache_kernels.

Adapts the aiter JIT-compiled HIP kernel workflow to the AgentKernelArena
evaluator interface (compile / correctness / performance).

Key paths (aiter pip-installed environment):
  - Kernel source : /opt/venv/lib/python3.12/site-packages/aiter_meta/csrc/kernels/cache_kernels.cu
  - JIT module    : module_cache
  - JIT .so cache : /opt/venv/lib/python3.12/site-packages/aiter/jit/module_cache.so
  - JIT build dir : /opt/venv/lib/python3.12/site-packages/aiter/jit/build/module_cache/
  - Test file     : /workspace/aiter-src/op_tests/test_kvcache.py

Output format (NON-STANDARD text, not a pandas markdown table):
  prefill part: ref vs aiter  16735.00us vs  2487.00us
  decode part:  ref vs aiter   6057.00us vs     5.00us
  finish test ctx_lens=4097 bs=128 num_heads=(8, 1) head_size=128 block_size=16 ...

Note: The test uses small batch args (-b 1 -c 100) to avoid GPU OOM in shared
multi-user environments. The default args (bs=[64,128,257], ctx=[4097,12800])
require ~45+ GiB of GPU memory for cache rotation buffers and fail on busy systems.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

TASK_NAME = "repository/aiter/cache_kernels"
MODULE_NAME = "module_cache"

# aiter pip-installed paths
AITER_JIT_DIR = Path("/opt/venv/lib/python3.12/site-packages/aiter/jit")
AITER_SRC_DIR = Path("/workspace/aiter-src")
TEST_SCRIPT = AITER_SRC_DIR / "op_tests" / "test_kvcache.py"

# Minimum free VRAM (bytes) required to run the test (~2 GiB safety margin
# on top of the ~800 MiB actually needed for 101 x 8 MiB tensor copies).
_MIN_FREE_VRAM = 2 * 1024 ** 3


def _select_gpu() -> None:
    """Set HIP_VISIBLE_DEVICES to the GPU with the most free VRAM if the
    current default GPU (0) does not have enough memory.  This handles
    shared-machine environments where other workloads occupy GPU 0.
    """
    if "HIP_VISIBLE_DEVICES" in os.environ:
        return  # caller already specified a GPU

    try:
        import torch

        num_gpus = torch.cuda.device_count()
        best_gpu = 0
        best_free = torch.cuda.mem_get_info(0)[0]

        for gpu_id in range(1, num_gpus):
            free = torch.cuda.mem_get_info(gpu_id)[0]
            if free > best_free:
                best_free = free
                best_gpu = gpu_id

        if best_gpu != 0:
            print(
                f"[GPU select] GPU {best_gpu} has most free VRAM "
                f"({best_free // 1024**2} MiB); setting HIP_VISIBLE_DEVICES={best_gpu}"
            )
            os.environ["HIP_VISIBLE_DEVICES"] = str(best_gpu)
        else:
            print(
                f"[GPU select] Using default GPU 0 "
                f"({best_free // 1024**2} MiB free)"
            )
    except Exception as e:
        print(f"[GPU select] Could not query GPU memory: {e}; using default")


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

    # Use small args to avoid OOM in shared GPU environments.
    # The perftest decorator in aiter deep-copies all input tensors up to ~101 times
    # for cache-flushing; large default args (bs=64-257, ctx=4097-12800) require
    # 40+ GiB for rotation buffers. bs=1, ctx=100 keeps tensors at ~8 MiB each.
    ok, out = _run_cmd(
        ["python3", str(TEST_SCRIPT), "-b", "1", "-c", "100"],
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
    """Run correctness tests across all KV cache configurations.

    Output format is NON-STANDARD text lines. A test passes when it prints:
      finish test ctx_lens=X bs=X ...
    Failures manifest as exceptions/tracebacks (FAIL, Error, Traceback).
    """
    ok, out = _run_cmd(
        ["python3", str(TEST_SCRIPT), "-b", "1", "-c", "100"],
        cwd=AITER_SRC_DIR,
        timeout_s=600,
    )

    # Count "finish test" occurrences as passed tests
    passed = len(re.findall(r"finish test ", out))
    # Count explicit failures: FAIL, Error, Traceback lines (exclude aiter log lines)
    failed = len(re.findall(r"(?:^|\n)(?:FAIL|Error|Traceback)", out))

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

    Output format from test_kvcache.py (per test configuration):
      prefill part: ref vs aiter  16735.00us vs  2487.00us
      decode part:  ref vs aiter   6057.00us vs     5.00us
      finish test ctx_lens=4097 bs=128 num_heads=(8, 1) head_size=128 block_size=16 DType_KV=torch.bfloat16 DType_KVCache=torch.bfloat16

    We extract the aiter timing (second number) for both prefill and decode parts.
    The test_case_id encodes the configuration from the "finish test" line.
    """
    ok, out = _run_cmd(
        ["python3", str(TEST_SCRIPT), "-b", "1", "-c", "100"],
        cwd=AITER_SRC_DIR,
        timeout_s=600,
    )

    if not ok:
        (_report_root() / "performance_report.json").write_text("[]")
        print("Performance: FAIL")
        sys.exit(1)

    # Parse prefill timing lines:
    # "prefill part: ref vs aiter  16735.00us vs  2487.00us"
    prefill_pattern = re.compile(
        r"prefill part:\s*ref vs aiter\s*([\d.]+)us\s*vs\s*([\d.]+)us"
    )
    # Parse decode timing lines:
    # "decode part: ref vs aiter   6057.00us vs     5.00us"  (may have extra spaces)
    decode_pattern = re.compile(
        r"decode\s+part:\s*ref vs aiter\s*([\d.]+)us\s*vs\s*([\d.]+)us"
    )
    # Parse "finish test" lines for config info
    finish_pattern = re.compile(
        r"finish test\s+ctx_lens=(\d+)\s+bs=(\d+)\s+num_heads=\((\d+),\s*(\d+)\)"
        r"\s+head_size=(\d+)\s+block_size=(\d+)"
        r"\s+DType_KV=(\S+)\s+DType_KVCache=(\S+)"
    )

    prefill_timings = prefill_pattern.findall(out)
    decode_timings = decode_pattern.findall(out)
    finish_matches = list(finish_pattern.finditer(out))

    results = []

    for i, fm in enumerate(finish_matches):
        ctx_lens = fm.group(1)
        bs = fm.group(2)
        qhead = fm.group(3)
        kvhead = fm.group(4)
        head_size = fm.group(5)
        block_size = fm.group(6)
        dtype_kv = fm.group(7).replace("torch.", "")
        dtype_kvcache = fm.group(8).replace("torch.", "")

        config_id = (
            f"ctx{ctx_lens}_bs{bs}_qh{qhead}kh{kvhead}"
            f"_hs{head_size}_blk{block_size}_{dtype_kv}to{dtype_kvcache}"
        )

        # Add prefill result
        if i < len(prefill_timings):
            ref_us, aiter_us = float(prefill_timings[i][0]), float(prefill_timings[i][1])
            results.append({
                "test_case_id": f"{config_id}_prefill",
                "execution_time_ms": aiter_us / 1000.0,
                "metadata": {
                    "ctx_lens": int(ctx_lens),
                    "bs": int(bs),
                    "qhead": int(qhead),
                    "kvhead": int(kvhead),
                    "head_size": int(head_size),
                    "block_size": int(block_size),
                    "dtype_kv": dtype_kv,
                    "dtype_kvcache": dtype_kvcache,
                    "phase": "prefill",
                    "ref_us": ref_us,
                    "aiter_us": aiter_us,
                    "unit_original": "us",
                },
            })

        # Add decode result
        if i < len(decode_timings):
            ref_us, aiter_us = float(decode_timings[i][0]), float(decode_timings[i][1])
            results.append({
                "test_case_id": f"{config_id}_decode",
                "execution_time_ms": aiter_us / 1000.0,
                "metadata": {
                    "ctx_lens": int(ctx_lens),
                    "bs": int(bs),
                    "qhead": int(qhead),
                    "kvhead": int(kvhead),
                    "head_size": int(head_size),
                    "block_size": int(block_size),
                    "dtype_kv": dtype_kv,
                    "dtype_kvcache": dtype_kvcache,
                    "phase": "decode",
                    "ref_us": ref_us,
                    "aiter_us": aiter_us,
                    "unit_original": "us",
                },
            })

    # Fallback: if finish_pattern didn't match but we have timing data, store raw
    if not results and (prefill_timings or decode_timings):
        for j, (ref_us_s, aiter_us_s) in enumerate(prefill_timings):
            results.append({
                "test_case_id": f"prefill_{j}",
                "execution_time_ms": float(aiter_us_s) / 1000.0,
                "metadata": {
                    "phase": "prefill",
                    "ref_us": float(ref_us_s),
                    "aiter_us": float(aiter_us_s),
                    "unit_original": "us",
                },
            })
        for j, (ref_us_s, aiter_us_s) in enumerate(decode_timings):
            results.append({
                "test_case_id": f"decode_{j}",
                "execution_time_ms": float(aiter_us_s) / 1000.0,
                "metadata": {
                    "phase": "decode",
                    "ref_us": float(ref_us_s),
                    "aiter_us": float(aiter_us_s),
                    "unit_original": "us",
                },
            })

    (_report_root() / "performance_report.json").write_text(json.dumps(results, indent=2))

    if results:
        avg_us = sum(r["metadata"]["aiter_us"] for r in results) / len(results)
        print(f"Performance: {len(results)} test cases, avg {avg_us:.2f} us")
        for r in results:
            print(f"  {r['test_case_id']}: {r['metadata']['aiter_us']:.2f} us")
    else:
        print("Performance: no results parsed")
        sys.exit(1)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=f"Task runner for {TASK_NAME}")
    parser.add_argument("mode", choices=["compile", "correctness", "performance"])
    args = parser.parse_args()

    _select_gpu()

    if args.mode == "compile":
        run_compile()
    elif args.mode == "correctness":
        run_correctness()
    else:
        run_performance()


if __name__ == "__main__":
    main()
