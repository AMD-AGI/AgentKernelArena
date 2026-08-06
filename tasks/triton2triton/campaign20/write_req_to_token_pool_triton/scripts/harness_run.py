#!/usr/bin/env python3
"""Real launcher + benchmark for the compile-only @triton.jit kernel
``write_req_to_token_pool_triton`` (sglang.srt.mem_cache.common).

WORKLOAD REGIME: input seqlen = output seqlen = 1024; concurrency B in {2,32,64}.
This is a token-parallel KV-write op. grid = (B,) (one program per request).
We run it prefill-style: each request writes seq_len=1024 tokens into its
req_to_token row. We use a small prefix (pre_len=PREFIX_LEN) so BOTH the
prefix-copy loop and the extend-copy loop are exercised, and provide REAL
device pointers for ``prefix_tensors`` so the int64* dereference is valid.

shape_mapping:
  B in {2,32,64} -> grid=(B,), one program per request (token-parallel).
  seq_len = 1024 per request (prefill: input==output==1024).
  pre_len = PREFIX_LEN (small captured-style prefix), extend_len = seq_len - pre_len.
  out_cache_loc total = sum(extend_lens) = B * (1024 - PREFIX_LEN).
  Model dims kept from captured base case:
    req_to_token_ptr columns (max_context_len) = 202756, stride = 202756.
    req_to_token_ptr rows (max_batch) = 2048.
"""
import importlib.util
import json
import os
import sys

import torch

TASK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_FILE = os.path.join(TASK_DIR, "source", "triton_write_req_to_token_pool_triton.py")
GOLDEN_FILE = os.path.join(TASK_DIR, "source_golden", "triton_write_req_to_token_pool_triton.py")
KERNEL_NAME = "write_req_to_token_pool_triton"

# captured model dims
MAX_BATCH = 2048
MAX_CONTEXT_LEN = 202756
STRIDE = 202756
SEQ_LEN = 1024
PREFIX_LEN = 16  # small prefix so both loops run; pointer deref is exercised

CASES = [("c2", 2), ("c32", 32), ("c64", 64)]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, KERNEL_NAME)


def build_inputs(B, seed=42):
    """Deterministic, seeded inputs for one concurrency case.

    Returns (args_tuple, grid, keep_alive) where keep_alive holds python refs
    to the per-request prefix tensors so their data_ptr() stay valid during
    the launch.
    """
    g = torch.Generator(device="cuda").manual_seed(seed + B)

    # output buffer (the thing the kernel writes into) -- mutated in place
    req_to_token = torch.zeros((B, MAX_CONTEXT_LEN), dtype=torch.int32, device="cuda")

    # which row each request writes to: a unique permutation of rows
    req_pool_indices = torch.randperm(B, generator=g, device="cuda").to(torch.int64)

    pre_lens = torch.full((B,), PREFIX_LEN, dtype=torch.int64, device="cuda")
    seq_lens = torch.full((B,), SEQ_LEN, dtype=torch.int64, device="cuda")
    extend_lens = (seq_lens - pre_lens).to(torch.int64)  # all == SEQ_LEN-PREFIX_LEN

    total_extend = int(extend_lens.sum().item())
    out_cache_loc = torch.randint(
        0, MAX_CONTEXT_LEN, (total_extend,), dtype=torch.int64, device="cuda", generator=g
    )

    # prefix_tensors: a uint64 array holding device pointers to per-request
    # int64 prefix buffers of length pre_len. Keep python refs alive.
    keep_alive = []
    ptrs = torch.empty((B,), dtype=torch.uint64, device="cuda")
    ptr_vals = []
    for i in range(B):
        buf = torch.randint(
            0, MAX_CONTEXT_LEN, (PREFIX_LEN,), dtype=torch.int64, device="cuda", generator=g
        )
        keep_alive.append(buf)
        ptr_vals.append(buf.data_ptr())
    ptrs.copy_(torch.tensor(ptr_vals, dtype=torch.uint64, device="cuda"))

    args = (
        req_to_token,
        req_pool_indices,
        ptrs,           # prefix_tensors
        pre_lens,
        seq_lens,
        extend_lens,
        out_cache_loc,
        STRIDE,         # req_to_token_ptr_stride (constexpr int)
    )
    grid = (B,)
    return args, grid, keep_alive


def reference_output(B, seed=42):
    """Pure-torch golden of what the kernel should write into req_to_token,
    computed from the SAME seeded inputs. Returns the mutated req_to_token."""
    args, grid, keep = build_inputs(B, seed=seed)
    req_to_token, req_pool_indices, ptrs, pre_lens, seq_lens, extend_lens, out_cache_loc, stride = args
    out = torch.zeros_like(req_to_token)
    cumsum = 0
    for pid in range(B):
        row = int(req_pool_indices[pid].item())
        pl = int(pre_lens[pid].item())
        sl = int(seq_lens[pid].item())
        # prefix region
        prefix = keep[pid][:pl].to(torch.int32)
        out[row, 0:pl] = prefix
        # extend region
        n = sl - pl
        seg = out_cache_loc[cumsum:cumsum + n].to(torch.int32)
        out[row, pl:sl] = seg
        cumsum += int(extend_lens[pid].item())
    return out


def run_kernel(kern, B, seed=42):
    args, grid, keep = build_inputs(B, seed=seed)
    kern[grid](*args)
    torch.cuda.synchronize()
    return args[0]  # req_to_token (mutated in place)


def correctness():
    kern_edit = _load(SOURCE_FILE, "k_edit")
    kern_gold = _load(GOLDEN_FILE, "k_gold")
    for cid, B in CASES:
        got = run_kernel(kern_edit, B, seed=42)
        # golden run with identical inputs (same seed -> same tensors)
        gold = run_kernel(kern_gold, B, seed=42)
        ref = reference_output(B, seed=42)
        # integer/index kernel: require EXACT match
        if not torch.equal(got, gold):
            return False, f"{cid}: edited vs golden mismatch ({(got != gold).sum().item()} elems)"
        if not torch.equal(got, ref):
            return False, f"{cid}: edited vs torch-reference mismatch ({(got != ref).sum().item()} elems)"
    return True, None


def compile_smoke():
    """JIT-compile and launch the smallest workload case."""
    run_kernel(_load(SOURCE_FILE, "k_compile"), CASES[0][1], seed=42)


def performance(benchmark):
    kern = _load(SOURCE_FILE, "k_perf")
    results = []
    for cid, B in CASES:
        args, grid, keep = build_inputs(B, seed=42)
        avg, metadata = benchmark(
            lambda: kern[grid](*args),
            warmup=10,
            repetition=100,
        )
        results.append({
            "test_case_id": cid,
            "execution_time_ms": avg,
            **metadata,
            "params": {"B": B, "seq_len": SEQ_LEN, "pre_len": PREFIX_LEN, "grid": list(grid)},
        })
    return results


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "correctness"
    if mode == "correctness":
        ok, err = correctness()
        print("CORRECTNESS", "PASS" if ok else "FAIL", err or "")
    elif mode == "performance":
        raise SystemExit("run performance through scripts/task_runner.py")
