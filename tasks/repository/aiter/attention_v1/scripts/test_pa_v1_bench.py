#!/usr/bin/env python3
"""Standalone paged attention v1 benchmark for AgentKernelArena.

Tests attention_v1.cu by manually loading the JIT module `module_pa_v1` and
calling its `paged_attention_v1` function directly. This bypasses the default
`cpp_itfs` template path that aiter's Python API uses.

Self-contained: does not depend on aiter's op_tests/test_pa_v1.py.
Only depends on the installed aiter package (import aiter).

Usage:
  python3 test_pa_v1_bench.py --mode correctness
  python3 test_pa_v1_bench.py --mode performance
  python3 test_pa_v1_bench.py --mode correctness --ns 1 --nh 16 --nkv 16 --ctx 64
"""
from __future__ import annotations

import argparse
import sys
import torch

# Fixed constants
HEAD_SIZE = 128
BLOCK_SIZE = 16
PARTITION_SIZE = 256  # only supported value in attention_v1.cu
DTYPE = torch.float16


def _load_pa_v1():
    """Load the JIT module and return the paged_attention_v1 function.

    Since module_pa_v1 has no @compile_ops registration, we cannot use
    get_module() directly (it won't auto-trigger compilation). Instead,
    we check if the .so exists and trigger build_module() manually if needed.
    """
    import importlib
    from pathlib import Path
    import aiter as _aiter_pkg
    from aiter.jit.core import get_args_of_build, build_module

    jit_dir = Path(_aiter_pkg.__file__).parent / "jit"
    so_path = jit_dir / "module_pa_v1.so"
    md_name = "module_pa_v1"

    if not so_path.exists():
        print(f"[pa_v1] JIT module not found, triggering build...")
        d_args = get_args_of_build(md_name)
        build_module(
            md_name,
            d_args["srcs"], d_args["flags_extra_cc"], d_args["flags_extra_hip"],
            d_args["blob_gen_cmd"], d_args["extra_include"], d_args["extra_ldflags"],
            d_args["verbose"], d_args["is_python_module"], d_args["is_standalone"],
            d_args["torch_exclude"],
            # NOTE: aiter d098ae5 added `third_party` as 12th positional arg
            # to build_module(); pass-through with safe default for back-compat.
            d_args.get("third_party", []),
        )

    mod = importlib.import_module(f"aiter.jit.{md_name}")
    return mod.paged_attention_v1


def _create_inputs(num_seqs: int, num_heads: int, num_kv_heads: int,
                   context_len: int, device: str = "cuda"):
    """Create all input tensors for one paged attention v1 test case."""
    num_blocks = (context_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    max_num_partitions = (context_len + PARTITION_SIZE - 1) // PARTITION_SIZE

    # workspace_buffer layout (from attention_v1.cu lines 283-286):
    #   exp_sums:   [num_seqs, num_heads, max_num_partitions] float32
    #   max_logits: [num_seqs, num_heads, max_num_partitions] float32
    #   tmp_out:    [num_seqs, num_heads, max_num_partitions, HEAD_SIZE] fp16
    wb_bytes = (num_seqs * num_heads * max_num_partitions * 4  # exp_sums
                + num_seqs * num_heads * max_num_partitions * 4  # max_logits
                + num_seqs * num_heads * max_num_partitions * HEAD_SIZE * 2)  # tmp_out
    workspace_buffer = torch.zeros(wb_bytes, dtype=torch.uint8, device=device)

    out = torch.empty(num_seqs, num_heads, HEAD_SIZE, dtype=DTYPE, device=device)
    query = torch.randn(num_seqs, num_heads, HEAD_SIZE, dtype=DTYPE, device=device)

    # KV cache in HND layout: [num_blocks, num_kv_heads, block_size, head_size]
    key_cache = torch.randn(num_blocks, num_kv_heads, BLOCK_SIZE, HEAD_SIZE,
                            dtype=DTYPE, device=device)
    value_cache = torch.randn(num_blocks, num_kv_heads, BLOCK_SIZE, HEAD_SIZE,
                              dtype=DTYPE, device=device)

    scale = 1.0 / (HEAD_SIZE ** 0.5)

    # block_tables: each seq uses consecutive blocks
    block_tables = torch.zeros(num_seqs, num_blocks, dtype=torch.int32, device=device)
    for i in range(num_blocks):
        block_tables[:, i] = i  # all seqs share the same blocks (ok for test)

    # cu_query_lens: MUST be provided (reduce kernel dereferences without NULL check)
    # decode scenario: each seq has query length = 1
    cu_query_lens = torch.arange(num_seqs + 1, dtype=torch.int32, device=device)

    context_lens = torch.full((num_seqs,), context_len, dtype=torch.int32, device=device)

    k_scale = torch.ones(1, dtype=torch.float32, device=device)
    v_scale = torch.ones(1, dtype=torch.float32, device=device)

    return {
        "out": out,
        "workspace_buffer": workspace_buffer,
        "query": query,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "scale": scale,
        "block_tables": block_tables,
        "cu_query_lens": cu_query_lens,
        "context_lens": context_lens,
        "max_context_len": context_len,
        "k_scale": k_scale,
        "v_scale": v_scale,
        "num_seqs": num_seqs,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "num_blocks": num_blocks,
    }


def _run_kernel(pa_v1_func, d: dict):
    """Call paged_attention_v1 from the JIT module."""
    pa_v1_func(
        d["out"], d["workspace_buffer"],
        d["query"], d["key_cache"], d["value_cache"],
        d["scale"], d["block_tables"],
        d["cu_query_lens"],  # must not be None
        d["context_lens"], d["max_context_len"],
        None,       # alibi_slopes
        "auto",     # kv_cache_dtype
        "HND",      # kv_cache_layout
        0.0,        # logits_soft_cap
        d["k_scale"], d["v_scale"],
        None,       # fp8_out_scale
        PARTITION_SIZE,
    )


def _reference_attention(d: dict) -> torch.Tensor:
    """Compute reference paged attention output using PyTorch.

    Reconstructs full K, V from paged cache, applies GQA expansion,
    and computes standard scaled dot-product attention.
    """
    query = d["query"]
    key_cache = d["key_cache"]
    value_cache = d["value_cache"]
    num_seqs = d["num_seqs"]
    num_heads = d["num_heads"]
    num_kv_heads = d["num_kv_heads"]
    context_len = d["max_context_len"]
    scale = d["scale"]
    num_blocks = d["num_blocks"]

    # Reconstruct full K, V from paged cache (HND layout)
    # key_cache: [num_blocks, num_kv_heads, block_size, head_size]
    k_full = key_cache.permute(1, 0, 2, 3).reshape(
        num_kv_heads, num_blocks * BLOCK_SIZE, HEAD_SIZE)[:, :context_len, :]
    v_full = value_cache.permute(1, 0, 2, 3).reshape(
        num_kv_heads, num_blocks * BLOCK_SIZE, HEAD_SIZE)[:, :context_len, :]

    # GQA expansion: num_kv_heads -> num_heads
    gqa_ratio = num_heads // num_kv_heads
    k_exp = k_full.repeat_interleave(gqa_ratio, dim=0)  # [num_heads, ctx, head_size]
    v_exp = v_full.repeat_interleave(gqa_ratio, dim=0)

    outputs = []
    for s in range(num_seqs):
        q_s = query[s].float().unsqueeze(1)  # [num_heads, 1, head_size]
        scores = torch.matmul(q_s, k_exp.float().transpose(1, 2)) * scale
        weights = torch.softmax(scores, dim=-1)
        out_s = torch.matmul(weights, v_exp.float()).squeeze(1)  # [num_heads, head_size]
        outputs.append(out_s.to(DTYPE))

    return torch.stack(outputs)  # [num_seqs, num_heads, head_size]


def _valid_config(nh, nkv):
    """Check if num_heads / num_kv_heads is a valid integer ratio."""
    return nh >= nkv and nh % nkv == 0


def run_correctness(pa_v1_func, ns_list, nh_list, nkv_list, ctx_list):
    """Run correctness tests and print results."""
    passed = failed = 0
    for nh in nh_list:
        for nkv in nkv_list:
            if not _valid_config(nh, nkv):
                continue
            for ns in ns_list:
                for ctx in ctx_list:
                    tag = f"ns={ns} nh={nh} nkv={nkv} ctx={ctx}"
                    try:
                        d = _create_inputs(ns, nh, nkv, ctx)
                        _run_kernel(pa_v1_func, d)
                        torch.cuda.synchronize()
                        ref = _reference_attention(d)
                        max_err = (d["out"].float() - ref.float()).abs().max().item()
                        ok = max_err < 0.02
                        if ok:
                            print(f"[RESULT] {tag} status=PASS err={max_err:.6f}")
                            passed += 1
                        else:
                            print(f"[RESULT] {tag} status=FAIL err={max_err:.6f}")
                            failed += 1
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            print(f"[RESULT] {tag} status=SKIP (OOM)")
                        else:
                            print(f"[RESULT] {tag} status=FAIL error={e}")
                            failed += 1
                    finally:
                        torch.cuda.empty_cache()

    print(f"\nSUMMARY: {passed} passed, {failed} failed")
    return failed == 0


def run_performance(pa_v1_func, ns_list, nh_list, nkv_list, ctx_list,
                    num_warmup=5, num_iters=20):
    """Run performance benchmarks and print results."""
    results = []
    for nh in nh_list:
        for nkv in nkv_list:
            if not _valid_config(nh, nkv):
                continue
            for ns in ns_list:
                for ctx in ctx_list:
                    tag = f"ns={ns} nh={nh} nkv={nkv} ctx={ctx}"
                    try:
                        d = _create_inputs(ns, nh, nkv, ctx)
                        # warmup
                        for _ in range(num_warmup):
                            _run_kernel(pa_v1_func, d)
                        torch.cuda.synchronize()

                        # benchmark
                        times_us = []
                        for _ in range(num_iters):
                            start = torch.cuda.Event(enable_timing=True)
                            end = torch.cuda.Event(enable_timing=True)
                            start.record()
                            _run_kernel(pa_v1_func, d)
                            end.record()
                            torch.cuda.synchronize()
                            times_us.append(start.elapsed_time(end) * 1000.0)

                        times_us.sort()
                        median_us = times_us[len(times_us) // 2]
                        print(f"[RESULT] {tag} us={median_us:.2f}")
                        results.append({
                            "ns": ns, "nh": nh, "nkv": nkv, "ctx": ctx,
                            "us": median_us,
                        })
                    except RuntimeError as e:
                        if "out of memory" in str(e).lower():
                            print(f"[RESULT] {tag} us=nan (OOM)")
                        else:
                            print(f"[RESULT] {tag} us=nan error={e}")
                    finally:
                        torch.cuda.empty_cache()

    if results:
        avg = sum(r["us"] for r in results) / len(results)
        print(f"\nSUMMARY: {len(results)} cases, avg {avg:.2f} us")
    return results


def main():
    parser = argparse.ArgumentParser(description="Paged attention v1 benchmark")
    parser.add_argument("--mode", choices=["correctness", "performance"], required=True)
    parser.add_argument("--ns", type=int, nargs="+", default=[1, 4, 16, 32])
    parser.add_argument("--nh", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--nkv", type=int, nargs="+", default=[16, 8])
    parser.add_argument("--ctx", type=int, nargs="+", default=[64, 256, 512])
    args = parser.parse_args()

    pa_v1_func = _load_pa_v1()

    if args.mode == "correctness":
        ok = run_correctness(pa_v1_func, args.ns, args.nh, args.nkv, args.ctx)
        sys.exit(0 if ok else 1)
    else:
        results = run_performance(pa_v1_func, args.ns, args.nh, args.nkv, args.ctx)
        sys.exit(0 if results else 1)


if __name__ == "__main__":
    main()
