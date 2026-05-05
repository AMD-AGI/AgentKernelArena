#!/usr/bin/env python3
"""
Test harness for the persistent lean attention + paged attention Triton kernel.

Modes: --correctness, --profile, --benchmark, --full-benchmark

Per-task divergence from the standard dispatch convention: --correctness
uses CORRECTNESS_CONFIGS (a small fixed correctness-focused set), not
HARNESS_CONFIGS. CORRECTNESS_CONFIGS and ALL_CONFIGS are different sets in
this task and the project plan calls for preserving the kernel.py-side
convention here.
"""
import argparse
import math
import os
import sys
import torch

from kernel import (
    persistent_lean_attention_paged,
    _make_test_case, _config_tag,
    CORRECTNESS_CONFIGS, ALL_CONFIGS, HARNESS_CONFIGS, PROFILE_CONFIGS,
    RTOL, ATOL,
)


# ============================================================================
# SHAPE SUBSETS
# ============================================================================

# kernel.py already pre-samples HARNESS_CONFIGS (25) and PROFILE_CONFIGS (5)
# from ALL_CONFIGS, so we just re-export under the standard names here.
ALL_SHAPES = ALL_CONFIGS
HARNESS_SHAPES = HARNESS_CONFIGS
PROFILE_SHAPES = PROFILE_CONFIGS


# ============================================================================
# PYTORCH REFERENCE (moved from kernel.py; correctness-only)
# ============================================================================

def torch_op(q, k, v, ref_indices, n_ctx_q, sm_scale):
    ref_out = torch.empty_like(q, dtype=v.dtype)
    for head_idx in range(q.shape[0]):
        start_q = 0
        for batch_idx in range(len(ref_indices[head_idx])):
            qb = q[head_idx, start_q : start_q + n_ctx_q, :]
            idxs = ref_indices[head_idx][batch_idx]
            kb = torch.index_select(k[head_idx], dim=0, index=idxs)
            vb = torch.index_select(v[head_idx], dim=0, index=idxs)
            p = torch.matmul(qb, kb.transpose(0, 1)) * sm_scale
            p = torch.softmax(p.float(), dim=-1).to(q.dtype)
            ref_out[head_idx, start_q : start_q + n_ctx_q, :] = torch.matmul(p, vb)
            start_q += n_ctx_q
    return ref_out


# ============================================================================
# Helpers
# ============================================================================

def _call_triton(case, cfg):
    batch, h, n_ctx_q, n_ctx, d, total_programs, dtype, block_m, block_n, waves_per_eu, num_warps = cfg
    return persistent_lean_attention_paged(
        q=case["q"], k=case["k"], v=case["v"],
        kv_block_tables=case["kv_block_tables"],
        Mp=case["Mp"], Lp=case["Lp"], Op=case["Op"], locks=case["locks"],
        batch_num_block_n=case["batch_num_block_n"],
        total_programs=total_programs,
        BLOCK_M=block_m, BLOCK_N=block_n,
        batch_size=batch,
        sm_scale=case["sm_scale"],
        num_warps=case["num_warps"],
        waves_per_eu=case["waves_per_eu"],
    )


# ============================================================================
# TEST HARNESS
# ============================================================================

def run_correctness(shapes=None, verbose=True):
    if shapes is None:
        shapes = CORRECTNESS_CONFIGS
    if verbose:
        print(f"Running correctness on {len(shapes)} shapes...")

    results, failures = [], []

    for cfg in shapes:
        batch, h, n_ctx_q, n_ctx, d, total_programs, dtype, block_m, block_n, waves_per_eu, num_warps = cfg
        tag = _config_tag(batch, h, n_ctx_q, n_ctx, d, total_programs, block_m, block_n, waves_per_eu, num_warps)
        try:
            case = _make_test_case(*cfg)
            out_triton = _call_triton(case, cfg)
            out_torch = torch_op(case["q"], case["k"], case["v"],
                                 case["ref_indices"], n_ctx_q, case["sm_scale"])
            torch.cuda.synchronize()

            torch.testing.assert_close(out_torch, out_triton, atol=ATOL, rtol=RTOL)
            results.append({"config": tag, "correct": True})
            if verbose:
                print(f"  PASS: {tag}")
        except Exception as exc:
            failures.append({"config": tag, "error": str(exc)})
            if verbose:
                print(f"  FAIL: {tag} - {str(exc)[:120]}")
        torch.cuda.empty_cache()

    if verbose:
        print("-" * 70)
        status = "ALL PASS" if not failures else f"FAILED ({len(failures)}/{len(shapes)})"
        print(f"{'Status:':<22} {status}")

    return {
        "correct": len(failures) == 0,
        "num_correct": len(results),
        "num_failed": len(failures),
        "failures": failures,
        "results": results,
    }


def run_profile(shapes=None, warmup=50, iters=200, verbose=True):
    if shapes is None:
        shapes = PROFILE_SHAPES
    if verbose:
        print(f"Profile: {len(shapes)} config(s), {warmup} warmup, {iters} iter(s)")

    for cfg in shapes:
        case = _make_test_case(*cfg)
        for _ in range(warmup):
            _call_triton(case, cfg)
        torch.cuda.synchronize()
        for _ in range(iters):
            _call_triton(case, cfg)
        torch.cuda.synchronize()
        if verbose:
            batch, h, n_ctx_q, n_ctx, d, total_programs, dtype, block_m, block_n, waves_per_eu, num_warps = cfg
            tag = _config_tag(batch, h, n_ctx_q, n_ctx, d, total_programs, block_m, block_n, waves_per_eu, num_warps)
            print(f"  {tag} done")
        torch.cuda.empty_cache()


def run_benchmark(shapes=None, warmup=50, iters=200, verbose=True):
    if shapes is None:
        shapes = HARNESS_SHAPES

    latencies = []

    print(f"Running benchmark on {len(shapes)} shapes, {warmup} warmup, {iters} iterations each...")
    if verbose:
        print(f"{'Config':<72} {'Triton':>10}")
        print("-" * 84)

    for cfg in shapes:
        batch, h, n_ctx_q, n_ctx, d, total_programs, dtype, block_m, block_n, waves_per_eu, num_warps = cfg
        tag = _config_tag(batch, h, n_ctx_q, n_ctx, d, total_programs, block_m, block_n, waves_per_eu, num_warps)
        case = _make_test_case(*cfg)

        for _ in range(warmup):
            _call_triton(case, cfg)
        torch.cuda.synchronize()

        triton_times = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _call_triton(case, cfg)
            end.record()
            torch.cuda.synchronize()
            triton_times.append(start.elapsed_time(end))

        triton_ms = sorted(triton_times)[len(triton_times) // 2]
        latencies.append(triton_ms)

        if verbose:
            print(f"{tag:<72} {triton_ms:>8.4f}ms", flush=True)

        torch.cuda.empty_cache()

    geomean_latency = math.exp(sum(math.log(l) for l in latencies) / len(latencies))

    print("-" * 84)
    print(f"{'Geometric mean latency:':<72} {geomean_latency:.4f} ms")
    print(f"GEAK_SHAPES_USED={list(range(len(shapes)))}")
    print(f"GEAK_RESULT_LATENCY_MS={geomean_latency:.4f}", flush=True)

    return {"geomean_latency_ms": geomean_latency, "latencies": latencies}


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lean Attention (Paged) Test Harness")
    parser.add_argument("--correctness", action="store_true",
                        help="Run correctness tests on CORRECTNESS_CONFIGS")
    parser.add_argument("--profile", action="store_true",
                        help="Run minimal profiling workload")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run benchmark on HARNESS_SHAPES (25 uniformly sampled)")
    parser.add_argument("--full-benchmark", action="store_true",
                        help="Run benchmark on ALL_SHAPES (complete set)")
    parser.add_argument("--warmup", type=int, default=50,
                        help="Number of warmup iterations (default: 50)")
    parser.add_argument("--iterations", type=int,
                        default=int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "200")),
                        help="Number of benchmark iterations (default: GEAK_BENCHMARK_ITERATIONS or 200)")
    args = parser.parse_args()

    print("=" * 70)
    print("Lean Attention (Paged) Test Harness")
    print("=" * 70)

    if args.correctness:
        print("\n[Correctness Mode]")
        result = run_correctness(CORRECTNESS_CONFIGS)
        sys.exit(0 if result["correct"] else 1)
    elif args.profile:
        print("\n[Profile Mode]")
        run_profile(PROFILE_SHAPES, warmup=args.warmup, iters=args.iterations)
    elif args.full_benchmark:
        print("\n[Full Benchmark Mode]")
        run_benchmark(ALL_SHAPES, warmup=args.warmup, iters=args.iterations)
    else:
        print("\n[Benchmark Mode]")
        run_benchmark(HARNESS_SHAPES, warmup=args.warmup, iters=args.iterations)

    print("=" * 70)
