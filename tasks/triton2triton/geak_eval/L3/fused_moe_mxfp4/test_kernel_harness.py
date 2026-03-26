#!/usr/bin/env python3
# Test harness for moe_op_mxfp4 kernel
# Shape source: op_tests/triton_tests/moe/test_moe_mx.py

import argparse
import os
import sys
import math

# Resolve repo root
REPO_ROOT = os.environ.get(
    "GEAK_WORK_DIR",
    os.environ.get(
        "GEAK_REPO_ROOT",
        os.path.dirname(os.path.abspath(__file__)),
    ),
)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch

# -- Imports from the repo --
from aiter.ops.triton.moe_op_mxfp4 import fused_moe_mxfp4
from aiter.ops.triton.utils.types import torch_to_triton_dtype
import aiter.ops.triton.utils._triton.arch_info as arch_info

# input_helper builds all tensors needed for the kernel
from op_tests.triton_tests.test_moe_mx import (
    input_helper,
    torch_mxfp4_to_fp32,
)
# Reference implementation for correctness
from op_tests.triton_tests.test_moe import torch_moe_ref

# -- Fixed constants --
WARMUP = 50
ITERATIONS = int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "200"))

# -- Full config list from test_moe_mx.py (ordered exactly as in the file) --
# Each entry: (M, N, K, E, top_k)
ALL_CONFIGS = [
    (64, 64, 128, 8, 2),
    (16, 256, 256, 128, 4),
    (1000, 704, 800, 3, 1),
    (1000, 704, 800, 8, 2),
    (64, 14336, 4096, 8, 2),
    (16, 14336, 128, 8, 2),
    (16, 14336, 4096, 4, 1),
    (1, 14336, 128, 4, 2),
    (3, 14336, 128, 4, 2),
    (16, 14336, 128, 1, 1),
    (64, 7186, 128, 8, 2),
    (64, 3584, 128, 8, 2),
    (64, 1792, 128, 8, 2),
    (64, 64, 128, 8, 2),
    (1, 1024, 16384, 2, 1),
]

# Fixed dtype parameters (the only supported combination in the test)
A_DTYPE_STR = "mxfp4_e2m1"
B_DTYPE_STR = "mxfp4_e2m1"
ROUTED_WEIGHT = False
SWIZZLE_MX = False


def _pick(configs, count):
    if len(configs) <= count:
        return list(range(len(configs)))
    n = len(configs)
    return [round(i * (n - 1) / (count - 1)) for i in range(count)]


def _format_config(cfg):
    M, N, K, E, top_k = cfg
    return "M={} N={} K={} E={} top_k={}".format(M, N, K, E, top_k)


def build_inputs(cfg):
    """Build inputs using the repo's input_helper."""
    M, N, K, E, top_k = cfg
    return input_helper(M, N, K, top_k, E, A_DTYPE_STR, B_DTYPE_STR)


def make_kernel_fn(inputs_tuple):
    """Create a callable that runs fused_moe_mxfp4."""
    (
        a_tri, b_tri, c_tri, c_tri_silu,
        a_scale, b_scale, a_mx_scales, b_mx_scales,
        topk_weights, topk_ids,
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        top_k_out, config,
    ) = inputs_tuple

    def fn():
        fused_moe_mxfp4(
            a_tri, b_tri, c_tri,
            a_scale, b_scale,
            a_mx_scales, b_mx_scales,
            topk_weights, topk_ids,
            sorted_token_ids, expert_ids, num_tokens_post_padded,
            ROUTED_WEIGHT, top_k_out,
            SWIZZLE_MX, SWIZZLE_MX,
            config,
            torch_to_triton_dtype[c_tri.dtype],
        )

    return fn, c_tri


def do_correctness(indices):
    """Run correctness checks on selected configs. Exit non-zero on failure."""
    torch.manual_seed(42)
    fp16_dtype = torch.bfloat16  # mxfp4 uses bf16 as the fp16 dtype

    failures = 0
    for idx in indices:
        cfg = ALL_CONFIGS[idx]
        M, N, K, E, top_k = cfg
        torch.cuda.empty_cache()

        inputs_tuple = build_inputs(cfg)
        (
            a_tri, b_tri, c_tri, c_tri_silu,
            a_scale, b_scale, a_mx_scales, b_mx_scales,
            topk_weights, topk_ids,
            sorted_token_ids, expert_ids, num_tokens_post_padded,
            top_k_out, config,
        ) = inputs_tuple

        # Clone for reference
        a_ref = a_tri.clone()
        b_ref = b_tri.clone()
        c_ref = c_tri.clone()

        # Run triton kernel
        fn, c_out = make_kernel_fn(inputs_tuple)
        fn()
        torch.cuda.synchronize()

        # Compute reference
        a_ref_fp32 = torch_mxfp4_to_fp32(a_ref, a_mx_scales)
        b_ref_fp32 = torch_mxfp4_to_fp32(b_ref, b_mx_scales)

        c_ref_out = torch_moe_ref(
            a_ref_fp32, b_ref_fp32, c_ref,
            a_scale, b_scale,
            None,  # b_zp
            0,     # group_size
            topk_ids, topk_weights,
            ROUTED_WEIGHT,
            sorted_token_ids, expert_ids, num_tokens_post_padded,
            dtype=fp16_dtype,
            fp8_w8a8=False,
            int8_w8a16=False,
            int4_w4a16=False,
        )

        try:
            torch.testing.assert_close(
                c_out.to(fp16_dtype), c_ref_out.to(fp16_dtype),
                atol=1e-1, rtol=1e-1,
            )
            print("  [PASS] {}".format(_format_config(cfg)))
        except AssertionError as e:
            print("  [FAIL] {}: {}".format(_format_config(cfg), e))
            failures += 1

    return failures


def do_benchmark(indices):
    """Benchmark selected configs, return list of latencies."""
    torch.manual_seed(42)
    latencies = []

    for idx in indices:
        cfg = ALL_CONFIGS[idx]
        torch.cuda.empty_cache()

        inputs_tuple = build_inputs(cfg)
        fn, _ = make_kernel_fn(inputs_tuple)

        # Warmup
        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()

        # Timed iterations
        times = []
        for _ in range(ITERATIONS):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        times.sort()
        median_ms = times[len(times) // 2]
        latencies.append(median_ms)
        print("  {}  {:.4f}ms".format(_format_config(cfg), median_ms))

    return latencies


def geometric_mean(values):
    if not values:
        return 0.0
    log_sum = sum(math.log(v) for v in values if v > 0)
    return math.exp(log_sum / len(values))


def main():
    parser = argparse.ArgumentParser(description="Test harness for moe_op_mxfp4")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--full-benchmark", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    if not arch_info.is_fp4_avail():
        print("MXFP4 not supported on this architecture")
        sys.exit(1)

    if args.correctness:
        indices = _pick(ALL_CONFIGS, 25)
        print("Running correctness on {} configs...".format(len(indices)))
        failures = do_correctness(indices)
        print("GEAK_SHAPES_USED={}".format(indices))
        if failures > 0:
            print("FAILED: {} correctness checks failed".format(failures))
            sys.exit(1)
        print("All correctness checks passed")

    elif args.benchmark:
        indices = _pick(ALL_CONFIGS, 25)
        print("Running benchmark on {} configs...".format(len(indices)))
        latencies = do_benchmark(indices)
        print("GEAK_SHAPES_USED={}".format(indices))
        gm = geometric_mean(latencies)
        print("GEAK_RESULT_LATENCY_MS={:.4f}".format(gm))

    elif args.full_benchmark:
        indices = list(range(len(ALL_CONFIGS)))
        print("Running full benchmark on {} configs...".format(len(indices)))
        latencies = do_benchmark(indices)
        print("GEAK_SHAPES_USED={}".format(indices))
        gm = geometric_mean(latencies)
        print("GEAK_RESULT_LATENCY_MS={:.4f}".format(gm))

    elif args.profile:
        indices = _pick(ALL_CONFIGS, 5)
        print("Running profile on {} configs...".format(len(indices)))
        for idx in indices:
            cfg = ALL_CONFIGS[idx]
            torch.cuda.empty_cache()
            inputs_tuple = build_inputs(cfg)
            fn, _ = make_kernel_fn(inputs_tuple)
            # Just run the kernel a few times for profiling
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            print("  {}".format(_format_config(cfg)))
        print("GEAK_SHAPES_USED={}".format(indices))

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
