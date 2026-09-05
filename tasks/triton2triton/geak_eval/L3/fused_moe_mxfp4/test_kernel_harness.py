#!/usr/bin/env python3
# Test harness for moe_op_mxfp4 kernel
# Shape source: op_tests/triton_tests/moe/test_moe_mx.py

import argparse
import json
import os
import sys
import math
from pathlib import Path

import torch
import triton
from _aka_benchmark import benchmark_cuda_graph_or_events_samples


def benchmark_cuda_graph_or_events(*args, **kwargs):
    samples, metadata = benchmark_cuda_graph_or_events_samples(*args, **kwargs)
    values = sorted(samples)
    midpoint = len(values) // 2
    median_ms = (
        values[midpoint]
        if len(values) % 2
        else (values[midpoint - 1] + values[midpoint]) / 2.0
    )
    return median_ms, metadata

# Kernel under test — kernel.py sits next to this harness (Python adds the
# script's directory to sys.path[0] automatically), and GEAK copies both files
# side-by-side into each per-task workspace. Importing from `kernel` guarantees
# the agent's edits are what we exercise.
from kernel import fused_moe_mxfp4, torch_to_triton_dtype  # noqa: E402


def _is_fp4_avail() -> bool:
    """MXFP4 support is gated on gfx950 / gfx1250."""
    try:
        arch = triton.runtime.driver.active.get_current_target().arch
    except Exception:
        return False
    return arch in ("gfx950", "gfx1250")


# ============================================================================
# INLINED REFERENCE HELPERS (from aiter/op_tests/triton_tests/moe/*)
# ----------------------------------------------------------------------------
# Kept self-contained so this harness has zero aiter dependency. If aiter's
# reference numerics change upstream, mirror the relevant change here.
# ============================================================================


# MXFP4 MoE tuned configs, copied verbatim from
# aiter/ops/triton/configs/moe/gfx950-MOE-MX_FP4.json
_MXFP4_MOE_CONFIGS = {
    "small_M": {
        "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 2, "num_warps": 4, "num_stages": 4,
        "waves_per_eu": 0, "matrix_instr_nonkdim": 16, "kpack": 1,
    },
    "medium_M": {
        "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 4, "num_warps": 8, "num_stages": 2,
        "waves_per_eu": 0, "matrix_instr_nonkdim": 16, "kpack": 1,
    },
    "large_M": {
        "BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 8, "num_warps": 8, "num_stages": 4,
        "waves_per_eu": 0, "matrix_instr_nonkdim": 16, "kpack": 1,
    },
}
_M_THRESHOLD_SMALL = 256
_M_THRESHOLD_MEDIUM = 1024


def _get_mxfp4_moe_config(M: int) -> dict:
    if M < _M_THRESHOLD_SMALL:
        return dict(_MXFP4_MOE_CONFIGS["small_M"])
    if M < _M_THRESHOLD_MEDIUM:
        return dict(_MXFP4_MOE_CONFIGS["medium_M"])
    return dict(_MXFP4_MOE_CONFIGS["large_M"])


def _alloc_rand(shape, device, dtype):
    if dtype.itemsize == 1:
        tmp = 2 ** -(torch.randint(4, 8, shape, device=device, dtype=torch.float16))
        return tmp.to(dtype)
    return torch.randn(shape, device=device, dtype=dtype)


def _torch_dynamic_mxfp4_quant(x: torch.Tensor):
    """Quantize a bf16/fp16 tensor to MXFP4 (packed uint8) + E8M0 scale."""
    MXFP4_QUANT_BLOCK_SIZE = 32
    x_shape = x.shape
    if x.shape[-1] % MXFP4_QUANT_BLOCK_SIZE != 0:
        shape = list(x_shape)
        shape[-1] = (
            (shape[-1] - 1 + MXFP4_QUANT_BLOCK_SIZE) // MXFP4_QUANT_BLOCK_SIZE
        ) * MXFP4_QUANT_BLOCK_SIZE
        x_padded = torch.zeros(tuple(shape), device=x.device, dtype=x.dtype)
        x_padded[..., : x.shape[-1]] = x
    else:
        x_padded = x

    x_padded = x_padded.reshape(
        -1, x_padded.shape[-1] // MXFP4_QUANT_BLOCK_SIZE, MXFP4_QUANT_BLOCK_SIZE
    ).to(torch.float32)
    amax, _ = torch.max(torch.abs(x_padded), dim=-1)
    amax = amax.view(torch.int32)
    amax = (amax + 0x200000) & 0xFF800000
    amax = amax.view(torch.float32)
    scale_e8m0_unbiased = torch.log2(amax).floor() - 2
    scale_e8m0_unbiased = torch.clamp(scale_e8m0_unbiased, min=-127, max=127)
    quant_scale = torch.exp2(-scale_e8m0_unbiased)
    qx = x_padded * quant_scale.unsqueeze(-1)
    bs_e8m0 = scale_e8m0_unbiased.to(torch.uint8) + 127

    qx = qx.view(torch.int32)
    s = qx & 0x80000000
    e = (qx >> 23) & 0xFF
    m = qx & 0x7FFFFF
    E8_BIAS = 127
    E2_BIAS = 1
    adjusted_exponents = E8_BIAS - e - 1
    m = torch.where(e < E8_BIAS, (0x400000 | (m >> 1)) >> adjusted_exponents, m)
    e = torch.where(e > E8_BIAS - E2_BIAS, e, E8_BIAS - E2_BIAS) - (E8_BIAS - E2_BIAS)
    combined_val = (((e << 2) | (m >> 21)) + 1) >> 1
    e2m1_tmp = torch.where(combined_val < 0x7, combined_val, 0x7)
    e2m1_value = (((s >> 28) & 0xF) | e2m1_tmp).to(torch.uint8)
    x_mxfp4 = e2m1_value[..., ::2] | (e2m1_value[..., 1::2] << 4)
    x_mxfp4 = torch.flatten(x_mxfp4, -2, -1)
    if x.shape[-1] % MXFP4_QUANT_BLOCK_SIZE != 0:
        x_mxfp4 = x_mxfp4[..., : x.shape[-1] // 2]

    mxfp4_shape = tuple(list(x_shape)[:-1] + [x_shape[-1] // 2])
    x_mxfp4 = x_mxfp4.reshape(mxfp4_shape)
    bs_e8m0_shape = tuple(
        list(x_shape)[:-1] + [x_shape[-1] // MXFP4_QUANT_BLOCK_SIZE]
    )
    bs_e8m0 = bs_e8m0.reshape(bs_e8m0_shape)
    return x_mxfp4, bs_e8m0


_MXFP4_LUT = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
]


def _mxfp4_to_f32(x):
    x = x.repeat_interleave(2, dim=-1)
    x[..., ::2] = x[..., ::2] & 0xF
    x[..., 1::2] = x[..., 1::2] >> 4
    lut = torch.tensor(_MXFP4_LUT, dtype=torch.float32, device=x.device)
    return lut[x.long()]


def _e8m0_to_f32(x):
    x_f32 = 2 ** ((x - 127).to(torch.float32))
    x_f32[x_f32 == 128] = float("nan")
    return x_f32


def torch_mxfp4_to_fp32(x, x_scales):
    x_f32 = _mxfp4_to_f32(x)
    x_scales = x_scales.repeat_interleave(32, dim=-1).to(torch.float32)
    return x_f32 * _e8m0_to_f32(x_scales)


def _moe_align_block_size(
    topk_ids, num_experts, top_k, block_size,
    sorted_token_ids, expert_ids, num_tokens_post_pad,
):
    M, top_k = topk_ids.shape
    expert_to_tokens = [[] for _ in range(num_experts)]
    for token_id in range(M):
        for j in range(top_k):
            e_id = topk_ids[token_id, j].item()
            expert_to_tokens[e_id].append(token_id * top_k + j)

    reordered_token_ids = []
    reordered_expert_ids = []
    for e_id in range(num_experts):
        tokens_for_expert = expert_to_tokens[e_id]
        num_tokens = len(tokens_for_expert)
        n_blocks = (num_tokens + block_size - 1) // block_size
        padded_size = n_blocks * block_size
        reordered_token_ids.extend(tokens_for_expert)
        reordered_expert_ids.extend([e_id] * n_blocks)
        if padded_size > num_tokens:
            reordered_token_ids.extend([topk_ids.numel()] * (padded_size - num_tokens))

    token_length = len(reordered_token_ids)
    expert_length = len(reordered_expert_ids)
    sorted_token_ids[:token_length] = torch.tensor(
        reordered_token_ids,
        dtype=sorted_token_ids.dtype, device=sorted_token_ids.device,
    )
    expert_ids[:expert_length] = torch.tensor(
        reordered_expert_ids, dtype=expert_ids.dtype, device=expert_ids.device,
    )
    if token_length < sorted_token_ids.numel():
        sorted_token_ids[token_length:] = topk_ids.numel()
    if expert_length < expert_ids.numel():
        expert_ids[expert_length:] = topk_ids.numel()
    num_tokens_post_pad.fill_(token_length)


def _torch_moe_align_block_size_ref(topk_ids, block_size, num_experts):
    sorted_ids = torch.empty(
        (topk_ids.numel() + num_experts * (block_size - 1),),
        dtype=torch.int32, device=topk_ids.device,
    )
    expert_ids = torch.empty(
        (topk_ids.numel() + num_experts,), dtype=torch.int32, device=topk_ids.device,
    )
    sorted_ids.fill_(topk_ids.numel())
    num_tokens_post_pad = torch.empty((1,), dtype=torch.int32, device=topk_ids.device)
    _moe_align_block_size(
        topk_ids, num_experts, topk_ids.shape[1], block_size,
        sorted_ids, expert_ids, num_tokens_post_pad,
    )
    return sorted_ids, expert_ids, num_tokens_post_pad


def torch_moe_ref(a, b, c, topk_ids, dtype):
    """Slim MoE reference — covers only the path this harness uses:
    no fp8/int8/int4 quant, no routed-weight, no activation fusion."""
    M, top_k, N = c.shape
    a_expanded = a.unsqueeze(1).repeat(1, top_k, 1)  # (M, top_k, K)
    b_indexed = b[topk_ids]                           # (M, top_k, N, K)
    return torch.einsum("mek,menk->men", a_expanded.to(dtype), b_indexed.to(dtype))


def input_helper(M, N, K, top_k, E):
    """MXFP4 A & B inputs. Mirrors aiter's input_helper for the mxfp4/mxfp4 case."""
    fp16_dtype = torch.bfloat16
    c_dtype = torch.bfloat16

    a_tri = _alloc_rand((M, K), dtype=fp16_dtype, device="cuda")
    b_tri = _alloc_rand((E, N, K), dtype=fp16_dtype, device="cuda")
    c_tri = torch.zeros((M, top_k, N), dtype=c_dtype, device="cuda")

    a_scale = torch.tensor([1.00], dtype=torch.float32, device="cuda")
    b_scale = torch.tensor([1.00] * E, dtype=torch.float32, device="cuda")

    config = _get_mxfp4_moe_config(M)

    values = torch.randn(M, E, dtype=torch.float16, device="cuda")
    topk_weights, topk_ids = torch.topk(torch.softmax(values, dim=1), k=top_k, dim=1)

    sorted_token_ids, expert_ids, num_tokens_post_padded = (
        _torch_moe_align_block_size_ref(topk_ids, config["BLOCK_SIZE_M"], E)
    )
    a_tri, a_mx_scales = _torch_dynamic_mxfp4_quant(a_tri)
    b_tri, b_mx_scales = _torch_dynamic_mxfp4_quant(b_tri)

    return (
        a_tri, b_tri, c_tri,
        a_scale, b_scale, a_mx_scales, b_mx_scales,
        topk_weights, topk_ids,
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        top_k, config,
    )

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

# Fixed kernel-call parameters (the only supported combination in the test)
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
    """Build inputs using the inlined input_helper."""
    M, N, K, E, top_k = cfg
    return input_helper(M, N, K, top_k, E)


def make_kernel_fn(inputs_tuple):
    """Create a callable that runs fused_moe_mxfp4."""
    (
        a_tri, b_tri, c_tri,
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
            a_tri, b_tri, c_tri,
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
            a_ref_fp32, b_ref_fp32, c_ref, topk_ids, dtype=fp16_dtype,
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
    methods = []
    report_cases = []

    for idx in indices:
        cfg = ALL_CONFIGS[idx]
        torch.cuda.empty_cache()

        inputs_tuple = build_inputs(cfg)
        fn, _ = make_kernel_fn(inputs_tuple)

        median_ms, benchmark_meta = benchmark_cuda_graph_or_events(
            fn, warmup=WARMUP, repetition=ITERATIONS,
        )
        latencies.append(median_ms)
        methods.append(benchmark_meta["benchmark_method"])
        report_cases.append({
            # The upstream list intentionally contains one repeated shape. Keep
            # both workloads, but include the stable list index so baseline and
            # candidate records remain uniquely matchable.
            "test_case_id": "case={} {}".format(idx, _format_config(cfg)),
            "params": {
                "case_index": idx,
                "M": cfg[0],
                "N": cfg[1],
                "K": cfg[2],
                "E": cfg[3],
                "top_k": cfg[4],
            },
            "execution_time_ms": median_ms,
            **benchmark_meta,
        })
        print("  {}  {:.4f}ms".format(_format_config(cfg), median_ms))

    report_path = Path("build/performance_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report_cases, indent=2))
    return latencies, methods


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
    parser.add_argument("--iterations", type=int, default=None, help="Number of benchmark iterations (overrides GEAK_BENCHMARK_ITERATIONS env var)")
    args = parser.parse_args()
    if args.iterations is not None:
        global ITERATIONS
        ITERATIONS = args.iterations

    if not _is_fp4_avail():
        print("MXFP4 not supported on this architecture")
        sys.exit(1)

    if args.correctness:
        indices = list(range(len(ALL_CONFIGS)))
        print("Running correctness on {} configs...".format(len(indices)))
        failures = do_correctness(indices)
        print("GEAK_SHAPES_USED={}".format(indices))
        if failures > 0:
            print("FAILED: {} correctness checks failed".format(failures))
            sys.exit(1)
        print("All correctness checks passed")

    elif args.benchmark:
        indices = list(range(len(ALL_CONFIGS)))  # use all configs so benchmark matches full-benchmark
        print("Running benchmark on {} configs...".format(len(indices)))
        latencies, methods = do_benchmark(indices)
        print("GEAK_SHAPES_USED={}".format(indices))
        gm = geometric_mean(latencies)
        print("GEAK_RESULT_LATENCY_MS={:.4f}".format(gm))
        print("GEAK_BENCHMARK_METHOD={}".format(
            methods[0] if len(set(methods)) == 1
            else "mixed:" + ",".join(sorted(set(methods)))
        ))

    elif args.full_benchmark:
        indices = list(range(len(ALL_CONFIGS)))
        print("Running full benchmark on {} configs...".format(len(indices)))
        latencies, methods = do_benchmark(indices)
        print("GEAK_SHAPES_USED={}".format(indices))
        gm = geometric_mean(latencies)
        print("GEAK_RESULT_LATENCY_MS={:.4f}".format(gm))
        print("GEAK_BENCHMARK_METHOD={}".format(
            methods[0] if len(set(methods)) == 1
            else "mixed:" + ",".join(sorted(set(methods)))
        ))

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
