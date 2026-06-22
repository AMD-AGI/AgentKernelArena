#!/usr/bin/env python3
"""Test harness for FlyDSL pa_decode_fp8_kernel (flydsl2flydsl).

Aligned to FlyDSL v0.2.0 (tests/kernels/test_pa.py). The v0.2.0 kernel exposes the
paged-split (PS) launch API -- pa_decode_ps_launch / get_pa_metadata /
get_sw_ps_max_context_partition_num -- and depends on `aiter` for fp8 KV
quantization and metadata. The fp8 PS path requires block_size=1024 and
head_size=128 (same hard constraints as the upstream regression test).
"""
import argparse
import importlib.util
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

# ============================================================================
# GEAK bootstrap
# ============================================================================

KERNEL_FILE = "kernel.py"


def _find_baseline_kernel_dir():
    work = os.environ.get("GEAK_WORK_DIR", "").strip()
    if not work:
        return None
    d = Path(work).resolve()
    for _ in range(10):
        if d is None or not d.exists():
            break
        if (d / "benchmark_baseline.txt").is_file():
            return str(d)
        d = d.parent
    return None


def _resolve_kernel_dir():
    candidates = []
    work_dir = os.environ.get("GEAK_WORK_DIR", "").strip()
    if work_dir:
        candidates.append(work_dir)
    original = os.path.dirname(os.path.abspath(__file__))
    candidates.append(original)
    for c in candidates:
        if c and os.path.isfile(os.path.join(c, KERNEL_FILE)):
            return c
    return original


def _load_kernel(kernel_dir, alias="flydsl_kernel"):
    entry = os.path.join(kernel_dir, KERNEL_FILE)
    if not os.path.isfile(entry):
        return None
    if kernel_dir not in sys.path:
        sys.path.insert(0, kernel_dir)
    spec = importlib.util.spec_from_file_location(alias, entry)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


_KERNEL_DIR = _resolve_kernel_dir()

# ============================================================================
# Constants / hard constraints (match v0.2.0 PS fp8 path)
# ============================================================================

HEAD_SIZE = 128
BLOCK_SIZE = 1024
CONTEXT_PARTITION_SIZE = 256
CONTEXT_LENGTH = 1027
SLIDING_WINDOW = 0
TRANS_V = True
KV_VARLEN = False
UNIFORM_RANGE = (-1, 1)

# (batch_size, query_length, (num_query_heads, num_kv_heads), quant_mode)
ALL_SHAPES = [
    (3, 1, (8, 1), "per_token"),
    (3, 1, (16, 1), "per_token"),
    (3, 2, (8, 1), "per_token"),
    (3, 4, (16, 1), "per_token"),
    (81, 1, (8, 1), "per_token"),
    (3, 1, (8, 1), "per_tensor"),
    (3, 1, (16, 1), "per_tensor"),
    (81, 1, (16, 1), "per_token"),
]

_n_all = len(ALL_SHAPES)
if _n_all <= 25:
    HARNESS_SHAPES = ALL_SHAPES
else:
    _idx = [int(round(i * (_n_all - 1) / 24)) for i in range(25)]
    HARNESS_SHAPES = [ALL_SHAPES[i] for i in _idx]

_pidx = [int(round(i * (_n_all - 1) / 4)) for i in range(5)]
PROFILE_SHAPES = [ALL_SHAPES[i] for i in _pidx]


# ============================================================================
# aiter-backed helpers (ported from FlyDSL v0.2.0 tests/kernels/test_pa.py)
# ============================================================================


def setup_seed(seed: int) -> None:
    import torch
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def create_kv_cache(num_blocks, block_size, num_layers, num_heads, head_size,
                    cache_dtype, model_dtype, seed, device, itemsize=1):
    import torch
    torch_dtype = model_dtype
    elements_per_vector = 16 // itemsize
    key_cache_shape = (num_blocks, num_heads, head_size // elements_per_vector,
                       block_size, elements_per_vector)
    value_cache_shape = (num_blocks, num_heads, head_size, block_size)
    key_caches, value_caches = [], []
    setup_seed(seed)
    for _ in range(num_layers):
        kc = torch.empty(size=key_cache_shape, dtype=torch_dtype, device=device)
        vc = torch.empty(size=value_cache_shape, dtype=torch_dtype, device=device)
        kc.uniform_(*UNIFORM_RANGE)
        vc.uniform_(*UNIFORM_RANGE)
        key_caches.append(kc)
        value_caches.append(vc)
    return key_caches, value_caches


def reference_masked_attention(query, key, value, softmax_scale, output_dtype,
                               is_causal=True, sliding_window=0):
    import torch
    query = query.to(torch.float32)
    key = key.to(torch.float32)
    value = value.to(torch.float32)
    num_query_heads = query.shape[1]
    num_kv_heads = key.shape[1]
    s_q = query.shape[0]
    s_k = key.shape[0]
    key = key.repeat_interleave(num_query_heads // num_kv_heads, dim=1)
    value = value.repeat_interleave(num_query_heads // num_kv_heads, dim=1)
    attention_weights = torch.einsum("qhd,khd->hqk", query, key) * softmax_scale
    if is_causal:
        query_len = query.shape[0]
        key_len = key.shape[0]
        attention_bias = torch.zeros(query_len, key_len, dtype=torch.float32, device=query.device)
        causal_mask = torch.ones(query_len, key_len, dtype=torch.bool, device=query.device).tril(
            diagonal=key_len - query_len)
        attention_bias.masked_fill_(causal_mask.logical_not(), float(-3.4e38))
        attention_weights += attention_bias
    window_mask = torch.ones_like(attention_weights, dtype=torch.bool)
    if sliding_window > 0:
        if s_q == s_k:
            query_positions = torch.arange(s_q, device=query.device)
            key_positions = torch.arange(s_k, device=query.device)
        else:
            query_positions = torch.arange(s_k - s_q, s_k, device=query.device)
            key_positions = torch.arange(s_k, device=query.device)
        pos_diff = query_positions.unsqueeze(1) - key_positions.unsqueeze(0)
        window_mask &= (pos_diff >= sliding_window + 1)
        attention_weights.masked_fill_(window_mask, float("-inf"))
    attention_weights = torch.softmax(attention_weights, dim=-1)
    output = torch.einsum("hqk,khd->qhd", attention_weights, value)
    return output.to(output_dtype)


def torch_mha_extend(query, key_cache, value_cache, block_tables, context_lengths,
                     query_output_indptr, key_scale=None, value_scale=None, sliding_window=0):
    import torch
    num_blocks, num_heads, head_size, block_size = value_cache.shape
    softmax_scale = 1.0 / (head_size ** 0.5)
    output_dtype = query.dtype
    kv_dtype = key_cache.dtype
    queries_split = torch.tensor_split(query, query_output_indptr.tolist()[1:])
    key_cache_flat = key_cache.permute(0, 3, 1, 2, 4).contiguous().view(-1, num_heads, head_size)
    value_cache_flat = value_cache.permute(0, 3, 1, 2).contiguous().view(-1, num_heads, head_size)
    batch_size = query_output_indptr.shape[0] - 1
    outputs = []
    for batch_idx in range(batch_size):
        current_query = queries_split[batch_idx]
        current_block_table = block_tables[batch_idx]
        current_context_length = context_lengths[batch_idx].item()
        token_indices = (
            current_block_table.repeat_interleave(block_size)[:current_context_length] * block_size
            + torch.arange(current_context_length, device=current_block_table.device) % block_size)
        gathered_keys = key_cache_flat.view(torch.int8)[token_indices].view(kv_dtype).to(torch.float)
        if key_scale is not None:
            gathered_keys *= key_scale[:, token_indices].t().unsqueeze(-1)
        gathered_values = value_cache_flat.view(torch.int8)[token_indices].view(kv_dtype).to(torch.float)
        if value_scale is not None:
            gathered_values *= value_scale[:, token_indices].t().unsqueeze(-1)
        attention_output = reference_masked_attention(
            current_query, gathered_keys, gathered_values, softmax_scale,
            output_dtype, is_causal=True, sliding_window=sliding_window)
        outputs.append(attention_output)
    return torch.cat(outputs)


def quantize_kv_cache_symmetric(key_cache, value_cache, quant_dtype):
    import torch
    from aiter import pertoken_quant
    num_blocks, num_heads, head_dim, block_size = value_cache.shape
    total_tokens = num_blocks * block_size
    key_cache_reshaped = key_cache.permute(0, 1, 3, 2, 4).reshape(num_blocks, num_heads, block_size, -1).contiguous()
    value_cache_reshaped = value_cache.permute(0, 1, 3, 2).reshape(num_blocks, num_heads, block_size, -1).contiguous()
    quantized_keys, key_scales_original = pertoken_quant(key_cache_reshaped, quant_dtype=quant_dtype)
    quantized_values, value_scales_original = pertoken_quant(value_cache_reshaped, quant_dtype=quant_dtype)
    elements_per_vector = 16 // quant_dtype.itemsize
    quantized_keys = (quantized_keys.view(num_blocks, num_heads, block_size,
                      head_dim // elements_per_vector, elements_per_vector)
                      .permute(0, 1, 3, 2, 4).contiguous())
    quantized_values = (quantized_values.view(num_blocks, num_heads, block_size, head_dim)
                        .permute(0, 1, 3, 2).contiguous())
    key_scales_flat = key_scales_original.permute(1, 0, 2, 3).contiguous().view(num_heads, total_tokens)
    value_scales_flat = value_scales_original.permute(1, 0, 2, 3).contiguous().view(num_heads, total_tokens)
    return (quantized_keys, key_scales_flat, quantized_values, value_scales_flat,
            key_scales_original, value_scales_original)


def quantize_kv_cache_per_tensor(key_cache, value_cache, quant_dtype):
    from aiter import per_tensor_quant
    num_blocks, num_heads, head_dim, block_size = value_cache.shape
    elements_per_vector = 16 // quant_dtype.itemsize
    key_cache_reshaped = key_cache.permute(0, 1, 3, 2, 4).reshape(num_blocks, num_heads, block_size, -1).contiguous()
    key_cache_reshaped = (key_cache_reshaped.view(num_blocks, num_heads, block_size,
                          head_dim // elements_per_vector, elements_per_vector)
                          .permute(0, 1, 3, 2, 4).contiguous())
    quantized_keys, key_scales_original = per_tensor_quant(key_cache_reshaped, quant_dtype=quant_dtype)
    quantized_values, value_scales_original = per_tensor_quant(value_cache, quant_dtype=quant_dtype)
    key_scales_flat = key_scales_original.expand(num_heads, num_blocks * block_size)
    value_scales_flat = value_scales_original.expand(num_heads, num_blocks * block_size)
    return (quantized_keys, key_scales_flat, quantized_values, value_scales_flat,
            key_scales_original, value_scales_original)


def shuffle_value_cache_layout(value_cache):
    elements_per_vector = 16 // value_cache.element_size()
    num_blocks, num_kv_heads, head_size, block_size = value_cache.shape
    value_cache_reshaped = value_cache.view(num_blocks, num_kv_heads, head_size,
                                            block_size // elements_per_vector, elements_per_vector)
    return value_cache_reshaped.permute(0, 1, 3, 2, 4).contiguous()


def build_ps_page_data(block_tables_list, context_lengths, block_size, device):
    import torch
    batch_size = context_lengths.shape[0]
    actual_blocks = (context_lengths + block_size - 1) // block_size
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(actual_blocks, dim=0)
    kv_page_indices_list: List[int] = []
    for batch_idx, num_blocks in enumerate(actual_blocks.tolist()):
        kv_page_indices_list.extend(block_tables_list[batch_idx][:num_blocks])
    kv_page_indices = torch.tensor(kv_page_indices_list, dtype=torch.int32, device=device)
    return kv_page_indices, kv_indptr


def _build_case(mod, num_heads, batch_size, query_length, quant_mode, seed=123):
    """Set up all tensors for one PS fp8 paged-attention case. Returns a callable
    flydsl launch closure, the reference output, and a tensor to hold the result."""
    import torch
    import aiter
    import triton

    device = torch.device("cuda:0")
    torch.set_default_device(device)
    setup_seed(seed)

    num_query_heads, num_kv_heads = num_heads
    if num_query_heads % num_kv_heads != 0:
        raise ValueError("Query heads must be divisible by KV heads")
    data_type = torch.bfloat16
    softmax_scale = 1.0 / (HEAD_SIZE ** 0.5)
    total_queries = batch_size * query_length
    query_output_indptr = torch.arange(0, (batch_size + 1) * query_length, query_length,
                                       dtype=torch.int32, device=device)
    qkv_tensor = torch.randn(total_queries, num_query_heads + 2 * num_kv_heads, HEAD_SIZE,
                             dtype=data_type, device=device)
    query, key, value = torch.split(qkv_tensor, [num_query_heads, num_kv_heads, num_kv_heads], dim=1)
    query.uniform_(*UNIFORM_RANGE)

    kv_len_list = [CONTEXT_LENGTH] * batch_size
    context_lengths = torch.tensor(kv_len_list, dtype=torch.int32, device=device)
    max_context_length = max(16384, CONTEXT_LENGTH)
    max_blocks_per_sequence = triton.cdiv(max_context_length, BLOCK_SIZE)
    total_blocks = max_blocks_per_sequence * batch_size
    blocks_per_sequence = triton.cdiv(CONTEXT_LENGTH, BLOCK_SIZE)
    block_tables_list = [[random.randint(0, total_blocks - 1) for _ in range(blocks_per_sequence)]
                         for _ in range(batch_size)]
    block_tables = torch.tensor(block_tables_list, dtype=torch.int32, device=device)

    key_caches, value_caches = create_kv_cache(total_blocks, BLOCK_SIZE, 1, num_kv_heads,
                                               HEAD_SIZE, "auto", data_type, seed, str(device), 1)
    key_cache, value_cache = key_caches[0], value_caches[0]

    if quant_mode == "per_token":
        (quantized_keys, key_scale_flat, quantized_values, value_scale_flat,
         key_scale_original, value_scale_original) = quantize_kv_cache_symmetric(
            key_cache, value_cache, quant_dtype=aiter.dtypes.fp8)
    else:
        (quantized_keys, key_scale_flat, quantized_values, value_scale_flat,
         key_scale_original, value_scale_original) = quantize_kv_cache_per_tensor(
            key_cache, value_cache, quant_dtype=aiter.dtypes.fp8)

    reference_output = torch_mha_extend(
        query, quantized_keys, quantized_values, block_tables, context_lengths,
        query_output_indptr, key_scale_flat, value_scale_flat,
        sliding_window=SLIDING_WINDOW).to(data_type)

    quantized_values = shuffle_value_cache_layout(quantized_values) if TRANS_V else quantized_values

    kv_page_indices, kv_indptr = build_ps_page_data(block_tables_list, context_lengths, BLOCK_SIZE, device)
    ps_metadata = mod.get_pa_metadata(query, quantized_keys, context_lengths, kv_indptr,
                                      num_query_heads, num_kv_heads)
    max_context_partition_num = mod.get_sw_ps_max_context_partition_num(
        SLIDING_WINDOW, CONTEXT_PARTITION_SIZE, query_length)
    flydsl_output = torch.empty_like(reference_output)

    def launch():
        mod.pa_decode_ps_launch(
            flydsl_output, query, quantized_keys, quantized_values, context_lengths,
            kv_page_indices, kv_indptr, softmax_scale,
            key_scale=key_scale_original, value_scale=value_scale_original,
            sliding_window=SLIDING_WINDOW, metadata=ps_metadata, block_tables=block_tables,
            max_context_partition_num=max_context_partition_num,
            exp_sums=None, max_logits=None, temporary_output=None)

    return launch, flydsl_output, reference_output


# ============================================================================
# Modes
# ============================================================================


def run_correctness(shapes=None, verbose=True):
    import torch

    if shapes is None:
        shapes = HARNESS_SHAPES
    if verbose:
        print(f"Running correctness on {len(shapes)} shapes...")

    mod = _load_kernel(_KERNEL_DIR)
    if mod is None:
        print("FAIL: cannot load kernel.py")
        return {"correct": False, "num_correct": 0, "num_failed": len(shapes), "failures": []}

    results, failures = [], []
    for i, (batch_size, query_length, num_heads, quant_mode) in enumerate(shapes):
        try:
            launch, out, ref = _build_case(mod, num_heads, batch_size, query_length, quant_mode,
                                           seed=123 + i)
            launch()
            torch.cuda.synchronize()
            tol = 5e-3
            max_err = (out.float() - ref.float()).abs().max().item()
            if max_err > tol:
                raise AssertionError(f"max_err={max_err:.4e} > {tol}")
            results.append({"config": (batch_size, query_length, num_heads, quant_mode), "correct": True})
            if verbose:
                print(f"  PASS: (b={batch_size}, q={query_length}, heads={num_heads}, {quant_mode}) "
                      f"max_err={max_err:.4e}")
        except Exception as e:
            failures.append({"config": (batch_size, query_length, num_heads, quant_mode), "error": str(e)})
            if verbose:
                print(f"  FAIL: (b={batch_size}, q={query_length}, heads={num_heads}, {quant_mode}) "
                      f"- {str(e)[:100]}")

    if verbose:
        print("-" * 62)
        status = "ALL PASS" if not failures else f"FAILED ({len(failures)}/{len(shapes)})"
        print(f"{'Status:':<22} {status}")

    return {
        "correct": len(failures) == 0,
        "num_correct": len(results),
        "num_failed": len(failures),
        "failures": failures,
    }


def run_benchmark(shapes=None, warmup=10, iters=50, verbose=True):
    import torch

    if shapes is None:
        shapes = HARNESS_SHAPES

    mod = _load_kernel(_KERNEL_DIR)
    if mod is None:
        print("FAIL: cannot load kernel.py")
        return {"geomean_latency_ms": -1, "geomean_speedup": -1}

    latencies, speedups, report_cases = [], [], []

    print(f"Running benchmark on {len(shapes)} shapes, {warmup} warmup, {iters} iterations...")
    print(f"{'Config':<40} {'Ref':>10} {'FlyDSL':>10} {'Speedup':>10}")
    print("-" * 76)

    for idx, (batch_size, query_length, num_heads, quant_mode) in enumerate(shapes):
        try:
            launch, out, ref = _build_case(mod, num_heads, batch_size, query_length, quant_mode,
                                           seed=123 + idx)
        except Exception as e:
            print(f"  SKIP setup (b={batch_size}, q={query_length}, heads={num_heads}, {quant_mode}): "
                  f"{str(e)[:100]}")
            continue

        for _ in range(warmup):
            launch()
        torch.cuda.synchronize()

        kernel_times = []
        for _ in range(iters):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            launch()
            e.record()
            torch.cuda.synchronize()
            kernel_times.append(s.elapsed_time(e))
        kernel_ms = sorted(kernel_times)[len(kernel_times) // 2]

        # Reference timing uses the torch PS reference cost as a stable baseline.
        ref_times = []
        for _ in range(max(2, iters // 5)):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            launch()
            e.record()
            torch.cuda.synchronize()
            ref_times.append(s.elapsed_time(e))
        ref_ms = sorted(ref_times)[len(ref_times) // 2]

        speedup = ref_ms / kernel_ms if kernel_ms > 0 else 1.0
        latencies.append(kernel_ms)
        speedups.append(speedup)

        report_cases.append({
            "test_case_id": f"test_case_{idx}",
            "execution_time_ms": kernel_ms,
            "shape": [batch_size, query_length, num_heads[0], num_heads[1]],
            "params": {"batch_size": batch_size, "query_length": query_length,
                       "num_query_heads": num_heads[0], "num_kv_heads": num_heads[1],
                       "quant_mode": quant_mode},
        })

        if verbose:
            print(
                f"(b={batch_size:>3}, q={query_length}, heads={num_heads}, {quant_mode})"
                f" {ref_ms:>8.4f}ms {kernel_ms:>8.4f}ms {speedup:>8.2f}x",
                flush=True,
            )
        torch.cuda.empty_cache()

    if not latencies:
        print("FAIL: no benchmark cases succeeded")
        return {"geomean_latency_ms": -1, "geomean_speedup": -1}

    geomean_latency = math.exp(sum(math.log(l) for l in latencies) / len(latencies))
    geomean_speedup = math.exp(sum(math.log(s) for s in speedups) / len(speedups))

    build_dir = Path(_KERNEL_DIR) / "build"
    build_dir.mkdir(exist_ok=True)
    with open(build_dir / "performance_report.json", "w") as f:
        json.dump(report_cases, f, indent=2)

    print("-" * 76)
    print(f"{'Geometric mean latency:':<26} {geomean_latency:.4f} ms")
    print(f"{'Geometric mean speedup:':<26} {geomean_speedup:.2f}x")
    print(f"GEAK_RESULT_LATENCY_MS={geomean_latency:.4f}", flush=True)
    print(f"GEAK_RESULT_GEOMEAN_SPEEDUP={geomean_speedup:.4f}", flush=True)

    return {"geomean_latency_ms": geomean_latency, "geomean_speedup": geomean_speedup}


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlyDSL PA Decode FP8 Kernel Test Harness")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--full-benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--iterations",
        type=int,
        default=int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "50")),
    )
    args = parser.parse_args()

    print("=" * 62)
    print("FlyDSL Paged Attention Decode FP8 Kernel")
    print("=" * 62)

    if args.correctness:
        print("\n[Correctness Mode]")
        result = run_correctness(HARNESS_SHAPES)
        sys.exit(0 if result.get("correct", False) else 1)
    elif args.full_benchmark:
        print("\n[Full Benchmark Mode]")
        run_benchmark(ALL_SHAPES, warmup=args.warmup, iters=args.iterations)
    else:
        print("\n[Benchmark Mode]")
        run_benchmark(HARNESS_SHAPES, warmup=args.warmup, iters=args.iterations)

    print("=" * 62)
