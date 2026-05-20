#!/usr/bin/env python3
"""Test harness for FlyDSL pa_decode_fp8_kernel (flydsl2flydsl).

Uses the PS (persistent scheduling) API: compile_pa_decode_ps,
get_pa_metadata, pa_decode_ps_launch.  Requires ``aiter`` for KV
quantisation and metadata helpers.
"""
import argparse
import importlib.util
import json
import math
import os
import random
import sys
from pathlib import Path

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
    _flydsl2flydsl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    if _flydsl2flydsl_dir not in sys.path:
        sys.path.insert(0, _flydsl2flydsl_dir)
    spec = importlib.util.spec_from_file_location(alias, entry)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


_KERNEL_DIR = _resolve_kernel_dir()



# ============================================================================
# Constants and test shapes
# ============================================================================

HEAD_SIZE = 128
QUERY_GROUP_SIZE = 16
KV_BLOCK_SIZE = 1024
CONTEXT_PARTITION_SIZE = 256

# (batch_size, context_length, num_kv_heads, query_length, quant_mode)
ALL_SHAPES = [
    (3, 1027, 1, 1, "per_tensor"),
    (3, 1027, 1, 1, "per_token"),
    (81, 1027, 1, 1, "per_tensor"),
    (3, 1027, 1, 2, "per_tensor"),
    (3, 1027, 1, 4, "per_tensor"),
    (81, 1027, 1, 2, "per_tensor"),
    (3, 1027, 2, 1, "per_tensor"),
    (81, 1027, 2, 1, "per_tensor"),
]

_n_all = len(ALL_SHAPES)
if _n_all <= 25:
    HARNESS_SHAPES = ALL_SHAPES
else:
    _idx = [int(round(i * (_n_all - 1) / 24)) for i in range(25)]
    HARNESS_SHAPES = [ALL_SHAPES[i] for i in _idx]

_pidx = [int(round(i * (_n_all - 1) / 4)) for i in range(5)]
PROFILE_SHAPES = [ALL_SHAPES[i] for i in _pidx]

UNIFORM_RANGE = (-1, 1)


# ============================================================================
# Helpers
# ============================================================================


def _setup_seed(seed: int):
    random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _create_kv_cache(num_blocks, block_size, num_heads, head_size, dtype, device="cuda"):
    """Create BF16 KV caches in vLLM-style layout."""
    import torch
    elements_per_vector = 16
    key_cache_shape = (num_blocks, num_heads, head_size // elements_per_vector,
                       block_size, elements_per_vector)
    value_cache_shape = (num_blocks, num_heads, head_size, block_size)
    key_cache = torch.empty(key_cache_shape, dtype=dtype, device=device)
    value_cache = torch.empty(value_cache_shape, dtype=dtype, device=device)
    key_cache.uniform_(*UNIFORM_RANGE)
    value_cache.uniform_(*UNIFORM_RANGE)
    return key_cache, value_cache


def _quantize_kv_per_tensor(key_cache, value_cache):
    """Per-tensor FP8 quantisation using aiter."""
    from aiter import per_tensor_quant, dtypes
    num_blocks, num_heads, head_dim, block_size = value_cache.shape
    elements_per_vector = 16 // dtypes.fp8.itemsize

    key_reshaped = (key_cache.permute(0, 1, 3, 2, 4)
                    .reshape(num_blocks, num_heads, block_size, -1)
                    .contiguous()
                    .view(num_blocks, num_heads, block_size,
                          head_dim // elements_per_vector, elements_per_vector)
                    .permute(0, 1, 3, 2, 4).contiguous())
    q_keys, key_scale = per_tensor_quant(key_reshaped, quant_dtype=dtypes.fp8)
    q_vals, val_scale = per_tensor_quant(value_cache, quant_dtype=dtypes.fp8)

    key_scale_flat = key_scale.expand(num_heads, num_blocks * block_size)
    val_scale_flat = val_scale.expand(num_heads, num_blocks * block_size)
    return q_keys, key_scale_flat, q_vals, val_scale_flat, key_scale, val_scale


def _quantize_kv_per_token(key_cache, value_cache):
    """Per-token FP8 quantisation using aiter."""
    from aiter import pertoken_quant, dtypes
    num_blocks, num_heads, head_dim, block_size = value_cache.shape
    total_tokens = num_blocks * block_size
    elements_per_vector = 16 // dtypes.fp8.itemsize

    key_reshaped = (key_cache.permute(0, 1, 3, 2, 4)
                    .reshape(num_blocks, num_heads, block_size, -1).contiguous())
    val_reshaped = (value_cache.permute(0, 1, 3, 2)
                    .reshape(num_blocks, num_heads, block_size, -1).contiguous())

    q_keys, key_scales_orig = pertoken_quant(key_reshaped, quant_dtype=dtypes.fp8)
    q_vals, val_scales_orig = pertoken_quant(val_reshaped, quant_dtype=dtypes.fp8)

    q_keys = (q_keys.view(num_blocks, num_heads, block_size,
                           head_dim // elements_per_vector, elements_per_vector)
              .permute(0, 1, 3, 2, 4).contiguous())
    q_vals = (q_vals.view(num_blocks, num_heads, block_size, head_dim)
              .permute(0, 1, 3, 2).contiguous())

    key_scale_flat = key_scales_orig.permute(1, 0, 2, 3).contiguous().view(num_heads, total_tokens)
    val_scale_flat = val_scales_orig.permute(1, 0, 2, 3).contiguous().view(num_heads, total_tokens)
    return q_keys, key_scale_flat, q_vals, val_scale_flat, key_scales_orig, val_scales_orig


def _shuffle_value_cache(value_cache):
    """Transpose V layout for PS kernel."""
    elements_per_vector = 16 // value_cache.element_size()
    nb, nh, hs, bs = value_cache.shape
    return (value_cache.view(nb, nh, hs, bs // elements_per_vector, elements_per_vector)
            .permute(0, 1, 3, 2, 4).contiguous())


def _build_ps_page_data(block_tables_list, context_lengths, block_size, device):
    """Build kv_page_indices and kv_indptr for PS scheduler."""
    import torch
    batch_size = context_lengths.shape[0]
    actual_blocks = (context_lengths + block_size - 1) // block_size
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(actual_blocks, dim=0)
    indices = []
    for b, nb in enumerate(actual_blocks.tolist()):
        indices.extend(block_tables_list[b][:nb])
    kv_page_indices = torch.tensor(indices, dtype=torch.int32, device=device)
    return kv_page_indices, kv_indptr


def _create_test_data(batch_size, context_length, num_kv_heads, query_length=1,
                      quant_mode="per_tensor"):
    """Create paged KV cache test data compatible with the PS API."""
    import torch
    import triton
    device = "cuda"
    num_query_heads = num_kv_heads * QUERY_GROUP_SIZE
    max_context_length = max(16384, context_length)
    max_blocks_per_seq = triton.cdiv(max_context_length, KV_BLOCK_SIZE)
    total_blocks = max_blocks_per_seq * batch_size

    total_queries = batch_size * query_length
    query = torch.randn(total_queries, num_query_heads, HEAD_SIZE,
                        dtype=torch.bfloat16, device=device)
    query.uniform_(*UNIFORM_RANGE)

    key_cache, value_cache = _create_kv_cache(
        total_blocks, KV_BLOCK_SIZE, num_kv_heads, HEAD_SIZE,
        torch.bfloat16, device)

    blocks_per_seq = triton.cdiv(context_length, KV_BLOCK_SIZE)
    block_tables_list = []
    for _ in range(batch_size):
        block_tables_list.append(
            [random.randint(0, total_blocks - 1) for _ in range(blocks_per_seq)])
    block_tables = torch.tensor(block_tables_list, dtype=torch.int32, device=device)
    context_lengths_t = torch.full((batch_size,), context_length,
                                   dtype=torch.int32, device=device)

    if quant_mode == "per_tensor":
        q_keys, k_scale_flat, q_vals, v_scale_flat, k_scale_orig, v_scale_orig = \
            _quantize_kv_per_tensor(key_cache, value_cache)
    else:
        q_keys, k_scale_flat, q_vals, v_scale_flat, k_scale_orig, v_scale_orig = \
            _quantize_kv_per_token(key_cache, value_cache)

    q_vals_shuffled = _shuffle_value_cache(q_vals)

    query_output_indptr = torch.arange(0, (batch_size + 1) * query_length,
                                       query_length, dtype=torch.int32, device=device)

    kv_page_indices, kv_indptr = _build_ps_page_data(
        block_tables_list, context_lengths_t, KV_BLOCK_SIZE, device)

    return {
        "query": query,
        "key_cache_q": q_keys,
        "value_cache_q": q_vals_shuffled,
        "key_cache_bf16": key_cache,
        "value_cache_bf16": value_cache,
        "key_scale_flat": k_scale_flat,
        "value_scale_flat": v_scale_flat,
        "key_scale_orig": k_scale_orig,
        "value_scale_orig": v_scale_orig,
        "block_tables": block_tables,
        "block_tables_list": block_tables_list,
        "context_lengths": context_lengths_t,
        "query_output_indptr": query_output_indptr,
        "kv_page_indices": kv_page_indices,
        "kv_indptr": kv_indptr,
        "num_query_heads": num_query_heads,
        "total_blocks": total_blocks,
        "softmax_scale": 1.0 / math.sqrt(HEAD_SIZE),
    }


def _torch_ref_attention(data, num_kv_heads):
    """PyTorch reference paged attention (FP8 dequant -> BF16 MHA)."""
    import torch
    query = data["query"]
    key_cache = data["key_cache_q"]
    value_cache_raw = data["value_cache_q"]
    block_tables = data["block_tables"]
    context_lengths = data["context_lengths"]
    query_output_indptr = data["query_output_indptr"]
    k_scale = data["key_scale_flat"]
    v_scale = data["value_scale_flat"]

    num_blocks, num_heads, head_size, block_size = data["value_cache_bf16"].shape
    softmax_scale = data["softmax_scale"]

    kv_dtype = key_cache.dtype
    key_flat = key_cache.permute(0, 1, 3, 2, 4).contiguous().view(-1, num_heads, head_size)
    elem_per_vec = 16 // value_cache_raw.element_size()
    v_unshuf = value_cache_raw.permute(0, 1, 3, 2, 4).contiguous()
    v_unshuf = v_unshuf.view(num_blocks, num_heads, block_size, head_size)
    val_flat = v_unshuf.permute(0, 2, 1, 3).contiguous().view(-1, num_heads, head_size)

    batch_size = query_output_indptr.shape[0] - 1
    queries_split = torch.tensor_split(query, query_output_indptr.tolist()[1:])

    outputs = []
    for b in range(batch_size):
        q_b = queries_split[b].float()
        bt = block_tables[b]
        ctx_len = context_lengths[b].item()
        tok_idx = (bt.repeat_interleave(block_size)[:ctx_len] * block_size
                   + torch.arange(ctx_len, device="cuda") % block_size)

        keys = key_flat.view(torch.int8)[tok_idx].view(kv_dtype).float()
        vals = val_flat.view(torch.int8)[tok_idx].view(kv_dtype).float()
        if k_scale is not None:
            keys *= k_scale[:, tok_idx].t().unsqueeze(-1)
        if v_scale is not None:
            vals *= v_scale[:, tok_idx].t().unsqueeze(-1)

        num_query_heads = q_b.shape[1]
        group_size = num_query_heads // num_kv_heads
        keys_expanded = keys.unsqueeze(1).expand(-1, group_size, -1, -1).reshape(
            ctx_len, num_query_heads, head_size) if group_size > 1 else keys
        vals_expanded = vals.unsqueeze(1).expand(-1, group_size, -1, -1).reshape(
            ctx_len, num_query_heads, head_size) if group_size > 1 else vals

        attn = torch.einsum("qhd,khd->hqk", q_b, keys_expanded) * softmax_scale
        probs = torch.softmax(attn, dim=-1)
        out = torch.einsum("hqk,khd->qhd", probs, vals_expanded)
        outputs.append(out)

    return torch.cat(outputs).to(torch.bfloat16)


def _precompute_ps_metadata(mod, data, num_kv_heads, query_length=1):
    """Pre-compute PS metadata (not part of kernel hot path)."""
    from kernel import get_sw_ps_max_context_partition_num

    metadata = mod.get_pa_metadata(
        data["query"], data["key_cache_q"],
        data["context_lengths"], data["kv_indptr"],
        data["num_query_heads"], num_kv_heads,
    )
    max_context_partition_num = get_sw_ps_max_context_partition_num(
        0, CONTEXT_PARTITION_SIZE, query_length)
    return metadata, max_context_partition_num


def _launch_ps_kernel(mod, data, metadata, max_context_partition_num, output):
    """Launch only the PA decode kernel (no metadata recomputation)."""
    mod.pa_decode_ps_launch(
        output, data["query"],
        data["key_cache_q"], data["value_cache_q"],
        data["context_lengths"],
        data["kv_page_indices"], data["kv_indptr"],
        data["softmax_scale"],
        key_scale=data["key_scale_orig"],
        value_scale=data["value_scale_orig"],
        sliding_window=0,
        metadata=metadata,
        block_tables=data["block_tables"],
        max_context_partition_num=max_context_partition_num,
    )


def _run_flydsl_ps(mod, data, num_kv_heads, query_length=1):
    """Run FlyDSL PS kernel and return output tensor."""
    import torch

    batch_size = data["context_lengths"].shape[0]
    total_queries = batch_size * query_length

    metadata, max_context_partition_num = _precompute_ps_metadata(
        mod, data, num_kv_heads, query_length)

    output = torch.empty(total_queries, data["num_query_heads"], HEAD_SIZE,
                         dtype=torch.bfloat16, device="cuda")

    _launch_ps_kernel(mod, data, metadata, max_context_partition_num, output)
    torch.cuda.synchronize()
    return output


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
    for i, (batch_size, context_length, num_kv_heads, query_length, quant_mode) in enumerate(shapes):
        try:
            _setup_seed(123 + i)
            data = _create_test_data(batch_size, context_length, num_kv_heads,
                                     query_length, quant_mode)

            output = _run_flydsl_ps(mod, data, num_kv_heads, query_length)
            ref = _torch_ref_attention(data, num_kv_heads)
            max_err = (output.float() - ref.float()).abs().max().item()
            passed = max_err < 0.15

            if not passed:
                raise AssertionError(f"max_err={max_err:.4e} > 0.15")

            cfg = f"(B={batch_size}, ctx={context_length}, kv_h={num_kv_heads}, ql={query_length}, {quant_mode})"
            results.append({"config": cfg, "correct": True})
            if verbose:
                print(f"  PASS: {cfg} max_err={max_err:.4e}")
        except Exception as e:
            cfg = f"(B={batch_size}, ctx={context_length}, kv_h={num_kv_heads}, ql={query_length}, {quant_mode})"
            failures.append({"config": cfg, "error": str(e)})
            if verbose:
                print(f"  FAIL: {cfg} - {str(e)[:120]}")

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

    print(f"Pre-compiling all {len(shapes)} shapes to eliminate JIT overhead...")
    for batch_size, context_length, num_kv_heads, query_length, quant_mode in shapes:
        try:
            _setup_seed(123)
            data = _create_test_data(batch_size, context_length, num_kv_heads,
                                     query_length, quant_mode)
            _run_flydsl_ps(mod, data, num_kv_heads, query_length)
        except Exception:
            pass
    torch.cuda.synchronize()

    print(f"Running benchmark on {len(shapes)} shapes, {warmup} warmup, {iters} iterations...")
    print(f"{'Config':<45} {'Ref':>10} {'FlyDSL':>10} {'Speedup':>10}")
    print("-" * 80)

    for idx, (batch_size, context_length, num_kv_heads, query_length, quant_mode) in enumerate(shapes):
        _setup_seed(123)
        data = _create_test_data(batch_size, context_length, num_kv_heads,
                                 query_length, quant_mode)

        total_queries = batch_size * query_length
        metadata, max_cp_num = _precompute_ps_metadata(
            mod, data, num_kv_heads, query_length)
        output = torch.empty(total_queries, data["num_query_heads"], HEAD_SIZE,
                             dtype=torch.bfloat16, device="cuda")

        for _ in range(warmup):
            _launch_ps_kernel(mod, data, metadata, max_cp_num, output)
            torch.cuda.synchronize()

        batch_n = 10
        kernel_times = []
        for _ in range(iters):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _b in range(batch_n):
                _launch_ps_kernel(mod, data, metadata, max_cp_num, output)
            e.record()
            torch.cuda.synchronize()
            kernel_times.append(s.elapsed_time(e) / batch_n)
        kernel_ms = sorted(kernel_times)[len(kernel_times) // 2]

        ref_times = []
        for _ in range(iters):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _b in range(batch_n):
                _torch_ref_attention(data, num_kv_heads)
            e.record()
            torch.cuda.synchronize()
            ref_times.append(s.elapsed_time(e) / batch_n)
        ref_ms = sorted(ref_times)[len(ref_times) // 2]

        speedup = ref_ms / kernel_ms if kernel_ms > 0 else 1.0
        latencies.append(kernel_ms)
        speedups.append(speedup)

        cfg = f"(B={batch_size}, ctx={context_length}, kv_h={num_kv_heads}, ql={query_length}, {quant_mode})"
        report_cases.append({
            "test_case_id": f"test_case_{idx}",
            "execution_time_ms": kernel_ms,
            "shape": [batch_size, context_length, num_kv_heads, query_length],
            "params": {"batch_size": batch_size, "context_length": context_length,
                       "num_kv_heads": num_kv_heads, "query_length": query_length,
                       "quant_mode": quant_mode},
        })

        marker = " *" if speedup > 1.0 else ""
        if verbose:
            print(f"{cfg:<45} {ref_ms:>8.4f}ms {kernel_ms:>8.4f}ms {speedup:>8.2f}x{marker}",
                  flush=True)

        torch.cuda.empty_cache()

    geomean_latency = math.exp(sum(math.log(l) for l in latencies) / len(latencies))
    geomean_speedup = math.exp(sum(math.log(s) for s in speedups) / len(speedups))

    build_dir = Path(_KERNEL_DIR) / "build"
    build_dir.mkdir(exist_ok=True)
    with open(build_dir / "performance_report.json", "w") as f:
        json.dump(report_cases, f, indent=2)

    print("-" * 80)
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
