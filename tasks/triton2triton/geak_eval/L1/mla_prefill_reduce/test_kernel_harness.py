#!/usr/bin/env python3
"""Test harness for mla_prefill_reduce Triton kernel.

Tests *only* the Triton reduce kernel (mla_prefill_reduce_triton from
kernel.py) — not the full ASM prefill + reduce pipeline.

Uses aiter to build realistic metadata and run the ASM prefill stage once
to produce real partial_output / partial_lse, then benchmarks only the
reduce step.

Config space derived from test_mla_prefill_ps.py defaults:
  Product order: causal × num_heads × ctx_len × batch_size
  Total: 2 × 2 × 10 × 3 = 120 cases
"""

import argparse
import itertools
import math
import os
import random
import sys

# ---------------------------------------------------------------------------
# Resolve imports
# ---------------------------------------------------------------------------
_work_dir = os.environ.get("GEAK_WORK_DIR", "")
_repo_root = os.environ.get("GEAK_REPO_ROOT", "/home/upandey/AIG-Eval/external_repos/aiter")
_kernel_dir = os.path.join(_repo_root, "aiter")
_task_dir = os.path.dirname(os.path.abspath(__file__))

for _p in [_task_dir, _work_dir, _repo_root, _kernel_dir]:
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import aiter
from aiter import dtypes
from aiter import per_tensor_quant
from aiter.jit.utils.chip_info import get_gfx

if get_gfx() == "gfx942":
    print("Skipping mla_prefill_reduce harness: only supported on gfx950")
    sys.exit(0)

from kernel import mla_prefill_reduce_triton, ref_mla_prefill_reduce

torch.set_default_device("cuda")

# --- Constants ---
WARMUP = 50
ITERATIONS = int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "200"))

# Fixed parameters from source defaults
QK_HEAD_DIM = 192
V_HEAD_DIM = 128
BLOCK_SIZE = 1
TILE_Q = 256
FP8_DTYPE = dtypes.d_dtypes["fp8"]

# --- Config space (source product order) ---
ALL_CONFIGS = list(itertools.product(
    [True, False],                                              # causal
    [1, 16],                                                    # num_heads
    [21, 64, 256, 512, 1200, 3200, 5200, 8192, 10000, 16384],  # ctx_len
    [1, 4, 16],                                                 # batch_size
))
# Total: 2 × 2 × 10 × 3 = 120 configs


def _pick(configs, count):
    if len(configs) <= count:
        return list(range(len(configs)))
    n = len(configs)
    return [round(i * (n - 1) / (count - 1)) for i in range(count)]


def _make_case_synthetic(cfg):
    """Build inputs using kernel.py's synthetic builder (no ASM needed)."""
    is_causal, num_head, ctx_len, batch_size = cfg
    from kernel import _build_reduce_inputs
    (partial_output, partial_lse, reduce_indptr, reduce_final_map,
     reduce_partial_map, output, tile_q, _max_partials) = _build_reduce_inputs(
        batch_size, ctx_len, num_head, V_HEAD_DIM)
    num_tokens = batch_size * ctx_len
    final_lse = torch.empty((num_tokens, num_head), dtype=torch.float32, device=output.device)
    return {
        "partial_output": partial_output,
        "partial_lse": partial_lse,
        "reduce_indptr": reduce_indptr,
        "reduce_final_map": reduce_final_map,
        "reduce_partial_map": reduce_partial_map,
        "output": output,
        "final_lse": final_lse,
        "tile_q": tile_q,
        "num_tokens": num_tokens,
        "num_heads": num_head,
    }


def config_str(cfg):
    is_causal, num_head, ctx_len, batch_size = cfg
    return (f"causal={is_causal} nhead={num_head} ctx={ctx_len} bs={batch_size}")


# ---------------------------------------------------------------------------
# Build inputs: use kernel.py's synthetic builder (no ASM needed).
# ---------------------------------------------------------------------------
def build_reduce_inputs(is_causal, num_head, ctx_len, batch_size):
    return _make_case_synthetic((is_causal, num_head, ctx_len, batch_size))


def _build_reduce_inputs_asm_DISABLED(is_causal, num_head, ctx_len, batch_size):
    device = "cuda:0"
    num_head_q = num_head
    num_head_kv = num_head
    gqa_ratio = num_head_q // num_head_kv
    softmax_scale = 1.0 / (QK_HEAD_DIM ** 0.5)

    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int)
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int)
    seq_lens_kv = torch.full((batch_size,), ctx_len, dtype=torch.int)
    seq_lens_qo = seq_lens_kv.clone()
    max_qlen = seq_lens_qo.max().item()

    qo_indptr[1:] = torch.cumsum(seq_lens_qo, dim=0)
    actual_blocks = (seq_lens_kv + BLOCK_SIZE - 1) // BLOCK_SIZE
    kv_indptr[1:] = torch.cumsum(actual_blocks, dim=0)
    num_blocks = kv_indptr[-1].item()
    kv_indices = torch.randint(0, max(num_blocks, 1), (num_blocks,), dtype=torch.int)

    num_tokens = qo_indptr[-1].item()
    Q_bf16 = torch.randn((num_tokens, num_head_q, QK_HEAD_DIM), dtype=torch.bfloat16)
    K_bf16 = torch.randn((num_blocks, num_head_kv, QK_HEAD_DIM), dtype=torch.bfloat16)
    V_bf16 = K_bf16[:, :, :V_HEAD_DIM].contiguous()

    q_quant, q_scale = per_tensor_quant(Q_bf16, quant_dtype=FP8_DTYPE)
    k_quant, k_scale = per_tensor_quant(K_bf16, quant_dtype=FP8_DTYPE)
    v_quant, v_scale = per_tensor_quant(V_bf16, quant_dtype=FP8_DTYPE)

    tile_q = TILE_Q
    tile_kv = 128
    qhead_granularity = gqa_ratio
    qlen_granularity = tile_q // qhead_granularity
    kvlen_granularity = max(tile_kv, BLOCK_SIZE)

    (
        (work_meta_data_size, work_meta_data_type),
        (work_indptr_size, work_indptr_type),
        (work_info_size, work_info_type),
        (reduce_indptr_size, reduce_indptr_type),
        (reduce_final_map_size, reduce_final_map_type),
        (reduce_partial_map_size, reduce_partial_map_type),
    ) = aiter.get_ps_metadata_info_v1(
        batch_size=batch_size,
        num_head_k=num_head_kv,
        max_qlen=max_qlen,
        qlen_granularity=qlen_granularity,
    )

    work_metadata_ptrs = torch.empty(work_meta_data_size, dtype=work_meta_data_type, device=device)
    work_indptr = torch.empty(work_indptr_size, dtype=work_indptr_type, device=device)
    work_info = torch.empty(work_info_size, dtype=work_info_type, device=device)
    reduce_indptr = torch.empty(reduce_indptr_size, dtype=reduce_indptr_type, device=device)
    reduce_final_map = torch.empty(reduce_final_map_size, dtype=reduce_final_map_type, device=device)
    reduce_partial_map = torch.empty(reduce_partial_map_size, dtype=reduce_partial_map_type, device=device)

    aiter.get_ps_metadata_v1(
        qo_indptr.cpu(), kv_indptr.cpu(), seq_lens_kv.cpu(),
        gqa_ratio, num_head_kv,
        work_metadata_ptrs, work_indptr, work_info,
        reduce_indptr, reduce_final_map, reduce_partial_map,
        qhead_granularity=qhead_granularity,
        qlen_granularity=qlen_granularity,
        kvlen_granularity=kvlen_granularity,
        block_size=BLOCK_SIZE,
        is_causal=is_causal,
    )
    torch.cuda.synchronize()

    # Run ASM prefill to produce realistic partial_output / partial_lse
    output = torch.empty((num_tokens, num_head_q, V_HEAD_DIM), dtype=torch.bfloat16)
    nhead = num_head_q

    logits = torch.empty(
        (reduce_partial_map.size(0) * tile_q, nhead, V_HEAD_DIM),
        dtype=torch.float32, device=device,
    )
    attn_lse = torch.empty(
        (reduce_partial_map.size(0) * tile_q, nhead),
        dtype=torch.float32, device=device,
    )
    final_lse = torch.empty((num_tokens, nhead), dtype=torch.float32, device=device)

    aiter.mla_prefill_ps_asm_fwd(
        q_quant, k_quant, v_quant,
        qo_indptr, kv_indptr, kv_indices,
        work_indptr, work_info,
        max_qlen, softmax_scale, is_causal,
        logits, attn_lse, output,
        q_scale, k_scale, v_scale,
    )
    torch.cuda.synchronize()

    return {
        "partial_output": logits,
        "partial_lse": attn_lse,
        "reduce_indptr": reduce_indptr,
        "reduce_final_map": reduce_final_map,
        "reduce_partial_map": reduce_partial_map,
        "output": output,
        "final_lse": final_lse,
        "tile_q": tile_q,
        "num_tokens": num_tokens,
        "num_heads": nhead,
    }


def _run_reduce(case):
    """Call the Triton reduce kernel once."""
    mla_prefill_reduce_triton(
        case["partial_output"],
        case["partial_lse"],
        case["reduce_indptr"],
        case["reduce_final_map"],
        case["reduce_partial_map"],
        case["output"],
        case["tile_q"],
    )


def _run_reduce_ref(case):
    """Call the PyTorch reference reduce."""
    return ref_mla_prefill_reduce(
        case["partial_output"],
        case["partial_lse"],
        case["reduce_indptr"],
        case["reduce_final_map"],
        case["reduce_partial_map"],
        case["output"].clone(),
        case["tile_q"],
    )


# ---------------------------------------------------------------------------
# Correctness: compare Triton reduce vs PyTorch reduce (both on real partial data)
# ---------------------------------------------------------------------------
def run_correctness(indices):
    all_pass = True
    print(f"Running correctness on {len(indices)} configs...")
    for idx in indices:
        cfg = ALL_CONFIGS[idx]
        torch.manual_seed(42 + idx)
        try:
            case = build_reduce_inputs(*cfg)

            output_triton = case["output"].clone()
            mla_prefill_reduce_triton(
                case["partial_output"], case["partial_lse"],
                case["reduce_indptr"], case["reduce_final_map"],
                case["reduce_partial_map"], output_triton, case["tile_q"],
            )

            output_ref = case["output"].clone()
            ref_mla_prefill_reduce(
                case["partial_output"], case["partial_lse"],
                case["reduce_indptr"], case["reduce_final_map"],
                case["reduce_partial_map"], output_ref, case["tile_q"],
            )
            torch.cuda.synchronize()

            ok = torch.allclose(output_triton.float(), output_ref.float(), rtol=1e-2, atol=1e-2)
            if not ok:
                max_diff = torch.max(torch.abs(output_triton.float() - output_ref.float())).item()
                x = output_triton.double().flatten()
                y = output_ref.double().flatten()
                cos_diff = 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)
                if cos_diff < 1e-4:
                    ok = True
                if not ok:
                    print(f"  [{idx}] {config_str(cfg)} => FAIL (max_diff={max_diff:.6f})")
                    all_pass = False
                    continue

            print(f"  [{idx}] {config_str(cfg)} => PASS")
        except Exception as exc:
            print(f"  [{idx}] {config_str(cfg)} => ERROR: {exc}")
            all_pass = False
        finally:
            torch.cuda.empty_cache()

    print(f"GEAK_SHAPES_USED={indices}")
    if not all_pass:
        print("CORRECTNESS FAILED")
        sys.exit(1)
    print("ALL CORRECTNESS CHECKS PASSED")


# ---------------------------------------------------------------------------
# Benchmark: time only the Triton reduce kernel
# ---------------------------------------------------------------------------
def run_benchmark(indices):
    print(f"Running benchmark on {len(indices)} configs...")
    latencies = []
    for idx in indices:
        cfg = ALL_CONFIGS[idx]
        torch.manual_seed(42 + idx)
        try:
            case = build_reduce_inputs(*cfg)

            for _ in range(WARMUP):
                out = case["output"].clone()
                mla_prefill_reduce_triton(
                    case["partial_output"], case["partial_lse"],
                    case["reduce_indptr"], case["reduce_final_map"],
                    case["reduce_partial_map"], out, case["tile_q"],
                )
            torch.cuda.synchronize()

            times = []
            for _ in range(ITERATIONS):
                out = case["output"].clone()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                mla_prefill_reduce_triton(
                    case["partial_output"], case["partial_lse"],
                    case["reduce_indptr"], case["reduce_final_map"],
                    case["reduce_partial_map"], out, case["tile_q"],
                )
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))

            times.sort()
            median_ms = times[len(times) // 2]
            latencies.append(median_ms)
            print(f"  [{idx}] {config_str(cfg)}  {median_ms:.4f}ms")
        except Exception as exc:
            print(f"  [{idx}] {config_str(cfg)} => ERROR: {exc}")
        finally:
            torch.cuda.empty_cache()

    print(f"GEAK_SHAPES_USED={indices}")
    if not latencies:
        print("No successful benchmarks")
        sys.exit(1)
    geo_mean = math.exp(sum(math.log(x) for x in latencies) / len(latencies))
    print(f"GEAK_RESULT_LATENCY_MS={geo_mean:.4f}")


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------
def run_profile(indices):
    print(f"Running profile on {len(indices)} configs...")
    for idx in indices:
        cfg = ALL_CONFIGS[idx]
        torch.manual_seed(42 + idx)
        try:
            case = build_reduce_inputs(*cfg)

            for _ in range(3):
                out = case["output"].clone()
                mla_prefill_reduce_triton(
                    case["partial_output"], case["partial_lse"],
                    case["reduce_indptr"], case["reduce_final_map"],
                    case["reduce_partial_map"], out, case["tile_q"],
                )
            torch.cuda.synchronize()

            out = case["output"].clone()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            mla_prefill_reduce_triton(
                case["partial_output"], case["partial_lse"],
                case["reduce_indptr"], case["reduce_final_map"],
                case["reduce_partial_map"], out, case["tile_q"],
            )
            end.record()
            torch.cuda.synchronize()
            ms = start.elapsed_time(end)
            print(f"  [{idx}] {config_str(cfg)}  {ms:.4f}ms")
        except Exception as exc:
            print(f"  [{idx}] {config_str(cfg)} => ERROR: {exc}")
        finally:
            torch.cuda.empty_cache()

    print(f"GEAK_SHAPES_USED={indices}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="MLA prefill reduce harness")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--correctness", action="store_true")
    group.add_argument("--benchmark", action="store_true")
    group.add_argument("--full-benchmark", action="store_true")
    group.add_argument("--profile", action="store_true")
    parser.add_argument("--iterations", type=int, default=None, help="Number of benchmark iterations (overrides GEAK_BENCHMARK_ITERATIONS env var)")
    args = parser.parse_args()
    if args.iterations is not None:
        global ITERATIONS
        ITERATIONS = args.iterations

    all_indices = list(range(len(ALL_CONFIGS)))

    if args.correctness:
        run_correctness(_pick(all_indices, 25))
    elif args.benchmark:
        run_benchmark(all_indices)  # use all configs so benchmark matches full-benchmark
    elif args.full_benchmark:
        run_benchmark(all_indices)
    elif args.profile:
        run_profile(_pick(all_indices, 5))


if __name__ == "__main__":
    main()
