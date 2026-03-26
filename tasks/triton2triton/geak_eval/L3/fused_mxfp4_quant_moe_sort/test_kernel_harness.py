#!/usr/bin/env python3
"""
Test harness for fused_dynamic_mxfp4_quant_moe_sort kernel.
Modes: --correctness, --profile, --benchmark, --full-benchmark
"""

import argparse
import itertools
import math
import os
import sys

# Ensure line-buffered stdout
sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Resolve repo root so imports work regardless of where this script lives.
# ---------------------------------------------------------------------------
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
import triton

torch.manual_seed(42)

# ---------------------------------------------------------------------------
# Imports from the repo
# ---------------------------------------------------------------------------

# ── Dynamic kernel.py loader (matches old kernel pattern) ──────────────────
import importlib.util
import types

def _resolve_geak_kernel_dir():
    candidates = []
    work_dir = os.environ.get("GEAK_WORK_DIR", "").strip()
    if work_dir:
        candidates.append(work_dir)
    repo_root = os.environ.get("GEAK_REPO_ROOT", "").strip()
    if repo_root:
        candidates.append(os.path.join(repo_root, '.'))
    original_kernel_dir = os.path.dirname(os.path.abspath(__file__))
    if original_kernel_dir:
        candidates.append(original_kernel_dir)
    for candidate in candidates:
        if candidate and os.path.isfile(os.path.join(candidate, "kernel.py")):
            return candidate
    return original_kernel_dir or os.getcwd()

def _ensure_geak_package(module_name):
    parts = module_name.split(".")
    for idx in range(1, len(parts)):
        prefix = ".".join(parts[:idx])
        if prefix in sys.modules:
            continue
        pkg = types.ModuleType(prefix)
        pkg.__path__ = []
        sys.modules[prefix] = pkg

def _register_geak_aliases(kernel_dir):
    aliases = ['fused_mxfp4_quant', 'aiter.ops.triton.fused_mxfp4_quant']
    entry_file = os.path.join(kernel_dir, "kernel.py")
    if not os.path.isfile(entry_file):
        return
    for alias in aliases:
        if alias in sys.modules:
            continue
        _ensure_geak_package(alias)
        spec = importlib.util.spec_from_file_location(alias, entry_file)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[alias] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            pass

_KERNEL_DIR = _resolve_geak_kernel_dir()
if _KERNEL_DIR and _KERNEL_DIR not in sys.path:
    sys.path.insert(0, _KERNEL_DIR)
_register_geak_aliases(_KERNEL_DIR)
# ── End dynamic loader ─────────────────────────────────────────────────────

from aiter.ops.triton.fused_mxfp4_quant import (
    fused_dynamic_mxfp4_quant_moe_sort,
)
from op_tests.triton_tests.test_fused_mxfp4_quant import (
    run_fused_dynamic_mxfp4_quant_moe_sort_ref,
    run_fused_dynamic_mxfp4_quant_moe_sort_triton,
    convert_mxfp4_to_fp32,
)
from aiter.utility.fp4_utils import dynamic_mxfp4_quant

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WARMUP = 50
ITERATIONS = int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "200"))

# ---------------------------------------------------------------------------
# Build the ordered full case stream (matches pytest parametrize order)
# pytest decorators (top-to-bottom): hidden_dim, token_num, (tns,nvi), topk, dtype
# pytest iterates outermost = last decorator (dtype), innermost = first (hidden_dim)
# So: dtype (outer) x topk x (token_num_sort,num_valid_ids_0) x token_num x hidden_dim (inner)
# ---------------------------------------------------------------------------
_dtypes = [torch.bfloat16]
_topks = [1, 8]
_token_num_sort_valid = [(1, 1), (32, 32), (1024, 1024), (1024, 512)]
_token_nums = [1, 32, 1024]
_hidden_dims = [256]

ALL_CONFIGS_RAW = list(
    itertools.product(
        _dtypes,
        _topks,
        _token_num_sort_valid,
        _token_nums,
        _hidden_dims,
    )
)
# Repack each entry to (hidden_dim, token_num, (token_num_sort, num_valid_ids_0), topk, dtype)
# so downstream code stays unchanged.
ALL_CONFIGS = [
    (hd, tn, tns_nvi, topk, dtype)
    for dtype, topk, tns_nvi, tn, hd in ALL_CONFIGS_RAW
]


def _pick(configs, count):
    if len(configs) <= count:
        return list(range(len(configs)))
    n = len(configs)
    return [round(i * (n - 1) / (count - 1)) for i in range(count)]


def _make_inputs(cfg):
    """Build inputs for a single config, returns dict of tensors + metadata."""
    hidden_dim, token_num, (token_num_sort, num_valid_ids_0), topk, dtype = cfg
    block_size_M = 128
    q_dtype_a = torch.float4_e2m1fn_x2

    torch.manual_seed(42)

    num_valid_ids = torch.zeros(2, dtype=torch.int64, device="cuda")
    num_valid_ids[0] = num_valid_ids_0
    num_valid_ids[1] = token_num

    topk_ids = torch.randint(0, max(topk, 1), (token_num_sort,), device="cuda")
    topk_ids, _ = torch.sort(topk_ids)
    sorted_ids = torch.randint(0, token_num, (token_num_sort,), device="cuda")
    sorted_ids = (topk_ids << 24) | sorted_ids

    x = torch.randn((token_num, topk, hidden_dim), dtype=dtype, device="cuda") / 20
    x = x.view(-1, hidden_dim)

    return dict(
        x=x,
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=token_num,
        topk=topk,
        block_size_M=block_size_M,
        q_dtype_a=q_dtype_a,
        hidden_dim=hidden_dim,
        token_num_sort=token_num_sort,
        num_valid_ids_0=num_valid_ids_0,
        dtype=dtype,
    )


def _cfg_label(cfg):
    hidden_dim, token_num, (token_num_sort, num_valid_ids_0), topk, dtype = cfg
    return (
        f"hidden_dim={hidden_dim} token_num={token_num} "
        f"token_num_sort={token_num_sort} num_valid_ids_0={num_valid_ids_0} "
        f"topk={topk}"
    )


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------
def run_correctness(indices):
    print(f"Running correctness on {len(indices)} configs...")
    all_pass = True
    for idx in indices:
        cfg = ALL_CONFIGS[idx]
        label = _cfg_label(cfg)
        inp = _make_inputs(cfg)

        try:
            # Reference
            x_fp4_ref, x_scales_ref, x_scales_ref_not_sorted = (
                run_fused_dynamic_mxfp4_quant_moe_sort_ref(
                    inp["x"],
                    inp["sorted_ids"],
                    inp["token_num"],
                    inp["topk"],
                    inp["q_dtype_a"],
                    None,  # num_local_tokens
                    inp["num_valid_ids"],
                    inp["block_size_M"],
                )
            )

            # Triton
            x_fp4_triton, x_scales_triton = run_fused_dynamic_mxfp4_quant_moe_sort_triton(
                inp["x"],
                inp["sorted_ids"],
                inp["token_num"],
                inp["topk"],
                inp["q_dtype_a"],
                None,  # num_local_tokens
                inp["num_valid_ids"],
                inp["block_size_M"],
            )

            tol = 0.1
            nvi = inp["num_valid_ids"][0].item()
            x_scales_ref_c = x_scales_ref[:nvi]
            x_scales_triton_c = x_scales_triton[:nvi]
            torch.testing.assert_close(
                x_scales_ref_c.view(torch.uint8),
                x_scales_triton_c.view(torch.uint8),
                atol=tol,
                rtol=tol,
            )

            # Also check fp4 values via dequant round-trip
            _, x_scales_ref_triton_ns = dynamic_mxfp4_quant(inp["x"])
            x_scales_ref_triton_ns = x_scales_ref_triton_ns[
                : x_scales_ref_not_sorted.shape[0],
                : x_scales_ref_not_sorted.shape[1],
            ]
            x_ref = convert_mxfp4_to_fp32(
                x_fp4_ref.view(torch.uint8),
                x_scales_ref_not_sorted.view(torch.uint8),
            )
            x_triton = convert_mxfp4_to_fp32(
                x_fp4_triton.view(torch.uint8),
                x_scales_ref_triton_ns.view(torch.uint8),
            )
            torch.testing.assert_close(x_ref, x_triton, atol=tol, rtol=tol)

            print(f"  [{idx}] PASS  {label}")
        except Exception as e:
            print(f"  [{idx}] FAIL  {label}: {e}")
            all_pass = False

    return all_pass


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def run_benchmark(indices):
    print(f"Running benchmark on {len(indices)} configs...")
    latencies = []
    for idx in indices:
        cfg = ALL_CONFIGS[idx]
        label = _cfg_label(cfg)
        inp = _make_inputs(cfg)

        def fn():
            fused_dynamic_mxfp4_quant_moe_sort(
                inp["x"],
                sorted_ids=inp["sorted_ids"],
                num_valid_ids=inp["num_valid_ids"],
                token_num=inp["token_num"],
                topk=inp["topk"],
                block_size=inp["block_size_M"],
            )

        ms = triton.testing.do_bench(fn, warmup=WARMUP, rep=ITERATIONS)
        latencies.append(ms)
        print(f"  [{idx}] {label}  {ms:.4f}ms")

    # Geometric mean
    log_sum = sum(math.log(max(lat, 1e-12)) for lat in latencies)
    geo_mean = math.exp(log_sum / len(latencies))

    print(f"GEAK_SHAPES_USED={indices}")
    print(f"GEAK_RESULT_LATENCY_MS={geo_mean:.6f}")
    return geo_mean


# ---------------------------------------------------------------------------
# Profile (just run the kernel, no correctness)
# ---------------------------------------------------------------------------
def run_profile(indices):
    print(f"Running profile on {len(indices)} configs...")
    for idx in indices:
        cfg = ALL_CONFIGS[idx]
        label = _cfg_label(cfg)
        inp = _make_inputs(cfg)

        # Warmup
        for _ in range(3):
            fused_dynamic_mxfp4_quant_moe_sort(
                inp["x"],
                sorted_ids=inp["sorted_ids"],
                num_valid_ids=inp["num_valid_ids"],
                token_num=inp["token_num"],
                topk=inp["topk"],
                block_size=inp["block_size_M"],
            )
        torch.cuda.synchronize()

        # Timed run
        fused_dynamic_mxfp4_quant_moe_sort(
            inp["x"],
            sorted_ids=inp["sorted_ids"],
            num_valid_ids=inp["num_valid_ids"],
            token_num=inp["token_num"],
            topk=inp["topk"],
            block_size=inp["block_size_M"],
        )
        torch.cuda.synchronize()
        print(f"  [{idx}] profiled  {label}")

    print(f"GEAK_SHAPES_USED={indices}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--correctness", action="store_true")
    group.add_argument("--benchmark", action="store_true")
    group.add_argument("--full-benchmark", action="store_true")
    group.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    all_indices = list(range(len(ALL_CONFIGS)))

    if args.correctness:
        indices = _pick(ALL_CONFIGS, 25)
        ok = run_correctness(indices)
        print(f"GEAK_SHAPES_USED={indices}")
        if not ok:
            sys.exit(1)

    elif args.benchmark:
        indices = _pick(ALL_CONFIGS, 25)
        run_benchmark(indices)

    elif args.full_benchmark:
        run_benchmark(all_indices)

    elif args.profile:
        indices = _pick(ALL_CONFIGS, 5)
        run_profile(indices)


if __name__ == "__main__":
    main()
