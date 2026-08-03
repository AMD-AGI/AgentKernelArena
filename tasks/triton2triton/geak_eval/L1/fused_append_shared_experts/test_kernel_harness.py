#!/usr/bin/env python3
"""
Test harness for fused_append_shared_experts kernel from
sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_kernels

Modes:
  --correctness      Validate kernel output against a pure-Python reference.
  --profile          Run 5 representative configs (for profiling tools).
  --benchmark        Run up to 25 configs, report per-shape latency + geomean.
  --full-benchmark   Run ALL configs, report per-shape latency + geomean.
"""

import argparse
import math
import os
import sys
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

# ── Constants ──────────────────────────────────────────────────────────────
WARMUP = 50
ITERATIONS = int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "200"))

# kernel.py lives next to this harness; Python puts the script dir on sys.path[0].
_HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))
if _HARNESS_DIR not in sys.path:
    sys.path.insert(0, _HARNESS_DIR)

from kernel import fused_append_shared_experts

import torch

# ── Config list (ordered full case stream) ────────────────────────────────
# Source of truth for the case stream:
#   common_utils.get_default_batch_sizes()
#   [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 1536, 2048, 3072, 4096]
#
# This kernel is called from topk.py with:
#   - M = router_logits batch/token count
#   - K = routed top-k width before shared experts are appended
#   - S = num_fused_shared_experts
#   - N = base expert count used as the starting shared-expert id
#
# There is no repo-native benchmark that sweeps K/S/N for this specific kernel,
# so keep the source batch-size stream and use one real call-site-style tuple:
#   K = 2, N = 8 from the default SGLang fused-MoE benchmark model path
#   S = 1 because SGLang shared-expert model paths assert one fused shared expert
#   scale_factor = 1.0 (topk.py default when no explicit scaling factor is provided)
_BATCH_SIZES = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024, 1536, 2048, 3072, 4096]
_ROUTED_TOPK = 2
_NUM_SHARED = 1
_NUM_BASE_EXPERTS = 8
_SCALE_FACTOR = 1.0

ALL_CONFIGS = [
    {"M": M, "K": _ROUTED_TOPK, "S": _NUM_SHARED, "N": _NUM_BASE_EXPERTS, "scale_factor": _SCALE_FACTOR}
    for M in _BATCH_SIZES
]


# ── Subsetting ────────────────────────────────────────────────────────────
def _pick(configs, count):
    if len(configs) <= count:
        return list(range(len(configs)))
    n = len(configs)
    return [round(i * (n - 1) / (count - 1)) for i in range(count)]


# ── Reference implementation ─────────────────────────────────────────────
def reference_fused_append(topk_ids, topk_weights, S, scale_factor, N):
    """Pure-PyTorch reference for correctness checking."""
    M, K = topk_ids.shape
    out_ids = torch.empty((M, K + S), dtype=topk_ids.dtype, device=topk_ids.device)
    out_weights = torch.empty(
        (M, K + S), dtype=topk_weights.dtype, device=topk_ids.device
    )
    out_ids[:, :K] = topk_ids
    out_weights[:, :K] = topk_weights
    for s in range(S):
        out_ids[:, K + s] = N + s
        out_weights[:, K + s] = scale_factor
    return out_ids, out_weights


# ── Build inputs ──────────────────────────────────────────────────────────
def build_inputs(cfg, device="cuda"):
    M, K, S, N = cfg["M"], cfg["K"], cfg["S"], cfg["N"]
    topk_ids = torch.randint(0, N, (M, K), dtype=torch.int32, device=device)
    topk_weights = torch.rand(M, K, dtype=torch.float32, device=device)
    return topk_ids, topk_weights


# ── Config label ──────────────────────────────────────────────────────────
def cfg_label(cfg):
    return "M={} K={} S={} N={}".format(cfg["M"], cfg["K"], cfg["S"], cfg["N"])


# ── Correctness ───────────────────────────────────────────────────────────
def run_correctness(indices):
    torch.manual_seed(42)
    print("Running correctness on {} configs ...".format(len(indices)))
    for idx in indices:
        cfg = ALL_CONFIGS[idx]
        topk_ids, topk_weights = build_inputs(cfg)
        # Kernel under test
        out_ids, out_weights = fused_append_shared_experts(
            topk_ids, topk_weights, cfg["S"], cfg["scale_factor"], N=cfg["N"]
        )
        # Reference
        ref_ids, ref_weights = reference_fused_append(
            topk_ids, topk_weights, cfg["S"], cfg["scale_factor"], cfg["N"]
        )
        torch.testing.assert_close(out_ids, ref_ids, atol=0, rtol=0)
        torch.testing.assert_close(out_weights, ref_weights, atol=1e-6, rtol=1e-5)
        print("  [{}] {}  PASS".format(idx, cfg_label(cfg)))
    print("GEAK_SHAPES_USED={}".format(indices))
    print("All correctness checks passed.")


# ── Benchmark ─────────────────────────────────────────────────────────────
def run_benchmark(indices):
    torch.manual_seed(42)
    latencies = []
    methods = []
    print("Running benchmark on {} configs ...".format(len(indices)))
    for idx in indices:
        cfg = ALL_CONFIGS[idx]
        topk_ids, topk_weights = build_inputs(cfg)
        def kernel_call():
            fused_append_shared_experts(
                topk_ids, topk_weights, cfg["S"], cfg["scale_factor"], N=cfg["N"]
            )
        latency_ms, metadata = benchmark_cuda_graph_or_events(
            kernel_call, warmup=WARMUP, repetition=ITERATIONS
        )
        latencies.append(latency_ms)
        methods.append(metadata["benchmark_method"])
        print("  [{}] {}  {:.4f}ms".format(idx, cfg_label(cfg), latency_ms))
    # Geometric mean
    log_sum = sum(math.log(t) for t in latencies)
    geomean = math.exp(log_sum / len(latencies))
    print("GEAK_SHAPES_USED={}".format(indices))
    print("GEAK_RESULT_LATENCY_MS={:.4f}".format(geomean))
    print("GEAK_BENCHMARK_METHOD={}".format(
        methods[0] if len(set(methods)) == 1 else "mixed:" + ",".join(sorted(set(methods)))
    ))


# ── Profile ───────────────────────────────────────────────────────────────
def run_profile(indices):
    torch.manual_seed(42)
    print("Running profile on {} configs ...".format(len(indices)))
    for idx in indices:
        cfg = ALL_CONFIGS[idx]
        topk_ids, topk_weights = build_inputs(cfg)
        # Warmup
        for _ in range(WARMUP):
            fused_append_shared_experts(
                topk_ids, topk_weights, cfg["S"], cfg["scale_factor"], N=cfg["N"]
            )
        torch.cuda.synchronize()
        # Single timed run for profiler
        for _ in range(10):
            fused_append_shared_experts(
                topk_ids, topk_weights, cfg["S"], cfg["scale_factor"], N=cfg["N"]
            )
        torch.cuda.synchronize()
        print("  [{}] {}  done".format(idx, cfg_label(cfg)))
    print("GEAK_SHAPES_USED={}".format(indices))


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Test harness for fused_append_shared_experts"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--correctness", action="store_true")
    group.add_argument("--profile", action="store_true")
    group.add_argument("--benchmark", action="store_true")
    group.add_argument("--full-benchmark", action="store_true")
    parser.add_argument("--iterations", type=int, default=None, help="Number of benchmark iterations (overrides GEAK_BENCHMARK_ITERATIONS env var)")
    args, _ = parser.parse_known_args()
    if args.iterations is not None:
        global ITERATIONS
        ITERATIONS = args.iterations

    if args.correctness:
        indices = list(range(len(ALL_CONFIGS)))
        run_correctness(indices)
    elif args.profile:
        indices = _pick(ALL_CONFIGS, 5)
        run_profile(indices)
    elif args.benchmark:
        indices = list(range(len(ALL_CONFIGS)))  # use all configs so benchmark matches full-benchmark
        run_benchmark(indices)
    elif args.full_benchmark:
        indices = list(range(len(ALL_CONFIGS)))
        run_benchmark(indices)


if __name__ == "__main__":
    main()
