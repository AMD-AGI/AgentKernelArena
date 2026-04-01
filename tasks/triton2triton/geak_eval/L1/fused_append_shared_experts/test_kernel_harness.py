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
import types
import importlib.util

# ── Constants ──────────────────────────────────────────────────────────────
WARMUP = 50
ITERATIONS = int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "200"))

# ── Resolve kernel location ───────────────────────────────────────────────
_REPO_REL = "python/sglang/srt/layers/moe/fused_moe_triton"
_KERNEL_FILENAME = "fused_moe_triton_kernels.py"


def _resolve_kernel_path():
    """Find the kernel file using GEAK env vars or fallback."""
    candidates = []
    work_dir = os.environ.get("GEAK_WORK_DIR")
    if work_dir:
        candidates.append(os.path.join(work_dir, _REPO_REL, _KERNEL_FILENAME))
    repo_root = os.environ.get("GEAK_REPO_ROOT")
    if repo_root:
        candidates.append(os.path.join(repo_root, _REPO_REL, _KERNEL_FILENAME))
    # Original location
    candidates.append(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     _REPO_REL, _KERNEL_FILENAME)
    )
    # Also check for kernel.py renamed in task directory
    candidates.append(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), _KERNEL_FILENAME)
    )
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        "Cannot find {} in any of: {}".format(_KERNEL_FILENAME, candidates)
    )


def _setup_sgl_kernel_mock():
    """Mock sgl_kernel so the kernel file can be imported on ROCm
    without the CUDA-only sgl_kernel native library."""
    if "sgl_kernel" in sys.modules:
        return
    mock_sgl = types.ModuleType("sgl_kernel")
    mock_sgl.__path__ = []
    mock_sgl.__file__ = "mock"

    def _noop(*a, **kw):
        return None

    for name in [
        "gelu_and_mul", "silu_and_mul", "moe_align_block_size",
        "moe_sum_reduce", "per_token_group_quant_fp8",
        "transfer_kv_all_layer",
    ]:
        setattr(mock_sgl, name, _noop)
    for submod_name in ["kvcacheio", "moe", "quantization", "elementwise"]:
        submod = types.ModuleType("sgl_kernel.{}".format(submod_name))
        for attr in ["transfer_kv_all_layer", "HostKVCache", "moe_align_block_size"]:
            setattr(submod, attr, _noop)
        sys.modules["sgl_kernel.{}".format(submod_name)] = submod
        setattr(mock_sgl, submod_name, submod)
    sys.modules["sgl_kernel"] = mock_sgl


def _load_kernel_module():
    """Load the kernel module directly, bypassing __init__.py chains."""
    _setup_sgl_kernel_mock()
    kernel_path = _resolve_kernel_path()
    # Walk up to find the 'python' directory and add it to sys.path
    parts = kernel_path.split(os.sep)
    for i, part in enumerate(parts):
        if part == "python":
            py_root = os.sep.join(parts[: i + 1])
            if py_root not in sys.path:
                sys.path.insert(0, py_root)
            break
    spec = importlib.util.spec_from_file_location(
        "sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_kernels",
        kernel_path,
        submodule_search_locations=[],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[
        "sglang.srt.layers.moe.fused_moe_triton.fused_moe_triton_kernels"
    ] = mod
    spec.loader.exec_module(mod)
    return mod


# ── Load kernel ───────────────────────────────────────────────────────────
_kernel_mod = _load_kernel_module()
fused_append_shared_experts = _kernel_mod.fused_append_shared_experts

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
    print("Running benchmark on {} configs ...".format(len(indices)))
    for idx in indices:
        cfg = ALL_CONFIGS[idx]
        topk_ids, topk_weights = build_inputs(cfg)
        # Warmup
        for _ in range(WARMUP):
            fused_append_shared_experts(
                topk_ids, topk_weights, cfg["S"], cfg["scale_factor"], N=cfg["N"]
            )
        torch.cuda.synchronize()
        # Timed iterations
        times = []
        for _ in range(ITERATIONS):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fused_append_shared_experts(
                topk_ids, topk_weights, cfg["S"], cfg["scale_factor"], N=cfg["N"]
            )
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
        times.sort()
        median_ms = times[len(times) // 2]
        latencies.append(median_ms)
        print("  [{}] {}  {:.4f}ms".format(idx, cfg_label(cfg), median_ms))
    # Geometric mean
    log_sum = sum(math.log(t) for t in latencies)
    geomean = math.exp(log_sum / len(latencies))
    print("GEAK_SHAPES_USED={}".format(indices))
    print("GEAK_RESULT_LATENCY_MS={:.4f}".format(geomean))


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
        indices = _pick(ALL_CONFIGS, 25)
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
