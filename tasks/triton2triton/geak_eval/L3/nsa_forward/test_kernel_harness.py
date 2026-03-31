#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test harness for native_sparse_attention.ops.parallel (parallel_nsa kernel).

Modes:
  --correctness     : Validate parallel_nsa forward against naive_nsa reference.
  --benchmark       : Benchmark up to 25 configs (forward only).
  --full-benchmark  : Benchmark ALL configs (forward only).
  --profile         : Run 5 configs for profiling.
"""

import argparse
import math
import os
import sys
import warnings

# ---------------------------------------------------------------------------
# Ensure GPU visibility
# ---------------------------------------------------------------------------
for _var in ['HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES']:
    if _var in os.environ:
        del os.environ[_var]

# ---------------------------------------------------------------------------
# Resolve imports
# ---------------------------------------------------------------------------
_REPO_ROOT = '/home/upandey/AIG-Eval/external_repos/native-sparse-attention'

_work_dir = os.environ.get('GEAK_WORK_DIR', '')
_repo_root = os.environ.get('GEAK_REPO_ROOT', '')

if _work_dir and os.path.isdir(_work_dir):
    if _work_dir not in sys.path:
        sys.path.insert(0, _work_dir)
elif _repo_root and os.path.isdir(_repo_root):
    if _repo_root not in sys.path:
        sys.path.insert(0, _repo_root)
elif os.path.isdir(_REPO_ROOT):
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

os.environ['TRITON_F32_DEFAULT'] = 'ieee'

import torch
import triton

if not torch.cuda.is_available():
    print("ERROR: No GPU available.")
    sys.exit(1)

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=ImportWarning)

# The fla library bundled with native-sparse-attention registers model configs
# that conflict with newer transformers versions.  Patch the registration
# methods to silently skip duplicates.
import transformers.models.auto.configuration_auto as _auto_cfg
import transformers.models.auto.auto_factory as _auto_fac

if hasattr(_auto_cfg, '_LazyConfigMapping'):
    _orig_cfg_register = _auto_cfg._LazyConfigMapping.register
    def _safe_cfg_register(self, key, value, exist_ok=False):
        try:
            return _orig_cfg_register(self, key, value, exist_ok=True)
        except TypeError:
            try:
                return _orig_cfg_register(self, key, value)
            except (ValueError, KeyError):
                pass
        except (ValueError, KeyError):
            pass
    _auto_cfg._LazyConfigMapping.register = _safe_cfg_register

if hasattr(_auto_fac, '_BaseAutoModelClass'):
    _orig_model_register = _auto_fac._BaseAutoModelClass.register
    @classmethod
    def _safe_model_register(cls, *args, exist_ok=False, **kwargs):
        try:
            return _orig_model_register.__func__(cls, *args, exist_ok=True, **kwargs)
        except TypeError:
            try:
                return _orig_model_register.__func__(cls, *args, **kwargs)
            except (ValueError, KeyError):
                pass
        except (ValueError, KeyError):
            pass
    _auto_fac._BaseAutoModelClass.register = _safe_model_register

from native_sparse_attention.ops.parallel import parallel_nsa
from native_sparse_attention.ops.naive import naive_nsa

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WARMUP = 50
ITERATIONS = int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "200"))

# ---------------------------------------------------------------------------
# Config lists
# Each config: (B, T, H, HQ, D, S, block_size, window_size, scale)
#
# Correctness source: tests/test_nsa.py -> test_parallel
#   B=1, H=4, HQ=64, S=16, block_size=32, dtype=bf16, scale=0.1
#   T in [256, 1024, 2000], D in [100, 64], window_size in [0, 32]
#
# Benchmark source: benchmarks/benchmark_nsa.py
#   B=4, H=4, HQ=64, D=128, S=16, block_size=64, window_size=64, scale=None
#   T in [1024, 2048, 4096, 8192, 16384, 32768]
# ---------------------------------------------------------------------------
TEST_CONFIGS = [
    (1, 256,  4, 64, 100, 16, 32, 0,  0.1),
    (1, 256,  4, 64, 100, 16, 32, 32, 0.1),
    (1, 256,  4, 64, 64,  16, 32, 0,  0.1),
    (1, 256,  4, 64, 64,  16, 32, 32, 0.1),
    (1, 1024, 4, 64, 100, 16, 32, 0,  0.1),
    (1, 1024, 4, 64, 100, 16, 32, 32, 0.1),
    (1, 1024, 4, 64, 64,  16, 32, 0,  0.1),
    (1, 1024, 4, 64, 64,  16, 32, 32, 0.1),
    (1, 2000, 4, 64, 100, 16, 32, 0,  0.1),
    (1, 2000, 4, 64, 100, 16, 32, 32, 0.1),
    (1, 2000, 4, 64, 64,  16, 32, 0,  0.1),
    (1, 2000, 4, 64, 64,  16, 32, 32, 0.1),
]

BENCHMARK_CONFIGS = [
    (4, 1024,  4, 64, 128, 16, 64, 64, None),
    (4, 2048,  4, 64, 128, 16, 64, 64, None),
    (4, 4096,  4, 64, 128, 16, 64, 64, None),
    (4, 8192,  4, 64, 128, 16, 64, 64, None),
    (4, 16384, 4, 64, 128, 16, 64, 64, None),
    (4, 32768, 4, 64, 128, 16, 64, 64, None),
]


def _pick(configs, count):
    """Deterministic uniform subsample."""
    if len(configs) <= count:
        return list(range(len(configs)))
    n = len(configs)
    return [round(i * (n - 1) / (count - 1)) for i in range(count)]


def _config_str(cfg):
    B, T, H, HQ, D, S, block_size, window_size, scale = cfg
    s = "B={} T={} H={} HQ={} D={} S={} bs={} ws={}".format(
        B, T, H, HQ, D, S, block_size, window_size)
    if scale is not None:
        s += " scale={}".format(scale)
    return s


def _make_test_inputs(cfg, device='cuda', dtype=torch.bfloat16):
    """Create inputs matching tests/test_nsa.py."""
    B, T, H, HQ, D, S, block_size, window_size, scale = cfg
    torch.manual_seed(42)

    perm_q = torch.randperm(T, device=device)
    perm_k = torch.randperm(T, device=device)
    perm_v = torch.randperm(T, device=device)
    q = torch.linspace(0, 1, steps=T, dtype=dtype, device=device)[perm_q].view(1, T, 1, 1).expand(B, T, HQ, D).clone()
    k = torch.linspace(0, 1, steps=T, dtype=dtype, device=device)[perm_k].view(1, T, 1, 1).expand(B, T, H, D).clone()
    v = torch.linspace(0, 1, steps=T, dtype=dtype, device=device)[perm_v].view(1, T, 1, 1).expand(B, T, H, D).clone()
    g_slc = torch.rand((B, T, HQ), dtype=dtype, device=device)
    g_swa = torch.rand((B, T, HQ), dtype=dtype, device=device)

    block_indices = torch.full((B, T, H, S), T, dtype=torch.long, device=device)
    for b in range(B):
        for t in range(T):
            for h in range(H):
                i_i = torch.randperm(max(1, triton.cdiv(t, block_size)))[:S]
                block_indices[b, t, h, :len(i_i)] = i_i
    block_indices = block_indices.sort(-1)[0]
    block_counts = torch.randint(1, S + 1, (B, T, H), dtype=torch.long, device=device)

    return dict(
        q=q, k=k, v=v,
        g_slc=g_slc, g_swa=g_swa,
        block_indices=block_indices,
        block_counts=block_counts,
        block_size=block_size,
        window_size=window_size,
        scale=scale,
    )


def _make_benchmark_inputs(cfg, device='cuda', dtype=torch.bfloat16):
    """Create inputs matching benchmarks/benchmark_nsa.py."""
    B, T, H, HQ, D, S, block_size, window_size, scale = cfg
    q = torch.randn(B, T, HQ, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, T, H, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, T, H, D, device=device, dtype=dtype, requires_grad=True)
    g_slc = torch.rand((B, T, HQ), dtype=dtype, device=device).requires_grad_(True)
    g_swa = torch.rand((B, T, HQ), dtype=dtype, device=device).requires_grad_(True)

    block_indices = torch.full((B, T, H, S), T, dtype=torch.long, device=device)
    for b in range(B):
        for t in range(T):
            for h in range(H):
                i_i = torch.randperm(max(1, triton.cdiv(t, block_size)), device=device)[:S]
                block_indices[b, t, h, :len(i_i)] = i_i
    block_indices = block_indices.sort(-1)[0]
    block_counts = torch.randint(1, S + 1, (B, T, H), dtype=torch.long, device=device)

    return dict(
        q=q, k=k, v=v,
        g_slc=g_slc, g_swa=g_swa,
        block_indices=block_indices,
        block_counts=block_counts,
        block_size=block_size,
        window_size=window_size,
        scale=scale,
    )


def get_err_ratio(x, y):
    err = (x - y).flatten().square().mean().sqrt().item()
    base = x.flatten().square().mean().sqrt().item()
    return err / max(base, 1e-12)


def _call_parallel_nsa(inputs):
    """Call parallel_nsa with g_cmp=None (no compression path)."""
    return parallel_nsa(
        q=inputs['q'],
        k=inputs['k'],
        v=inputs['v'],
        g_cmp=None,
        g_slc=inputs['g_slc'],
        g_swa=inputs['g_swa'],
        block_indices=inputs['block_indices'],
        block_counts=inputs['block_counts'],
        block_size=inputs['block_size'],
        window_size=inputs['window_size'],
        scale=inputs['scale'],
    )


def _call_naive_nsa(inputs):
    """Call naive_nsa (no g_cmp parameter)."""
    return naive_nsa(
        q=inputs['q'],
        k=inputs['k'],
        v=inputs['v'],
        g_slc=inputs['g_slc'],
        g_swa=inputs['g_swa'],
        block_indices=inputs['block_indices'],
        block_counts=inputs['block_counts'],
        block_size=inputs['block_size'],
        window_size=inputs['window_size'],
        scale=inputs['scale'],
    )


# ---------------------------------------------------------------------------
# Correctness - forward output only
# ---------------------------------------------------------------------------
def run_correctness(indices):
    print("Running correctness on {} configs (forward only)...".format(len(indices)))
    all_pass = True
    for idx in indices:
        cfg = BENCHMARK_CONFIGS[idx]
        label = _config_str(cfg)
        try:
            inputs = _make_benchmark_inputs(cfg)

            # Reference (naive) - forward only
            with torch.no_grad():
                ref = _call_naive_nsa(inputs)

            # Triton (parallel) - forward only
            with torch.no_grad():
                tri = _call_parallel_nsa(inputs)

            ratio_o = get_err_ratio(ref, tri)

            status = "PASS" if ratio_o < 0.005 else "FAIL"
            if status == "FAIL":
                all_pass = False
            print("  [{}] {}  o_ratio={:.6f}".format(status, label, ratio_o))

            # Free memory
            del ref, tri, inputs
            torch.cuda.empty_cache()

        except Exception as e:
            print("  [FAIL] {}  Error: {}".format(label, e))
            import traceback
            traceback.print_exc()
            all_pass = False
            torch.cuda.empty_cache()

    print("GEAK_SHAPES_USED={}".format(indices))
    if not all_pass:
        print("CORRECTNESS FAILED")
        sys.exit(1)
    print("ALL CORRECTNESS CHECKS PASSED")


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def run_benchmark(indices):
    print("Running benchmark on {} configs...".format(len(indices)))
    latencies = []
    for idx in indices:
        cfg = BENCHMARK_CONFIGS[idx]
        label = _config_str(cfg)
        inputs = _make_benchmark_inputs(cfg)

        def run_fn():
            return _call_parallel_nsa(inputs)

        # Warmup
        for _ in range(WARMUP):
            run_fn()
        torch.cuda.synchronize()

        # Timed iterations using GPU events
        times = []
        for _ in range(ITERATIONS):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            run_fn()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        times.sort()
        median_ms = times[len(times) // 2]
        latencies.append(median_ms)
        print("  {}  {:.4f}ms".format(label, median_ms))

        del inputs
        torch.cuda.empty_cache()

    print("GEAK_SHAPES_USED={}".format(indices))
    # Geometric mean
    log_sum = sum(math.log(t) for t in latencies)
    geo_mean = math.exp(log_sum / len(latencies))
    print("GEAK_RESULT_LATENCY_MS={:.4f}".format(geo_mean))


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------
def run_profile(indices):
    print("Running profile on {} configs...".format(len(indices)))
    for idx in indices:
        cfg = BENCHMARK_CONFIGS[idx]
        label = _config_str(cfg)
        inputs = _make_benchmark_inputs(cfg)

        # Warmup
        for _ in range(3):
            _call_parallel_nsa(inputs)
        torch.cuda.synchronize()

        # Single timed run
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _call_parallel_nsa(inputs)
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end)
        print("  {}  {:.4f}ms".format(label, ms))

        del inputs
        torch.cuda.empty_cache()

    print("GEAK_SHAPES_USED={}".format(indices))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Test harness for parallel_nsa")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--correctness', action='store_true')
    group.add_argument('--benchmark', action='store_true')
    group.add_argument('--full-benchmark', action='store_true')
    group.add_argument('--profile', action='store_true')
    args = parser.parse_args()

    if args.correctness:
        indices = _pick(BENCHMARK_CONFIGS, 25)
        run_correctness(indices)
    elif args.benchmark:
        indices = list(range(len(BENCHMARK_CONFIGS)))  # use all configs so benchmark matches full-benchmark
        run_benchmark(indices)
    elif args.full_benchmark:
        indices = list(range(len(BENCHMARK_CONFIGS)))
        run_benchmark(indices)
    elif args.profile:
        indices = _pick(BENCHMARK_CONFIGS, 5)
        run_profile(indices)


if __name__ == '__main__':
    main()
