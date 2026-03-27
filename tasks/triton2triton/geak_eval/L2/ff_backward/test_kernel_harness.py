#!/usr/bin/env python3
"""Test harness for the SwiGLU kernel (LigerSiLUMulFunction / swiglu_forward / swiglu_backward).

Modes:
  --correctness      : validate against torch reference (silu(a)*b), forward + backward
  --profile          : run up to 5 configs for profiling
  --benchmark        : run up to 25 configs, report per-shape latency + geo-mean
  --full-benchmark   : run ALL configs, report per-shape latency + geo-mean

Shape source: benchmark/scripts/benchmark_swiglu.py (adapted to direct kernel invocation).
Configs replicate the source's single-model (llama_3_8b) seq-len sweep with
dynamic batch_size/seq_len computed from GPU memory (same logic as
compute_seq_len_sweep_config).
"""

import argparse
import gc
import math
import os
import subprocess
import sys

# ---------------------------------------------------------------------------
# GPU device auto-detection: ensure torch.cuda works on ROCm systems.
# Must happen BEFORE importing torch.
# ---------------------------------------------------------------------------
def _fix_gpu_env():
    """Ensure at least one GPU is visible to PyTorch on ROCm systems.

    On some ROCm setups, ROCR_VISIBLE_DEVICES or HIP_VISIBLE_DEVICES may
    point to devices that PyTorch cannot open. We probe with a subprocess
    (so we don't taint the current process's CUDA init) and fix env vars.
    """
    # Quick probe: does the current env work?
    probe = subprocess.run(
        [sys.executable, "-c",
         "import torch; exit(0 if torch.cuda.is_available() and torch.cuda.device_count()>0 else 1)"],
        capture_output=True, timeout=30,
    )
    if probe.returncode == 0:
        return  # current env is fine

    # Try unsetting ROCR/HIP and using CUDA_VISIBLE_DEVICES
    test_env = dict(os.environ)
    for key in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        test_env.pop(key, None)
    if "CUDA_VISIBLE_DEVICES" not in test_env:
        test_env["CUDA_VISIBLE_DEVICES"] = "0"

    probe2 = subprocess.run(
        [sys.executable, "-c",
         "import torch; exit(0 if torch.cuda.is_available() and torch.cuda.device_count()>0 else 1)"],
        capture_output=True, timeout=30, env=test_env,
    )
    if probe2.returncode == 0:
        for key in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
            os.environ.pop(key, None)
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"

_fix_gpu_env()

# ---------------------------------------------------------------------------
# Resolve imports: prefer GEAK_WORK_DIR, then GEAK_REPO_ROOT + kernel subdir,
# then the original kernel directory.
# ---------------------------------------------------------------------------
_repo_root = os.environ.get(
    "GEAK_WORK_DIR",
    os.environ.get(
        "GEAK_REPO_ROOT",
        "/home/upandey/AIG-Eval/external_repos/Liger-Kernel",
    ),
)
_src_path = os.path.join(_repo_root, "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
_benchmark_scripts_path = os.path.join(_repo_root, "benchmark", "scripts")
if _benchmark_scripts_path not in sys.path:
    sys.path.insert(0, _benchmark_scripts_path)

import torch
import torch.nn.functional as F

from liger_kernel.ops.swiglu import LigerSiLUMulFunction  # noqa: E402
from benchmark_model_configs import (  # noqa: E402
    compute_seq_len_sweep_config,
    estimate_kernel_peak_memory,
    get_benchmark_model_config,
)
from benchmark_swiglu import _setup_swiglu  # noqa: E402
from utils import SingleBenchmarkRunInput  # noqa: E402

# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------
def _get_device():
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return "cuda"
    raise RuntimeError(
        "No GPU device available. "
        "Check CUDA_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES / ROCR_VISIBLE_DEVICES."
    )

DEVICE = _get_device()


def _cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _chunked_assert_close(actual, expected, atol, rtol, seq_chunk=128):
    """Compare large tensors in chunks to avoid temporary OOM in assert_close."""
    if actual.shape != expected.shape:
        raise AssertionError(f"shape mismatch: {actual.shape} != {expected.shape}")

    if actual.ndim < 2:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        return

    dim1 = actual.shape[1]
    for start in range(0, dim1, seq_chunk):
        end = min(start + seq_chunk, dim1)
        torch.testing.assert_close(
            actual[:, start:end],
            expected[:, start:end],
            atol=atol,
            rtol=rtol,
        )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WARMUP = 50
ITERATIONS = int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "200"))

# ---------------------------------------------------------------------------
# Full ordered config list.
# Each config: (bsz, seq_len, n_cols, dtype)
#
# Derived from benchmark_swiglu.py sweep.  The source file uses a single
# model config (llama_3_8b by default: hidden_size=4096,
# intermediate_size=14336, dtype=bf16) and sweeps sequence length as
# powers of 2 from 1024 up to config.seq_len (max_position_embeddings=8192).
# batch_size is computed dynamically from GPU memory; we replicate the
# computation here using the same helper.
# ---------------------------------------------------------------------------

def _build_configs():
    """Use the exact source helper flow from benchmark_swiglu.py."""
    model = get_benchmark_model_config(None)
    probe_seq_len = 1024

    def _probe():
        probe_input = SingleBenchmarkRunInput(
            x=probe_seq_len,
            kernel_provider="huggingface",
            extra_benchmark_config={
                "bsz": 1,
                "hidden_size": model.hidden_size,
                "intermediate_size": model.intermediate_size,
                "hidden_act": "silu",
                "dtype": model.dtype,
            },
        )
        x, layer = _setup_swiglu(probe_input)
        return layer(x)

    peak_bytes = estimate_kernel_peak_memory(probe_fn=_probe)
    kernel_bpt = peak_bytes // probe_seq_len
    config = compute_seq_len_sweep_config(model, kernel_bytes_per_token=kernel_bpt)

    x_values = [2**i for i in range(10, int(math.log2(config.seq_len)) + 1)]
    return [
        (config.batch_size, seq_len, model.intermediate_size, model.dtype)
        for seq_len in x_values
    ]

ALL_CONFIGS = _build_configs()


def _pick(configs, count):
    """Deterministic uniform subsample."""
    if len(configs) <= count:
        return list(range(len(configs))), configs
    n = len(configs)
    indices = [round(i * (n - 1) / (count - 1)) for i in range(count)]
    return indices, [configs[i] for i in indices]


# ---------------------------------------------------------------------------
# Reference implementation (pure PyTorch)
# ---------------------------------------------------------------------------
def torch_swiglu_forward(a, b):
    """Reference: silu(a) * b, with silu computed in float32 like the kernel."""
    return F.silu(a.float()).to(a.dtype) * b


def torch_swiglu_backward(a, b, dc):
    """Reference backward via autograd."""
    a_ref = a.clone().detach().requires_grad_(True)
    b_ref = b.clone().detach().requires_grad_(True)
    y = F.silu(a_ref.float()).to(a_ref.dtype) * b_ref
    y.backward(dc)
    return a_ref.grad, b_ref.grad


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------
def run_correctness(configs, indices):
    torch.manual_seed(42)

    print(f"Running correctness on {len(configs)} configs on {DEVICE}")
    all_pass = True

    for idx, (bsz, seq_len, n_cols, dtype) in zip(indices, configs):
        torch.manual_seed(42 + idx)
        try:
            a_data = torch.randn(bsz, seq_len, n_cols, device=DEVICE, dtype=dtype)
            b_data = torch.randn(bsz, seq_len, n_cols, device=DEVICE, dtype=dtype)

            # --- Forward ---
            a1 = a_data.clone().requires_grad_(True)
            b1 = b_data.clone().requires_grad_(True)
            y_liger = LigerSiLUMulFunction.apply(a1, b1)

            y_ref = torch_swiglu_forward(a_data, b_data)

            if dtype == torch.bfloat16:
                atol, rtol = 1e-1, 1e-2
            else:
                atol, rtol = 1e-4, 1e-5

            _chunked_assert_close(y_liger, y_ref, atol=atol, rtol=rtol)
        except AssertionError as e:
            print(f"  FAIL forward  config[{idx}] bsz={bsz} seq={seq_len} cols={n_cols} dtype={dtype}: {e}")
            all_pass = False
            _cleanup()
            continue

        # --- Backward ---
        dc = torch.randn_like(y_liger)
        y_liger.backward(dc.clone())
        da_liger = a1.grad
        db_liger = b1.grad

        da_ref, db_ref = torch_swiglu_backward(a_data, b_data, dc)

        try:
            _chunked_assert_close(da_liger, da_ref, atol=atol, rtol=rtol)
            _chunked_assert_close(db_liger, db_ref, atol=atol, rtol=rtol)
        except AssertionError as e:
            print(f"  FAIL backward config[{idx}] bsz={bsz} seq={seq_len} cols={n_cols} dtype={dtype}: {e}")
            all_pass = False
            _cleanup()
            continue

        print(f"  PASS config[{idx}] bsz={bsz} seq={seq_len} cols={n_cols} dtype={dtype}")
        del a_data, b_data, a1, b1, y_liger, y_ref, dc, da_liger, db_liger, da_ref, db_ref
        _cleanup()
    _cleanup()
    _cleanup()

    print(f"GEAK_SHAPES_USED={indices}")
    if not all_pass:
        print("CORRECTNESS FAILED")
        sys.exit(1)
    print("ALL CORRECTNESS CHECKS PASSED")


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def run_benchmark(configs, indices):
    torch.manual_seed(42)

    print(f"Running benchmark on {len(configs)} configs on {DEVICE}")
    latencies = []

    for idx, (bsz, seq_len, n_cols, dtype) in zip(indices, configs):
        a = torch.randn(bsz, seq_len, n_cols, device=DEVICE, dtype=dtype)
        b = torch.randn(bsz, seq_len, n_cols, device=DEVICE, dtype=dtype)

        # Benchmark forward + backward (full pass)
        def run_fn():
            a_t = a.clone().requires_grad_(True)
            b_t = b.clone().requires_grad_(True)
            y = LigerSiLUMulFunction.apply(a_t, b_t)
            y.backward(torch.randn_like(y))

        # Warmup
        for _ in range(WARMUP):
            run_fn()
        torch.cuda.synchronize()

        # Timed iterations
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
        print(f"  bsz={bsz} seq={seq_len} cols={n_cols} dtype={dtype}  {median_ms:.4f}ms")
        del a, b
        _cleanup()

    # Geometric mean
    log_sum = sum(math.log(t) for t in latencies)
    geo_mean = math.exp(log_sum / len(latencies))

    print(f"GEAK_SHAPES_USED={indices}")
    print(f"GEAK_RESULT_LATENCY_MS={geo_mean:.4f}")


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------
def run_profile(configs, indices):
    torch.manual_seed(42)

    print(f"Running profile on {len(configs)} configs on {DEVICE}")

    for idx, (bsz, seq_len, n_cols, dtype) in zip(indices, configs):
        a = torch.randn(bsz, seq_len, n_cols, device=DEVICE, dtype=dtype)
        b = torch.randn(bsz, seq_len, n_cols, device=DEVICE, dtype=dtype)

        a_t = a.clone().requires_grad_(True)
        b_t = b.clone().requires_grad_(True)
        y = LigerSiLUMulFunction.apply(a_t, b_t)
        y.backward(torch.randn_like(y))
        torch.cuda.synchronize()

        print(f"  Profiled config[{idx}] bsz={bsz} seq={seq_len} cols={n_cols} dtype={dtype}")
        del a, b, a_t, b_t, y
        _cleanup()

    print(f"GEAK_SHAPES_USED={indices}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="SwiGLU kernel test harness")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--correctness", action="store_true")
    group.add_argument("--profile", action="store_true")
    group.add_argument("--benchmark", action="store_true")
    group.add_argument("--full-benchmark", action="store_true")
    args = parser.parse_args()

    if args.correctness:
        indices, configs = _pick(ALL_CONFIGS, 25)
        run_correctness(configs, indices)
    elif args.profile:
        indices, configs = _pick(ALL_CONFIGS, 5)
        run_profile(configs, indices)
    elif args.benchmark:
        indices, configs = _pick(ALL_CONFIGS, 25)
        run_benchmark(configs, indices)
    elif args.full_benchmark:
        indices = list(range(len(ALL_CONFIGS)))
        run_benchmark(ALL_CONFIGS, indices)


if __name__ == "__main__":
    main()
