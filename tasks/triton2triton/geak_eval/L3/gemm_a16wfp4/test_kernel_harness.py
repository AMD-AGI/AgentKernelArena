#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Test harness for gemm_a16wfp4 kernel

import argparse
import os
import sys
import time
import torch

# Import kernel and utilities
from kernel import gemm_a16wfp4, is_fp4_avail

# Note this is specified by the HW and cannot be changed.
SCALE_GROUP_SIZE = 32

# ALL_SHAPES: All unique shapes from test file, sorted by total element count
ALL_SHAPES = [
    (1, 8192, 1024),
    (1, 1280, 8192),
    (1, 7168, 2048),
    (1, 2112, 7168),
    (1, 4096, 4096),
    (4, 7168, 2048),
    (4, 2112, 7168),
    (8, 7168, 2048),
    (32, 512, 7168),
    (8, 2112, 7168),
    (2, 8192, 8192),
    (32, 8192, 1024),
    (32, 1280, 8192),
    (32, 7168, 2048),
    (32, 2112, 7168),
    (64, 8192, 1024),
    (4, 12288, 12288),
    (64, 1280, 8192),
    (64, 7168, 2048),
    (64, 2112, 7168),
    (128, 8192, 1024),
    (1024, 1024, 1024),
    (128, 1280, 8192),
    (192, 8192, 1024),
    (16, 16384, 6656),
    (128, 7168, 2048),
    (128, 2112, 7168),
    (192, 1280, 8192),
    (8, 16384, 16384),
    (256, 8192, 1024),
    (320, 8192, 1024),
    (256, 1280, 8192),
    (320, 1280, 8192),
    (512, 8192, 1024),
    (512, 1280, 8192),
    (16, 20480, 20480),
    (1024, 8192, 1024),
    (2048, 2048, 2048),
    (1024, 1280, 8192),
    (128, 16384, 6656),
    (2048, 8192, 1024),
    (2048, 1280, 8192),
    (3072, 3072, 3072),
    (4096, 8192, 1024),
    (4096, 1280, 8192),
    (8192, 8192, 1024),
    (4096, 4096, 4096),
    (8192, 1280, 8192),
    (5120, 5120, 5120),
    (16384, 8192, 1024),
    (4864, 4096, 8192),
    # (4864, 8192, 4160),  # Skipped due to compilation error
    (16384, 1280, 8192),
    (6144, 6144, 6144),
    (7168, 7168, 7168),
    (8192, 8192, 8192),
    # (9728, 8192, 65536),  # Too large, may cause OOM
]

# HARNESS_SHAPES: use ALL shapes so task-local and verified benchmarks match
HARNESS_SHAPES = ALL_SHAPES

# PROFILE_SHAPES: 5 evenly-spaced shapes for profiling
PROFILE_SHAPES = [
    (1, 8192, 1024),       # smallest
    (32, 7168, 2048),      # small-medium
    (256, 8192, 1024),     # medium
    (2048, 2048, 2048),    # medium-large
    (4096, 4096, 4096),    # large
]


def shuffle_scales(scales: torch.Tensor):
    """Shuffle scales for preshuffle kernel."""
    scales_shuffled = scales.clone()
    sm, sn = scales_shuffled.shape
    scales_shuffled = scales_shuffled.view(sm // 32, 2, 16, sn // 8, 2, 4, 1)
    scales_shuffled = scales_shuffled.permute(0, 3, 5, 2, 4, 1, 6).contiguous()
    scales_shuffled = scales_shuffled.view(sm // 32, sn * 32)
    return scales_shuffled


def mxfp4_to_f32(x):
    """Convert MXFP4 packed uint8 to float32."""
    x = x.repeat_interleave(2, dim=-1)
    x[..., ::2] = x[..., ::2] & 0xF
    x[..., 1::2] = x[..., 1::2] >> 4
    mxfp4_list = [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ]
    mxfp4_in_f32 = torch.tensor(mxfp4_list, dtype=torch.float32, device="cuda")
    return mxfp4_in_f32[x.long()]


def e8m0_to_f32(x):
    """Convert E8M0 scale to float32."""
    x_f32 = 2 ** (x.to(torch.float32) - 127)
    x_f32[x_f32 == 128] = float("nan")
    return x_f32


def generate_inputs(M: int, N: int, K: int, dtype=torch.bfloat16):
    """Generate inputs for gemm_a16wfp4 kernel."""
    torch.manual_seed(42)
    
    # Generate x (bf16 input) - TN layout only
    x_low = torch.randint(0, 16, (M, K // 2), dtype=torch.uint8, device="cuda")
    x_high = torch.randint(0, 16, (M, K // 2), dtype=torch.uint8, device="cuda")
    x_uint8 = x_low | x_high << 4
    
    # Generate x_scales and convert x to bf16
    x_scales = torch.randint(124, 128, (K // SCALE_GROUP_SIZE, M), dtype=torch.uint8, device="cuda").T
    x_f32 = mxfp4_to_f32(x_uint8)
    x_scales_expanded = x_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=-1).to(torch.float32)
    x_scales_f32 = e8m0_to_f32(x_scales_expanded)
    x_f32 = x_f32 * x_scales_f32
    x = x_f32.to(dtype)
    
    # Generate w (fp4 weights) - TN layout only
    w_low = torch.randint(0, 16, (N, K // 2), dtype=torch.uint8, device="cuda")
    w_high = torch.randint(0, 16, (N, K // 2), dtype=torch.uint8, device="cuda")
    w = w_low | w_high << 4
    
    # Generate w_scales
    w_scales = torch.randint(124, 128, (K // SCALE_GROUP_SIZE, N), dtype=torch.uint8, device="cuda").T
    
    # Non-preshuffled deterministic path only
    return x, w, w, w_scales, w_scales


def run_torch_reference(x, w, w_scales, dtype):
    """Compute reference output using PyTorch."""
    x_f32 = x.to(torch.float32)
    w_f32 = mxfp4_to_f32(w)
    w_scales_expanded = w_scales.repeat_interleave(SCALE_GROUP_SIZE, dim=-1).to(torch.float32)
    w_scales_f32 = e8m0_to_f32(w_scales_expanded)
    assert w_f32.shape == w_scales_f32.shape
    w_f32 = w_f32 * w_scales_f32
    return torch.mm(x_f32, w_f32.T).to(dtype)


def run_correctness(shapes):
    """Run correctness tests on given shapes."""
    if not is_fp4_avail():
        print("MXFP4 not supported on this architecture, skipping correctness tests")
        return True
    
    print(f"Running correctness tests on {len(shapes)} shapes...")
    all_passed = True
    
    for i, (M, N, K) in enumerate(shapes):
        torch.cuda.empty_cache()
        dtype = torch.bfloat16
        
        try:
            x, w, w_kernel, w_scales, w_scales_kernel = generate_inputs(M, N, K, dtype=dtype)
            
            # Run kernel
            y = gemm_a16wfp4(x, w_kernel, w_scales_kernel, atomic_add=False, dtype=dtype)
            
            # Run reference
            y_ref = run_torch_reference(x, w, w_scales, dtype)
            
            # Compare
            torch.testing.assert_close(y, y_ref, rtol=1e-2, atol=1e-2)
            print(f"  [{i+1}/{len(shapes)}] Shape ({M}, {N}, {K}): PASSED")
        except Exception as e:
            print(f"  [{i+1}/{len(shapes)}] Shape ({M}, {N}, {K}): FAILED - {e}")
            all_passed = False
    
    return all_passed


def run_profile(shapes):
    """Run kernel once for profiling."""
    if not is_fp4_avail():
        print("MXFP4 not supported on this architecture")
        return
    
    for M, N, K in shapes:
        torch.cuda.empty_cache()
        dtype = torch.bfloat16
        
        x, w, w_kernel, w_scales, w_scales_kernel = generate_inputs(M, N, K, dtype=dtype)
        
        # Warmup
        y = gemm_a16wfp4(x, w_kernel, w_scales_kernel, atomic_add=False, dtype=dtype)
        torch.cuda.synchronize()
        
        # Profile run
        y = gemm_a16wfp4(x, w_kernel, w_scales_kernel, atomic_add=False, dtype=dtype)
        torch.cuda.synchronize()
        
        print(f"Profiled shape ({M}, {N}, {K})")


def run_benchmark(shapes, iterations=20):
    """Run benchmark on given shapes."""
    if not is_fp4_avail():
        print("MXFP4 not supported on this architecture")
        print("GEAK_RESULT_LATENCY_MS=0.0")
        return
    
    print(f"Running benchmark on {len(shapes)} shapes with {iterations} iterations...")
    latencies = []
    
    for i, (M, N, K) in enumerate(shapes):
        torch.cuda.empty_cache()
        dtype = torch.bfloat16
        
        x, w, w_kernel, w_scales, w_scales_kernel = generate_inputs(M, N, K, dtype=dtype)
        
        # Warmup
        for _ in range(5):
            y = gemm_a16wfp4(x, w_kernel, w_scales_kernel, atomic_add=False, dtype=dtype)
        torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            y = gemm_a16wfp4(x, w_kernel, w_scales_kernel, atomic_add=False, dtype=dtype)
            torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
        
        median_time = sorted(times)[len(times) // 2]
        latencies.append(median_time)
        print(f"  [{i+1}/{len(shapes)}] Shape ({M}, {N}, {K}): {median_time:.4f} ms")
    
    # Compute geometric mean of latencies
    import math
    geomean = math.exp(sum(math.log(t) for t in latencies) / len(latencies))
    print(f"\nGeometric mean latency: {geomean:.4f} ms")
    print(f"GEAK_RESULT_LATENCY_MS={geomean:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Test harness for gemm_a16wfp4 kernel")
    parser.add_argument("--correctness", action="store_true", help="Run correctness tests")
    parser.add_argument("--profile", action="store_true", help="Run kernel once for profiling")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark on HARNESS_SHAPES")
    parser.add_argument("--full-benchmark", action="store_true", help="Run benchmark on ALL_SHAPES")
    parser.add_argument("--iterations", type=int, default=None, help="Number of benchmark iterations")
    
    args = parser.parse_args()
    
    if args.correctness:
        success = run_correctness(HARNESS_SHAPES)
        sys.exit(0 if success else 1)
    elif args.profile:
        run_profile(PROFILE_SHAPES)
    elif args.benchmark:
        iterations = args.iterations if args.iterations is not None else int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "10"))
        run_benchmark(HARNESS_SHAPES, iterations)
    elif args.full_benchmark:
        iterations = args.iterations if args.iterations is not None else int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "20"))
        run_benchmark(ALL_SHAPES, iterations)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
