#!/usr/bin/env python3
"""Test harness for the FP8 block-scale GEMM kernel.

Timing, inputs, cases, tolerances, and the reference implementation live HERE,
not in kernel.py, because the agent edits kernel.py. The harness imports only
the candidate operation from the editable module.
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

_HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))
if _HARNESS_DIR not in sys.path:
    sys.path.insert(0, _HARNESS_DIR)

import torch

from kernel import fp8_blockwise_mm_triton

WARMUP = 50
ITERATIONS = int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "200"))
BLOCK_SHAPE_N = 128
BLOCK_SHAPE_K = 128
RTOL, ATOL = 2e-2, 1e-3

TEST_CONFIGS = [
    {"m": 64, "n": 64, "k": 128, "seed": 6635},
    {"m": 64, "n": 1536, "k": 7168, "seed": 6635},
    {"m": 64, "n": 3072, "k": 1536, "seed": 1236},
    {"m": 64, "n": 576, "k": 7168, "seed": 542},
    {"m": 96, "n": 7168, "k": 256, "seed": 1234},
    {"m": 96, "n": 7168, "k": 2048, "seed": 4153},
    {"m": 96, "n": 4608, "k": 7168, "seed": 412},
    {"m": 128, "n": 7168, "k": 2304, "seed": 624},
    {"m": 128, "n": 512, "k": 7168, "seed": 2514},
    {"m": 512, "n": 4096, "k": 512, "seed": 543},
    {"m": 512, "n": 1536, "k": 7168, "seed": 12341},
]

BENCHMARK_CONFIGS = [
    {"m": 1024, "n": 1536, "k": 7168, "seed": 8135},
    {"m": 1024, "n": 3072, "k": 1536, "seed": 6251},
    {"m": 1024, "n": 576, "k": 7168, "seed": 12346},
    {"m": 1024, "n": 7168, "k": 256, "seed": 5364},
    {"m": 1024, "n": 7168, "k": 2048, "seed": 6132},
    {"m": 1024, "n": 4608, "k": 7168, "seed": 7531},
    {"m": 1024, "n": 7168, "k": 2304, "seed": 12345},
    {"m": 1024, "n": 512, "k": 7168, "seed": 6563},
    {"m": 1024, "n": 4096, "k": 512, "seed": 17512},
    {"m": 6144, "n": 1536, "k": 7168, "seed": 6543},
    {"m": 6144, "n": 3072, "k": 1536, "seed": 234},
    {"m": 6144, "n": 576, "k": 7168, "seed": 9863},
    {"m": 6144, "n": 7168, "k": 256, "seed": 764243},
    {"m": 6144, "n": 7168, "k": 2048, "seed": 76547},
    {"m": 6144, "n": 4608, "k": 7168, "seed": 65436},
    {"m": 6144, "n": 7168, "k": 2304, "seed": 452345},
    {"m": 6144, "n": 512, "k": 7168, "seed": 12341},
    {"m": 6144, "n": 4096, "k": 512, "seed": 45245},
]

ALL_CONFIGS = TEST_CONFIGS + BENCHMARK_CONFIGS


def get_inputs(m, n, k, seed=42, device="cuda"):
    """Build representative inputs independently of the editable candidate."""
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    scale_n = (n + BLOCK_SHAPE_N - 1) // BLOCK_SHAPE_N
    scale_k = (k + BLOCK_SHAPE_K - 1) // BLOCK_SHAPE_K

    a = torch.randn((k, m), dtype=torch.bfloat16, device=device, generator=gen).to(
        torch.float8_e4m3fnuz
    )
    b = torch.randn((k, n), dtype=torch.bfloat16, device=device, generator=gen).to(
        torch.float8_e4m3fnuz
    )
    a_scale = torch.randn(
        [scale_k, m], dtype=torch.float32, device=device, generator=gen
    )
    b_scale = torch.randn(
        [scale_k, scale_n], dtype=torch.float32, device=device, generator=gen
    )
    c = torch.zeros((m, n), dtype=torch.bfloat16, device=device)
    return (a.T, b.T, a_scale.T, b_scale.T, c)


def fp8_blockwise_mm_pytorch(a, b, a_scale, b_scale, c):
    """Protected PyTorch reference for the blockwise-scaled multiplication."""
    a_c = a.contiguous()
    a_s = a_scale.contiguous()
    b_s = b_scale.contiguous()

    m, k = a_c.shape
    n = b.shape[0]
    sn, sk = b_s.shape

    a_sc = (
        a_s.unsqueeze(-1)
        .repeat(1, 1, BLOCK_SHAPE_K)
        .reshape(m, sk * BLOCK_SHAPE_K)[:, :k]
    )
    a_deq = a_c.to(a_sc.dtype) * a_sc

    b_sc = (
        b_s.view(-1, 1)
        .repeat(1, BLOCK_SHAPE_N * BLOCK_SHAPE_K)
        .view(sn, sk, BLOCK_SHAPE_N, BLOCK_SHAPE_K)
        .permute(0, 2, 1, 3)
        .reshape(sn * BLOCK_SHAPE_N, sk * BLOCK_SHAPE_K)
    )[:n, :k]
    b_deq = b.to(b_sc.dtype) * b_sc

    c[...] = (a_deq @ b_deq.T).to(torch.bfloat16)
    return c


def _pick(configs, count):
    if len(configs) <= count:
        return list(range(len(configs)))
    n = len(configs)
    return [round(i * (n - 1) / (count - 1)) for i in range(count)]


def _label(cfg):
    return "M={} N={} K={}".format(cfg["m"], cfg["n"], cfg["k"])


def check_correctness(cfg):
    a, b, a_scale, b_scale, c_triton = get_inputs(**cfg)
    c_ref = c_triton.clone()
    fp8_blockwise_mm_triton(a, b, a_scale, b_scale, c_triton)
    fp8_blockwise_mm_pytorch(a, b, a_scale, b_scale, c_ref)
    torch.cuda.synchronize()
    return torch.allclose(c_triton.float(), c_ref.float(), rtol=RTOL, atol=ATOL)


def _bench_one(cfg, warmup, iters):
    a, b, a_scale, b_scale, c = get_inputs(**cfg)
    output = c.clone()
    return benchmark_cuda_graph_or_events(
        lambda: fp8_blockwise_mm_triton(a, b, a_scale, b_scale, output),
        warmup=warmup,
        repetition=iters,
    )


def run_correctness(indices):
    torch.manual_seed(42)
    print("Running correctness on {} configs ...".format(len(indices)))
    all_ok = True
    for idx in indices:
        cfg = ALL_CONFIGS[idx]
        try:
            ok = check_correctness(cfg)
        except Exception as e:  # noqa: BLE001
            print("  [{}] {}  FAIL: {}".format(idx, _label(cfg), str(e)[:80]))
            all_ok = False
            continue
        if ok:
            print("  [{}] {}  PASS".format(idx, _label(cfg)))
        else:
            print("  [{}] {}  FAIL".format(idx, _label(cfg)))
            all_ok = False
    print("GEAK_SHAPES_USED={}".format(indices))
    if not all_ok:
        print("CORRECTNESS FAILED")
        sys.exit(1)
    print("All correctness checks passed.")


def run_benchmark(indices, warmup, iters):
    torch.manual_seed(42)
    print("Running benchmark on {} configs ...".format(len(indices)))
    latencies = []
    methods = []
    for idx in indices:
        cfg = ALL_CONFIGS[idx]
        ms, metadata = _bench_one(cfg, warmup, iters)
        latencies.append(ms)
        methods.append(metadata["benchmark_method"])
        print("  [{}] {}  {:.4f}ms".format(idx, _label(cfg), ms))
    geo = math.exp(sum(math.log(l) for l in latencies) / len(latencies))
    print("GEAK_SHAPES_USED={}".format(indices))
    print("GEAK_RESULT_LATENCY_MS={:.4f}".format(geo))
    print("GEAK_BENCHMARK_METHOD={}".format(
        methods[0] if len(set(methods)) == 1 else "mixed:" + ",".join(sorted(set(methods)))
    ))


def run_profile(indices):
    torch.manual_seed(42)
    print("Running profile on {} configs ...".format(len(indices)))
    for idx in indices:
        cfg = ALL_CONFIGS[idx]
        a, b, a_scale, b_scale, c = get_inputs(**cfg)
        for _ in range(3):
            fp8_blockwise_mm_triton(a, b, a_scale, b_scale, c.clone())
        torch.cuda.synchronize()
        print("  [{}] {}  done".format(idx, _label(cfg)))
    print("GEAK_SHAPES_USED={}".format(indices))


def main():
    parser = argparse.ArgumentParser(description="Test harness for fp8 blockwise mm")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--correctness", action="store_true")
    group.add_argument("--benchmark", action="store_true")
    group.add_argument("--full-benchmark", action="store_true")
    group.add_argument("--profile", action="store_true")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    args = parser.parse_args()
    iters = args.iterations if args.iterations is not None else ITERATIONS

    if args.correctness:
        run_correctness(_pick(ALL_CONFIGS, 25))
    elif args.benchmark:
        run_benchmark(_pick(ALL_CONFIGS, 25), args.warmup, iters)
    elif args.full_benchmark:
        run_benchmark(list(range(len(ALL_CONFIGS))), args.warmup, iters)
    elif args.profile:
        run_profile(_pick(ALL_CONFIGS, 5))


if __name__ == "__main__":
    main()
