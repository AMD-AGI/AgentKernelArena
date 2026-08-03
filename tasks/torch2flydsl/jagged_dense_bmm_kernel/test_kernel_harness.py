#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Test harness for the torch2flydsl jagged_dense_bmm (jdbba) task.

Builds the PyTorch reference from model.py (`Model`/`get_inputs`/
`get_init_inputs`) and runs the FlyDSL kernel from kernel.py over a set of
jagged shapes, comparing for correctness and benchmarking against the torch
reference.

Modes:
  --correctness     compare FlyDSL output to the PyTorch Model reference at an
                    element-wise gate (max|ref-out| / max|ref| <= 1e-2)
  --full-benchmark  time FlyDSL vs the torch reference, write a perf report
"""
import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from _aka_benchmark import benchmark_cuda_graph_or_events

KERNEL_FILE = "kernel.py"
MODEL_FILE = "model.py"


def _resolve_kernel_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    if os.path.isfile(os.path.join(here, KERNEL_FILE)):
        return here
    cwd = os.getcwd()
    if os.path.isfile(os.path.join(cwd, KERNEL_FILE)):
        return cwd
    return here


def _load_module(kernel_dir, filename, alias):
    entry = os.path.join(kernel_dir, filename)
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

# Jagged configs (B groups, per-group row counts M_b). N=K=128 is the kernel's
# fixed problem shape. Includes unaligned M_b, M_b > BLOCK_M, empty groups, and
# a larger many-group case.
SHAPES = [
    {"name": "b4_varied", "m_per_group": [100, 128, 64, 200]},
    {"name": "b3_aligned", "m_per_group": [128, 256, 128]},
    {"name": "b5_with_empty", "m_per_group": [0, 130, 0, 300, 50]},
    {"name": "b8_small", "m_per_group": [10, 33, 128, 200, 1, 64, 129, 255]},
    {"name": "b2_single_tile", "m_per_group": [64, 96]},
]

N, K = 128, 128

# bf16 GEMM with fp32 accumulate: relative worst-element gate.
REL_GATE = 1e-2
SEED = 20260601


def _make_inputs(m_per_group, device="cuda"):
    import torch

    gen = torch.Generator(device=device)
    gen.manual_seed(SEED)
    B = len(m_per_group)
    total_M = sum(m_per_group)
    seq_offsets = torch.zeros(B + 1, dtype=torch.int32, device=device)
    for i, m in enumerate(m_per_group):
        seq_offsets[i + 1] = seq_offsets[i] + int(m)
    jagged = torch.randn((max(total_M, 1), K), generator=gen, device=device, dtype=torch.bfloat16)
    dense = torch.randn((B, N, K), generator=gen, device=device, dtype=torch.bfloat16)
    bias = torch.randn((B, N), generator=gen, device=device, dtype=torch.bfloat16)
    return jagged, dense, bias, seq_offsets


def _make_candidate_runner(kmod, jagged, dense, bias, seq_offsets, max_seq_len):
    """Prepare host shape metadata and stable buffers before graph capture."""
    import torch

    B, N, K = dense.shape
    L = jagged.shape[0]
    dense_tall = dense.reshape(B * N, K).contiguous()
    bias_flat = bias.reshape(B * N).contiguous()
    out = torch.zeros(
        (L + kmod.BLOCK_M, N), dtype=torch.bfloat16, device=jagged.device
    )
    tA = kmod.flyc.from_dlpack(jagged).mark_layout_dynamic(
        leading_dim=1, divisibility=8
    )
    tC = kmod.flyc.from_dlpack(out).mark_layout_dynamic(
        leading_dim=1, divisibility=8
    )

    def launch():
        kmod.jagged_dense_bmm(
            tC,
            tA,
            dense_tall,
            bias_flat,
            seq_offsets,
            B,
            max_seq_len,
            stream=kmod.fx.Stream(torch.cuda.current_stream()),
        )
        return out[:L]

    return launch


def _make_reference_runner(jagged, dense, bias, m_per_group):
    """Equivalent fixed-shape reference without timed seq_offsets.item()."""
    import torch

    out = torch.zeros(
        (jagged.shape[0], dense.shape[1]), dtype=jagged.dtype, device=jagged.device
    )
    groups = []
    start = 0
    for b, rows in enumerate(m_per_group):
        groups.append((b, start, start + rows))
        start += rows

    def launch():
        for b, start, end in groups:
            value = (
                jagged[start:end].float() @ dense[b].float().transpose(-1, -2)
                + bias[b].float()[None, :]
            ).to(jagged.dtype)
            out[start:end].copy_(value)
        return out

    return launch


def run_correctness(verbose=True):
    import torch

    kmod = _load_module(_KERNEL_DIR, KERNEL_FILE, "flydsl_kernel")
    mmod = _load_module(_KERNEL_DIR, MODEL_FILE, "torch_model")
    if kmod is None or mmod is None:
        print("FAIL: cannot load kernel.py / model.py")
        return {"correct": False}

    init = mmod.get_init_inputs()
    model = mmod.Model(*init).to("cuda").eval() if init else mmod.Model().to("cuda").eval()

    failures = []
    worst_rel = 0.0
    for shape in SHAPES:
        try:
            jagged, dense, bias, seq_offsets = _make_inputs(shape["m_per_group"])
            with torch.no_grad():
                ref = model(jagged, dense, bias, seq_offsets)
            out = kmod.flydsl_jagged_dense_bmm(jagged, dense, bias, seq_offsets)
            torch.cuda.synchronize()

            ref_f, out_f = ref.float(), out.float()
            denom = ref_f.abs().max().item()
            max_delta = (ref_f - out_f).abs().max().item()
            rel = max_delta / denom if denom > 0 else max_delta
            worst_rel = max(worst_rel, rel)
            ok = rel <= REL_GATE
            if verbose:
                print(
                    f"  {'PASS' if ok else 'FAIL'}: {shape['name']} "
                    f"(B={len(shape['m_per_group'])}, total_M={sum(shape['m_per_group'])}, "
                    f"N={N}, K={K}) rel_max={rel:.2e} (max_delta={max_delta:.5f}, "
                    f"max|ref|={denom:.5f})"
                )
            if not ok:
                failures.append(shape["name"])
        except Exception as e:  # noqa: BLE001
            failures.append(shape["name"])
            if verbose:
                print(f"  FAIL: {shape['name']} - {str(e)[:160]}")

    status = "ALL PASS" if not failures else f"FAILED ({len(failures)}/{len(SHAPES)})"
    print(f"Status: {status}")
    print(f"Tight gate: rel_max = max|ref-out| / max|ref| <= {REL_GATE:.0e}")
    print(f"Worst-element relative error across shapes: {worst_rel:.2e}")
    print(f"correctness: {'pass' if not failures else 'fail'}")
    assert not failures, f"correctness FAILED for: {failures}"
    return True


def run_benchmark(warmup=10, iters=100, verbose=True):
    import torch

    kmod = _load_module(_KERNEL_DIR, KERNEL_FILE, "flydsl_kernel")
    mmod = _load_module(_KERNEL_DIR, MODEL_FILE, "torch_model")
    if kmod is None or mmod is None:
        print("FAIL: cannot load kernel.py / model.py")
        return {"geomean_latency_ms": -1, "geomean_speedup": -1}

    model = mmod.Model().to("cuda").eval()

    latencies, speedups, report = [], [], []
    print(f"{'Config':<22} {'Ref':>10} {'FlyDSL':>10} {'Speedup':>10}")
    print("-" * 56)
    for idx, shape in enumerate(SHAPES):
        jagged, dense, bias, seq_offsets = _make_inputs(shape["m_per_group"])
        B = len(shape["m_per_group"])
        total_M = sum(shape["m_per_group"])
        run_kernel = _make_candidate_runner(
            kmod,
            jagged,
            dense,
            bias,
            seq_offsets,
            max(shape["m_per_group"]),
        )
        run_ref = _make_reference_runner(
            jagged, dense, bias, shape["m_per_group"]
        )

        run_kernel()
        torch.cuda.synchronize()
        for _ in range(warmup):
            run_kernel()
        torch.cuda.synchronize()

        kernel_ms, kernel_bench_meta = benchmark_cuda_graph_or_events(
            run_kernel, warmup=0, repetition=iters
        )

        with torch.no_grad():
            ref_ms, ref_bench_meta = benchmark_cuda_graph_or_events(
                run_ref,
                warmup=warmup,
                repetition=iters,
            )

        methods_match = kernel_bench_meta["benchmark_method"] == ref_bench_meta["benchmark_method"]
        speedup = (
            ref_ms / kernel_ms if methods_match and kernel_ms > 0 else None
        )
        speedup_display = (
            format(speedup, ">8.2f") + "x"
            if speedup is not None
            else f"{'N/A':>9}"
        )
        latencies.append(kernel_ms)
        if speedup is not None:
            speedups.append(speedup)
        flops = 2.0 * total_M * N * K
        tflops = flops / (kernel_ms * 1e-3) / 1e12 if kernel_ms > 0 else 0.0
        report.append({
            "test_case_id": f"test_case_{idx}",
            "execution_time_ms": kernel_ms,
            **kernel_bench_meta,
            "reference_benchmark_method": ref_bench_meta["benchmark_method"],
            "benchmark_method_consistent": kernel_bench_meta["benchmark_method"] == ref_bench_meta["benchmark_method"],
            "shape": [B, total_M, N, K],
            "params": {"B": B, "total_M": total_M, "N": N, "K": K, "dtype": "bf16"},
            "tflops": tflops,
        })
        if verbose:
            print(f"{shape['name']:<22} {ref_ms:>8.4f}ms {kernel_ms:>8.4f}ms {speedup_display}")
        del jagged, dense, bias, seq_offsets
        torch.cuda.empty_cache()

    geomean_latency = math.exp(sum(math.log(x) for x in latencies) / len(latencies))
    geomean_speedup = math.exp(sum(math.log(x) for x in speedups) / len(speedups)) if speedups else None
    geomean_speedup_display = (
        format(geomean_speedup, ".2f") + "x"
        if geomean_speedup is not None
        else "N/A"
    )

    build_dir = Path(_KERNEL_DIR) / "build"
    build_dir.mkdir(exist_ok=True)
    with open(build_dir / "performance_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("-" * 56)
    print(f"Geometric mean latency: {geomean_latency:.4f} ms")
    print(f"Geometric mean speedup: {geomean_speedup_display}")
    return {"geomean_latency_ms": geomean_latency, "geomean_speedup": geomean_speedup}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="torch2flydsl jdbba harness")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--full-benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    print("=" * 56)
    print("torch2flydsl jagged_dense_bmm_broadcast_add (jdbba)")
    print("=" * 56)

    if args.correctness:
        try:
            run_correctness()
        except AssertionError as exc:
            print(f"ASSERTION: {exc}")
            sys.exit(1)
        sys.exit(0)
    else:
        run_benchmark(warmup=args.warmup, iters=args.iterations)
