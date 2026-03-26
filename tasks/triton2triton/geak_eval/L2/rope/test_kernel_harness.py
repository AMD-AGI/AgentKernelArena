#!/usr/bin/env python3
"""
Test harness for the RoPE (Rotary Position Embedding) Triton kernel.
Supports --correctness, --benchmark, --full-benchmark, --profile modes.

Shape source: op_tests/triton_tests/rope/test_rope.py (test_rope_sbhd_fwd parametrize)
"""

import argparse
import math
import os
import sys
import random

import torch

# ---------------------------------------------------------------------------
# Resolve repo root so imports work regardless of where the harness lives.
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

# ---------------------------------------------------------------------------
# Imports from the kernel package
# ---------------------------------------------------------------------------
from aiter.ops.triton.rope import (
    rope_fwd,
    rope_fwd_inplace,
    RotateStyle,
)

# ---------------------------------------------------------------------------
# Reference implementation from the test file
# ---------------------------------------------------------------------------
from op_tests.test_rope import ref_rope_sbhd_fwd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WARMUP = 50
ITERATIONS = int(os.environ.get("GEAK_BENCHMARK_ITERATIONS", "200"))


# ---------------------------------------------------------------------------
# generate_rope_inputs - copied from op_tests/triton_tests/rope/test_rope.py
# to avoid the triton driver import issue in op_tests.triton_tests.__init__
# ---------------------------------------------------------------------------
def generate_rope_inputs(
    B, S, H, Q, D,
    cached, reuse_freqs_front_part, nope,
    pos, offs, two_inputs, layout, dtype, bwd=False,
):
    torch.manual_seed(20)
    random.seed(20)

    device = "cuda"
    if layout == "thd":
        assert B == 1
        input_x_shape = (S, Q * H, D)
        input_y_shape = (S, H, D)
        pos_offs_shape = (S,)
    elif layout == "sbhd":
        input_x_shape = (S, B, Q * H, D)
        input_y_shape = (S, B, H, D)
        pos_offs_shape = (S, B)
    else:
        raise NotImplementedError(f"layout '{layout}' not supported")

    x = torch.randn(input_x_shape, dtype=dtype, device="cuda", requires_grad=bwd)
    y = (
        torch.randn(input_y_shape, dtype=dtype, device="cuda", requires_grad=bwd)
        if two_inputs else None
    )
    gx = torch.randn(input_x_shape, dtype=dtype, device="cuda") if bwd else None
    gy = (
        torch.randn(input_y_shape, dtype=dtype, device="cuda")
        if bwd and two_inputs else None
    )

    freqs_D = D
    if nope:
        freqs_D = freqs_D // 2
    if reuse_freqs_front_part:
        freqs_D = freqs_D // 2

    freqs = torch.randn((S, 1, 1, freqs_D), dtype=dtype, device="cuda")
    positions = (
        torch.randint(
            max(0, int(S * 0.25) if offs else 0),
            max(1, int(S * 0.75) if offs else S),
            pos_offs_shape, device=device,
        )
        if pos else None
    )
    offsets = (
        torch.randint(
            max(0, int(S * -0.25)),
            max(1, int(S * 0.25)),
            pos_offs_shape, device="cuda",
        )
        if offs else None
    )

    cos = torch.cos(freqs) if cached else None
    sin = torch.sin(freqs) if cached else None

    if cached and layout == "thd":
        cos = cos.reshape(S, freqs_D)
        sin = sin.reshape(S, freqs_D)

    return x, y, gx, gy, freqs, positions, offsets, cos, sin


# ---------------------------------------------------------------------------
# Build the ordered full case stream.
# Based on test_rope_sbhd_fwd parametrize – mirrors the full parameter space
# and pytest nesting order exactly.
# ---------------------------------------------------------------------------
def _build_all_configs():
    configs = []
    B_vals = [1, 2, 15, 32, 57]
    S_vals = [2, 10, 32]
    H_vals = [1, 8, 32]
    D_vals = [4, 64, 128]
    rotate_styles = [RotateStyle.GPTJ, RotateStyle.NEOX]
    nope_nope_first_vals = [(False, False), (True, False), (True, True)]
    reuse_vals = [False, True]
    dtype_vals = [torch.float16, torch.bfloat16]
    inplace_vals = [True, False]

    for inplace in inplace_vals:
        for dtype in dtype_vals:
            for reuse in reuse_vals:
                for nope, nope_first in nope_nope_first_vals:
                    for rs in rotate_styles:
                        for D in D_vals:
                            for H in H_vals:
                                for S in S_vals:
                                    for B in B_vals:
                                        configs.append({
                                            'B': B, 'S': S, 'H': H, 'D': D,
                                            'rotate_style': rs,
                                            'nope': nope, 'nope_first': nope_first,
                                            'reuse_freqs_front_part': reuse,
                                            'dtype': dtype,
                                            'inplace': inplace,
                                        })
    return configs


ALL_CONFIGS = _build_all_configs()


def _pick(configs, count):
    if len(configs) <= count:
        return list(range(len(configs))), configs
    n = len(configs)
    indices = [round(i * (n - 1) / (count - 1)) for i in range(count)]
    return indices, [configs[i] for i in indices]


def _config_label(cfg):
    rs_name = "GPTJ" if cfg['rotate_style'] == RotateStyle.GPTJ else "NEOX"
    dt_name = "fp16" if cfg['dtype'] == torch.float16 else "bf16"
    return (
        f"B={cfg['B']} S={cfg['S']} H={cfg['H']} D={cfg['D']} "
        f"rs={rs_name} nope={cfg['nope']} nf={cfg['nope_first']} "
        f"reuse={cfg['reuse_freqs_front_part']} dt={dt_name} "
        f"inplace={cfg['inplace']}"
    )


def _run_kernel(cfg):
    x, y, gx, gy, freqs, positions, offsets, cos, sin = generate_rope_inputs(
        cfg['B'], cfg['S'], cfg['H'], 1, cfg['D'],
        cached=False,
        reuse_freqs_front_part=cfg['reuse_freqs_front_part'],
        nope=cfg['nope'],
        pos=False, offs=False, two_inputs=False,
        layout="sbhd", dtype=cfg['dtype'],
    )
    if cfg['inplace']:
        out = rope_fwd_inplace(
            x, freqs,
            rotate_style=cfg['rotate_style'],
            reuse_freqs_front_part=cfg['reuse_freqs_front_part'],
            nope_first=cfg['nope_first'],
            transpose_output=False,
        )
    else:
        out = rope_fwd(
            x, freqs,
            rotate_style=cfg['rotate_style'],
            reuse_freqs_front_part=cfg['reuse_freqs_front_part'],
            nope_first=cfg['nope_first'],
            transpose_output=False,
        )
    return out


def _run_ref(cfg):
    x, y, gx, gy, freqs, positions, offsets, cos, sin = generate_rope_inputs(
        cfg['B'], cfg['S'], cfg['H'], 1, cfg['D'],
        cached=False,
        reuse_freqs_front_part=cfg['reuse_freqs_front_part'],
        nope=cfg['nope'],
        pos=False, offs=False, two_inputs=False,
        layout="sbhd", dtype=cfg['dtype'],
    )
    ref_out = ref_rope_sbhd_fwd(
        x, freqs,
        rotate_style=cfg['rotate_style'],
        reuse_freqs_front_part=cfg['reuse_freqs_front_part'],
        nope_first=cfg['nope_first'],
    )
    return ref_out


def _make_bench_fn(cfg):
    x, y, gx, gy, freqs, positions, offsets, cos, sin = generate_rope_inputs(
        cfg['B'], cfg['S'], cfg['H'], 1, cfg['D'],
        cached=False,
        reuse_freqs_front_part=cfg['reuse_freqs_front_part'],
        nope=cfg['nope'],
        pos=False, offs=False, two_inputs=False,
        layout="sbhd", dtype=cfg['dtype'],
    )
    if cfg['inplace']:
        def fn():
            return rope_fwd_inplace(
                x.clone(), freqs,
                rotate_style=cfg['rotate_style'],
                reuse_freqs_front_part=cfg['reuse_freqs_front_part'],
                nope_first=cfg['nope_first'],
                transpose_output=False,
            )
    else:
        def fn():
            return rope_fwd(
                x, freqs,
                rotate_style=cfg['rotate_style'],
                reuse_freqs_front_part=cfg['reuse_freqs_front_part'],
                nope_first=cfg['nope_first'],
                transpose_output=False,
            )
    return fn


# ---------------------------------------------------------------------------
# Mode: --correctness
# ---------------------------------------------------------------------------
def run_correctness():
    indices, configs = _pick(ALL_CONFIGS, 25)
    print(f"Running correctness on {len(configs)} configs...")
    all_pass = True
    for idx, cfg in zip(indices, configs):
        label = _config_label(cfg)
        try:
            triton_out = _run_kernel(cfg)
            ref_out = _run_ref(cfg)
            torch.testing.assert_close(triton_out, ref_out, atol=1e-1, rtol=1e-1)
            print(f"  [{idx}] PASS  {label}")
        except Exception as e:
            print(f"  [{idx}] FAIL  {label}: {e}")
            all_pass = False
    print(f"GEAK_SHAPES_USED={indices}")
    if not all_pass:
        print("CORRECTNESS FAILED")
        sys.exit(1)
    print("ALL CORRECTNESS CHECKS PASSED")


# ---------------------------------------------------------------------------
# Mode: --benchmark / --full-benchmark
# ---------------------------------------------------------------------------
def run_benchmark(full=False):
    if full:
        indices = list(range(len(ALL_CONFIGS)))
        configs = ALL_CONFIGS
    else:
        indices, configs = _pick(ALL_CONFIGS, 25)

    print(f"Running benchmark on {len(configs)} configs (WARMUP={WARMUP}, ITERATIONS={ITERATIONS})...")
    latencies = []
    for idx, cfg in zip(indices, configs):
        label = _config_label(cfg)
        fn = _make_bench_fn(cfg)

        # Warmup
        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()

        # Timed iterations
        times = []
        for _ in range(ITERATIONS):
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            fn()
            end_evt.record()
            torch.cuda.synchronize()
            times.append(start_evt.elapsed_time(end_evt))

        median_ms = sorted(times)[len(times) // 2]
        latencies.append(median_ms)
        print(f"  [{idx}] {label}  {median_ms:.4f}ms")

    print(f"GEAK_SHAPES_USED={indices}")
    log_sum = sum(math.log(t) for t in latencies)
    geo_mean = math.exp(log_sum / len(latencies))
    print(f"GEAK_RESULT_LATENCY_MS={geo_mean:.4f}")


# ---------------------------------------------------------------------------
# Mode: --profile
# ---------------------------------------------------------------------------
def run_profile():
    indices, configs = _pick(ALL_CONFIGS, 5)
    print(f"Running profile on {len(configs)} configs...")
    for idx, cfg in zip(indices, configs):
        label = _config_label(cfg)
        fn = _make_bench_fn(cfg)
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        fn()
        torch.cuda.synchronize()
        print(f"  [{idx}] {label}")
    print(f"GEAK_SHAPES_USED={indices}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="RoPE kernel test harness")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--correctness", action="store_true")
    group.add_argument("--benchmark", action="store_true")
    group.add_argument("--full-benchmark", action="store_true")
    group.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(42)

    if args.correctness:
        run_correctness()
    elif args.benchmark:
        run_benchmark(full=False)
    elif args.full_benchmark:
        run_benchmark(full=True)
    elif args.profile:
        run_profile()


if __name__ == "__main__":
    main()
