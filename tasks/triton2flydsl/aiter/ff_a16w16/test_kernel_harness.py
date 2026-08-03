#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Test harness for triton2flydsl/aiter/ff_a16w16 (FLAT layout).

The kernel under test is AITER's 16-bit ungated feed-forward block
(`ff_a16w16_nogate`), which composes two `_gemm_a16_w16_kernel` matmuls with an
activation on the up-projection:
    intermediate = act(X @ W_up^T);  Y = intermediate @ W_down^T
fp32 accumulation, bf16/fp16 output. The standalone source inlines the basic
a16w16 device kernel, the pid/XCD helpers, the elementwise activation helpers, and
replaces the on-disk tuned-config lookup with a static tile config (NUM_KSPLIT==1).

Modes:
  --compile         ast-parse + import the standalone source, assert entry/kernel symbols
  --correctness     run the Triton FFN on TEST_SHAPES x activations, assert finite
                    output AND match the fp32 torch reference at the upstream FFN
                    tolerance (atol=5e-2, rtol=5e-2)
                    [mirrors gemm/feed_forward/ff_test_utils.py:ff_ungated_test]
  --full-benchmark  graph-first GPU timing, write build/performance_report.json
"""
import argparse
import ast
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from _aka_benchmark import benchmark_cuda_graph_or_events

SOURCE_FILE = "ff_a16w16.py"
ENTRY = "ff_a16w16_nogate"
KERNEL = "_gemm_a16_w16_kernel"

# (batch, hidden_dim, intermediate_dim): real FFN shapes drawn from
# op_tests/.../gemm/basic/test_gemm_a16w16.py:get_x_vals (used by the ff tests),
# kept to a bounded subset (minimal/irregular + small/decode/GPT-OSS-style FFN).
TEST_SHAPES = [
    {"name": "b1_h1_i1", "batch": 1, "hidden": 1, "intermediate": 1},
    {"name": "b3_h5_i2", "batch": 3, "hidden": 5, "intermediate": 2},
    {"name": "b32_h1024_i1024", "batch": 32, "hidden": 1024, "intermediate": 1024},
    {"name": "b128_h2048_i8192", "batch": 128, "hidden": 2048, "intermediate": 8192},
    {"name": "b64_h5120_i2880", "batch": 64, "hidden": 5120, "intermediate": 2880},
]

ACTIVATIONS = ["gelu_tanh", "silu_exp2", "relu", None]
SEED = 0  # ff tests call torch.manual_seed(0)
WARMUP, ITERS = 10, 100

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_source():
    entry = os.path.join(_HERE, SOURCE_FILE)
    spec = importlib.util.spec_from_file_location("ff_a16w16_src", entry)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_inputs(batch, hidden, intermediate, dtype, device="cuda"):
    # Mirrors ff_test_utils.generate_ff_inputs (layout TN, gating=False).
    import torch

    torch.manual_seed(SEED)
    x = torch.randn((batch, hidden), dtype=dtype, device=device)
    w1 = torch.randn((intermediate, hidden), dtype=dtype, device=device)
    w2 = torch.randn((hidden, intermediate), dtype=dtype, device=device).T
    w1 = w1 / (intermediate**0.5)
    w2 = w2 / (hidden**0.5)
    return x, w1, w2


def _torch_ref(x, w1, w2, activation):
    # fp32 reference (matches ff_test_utils.ff_ungated_test).
    import torch.nn.functional as F

    torch_out = F.linear(x, w1, bias=None)
    if activation in ("gelu", "gelu_tanh"):
        torch_out = F.gelu(torch_out, approximate="tanh")
    elif activation in ("silu", "silu_exp2"):
        torch_out = F.silu(torch_out)
    elif activation == "relu":
        torch_out = F.relu(torch_out)
    elif activation is None:
        pass
    else:
        raise ValueError(f"Unsupported activation: {activation}")
    return torch_out @ w2


def run_compile():
    with open(os.path.join(_HERE, SOURCE_FILE)) as f:
        ast.parse(f.read())
    mod = _load_source()
    assert hasattr(mod, ENTRY), f"Missing entry {ENTRY}"
    assert hasattr(mod, KERNEL), f"Missing kernel {KERNEL}"
    print("Compilation: PASS")
    return True


def run_correctness(verbose=True):
    import torch

    mod = _load_source()
    dtype = torch.bfloat16
    failures = []
    for shape in TEST_SHAPES:
        for act in ACTIVATIONS:
            tag = f"{shape['name']}_{act or 'none'}"
            try:
                x, w1, w2 = _make_inputs(
                    shape["batch"], shape["hidden"], shape["intermediate"], dtype
                )
                y = mod.ff_a16w16_nogate(x, w1, w2, dtype, activation=act)
                torch.cuda.synchronize()
                ref = _torch_ref(x, w1, w2, act)
                finite = bool(torch.isfinite(y).all().item())
                close = torch.allclose(y, ref, atol=5e-2, rtol=5e-2)
                ok = finite and close
                if verbose:
                    print(
                        f"  {'PASS' if ok else 'FAIL'}: {tag} "
                        f"(b={shape['batch']},h={shape['hidden']},i={shape['intermediate']}) "
                        f"out={tuple(y.shape)} finite={finite} close={close}"
                    )
                if not ok:
                    failures.append(tag)
            except Exception as e:  # noqa: BLE001
                failures.append(tag)
                if verbose:
                    print(f"  FAIL: {tag} - {str(e)[:160]}")

    total = len(TEST_SHAPES) * len(ACTIVATIONS)
    status = "ALL PASS" if not failures else f"FAILED ({len(failures)}/{total})"
    print(f"Status: {status}")
    print(f"correctness: {'pass' if not failures else 'fail'}")
    return not failures


def run_benchmark(verbose=True):
    import torch

    mod = _load_source()
    dtype = torch.bfloat16
    report, latencies = [], []
    for idx, shape in enumerate(TEST_SHAPES):
        x, w1, w2 = _make_inputs(
            shape["batch"], shape["hidden"], shape["intermediate"], dtype
        )
        fn = lambda: mod.ff_a16w16_nogate(x, w1, w2, dtype, activation="silu_exp2")  # noqa: E731
        fn()
        torch.cuda.synchronize()
        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()
        ms, bench_meta = benchmark_cuda_graph_or_events(
            fn, warmup=0, repetition=ITERS
        )
        latencies.append(ms)
        flops = 4.0 * shape["batch"] * shape["hidden"] * shape["intermediate"]
        report.append(
            {
                "test_case_id": f"perf{idx + 1}",
                "execution_time_ms": ms,
                **bench_meta,
                "params": {k: shape[k] for k in ("batch", "hidden", "intermediate")},
                "tflops": flops / (ms * 1e-3) / 1e12,
            }
        )
        if verbose:
            print(f"  {shape['name']}: {ms:.4f} ms")

    build_dir = Path(_HERE) / "build"
    build_dir.mkdir(exist_ok=True)
    with open(build_dir / "performance_report.json", "w") as f:
        json.dump(report, f, indent=2)
    geomean = math.exp(sum(math.log(x) for x in latencies) / len(latencies))
    print(f"Geometric mean latency: {geomean:.4f} ms")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ff_a16w16 harness")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--full-benchmark", action="store_true")
    args = parser.parse_args()

    print("=" * 62)
    print("triton2flydsl 16-bit ungated feed-forward (ff_a16w16)")
    print("=" * 62)

    if args.compile:
        try:
            run_compile()
            sys.exit(0)
        except Exception as e:  # noqa: BLE001
            print(f"Compilation: FAIL\nError: {e}")
            sys.exit(1)
    elif args.correctness:
        sys.exit(0 if run_correctness() else 1)
    else:
        run_benchmark()
        sys.exit(0)
