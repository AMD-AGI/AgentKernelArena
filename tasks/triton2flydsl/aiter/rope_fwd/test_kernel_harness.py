#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Test harness for triton2flydsl/aiter/rope_fwd (FLAT layout).

The kernel under test is AITER's RoPE forward (sbhd) Triton kernel (`rope_fwd` ->
`_rope_kernel_sbhd_fwd` + `_get_neox_rotated_x` / `_get_gptj_rotated_x`): rotary
position embedding over [S, B, H, D] with cos/sin(freqs) computed in fp32,
NEOX/GPTJ rotate_style, `reuse_freqs_front_part`, and an optional NOPE
(non-rotary) split (`nope_first`). Output is the input dtype.

Modes:
  --compile         ast-parse + import the standalone source, assert entry/kernel symbols
  --correctness     run the Triton kernel on TEST_SHAPES x configs, assert finite
                    output AND match the torch reference at the upstream tolerance
                    (atol=1e-1, rtol=1e-1)
                    [mirrors op_tests/test_rope.py:ref_rope_sbhd_fwd]
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

SOURCE_FILE = "rope_fwd.py"
ENTRY = "rope_fwd"
KERNEL = "_rope_kernel_sbhd_fwd"

# (B, S, H, D) from op_tests/triton_tests/rope/test_rope.py::test_rope_sbhd_fwd
# (B in {1,32}, S in {1,32}, H=8, D=64) crossed with rotate_style (NEOX/GPTJ),
# (nope, nope_first), and reuse_freqs_front_part.
BS = [1, 32]
SS = [1, 32]
H = 8
D = 64
NOPE_CFG = [(False, False), (True, False), (True, True)]  # (nope, nope_first)
REUSE = [False, True]

SEED = 20  # match upstream generate_rope_inputs (torch.manual_seed(20))
WARMUP, ITERS = 10, 100

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load_source():
    entry = os.path.join(_HERE, SOURCE_FILE)
    spec = importlib.util.spec_from_file_location("rope_fwd_src", entry)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_inputs(B, S, nope, reuse, device="cuda"):
    # Mirrors generate_rope_inputs (sbhd, cached=False, two_inputs=False).
    import torch

    torch.manual_seed(SEED)
    x = torch.randn((S, B, H, D), dtype=torch.bfloat16, device=device)
    freqs_D = D
    if nope:
        freqs_D = freqs_D // 2
    if reuse:
        freqs_D = freqs_D // 2
    freqs = torch.randn((S, 1, 1, freqs_D), dtype=torch.bfloat16, device=device)
    return x, freqs


def _rotate_half_neox(x):
    import torch

    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _rotate_half_gptj(x):
    import torch

    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


def _ref_rope_sbhd_fwd(x, freqs, rotate_style, reuse_freqs_front_part, nope_first, NEOX):
    # mirrors op_tests/test_rope.py:ref_rope_sbhd_fwd (bf16 path).
    import torch

    rotate_half = _rotate_half_neox if rotate_style == NEOX else _rotate_half_gptj
    rotate_dim = freqs.shape[-1] * (2 if reuse_freqs_front_part else 1)
    if nope_first:
        d = x.shape[-1]
        x, x_forward = x[..., d - rotate_dim :], x[..., : d - rotate_dim]
    else:
        x, x_forward = x[..., :rotate_dim], x[..., rotate_dim:]
    if reuse_freqs_front_part:
        if rotate_style == NEOX:
            freqs = freqs.repeat([1] * (freqs.dim() - 1) + [2])
        else:
            freqs = freqs.repeat_interleave(2, dim=-1)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    x_embed = (x * cos) + (rotate_half(x) * sin)
    if nope_first:
        return torch.cat((x_forward, x_embed.to(dtype=x.dtype)), dim=-1).to(x.dtype)
    return torch.cat((x_embed.to(dtype=x.dtype), x_forward), dim=-1).to(x.dtype)


def _cases():
    cases = []
    for B in BS:
        for S in SS:
            for style in ("NEOX", "GPTJ"):
                for nope, nope_first in NOPE_CFG:
                    for reuse in REUSE:
                        name = (
                            f"b{B}_s{S}_{style.lower()}_nope{int(nope)}"
                            f"first{int(nope_first)}_reuse{int(reuse)}"
                        )
                        cases.append(
                            {
                                "name": name,
                                "B": B,
                                "S": S,
                                "style": style,
                                "nope": nope,
                                "nope_first": nope_first,
                                "reuse": reuse,
                            }
                        )
    return cases


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
    NEOX = mod.RotateStyle.NEOX
    GPTJ = mod.RotateStyle.GPTJ
    failures = []
    for c in _cases():
        tag = c["name"]
        try:
            style = NEOX if c["style"] == "NEOX" else GPTJ
            x, freqs = _make_inputs(c["B"], c["S"], c["nope"], c["reuse"])
            y = mod.rope_fwd(
                x,
                freqs,
                rotate_style=style,
                reuse_freqs_front_part=c["reuse"],
                nope_first=c["nope_first"],
            )
            torch.cuda.synchronize()
            ref = _ref_rope_sbhd_fwd(
                x, freqs, style, c["reuse"], c["nope_first"], NEOX
            )
            finite = bool(torch.isfinite(y).all().item())
            close = torch.allclose(y, ref, atol=1e-1, rtol=1e-1)
            ok = finite and close and (y.shape == ref.shape)
            if verbose:
                print(
                    f"  {'PASS' if ok else 'FAIL'}: {tag} "
                    f"out={tuple(y.shape)} finite={finite} close={close}"
                )
            if not ok:
                failures.append(tag)
        except Exception as e:  # noqa: BLE001
            failures.append(tag)
            if verbose:
                print(f"  FAIL: {tag} - {str(e)[:160]}")

    cases = _cases()
    status = "ALL PASS" if not failures else f"FAILED ({len(failures)}/{len(cases)})"
    print(f"Status: {status}")
    print(f"correctness: {'pass' if not failures else 'fail'}")
    return not failures


def run_benchmark(verbose=True):
    import torch

    mod = _load_source()
    NEOX = mod.RotateStyle.NEOX
    # bench a representative subset (neox, no nope, reuse) over (B, S).
    shapes = [
        {"name": f"b{B}_s{S}", "B": B, "S": S} for B in (1, 32) for S in (1, 32)
    ]
    report, latencies = [], []
    for idx, shape in enumerate(shapes):
        x, freqs = _make_inputs(shape["B"], shape["S"], False, True)
        fn = lambda: mod.rope_fwd(  # noqa: E731
            x, freqs, rotate_style=NEOX, reuse_freqs_front_part=True, nope_first=False
        )
        fn()
        torch.cuda.synchronize()
        for _ in range(WARMUP):
            fn()
        torch.cuda.synchronize()
        ms, bench_meta = benchmark_cuda_graph_or_events(
            fn, warmup=0, repetition=ITERS
        )
        latencies.append(ms)
        nbytes = 2.0 * shape["S"] * shape["B"] * H * D * 2  # bf16 read+write
        report.append(
            {
                "test_case_id": f"perf{idx + 1}",
                "execution_time_ms": ms,
                **bench_meta,
                "params": {"B": shape["B"], "S": shape["S"], "H": H, "D": D},
                "gbps": nbytes / (ms * 1e-3) / 1e9,
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
    parser = argparse.ArgumentParser(description="rope_fwd harness")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--correctness", action="store_true")
    parser.add_argument("--full-benchmark", action="store_true")
    args = parser.parse_args()

    print("=" * 62)
    print("triton2flydsl RoPE forward (sbhd)")
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
