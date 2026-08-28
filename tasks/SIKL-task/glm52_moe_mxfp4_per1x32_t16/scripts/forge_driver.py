#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Dual-path measurement driver for the GLM-5.2 MXFP4 MoE rewrite task.

THE OPERATOR
    GLM-5.2 routed-expert MoE, one layer, per-rank (TP=8) config: model_dim
    6144, inter_dim 256, 257 experts (256 routed + 1 fused shared), topk 9,
    MXFP4 group_size 32 (QuantType.per_1x32), SiLU, bf16 in/out. Both operands
    are MXFP4 (afp4_wfp4): the weights arrive pre-quantized and preshuffled,
    the activation is quantized on the fly, once into stage 1 and again on the
    stage-1 output into stage 2.

THE BASELINE IMPLEMENTATION TO REPLACE (read these, they are the real thing)
    entry            /sgl-workspace/aiter/aiter/fused_moe.py:441  fused_moe
    stage dispatch   /sgl-workspace/aiter/aiter/fused_moe.py:1169 _flydsl_stage1_wrapper
                     /sgl-workspace/aiter/aiter/fused_moe.py:1243 _flydsl_stage2_wrapper
    stage entries    /sgl-workspace/aiter/aiter/ops/flydsl/moe_kernels.py:1260 flydsl_moe_stage1
                     /sgl-workspace/aiter/aiter/ops/flydsl/moe_kernels.py:1645 flydsl_moe_stage2
    FlyDSL kernels   /sgl-workspace/aiter/aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py:313
                     /sgl-workspace/aiter/aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py:7432
    FlyDSL reduce    /sgl-workspace/aiter/aiter/ops/flydsl/kernels/moe_gemm_2stage.py
    HIP quant/sort   /sgl-workspace/aiter/csrc/kernels/quant_kernels.cu:1734
                     /sgl-workspace/aiter/csrc/include/moe_sorting_opus.h
    On gfx950 at num_tokens=64 the tuned table picks
    flydsl_moe1_afp4_wfp4_bf16_t32x128x256_w4 and
    flydsl_moe2_afp4_wfp4_bf16_t32x128x256_reduce_persist; the seven device
    kernels sum to ~117 us (two grouped GEMMs ~91 us of it).

THE INTERFACE THE PORT MUST EXPOSE
    The FlyDSL candidate module must define the builder symbol named by
    KERNELFORGE_REWRITE_BUILDER_SYMBOL:

        build_<slug>_module(num_tokens, model_dim, inter_dim, num_experts, topk)
            -> launch

        launch(hidden_states, w1, w2, topk_weight, topk_ids,
               w1_scale, w2_scale, activation, doweight_stage1)
            -> out                      # bf16 [num_tokens, model_dim]

    Tensor layouts are exactly what the operator receives (see task_inputs.py):
        hidden_states  [num_tokens, 6144]  bfloat16
        w1             [257, 512, 3072]    float4_e2m1fn_x2   (preshuffled)
        w1_scale       [257, 512, 192]     float8_e8m0fnu     (preshuffled)
        w2             [257, 6144, 128]    float4_e2m1fn_x2   (preshuffled)
        w2_scale       [257, 6144, 8]      float8_e8m0fnu     (preshuffled)
        topk_weight    [num_tokens, 9]     float32
        topk_ids       [num_tokens, 9]     int32
        activation     int   (0 = SiLU)
        doweight_stage1 bool (False: routing weights applied in the stage-2 reduction)

RULES THE CANDIDATE MUST SATISFY (checked, not just stated)
    The candidate may NOT import the framework under test. Importing aiter --
    including its FlyDSL kernel modules under aiter/ops/flydsl/kernels/ -- means
    launching the implementation this task exists to replace, so the port would
    measure the baseline against itself. Write the kernels in FlyDSL: import
    flydsl (and torch for tensor plumbing) only.

MODES
    (no flag)          correctness: candidate vs task_reference, prints
                       `allclose:` and `SNR: <db> dB`
    --ref-bench-mode   times the baseline (task_baseline = aiter.fused_moe)
    --bench-mode       times the FlyDSL candidate
    --profile-run      builds and warms the candidate, prints no timing

Both bench modes use Arena's canonical benchmark helper (CUDA-graph replay with
an event fallback), the same one that produces the task's score. Timing the
operator eagerly instead would make per-call host dispatch dominate the ~112 us
of device work, so a candidate that merely pre-builds for a fixed shape would
report a large speedup while running identical kernels -- and the win would not
exist in a graph-captured server.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from _aka_benchmark import benchmark_cuda_graph_or_events

import task_baseline
import task_inputs
import task_reference

CASE_ID = task_inputs.CASE_ID


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--ref-bench-mode", action="store_true")
    parser.add_argument("--bench-mode", action="store_true")
    parser.add_argument("--profile-run", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    # Unknown flags are ignored by convention: the nested forge-loop tools pass
    # arguments this driver does not define, and refusing them would read as
    # "this driver does not support the mode".
    args, _unknown = parser.parse_known_args(argv)
    return args


def _load_candidate_builder():
    """Import the FlyDSL candidate by path and return its builder symbol."""
    path = os.environ.get("KERNELFORGE_REWRITE_CANDIDATE_KERNEL", "")
    symbol = os.environ.get("KERNELFORGE_REWRITE_BUILDER_SYMBOL", "")
    if not path or not symbol:
        raise RuntimeError(
            "KERNELFORGE_REWRITE_CANDIDATE_KERNEL and "
            "KERNELFORGE_REWRITE_BUILDER_SYMBOL must be set by the rewrite driver "
            "environment"
        )
    task_inputs.assert_candidate_is_independent(Path(path).read_text())
    spec = importlib.util.spec_from_file_location("forge_flydsl_candidate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import the FlyDSL candidate at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    builder = getattr(module, symbol, None)
    if builder is None:
        raise RuntimeError(f"{path} does not define the builder symbol {symbol}")
    return builder


def _candidate_launch():
    builder = _load_candidate_builder()
    return builder(
        num_tokens=task_inputs.NUM_TOKENS,
        model_dim=task_inputs.MODEL_DIM,
        inter_dim=task_inputs.INTER_DIM,
        num_experts=task_inputs.NUM_EXPERTS,
        topk=task_inputs.TOPK,
    )


def _call_candidate(launch, inputs: dict) -> torch.Tensor:
    return launch(
        inputs["hidden_states"],
        inputs["w1"],
        inputs["w2"],
        inputs["topk_weight"],
        inputs["topk_ids"],
        inputs["w1_scale"],
        inputs["w2_scale"],
        inputs["activation"],
        inputs["doweight_stage1"],
    )


def _timed_ms(fn, warmup: int, iters: int) -> tuple[float, dict]:
    """Time one implementation with Arena's canonical benchmark helper."""
    return benchmark_cuda_graph_or_events(
        fn,
        warmup=max(0, warmup),
        repetition=max(1, iters),
        target_ms=task_inputs.BENCH_TARGET_MS,
    )


def _report_timing(ms: float, metadata: dict) -> None:
    print(f"case_ms: {CASE_ID} {ms:.6f}")
    print(f"median_ms: {ms:.6f}")
    print(f"benchmark_method: {metadata.get('benchmark_method')}")


def _snr_db(reference: torch.Tensor, got: torch.Tensor) -> float:
    reference_f32 = reference.float()
    noise = got.float() - reference_f32
    signal_power = reference_f32.pow(2).sum().item()
    noise_power = noise.pow(2).sum().item()
    if noise_power <= 0.0:
        return float("inf")
    if signal_power <= 0.0:
        return float("-inf")
    return 10.0 * math.log10(signal_power / noise_power)


def run_correctness(inputs: dict) -> int:
    launch = _candidate_launch()
    got = _call_candidate(launch, inputs)
    torch.cuda.synchronize()
    expected = task_reference.run(**task_inputs.call_kwargs(inputs))
    torch.cuda.synchronize()

    if got.shape != expected.shape:
        print(f"shape mismatch: candidate {tuple(got.shape)} vs "
              f"reference {tuple(expected.shape)}")
        print("allclose: False")
        return 1

    relative_error = task_inputs.relative_error(got, expected)
    snr = _snr_db(expected, got)
    passed = bool(
        torch.isfinite(got.float()).all().item()
        and relative_error <= task_inputs.MAX_RELERR
    )
    print(f"# case {CASE_ID}:")
    print(f"mean relative error: {relative_error:.6f} (gate {task_inputs.MAX_RELERR})")
    print(f"SNR: {snr:.2f} dB")
    print(f"allclose: {passed}")
    return 0 if passed else 1


def run_reference_bench(inputs: dict, warmup: int, iters: int) -> int:
    kwargs = task_inputs.call_kwargs(inputs)
    _report_timing(*_timed_ms(lambda: task_baseline.run(**kwargs), warmup, iters))
    return 0


def run_candidate_bench(inputs: dict, warmup: int, iters: int) -> int:
    launch = _candidate_launch()
    _report_timing(*_timed_ms(lambda: _call_candidate(launch, inputs), warmup, iters))
    return 0


def run_profile(inputs: dict) -> int:
    launch = _candidate_launch()
    for _ in range(3):
        _call_candidate(launch, inputs)
    torch.cuda.synchronize()
    print(f"profile run complete for case {CASE_ID}")
    return 0


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("this driver requires a ROCm device")
    inputs = task_inputs.build_inputs()

    if args.ref_bench_mode:
        return run_reference_bench(inputs, args.warmup, args.iters)
    if args.bench_mode:
        return run_candidate_bench(inputs, args.warmup, args.iters)
    if args.profile_run:
        return run_profile(inputs)
    return run_correctness(inputs)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
