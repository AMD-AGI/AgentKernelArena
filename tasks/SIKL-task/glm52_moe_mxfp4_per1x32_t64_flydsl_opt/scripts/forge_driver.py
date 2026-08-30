#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""forge-loop measurement driver for the GLM-5.2 MXFP4 MoE FlyDSL kernels.

WHAT YOU ARE OPTIMIZING
    The GLM-5.2 routed-expert MoE layer as aiter ships it, in place. Per-rank
    (TP=8): model_dim 6144, inter_dim 256, 257 experts (256 routed + 1 fused
    shared), topk 9, MXFP4 group_size 32 (QuantType.per_1x32), SiLU, bf16 in and
    out. This is the afp4_wfp4 path -- BOTH operands are MXFP4: the weights
    arrive pre-quantized and preshuffled, the activation is quantized on the fly
    once into stage 1 and again on the stage-1 output into stage 2.

    The measured entry is aiter.fused_moe.fused_moe, resolved from the aiter
    copy seeded into this workspace. Your edits to that copy are what this
    driver measures; edits anywhere else are invisible.

WHERE THE WORK IS (num_tokens=64, ~112 us of device time in 7 kernels)
    61.8 us  mfma_moe1_silu_mul_afp4_wfp4_bf16_t32x128x256_pm1_async_v32
             aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py:313
    29.1 us  mfma_moe2_afp4_wfp4_bf16_cshuffle_..._persist_cu256_acc0
             aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py:7432
     6.8 us  fused_mx_quant_moe_sort_kernel<bf16, fp4_t, 256, 32>   (HIP, not editable)
     4.4 us  opus_moe_sorting_entry<P0_v2>                          (HIP, not editable)
     4.1 us  fused_mx_quant_moe_sort_kernel<bf16, fp4_t, 64, 8>     (HIP, not editable)
     3.9 us  moe_reduction_kernel_plain_bf16_topk9_md6144
             aiter/ops/flydsl/kernels/moe_gemm_2stage.py
     3.7 us  opus_moe_sorting_entry<P23>                            (HIP, not editable)

    Dispatch runs through aiter/fused_moe.py:441 -> _flydsl_stage1_wrapper:1169 /
    _flydsl_stage2_wrapper:1243 -> aiter/ops/flydsl/moe_kernels.py:1260 / :1645.
    The kernel pair is chosen by the tuned table row for this shape in
    aiter/configs/model_configs/glm5_fp4_tuned_fmoe.csv, which is editable too:
    picking a better-suited variant is a legitimate optimization.

    Only FlyDSL sources and the dispatch around them are in the edit scope. The
    HIP quant/sort kernels are out of scope for this task, and so is the
    activation quantizer the correctness reference uses.

MODES
    (no flag)        correctness against scripts/task_reference.py, prints
                     `allclose:` and `SNR: <db> dB`
    --bench-mode     times the operator, prints `case_ms:` and `mean_ms:`
    --profile-run    warms the operator, prints no timing

Timing uses Arena's canonical CUDA-graph helper under the task's own warmup and
repetition counts, the same helper and the same counts that produce the task's
score, so what this loop optimizes is what the task reports. `--warmup`/`--iters`
are accepted for contract compatibility but do not change the protocol.
"""

from __future__ import annotations

import argparse
import math
import sys

from pathlib import Path

_DRIVER_DIR = Path(__file__).resolve().parent


def _task_modules_dir() -> Path:
    """Locate the task's helper modules, whichever layout the driver was copied into.

    Arena's forge launcher copies this driver to the workspace ROOT while the
    task keeps its modules under scripts/; the rewrite launcher copies the driver
    and its modules side by side into a scratch workspace. Both have to resolve,
    and a driver that cannot import its modules exits non-zero in every mode,
    which KernelForge reports as a non-conforming task rather than a path bug.
    """
    for candidate in (_DRIVER_DIR, _DRIVER_DIR / "scripts", _DRIVER_DIR.parent / "scripts"):
        if (candidate / "task_inputs.py").is_file():
            return candidate
    raise RuntimeError(
        f"task_inputs.py not found next to {_DRIVER_DIR}, in its scripts/ or in "
        f"{_DRIVER_DIR.parent / 'scripts'}"
    )


sys.path.insert(0, str(_task_modules_dir()))

import task_inputs

# Must precede the first `import aiter` anywhere in this process.
task_inputs.use_workspace_aiter()

import torch

from _aka_benchmark import benchmark_cuda_graph_or_events

import task_baseline
import task_reference

CASE_ID = task_inputs.CASE_ID


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--bench-mode", action="store_true")
    parser.add_argument("--profile-run", action="store_true")
    # Accepted for driver-contract compatibility and deliberately unused: the
    # sampling protocol belongs to the task (see run_bench), so that every timing
    # this driver prints is comparable with the one the task is scored on.
    parser.add_argument("--warmup", type=int, default=task_inputs.BENCH_WARMUP)
    parser.add_argument("--iters", type=int, default=task_inputs.BENCH_REPETITION)
    # Unknown flags are ignored by convention: forge-loop's tools pass arguments
    # this driver does not define.
    args, _unknown = parser.parse_known_args(argv)
    return args


def _prepare():
    if not torch.cuda.is_available():
        raise RuntimeError("this driver requires a ROCm device")
    task_inputs.assert_aiter_is_workspace_copy()
    inputs = task_inputs.build_inputs()
    return task_inputs.call_kwargs(inputs)


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


def run_correctness(kwargs: dict) -> int:
    got = task_baseline.run(**kwargs)
    torch.cuda.synchronize()
    expected = task_reference.run(**kwargs)
    torch.cuda.synchronize()

    if got.shape != expected.shape:
        print(f"shape mismatch: {tuple(got.shape)} vs {tuple(expected.shape)}")
        print("allclose: False")
        return 1

    relative_error = task_inputs.relative_error(got, expected)
    passed = bool(
        torch.isfinite(got.float()).all().item()
        and relative_error <= task_inputs.MAX_RELERR
    )
    print(f"# case {CASE_ID}:")
    print(f"mean relative error: {relative_error:.6f} (gate {task_inputs.MAX_RELERR})")
    print(f"SNR: {_snr_db(expected, got):.2f} dB")
    print(f"allclose: {passed}")
    return 0 if passed else 1


def run_bench(kwargs: dict) -> int:
    """Time the candidate the way the task is scored.

    The sampling protocol is the task's, not the caller's: a candidate is only
    worth keeping if it holds up under the protocol that decides the task's
    score, and honouring a caller's smaller --warmup/--iters would report a
    number that cannot be compared against it. The extra samples cost ~80ms of
    device time against several seconds of process startup per invocation, so
    pinning them is close to free.
    """
    execution_time_ms, metadata = benchmark_cuda_graph_or_events(
        lambda: task_baseline.run(**kwargs),
        warmup=task_inputs.BENCH_WARMUP,
        repetition=task_inputs.BENCH_REPETITION,
        target_ms=task_inputs.BENCH_TARGET_MS,
    )
    # `mean_ms`, not `median_ms`: Arena's helper averages its per-replay samples.
    # The driver contract accepts either key and asks for the one that names the
    # statistic actually computed.
    print(f"case_ms: {CASE_ID} {execution_time_ms:.6f}")
    print(f"mean_ms: {execution_time_ms:.6f}")
    print(f"benchmark_method: {metadata.get('benchmark_method')}")
    return 0


def run_profile(kwargs: dict) -> int:
    for _ in range(3):
        task_baseline.run(**kwargs)
    torch.cuda.synchronize()
    print(f"profile run complete for case {CASE_ID}")
    return 0


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    kwargs = _prepare()
    if args.bench_mode:
        return run_bench(kwargs)
    if args.profile_run:
        return run_profile(kwargs)
    return run_correctness(kwargs)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
