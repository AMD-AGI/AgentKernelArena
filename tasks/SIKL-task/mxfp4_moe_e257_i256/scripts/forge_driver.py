#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Dual-path measurement driver for the MXFP4 fused-MoE rewrite task.

THE OPERATOR
    A routed-expert MoE layer, one layer, MXFP4 group_size 32
    (QuantType.per_1x32), SiLU, bf16 in and out. Both operands are MXFP4
    (afp4_wfp4): the weights arrive pre-quantized and preshuffled, the
    activation is quantized on the fly, once into stage 1 and again on the
    stage-1 output into stage 2. The axes, the list of scored cases, the seed
    and the gate policy live in the task's workload.json; scripts/task_inputs.py
    is the single place that reads them, and both this driver and the Arena
    harness build their inputs through it.

THE BASELINE IMPLEMENTATION TO REPLACE (read these, they are the real thing)
    entry            /sgl-workspace/aiter/aiter/fused_moe.py  fused_moe
    stage dispatch   /sgl-workspace/aiter/aiter/fused_moe.py  _flydsl_stage1_wrapper
                     /sgl-workspace/aiter/aiter/fused_moe.py  _flydsl_stage2_wrapper
    stage entries    /sgl-workspace/aiter/aiter/ops/flydsl/moe_kernels.py
                       flydsl_moe_stage1 / flydsl_moe_stage2
    FlyDSL kernels   /sgl-workspace/aiter/aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py
    FlyDSL reduce    /sgl-workspace/aiter/aiter/ops/flydsl/kernels/moe_gemm_2stage.py
    HIP quant/sort   /sgl-workspace/aiter/csrc/kernels/quant_kernels.cu
                     /sgl-workspace/aiter/csrc/include/moe_sorting_opus.h

    fused_moe is a tuned DISPATCH, not one kernel: it selects a stage-1 and a
    stage-2 kernel per M bucket out of aiter/configs/model_configs. Which pair a
    given case faces is not recorded anywhere in the task -- set
    AITER_LOG_TUNED_CONFIG to have aiter report the resolved kernel names.

CORRECTNESS GATE
    Derived, not fixed. This driver measures the production implementation's own
    distance to the reference at every case and admits a candidate within
    `gate_multiplier` (workload.json) times the worst of those. So the bar is
    "no worse than what ships", evaluated on the machine you are running on.
    That distance is not uniform across the cases: the tuned stage-1 variants
    that fuse the output quantization into their epilogue sit an order of
    magnitude further from the reference than the others.

THE INTERFACE THE PORT MUST EXPOSE
    The FlyDSL candidate module must define the builder symbol named by
    KERNELFORGE_REWRITE_BUILDER_SYMBOL:

        build_<slug>_module(num_tokens, model_dim, inter_dim, num_experts, topk)
            -> launch

        launch(hidden_states, w1, w2, topk_weight, topk_ids,
               w1_scale, w2_scale, activation, doweight_stage1)
            -> out

    The builder is called with keyword arguments, ONCE PER CASE, and the
    returned launch is what gets timed, so build all shape-dependent work --
    FlyDSL compilation, tile and split-k selection, any scratch allocation --
    inside the builder.

    Tensor layouts are exactly what the operator receives (see task_inputs.py),
    with E experts, D model_dim and I inter_dim:
        hidden_states   [num_tokens, D]     bfloat16
        w1              [E, 2*I, D/2]       float4_e2m1fn_x2  (preshuffled)
        w1_scale        [E, 2*I, D/32]      float8_e8m0fnu    (preshuffled)
        w2              [E, D, I/2]         float4_e2m1fn_x2  (preshuffled)
        w2_scale        [E, D, I/32]        float8_e8m0fnu    (preshuffled)
        topk_weight     [num_tokens, topk]  float32
        topk_ids        [num_tokens, topk]  int32
        activation      int   (0 = SiLU)
        doweight_stage1 bool  (False: routing weights applied in the stage-2 reduction)
        out             [num_tokens, D]     bfloat16  (returned, not written into an argument)

    w1 holds [gate | up] along its rows, which is what shuffle_weight((16, 16))
    implies together with doweight_stage1=False.

WHAT THE CORRECTNESS SUITE CHECKS BEFORE SCORING
    The layer has to come out of kernels the candidate itself writes in FlyDSL.
    Importing aiter -- including its FlyDSL kernel modules under
    aiter/ops/flydsl/kernels/ -- launches the implementation this task exists to
    replace, so the port would measure the baseline against itself; that is
    checked mechanically and costs an attempt rather than producing a score. The
    same goes for handing the computation to torch or another GPU library, which
    is otherwise free to use for tensor plumbing.

    Beyond that the implementation is open: how you stage, fuse, quantize and
    dispatch the layer in FlyDSL is your call.

MODES
    (no flag)          correctness: candidate vs task_reference over every case,
                       prints one `SNR: <db> dB` (the worst case) and one
                       `allclose:` verdict
    --ref-bench-mode   times the baseline (task_baseline = aiter.fused_moe)
    --bench-mode       times the FlyDSL candidate
    --profile-run      builds and warms the candidate, prints no timing

Both bench modes print one `case_ms:` line per case plus a `mean_ms:` aggregate,
and both time through scripts/task_measure.py -- the same module the Arena
harness scores with, so the number this driver reports and the number the task
is scored on come from one implementation rather than two that agree today.
`--warmup` and `--iters` are accepted for contract compatibility but do not
change the protocol: a candidate is only worth keeping if it holds up under the
protocol that decides the score, and honouring a smaller count would report a
number that cannot be compared against it.

Timing the operator eagerly instead would let per-call host dispatch dominate
the device work at the small-token cases, so a candidate that merely pre-builds
for a fixed shape would report a large speedup while running identical kernels
-- and the win would not exist in a graph-captured server.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

_DRIVER_DIR = Path(__file__).resolve().parent


def _task_modules_dir() -> Path:
    """Locate the task's helper modules, whichever layout the driver was copied into.

    The task keeps its modules under ``scripts/`` next to the harness; the
    rewrite launcher copies the driver and those modules side by side into a
    scratch workspace. Both have to resolve, and a driver that cannot import its
    modules exits non-zero in every mode, which KernelForge reports as a
    non-conforming task rather than a path bug.
    """
    for candidate in (_DRIVER_DIR, _DRIVER_DIR / "scripts", _DRIVER_DIR.parent / "scripts"):
        if (candidate / "task_inputs.py").is_file():
            return candidate
    raise RuntimeError(
        f"task_inputs.py not found next to {_DRIVER_DIR}, in its scripts/ or in "
        f"{_DRIVER_DIR.parent / 'scripts'}"
    )


sys.path.insert(0, str(_task_modules_dir()))

import torch

import task_inputs
import task_measure


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--ref-bench-mode", action="store_true")
    parser.add_argument("--bench-mode", action="store_true")
    parser.add_argument("--profile-run", action="store_true")
    # Accepted for driver-contract compatibility and deliberately unused: the
    # sampling protocol belongs to the task, so that every timing this driver
    # prints is comparable with the one the task is scored on.
    parser.add_argument("--warmup", type=int, default=task_inputs.BENCH_WARMUP)
    parser.add_argument("--iters", type=int, default=task_inputs.BENCH_REPETITION)
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


def _candidate_calls(inputs: dict):
    """Build the candidate for every case, refusing an unimplemented skeleton."""
    launches = task_measure.build_launches(_load_candidate_builder(), inputs)
    if launches is None:
        raise RuntimeError(
            "the FlyDSL candidate is still an unimplemented skeleton; its launch "
            "raises NotImplementedError"
        )
    return task_measure.candidate_calls(inputs, launches)


def _report_timings(samples: list[dict]) -> None:
    """Print the per-case timings plus the aggregate the contract reads.

    Exactly one aggregate line is printed, and it is spelled `mean_ms` rather
    than the contract's canonical `median_ms` because the shared timing helper
    averages its per-replay samples; naming it after a statistic it does not
    compute would be worse than the deprecation warning the spelling earns.
    """
    for sample in samples:
        print(f"case_ms: {sample['case_id']} {sample['execution_time_ms']:.6f}")
    mean_ms = sum(s["execution_time_ms"] for s in samples) / len(samples)
    methods = sorted({str(s["metadata"].get("benchmark_method")) for s in samples})
    print(f"mean_ms: {mean_ms:.6f}")
    print(f"benchmark_method: {','.join(methods)}")


def run_correctness(inputs: dict) -> int:
    """Compare the candidate against the reference on every scored case.

    Only one `SNR:` line and one `allclose:` line are printed: the contract reads
    the first match of each, so the aggregate has to be unambiguous. The
    per-case detail is emitted as `# case <id>:` comments, which is also how the
    contract learns which cases this path covered.
    """
    calls = _candidate_calls(inputs)
    expected, baseline_errors, gate = task_measure.reference_and_gate(inputs)
    print(f"# {task_inputs.gate_explanation(baseline_errors)}")

    worst_snr = float("inf")
    passed = True
    for record in task_measure.compare_cases(calls, expected):
        print(f"# case {record['case_id']}:")
        if record["shape_mismatch"] is not None:
            got_shape, expected_shape = record["shape_mismatch"]
            print(f"#   shape mismatch: candidate {got_shape} vs reference {expected_shape}")
        else:
            print(f"#   mean relative error {record['error']:.8f}")
            print(f"#   snr {record['snr']:.2f} dB")
        worst_snr = min(worst_snr, record["snr"])
        passed = passed and task_measure.passes(record, gate)

    print(f"SNR: {worst_snr:.2f} dB")
    print(f"allclose: {passed}")
    return 0 if passed else 1


def run_reference_bench(inputs: dict) -> int:
    _report_timings(task_measure.time_cases(task_measure.baseline_calls(inputs)))
    return 0


def run_candidate_bench(inputs: dict) -> int:
    _report_timings(task_measure.time_cases(_candidate_calls(inputs)))
    return 0


def run_profile(inputs: dict) -> int:
    calls = _candidate_calls(inputs)
    for _case, call in calls:
        for _ in range(3):
            call()
    torch.cuda.synchronize()
    print(f"profile run complete for {len(calls)} cases")
    return 0


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("this driver requires a ROCm device")
    inputs = task_inputs.build_inputs()

    if args.ref_bench_mode:
        return run_reference_bench(inputs)
    if args.bench_mode:
        return run_candidate_bench(inputs)
    if args.profile_run:
        return run_profile(inputs)
    return run_correctness(inputs)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
