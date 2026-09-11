#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Dual-path measurement driver for the MXFP4 fused-MoE rewrite task.

THE OPERATOR
    A routed-expert MoE layer, one layer, MXFP4 group_size 32
    (QuantType.per_1x32), SiLU, bf16 in and out. Both operands are MXFP4
    (afp4_wfp4): the weights arrive pre-quantized and preshuffled, the
    activation is quantized on the fly, once into stage 1 and again on the
    stage-1 output into stage 2. The axes, the list of scored cases and the seed
    live in the task's workload.json; scripts/task_inputs.py is the single place
    that reads them, and both this driver and the Arena harness build their
    inputs through it.

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
    scripts/task_compare.py, which is the workload schema's own comparison
    callback, copied from the bundle without edit. It owns the threshold and the
    task does not set, scale or relax it -- the acceptance run that verifies a
    submitted result applies the same file, so a candidate this driver keeps is
    a candidate that run accepts.

    scripts/task_initialize.py is the bundle's input callback on the same terms;
    it also owns the MXFP4 quantization, so no aiter quantizer sits between the
    weights you are given and the weights the acceptance run generates. Inputs
    are built one case at a time because the bundle initializes one workload
    point at a time.

    The production implementation is judged by the same callback beside every
    case and the result is printed, but it does not move the bar.

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
                       prints one `allclose:` verdict and no `SNR:` line (see
                       run_correctness for why)
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


def _candidate_launches():
    """Build the candidate for every case, refusing an unimplemented skeleton."""
    launches = task_measure.build_launches(_load_candidate_builder())
    if launches is None:
        raise RuntimeError(
            "the FlyDSL candidate is still an unimplemented skeleton; its launch "
            "raises NotImplementedError"
        )
    return launches


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


def run_correctness() -> int:
    """Compare the candidate against the reference on every scored case.

    Exactly one `allclose:` line is printed and no `SNR:` line, on purpose.
    KernelForge's correctness stage prefers an SNR reading over the driver's own
    verdict whenever one is present -- it applies `snr_db >= --snr-threshold`
    and never looks at `allclose` -- so printing an SNR here would hand the
    keep/revert decision to a threshold that knows nothing about the bundle's
    comparison. Withholding it makes the bundle's verdict the pipeline's
    verdict, which is the only way PORT and OPTIMIZE keep candidates that the
    acceptance run will also accept. The per-case detail stays in
    `# case <id>:` comments, which is also how the contract learns which cases
    this path covered.
    """
    launches = _candidate_launches()
    print(f"# {task_inputs.GATE_EXPLANATION}")

    passed = True
    for record in task_measure.compare_cases(launches):
        print(f"# case {record['case_id']}:")
        print(f"#   candidate: {'pass' if record['passed'] else 'fail'} -- {record['detail']}")
        print(
            f"#   production: {'pass' if record['baseline_passed'] else 'fail'} -- "
            f"{record['baseline_detail']}"
        )
        passed = passed and record["passed"]

    print(f"allclose: {passed}")
    return 0 if passed else 1


def run_reference_bench() -> int:
    _report_timings(task_measure.time_cases(None))
    return 0


def run_candidate_bench() -> int:
    _report_timings(task_measure.time_cases(_candidate_launches()))
    return 0


def run_profile() -> int:
    launches = _candidate_launches()
    for case, launch in zip(task_inputs.CASES, launches):
        inputs = task_inputs.build_case_inputs(case)
        call = task_measure.case_call(inputs, launch)
        for _ in range(3):
            call()
    torch.cuda.synchronize()
    print(f"profile run complete for {len(launches)} cases")
    return 0


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("this driver requires a ROCm device")
    if args.ref_bench_mode:
        return run_reference_bench()
    if args.bench_mode:
        return run_candidate_bench()
    if args.profile_run:
        return run_profile()
    return run_correctness()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
