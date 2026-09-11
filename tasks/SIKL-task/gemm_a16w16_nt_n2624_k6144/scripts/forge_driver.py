#!/usr/bin/env python3
# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Dual-path measurement driver for the a16w16 GEMM rewrite task.

THE OPERATOR
    A 16-bit GEMM with a transposed weight: ``out = a @ b.T``. The constant axes
    (n, k), the list of scored m cases, the seed and the gate policy all live in
    the task's workload.json; scripts/task_inputs.py is the single place that
    reads them, and both this driver and the Arena harness build their inputs
    through it.

THE BASELINE IMPLEMENTATION TO REPLACE (read it, it is the real thing)
    entry    /sgl-workspace/aiter/aiter/tuned_gemm.py:354  gemm_a16w16

    `gemm_a16w16` is a tuned DISPATCH, not one kernel. It looks the shape up in
    aiter's merged bf16 tuned table (configs/bf16_tuned_gemm.csv merged with
    configs/model_configs/*_bf16_tuned_gemm.csv) and selects a libtype per M
    bucket. One libtype is `flydsl` -- aiter's own FlyDSL kernels under
    aiter/ops/flydsl/kernels/splitk_hgemm.py and small_m_hgemm.py, built through
    aiter/ops/flydsl/kernels/hgemm_dispatch.py -- and the other is `torch`
    (native matmul, i.e. hipBLASLt).

    The selection is NOT monotonic in M, so do not assume a small-M / large-M
    split: on gfx950/256CU at n=k=6144 it picks `flydsl` at m=1..128 and again at
    m=512, and `torch` at m=256, 1024, 2048 and 4096. Which one a given case
    faces is not recorded anywhere in the task: query it with
    `aiter.tuned_gemm.get_GEMM_A16W16_config(M=.., N=.., K=.., bias=False,
    dtype="torch.bfloat16", otype="torch.bfloat16")`, which is the same lookup
    the baseline performs.

CORRECTNESS GATE
    Derived, not fixed. This driver measures the production implementation's own
    distance to the fp32 reference at every case and admits a candidate within
    `gate_multiplier` (workload.json) times the worst of those. So the bar is
    "no worse than what ships", evaluated on the machine you are running on.

THE INTERFACE THE PORT MUST EXPOSE
    The FlyDSL candidate module must define the builder symbol named by
    KERNELFORGE_REWRITE_BUILDER_SYMBOL:

        build_<slug>_module(m, n, k) -> launch

        launch(a, b) -> out

    Called with keyword arguments (``m=``, ``n=``, ``k=``). The builder is
    called ONCE PER CASE and the returned launch is what gets timed, so build
    all shape-dependent work -- FlyDSL compilation, tile and split-k selection,
    any scratch allocation -- inside the builder. `m` is passed because the
    tuned table picks a different kernel per M bucket; aiter's own FlyDSL
    factories compile on (dtype, n, k) plus a tile config and take M as a
    runtime grid dimension, so `m` is a selection input rather than a
    specialization one.

    Tensor layouts are exactly what the operator receives (see task_inputs.py):
        a      [m, k]  bfloat16
        b      [n, k]  bfloat16      (trans_b: the product is a @ b.T)
        out    [m, n]  bfloat16      (returned, not written into an argument)

WHAT THE CORRECTNESS SUITE CHECKS BEFORE SCORING
    The product has to come out of kernels the candidate itself writes in
    FlyDSL. Two ways of not doing that are checked mechanically, so they cost an
    attempt rather than producing a score:

    Importing aiter -- including its FlyDSL kernel modules under
    aiter/ops/flydsl/kernels/ -- launches the implementation this task exists to
    replace.

    Computing the product with torch (`@`, torch.matmul / mm / bmm / einsum /
    addmm / F.linear) resolves to hipBLASLt, which IS the baseline at the larger
    M cases, so it would tie the baseline while implementing no kernel at all.
    Torch is otherwise free to use for tensor plumbing.

    Beyond that the implementation is open: any FlyDSL structure, tiling,
    pipelining or per-shape dispatch you can make fast is fair game.

MODES
    (no flag)          correctness: candidate vs task_reference over every case,
                       prints one `SNR: <db> dB` (the worst case) and one
                       `allclose:` verdict
    --ref-bench-mode   times the baseline (task_baseline = aiter.tuned_gemm)
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

Timing the operator eagerly instead would let per-call host dispatch dominate the
tens of microseconds of device work at the small-M cases, so a candidate that
merely pre-builds for a fixed shape would report a large speedup while running
identical kernels -- and the win would not exist in a graph-captured server.
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
    expected, baseline, gates = task_measure.reference_and_gate(inputs)
    print(f"# {task_inputs.gate_explanation(baseline)}")

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
        passed = passed and task_measure.passes(record, gates)

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
