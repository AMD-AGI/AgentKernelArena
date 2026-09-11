# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""The one implementation of how this task measures anything.

Two callers evaluate this task and they must not drift: the Arena harness
produces the score, and the rewrite driver produces the number KernelForge's
loop keeps or reverts a candidate on. If those two ever measured differently,
the pipeline would optimize for something the score does not reward.

They differ only in where the candidate comes from and how results are printed.
Everything else -- how launches are built, how the gate is derived, how a case is
compared to the reference, how a case is timed -- lives here and is imported by
both.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from _aka_benchmark import benchmark_cuda_graph_or_events

import task_baseline
import task_inputs
import task_reference


def build_launches(builder, inputs: dict[str, Any]) -> list | None:
    """Build one launch per scored case, or None while the candidate is a stub.

    Only ``NotImplementedError`` counts as "not ported yet"; every other failure
    propagates, so a broken port fails the task instead of silently scoring the
    baseline a second time.
    """
    if builder is None:
        return None
    launches = []
    for case in inputs["cases"]:
        try:
            launch = builder(m=int(case["m"]), n=task_inputs.N, k=task_inputs.K)
        except NotImplementedError:
            return None
        launches.append(launch)
    return launches


def baseline_calls(inputs: dict[str, Any]) -> list[tuple[dict, Callable]]:
    """Zero-argument calls into the production implementation, one per case."""
    return [
        (case, lambda kwargs=task_inputs.call_kwargs(inputs, case): task_baseline.run(**kwargs))
        for case in inputs["cases"]
    ]


def candidate_calls(inputs: dict[str, Any], launches: list) -> list[tuple[dict, Callable]]:
    """Zero-argument calls into a built candidate, one per case."""
    return [
        (case, lambda launch=launch, a=case["a"], b=inputs["b"]: launch(a, b))
        for case, launch in zip(inputs["cases"], launches)
    ]


def reference_and_baseline(
    inputs: dict[str, Any],
) -> tuple[list, list[dict[str, Any]]]:
    """Evaluate the fp32 reference, and the production implementation beside it.

    The gate is fixed by the task, so the baseline is measured for context
    rather than to set the bar: it is the only way a reader can tell a candidate
    that is genuinely wrong from one that misses the same elements the shipped
    kernel misses. Both readings describe the device, the framework build and
    the inputs that are about to score the candidate.
    """
    expected = []
    measured: list[dict[str, Any]] = []
    for case in inputs["cases"]:
        kwargs = task_inputs.call_kwargs(inputs, case)
        reference = task_reference.run(**kwargs)
        baseline = task_baseline.run(**kwargs)
        torch.cuda.synchronize()
        expected.append(reference)
        measured.append(
            {
                "case_id": str(case["case_id"]),
                "matched_ratio": task_inputs.matched_ratio(baseline, reference),
                "error": task_inputs.relative_error(baseline, reference),
                "snr": snr_db(reference, baseline),
            }
        )
    return expected, measured


def snr_db(reference: torch.Tensor, got: torch.Tensor) -> float:
    reference_f32 = reference.float()
    noise = got.float() - reference_f32
    signal_power = reference_f32.pow(2).sum().item()
    noise_power = noise.pow(2).sum().item()
    if noise_power <= 0.0:
        return float("inf")
    if signal_power <= 0.0:
        return float("-inf")
    return 10.0 * torch.log10(torch.tensor(signal_power / noise_power)).item()


def compare_cases(
    calls: list[tuple[dict, Callable]], expected_outputs: list
) -> list[dict[str, Any]]:
    """Run every case and report how far it lands from the reference."""
    results = []
    for (case, call), expected in zip(calls, expected_outputs):
        got = call()
        torch.cuda.synchronize()
        record: dict[str, Any] = {"case_id": case["case_id"]}
        if got.shape != expected.shape:
            record.update(
                shape_mismatch=(tuple(got.shape), tuple(expected.shape)),
                matched_ratio=0.0,
                error=float("inf"),
                snr=float("-inf"),
            )
        else:
            record.update(
                shape_mismatch=None,
                matched_ratio=task_inputs.matched_ratio(got, expected),
                error=task_inputs.relative_error(got, expected),
                snr=snr_db(expected, got),
                finite=bool(torch.isfinite(got.float()).all().item()),
            )
        results.append(record)
    return results


def passes(record: dict[str, Any]) -> bool:
    """Whether one compared case clears the correctness gate.

    Every element has to be within tolerance. ``error`` and ``snr`` are carried
    on the record for diagnosis only: both are aggregates, and the failure this
    task most needs to catch -- partial sums truncated to bf16 before reduction
    -- leaves both of them looking healthy while a tenth of the output is out of
    tolerance.
    """
    return (
        record["shape_mismatch"] is None
        and record.get("finite", False)
        and record["matched_ratio"] >= 1.0
    )


def time_cases(calls: list[tuple[dict, Callable]]) -> list[dict[str, Any]]:
    """Time every case under the task's own sampling protocol.

    The protocol belongs to the task, not to the caller: a candidate is only
    worth keeping if it holds up under the protocol that decides the score.
    """
    samples = []
    for case, call in calls:
        execution_time_ms, metadata = benchmark_cuda_graph_or_events(
            call,
            warmup=task_inputs.BENCH_WARMUP,
            repetition=task_inputs.BENCH_REPETITION,
            target_ms=task_inputs.BENCH_TARGET_MS,
        )
        samples.append(
            {
                "case_id": case["case_id"],
                "m": case["m"],
                "execution_time_ms": execution_time_ms,
                "metadata": metadata,
            }
        )
    return samples
