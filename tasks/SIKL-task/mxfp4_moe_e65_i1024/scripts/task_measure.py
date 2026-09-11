# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""The one implementation of how this task measures anything.

Two callers evaluate this task and they must not drift: the Arena harness
produces the score, and the rewrite driver produces the number KernelForge's
loop keeps or reverts a candidate on. If those two ever measured differently,
the pipeline would optimize for something the score does not reward.

They differ only in where the candidate comes from and how results are printed.
Everything else -- how launches are built, how a case is compared, how a case is
timed -- lives here and is imported by both.

Inputs are built one case at a time, because the schema bundle initializes one
workload point at a time and its activation is drawn before its expert weights:
a case's weights depend on the case's num_tokens, so the cases cannot share one
set. Each mode therefore builds a case, uses it, and lets it fall out of scope
before the next.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from _aka_benchmark import benchmark_cuda_graph_or_events

import task_baseline
import task_inputs
import task_reference


def build_launches(builder) -> list | None:
    """Build one launch per scored case, or None while the candidate is a stub.

    The builder takes the case shape and no data, so every launch is built
    before any input exists. Only ``NotImplementedError`` counts as "not ported
    yet"; every other failure propagates, so a broken port fails the task
    instead of silently scoring the baseline a second time.
    """
    if builder is None:
        return None
    launches = []
    for case in task_inputs.CASES:
        try:
            launch = builder(
                num_tokens=int(case["num_tokens"]),
                model_dim=task_inputs.MODEL_DIM,
                inter_dim=task_inputs.INTER_DIM,
                num_experts=task_inputs.NUM_EXPERTS,
                topk=task_inputs.TOPK,
            )
        except NotImplementedError:
            return None
        launches.append(launch)
    return launches


def case_call(inputs: dict[str, Any], launch) -> Callable:
    """A zero-argument call into the candidate, or into the baseline when stub."""
    if launch is None:
        kwargs = task_inputs.call_kwargs(inputs)
        return lambda: task_baseline.run(**kwargs)
    return lambda: launch(
        inputs["hidden_states"],
        inputs["w1"],
        inputs["w2"],
        inputs["topk_weights"],
        inputs["topk_ids"],
        inputs["w1_scale"],
        inputs["w2_scale"],
        inputs["activation"],
        inputs["doweight_stage1"],
    )


def compare_cases(launches: list | None) -> list[dict[str, Any]]:
    """Judge every case, and judge the production implementation beside it.

    The verdict is the bundle's comparison callback, applied to the candidate
    and -- separately -- to the shipped implementation. The second reading
    changes no outcome; it is the only thing that tells a reader whether a
    failing case is the candidate's fault or a bar production does not clear
    either.
    """
    results = []
    for index, case in enumerate(task_inputs.CASES):
        inputs = task_inputs.build_case_inputs(case)
        kwargs = task_inputs.call_kwargs(inputs)
        expected = task_reference.run(**kwargs)
        baseline_passed, baseline_detail = task_inputs.verdict(
            task_baseline.run(**kwargs), expected
        )
        got = case_call(inputs, None if launches is None else launches[index])()
        torch.cuda.synchronize()
        passed, detail = task_inputs.verdict(got, expected)
        results.append(
            {
                "case_id": str(case["case_id"]),
                "passed": passed,
                "detail": detail,
                "baseline_passed": baseline_passed,
                "baseline_detail": baseline_detail,
            }
        )
    return results


def time_cases(launches: list | None) -> list[dict[str, Any]]:
    """Time every case under the task's own sampling protocol.

    The protocol belongs to the task, not to the caller: a candidate is only
    worth keeping if it holds up under the protocol that decides the score.
    """
    samples = []
    for index, case in enumerate(task_inputs.CASES):
        inputs = task_inputs.build_case_inputs(case)
        call = case_call(inputs, None if launches is None else launches[index])
        execution_time_ms, metadata = benchmark_cuda_graph_or_events(
            call,
            warmup=task_inputs.BENCH_WARMUP,
            repetition=task_inputs.BENCH_REPETITION,
            target_ms=task_inputs.BENCH_TARGET_MS,
        )
        samples.append(
            {
                "case_id": str(case["case_id"]),
                "num_tokens": int(case["num_tokens"]),
                "execution_time_ms": execution_time_ms,
                "metadata": metadata,
            }
        )
    return samples
