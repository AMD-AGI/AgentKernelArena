#!/usr/bin/env python3
"""Protected task-owned arena-eval-v1 actions and measured replay checks."""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import secrets
import sys
import traceback

import torch

import task_api as api

ROOT = Path(__file__).resolve().parents[1]
WARMUP = 10
REPETITION = 100
ROTATION_DRAWS = 3
UNSEEN_DRAWS = 4
CHECKED_SAMPLES = 8
UNSEEN_COST_LIMIT = 1.5


def load_manifest():
    rows = json.loads((ROOT / "workloads.json").read_text())["cases"]
    if not rows or len({row["test_case_id"] for row in rows}) != len(rows):
        raise ValueError("Workload manifest must have unique, nonempty cases")
    for row in rows:
        if row["checks"] != ["correctness", "performance"]:
            raise ValueError("Every measured case must receive correctness checks")
        api.validate_params(row["params"])
    return rows


def tensor_copies(value, *, cpu=False):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone() if cpu else value.detach().clone()
    if isinstance(value, tuple):
        return tuple(tensor_copies(x, cpu=cpu) for x in value)
    if isinstance(value, list):
        return [tensor_copies(x, cpu=cpu) for x in value]
    if isinstance(value, dict):
        return {k: tensor_copies(v, cpu=cpu) for k, v in value.items()}
    return value


def immutable_snapshot(values):
    return {name: (value.detach().clone(), tuple(value.stride()), value.data_ptr())
            for name, value in api.readonly(values).items()}


def assert_unchanged(values, snapshot):
    current = api.readonly(values)
    if set(current) != set(snapshot):
        raise AssertionError("Read-only input inventory changed")
    for name, (before, stride, address) in snapshot.items():
        now = current[name]
        if (now.shape != before.shape or now.dtype != before.dtype
                or now.device != before.device or tuple(now.stride()) != stride
                or now.data_ptr() != address or not torch.equal(now, before)):
            raise AssertionError(f"Read-only input {name} was modified")


def changed_output(expected, kind):
    result = tensor_copies(expected)
    first = result[0] if isinstance(result, (tuple, list)) else result
    if kind == "nan":
        first.fill_(float("nan"))
    elif kind == "wrong":
        first.copy_(torch.where(first.float() >= 0, -10000.0, 10000.0).to(first.dtype))
    elif kind == "dtype":
        first = first.to(torch.float64)
        if isinstance(result, tuple):
            result = (first, *result[1:])
        elif isinstance(result, list):
            result[0] = first
        else:
            result = first
    else:
        raise ValueError(kind)
    return result


def reject_wrong_outputs(expected, params):
    api.compare(expected, expected, params)
    for kind in ("wrong", "nan", "dtype"):
        try:
            api.compare(changed_output(expected, kind), expected, params)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError(f"Comparator accepted deliberately {kind} output")
    api.extra_negative_checks(expected, params)


def measure_case(invoke, values, params):
    # Workspace setup materializes the canonical helper beside this entrypoint.
    from _aka_benchmark import TimedRun, benchmark_cuda_graph_or_events

    pristine = immutable_snapshot(values)
    seeds = [secrets.randbits(63) for _ in range(ROTATION_DRAWS + UNSEEN_DRAWS)]
    draws = [api.draw(values, params, seed) for seed in seeds]
    expected = []
    for draw in draws:
        oracle_values = dict(values, **draw)
        expected.append(api.reference(oracle_values, params))
    expected_cpu = [tensor_copies(output, cpu=True) for output in expected]
    selected = set(secrets.SystemRandom().sample(range(REPETITION), CHECKED_SAMPLES))
    state = {"next": 0, "draw": 0, "override": None, "samples": 0}
    observations = []

    def prepare():
        index = state["override"]
        if index is None:
            index = state["next"] % ROTATION_DRAWS
            state["next"] += 1
        state["draw"] = index
        for name, tensor in draws[index].items():
            values[name].copy_(tensor)

    def observe(output):
        index = state["samples"]
        if index in selected:
            api.check_output_contract(output, expected[state["draw"]])
            observations.append((state["draw"], tensor_copies(output, cpu=True)))
        state["samples"] += 1

    replay = TimedRun()
    replay.after_sample = observe
    elapsed, timing = benchmark_cuda_graph_or_events(
        lambda: invoke(values, params), warmup=WARMUP, repetition=REPETITION,
        target_ms=1.0, prepare_fn=prepare, timed_run=replay,
        **getattr(api, "timing_options", lambda: {})(),
    )
    if (timing.get("benchmark_effective_repeats") != 1
            or state["samples"] != REPETITION or len(observations) != CHECKED_SAMPLES):
        raise AssertionError("Missing observation or more than one logical call per sample")
    for index, output in observations:
        api.compare(output, expected_cpu[index], params)
    api.compare(replay.outputs, expected[state["draw"]], params)

    # Held-out draws are never loaded during warmup/capture/reported samples.
    # They follow the same preparation and device-timing path as reported draws.
    unseen_times = []
    for index in range(ROTATION_DRAWS, len(draws)):
        state["override"] = index
        unseen_times.append(replay.rerun_ms())
        api.compare(replay.outputs, expected[index], params)
    if any(not math.isfinite(t) or t <= 0 for t in unseen_times):
        raise AssertionError("Invalid unseen-draw device measurement")
    if min(unseen_times) > elapsed * UNSEEN_COST_LIMIT:
        raise AssertionError("Reported samples are too cheap compared with unseen inputs")

    # A separate poison check tests complete writes, after checking the actual
    # unpoisoned samples. Poisoning never identifies the scored invocations.
    api.poison_output(replay.outputs, params)
    output = replay.rerun()
    api.compare(output, expected[state["draw"]], params)

    # Mutable call inputs must still equal the last draw; fixed inputs must
    # retain their original storage, layout and values.
    current = api.readonly(values)
    for name, (before, stride, address) in pristine.items():
        want = draws[state["draw"]].get(name, before)
        got = current[name]
        if (got.shape != before.shape or got.dtype != before.dtype
                or got.device != before.device or tuple(got.stride()) != stride
                or got.data_ptr() != address or not torch.equal(got, want)):
            raise AssertionError(f"Timed invocation modified input {name}")
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise AssertionError("Device timing must be finite and positive")
    return {"execution_time_ms": elapsed, "benchmark_method": timing["benchmark_method"],
            "metadata": {"device_timing": timing, "draw_seeds": seeds,
                         "checked_sample_indices": sorted(selected),
                         "checked_timed_outputs": len(observations),
                         "exact_replay_validated": True, "poisoned_replay_validated": True,
                         "unseen_draw_ms": unseen_times, "unseen_cost_limit": UNSEEN_COST_LIMIT,
                         "readonly_inputs_unchanged": True}}


def evaluate(role, action):
    result = {"protocol": "arena-eval-v1", "role": role, "action": action,
              "status": "FAIL", "cases": [], "metadata": {}}
    try:
        rows = load_manifest()
        result["cases"] = [dict(copy.deepcopy(row), status="FAIL", reason="Not executed")
                           for row in rows]
        if not torch.cuda.is_available() or not torch.version.hip:
            raise RuntimeError("This task requires a compatible ROCm GPU")
        arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
        if arch != "gfx950":
            raise RuntimeError(f"Unqualified GPU architecture: {arch}")
        invoke = api.load_candidate()
        for row, record in zip(rows, result["cases"]):
            params = row["params"]
            values = api.make_inputs(params, seed=42)
            pristine = None if action == "performance" else immutable_snapshot(values)
            if action == "performance":
                record.update(measure_case(invoke, values, params))
            else:
                expected = api.reference(values, params)
                if action == "validate-task":
                    reject_wrong_outputs(expected, params)
                output = invoke(values, params)
                torch.cuda.synchronize()
                # Compilation really launches every specialization. Checking it
                # too preserves feasibility evidence for the implemented state.
                api.compare(output, expected, params)
                assert_unchanged(values, pristine)
                record["metrics"] = {"reference_checked": True,
                                     "readonly_inputs_unchanged": True}
                if action == "validate-task":
                    record["metrics"]["negative_controls_rejected"] = True
            record.update(status="PASS")
            record.pop("reason", None)
            print(f"{role} {action}: {row['test_case_id']} PASS", flush=True)
            del values, pristine
        result["status"] = "PASS"
        if action == "validate-task":
            result["metadata"]["candidate_state"] = "implemented"
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        result.update(status="FAIL", reason=f"{type(exc).__name__}: {exc}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("validate-task", "baseline", "candidate"))
    parser.add_argument("action", nargs="?", choices=("compile", "correctness", "performance"))
    args = parser.parse_args()
    if (args.role == "validate-task") != (args.action is None):
        parser.error("Use validate-task or <baseline|candidate> <compile|correctness|performance>")
    role, action = ("task", "validate-task") if args.role == "validate-task" else (args.role, args.action)
    result = evaluate(role, action)
    print("ARENA_EVAL_RESULT=" + json.dumps(result, allow_nan=False), flush=True)
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
