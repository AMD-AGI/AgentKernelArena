"""Complete checked graph replay for the five frozen native GEMM geometries."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics

from _dense_runtime import DenseCase, ReplayState, TimedGraph
from _routing_contract import (PROTOCOL, WARMUP, SAMPLES, CHECKED, PREPARED,
    fingerprint, require, source_context, specs, strict_json, validate_payload, validate_request)


def load_harness(ut):
    spec = importlib.util.spec_from_file_location("_dense_protected_harness", ut / "harness_lib.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checked_graph(state, *, warmup=WARMUP, samples=SAMPLES, benchmark=None):
    if benchmark is None:
        from _aka_benchmark import benchmark_cuda_graph_or_events_samples
        benchmark = benchmark_cuda_graph_or_events_samples
    timed = TimedGraph()
    values, metadata = benchmark(state.run, warmup=warmup, repetition=samples,
                                  prepare_fn=state.prepare, timed_run=timed)
    require(metadata.get("benchmark_method") == "cuda_graph"
            and metadata.get("benchmark_warmup") == warmup
            and metadata.get("benchmark_samples") == samples
            and metadata.get("benchmark_effective_repeats") == 1
            and "benchmark_fallback_reason" not in metadata
            and len(values) == samples
            and all(type(value) in (int, float) and math.isfinite(value) and value > 0 for value in values),
            "canonical helper did not return the complete checked graph series")
    result = timed.rerun()
    require(result is state.outputs and result is timed.outputs, "validation does not use the measured graph output")
    state.validate()
    require(state.checked == warmup + samples + 4, "not every warmup/graph replay was checked")
    require(state.case.generated == warmup + samples + 6, "not every graph preparation used fresh inputs")
    return values, metadata


def negative_controls(state):
    # Fixed evaluator probes; the editable policy cannot supply callbacks.
    state.prepare()
    state.outputs = state.torch.empty_like(state.outputs)
    state.outputs.fill_(float("nan"))
    state.capture_pending = False
    try:
        state.validate()
    except (ValueError, RuntimeError):
        pass
    else:
        raise RuntimeError("no-op output was accepted")
    state.needs_validation = False
    state.prepare()
    state.run()
    state.outputs.reshape(-1)[0] += 1024
    try:
        state.validate()
    except (ValueError, RuntimeError):
        pass
    else:
        raise RuntimeError("wrong output was accepted")
    state.needs_validation = False
    return {"no_op": "rejected", "wrong_output": "rejected"}


def measure_case(torch, meta, spec, context, compare, seed):
    case = DenseCase(torch, meta, spec, context, seed)
    state = ReplayState(case, compare)
    values, metadata = checked_graph(state)
    require(state.checked == CHECKED and case.generated == PREPARED, "fixed graph protocol drift")
    validation = state.validation()
    validation["measured_graph_validated"] = True
    return {"sig": spec["sig"] + "|" + spec["regime"], "params": case.params,
            "selected_route": case.route, "mean_ms": math.fsum(values) / len(values),
            "median_ms": statistics.median(values), "min_ms": min(values), "max_ms": max(values),
            "samples_ms": values, **metadata, "validation": validation}


def correctness_case(torch, meta, spec, context, compare, seed):
    case = DenseCase(torch, meta, spec, context, seed)
    state = ReplayState(case, compare)
    held = []
    for _ in range(2 + int(meta["random_draws"])):
        # Keep earlier eager outputs alive; preparation must not poison those
        # caller-owned values while checking the fresh-output API contract.
        state.outputs = None
        state.prepare(); state.run(); state.validate()
        held.append((state.outputs, state.outputs.detach().cpu().clone()))
    require(len({value.untyped_storage().data_ptr() for value, _ in held}) == len(held),
            "native eager calls share output storage")
    require(all(torch.equal(value.detach().cpu().view(torch.uint8), snapshot.view(torch.uint8))
                for value, snapshot in held), "later native call mutated an earlier output")
    eager_checks = state.checked
    controls = negative_controls(state)
    graph_case = DenseCase(torch, meta, spec, context, seed + 1000)
    graph_state = ReplayState(graph_case, compare)
    checked_graph(graph_state, warmup=3, samples=2)
    validation = state.validation()
    validation.update(checked_invocations=eager_checks + graph_state.checked,
        graph_checked_invocations=graph_state.checked, measured_graph_validated=True,
        negative_controls=controls, max_rel_err=max(state.worst_error, graph_state.worst_error),
        fresh_input_sets=case.generated + graph_case.generated,
        prior_outputs_unchanged=True, independent_output_storage=True)
    return {"sig": spec["sig"] + "|" + spec["regime"], "params": case.params,
            "selected_route": case.route, "validation": validation}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ut", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iters", type=int, default=SAMPLES)
    parser.add_argument("--correctness-only", action="store_true")
    args = parser.parse_args()
    require((args.warmup, args.iters) == (WARMUP, SAMPLES), "exactly ten warmups and one hundred samples required")
    task = args.ut.resolve().parent
    mode = "correctness" if args.correctness_only else "performance"
    request = strict_json(args.request.read_text())
    validate_request(request, task, mode)
    meta = strict_json((args.ut / "meta.json").read_text())
    context = source_context(task, meta)
    args.out.unlink(missing_ok=True)
    # Source policy is parsed and bound before any device/runtime import.
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "one assigned CUDA/HIP device is required")
    require("gfx950" in torch.cuda.get_device_properties(0).gcnArchName, "native routing requires gfx950")
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    harness = load_harness(args.ut)
    rows = []
    for index, spec in enumerate(specs(meta)):
        function = correctness_case if args.correctness_only else measure_case
        row = function(torch, meta, spec, context, harness.correct, 31000 + index)
        rows.append(row)
        diagnostics = args.out.parent / "case_diagnostics"
        diagnostics.mkdir(parents=True, exist_ok=True)
        (diagnostics / (str(index) + ".json")).write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
        print(f"CASE {row['sig']} PASS route={row['selected_route']} checked={row['validation']['checked_invocations']}", flush=True)
        torch.cuda.empty_cache()
    validate_request(request, task, mode)
    payload = {"status": "ok", "protocol": PROTOCOL, "request": request,
               "request_sha256": fingerprint(request), "cases": rows}
    if not args.correctness_only:
        payload.update(timer="cuda_graph", warmup=WARMUP, iters=SAMPLES)
    validate_payload(payload, meta, request)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.out)


if __name__ == "__main__":
    main()
