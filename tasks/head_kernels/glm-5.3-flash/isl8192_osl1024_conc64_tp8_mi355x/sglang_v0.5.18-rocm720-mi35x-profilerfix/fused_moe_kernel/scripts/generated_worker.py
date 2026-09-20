#!/usr/bin/env python3
"""Protected GLM worker: receives input seeds, emits outputs, never receives goldens."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys

PREFIX = "GLM_GENERATED_RESULT="


def support(name):
    # Installed and attested before the candidate overlay by the common _trusted_worker.
    module = sys.modules.get(name)
    if module is None:
        raise RuntimeError(f"trusted generated support was not preloaded: {name}")
    return module


def check():
    support("runtime_integrity").ACTIVE_GUARD.check()


def invoke_checked(call, values, contract, torch, previous, writable=()):
    check()
    saved = contract.snapshot_inputs((), values, torch)
    output = call(values)
    check()
    contract.require_inputs_unchanged(saved, torch, writable=writable)
    leaves = contract.tensor_leaves(output, torch)
    if saved and any(value.device != saved[0][0].device for value in leaves):
        raise RuntimeError("candidate moved output off the execution device")
    for prior, snapshot in previous:
        if not contract.compare_output(contract.encode_output(prior, torch), snapshot, 0, torch):
            raise RuntimeError("candidate mutated a previous independent output")
        if any(a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
               for a in contract.tensor_leaves(prior, torch) for b in leaves):
            raise RuntimeError("candidate reused storage across independent outputs")
    encoded = contract.encode_output(output, torch)
    aliases = contract.alias_contract(output, values, torch)
    previous.append((output, encoded))
    check()
    return {"output": encoded, "aliases": aliases}


def input_signature(values, contract, torch):
    groups = {}
    result = []
    for tensor in contract.tensor_leaves(values, torch):
        storage = tensor.untyped_storage()
        group = groups.setdefault((str(tensor.device), storage.data_ptr()), len(groups))
        result.append({"shape": list(tensor.shape), "stride": list(tensor.stride()),
                       "dtype": str(tensor.dtype), "storage_offset": tensor.storage_offset(),
                       "storage_group": group, "storage_nbytes": storage.nbytes(),
                       "tensor_attrs": dict(tensor.__dict__)})
    return result


def timed_row(row, call, cases, meta, reference, contract, torch):
    bench = support("_bench")
    values = row["args"]
    signature = input_signature(values, contract, torch)
    writable = tuple(values[name] for name in row.get("mutable_inputs", ()))
    state = bench.InputState(values, torch, bench.replay_probe(values, torch))
    transform = bench.output_transform(cases, meta, row, torch)
    checks = []
    if reference:
        for probe in (False, True, False):
            state.probe_enabled = probe
            check()
            state.restore()
            saved = contract.snapshot_inputs((), values, torch)
            output = bench.collect_output(lambda: call(values), state.restore, transform,
                                          int(row.get("validation_replays", 1)), torch)
            check()
            contract.require_inputs_unchanged(saved, torch, writable=writable)
            checks.append({"output": contract.encode_output(output, torch),
                           "aliases": contract.alias_contract(output, values, torch)})
        timing = None
    else:
        replay = bench.TimedReplay()
        check()
        samples, timing = support("_aka_benchmark").benchmark_cuda_graph_or_events_samples(
            lambda: call(values), warmup=10, repetition=100,
            prepare_fn=state.restore, timed_run=replay)
        check()
        if timing.get("benchmark_method") != "cuda_graph":
            raise RuntimeError("exact timed graph validation disallows timing fallback")
        if len(samples) != 100 or any(not math.isfinite(x) or x <= 0 for x in samples):
            raise RuntimeError("benchmark did not produce 100 positive device samples")
        for probe in (False, True, False):
            state.probe_enabled = probe
            state.restore()
            saved = contract.snapshot_inputs((), values, torch)
            # This is the exact callable captured by the canonical timing helper.
            output = bench.collect_output(replay.replay, lambda: None, transform,
                                          int(row.get("validation_replays", 1)), torch)
            check()
            contract.require_inputs_unchanged(saved, torch, writable=writable)
            checks.append({"output": contract.encode_output(output, torch),
                           "aliases": contract.alias_contract(output, values, torch)})
        timing = {"test_case_id": bench.case_identity(row),
                  "execution_time_ms": sum(samples) / len(samples),
                  "median_ms": sorted(samples)[len(samples)//2], "min_ms": min(samples),
                  "params": {"regime": row.get("regime"), "m": row.get("m")}, **timing,
                  "benchmark_method_consistent": True,
                  "benchmark_state_restore": "all_input_storages_before_each_replay",
                  "benchmark_output_validation": "exact_timed_graph_replay",
                  "benchmark_replay_probe": "input_change_and_restore",
                  "benchmark_validation_replays": int(row.get("validation_replays", 1))}
    check()
    return {"id": bench.case_identity(row), "inputs": signature, "checks": checks, "timing": timing}


def run_profile(ut, profile, seed, reference=False, device="cuda"):
    import torch
    contract = support("generated_contract")
    cases = support("_glm_generated_cases")
    h = support("harness_lib")
    compact = contract.prepare_cases(cases, ut, seed, torch, device)
    meta = json.loads((ut / "meta.json").read_text())
    if compact["task"] == "fused_moe_kernel":
        # Resolve only after the protected overlay, avoiding a stock-runtime
        # preflight import that would bypass a lazy frozen module binding.
        cases._bootstrap()
        fn = cases._resolve(meta["target_callable"])
        binding = support("_glm_device_binding")
        identity = binding.device_identity(ut)
        overlay = ut / ("baseline_overlay" if reference else "_cand_overlay")
        mapping = json.loads((overlay / "_overlay_manifest.json").read_text())
        contract_data = json.loads((ut / "device_source_contract.json").read_text())
        device_entry = next(e for e in mapping["modules"] if e["module"] == contract_data["device_module"])
        owner = sys.modules[fn.__module__]
        dispatch = next(e for e in mapping["modules"] if e["module"] == contract_data["dispatcher_module"])
        if (identity["file"] != str((overlay/device_entry["file"]).resolve())
                or Path(owner.__file__).resolve() != (overlay/dispatch["file"]).resolve()):
            raise RuntimeError("GLM generated worker did not bind the independent source modules")
        cases._C["fn"] = fn
    call = contract.reference_copy if reference and compact["task"] == "elementwise_copy_cluster" else cases.call
    rows, previous = [], []
    if profile == "recorded":
        for row in contract.recorded_cases(cases, compact):
            result = invoke_checked(call, row["args"], contract, torch, previous)
            contract.require_output_contract(result["output"], row["output_contract"])
            rows.append({"id": row["sig"], "inputs": input_signature(row["args"], contract, torch), **result})
    elif profile == "random":
        for shape in cases.random_shapes(h, meta):
            values = shape["make_inputs"](torch.Generator(device=device).manual_seed(seed))
            rows.append({"id": shape["sig"], "inputs": input_signature(values, contract, torch),
                         **invoke_checked(call, values, contract, torch, previous)})
    elif profile == "semantic_call":
        for row in contract.semantic_moe_call_cases(cases._blob(), cases._build_args, meta):
            values = row["args"]
            writable = tuple(values[name] for name in row["mutable_inputs"])
            rows.append({"id": row["sig"], "inputs": input_signature(values, contract, torch),
                         **invoke_checked(call, values, contract, torch, previous, writable=writable)})
    elif profile == "timed":
        for row in cases.timing_cases(h, meta):
            rows.append(timed_row(row, call, cases, meta, reference, contract, torch))
    else:
        raise ValueError(profile)
    check()
    return {"schema_version": 1, "profile": profile, "seed": seed,
            "reference": reference, "rows": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ut", type=Path, required=True)
    parser.add_argument("--profile", choices=("recorded", "random", "semantic_call", "timed"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    args = parser.parse_args(argv)
    result = run_profile(args.ut.resolve(), args.profile, args.seed, args.reference, args.device)
    check()
    print(PREFIX + json.dumps(result, separators=(",", ":")), flush=True)
    return 0
