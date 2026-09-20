#!/usr/bin/env python3
"""Kimi generated-input worker: independent role, no expected answers supplied."""
from __future__ import annotations
import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys

PREFIX = "KIMI_GENERATED_RESULT="
_PRINT = print


def load(name, path):
    existing = sys.modules.get(name)
    if existing is not None:
        if Path(getattr(existing, "__file__", "")).resolve() != Path(path).resolve():
            raise RuntimeError(f"trusted helper alias names a different file: {name}")
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def snapshot(value, contract, torch):
    result = []
    for tensor in contract.tensor_leaves(value, torch):
        storage = torch.Tensor.untyped_storage(tensor)
        raw = torch.empty(0, device=tensor.device, dtype=torch.uint8).set_(storage, 0, (storage.nbytes(),), (1,))
        signature = (tuple(tensor.shape), tuple(torch.Tensor.stride(tensor)), tensor.dtype,
                     tensor.device, torch.Tensor.storage_offset(tensor), storage.data_ptr(), dict(tensor.__dict__))
        result.append((tensor, signature, raw, raw.clone()))
    return result


def unchanged(saved, torch):
    for tensor, signature, raw, original in saved:
        storage = torch.Tensor.untyped_storage(tensor)
        current = (tuple(tensor.shape), tuple(torch.Tensor.stride(tensor)), tensor.dtype,
                   tensor.device, torch.Tensor.storage_offset(tensor), storage.data_ptr(), dict(tensor.__dict__))
        if current != signature or not torch.equal(raw, original):
            raise RuntimeError("candidate changed a read-only input or its layout/alias contract")


def checked_call(fn, values, contract, torch, check, previous):
    check()
    saved = snapshot(values, contract, torch)
    output = fn(values)
    check()
    unchanged(saved, torch)
    leaves = contract.tensor_leaves(output, torch)
    input_ptrs = {sig[5] for _, sig, _, _ in saved}
    for value in leaves:
        if torch.Tensor.untyped_storage(value).data_ptr() in input_ptrs:
            raise RuntimeError("output unexpectedly aliases a read-only input")
    for prior, encoded in previous:
        previous_ptrs = {torch.Tensor.untyped_storage(value).data_ptr()
                         for value in contract.tensor_leaves(prior, torch)}
        if any(torch.Tensor.untyped_storage(value).data_ptr() in previous_ptrs for value in leaves):
            raise RuntimeError("candidate reused a previous independent output buffer")
        if not contract.compare_output(contract.encode_output(prior, torch), encoded, 0, torch):
            raise RuntimeError("candidate changed a previous independent output")
    encoded = contract.encode_output(output, torch)
    previous.append((output, encoded))
    check()
    return encoded


def calibrate(scope):
    for item in scope["META"].get("splits_calibration", []):
        actual = scope["_num_kv_splits"](int(item["bs"]), int(item["seq_len"]))
        if not bool((actual == int(item["num_kv_splits"])).all()):
            raise RuntimeError("generated attention split schedule differs from captured calibration")


def profile_cases(scope, compact, profile, seed, contract, torch, device):
    if compact["kind"] == "attention":
        if profile in ("recorded", "sequence"):
            rows = [{"sig": row["sig"], "args": contract.attention_record(row, seed + index, torch, device),
                     "regime": row["regime"]} for index, row in enumerate(compact["records"])]
        else:
            # Preserve the established scored input recipe and RNG seed.
            generator = torch.Generator(device=device).manual_seed(1234 if profile == "performance" else seed)
            rows = [{"sig": sig, "regime": regime, "m": batch, "args": scope["_synth"](batch, ctx, generator)}
                    for sig, batch, ctx, regime in scope["_online_buckets"]()]
    else:
        specs = (scope["META"]["case_specs"] if profile != "performance"
                 else [entry["spec"] for entry in scope["timing_cases"]()])
        rows = [{"sig": spec["sig"], "regime": spec.get("regime", ""), "m": int(spec["token_num"]),
                 "args": scope["build_inputs"](spec)} for spec in specs]
    if profile == "sequence":
        order = scope["META"].get("call_sequence_idx", [0, 1, 0])
        rows = [rows[index] for index in order]
    return rows


def replay_moe(scope, fn, reference, contract, torch, check):
    scope["CANDIDATE_FN"] = fn
    bundle = scope["build_replay"](scope["META"]["replay_specs"])
    expected = [item["sig"] for item in sorted(scope["META"]["replay_specs"],
                                              key=lambda item: int(item["token_num"]), reverse=True)]
    if [row["sig"] for row in bundle["cases"]] != expected or len(expected) < 2:
        raise RuntimeError("generated replay omitted a captured boundary case")
    if any("ref" in row for row in bundle["cases"]):
        raise RuntimeError("candidate replay received a cached reference")
    if reference:
        run = bundle["run"]
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("required graph replay has no CUDA/HIP device")
        bundle["fill"](bundle["cases"][0])
        for _ in range(3):
            bundle["run"]()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            bundle["run"]()
        run = graph.replay
    rows = []
    for row in bundle["cases"] + bundle["cases"][:1]:
        check()
        bundle["fill"](row)
        saved = snapshot(bundle["state_inputs"], contract, torch)
        run()
        torch.cuda.synchronize()
        check()
        unchanged(saved, torch)
        rows.append({"id": row["sig"], "output": contract.encode_output(bundle["read_out"](), torch)})
    return rows


def benchmark_rows(rows, scope, fn, reference, contract, torch, check, helper):
    """Return only observed replay outputs; the independent parent compares them."""
    if not reference:
        from _aka_benchmark import benchmark_cuda_graph_or_events_samples
    if scope["META"].get("entry_attr"):
        fn = scope["h"].compiled_op(fn, scope["REGIME"])
    result = []
    for row in rows:
        values = row["args"]
        if scope["META"].get("entry_attr"):
            shape = ((int(values["token_num"]), scope["TOPK"], scope["INTER_DIM"])
                     if scope["META"]["entry_attr"] == "flydsl_moe_stage1" else
                     (int(values["token_num"]), scope["MODEL_DIM"]))
            output = torch.zeros(shape, dtype=torch.bfloat16, device=scope["DEV"])
            state_values = {"inputs": values, "output": output}
            call = lambda: scope["_invoke"](fn, values, output, zero=False)
        else:
            state_values = values
            caller = scope["_make_call"](fn)
            call = lambda: caller(values)
        state = helper.InputState(state_values, torch, helper.replay_probe(state_values, torch))
        if reference:
            run = lambda: (state.restore(), call())[1]
            timing = None
        else:
            timed = helper.TimedReplay()
            samples, timing = benchmark_cuda_graph_or_events_samples(
                call, warmup=10, repetition=100, prepare_fn=state.restore, timed_run=timed)
            check()
            if (timing.get("benchmark_method") != "cuda_graph" or len(samples) != 100
                    or any(not math.isfinite(value) or value <= 0 for value in samples)):
                raise RuntimeError("incomplete canonical graph measurement")
            run = timed.replay
        observed = []
        replays = int(scope["META"].get("median_launches", 1))
        for probe in (False, True, False):
            state.probe_enabled = probe
            values_out = []
            for _ in range(replays):
                state.restore()
                saved = snapshot(values, contract, torch)
                value = run()
                check()
                unchanged(saved, torch)
                if replays == 1:
                    break
                values_out.append(value.clone())
            if replays > 1:
                value = torch.stack(values_out).median(0).values
            observed.append(contract.encode_output(value, torch))
        item = {"id": helper.case_identity(row), "outputs": observed}
        if timing is not None:
            item["timing"] = {"test_case_id": item["id"], "execution_time_ms": sum(samples) / len(samples),
                              "params": {"regime": row.get("regime"), "m": row.get("m")}, **timing,
                              "benchmark_method_consistent": True,
                              "benchmark_output_validation": "exact_timed_graph_replay",
                              "benchmark_state_restore": "all_input_storages_before_each_replay",
                              "benchmark_replay_probe": "input_change_and_restore",
                              "benchmark_validation_replays": replays}
        result.append(item)
    return result


def run_profile(ut, profile, seed, reference=False, device="cuda"):
    import torch
    contract = load("generated_contract", ut / "generated_contract.py")
    helper = load("_kimi_timing_helpers", ut.parent / "scripts/_bench.py")
    compact = contract.load_contract(ut)
    scope = contract.legacy_definitions(ut, torch, device, compact, None if profile == "performance" else seed)
    import runtime_integrity
    guard = runtime_integrity.ACTIVE_GUARD
    if guard is None:
        raise RuntimeError("generated Kimi worker requires the shared trusted preloader")
    for module in (contract, helper, sys.modules[__name__]):
        guard.check_module(module)
    check = guard.check
    check()
    fn = None if profile == "single_launch" and reference else contract.bind(ut, scope["META"], reference)
    check()
    calibrate(scope)
    if profile == "single_launch":
        independent = load("kimi_single_launch", ut / "generated_single_launch.py")
        guard.check_module(independent)
        rows = independent.run(scope, fn, reference, seed, contract, torch, check)
    elif profile == "replay" and compact["kind"] == "moe":
        rows = replay_moe(scope, fn, reference, contract, torch, check)
    elif profile in ("performance", "replay"):
        cases = profile_cases(scope, compact, "performance" if profile == "performance" else "random",
                              seed, contract, torch, device)
        rows = benchmark_rows(cases, scope, fn, reference, contract, torch, check, helper)
    else:
        cases = profile_cases(scope, compact, profile, seed, contract, torch, device)
        caller = scope["_make_call"](fn) if compact["kind"] == "attention" else scope["make_call"](fn)
        previous = []
        rows = [{"id": row["sig"], "output": checked_call(caller, row["args"], contract, torch, check, previous)}
                for row in cases]
    check()
    return {"schema_version": 1, "profile": profile, "seed": seed, "reference": reference, "rows": rows,
            "correctness_policy": "independent_torch_single_launch" if profile == "single_launch" else "elementwise_median_21" if scope["META"].get("median_launches") == 21 else "single_launch",
            "single_launch_correctness_established": False if scope["META"].get("median_launches") == 21 else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ut", type=Path, required=True)
    parser.add_argument("--profile", choices=["recorded", "random", "sequence", "replay", "performance", "single_launch"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--reference", action="store_true")
    args = parser.parse_args()
    print_result, serialize = _PRINT, json.dumps
    result = run_profile(args.ut.resolve(), args.profile, args.seed, args.reference)
    print_result(PREFIX + serialize(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
