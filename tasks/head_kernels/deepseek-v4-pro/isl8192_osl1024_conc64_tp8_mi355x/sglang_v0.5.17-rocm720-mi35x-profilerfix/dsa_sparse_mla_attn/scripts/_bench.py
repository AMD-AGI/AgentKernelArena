#!/usr/bin/env python3
"""Benchmark the complete case set through each protected package's own APIs.

Reference and candidate workers run in separate overlay environments. The arena
materializes _aka_benchmark beside this file. Exact timed-graph output is checked;
capture, replay, case construction and comparison failures are always fatal.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def tensors(value, torch, path="args"):
    if torch.is_tensor(value):
        yield path, value
    elif isinstance(value, dict):
        for key in sorted(value, key=str):
            yield from tensors(value[key], torch, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from tensors(item, torch, f"{path}[{index}]")


def tensor_signature(tensor):
    return {"shape": list(tensor.shape), "stride": list(tensor.stride()),
            "dtype": str(tensor.dtype), "storage_offset": tensor.storage_offset()}


class InputState:
    """Restore storages in place, retaining views, aliases and dispatch attributes.

    Raw byte copies also handle packed FP4/E8M0 types without copy kernels.
    Every input storage is restored outside every measured interval on both
    sides. Tensor objects remain the same, including their is_shuffled flags.
    """

    def __init__(self, args, torch, probe=None):
        self.tensors, self.storages = [], []
        self.probe = probe
        self.probe_enabled = False
        seen_tensors, seen_storages = set(), set()
        for _, tensor in tensors(args, torch):
            if id(tensor) in seen_tensors:
                continue
            seen_tensors.add(id(tensor))
            storage = tensor.untyped_storage()
            self.tensors.append((tensor, tensor_signature(tensor), dict(tensor.__dict__),
                                 storage.data_ptr()))
            key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
            if key in seen_storages:
                continue
            seen_storages.add(key)
            view = torch.empty(0, device=tensor.device, dtype=torch.uint8)
            view.set_(storage, 0, (storage.nbytes(),), (1,))
            self.storages.append((view, view.clone()))

    def restore(self):
        for tensor, signature, attributes, pointer in self.tensors:
            if (tensor_signature(tensor) != signature
                    or tensor.untyped_storage().data_ptr() != pointer):
                raise RuntimeError("candidate changed an input tensor's storage, shape, stride or dtype")
            tensor.__dict__.clear()
            tensor.__dict__.update(attributes)
        for view, saved in self.storages:
            view.copy_(saved)
        if self.probe_enabled:
            if self.probe is None:
                raise RuntimeError("no input tensor was declared for replay validation")
            self.probe.copy_((self.probe.float() * -0.75 + 0.125).to(self.probe.dtype))


def cpu_copy(value, torch):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item, torch) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(cpu_copy(item, torch) for item in value)
    if isinstance(value, list):
        return [cpu_copy(item, torch) for item in value]
    return value


def case_identity(row):
    signature = row.get("sig")
    if not isinstance(signature, str) or not signature:
        raise RuntimeError("every benchmark case needs its full captured signature")
    return json.dumps([signature, row.get("regime", "")], separators=(",", ":"))


def validate_cases(rows):
    ids = [case_identity(row) for row in rows]
    if not ids or len(ids) != len(set(ids)):
        raise RuntimeError("benchmark case set is empty or contains duplicate identities")
    return ids


def replay_probe(values, torch):
    """Choose a numerical activation, never an index, scale encoding or output.

    The additional validation values keep the exact captured shape and layout.
    The independent baseline evaluates the same values, with the package's
    original comparator. They are never part of the scored timing samples.
    """
    candidates = []
    if isinstance(values, dict):
        for name in ("q", "hidden_states", "inter_states", "A", "a", "a1", "x", "mixed_qkv", "scale"):
            if name in values:
                candidates.append(values[name])
        for name in ("kwargs", "inputs"):
            if isinstance(values.get(name), dict):
                try:
                    return replay_probe(values[name], torch)
                except ValueError:
                    pass
        if values.get("positional"):
            candidates.append(values["positional"][2])  # paged attention query
        for name in ("pos", "args"):
            if values.get(name):
                candidates.append(values[name][0])
    for value in candidates:
        if torch.is_tensor(value) and value.is_floating_point() and value.numel():
            return value
    raise ValueError("package has no declared numerical activation for replay perturbation")


def independent_pair(module):
    """Require the protected task adapter's distinct reference/candidate pair."""
    baseline = getattr(module, "BASELINE_FN", None)
    candidate = getattr(module, "CANDIDATE_FN", None)
    if not callable(baseline) or not callable(candidate) or baseline is candidate:
        raise RuntimeError("protected adapter did not bind independent baseline and candidate callables")
    return baseline, candidate


def selected_cases(module, h, meta, torch, reference):
    """Adapt immutable package APIs; geometry and value recipes stay in the UT."""
    if hasattr(module, "selected_cases") and hasattr(module, "timing_case"):
        rows = [module.timing_case(case) for case in
                module.selected_cases(meta, meta["ledger_ids"])]
        call = module.baseline_call if reference else module.candidate_call
    elif hasattr(module, "load_live_cases"):
        rows = list(module.load_live_cases().values())
        # Reference workers resolve this through the baseline overlay; they do
        # not import the candidate-only frozen-binding registry.
        call = module.call
    elif hasattr(module, "load_live_case"):
        rows = [module.load_live_case()]
        # The package's timing path reuses caller-owned output/workspace buffers.
        rows = [{**row, "args": {**row["args"], "fresh": False}} for row in rows]
        call = module.call
    elif hasattr(module, "timing_cases") and hasattr(module, "call"):
        rows = list(module.timing_cases(h, meta))
        call = module.call  # selected by the worker's baseline/candidate overlay
    elif hasattr(module, "_online_buckets"):
        # unittest.py gets these functions from ut/bindings.py:resolve_pair.
        # Keep its output-slot allocation wrapper and captured geometry recipe.
        independent_pair(module)
        rng = torch.Generator(device=module.DEV).manual_seed(1234)
        rows = [{"sig": sig, "regime": regime, "m": bs,
                 "args": module._synth(bs, ctx, rng)}
                for sig, bs, ctx, regime in module._online_buckets()]
        call = module.baseline_call if reference else module.current_call
    elif hasattr(module, "BASELINE_FN") and hasattr(module, "CANDIDATE_FN"):
        # Kimi FlyDSL imports separate frozen/editable packages. Keep its
        # persistent output and single-launch timed seam; validate its declared
        # elementwise median using repeated replay of that same timed graph.
        baseline, candidate = independent_pair(module)
        fn = baseline if reference else candidate
        fn = h.compiled_op(fn, module.REGIME)
        rows = []
        for entry in module.timing_cases():
            inputs = module.build_inputs(entry["spec"])
            shape = ((int(inputs["token_num"]), module.TOPK, module.INTER_DIM)
                     if meta["entry_attr"] == "flydsl_moe_stage1" else
                     (int(inputs["token_num"]), module.MODEL_DIM))
            output = torch.zeros(shape, dtype=torch.bfloat16, device=module.DEV)
            rows.append({"sig": entry["spec"]["sig"], "regime": entry["regime"],
                         "m": entry["m"], "args": {"inputs": inputs, "output": output},
                         "validation_replays": int(getattr(module, "MEDIAN_LAUNCHES", 1))})
        call = lambda args: module._invoke(fn, args["inputs"], args["output"], zero=False)
    else:
        raise RuntimeError("package has no supported complete-case benchmark adapter")
    validate_cases(rows)
    return rows, call


def output_transform(module, meta, row, torch):
    # DSA has oracle-derived don't-care masks. Apply the package's exact mask
    # only during comparison, outside the measured op, as its eager UT does.
    if hasattr(module, "_undef_mask") and hasattr(module, "_spec_by_name"):
        spec = module._spec_by_name(meta, row["sig"])
        if spec is None:
            raise RuntimeError(f"no DSA case specification for {row['sig']}")
        mask = module._undef_mask(spec["source_sig"], torch, "cuda", m=spec.get("m"))
        if mask is not None:
            return lambda out: out.masked_fill(mask, 0)
    return lambda out: out


def collect_output(call, prepare, transform, count, torch):
    outputs = []
    for _ in range(count):
        prepare()
        output = transform(call())
        if count == 1:
            return output
        if not torch.is_tensor(output):
            raise RuntimeError("the package's median policy requires a tensor output")
        outputs.append(output.detach().clone())
    return torch.stack(outputs).median(dim=0).values


class TimedReplay:
    def _bind(self, replay, output):
        self._replay = replay

    def replay(self):
        if not hasattr(self, "_replay"):
            raise RuntimeError("benchmark helper did not expose the timed graph")
        return self._replay()


def measure_case(row, call, expected, module, h, meta, torch, warmup, iters, benchmark):
    values = row["args"]
    state = InputState(values, torch, replay_probe(values, torch))
    transform = output_transform(module, meta, row, torch)
    replay = TimedReplay()
    samples, timing = benchmark(lambda: call(values), warmup=warmup, repetition=iters,
                                prepare_fn=state.restore, timed_run=replay)
    if timing.get("benchmark_method") != "cuda_graph":
        raise RuntimeError("exact replay validation requires graph timing; no fallback is allowed")
    if len(samples) != iters or any(not math.isfinite(x) or x <= 0 for x in samples):
        raise RuntimeError("benchmark returned an incomplete or invalid sample set")
    # A/B/A checks the graph responds to changed input values and then restores
    # the original values. A graph replay returning a memoized warmup result
    # cannot pass merely because timing repeatedly used the same input buffers.
    for probe in (False, True, False):
        state.probe_enabled = probe
        output = collect_output(replay.replay, lambda: None, transform,
                                int(row.get("validation_replays", 1)), torch)
        refs = h.to_device_like(expected["probe" if probe else "base"], "cuda")
        integrity = getattr(sys.modules.get("runtime_integrity"), "ACTIVE_GUARD", None)
        compare = integrity.compare if integrity is not None else h.correct
        ok, error = compare(output, refs, float(meta["tol"]))
        if not ok:
            raise RuntimeError(f"timed graph output mismatch for {case_identity(row)} "
                               f"(input_probe={probe}): {error}")
    timing.update({"benchmark_method_consistent": True,
                   "benchmark_state_restore": "all_input_storages_before_each_replay",
                   "benchmark_output_validation": "exact_timed_graph_replay",
                   "benchmark_replay_probe": "input_change_and_restore",
                   "benchmark_validation_replays": int(row.get("validation_replays", 1))})
    return {"test_case_id": case_identity(row), "execution_time_ms": sum(samples) / len(samples),
            "params": {"regime": row.get("regime"), "m": row.get("m")},
            "median_ms": sorted(samples)[len(samples) // 2], "min_ms": min(samples), **timing}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ut", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--phase", choices=["reference", "measure"], required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()
    if args.warmup < 0 or args.iters < 1:
        raise ValueError("warmup must be nonnegative and iters must be positive")
    ut = Path(args.ut).resolve()
    meta = json.loads((ut / "meta.json").read_text())
    if meta.get("dispatch_config"):
        os.environ["AITER_CONFIG_GEMM_BF16"] = str(ut / meta["dispatch_config"])
    h = load_module("harness_lib", ut / "harness_lib.py")
    # unittest.py in the package must not shadow Python's standard library.
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != ut]
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA/HIP device visible")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    module = load_module("_headkernel_cases", ut / ("cases.py" if (ut / "cases.py").is_file()
                                                   else "unittest.py"))
    rows, call = selected_cases(module, h, meta, torch, args.phase == "reference")
    ids = validate_cases(rows)
    reference_path = Path(args.reference)
    if args.phase == "reference":
        outputs, signatures = {}, {}
        for row in rows:
            identity = case_identity(row)
            values = row["args"]
            state = InputState(values, torch, replay_probe(values, torch))
            signatures[identity] = [(path, tensor_signature(t)) for path, t in tensors(values, torch)]
            outputs[identity] = {}
            for probe in (False, True):
                state.probe_enabled = probe
                output = collect_output(lambda: call(values), state.restore,
                                        output_transform(module, meta, row, torch),
                                        int(row.get("validation_replays", 1)), torch)
                outputs[identity]["probe" if probe else "base"] = cpu_copy(output, torch)
            state.probe_enabled = False
            state.restore()
        temporary = reference_path.with_suffix(".tmp")
        torch.save({"case_ids": ids, "inputs": signatures, "outputs": outputs}, temporary)
        temporary.replace(reference_path)
        reference_path.with_suffix(".cases.json").write_text(json.dumps(ids) + "\n")
        return 0

    from _aka_benchmark import benchmark_cuda_graph_or_events_samples
    reference = torch.load(reference_path, map_location="cpu", weights_only=False)
    if reference.get("case_ids") != ids or set(reference.get("outputs", {})) != set(ids):
        raise RuntimeError("candidate case set does not match the complete baseline case set")
    results = []
    for row in rows:
        identity = case_identity(row)
        signature = [(path, tensor_signature(t)) for path, t in tensors(row["args"], torch)]
        if reference["inputs"].get(identity) != signature:
            raise RuntimeError(f"baseline/candidate input contract differs: {identity}")
        results.append(measure_case(row, call, reference["outputs"][identity], module,
                                    h, meta, torch, args.warmup, args.iters,
                                    benchmark_cuda_graph_or_events_samples))
    payload = {"status": "ok", "expected_case_ids": ids,
               "warmup_iterations": args.warmup, "benchmark_iterations": args.iters,
               "test_cases": results}
    out = Path(args.out)
    temporary = out.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
