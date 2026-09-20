#!/usr/bin/env python3
"""Protected generated-input worker; receives seeds, never expected outputs."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys

PREFIX = "MINIMAX_GENERATED_RESULT="


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


def invoke_checked(fn, args, kwargs, contract, torch, previous):
    saved = contract.snapshot_inputs(args, kwargs, torch)
    output = fn(*args, **kwargs)
    contract.require_inputs_unchanged(saved, torch)
    leaves = contract.tensor_leaves(output, torch)
    if saved and any(value.device != saved[0][0].device for value in leaves):
        raise RuntimeError("candidate moved its output off the execution device")
    input_storages = {value.untyped_storage().data_ptr() for value, _, _ in saved}
    if any(value.untyped_storage().data_ptr() in input_storages for value in leaves):
        raise RuntimeError("MiniMax result unexpectedly aliases a read-only input")
    for prior, snapshot in previous:
        if any(a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
               for a in contract.tensor_leaves(prior, torch) for b in leaves):
            raise RuntimeError("candidate reused output storage across independent calls")
        if not contract.compare_output(contract.encode_output(prior, torch), snapshot, 0, torch):
            raise RuntimeError("candidate mutated a previous independent output")
    encoded = contract.encode_output(output, torch)
    previous.append((output, encoded))
    return encoded


def graph_sequence(fn, values, contract, torch):
    if not torch.cuda.is_available():
        raise RuntimeError("required MiniMax graph replay has no CUDA/HIP device")
    if len(values) < 2:
        raise RuntimeError("required MiniMax replay needs every boundary variant")
    static = contract.clone_tree(values[0][1], torch)
    holder = {}

    def fill(kwargs):
        if set(kwargs) != set(static):
            raise RuntimeError("replay changed the callable argument set")
        for name, source in kwargs.items():
            dest = static[name]
            if torch.is_tensor(source):
                if (source.shape != dest.shape or source.stride() != dest.stride()
                        or source.dtype != dest.dtype):
                    raise RuntimeError("replay variant changed captured shape/stride/dtype")
                dest.copy_(source)
            elif dest != source:
                raise RuntimeError("replay variant changed a host scalar")

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            holder["output"] = fn(**static)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        holder["output"] = fn(**static)
    rows = []
    for signature, kwargs in values + values[:1]:
        fill(kwargs)
        saved = contract.snapshot_inputs((), static, torch)
        graph.replay()
        torch.cuda.synchronize()
        contract.require_inputs_unchanged(saved, torch)
        output = holder["output"]
        if saved and any(value.device != saved[0][0].device
                         for value in contract.tensor_leaves(output, torch)):
            raise RuntimeError("graph output moved off the execution device")
        inputs = {value.untyped_storage().data_ptr() for value, _, _ in saved}
        if any(value.untyped_storage().data_ptr() in inputs
               for value in contract.tensor_leaves(output, torch)):
            raise RuntimeError("graph result aliases a read-only input")
        rows.append({"id": signature, "output": contract.encode_output(output, torch)})
    return rows


def run_profile(ut, profile, seed, reference=False, device="cuda"):
    import torch
    contract = load("generated_contract", ut / "generated_contract.py")
    compact = contract.load_contract(ut)
    meta = json.loads((ut / "meta.json").read_text())
    # Do not put ut/ on sys.path: its unittest.py shadows the standard library.
    sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != ut]
    h = load("harness_lib", ut / "harness_lib.py")
    cases = load("_minimax_generated_cases", ut / "cases.py")
    fn = cases._resolve()
    rows, previous = [], []
    if profile == "recorded":
        for index, record in enumerate(compact["records"]):
            args, kwargs = contract.build_record(record, seed + index, torch, device)
            output = invoke_checked(fn, args, kwargs, contract, torch, previous)
            contract.require_output_contract(output, record["output_contract"])
            rows.append({"id": record["sig"], "output": output})
    elif profile == "random":
        for shape in cases.random_shapes(h, meta):
            generator = torch.Generator(device=device).manual_seed(seed)
            kwargs = shape["make_inputs"](generator)
            rows.append({"id": shape["sig"], "output": invoke_checked(
                fn, (), kwargs, contract, torch, previous)})
    elif profile == "replay":
        values = [(shape["sig"], shape["make_inputs"](
            torch.Generator(device=device).manual_seed(seed))) for shape in cases.replay_shapes(h, meta)]
        if reference:
            for signature, kwargs in values + values[:1]:
                rows.append({"id": signature, "output": invoke_checked(
                    fn, (), kwargs, contract, torch, previous)})
        else:
            rows = graph_sequence(fn, values, contract, torch)
    else:
        raise ValueError(profile)
    return {"schema_version": 1, "profile": profile, "seed": seed, "reference": reference, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ut", type=Path, required=True)
    parser.add_argument("--profile", choices=["recorded", "random", "replay"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--reference", action="store_true")
    args = parser.parse_args()
    result = run_profile(args.ut.resolve(), args.profile, args.seed, args.reference)
    print(PREFIX + json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    main()
