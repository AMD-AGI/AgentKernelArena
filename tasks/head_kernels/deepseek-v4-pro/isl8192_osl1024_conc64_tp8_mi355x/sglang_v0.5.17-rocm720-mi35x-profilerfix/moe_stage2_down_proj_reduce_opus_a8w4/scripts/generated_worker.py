"""Protected DeepSeek worker: generated inputs and actual outputs, never goldens."""

from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

PREFIX = "DEEPSEEK_GENERATED_RESULT="


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


def invoke(cases, args, contract, torch, guard, previous):
    check = guard.check
    encode = contract.encode_output
    snapshot = contract.snapshot_inputs
    unchanged = contract.require_unchanged
    check()
    saved = snapshot(args, torch)
    full = cases.call_full(args)
    check()
    unchanged(saved, torch)
    raw_primary = full[0] if isinstance(full, (tuple, list)) else full
    output = cases.comparison_output(full, args)
    leaves = contract.tensor_leaves(output, torch)
    if not leaves:
        raise RuntimeError("candidate returned no tensor output")
    input_pointers = {
        value.untyped_storage().data_ptr()
        for value in contract.tensor_leaves(args, torch)
    }
    if raw_primary.untyped_storage().data_ptr() in input_pointers:
        raise RuntimeError("candidate returned an aliased primary output")
    if any(value.device != leaves[0].device for value in leaves):
        raise RuntimeError("mixed output devices")
    raw_leaves = contract.tensor_leaves(full, torch)
    if any(
        value.untyped_storage().data_ptr() in input_pointers for value in raw_leaves
    ):
        raise RuntimeError("candidate returned an aliased output component")
    for prior_leaves, snapshots in previous:
        if any(
            current.untyped_storage().data_ptr() == prior.untyped_storage().data_ptr()
            for current in raw_leaves
            for prior in prior_leaves
        ):
            raise RuntimeError("candidate reused an earlier output component")
        for prior, snapshot in zip(prior_leaves, snapshots):
            if not torch.equal(prior.contiguous().view(torch.uint8), snapshot):
                raise RuntimeError("candidate mutated an earlier output component")
    previous.append(
        (
            raw_leaves,
            [value.contiguous().view(torch.uint8).clone() for value in raw_leaves],
        )
    )
    del previous[:-2]
    encoded = encode(output, torch)
    check()
    return encoded


def case_inputs(cases, profile, index, seed, torch):
    meta = cases.META
    device = "cuda"
    if profile in ("eager", "sequence"):
        ids = cases.GEN.profiles(meta)[profile]
        name = ids[index]
        if cases.CONTRACT["task_kind"] == "dsa":
            spec = cases._spec_by_name(meta, name)
            return cases._build_args(spec["source_sig"], torch, device, m=spec["m"])
        return next(row["args"] for row in cases._oracle() if row["sig"] == name)
    if profile == "random":
        row = cases.random_shapes(None, meta)[index]
        return row["make_inputs"](torch.Generator(device=device).manual_seed(seed))
    raise ValueError(profile)


def replay(cases, contract, torch, guard, reference):
    """Keep the original two boundary inputs and restore the first afterward."""
    meta = cases.META
    kind = cases.CONTRACT["task_kind"]
    ids = contract.profiles(meta)["replay"]
    if kind == "dsa":
        spec = next(row for row in meta["case_specs"] if row.get("replay"))
        first = cases._build_args(spec["source_sig"], torch, "cuda", m=spec["m"])
        second = cases._build_args(
            spec["source_sig"], torch, "cuda", m=spec["m"], index_mode="shortctx"
        )
        static = dict(first)
        varying = ("q", "indices", "extra_indices_in_kvcache")
        for name in varying:
            static[name] = first[name].clone()

        def fill(value):
            for name in varying:
                static[name].copy_(value[name])

        values = [first, second, first]
        fn = lambda: cases.call_full(static)
        read = lambda value: cases.comparison_output(value, static)
    else:
        gr = meta["graph_replay"]
        by_sig = {row["sig"]: row["args"] for row in cases._oracle()}
        first, second = by_sig[gr["capture_sig"]], by_sig[gr["second_sig"]]

        # The archived replay pads the smaller decode into the larger static M.
        def clone(value):
            if torch.is_tensor(value):
                result = torch.empty_strided(
                    value.shape, value.stride(), dtype=value.dtype, device=value.device
                )
                result.view(torch.uint8).copy_(value.view(torch.uint8))
                result.__dict__.update(value.__dict__)
                return result
            if isinstance(value, dict):
                return {k: clone(v) for k, v in value.items()}
            if isinstance(value, list):
                return [clone(v) for v in value]
            return value

        static = clone(first)
        state = {"m": cases._primary(first).shape[0]}
        out_index = meta.get("inplace_out_arg")
        out_key = meta.get("out_key", "out")
        if kind == "moe1":
            record = next(
                row
                for row in cases.CONTRACT["records"]
                if row["sig"] == gr["capture_sig"]
            )
            output = record["output_contract"]["items"][0]
            out = torch.zeros(
                output["shape"],
                dtype=contract.dtype(torch, output["dtype"]),
                device="cuda",
            )
            static["kwargs"][out_key] = out
        else:
            out = static["args"][out_index]

        def fill(value):
            state["m"] = cases._primary(value).shape[0]
            for dests, sources in (
                (static["args"], value["args"]),
                (static["kwargs"], value["kwargs"]),
            ):
                pairs = (
                    enumerate(sources) if isinstance(sources, list) else sources.items()
                )
                for key, source in pairs:
                    if (
                        key == out_key
                        or (isinstance(sources, list) and key == out_index)
                        or not torch.is_tensor(source)
                    ):
                        continue
                    dest = dests[key]
                    pad = (
                        dest[-1].clone()
                        if dest.dim() == 1 and dest.shape[0] > source.shape[0]
                        else None
                    )
                    d, s = dest.view(torch.uint8), source.view(torch.uint8)
                    d.zero_()
                    count = min(d.shape[0], s.shape[0])
                    d[:count].copy_(s[:count])
                    if count < dest.shape[0] and pad is not None:
                        dest[count:] = pad
            out.zero_()

        values = [first, second, first]
        fn = lambda: cases._resolve()(*static["args"], **static["kwargs"])
        read = (
            (
                lambda value: cases.comparison_output(
                    value, static, active_tokens=state["m"]
                )
            )
            if kind == "moe1"
            else (lambda value: out[: state["m"]])
        )
    rows = []
    holder = {}
    if not reference:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                fill(first)
                holder["out"] = fn()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            holder["out"] = fn()
    for name, value in zip(ids, values):
        fill(value)
        guard.check()
        if kind == "dsa":
            readonly = static
        else:
            pos = list(static["args"])
            kw = dict(static["kwargs"])
            if out_index is not None:
                pos[out_index] = None
            kw.pop(out_key, None)
            readonly = {"args": pos, "kwargs": kw}
        saved = contract.snapshot_inputs(readonly, torch)
        if reference:
            holder["out"] = fn()
        else:
            graph.replay()
        torch.cuda.synchronize()
        guard.check()
        contract.require_unchanged(saved, torch)
        rows.append(
            {"id": name, "output": contract.encode_output(read(holder["out"]), torch)}
        )
        guard.check()
    return rows


def intern_output(output, objects):
    digest = hashlib.sha256()

    def visit(value):
        if isinstance(value, dict):
            for key in sorted(value):
                digest.update(key.encode())
                visit(value[key])
        elif isinstance(value, list):
            for item in value:
                visit(item)
        else:
            digest.update(repr(value).encode())

    visit(output)
    key = digest.hexdigest()
    if key not in objects:
        objects[key] = output
    return key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ut", type=Path, required=True)
    parser.add_argument(
        "--profile", choices=["eager", "random", "sequence", "replay"], required=True
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--reference", action="store_true")
    args = parser.parse_args()
    ut = args.ut.resolve()
    import torch

    contract = load("generated_contract", ut / "generated_contract.py")
    contract.set_seed(args.seed)
    cases = load("_deepseek_generated_cases", ut / "cases.py")
    from runtime_integrity import ACTIVE_GUARD

    guard = ACTIVE_GUARD
    if guard is None:
        raise RuntimeError("generated worker requires the shared trusted bootstrap")
    guard.check()
    profiles = contract.profiles(cases.META)
    if args.profile == "replay":
        rows = replay(cases, contract, torch, guard, args.reference)
    elif args.profile == "sequence":
        previous = []
        rows = []
        objects = {}
        for index, name in enumerate(profiles["sequence"]):
            inputs = case_inputs(cases, "sequence", index, args.seed, torch)
            output = invoke(cases, inputs, contract, torch, guard, previous)
            rows.append({"id": name, "output_ref": intern_output(output, objects)})
    else:
        if not 0 <= args.index < len(profiles[args.profile]):
            raise ValueError("unexpected correctness case index")
        previous = []
        indices = [args.index]
        for index in indices:
            inputs = case_inputs(cases, args.profile, index, args.seed, torch)
            output = invoke(cases, inputs, contract, torch, guard, previous)
        rows = [{"id": profiles[args.profile][args.index], "output": output}]
    guard.check()
    print(
        PREFIX
        + json.dumps(
            {
                "schema_version": 1,
                "profile": args.profile,
                "index": args.index,
                "seed": args.seed,
                "reference": args.reference,
                "rows": rows,
                "objects": objects if args.profile == "sequence" else {},
            },
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
