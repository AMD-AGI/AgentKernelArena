#!/usr/bin/env python3
"""Execute generated Qwen cases without receiving any expected numeric outputs."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

PREFIX = "QWEN_GENERATED_RESULT="


def load(name, path):
    existing = sys.modules.get(name)
    if existing is not None:
        if Path(getattr(existing, "__file__", "")).resolve() != Path(path).resolve():
            raise RuntimeError(f"protected module alias names a different file: {name}")
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def payload(case):
    values = case["args"]
    return tuple(values.get("positional", ())), dict(values.get("kwargs", {}))


def snapshot(values, torch, contract):
    result = []
    for value in contract.tensor_leaves(values, torch):
        copy = value.view(torch.uint8).clone()
        signature = (contract.tensor_signature(value), value.untyped_storage().data_ptr(), dict(value.__dict__))
        result.append((value, signature, copy))
    return result


def unchanged(saved, torch, contract):
    for value, signature, copy in saved:
        current = (contract.tensor_signature(value), value.untyped_storage().data_ptr(), dict(value.__dict__))
        if current != signature or not torch.equal(value.view(torch.uint8), copy):
            raise RuntimeError("candidate changed a read-only input or its tensor contract")


def immutable_inputs(kind, args, kwargs):
    if kind == "paged_attention_decode":
        return args[2:]
    if kind == "fused_recurrent_gated_delta_rule_decode":
        return {key: value for key, value in kwargs.items() if key not in {"initial_state", "out"}}
    return kwargs


def state_result(out, kwargs, before, torch, contract, active_batch=None):
    if type(out) is not tuple or len(out) != 2:
        raise RuntimeError("gated delta must return the complete (out, initial_state) tuple")
    output, state = out
    if output.data_ptr() != kwargs["out"].data_ptr() or state.data_ptr() != kwargs["initial_state"].data_ptr():
        raise RuntimeError("gated delta changed the supplied output/state aliases")
    indices = kwargs["ssm_state_indices"]
    indices = indices[indices >= 0].long()
    if len(indices.unique()) != len(indices):
        raise RuntimeError("state-slot indices must remain unique")
    untouched = torch.ones(state.shape[0], dtype=torch.bool, device=state.device)
    untouched[indices] = False
    if not torch.equal(state[untouched], before[untouched]):
        raise RuntimeError("candidate changed an unselected state-pool row")
    if active_batch is not None:
        output = output[:active_batch]
    selected = state.index_select(0, indices)
    return {"sequence": "tuple", "items": [contract.encode_output(output, torch), {
        "state": True, "signature": contract.tensor_signature(state),
        "indices": indices.detach().cpu().tolist(), "unselected_rows_unchanged": True,
        "full_state_rms": float(state.float().square().mean().sqrt().item()),
        "selected_rows": contract.encode_output(selected, torch),
    }]}


def encode_checked(kind, out, args, kwargs, before, torch, contract, previous, active_batch=None):
    if kind == "fused_recurrent_gated_delta_rule_decode":
        return state_result(out, kwargs, before, torch, contract, active_batch)
    if type(out) is not torch.Tensor:
        raise RuntimeError("Qwen callable must return its complete tensor output")
    template = args[2] if kind == "paged_attention_decode" else kwargs["hidden_states"]
    if out.shape != template.shape or out.dtype != template.dtype or out.device != template.device:
        raise RuntimeError("candidate changed output shape, dtype or device")
    if kind == "paged_attention_decode":
        if out.data_ptr() != args[0].data_ptr():
            raise RuntimeError("paged attention must return the supplied output buffer")
    else:
        input_ptrs = {value.untyped_storage().data_ptr() for value in contract.tensor_leaves(kwargs, torch)}
        if out.untyped_storage().data_ptr() in input_ptrs:
            raise RuntimeError("MoE returned a read-only input as its output")
        for prior, copy in previous:
            if out.untyped_storage().data_ptr() == prior.untyped_storage().data_ptr():
                raise RuntimeError("MoE reused output storage across independent calls")
            if not torch.equal(prior, copy):
                raise RuntimeError("MoE changed a previous independent output")
        previous[:] = [(out, out.clone())]
    return contract.encode_output(out, torch)


def invoke(kind, fn, args, kwargs, torch, contract, previous, check_boundary):
    saved = snapshot(immutable_inputs(kind, args, kwargs), torch, contract)
    before = kwargs["initial_state"].clone() if kind == "fused_recurrent_gated_delta_rule_decode" else None
    check_boundary()
    result = fn(*args, **kwargs)
    check_boundary()
    unchanged(saved, torch, contract)
    return encode_checked(kind, result, args, kwargs, before, torch, contract, previous)


def case_rows(cases, h, meta, device, seed):
    if hasattr(cases, "load_live_cases"):
        return list(cases.load_live_cases(device=device, seed=seed).values())
    if hasattr(cases, "load_live_case"):
        return [cases.load_live_case(device=device, seed=seed)]
    return cases.eager_cases(h, meta, device=device, seed=seed)


def output_contracts(meta, compact):
    kind = meta["generated_inputs"]["kernel"]
    result = {}
    for record in compact["records"]:
        if kind == "fused_moe_2stage_mxfp4":
            case_id = f"{record['regime']}_m{record['kwargs']['hidden_states']['shape'][0]}_live"
        elif kind == "fused_recurrent_gated_delta_rule_decode":
            case_id = f"decode_b{record['kwargs']['mixed_qkv']['shape'][0]}_live"
        else:
            case_id = meta["generated_inputs"]["case_ids"][0]
        if case_id in result:
            raise RuntimeError("duplicate captured output contract")
        result[case_id] = record["output_contract"]
    if set(result) != set(meta["generated_inputs"]["case_ids"]):
        raise RuntimeError("captured output contracts do not cover every generated case")
    return result


def run_graph(kind, fn, case, torch, contract, reference, seed, check_boundary):
    """Compare a changed-input/repeated graph replay with independent eager output."""
    args, kwargs = payload(case)
    args, kwargs = contract.clone_tree((args, kwargs), torch)
    probe = args[2] if kind == "paged_attention_decode" else kwargs["hidden_states"]
    original = probe.clone()
    variants = [original, original * -0.75 + 0.125, original]
    if reference:
        rows, previous = [], []
        for index, value in enumerate(variants):
            probe.copy_(value)
            rows.append({"id": f"{case['sig']}|replay{index}",
                         "output": invoke(kind, fn, args, kwargs, torch, contract, previous, check_boundary)})
        return rows
    if not torch.cuda.is_available():
        raise RuntimeError("required graph replay has no CUDA/HIP device")
    holder = {}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            check_boundary()
            holder["output"] = fn(*args, **kwargs)
            check_boundary()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        holder["output"] = fn(*args, **kwargs)
    check_boundary()
    rows = []
    for index, value in enumerate(variants):
        probe.copy_(value)
        if kind == "paged_attention_decode":
            args[0].zero_()
            args[1].zero_()
        saved = snapshot(immutable_inputs(kind, args, kwargs), torch, contract)
        check_boundary()
        graph.replay()
        torch.cuda.synchronize()
        check_boundary()
        unchanged(saved, torch, contract)
        rows.append({"id": f"{case['sig']}|replay{index}", "output": encode_checked(
            kind, holder["output"], args, kwargs, None, torch, contract, [])})
    return rows


def run_state_graph(fn, cases, torch, contract, reference, check_boundary):
    """Retain max-batch capture, masked min-batch replay, and restore-first order."""
    cases = sorted(cases, key=lambda row: row["m"], reverse=True)
    sequence = cases + cases[:1]
    kind = "fused_recurrent_gated_delta_rule_decode"
    if reference:
        return [{"id": f"{row['sig']}|replay{i}", "output": invoke(
            kind, fn, *contract.clone_tree(payload(row), torch), torch, contract, [], check_boundary)}
                for i, row in enumerate(sequence)]
    if not torch.cuda.is_available():
        raise RuntimeError("required state graph replay has no CUDA/HIP device")
    _, static = contract.clone_tree(payload(cases[0]), torch)
    def fill(row):
        _, values = payload(row)
        batch = values["mixed_qkv"].shape[0]
        for key in ("mixed_qkv", "a", "b", "out"):
            static[key].zero_()
            static[key][:batch].copy_(values[key])
        static["ssm_state_indices"].fill_(-1)
        static["ssm_state_indices"][:batch].copy_(values["ssm_state_indices"])
        for key in ("A_log", "dt_bias", "initial_state"):
            static[key].copy_(values[key])
        for key in ("scale", "use_qk_l2norm_in_kernel"):
            if static[key] != values[key]:
                raise RuntimeError("graph replay changed a host scalar")
        return batch
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fill(cases[0])
            check_boundary()
            fn(**static)
            check_boundary()
    torch.cuda.current_stream().wait_stream(stream)
    fill(cases[0])
    holder = {}
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        holder["output"] = fn(**static)
    check_boundary()
    rows = []
    for index, row in enumerate(sequence):
        batch = fill(row)
        before = static["initial_state"].clone()
        saved = snapshot(immutable_inputs(kind, (), static), torch, contract)
        check_boundary()
        graph.replay()
        torch.cuda.synchronize()
        check_boundary()
        unchanged(saved, torch, contract)
        rows.append({"id": f"{row['sig']}|replay{index}", "output": state_result(
            holder["output"], static, before, torch, contract, active_batch=batch)})
    return rows


def run_profile(ut, profile, seed, reference=False, device="cuda"):
    import torch
    contract = load("generated_contract", ut / "generated_contract.py")
    meta = json.loads((ut / "meta.json").read_text())
    sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != ut]
    h = load("harness_lib", ut / "harness_lib.py")
    cases = load("_qwen_generated_cases", ut / "cases.py")
    # Attest the actually loaded helper/worker functions, including sys.modules
    # identities, before input generation and before/after each candidate call.
    monitor = sys.modules.get("runtime_integrity")
    access = getattr(monitor, "ACTIVE_GUARD", None)
    if access is None:
        raise RuntimeError("generated worker must run through the preloaded trusted boundary")
    check_boundary = access.check
    expected_outputs = output_contracts(meta, contract.load_contract(ut))
    rows = case_rows(cases, h, meta, device, seed)
    fn = cases.current_callable() if hasattr(cases, "current_callable") else cases._current_fn()
    check_boundary()
    kind = meta["generated_inputs"]["kernel"]
    output, previous = [], []
    if profile == "eager":
        for row in rows:
            # Two calls retain the prior independent output check for MoE.
            for index in range(2):
                args, kwargs = contract.clone_tree(payload(row), torch)
                output.append({"id": f"{row['sig']}|call{index}", "output": invoke(
                    kind, fn, args, kwargs, torch, contract, previous, check_boundary)})
    elif profile == "replay":
        if kind == "fused_recurrent_gated_delta_rule_decode":
            output = run_state_graph(fn, rows, torch, contract, reference, check_boundary)
        else:
            for row in rows:
                output.extend(run_graph(kind, fn, row, torch, contract, reference, seed, check_boundary))
    elif profile == "transitions" and kind == "fused_recurrent_gated_delta_rule_decode":
        ordered = sorted(rows, key=lambda row: row["m"])
        state = contract.clone_tree(payload(ordered[0])[1]["initial_state"], torch)
        for index, row in enumerate(ordered + ordered[:1]):
            args, kwargs = contract.clone_tree(payload(row), torch)
            kwargs["initial_state"] = state
            output.append({"id": f"{row['sig']}|transition{index}", "output": invoke(
                kind, fn, args, kwargs, torch, contract, [], check_boundary)})
    else:
        raise ValueError(f"unsupported generated profile: {profile}")
    for row in output:
        contract.require_output_contract(row["output"], expected_outputs[row["id"].split("|", 1)[0]])
    check_boundary()
    return {"schema_version": 1, "profile": profile, "seed": seed, "reference": reference, "rows": output}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ut", type=Path, required=True)
    parser.add_argument("--profile", choices=("eager", "replay", "transitions"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--reference", action="store_true")
    args = parser.parse_args()
    print(PREFIX + json.dumps(run_profile(args.ut.resolve(), args.profile, args.seed, args.reference), separators=(",", ":")))


if __name__ == "__main__":
    main()
