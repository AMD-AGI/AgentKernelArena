"""Independently reconstruct MiniMax paired input receipts from protected data.

Only CPU Torch and pinned harness modules are used. Candidate source modules
are never imported, and no device allocation or GPU operation is performed.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

FIELDS = ["input_seed", "variant_id", "actual_tensor_controls_sha256", "addressing", "private_cpu_storage_bytes"]
DEPENDENCIES = ["evaluation_contract", "served_contract", "minimax_work", "minimax_fixtures", "minimax_data", "paired_state_controls"]


def require(value, message):
    if not value:
        raise ValueError("MiniMax paired inputs: " + message)


def signature_from_geometry(geometry, seed, byte_count, expected_variant, work_controls, fingerprint):
    controls = {"inputs."+k.removeprefix("work."): v for k, v in work_controls(geometry).items()}
    variant = fingerprint(controls)
    require(variant == expected_variant, "prepared controls differ from selected distribution")
    addresses = {}
    for name in ("seq_lens", "req_to_token", "slot_ids", "topk_idx", "cu_seqlens", "cu_seqblocks_q", "prefix_lens"):
        if name in geometry and geometry[name] is not None:
            value = geometry[name]
            require(value.device.type == "cpu", "signature must use the owned CPU snapshot")
            value = value.contiguous()
            addresses[name] = {"dtype": str(value.dtype), "shape": list(value.shape),
                "sha256": hashlib.sha256(value.numpy().tobytes()).hexdigest()}
    return {"input_seed": seed, "variant_id": variant,
            "actual_tensor_controls_sha256": variant, "addressing": addresses,
            "private_cpu_storage_bytes": byte_count}


@contextmanager
def protected_modules(root, contract):
    expected = contract["files_sha256"]
    require(set(expected) == {"ut/"+name+".py" for name in DEPENDENCIES}, "validator dependency set differs")
    previous = {name: sys.modules.get(name) for name in DEPENDENCIES}
    loaded = {}
    try:
        for name in DEPENDENCIES:
            relative = "ut/"+name+".py"
            path = root/relative
            require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(root), "unsafe protected dependency")
            data = path.read_bytes()
            require(hashlib.sha256(data).hexdigest() == expected[relative], "protected dependency changed: "+relative)
            module = ModuleType(name); module.__file__ = str(path)
            sys.modules[name] = module
            exec(compile(data, str(path), "exec"), module.__dict__)
            loaded[name] = module
        yield loaded
    finally:
        for name, module in previous.items():
            if module is None: sys.modules.pop(name, None)
            else: sys.modules[name] = module


def expected_case_signatures(root, case, definition, distribution, modules, seed, verified):
    data, fixtures = modules["minimax_data"], modules["minimax_fixtures"]
    specs = {k: v for k, v in case["tensors"].items() if v["role"] == "input"}
    entries = fixtures.load_bundle(root, case, definition)
    states = [fixtures.geometry(entry, specs, verified) for entry in entries]
    inputs = data.Inputs.__new__(data.Inputs)
    inputs.case, inputs.definition = case, definition
    inputs.scalars = {k: v for k, v in case["scalars"].items() if not k.startswith(("result", "work."))}
    inputs.tensors = {k: SimpleNamespace(shape=tuple(v["shape"])) for k, v in specs.items()}
    sizes = {}
    for key, alias in case["input_aliases"].items():
        sizes[alias] = max(sizes.get(alias, 0), case["original_storage_nbytes"][key])
    byte_count = sum(sizes.values())
    for index in range(110):
        input_seed = seed+index
        index = input_seed % len(case["states"])
        state = states[index]
        variant = modules["paired_state_controls"].fingerprint(case["states"][index]["tensor_controls"])
        fresh = inputs._fresh_geometry(state, input_seed)
        yield signature_from_geometry(fresh, input_seed, byte_count, variant,
                                      data.work_controls, modules["paired_state_controls"].fingerprint)


def reconstruct_inputs(root, manifest, request, provenance):
    """CPU test/inspection API; no generated rows assert GPU execution."""
    import torch
    root = Path(root).resolve()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(min(previous_threads, 8))
    try:
        with protected_modules(root, provenance["input_receipt_contract"]) as modules:
            definition = json.loads((root/"task_definition.json").read_text())
            verified = set()
            return {case["case_id"]: list(expected_case_signatures(root, case, definition,
                None, modules, request["challenge_seed"], verified))
                for case in manifest["cases"]}
    finally:
        torch.set_num_threads(previous_threads)


def validate_paired_input_receipts(root, report, manifest, request, provenance, *, return_details=False):
    import torch
    root = Path(root).resolve()
    contract = provenance["input_receipt_contract"]
    require(contract["schema"] == "task-local-paired-input-validator-v1"
            and contract["validator"] == "ut/paired_input_validation.py"
            and contract["function"] == "validate_paired_input_receipts", "wrong receipt validator contract")
    require(request["phase"] == "performance", "receipt validation requires performance")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(min(previous_threads, 8))
    pairs = report["paired_reference_comparison"]["cases"]
    by_id = {row["case_id"]: row for row in pairs}
    outer = {row["case"]["case_id"]: row for row in report["cases"]}
    require(len(by_id) == len(pairs) == len(outer) == len(manifest["cases"]), "case coverage differs")
    checked = 0
    try:
        with protected_modules(root, contract) as modules:
            scope, data = modules["paired_state_controls"], modules["minimax_data"]
            fixtures = modules["minimax_fixtures"]
            definition = json.loads((root/"task_definition.json").read_text())
            verified = set()
            for case in manifest["cases"]:
                name = case["case_id"]
                require(provenance["input_signature_fields"][name] == FIELDS, "signature fields differ")
                distribution = None
                sampled = outer[name]["workload_state_sampling"]
                ids = sampled["warmup_variant_ids"] + sampled["measured_variant_ids"]
                schedule = by_id[name]["input_schedule"]
                rows = schedule["warmup_inputs"] + schedule["measured_inputs"]
                if "realized_input_signatures" in outer[name]:
                    require(outer[name]["realized_input_signatures"] == rows, "independent realized receipt contradicts paired schedule")
                if "paired_reference" in outer[name]:
                    require(outer[name]["paired_reference"] == by_id[name], "duplicate paired reference receipt differs")
                require(len(rows) == len(ids) == 110, "all 110 inputs are required")
                scope.validate_state_sampling(case, request, manifest["measurement"], outer[name], rows)
                # Reject work/draw corruption before reading fixture payloads.
                for index, (row, variant) in enumerate(zip(rows, ids)):
                    require(set(row) == set(FIELDS) and row["input_seed"] == request["challenge_seed"]+index
                            and row["variant_id"] == row["actual_tensor_controls_sha256"] == variant,
                            "realized draw contradicts protected histogram/request")
                expected_rows = expected_case_signatures(root, case, definition, distribution,
                    modules, request["challenge_seed"], verified)
                for row, expected in zip(rows, expected_rows):
                    require(row == expected, "realized addressing/control hashes differ from protected reconstruction")
                    checked += 1

    finally:
        torch.set_num_threads(previous_threads)
    result = {"status": "ok", "schema": "minimax-prefill-paired-input-validation-v1", "cases": len(manifest["cases"]),
            "reconstructed_inputs": checked, "validated_raw_timings": 100*len(manifest["cases"]),
            "addressing_recomputed_from_protected_fixtures": True, "candidate_code_imported": False,
            "GPU_actions": False}

    return result
