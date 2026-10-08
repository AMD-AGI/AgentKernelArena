"""Self-contained SIKL callable loading, tensor contracts and the bundle comparison."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
import types
from pathlib import Path

import torch

SCRIPTS = Path(__file__).resolve().parent


def load_solution(root: Path, entry: str):
    filename, symbol = entry.split("::")
    namespace = "_sikl_" + hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:16]
    if namespace not in sys.modules:
        initializer = root / "__init__.py"
        if initializer.is_file():
            spec = importlib.util.spec_from_file_location(namespace, initializer, submodule_search_locations=[str(root)])
            package = importlib.util.module_from_spec(spec)
            sys.modules[namespace] = package
            spec.loader.exec_module(package)
        else:
            package = types.ModuleType(namespace)
            package.__path__ = [str(root)]
            sys.modules[namespace] = package
    parts = list(Path(filename).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    module = importlib.import_module(".".join([namespace, *parts]))
    entrypoint = getattr(module, symbol)
    if not callable(entrypoint):
        raise TypeError(f"{entry} is not callable")
    return entrypoint


def dtype(name):
    names = {"float4_e2m1": "float4_e2m1fn_x2"}
    value = getattr(torch, names.get(name, name), None)
    if not isinstance(value, torch.dtype):
        raise ValueError(f"Unsupported dtype in this runtime: {name}")
    return value


def dimensions(definition, row):
    return {**{k: a["value"] for k, a in definition["axes"].items() if a["type"] == "const"},
            **row["workload"]["axes"]}


def shape_of(spec, axes):
    return tuple(axes[d] if isinstance(d, str) else d for d in spec["shape"])


def validate_inputs(values, definition, row, device):
    if not isinstance(values, dict) or set(values) != set(definition["inputs"]):
        raise ValueError("Input adapter must return exactly the definition's named arguments")
    axes = dimensions(definition, row)
    for name, spec in definition["inputs"].items():
        descriptor = row["workload"]["inputs"][name]
        value = values[name]
        if descriptor["type"] == "scalar":
            literal = descriptor["value"]
            if type(value) is not type(literal) or value != literal:
                raise ValueError(f"{name}: scalar changed from workload")
        else:
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name}: expected tensor")
            if tuple(value.shape) != shape_of(spec, axes) or value.dtype != dtype(spec["dtype"]):
                raise ValueError(f"{name}: input shape/dtype differs from definition")
            if value.device.type != torch.device(device).type:
                raise ValueError(f"{name}: wrong input device")


class OutputContractError(AssertionError):
    """A returned value violates the declared output names, shapes, dtypes or device."""


def outputs(value, definition, row, device):
    names = list(definition["outputs"])
    if isinstance(value, dict):
        if set(value) != set(names):
            raise OutputContractError("Output names differ from definition")
        result = [value[n] for n in names]
    elif isinstance(value, (tuple, list)):
        result = list(value)
    else:
        result = [value]
    if len(result) != len(names):
        raise OutputContractError("Output count differs from definition")
    axes = dimensions(definition, row)
    for name, tensor in zip(names, result):
        spec = definition["outputs"][name]
        if not isinstance(tensor, torch.Tensor):
            raise OutputContractError(f"{name}: output must be a tensor")
        if tensor.dtype != dtype(spec["dtype"]) or tuple(tensor.shape) != shape_of(spec, axes):
            raise OutputContractError(f"{name}: output shape/dtype differs from definition")
        if tensor.device.type != torch.device(device).type:
            raise OutputContractError(f"{name}: output must be on the requested device")
        if bool(torch.isnan(tensor).any()):
            raise OutputContractError(f"{name}: output must not contain NaN")
    return result


def compare_outputs(got, expected, definition, row, device):
    """Return the bundle comparison's verdict as a per-case result.

    Output-contract violations and numerical mismatches are distinguished, so a
    declared diagnostic baseline policy can apply to the latter only. An invalid
    reference raises instead of being scored as a candidate failure.
    """
    wanted = outputs(expected, definition, row, device)
    try:
        actual = outputs(got, definition, row, device)
        for a, b in zip(actual, wanted):
            if (not torch.equal(torch.isposinf(a), torch.isposinf(b))
                    or not torch.equal(torch.isneginf(a), torch.isneginf(b))):
                raise OutputContractError("Output infinity positions/signs differ from the reference")
    except OutputContractError as error:
        return {"status": "FAIL", "failure_kind": "output_contract", "reason": str(error)}
    compare = load_solution(SCRIPTS / "compare", "main.py::run")
    # Schema callbacks consume a tensor for one output and the declared names
    # for multiple outputs, independent of the solution's tuple ABI.
    callback_actual = actual[0] if len(actual) == 1 else dict(zip(definition["outputs"], actual))
    callback_expected = wanted[0] if len(wanted) == 1 else dict(zip(definition["outputs"], wanted))
    try:
        if compare(callback_actual, callback_expected) is not None:
            raise ValueError("compare must return None on success or raise AssertionError")
    except AssertionError as error:
        return {"status": "FAIL", "failure_kind": "numerical_mismatch", "reason": str(error),
                "metadata": {"comparison": "scripts/compare/main.py:run", "output_contract_passed": True}}
    return {"status": "PASS", "metadata": {"comparison": "scripts/compare/main.py:run",
                                           "output_contract_passed": True}}


def snapshot(values):
    """Byte copies of every tensor input; exact for packed and FP8 operands too."""
    return {name: value.detach().contiguous().view(torch.uint8).clone()
            for name, value in values.items() if isinstance(value, torch.Tensor)}


def assert_unmodified(values, before):
    for name, expected in before.items():
        if not torch.equal(values[name].detach().contiguous().view(torch.uint8), expected):
            raise RuntimeError(f"Operator modified protected input tensor: {name}")
