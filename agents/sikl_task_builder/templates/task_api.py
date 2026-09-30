"""Self-contained SIKL callable loading and tensor contract checks."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
import types
from pathlib import Path

import torch


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


def outputs(value, definition, row, device):
    names = list(definition["outputs"])
    if isinstance(value, dict):
        if set(value) != set(names):
            raise ValueError("Output names differ from definition")
        result = [value[n] for n in names]
    elif isinstance(value, (tuple, list)):
        result = list(value)
    else:
        result = [value]
    if len(result) != len(names):
        raise ValueError("Output count differs from definition")
    axes = dimensions(definition, row)
    for name, tensor in zip(names, result):
        spec = definition["outputs"][name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name}: output must be a tensor")
        if tensor.dtype != dtype(spec["dtype"]) or tuple(tensor.shape) != shape_of(spec, axes):
            raise ValueError(f"{name}: output shape/dtype differs from definition")
        if tensor.device.type != torch.device(device).type:
            raise ValueError(f"{name}: output must be on the requested device")
        if definition.get("compare"):
            if bool(torch.isnan(tensor).any()):
                raise ValueError(f"{name}: output must not contain NaN")
        elif not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name}: output must be finite")
    return result


def assert_outputs(got, expected, definition, row, policy, device):
    actual = outputs(got, definition, row, device)
    wanted = outputs(expected, definition, row, device)
    if definition.get("compare"):
        for a, b in zip(actual, wanted):
            # Complex outputs need component-wise signs; isposinf/isneginf do
            # not accept complex tensors, while the existing ABI permits them.
            a = torch.view_as_real(a.resolve_conj()) if a.is_complex() else a
            b = torch.view_as_real(b.resolve_conj()) if b.is_complex() else b
            # PyTorch's signed-infinity predicates do not support FP8. Widen
            # only these checks; callbacks still receive the original tensors.
            if a.is_floating_point() and a.element_size() == 1:
                a, b = a.float(), b.float()
            if (not torch.equal(torch.isposinf(a), torch.isposinf(b))
                    or not torch.equal(torch.isneginf(a), torch.isneginf(b))):
                raise AssertionError("Output infinity positions/signs differ from the reference")
        compare = load_solution(Path(__file__).parent / "compare", "main.py::run")
        # Callbacks receive the schema ABI, independent of the solution's
        # tuple/list/dict return container. Preserve declared insertion order.
        callback_actual = actual[0] if len(actual) == 1 else dict(zip(definition["outputs"], actual))
        callback_expected = wanted[0] if len(wanted) == 1 else dict(zip(definition["outputs"], wanted))
        if compare(callback_actual, callback_expected) is not None:
            raise ValueError("compare must return None on success or raise AssertionError")
        return
    for a, b in zip(actual, wanted):
        torch.testing.assert_close(a, b, rtol=policy["rtol"], atol=policy["atol"], equal_nan=False)


def poison_outputs(captured, expected, values, definition, row, device):
    actual = outputs(captured, definition, row, device)
    wanted = outputs(expected, definition, row, device)
    input_storage = {v.untyped_storage().data_ptr() for v in values.values() if isinstance(v, torch.Tensor)}
    for tensor, reference in zip(actual, wanted):
        if tensor.untyped_storage().data_ptr() in input_storage:
            raise ValueError("Output/input aliasing is unsupported by this functional task")
        if tensor.is_floating_point() or tensor.is_complex():
            tensor.fill_(float("nan"))
        else:
            tensor.copy_(torch.bitwise_not(reference))


def clone_inputs(values):
    # The supported contract is functional, with independent input tensors.
    # Preserve AITER's layout tag when making correctness-only copies.
    copied = {}
    for name, value in values.items():
        copied[name] = value.clone() if isinstance(value, torch.Tensor) else value
        if hasattr(value, "is_shuffled"):
            copied[name].is_shuffled = value.is_shuffled
    return copied


def assert_unmodified(before, after):
    for name, value in before.items():
        if isinstance(value, torch.Tensor):
            other = after[name]
            # Bytewise comparison works for packed FP4/E8M0 as well.
            if not torch.equal(value.contiguous().view(torch.uint8), other.contiguous().view(torch.uint8)):
                raise ValueError(f"Input mutation is unsupported: {name}")
