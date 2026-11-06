# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Deterministic inputs, preallocated outputs and their poisoning.

Seeds come from the resolved contract, never from task code. The reference sees
the logical inputs; baseline and candidate see the kernel-facing inputs after
the task's optional layout transform, all prepared outside any timed interval.
"""
from __future__ import annotations

import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from .contract import WORKSPACE, Case, Contract, ContractError


def _dtype(name: str) -> torch.dtype:
    aliases = {"bfloat16": "bfloat16", "float16": "float16", "float32": "float32",
               "float64": "float64", "int8": "int8", "uint8": "uint8", "int16": "int16",
               "int32": "int32", "int64": "int64", "bool": "bool",
               "float8_e4m3fn": "float8_e4m3fn", "float8_e4m3fnuz": "float8_e4m3fnuz",
               "float8_e5m2": "float8_e5m2", "float8_e5m2fnuz": "float8_e5m2fnuz",
               "float8_e8m0fnu": "float8_e8m0fnu", "float4_e2m1fn_x2": "float4_e2m1fn_x2"}
    if name not in aliases or not hasattr(torch, aliases[name]):
        raise ContractError(f"dtype {name!r} is unavailable in this PyTorch")
    return getattr(torch, aliases[name])


# Unit roundoff of each output dtype; integral and packed codes compare exactly.
UNIT_ROUNDOFF = {
    "float64": 2.0 ** -53, "float32": 2.0 ** -24, "float16": 2.0 ** -11, "bfloat16": 2.0 ** -8,
    "float8_e4m3fn": 2.0 ** -4, "float8_e4m3fnuz": 2.0 ** -4,
    "float8_e5m2": 2.0 ** -3, "float8_e5m2fnuz": 2.0 ** -3,
}
NAN_POISONED = frozenset({"float64", "float32", "float16", "bfloat16"})


def torch_dtype(name: str) -> torch.dtype:
    return _dtype(name)


def is_exact_dtype(name: str) -> bool:
    return name not in UNIT_ROUNDOFF


def _load_task_function(entry: dict | None, label: str) -> Callable | None:
    if entry is None:
        return None
    path = (WORKSPACE / entry["file"]).resolve(strict=True)
    if not path.is_relative_to(WORKSPACE):
        raise ContractError(f"{label} escapes the task")
    name = f"_aka_task_{label}"
    if name in sys.modules:
        module = sys.modules[name]
    else:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ContractError(f"Cannot load {label} from {entry['file']}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    function = getattr(module, entry["symbol"], None)
    if not callable(function):
        raise ContractError(f"{entry['file']} does not define callable {entry['symbol']}")
    return function


def reference_function(contract: Contract) -> Callable:
    return _load_task_function(contract.reference, "reference")


def _builtin(declaration: dict, shape: tuple[int, ...], generator: torch.Generator,
             device: torch.device) -> torch.Tensor:
    dist = declaration["distribution"]
    kind = dist["kind"]
    dtype = _dtype(declaration["dtype"])
    # In-place arithmetic keeps one float32 copy alive (the same operations, so the same values): large
    # weight stacks otherwise need three full-size temporaries while they are drawn.
    if kind == "normal":
        value = torch.randn(shape, generator=generator, device=device, dtype=torch.float32)
        value.mul_(float(dist.get("std", 1.0))).add_(float(dist.get("mean", 0.0)))
    elif kind == "uniform":
        low, high = float(dist["low"]), float(dist["high"])
        value = torch.rand(shape, generator=generator, device=device, dtype=torch.float32)
        value.mul_(high - low).add_(low)
    elif kind == "log_uniform":
        low, high = float(dist["log2_low"]), float(dist["log2_high"])
        value = torch.rand(shape, generator=generator, device=device, dtype=torch.float32)
        value.mul_(high - low).add_(low).exp2_()
        if dist.get("signed", False):
            sign = torch.randint(0, 2, shape, generator=generator, device=device)
            value.mul_(sign.mul_(2).sub_(1))
            del sign
    elif kind == "randint":
        return torch.randint(int(dist["low"]), int(dist["high"]), shape, generator=generator,
                             device=device, dtype=torch.int64).to(dtype)
    elif kind == "permutation":
        if len(shape) != 1:
            raise ContractError("A permutation input must be one-dimensional")
        return torch.randperm(shape[0], generator=generator, device=device).to(dtype)
    elif kind == "zeros":
        return torch.zeros(shape, device=device, dtype=dtype)
    else:
        raise ContractError(f"Unknown distribution {kind!r}")
    return value.to(dtype)


def device_generator(seed: int, device: torch.device) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


@dataclass
class CaseInputs:
    logical: dict[str, Any]
    kernel: dict[str, Any]


def make_inputs(contract: Contract, case: Case, seed: int, device: torch.device) -> CaseInputs:
    """Logical and kernel-facing inputs for one case and seed, validated against the interface."""
    symbols = contract.symbols(case)
    generator = device_generator(seed, device)
    scalars = contract.scalars(case)
    logical: dict[str, Any] = {}
    owned = []
    for name, declaration in contract.inputs.items():
        if declaration["kind"] == "scalar":
            logical[name] = scalars[name]
        elif declaration["distribution"]["kind"] == "generator":
            owned.append(name)
        else:
            logical[name] = _builtin(declaration, contract.shape(declaration, symbols), generator, device)
    task_generator = _load_task_function(contract.generator, "generator")
    if owned:
        if task_generator is None:
            raise ContractError("Inputs declare distribution generator but the task has no generator")
        produced = task_generator({"id": case.case_id, "axes": dict(symbols), "scalars": dict(scalars)},
                                  dict(logical), generator, device)
        if not isinstance(produced, dict) or set(produced) != set(owned):
            raise ContractError(f"Task generator must return exactly {sorted(owned)}")
        logical.update(produced)
    layout = _load_task_function(contract.layout, "layout")
    if layout is None:
        kernel = dict(logical)
    else:
        transformed = layout({"id": case.case_id, "axes": dict(symbols), "scalars": dict(scalars)},
                             {name: clone_value(value) for name, value in logical.items()})
        if not isinstance(transformed, dict) or set(transformed) != set(contract.inputs):
            raise ContractError("Task layout transform must return every declared input")
        if any(name in contract.mutable and transformed[name] is not None and
               not torch.equal(transformed[name], logical[name]) for name in contract.mutable):
            raise ContractError("A layout transform cannot change a mutable input")
        kernel = transformed
    for name, declaration in contract.inputs.items():
        value = kernel[name]
        if declaration["kind"] == "scalar":
            continue
        expected = contract.shape(declaration, symbols)
        if not isinstance(value, torch.Tensor):
            raise ContractError(f"Input {name} is not a tensor")
        if tuple(value.shape) != expected or value.dtype != _dtype(declaration["dtype"]):
            raise ContractError(f"Input {name} is {tuple(value.shape)}/{value.dtype}, "
                                f"declared {expected}/{declaration['dtype']}")
        if value.device != device:
            raise ContractError(f"Input {name} is not on {device}")
        if not value.is_contiguous():
            kernel[name] = value.contiguous()
    return CaseInputs(logical, kernel)


def clone_value(value: Any) -> Any:
    return value.clone() if isinstance(value, torch.Tensor) else value


def clone_inputs(values: dict[str, Any]) -> dict[str, Any]:
    return {name: clone_value(value) for name, value in values.items()}


def allocate_outputs(contract: Contract, case: Case, device: torch.device) -> dict[str, torch.Tensor]:
    symbols = contract.symbols(case)
    return {name: torch.empty(contract.shape(declaration, symbols), dtype=_dtype(declaration["dtype"]),
                              device=device)
            for name, declaration in contract.outputs.items()}


def poison(outputs: dict[str, torch.Tensor], salt: int) -> None:
    """Make every unwritten output element detectable, differently per call."""
    for index, (name, tensor) in enumerate(outputs.items()):
        if tensor.numel() == 0:
            continue
        dtype_name = str(tensor.dtype).removeprefix("torch.")
        if dtype_name in NAN_POISONED:
            tensor.fill_(math.nan)
        else:
            generator = device_generator((salt * 1000003 + index) & 0x7FFFFFFF, tensor.device)
            raw = tensor.view(torch.uint8)
            raw.copy_(torch.randint(0, 256, raw.shape, generator=generator, device=tensor.device,
                                    dtype=torch.int64).to(torch.uint8))


def copy_into(destination: dict[str, Any], source: dict[str, Any]) -> None:
    """Load new values into existing storage; scalars must be unchanged."""
    for name, value in source.items():
        if isinstance(value, torch.Tensor):
            destination[name].copy_(value)
        elif destination[name] != value:
            raise ContractError("Repeated-call inputs cannot change a scalar")


def snapshot(values: dict[str, Any], names) -> dict[str, torch.Tensor]:
    return {name: values[name].clone() for name in names if isinstance(values.get(name), torch.Tensor)}


def tensor_bytes(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().reshape(-1).view(torch.uint8)


def changed_inputs(before: dict[str, torch.Tensor], after: dict[str, Any]) -> list[str]:
    """Names whose bytes differ (NaN payloads included) after a call."""
    return [name for name, value in before.items()
            if after[name].shape != value.shape or after[name].dtype != value.dtype
            or not torch.equal(tensor_bytes(after[name]), tensor_bytes(value))]


def checked_values(contract: Contract, kernel_inputs: dict[str, Any],
                   outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    values = dict(outputs)
    for name in contract.mutable:
        values[name] = kernel_inputs[name]
    return values


def reference_outputs(contract: Contract, logical: dict[str, Any]) -> dict[str, torch.Tensor]:
    result = reference_function(contract)(**clone_inputs(logical))
    names = contract.checked_tensors
    if isinstance(result, torch.Tensor):
        result = (result,)
    if isinstance(result, (tuple, list)):
        if len(result) != len(names):
            raise ContractError(f"Reference returned {len(result)} tensors, expected {list(names)}")
        result = dict(zip(names, result))
    if not isinstance(result, dict) or set(result) != set(names):
        raise ContractError(f"Reference must return {list(names)}")
    for name, value in result.items():
        if not isinstance(value, torch.Tensor):
            raise ContractError(f"Reference output {name} is not a tensor")
    return result


def workspace_path(relative: str) -> Path:
    path = (WORKSPACE / relative).resolve(strict=True)
    if not path.is_relative_to(WORKSPACE):
        raise ContractError(f"{relative} escapes the workspace")
    return path
