# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Deterministic inputs, preallocated outputs and their poisoning.

Seeds come from the resolved contract, never from task code. The reference sees
the logical inputs; baseline and candidate see the kernel-facing inputs after
the task's optional layout transform, all prepared outside any timed interval.
"""
from __future__ import annotations

import hashlib
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


def make_inputs(contract: Contract, case: Case, seed: int, device: torch.device, *,
                rank: int | None = None) -> CaseInputs:
    """Logical and kernel-facing inputs for one case and seed, validated against the interface.

    For a multi-GPU task the declared interface is one rank's call: ``rank`` selects the rank whose
    shard (then layout) becomes the kernel-facing inputs. Without a rank only the logical inputs,
    which every rank draws identically from the same seed, are returned (``kernel`` is None).
    """
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
    if contract.distributed is not None and rank is None:
        return CaseInputs(logical, None)
    return CaseInputs(logical, kernel_inputs(contract, case, logical, device, rank=rank))


def kernel_inputs(contract: Contract, case: Case, logical: dict[str, Any], device: torch.device, *,
                  rank: int | None = None) -> dict[str, Any]:
    """The kernel-facing inputs: the rank's shard (multi-GPU tasks), then the layout transform."""
    symbols = contract.symbols(case)
    scalars = contract.scalars(case)
    if contract.distributed is None:
        source = logical
        owner = ""
    else:
        source = shard_inputs(contract, logical, rank)
        owner = f" for rank {rank}"
    layout = _load_task_function(contract.layout, "layout")
    if layout is None:
        kernel = dict(source)
    else:
        transformed = layout({"id": case.case_id, "axes": dict(symbols), "scalars": dict(scalars)},
                             {name: clone_value(value) for name, value in source.items()})
        if not isinstance(transformed, dict) or set(transformed) != set(contract.inputs):
            raise ContractError("Task layout transform must return every declared input")
        if any(name in contract.mutable and transformed[name] is not None and
               not torch.equal(transformed[name], source[name]) for name in contract.mutable):
            raise ContractError("A layout transform cannot change a mutable input")
        kernel = transformed
    for name, declaration in contract.inputs.items():
        value = kernel[name]
        if declaration["kind"] == "scalar":
            continue
        expected = contract.shape(declaration, symbols)
        if not isinstance(value, torch.Tensor):
            raise ContractError(f"Input {name}{owner} is not a tensor")
        if tuple(value.shape) != expected or value.dtype != _dtype(declaration["dtype"]):
            raise ContractError(f"Input {name}{owner} is {tuple(value.shape)}/{value.dtype}, "
                                f"declared {expected}/{declaration['dtype']}")
        if value.device != device:
            raise ContractError(f"Input {name}{owner} is not on {device}")
        if not value.is_contiguous():
            kernel[name] = value.contiguous()
        elif contract.distributed is not None and _is_partial_view(value):
            # A shard is often a slice of a full logical tensor; own it, so the full copy is freed.
            kernel[name] = value.clone()
    return kernel


def _is_partial_view(value: torch.Tensor) -> bool:
    return value.untyped_storage().nbytes() != value.numel() * value.element_size()


def shard_inputs(contract: Contract, logical: dict[str, Any], rank: int) -> dict[str, Any]:
    """One rank's kernel-facing inputs (before the layout transform) from the logical inputs."""
    world_size = contract.world_size
    if type(rank) is not int or not 0 <= rank < world_size:
        raise ContractError(f"rank {rank!r} is outside 0..{world_size - 1}")
    shard = _load_task_function(contract.distributed["shard"], "shard")
    produced = shard({name: clone_value(value) for name, value in logical.items()},
                     rank=rank, world_size=world_size)
    if not isinstance(produced, dict) or set(produced) != set(contract.inputs):
        missing = sorted(set(contract.inputs) - set(produced or {})) if isinstance(produced, dict) else None
        raise ContractError(f"Task shard function must return every declared input for rank {rank}"
                            + (f"; missing {missing}" if missing else ""))
    return produced


def expected_outputs(contract: Contract, case: Case, references: dict[str, torch.Tensor],
                     rank: int) -> dict[str, torch.Tensor]:
    """The tensors one rank must write, from the logical reference outputs."""
    expected = _load_task_function(contract.distributed["expected"], "expected")
    produced = expected(dict(references), rank=rank, world_size=contract.world_size)
    names = contract.checked_tensors
    if not isinstance(produced, dict) or set(produced) != set(names):
        raise ContractError(f"Task expected function must return exactly {list(names)} for rank {rank}")
    symbols = contract.symbols(case)
    for name in names:
        value = produced[name]
        declaration = contract.outputs.get(name) or contract.inputs[name]
        shape = contract.shape(declaration, symbols)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            got = tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__
            raise ContractError(f"Expected {name} for rank {rank} is {got}; one rank writes {shape}")
        if _is_partial_view(value):
            produced[name] = value.clone()  # do not keep the full logical reference alive
    return produced


_FINGERPRINT_SAMPLE = 1 << 20


def _host_bytes(flat: torch.Tensor) -> bytes:
    host = flat.cpu()
    try:
        return host.numpy().tobytes()
    except (RuntimeError, ImportError):  # PyTorch built without NumPy
        return bytes(host.tolist())


def tensor_digest(tensor: torch.Tensor) -> str:
    """SHA-256 of a tensor's dtype, shape and bytes: equal digests mean bitwise-equal tensors."""
    digest = hashlib.sha256(f"{tuple(tensor.shape)}|{tensor.dtype}|".encode())
    digest.update(_host_bytes(tensor_bytes(tensor)))
    return digest.hexdigest()


def logical_fingerprint(logical: dict[str, Any]) -> str:
    """A cheap identity of the logical inputs: every rank must draw the same values from one seed.

    Large tensors contribute an evenly spaced byte sample; a diverging random stream differs almost
    everywhere, so the sample detects it without copying whole weights to the host.
    """
    digest = hashlib.sha256()
    for name in sorted(logical):
        value = logical[name]
        digest.update(name.encode() + b"\0")
        if isinstance(value, torch.Tensor):
            digest.update(f"{tuple(value.shape)}|{value.dtype}|".encode())
            flat = tensor_bytes(value)
            if flat.numel() > _FINGERPRINT_SAMPLE:
                index = torch.linspace(0, flat.numel() - 1, _FINGERPRINT_SAMPLE, device=flat.device,
                                       dtype=torch.float64).to(torch.int64)
                flat = flat[index]
            digest.update(_host_bytes(flat))
        else:
            digest.update(repr(value).encode())
    return digest.hexdigest()


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


def check_rank_views(contract: Contract, case: Case, logical: dict[str, Any],
                     references: dict[str, torch.Tensor], device: torch.device) -> list[dict]:
    """Every rank's shard and expected outputs exist and match the one-rank interface.

    Runs the task's shard, layout and expected functions for each rank in one process, without
    kernels, so validate-task and the CPU pre-check can reject a malformed multi-GPU task early.
    """
    views = []
    for rank in range(contract.world_size):
        kernel = kernel_inputs(contract, case, logical, device, rank=rank)
        expected = expected_outputs(contract, case, references, rank)
        views.append({"rank": rank,
                      "inputs": {n: list(v.shape) for n, v in kernel.items() if isinstance(v, torch.Tensor)},
                      "expected": {n: list(v.shape) for n, v in expected.items()}})
        del kernel, expected
    return views
