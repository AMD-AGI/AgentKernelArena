"""Input construction through the bundle's own ``initialize`` callback.

Buffers are allocated in the definition's shapes and dtypes and filled in place
by the bundle callback; no distribution or encoding is decided here. The
workload policy names the operands a production caller holds across calls
(``persistent_inputs``); timing redraws only the others.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from scripts.task_api import dimensions, dtype, load_solution, shape_of, validate_inputs


def row_seed(policy, row):
    """The base draw's seed, a function of the policy seed and the row identity."""
    return int.from_bytes(hashlib.sha256(
        f"{policy['seed']}:{row['workload']['uuid']}".encode()).digest()[:8], "little") % (2**63)


def allocate(definition, row, device="cuda"):
    axes = dimensions(definition, row)
    return {name: (row["workload"]["inputs"][name]["value"] if spec.get("shape") is None else
                   torch.empty(shape_of(spec, axes), dtype=dtype(spec["dtype"]), device=device))
            for name, spec in definition["inputs"].items()}


def initialize_buffers(values, definition, row, seed, device):
    initialize = load_solution(Path(__file__).parent / "initialize", "main.py::run")
    original = dict(values)
    storage = {name: (v.data_ptr(), v.stride()) for name, v in values.items() if isinstance(v, torch.Tensor)}
    if initialize(values, seed=seed) is not values:
        raise ValueError("initialize must return the original input dictionary")
    validate_inputs(values, definition, row, device)
    for name, (pointer, stride) in storage.items():
        if values[name] is not original[name] or values[name].data_ptr() != pointer or values[name].stride() != stride:
            raise ValueError(f"initialize replaced input buffer: {name}")
    return values


def make_inputs(definition, row, policy, device="cuda"):
    if not definition.get("initialize"):
        raise ValueError("This task requires the definition's initialize callback")
    return initialize_buffers(allocate(definition, row, device), definition, row, row_seed(policy, row), device)


def persistent_inputs(definition, policy):
    names = tuple(policy["persistent_inputs"])
    tensors = {name for name, spec in definition["inputs"].items() if spec.get("shape") is not None}
    if not set(names) < tensors:
        raise ValueError("persistent_inputs must be a proper subset of the tensor inputs")
    return names


def call_varying_draws(values, definition, row, policy, seeds, device="cuda"):
    """One snapshot of the call-varying operands per seed, drawn by the bundle.

    Every draw runs the full callback on scratch buffers; the persistent
    operands of the live buffers are untouched, so a draw differs from the base
    exactly where two calls on a live model differ.
    """
    held = persistent_inputs(definition, policy)
    draws = []
    for seed in seeds:
        scratch = initialize_buffers(allocate(definition, row, device), definition, row, seed, device)
        draws.append({name: value for name, value in scratch.items()
                      if isinstance(value, torch.Tensor) and name not in held})
    return draws


def load_draw(values, draw):
    """Copy a snapshot into the live buffers, keeping their storage."""
    for name, value in draw.items():
        values[name].copy_(value)
