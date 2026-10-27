# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

import traceback
from functools import wraps

from flydsl._mlir import ir


def _to_raw_value(obj):
    if isinstance(obj, ir.Value):
        return obj
    if isinstance(obj, type):
        return obj
    if hasattr(obj, "__extract_to_ir_values__"):
        values = obj.__extract_to_ir_values__()
        if len(values) != 1:
            raise ValueError(f"Primitive function expects 1 value, got {len(values)}")
        return values[0]
    if isinstance(obj, tuple):
        return tuple(_to_raw_value(e) for e in obj)
    if isinstance(obj, list):
        return [_to_raw_value(e) for e in obj]
    return obj


def _flatten_args(args, kwargs):
    new_args = tuple(_to_raw_value(a) for a in args)
    new_kwargs = {k: _to_raw_value(v) if k not in ("loc", "ip") else v for k, v in kwargs.items()}
    return new_args, new_kwargs


def _caller_location(depth=1):
    """Build a line-level MLIR location without retaining Python frame objects."""
    limit = max(1, depth + 2)
    stack = traceback.extract_stack(limit=limit)
    if len(stack) < limit:
        return ir.Location.unknown()
    info = stack[0]
    file_loc = ir.Location.file(info.filename, info.lineno, 0)
    label = info.line.strip() if info.line else info.name
    return ir.Location.name(label, childLoc=file_loc)


def traced_op(op):
    @wraps(op)
    def wrapper(*args, **kwargs):
        loc = kwargs.pop("loc", None)
        if loc is None:
            loc = _caller_location(depth=1)
        args, kwargs = _flatten_args(args, kwargs)
        with loc:
            return op(*args, **kwargs)

    return wrapper
