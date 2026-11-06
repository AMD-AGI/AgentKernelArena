# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""traced_op from FlyDSL python/flydsl/expr/meta.py for runtimes without flydsl.expr.buffer_ops.

The wrapper unwraps DSL values to raw MLIR values and runs the op under a location. Ops without an
explicit ``loc`` get an unknown location instead of one built from the Python call stack.
"""

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


def traced_op(op):
    @wraps(op)
    def wrapper(*args, **kwargs):
        loc = kwargs.pop("loc", None)
        if loc is None:
            loc = ir.Location.unknown()
        args, kwargs = _flatten_args(args, kwargs)
        with loc:
            return op(*args, **kwargs)

    return wrapper
