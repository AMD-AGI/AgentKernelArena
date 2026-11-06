# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Vector dialect helpers used by the kernel (subset of ROCm/FlyDSL python/flydsl/expr/vector.py)."""

from flydsl._mlir.dialects import vector as _vector
from flydsl.expr.typing import ReductionOp  # noqa: F401
from .meta import traced_op


@traced_op
def from_elements(*args, loc=None, ip=None, **kwargs):
    """Construct a vector from scalar elements, auto-unwrapping ArithValue wrappers."""
    from flydsl.expr import arith as _arith_ext

    if len(args) >= 2:
        args = list(args)
        elems = args[1]
        if isinstance(elems, (list, tuple)):
            args[1] = [_arith_ext.unwrap(v) for v in elems]
        return _vector.from_elements(*args, loc=loc, ip=ip, **kwargs)

    return _vector.from_elements(*args, loc=loc, ip=ip, **kwargs)


@traced_op
def bitcast(result_type, source, *, loc=None, ip=None):
    """Wrapper around `vector.BitCastOp(...).result`."""
    from flydsl.expr import arith as _arith_ext

    return _vector.BitCastOp(
        result_type,
        _arith_ext.unwrap(source),
        loc=loc,
        ip=ip,
    ).result
