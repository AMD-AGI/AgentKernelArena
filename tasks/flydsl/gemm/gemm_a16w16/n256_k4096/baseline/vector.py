# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.
"""Only the MLIR vector builders used by this task's FlyDSL kernel.

FlyDSL 0.3 exposes the raw vector dialect rather than ``expr.vector``.  These
small adapters convert its DSL values to the raw operands expected by MLIR.
"""

from flydsl._mlir import ir
from flydsl._mlir.dialects import arith, vector
from flydsl.expr.typing import as_ir_value


def _index(value):
    value = as_ir_value(value)
    if isinstance(value.type, ir.IntegerType):
        return arith.IndexCastOp(ir.IndexType.get(), value).result
    return value


def from_elements(result_type, elements):
    return vector.from_elements(result_type, [as_ir_value(value) for value in elements])


def bitcast(result_type, source):
    return vector.bitcast(result_type, as_ir_value(source))


def extract(source, *, static_position, dynamic_position):
    return vector.extract(
        as_ir_value(source),
        [_index(value) for value in dynamic_position],
        static_position,
    )


def load_op(result_type, memref, indices):
    return vector.load(result_type, as_ir_value(memref), [_index(value) for value in indices])


def store(value, memref, indices, *, alignment):
    return vector.store(
        as_ir_value(value), as_ir_value(memref), [_index(index) for index in indices], alignment=alignment
    )


def broadcast(result_type, source):
    return vector.broadcast(result_type, as_ir_value(source))
