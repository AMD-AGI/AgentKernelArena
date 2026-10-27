"""CPU control for the hgemm task-local MLIR buffer offset adapter."""

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest


BUFFER_OPS = (
    Path(__file__).resolve().parents[1]
    / "tasks/torch2flydsl/hgemm_kernel/hgemm_kernel_ops/buffer_ops.py"
)


@pytest.mark.parametrize('shape,width,expected', [
    ((2, 3), 16, 12), ((7, 2), 32, 56), ((2, 0), 16, 0),
    ((-1, 3), 16, None), ((2, 3), None, None), ((2, 3), 4, None),
])
def test_static_buffer_byte_extent_preserves_width_and_unknown_cases(shape, width, expected):
    compatibility = BUFFER_OPS.parents[1] / 'flydsl_compat/buffer_ops.py'
    tree = ast.parse(compatibility.read_text())
    function = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == '_num_records_from_memref_type')
    metadata = SimpleNamespace(shape=shape, element_type=(
        SimpleNamespace() if width is None else SimpleNamespace(width=width)))
    namespace = dict(Optional=Optional, ir=SimpleNamespace(MemRefType=lambda value: value),
                     _unwrap_value=lambda value: value, memref_val=SimpleNamespace(type=metadata))
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(compatibility), 'exec'), namespace)
    assert namespace['_num_records_from_memref_type']() == expected


def test_integer_widths_use_matching_mlir_offset_conversion():
    source = ast.parse(BUFFER_OPS.read_text())
    function = next(
        node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "_to_i32"
    )
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    calls = []

    class IntegerType:
        def __init__(self, width):
            self.width = width

    class IndexType:
        pass

    class Value:
        def __init__(self, type_):
            self.type = type_

    def convert(name):
        def operation(target, source):
            calls.append(name)
            return SimpleNamespace(result=Value(target))
        return operation

    namespace = {
        "ir": SimpleNamespace(IntegerType=IntegerType),
        "T": SimpleNamespace(i32=lambda: IntegerType(32)),
        "arith": SimpleNamespace(
            TruncIOp=convert("trunc"),
            ExtSIOp=convert("sign_extend"),
            IndexCastOp=convert("index_cast"),
        ),
        "as_ir_value": lambda value: value,
        "_i32": lambda value: Value(IntegerType(32)),
    }
    exec(compile(module, str(BUFFER_OPS), "exec"), namespace)
    to_i32 = namespace["_to_i32"]
    for original, expected in (
        (Value(IntegerType(8)), "sign_extend"),
        (Value(IntegerType(64)), "trunc"),
        (Value(IndexType()), "index_cast"),
    ):
        calls.clear()
        assert to_i32(original).type.width == 32
        assert calls == [expected]
    calls.clear()
    i32 = Value(IntegerType(32))
    assert to_i32(i32) is i32
    assert calls == []
