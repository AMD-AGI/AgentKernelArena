"""CPU control for the hgemm task-local MLIR buffer offset adapter."""

import ast
from pathlib import Path
from types import SimpleNamespace


BUFFER_OPS = (
    Path(__file__).resolve().parents[1]
    / "tasks/torch2flydsl/hgemm_kernel/kernels/buffer_ops.py"
)


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
