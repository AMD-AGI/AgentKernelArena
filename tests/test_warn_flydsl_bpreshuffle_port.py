"""CPU checks for the task-local FlyDSL 0.3.2 raw-buffer adapter."""

import ast
from pathlib import Path
from types import SimpleNamespace


KERNEL = (
    Path(__file__).resolve().parents[1]
    / "tasks/torch2flydsl/gemm_a8w8_bpreshuffle_kernel/kernel.py"
)


class Type:
    def __init__(self, width=None, element_type=None):
        self.width = width
        if element_type is not None:
            self.element_type = element_type


class Value:
    def __init__(self, value, type_):
        self.value = value
        self.type = type_


class Op:
    def __init__(self, result):
        self.result = result


def _adapter():
    """Execute only the adapter class with recording MLIR stand-ins."""
    source = ast.parse(KERNEL.read_text())
    adapter = next(
        node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "_RawBufferOps"
    )
    module = ast.fix_missing_locations(ast.Module(body=[adapter], type_ignores=[]))
    calls = []

    class IntegerType(Type):
        @classmethod
        def get_signless(cls, bits):
            return cls(bits)

    class IndexType(Type):
        pass

    class VectorType(Type):
        @classmethod
        def get(cls, shape, elem):
            return cls(elem.width * shape[0], elem)

    ir = SimpleNamespace(
        Value=Value,
        IntegerType=IntegerType,
        IndexType=IndexType,
        VectorType=VectorType,
        F32Type=SimpleNamespace(get=lambda: Type(32)),
        IntegerAttr=SimpleNamespace(get=lambda _ty, n: n),
        Type=SimpleNamespace(parse=lambda name: Type()),
    )
    arith = SimpleNamespace(
        ConstantOp=lambda ty, n: Op(Value(n, ty)),
        IndexCastOp=lambda ty, src: record("index_cast", Value(src.value, ty), ty, src),
        TruncIOp=lambda ty, src: record("trunc", Value(src.value, ty), ty, src),
        ExtSIOp=lambda ty, src: record("sign_extend", Value(src.value, ty), ty, src),
        MulIOp=lambda a, b: Op(Value(a.value * b.value, a.type)),
        AddIOp=lambda a, b: Op(Value(a.value + b.value, a.type)),
    )

    def record(name, result, *args, **kwargs):
        calls.append((name, args, kwargs))
        return Op(result)

    llvm = SimpleNamespace(
        IntToPtrOp=lambda ty, src: record("ptr", Value(src.value, ty), ty, src),
        GEPOp=lambda *args: record("gep", Value(0, args[0]), *args),
    )
    rocdl = SimpleNamespace(
        MakeBufferRsrcOp=lambda *args: record("rsrc", Value(0, args[0]), *args),
        RawPtrBufferLoadOp=lambda *args, **kwargs: record(
            "load", Value(0, args[0]), *args, **kwargs
        ),
        RawPtrBufferStoreOp=lambda *args, **kwargs: record(
            "store", None, *args, **kwargs
        ),
    )
    namespace = {
        "ir": ir,
        "mlir_arith": arith,
        "llvm": llvm,
        "mlir_rocdl": rocdl,
        "get_hip_arch": lambda: "gfx950",
        "is_rdna_arch": lambda arch: arch.startswith("gfx11"),
    }
    exec(compile(module, str(KERNEL), "exec"), namespace)
    return namespace["_RawBufferOps"], ir, calls


def test_buffer_descriptor_and_load_store_offsets():
    adapter, ir, calls = _adapter()
    i32 = ir.IntegerType.get_signless(32)
    resource = adapter.create_buffer_resource_from_addr(
        Value(0x1000, ir.IntegerType.get_signless(64)), num_records_bytes=8192
    )
    descriptor = next(args for name, args, _ in calls if name == "rsrc")
    assert descriptor[3].value == 8192
    assert descriptor[4].value == (7 << 12) | (4 << 15)

    adapter.buffer_load(resource, Value(3, i32), vec_width=4, dtype=i32)
    load = next(args for name, args, _ in calls if name == "load")
    assert load[2].value == 12  # i32 element offset -> byte offset
    assert load[0].element_type.width == 32

    data = Value(0, ir.VectorType.get([2], i32))
    adapter.buffer_store(data, resource, Value(7, i32))
    adapter.buffer_store(data, resource, Value(14, i32), offset_is_bytes=True)
    adapter.buffer_store(Value(0, ir.IntegerType.get_signless(16)), resource, Value(7, i32))
    stores = [args for name, args, _ in calls if name == "store"]
    assert [args[2].value for args in stores] == [28, 14, 14]

    pointer = Value(0x2000, ir.Type.parse("!llvm.ptr<3>"))
    adapter.get_element_ptr(pointer, Value(48, i32), static_byte_offset=64)
    gep = next(args for name, args, _ in calls if name == "gep")
    assert gep[2][0].value == 112  # async LDS pointer offset remains in bytes


def test_buffer_offset_integer_widths_use_valid_mlir_casts():
    adapter, ir, calls = _adapter()
    for value, expected in (
        (Value(7, ir.IntegerType.get_signless(8)), "sign_extend"),
        (Value(7, ir.IntegerType.get_signless(64)), "trunc"),
        (Value(7, ir.IndexType()), "index_cast"),
    ):
        calls.clear()
        converted = adapter._i32(value)
        assert converted.type.width == 32
        assert [name for name, _, _ in calls] == [expected]
    calls.clear()
    i32 = Value(7, ir.IntegerType.get_signless(32))
    assert adapter._i32(i32) is i32
    assert calls == []


def test_kernel_uses_installed_flydsl_primitives_without_forbidden_backend():
    tree = ast.parse(KERNEL.read_text())
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            modules.append(node.module or "")
        elif isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        if isinstance(node, ast.Attribute) and node.attr == "load_op":
            raise AssertionError("removed FlyDSL vector.load_op API")
    assert all(not module.startswith(("aiter", "kernels")) for module in modules)
    assert any(module == "flydsl._mlir.dialects" for module in modules)
    assert any(isinstance(node, ast.ClassDef) and node.name == "_RawBufferOps" for node in tree.body)
