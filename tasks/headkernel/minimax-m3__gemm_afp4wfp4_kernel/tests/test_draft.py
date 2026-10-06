"""CPU format, protected binding and capture-adapter checks; no GPU claims."""
import ast
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ut"))
from binding import bind_globals, raw_wrapper_tree
from capture_contract import validate_abi, validate_launch
from reference import FP4, decode_e8m0, matmul, storage_rows, unpack_row
from source_guard import validate_sources


def test_all_fp4_codes_and_nibble_order():
    assert unpack_row([0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE] * 2, [127]) == list(FP4) * 2
    assert math.copysign(1, unpack_row([0x88] * 16, [127])[0]) == -1


def test_e8m0_zero_and_nan_are_not_silently_erased():
    assert decode_e8m0(0) == 2**-127
    assert decode_e8m0(127) == 1
    assert decode_e8m0(128) == 2
    assert math.isnan(decode_e8m0(255))
    with pytest.raises(ValueError, match="NaN"):
        unpack_row([0] * 16, [255])


def test_reference_signed_products_and_splitk():
    x = [[0x22] * 16 + [0xAA] * 16]
    w = [[0x22] * 32, [0xAA] * 32]
    assert matmul(x, w, [[128, 127]], [[127, 127], [127, 127]]) == [[32.0, -32.0]]
    assert matmul(x, w, [[128, 127]], [[127, 127], [127, 127]], splitk_block_size=32) == [[[64.0, -64.0]], [[-32.0, 32.0]]]


def test_strided_scale_storage_and_bounds():
    assert storage_rows(bytes(range(30)), [2, 3], [2, 7], 1) == [[1, 8, 15], [3, 10, 17]]
    with pytest.raises(ValueError, match="exceeds"):
        storage_rows(bytes(8), [2, 3], [2, 7], 1)


def test_wrapper_transformation_removes_only_registration():
    text = (ROOT / "ut/native/wrapper.py").read_text()
    expected = ast.parse(text)
    fn, = [node for node in expected.body if isinstance(node, ast.FunctionDef) and node.name == "gemm_afp4wfp4_"]
    fn.decorator_list = []
    assert ast.dump(raw_wrapper_tree(text)) == ast.dump(expected)
    with pytest.raises(ValueError, match="context"):
        raw_wrapper_tree(text.replace("gen_fake=gemm_afp4wfp4_fake_tensor", "gen_fake=None"))


def test_private_binding_invokes_each_selected_leg():
    wrappers = []
    for value in ("reference", "candidate"):
        module = ModuleType(value)
        exec("def gemm_afp4wfp4_(): return _triton_gemm_afp4wfp4_kernel()\ndef gemm_afp4wfp4(): return gemm_afp4wfp4_()", module.__dict__)
        kernel = SimpleNamespace(_gemm_afp4wfp4_kernel=lambda value=value: value, _get_config=lambda: None)
        wrappers.append(bind_globals(module, kernel))
    assert [call() for call in wrappers] == ["reference", "candidate"]


@pytest.mark.parametrize("edit,accepted", [("zero", True), ("host", False), ("decorator", False)])
def test_source_guard_accepts_device_control_and_rejects_host_edits(tmp_path, edit, accepted):
    shutil.copytree(ROOT / "source", tmp_path / "source")
    path = tmp_path / "source/kernel.py"
    text = path.read_text()
    if edit == "zero":
        text = text.replace("c = accumulator.to(c_ptr.type.element_ty)", "c = (accumulator * 0).to(c_ptr.type.element_ty)", 1)
    elif edit == "host":
        text = text.replace("tl.assume(stride_am > 0)", "open('forged-report', 'w')", 1)
    else:
        text = text.replace("@triton.jit(repr=_gemm_afp4wfp4_repr)", "@triton.jit")
    path.write_text(text)
    if accepted:
        assert validate_sources(tmp_path, ROOT)
    else:
        with pytest.raises(ValueError):
            validate_sources(tmp_path, ROOT)


def test_abi_distinguishes_packed_k_and_scale_layout():
    tensor = lambda shape, strides: {"dtype": "uint8", "shape": shape, "strides": strides, "storage_offset": 0}
    tensors = {"x": tensor([2, 16], [16, 1]), "w": tensor([3, 16], [1, 3]),
               "x_scales": tensor([2, 1], [1, 2]), "w_scales": tensor([3, 1], [1, 3])}
    controls = {"dtype": "bfloat16", "skip_reduce": False, "use_splitk_bf16": False}
    abi = validate_abi(tensors, controls)
    assert abi == {"M": 2, "N": 3, "K_packed_bytes": 16, "K_logical_values": 32}
    launch = {"arguments": {"M": 2, "N": 3, "K": 16, "BLOCK_SIZE_M": 32,
        "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 128, "GROUP_SIZE_M": 1, "NUM_KSPLIT": 1, "SPLITK_BLOCK_SIZE": 32}}
    validate_launch(abi, launch)
    launch["arguments"]["K"] = 32
    with pytest.raises(ValueError, match="operand ABI"):
        validate_launch(abi, launch)


@pytest.mark.parametrize("kind", ["no_op", "wrong_output"])
def test_generated_source_controls_pass_the_unchanged_boundary(tmp_path, kind):
    spec = importlib.util.spec_from_file_location("fp4_controls", ROOT / "scripts/make_source_controls.py")
    controls = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controls)
    (tmp_path / "source").mkdir()
    source = controls.control_source(kind)
    assert source != (ROOT / "ut/reference/kernel.py").read_text()
    (tmp_path / "source/kernel.py").write_text(source)
    assert validate_sources(tmp_path, ROOT)


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("retained_globals", [False, True])
@pytest.mark.parametrize("launch_count", [0, 1, 2])
def test_capture_hooks_cached_quark_global_and_records_native_launch(monkeypatch, graph, retained_globals, launch_count):
    spec = importlib.util.spec_from_file_location("fp4_capture_adapter", ROOT / "capture/adapter.py")
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    calls = []
    class Jit:
        def run(self, *args, **kwargs):
            return SimpleNamespace(name="native_fp4", hash="compiled-hash")
        def __getitem__(self, grid):
            # Triton's KernelInterface resolves run dynamically on this object.
            return lambda *args, **kwargs: self.run(*args, grid=grid, warmup=False, **kwargs)
    basic = ModuleType("fake_basic")
    basic.__file__ = str(ROOT / "ut/native/wrapper.py")
    basic._triton_gemm_afp4wfp4_kernel = Jit()
    basic._USE_GEMM_SPLITK_BF16 = False
    kernel_module = SimpleNamespace(__file__=str(ROOT / "source/kernel.py"),
                                    _gemm_afp4wfp4_kernel=basic._triton_gemm_afp4wfp4_kernel)
    # Model AITER's redirect: a cached custom-op function keeps the first
    # execution's globals, while the current module can be a second instance.
    raw_globals = dict(basic.__dict__) if retained_globals else basic.__dict__
    raw_globals["_USE_GEMM_SPLITK_BF16"] = retained_globals
    raw_globals["launch_count"] = launch_count
    code = """def gemm_afp4wfp4_(x, w, x_scales, w_scales, dtype, y, config, skip_reduce):
    for _ in range(launch_count):
        _triton_gemm_afp4wfp4_kernel[(1,)](x, w, y, x_scales, w_scales, 2, 3, 16,
            BLOCK_SIZE_M=32, BLOCK_SIZE_N=32, BLOCK_SIZE_K=128, NUM_KSPLIT=1)
    return y
"""
    exec(compile(code, basic.__file__, "exec"), raw_globals)
    registered_native = raw_globals["gemm_afp4wfp4_"]
    def original(x, w, x_scales, w_scales, dtype="bfloat16", y=None, config=None, skip_reduce=False):
        return registered_native(x, w, x_scales, w_scales, dtype, y, config, skip_reduce)
    basic.gemm_afp4wfp4 = original
    quark = ModuleType("fake_quark")
    quark.__file__ = str(ROOT / "ut/native/quark_linear.py")
    quark._gemm_afp4wfp4_orig = original
    exec("def registered(*args): return _gemm_afp4wfp4_orig(*args)", quark.__dict__)
    cached = quark.registered
    recorder = SimpleNamespace(begin_eager=lambda *args: SimpleNamespace(controls=args[2]),
        begin_graph=lambda *args, **kwargs: SimpleNamespace(controls=args[2]),
        finish=lambda handle, outputs: calls.append((handle.controls, outputs)),
        abort=lambda *args: None)
    cap = SimpleNamespace(__file__=str(ROOT / "capture/adapter.py"), capturing=lambda: graph,
        controls_json=lambda value: value, Role=lambda *args, **kw: (args, kw),
        Family=lambda *args: args)
    monkeypatch.setattr(adapter, "COMMON_SHA256", adapter.hashlib.sha256(Path(cap.__file__).read_bytes()).hexdigest())
    adapter.install(cap, quark, basic, kernel_module, lambda: {"recorder": recorder, "served": object(),
        "graph_id": "graph-1", "slot_id": "minimax_fp4_gemm:0", "bucket": "actual-observed"})
    result = object()
    if launch_count != 1:
        with pytest.raises(ValueError, match="exactly one declared FP4 GEMM launch: " + str(launch_count)):
            cached(object(), object(), object(), object(), "bfloat16", result)
        assert calls == []
        return
    assert cached(object(), object(), object(), object(), "bfloat16", result) is result
    assert calls[0][0]["launches"][0]["arguments"]["K"] == 16
    assert "warmup" not in calls[0][0]["launches"][0]["arguments"]
    assert "grid" not in calls[0][0]["launches"][0]["arguments"]
    assert calls[0][0]["use_splitk_bf16"] is retained_globals
    assert basic._triton_gemm_afp4wfp4_kernel is kernel_module._gemm_afp4wfp4_kernel
    assert calls[0][1]["result"] is result
