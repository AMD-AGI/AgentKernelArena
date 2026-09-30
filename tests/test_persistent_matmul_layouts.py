"""Exercise the public wrapper's layout dispatch; GPU tests require real hardware."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


SOURCE = (Path(__file__).resolve().parents[1] / "tasks/triton2triton/vllm/"
          "triton_matmul_persistent/source/triton_matmul_persistent.py")


class TensorMetadata:
    def __init__(self, shape, strides=None, *, dtype="fp16", device="cuda:3"):
        self.shape, self.dtype, self.device = tuple(shape), dtype, device
        packed = []
        count = 1
        for size in reversed(shape):
            packed.insert(0, count)
            count *= size
        self.packed_strides = tuple(packed)
        self.strides = tuple(strides) if strides is not None else self.packed_strides

    def stride(self, index=None):
        return self.strides if index is None else self.strides[index]

    def dim(self):
        return len(self.shape)

    def numel(self):
        import math
        return math.prod(self.shape)

    def is_contiguous(self):
        return self.strides == self.packed_strides

    def contiguous(self):
        return TensorMetadata(self.shape, dtype=self.dtype, device=self.device)


@pytest.fixture
def wrapper(monkeypatch):
    launches, device_queries = [], []

    class Kernel:
        def __init__(self, fn):
            self.fn = fn

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                launches.append((args, kwargs))
            return launch

    def properties(device):
        device_queries.append(device)
        return SimpleNamespace(multi_processor_count=304)

    language = SimpleNamespace(constexpr=object())
    monkeypatch.setitem(sys.modules, "triton", SimpleNamespace(jit=Kernel, language=language))
    monkeypatch.setitem(sys.modules, "triton.language", language)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        Tensor=TensorMetadata, float16="fp16", bfloat16="bf16", float32="fp32",
        cuda=SimpleNamespace(get_device_properties=properties),
        empty=lambda shape, **kwargs: TensorMetadata(shape, **kwargs)))
    spec = importlib.util.spec_from_file_location("_matmul_layout_fixture", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.matmul_persistent, launches, device_queries


@pytest.mark.parametrize("shape,strides,expected", [
    ((2, 2), (2**31 - 2, 1), False),  # Last element is exactly INT32_MAX.
    ((2, 2), (2**31 - 1, 1), True),
    ((2, 2), (1, 2**31), True),
    ((2, 2), (0, 1), False),  # Preserve valid broadcast views.
    ((4, 4), (9, 2), False),
])
def test_dispatch_uses_address_span_instead_of_element_count(wrapper, shape, strides, expected):
    call, launches, _ = wrapper
    a = TensorMetadata(shape, strides)
    b = TensorMetadata((shape[1], shape[0]), tuple(reversed(strides)))
    call(a, b)
    args, flags = launches[-1]
    assert args[0] is a and args[1] is b
    assert flags["A_LARGE"] is expected
    assert flags["B_LARGE"] is expected
    assert flags["C_LARGE"] is False


def test_output_index_boundary_and_device_selection(wrapper):
    call, launches, queries = wrapper
    call(TensorMetadata((2, 1)), TensorMetadata((1, 2**30 + 1)))
    assert launches[-1][1]["C_LARGE"] is True
    assert queries == ["cuda:3"]


@pytest.mark.parametrize("dtype", ["fp16", "bf16", "fp32"])
def test_noncontiguous_bias_is_materialized_without_narrowing_dtype_support(wrapper, dtype):
    call, launches, _ = wrapper
    bias = TensorMetadata((3,), (2,), dtype=dtype)
    call(TensorMetadata((2, 2), dtype=dtype), TensorMetadata((2, 3), dtype=dtype), bias)
    received = launches[-1][0][3]
    assert received is not bias
    assert received.stride() == (1,)
    assert received.dtype == dtype
    assert bias.stride() == (2,)


def test_contiguous_bias_is_not_copied(wrapper):
    call, launches, _ = wrapper
    bias = TensorMetadata((3,))
    call(TensorMetadata((2, 2)), TensorMetadata((2, 3)), bias)
    assert launches[-1][0][3] is bias


@pytest.mark.parametrize("large_operand", [None, "a", "b"])
def test_gpu_strided_bias_and_large_address_span(large_operand):
    import torch
    if not torch.cuda.is_available():
        pytest.skip("Requires a GPU; CPU metadata tests do not qualify GPU indexing")
    pytest.importorskip("triton")
    spec = importlib.util.spec_from_file_location("_matmul_gpu_layout", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    device = torch.device("cuda", torch.cuda.current_device())
    a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float16, device=device)
    b = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float16, device=device)
    if large_operand is not None:
        original = a if large_operand == "a" else b
        # About 4 GiB of address space, but only six logical elements are used.
        storage = torch.empty(2**31 + original.shape[1], dtype=original.dtype, device=device)
        view = storage.as_strided(original.shape, (2**31 - 1, 1))
        view.copy_(original)
        if large_operand == "a":
            a = view
        else:
            b = view
    bias = torch.tensor([1, 99, 2, 99, 3, 99], dtype=a.dtype, device=device)[::2]
    expected = (a.clone().float() @ b.clone().float() + bias.float()).to(a.dtype)
    actual = module.matmul_persistent(a, b, bias)
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)
