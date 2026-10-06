# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Input construction for this workload, copied from the schema bundle.

Everything below this docstring is the definition's ``initialize`` callback, taken
without edit from schema v2 so that this task and the acceptance run that
verifies its result apply one implementation rather than two that agree today.
Do not edit it here: change it in the bundle and copy it down again, or the two
silently diverge -- which is exactly the failure this file exists to prevent.

``run`` is the entry point the bundle exports.
"""

from __future__ import annotations
from abc import ABC as ABC
from abc import abstractmethod as abstractmethod
from collections.abc import Mapping as Mapping
from dataclasses import dataclass as dataclass
from functools import partial as partial
import math as math
import torch as torch

def check_init_buffers(inputs, tensor_names, seed=0) -> torch.Generator:
    """Validate metadata before writes; independent buffers must not overlap."""
    if type(inputs) is not dict:
        raise ValueError("inputs must be a dictionary of canonical input names")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    device = None
    ranges = []
    for name in tensor_names:
        tensor = inputs.get(name)
        if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
            raise ValueError(f"{name}: expected a dense strided Tensor")
        if not tensor.is_contiguous() or tensor.device.type not in ("cpu", "cuda"):
            raise ValueError(f"{name}: expected a contiguous CPU/CUDA tensor")
        if device is not None and device != tensor.device:
            raise ValueError("all input buffers must use the same device")
        device = tensor.device
        if tensor.numel():
            start = tensor.data_ptr()
            end = start + tensor.numel() * tensor.element_size()
            for other, low, high in ranges:
                if start < high and low < end:
                    raise ValueError(f"input buffers overlap: {other} and {name}")
            ranges.append((name, start, end))
    if device is None:
        raise ValueError("at least one input tensor is required")
    return torch.Generator(device=device).manual_seed(seed)


class Initializer(ABC):
    """Initialize one or more buffers, independently of their input names."""

    @abstractmethod
    def validate(self, *tensors: torch.Tensor) -> None:
        """Check parameters and buffer metadata without reading their contents."""

    @abstractmethod
    def initialize(self, *tensors: torch.Tensor, generator: torch.Generator) -> None:
        """Write validated buffers in place; use only the supplied RNG."""

    def __call__(self, *tensors, seed=None, generator=None):
        """Return the original tensor, or a tuple for multiple tensor arguments."""
        if generator is not None and seed is not None:
            raise ValueError("specify seed or generator, not both")
        buffers = {f"tensor_{i}": tensor for i, tensor in enumerate(tensors)}
        local = check_init_buffers(buffers, tuple(buffers), 0 if seed is None else seed)
        if generator is not None and generator.device != local.device:
            raise ValueError("generator and tensor must use the same device")
        self.validate(*tensors)
        self.initialize(*tensors, generator=local if generator is None else generator)
        return tensors[0] if len(tensors) == 1 else tensors


def _finite_parameter(name, value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number")


def _require_float(tensor):
    # Packed FP4 and E8M0 describe encodings, not ordinary floating-point data.
    supported = (torch.float16, torch.bfloat16, torch.float32, torch.float64)
    fp8 = tuple(
        getattr(torch, name)
        for name in (
            "float8_e4m3fn",
            "float8_e5m2",
            "float8_e4m3fnuz",
            "float8_e5m2fnuz",
        )
        if hasattr(torch, name)
    )
    if tensor.dtype not in supported + fp8:
        raise ValueError(
            f"{tensor.dtype} requires an explicit integer or encoding initializer"
        )


@dataclass(frozen=True)
class ConstantInit(Initializer):
    value: float = 0.0

    def validate(self, tensor):
        if not isinstance(self.value, bool):
            _finite_parameter("value", self.value)
        if tensor.dtype == torch.bool:
            if self.value not in (0, 1):
                raise ValueError("boolean constants must be 0 or 1")
        elif tensor.is_floating_point():
            _require_float(tensor)
            if abs(self.value) > torch.finfo(tensor.dtype).max:
                raise ValueError("constant exceeds dtype range")
        else:
            try:
                limits = torch.iinfo(tensor.dtype)
            except TypeError as error:
                raise ValueError(
                    "encoded tensors require a joint initializer"
                ) from error
            if (
                int(self.value) != self.value
                or not limits.min <= self.value <= limits.max
            ):
                raise ValueError("constant must be an integer in the dtype range")

    def initialize(self, tensor, *, generator):
        # copy_ also works for the FP8 dtypes without a fill_ kernel.
        tensor.copy_(
            torch.full(
                tensor.shape,
                self.value,
                dtype=torch.float64 if tensor.is_floating_point() else tensor.dtype,
                device=tensor.device,
            )
        )


@dataclass(frozen=True, init=False, repr=False)
class InputInitializer:
    """Bind tensor names to strategies and validate all buffers before in-place writes."""

    bindings: tuple[tuple[tuple[str, ...], Initializer], ...]

    def __init__(
        self,
        initializers: Mapping[str | tuple[str, ...], Initializer] | None = None,
        /,
        **tensors: Initializer,
    ):
        if initializers is None:
            initializers = {}
        if not isinstance(initializers, Mapping):
            raise TypeError(
                "initializers must be a mapping of input names to strategies"
            )
        bindings, seen = [], set()
        for target, initializer in (*initializers.items(), *tensors.items()):
            names = (target,) if isinstance(target, str) else target
            if (
                not isinstance(names, tuple)
                or not names
                or not all(isinstance(name, str) and name for name in names)
            ):
                raise ValueError("targets must be a nonempty name or tuple of names")
            if not isinstance(initializer, Initializer):
                raise TypeError("strategies must be Initializer instances")
            for name in names:
                if name in seen:
                    raise ValueError(
                        f"an input buffer cannot be initialized twice: {name}"
                    )
                seen.add(name)
            bindings.append((names, initializer))
        object.__setattr__(self, "bindings", tuple(bindings))

    def __repr__(self):
        # Keep the positional mapping so callback export reserves no input names.
        return f"{type(self).__name__}({dict(self.bindings)!r})"

    @property
    def tensor_names(self):
        return tuple(name for names, _ in self.bindings for name in names)

    def __call__(self, inputs, *, seed=0):
        rng = check_init_buffers(inputs, self.tensor_names, seed)
        self.validate(inputs)
        self.initialize(inputs, generator=rng)
        return inputs

    def validate(self, inputs):
        actual_names = {
            name for name, value in inputs.items() if isinstance(value, torch.Tensor)
        }
        if actual_names != set(self.tensor_names):
            raise ValueError(
                f"expected tensor buffers {self.tensor_names}, got {sorted(actual_names)}"
            )
        for names, initializer in self.bindings:
            initializer.validate(*(inputs[name] for name in names))

    def initialize(self, inputs, *, generator):
        for names, initializer in self.bindings:
            initializer.initialize(
                *(inputs[name] for name in names), generator=generator
            )


@dataclass(frozen=True)
class NormalInit(Initializer):
    """Normal activations, sampled natively where torch supports the dtype."""

    mean: float = 0.0
    std: float = 1.0

    def validate(self, tensor):
        _require_float(tensor)
        _finite_parameter("mean", self.mean)
        _finite_parameter("std", self.std)
        if self.std < 0:
            raise ValueError("std must be nonnegative")
        limit = torch.finfo(tensor.dtype).max
        if abs(self.mean) > limit or self.std > limit:
            raise ValueError("normal parameters exceed the target dtype range")

    def initialize(self, tensor, *, generator):
        native = tensor.dtype in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        )
        work = tensor if native else torch.empty_like(tensor, dtype=torch.float32)
        work.normal_(mean=self.mean, std=self.std, generator=generator)
        limit = torch.finfo(tensor.dtype).max
        work.nan_to_num_(nan=0.0, posinf=limit, neginf=-limit).clamp_(-limit, limit)
        if not native:
            tensor.copy_(work)


@dataclass(frozen=True)
class UniformInit(Initializer):
    low: float = 0.0
    high: float = 1.0

    def validate(self, tensor):
        _require_float(tensor)
        _finite_parameter("low", self.low)
        _finite_parameter("high", self.high)
        limit = torch.finfo(tensor.dtype).max
        if not -limit <= self.low < self.high <= limit or self.high - self.low > limit:
            raise ValueError("uniform bounds must be ordered and representable")

    def initialize(self, tensor, *, generator):
        native = tensor.dtype in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        )
        work = tensor if native else torch.empty_like(tensor, dtype=torch.float32)
        work.uniform_(self.low, self.high, generator=generator)
        if not native:
            tensor.copy_(work)


_FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2,)


def _valid_blockwise_scaled_shapes(
    a, b, a_scale, b_scale, bias, *, block_size, scale_major_mode, trans_b
):
    """Metadata-only shape contract, also used by the exported initializer."""
    if (
        type(trans_b) is not bool
        or not isinstance(block_size, (tuple, list))
        or len(block_size) != 2
        or any(type(v) is not int or v <= 0 for v in block_size)
        or type(scale_major_mode) is not str
        or scale_major_mode not in ("K", "MN")
    ):
        return False
    if any(
        not isinstance(t, torch.Tensor) or t.ndim != 2 for t in (a, b, a_scale, b_scale)
    ):
        return False
    m, k = a.shape
    n = b.shape[0 if trans_b else 1]
    if b.shape[1 if trans_b else 0] != k:
        return False
    bn, bk = block_size
    sn, sk = (n + bn - 1) // bn, (k + bk - 1) // bk
    return (
        a_scale.shape == ((m, sk) if scale_major_mode == "K" else (sk, m))
        and b_scale.shape == (sn, sk)
        and (bias is None or (isinstance(bias, torch.Tensor) and bias.shape == (n,)))
    )


@torch.no_grad()
def initialize_blockwise_scaled_inputs(
    inputs,
    *,
    seed=0,
    block_size=(128, 128),
    scale_major_mode="K",
    trans_b=True,
    a_scale_storage="plain",
):
    """Fill logical values or encoded payloads without replacing buffers/strides.

    Raw and logical preshuffle fixtures share the same logical random matrices
    and nonuniform positive scales. Blob replay bypasses this initializer.
    """
    required = {"a", "b", "a_scale", "b_scale"}
    if type(inputs) is not dict or inputs.keys() not in (
        required,
        required | {"bias"},
    ):
        raise ValueError("expected a, b, a_scale, b_scale and optionally bias")
    check_init_buffers(inputs, tuple(inputs), seed)
    a, b = inputs["a"], inputs["b"]
    a_scale, b_scale = inputs["a_scale"], inputs["b_scale"]
    if not _valid_blockwise_scaled_shapes(
        a,
        b,
        a_scale,
        b_scale,
        inputs.get("bias"),
        block_size=block_size,
        scale_major_mode=scale_major_mode,
        trans_b=trans_b,
    ):
        raise ValueError(
            "invalid blockwise-scaled matrix/scale/bias shapes or tile sizes"
        )
    if (
        a.dtype not in _FP8_DTYPES
        or b.dtype not in _FP8_DTYPES
        or a_scale.dtype != torch.float32
        or b_scale.dtype != torch.float32
        or a_scale_storage not in ("plain", "raw", "logical")
    ):
        raise ValueError("expected FP8 operands, FP32 scales and a known storage mode")
    if a_scale_storage != "plain" and (
        not trans_b
        or tuple(block_size) != (128, 128)
        or scale_major_mode != "K"
        or "bias" in inputs
        or a.dtype != torch.float8_e4m3fn
        or b.dtype != a.dtype
        or min(a.shape[0], b.shape[0], a.shape[1]) <= 0
        or b.shape[0] % 16
        or a.shape[1] % 128
    ):
        raise ValueError("invalid AITER 16x16 preshuffle contract")
    strategies: dict[str, Initializer] = {
        "a": NormalInit(),
        "b": NormalInit(),
        "a_scale": UniformInit(0.125, 1.0),
        "b_scale": UniformInit(0.125, 1.0),
    }
    if "bias" in inputs:
        strategies["bias"] = ConstantInit(1)
    InputInitializer(**strategies)(inputs, seed=seed)
    if a_scale_storage != "plain":
        n, k = b.shape
        # Forward AITER shuffle_weight(layout=(16,16)), one-byte FP8 operands.
        b.copy_(
            b.reshape(n // 16, 16, k // 32, 2, 16)
            .permute(0, 2, 3, 1, 4)
            .contiguous()
            .reshape(n, k)
        )
        if a_scale_storage == "raw":
            a_scale.copy_(a_scale.T.contiguous().reshape(a_scale.shape))
    return inputs


_callable = partial(initialize_blockwise_scaled_inputs, **{'trans_b': True, 'block_size': (128, 128,), 'scale_major_mode': 'K', 'a_scale_storage': 'logical'})


def run(*args, **kwargs):
    return _callable(*args, **kwargs)
