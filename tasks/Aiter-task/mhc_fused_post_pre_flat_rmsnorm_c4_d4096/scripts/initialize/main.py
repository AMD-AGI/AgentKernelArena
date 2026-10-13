from __future__ import annotations
from abc import ABC as ABC
from abc import abstractmethod as abstractmethod
from collections.abc import Mapping as Mapping
from dataclasses import dataclass as dataclass
from functools import partial as partial
from typing import Any as Any
from typing import TypeGuard as TypeGuard
import math as math
import struct as struct
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
class FanInNormal(Initializer):
    """Projection weights with std=gain/sqrt(logical fan-in), never packed bytes."""

    fan_in: int | None = None
    dim: int = -1
    gain: float = 1.0

    def _normal(self, tensor):
        fan_in = self.fan_in
        if fan_in is None:
            if type(self.dim) is not int or not -tensor.ndim <= self.dim < tensor.ndim:
                raise ValueError("dim must select a logical reduction axis")
            fan_in = tensor.shape[self.dim]
        if type(fan_in) is not int or fan_in <= 0:
            raise ValueError("fan_in must be a positive integer")
        _finite_parameter("gain", self.gain)
        if self.gain < 0:
            raise ValueError("gain must be nonnegative")
        return NormalInit(std=self.gain / math.sqrt(fan_in))

    def validate(self, tensor):
        self._normal(tensor).validate(tensor)

    def initialize(self, tensor, *, generator):
        self._normal(tensor).initialize(tensor, generator=generator)


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


_PRE_SCALARS = {'rms_eps': 1e-06, 'pre_eps': 1e-06, 'sinkhorn_eps': 1e-06, 'post_multiplier': 1.0, 'sinkhorn_iters': 20}


def _valid_float32_scalar(value: Any, *, positive: bool) -> bool:
    """Require finite FP32 conversion, and a positive result when requested.

    Only Python int/float values are accepted. With ``positive=True``, values
    that round to zero are rejected; otherwise finite zero is allowed.
    """
    if type(value) not in (int, float):
        return False
    try:
        converted = struct.unpack("f", struct.pack("f", value))[0]
    except (OverflowError, struct.error):
        return False
    return math.isfinite(converted) and (not positive or converted > 0)


def _valid_pre_scalars(kwargs: dict[str, Any]) -> bool:
    for scalar, default in _PRE_SCALARS.items():
        value = kwargs.get(scalar, default)
        if scalar == "sinkhorn_iters":
            if type(value) is not int or not 1 <= value < 2**31:
                return False
        elif not _valid_float32_scalar(value, positive=scalar != "post_multiplier"):
            return False
    return True


def _valid_tensor(
    tensor: Any, like: torch.Tensor | None = None
) -> TypeGuard[torch.Tensor]:
    """Dense contiguous storage on the expected device, without reading contents."""
    return (
        isinstance(tensor, torch.Tensor)
        and tensor.layout == torch.strided
        and tensor.is_contiguous()
        and not tensor.is_conj()
        and not tensor.is_neg()
        and (like is None or tensor.device == like.device)
    )


def _valid_residual(residual: Any) -> TypeGuard[torch.Tensor]:
    return (
        _valid_tensor(residual)
        and residual.dtype == torch.bfloat16
        and residual.ndim == 3
        and residual.shape[0] > 0
        and residual.shape[1] == 4
        and residual.shape[2] > 0
    )


def initialize_mhc_inputs(inputs, *, seed=0, op, flat_post=False, with_norm=False):
    """Fill canonical buffers in place with non-saturating, seeded mHC inputs.

    Projection weights use fan-in scaling. Incoming post gates are bounded and
    combination matrices are positive and approximately doubly stochastic.
    Scalar parameters are validated but never changed.
    """
    if op not in ("mhc_pre", "mhc_post", "mhc_fused_post_pre"):
        raise ValueError(f"unknown mHC operation: {op}")
    pre = op != "mhc_post"
    post = op != "mhc_pre"
    if (flat_post and not post) or (with_norm and op != "mhc_fused_post_pre"):
        raise ValueError("unsupported mHC initialization variant")

    names = ["residual"]
    if post:
        names += ["x", "post_mix", "comb_mix"]
    if pre:
        names += ["proj_weight", "mix_scale", "mix_bias"]
    if with_norm:
        names += ["norm_weight"]
    check_init_buffers(inputs, names, seed)
    scalars = set(_PRE_SCALARS) if pre else set()
    if with_norm:
        scalars.add("norm_eps")
    if set(inputs) - set(names) - scalars:
        raise ValueError("unexpected mHC input names")
    residual = inputs["residual"]
    if not _valid_residual(residual):
        raise ValueError("residual must be a nonempty HC=4 BF16 tensor")
    tokens, streams, hidden = residual.shape
    specs = {}
    if post:
        specs.update(
            x=((tokens, hidden), torch.bfloat16),
            post_mix=(
                (tokens, streams) if flat_post else (tokens, streams, 1),
                torch.float32,
            ),
            comb_mix=((tokens, streams, streams), torch.float32),
        )
    if pre:
        specs.update(
            proj_weight=((24, streams * hidden), torch.float32),
            mix_scale=((3,), torch.float32),
            mix_bias=((24,), torch.float32),
        )
        if not _valid_pre_scalars(inputs):
            raise ValueError("invalid mHC pre scalar parameters")
    if with_norm:
        specs["norm_weight"] = ((hidden,), torch.bfloat16)
        if not _valid_float32_scalar(inputs.get("norm_eps", 1e-6), positive=True):
            raise ValueError("norm_eps must be positive and finite in FP32")
    for name, (shape, dtype) in specs.items():
        tensor = inputs[name]
        if (
            not _valid_tensor(tensor, residual)
            or tensor.shape != shape
            or tensor.dtype != dtype
        ):
            raise ValueError(f"{name}: expected contiguous {dtype} buffer {shape}")

    strategies: dict[str, Initializer] = {"residual": NormalInit()}
    if post:
        strategies.update(
            x=NormalInit(),
            post_mix=UniformInit(0.1, 0.9),
            comb_mix=NormalInit(),
        )
    if pre:
        strategies.update(
            proj_weight=FanInNormal(),
            mix_scale=ConstantInit(1),
            mix_bias=NormalInit(std=0.1),
        )
    if with_norm:
        strategies["norm_weight"] = UniformInit(0.5, 1.5)
    InputInitializer(**strategies)(inputs, seed=seed)
    if post:
        comb = inputs["comb_mix"]
        comb.copy_(comb.softmax(dim=-1))
        for _ in range(20):
            comb.div_(comb.sum(dim=-1, keepdim=True))
            comb.div_(comb.sum(dim=-2, keepdim=True))
    return inputs


_callable = partial(initialize_mhc_inputs, **{'op': 'mhc_fused_post_pre', 'flat_post': True, 'with_norm': True})


def run(*args, **kwargs):
    return _callable(*args, **kwargs)
