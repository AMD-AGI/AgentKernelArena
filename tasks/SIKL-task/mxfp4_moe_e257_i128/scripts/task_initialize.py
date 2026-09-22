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


def _n_ones(n: int) -> int:
    return (1 << n) - 1


def _f32_to_floatx_unpacked(x: torch.Tensor, ebits: int, mbits: int) -> torch.Tensor:
    """Convert FP32 to saturated sub-byte codes with round-to-nearest-even."""

    # FP4 conversion and packing adapted from torchao 0.17.0.
    EBITS_F32, MBITS_F32 = 8, 23
    F32_EXP_BIAS = _n_ones(EBITS_F32 - 1)

    assert x.dtype == torch.float
    assert 1 + ebits + mbits <= 8

    exp_bias = _n_ones(ebits - 1)
    max_int = _n_ones(ebits + mbits)
    sign_mask = 1 << (ebits + mbits)

    magic_adder = _n_ones(MBITS_F32 - mbits - 1)

    max_normal = 2 ** (_n_ones(ebits) - exp_bias) * (_n_ones(mbits + 1) / (2**mbits))

    min_normal = 2 ** (1 - exp_bias)

    denorm_exp = (F32_EXP_BIAS - exp_bias) + (MBITS_F32 - mbits) + 1
    denorm_mask_int = denorm_exp << MBITS_F32

    denorm_mask_float = torch.tensor(denorm_mask_int, dtype=torch.int32).view(
        torch.float32
    )

    # CPU bit shifts require int32 rather than uint32.
    x = x.view(torch.int32)
    sign = x & 0x80000000

    x = x ^ sign

    x = x.view(torch.float)

    saturate_mask = x >= max_normal
    denormal_mask = torch.logical_and(torch.logical_not(saturate_mask), x < min_normal)
    normal_mask = torch.logical_not(torch.logical_or(saturate_mask, denormal_mask))

    # Adding the exponent offset rounds subnormals to nearest-even.
    denormal_x = x + denorm_mask_float
    denormal_x = denormal_x.view(torch.int32)
    denormal_x -= denorm_mask_int
    denormal_x = denormal_x.to(torch.uint8)

    # Adjust the exponent and round the retained mantissa to nearest-even.
    normal_x = x.view(torch.int32)
    mant_odd = (normal_x >> (MBITS_F32 - mbits)) & 1
    val_to_add = ((exp_bias - F32_EXP_BIAS) << MBITS_F32) + magic_adder
    normal_x += val_to_add
    normal_x += mant_odd
    normal_x = normal_x >> (MBITS_F32 - mbits)
    normal_x = normal_x.to(torch.uint8)

    x = torch.full_like(x, max_int, dtype=torch.uint8)
    x = torch.where(denormal_mask, denormal_x, x)
    x = torch.where(normal_mask, normal_x, x)

    sign_lp = sign >> (MBITS_F32 + EBITS_F32 - mbits - ebits)
    sign_lp = sign_lp.to(torch.uint8)
    # Discard sign extension from the signed right shift.
    sign_lp = sign_lp & sign_mask
    x = x | sign_lp

    return x.to(torch.uint8)


def _quantize_mxfp4(
    x: torch.Tensor, block: int = 32, min_amax: float = 0.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack nearest-even E2M1 values with upward-rounded E8M0 block scales."""
    values = x.to(torch.float32).unflatten(-1, (-1, block))
    amax = values.abs().amax(dim=-1, keepdim=True)
    is_finite = torch.isfinite(amax)
    values = torch.where(is_finite, values, 0.0)
    amax = torch.where(is_finite, amax, 0.0).clamp_min(min_amax)
    # frexp preserves subnormal maxima; 6 = 0.75 * 2**3.
    mantissa, exponent = torch.frexp(amax)
    exponent = exponent - 3 + (mantissa > 0.75).to(torch.int32)
    exponent = torch.where(amax == 0, -127, exponent).clamp(-127, 127)
    biased_exponent = exponent + 127
    scaled = values * torch.exp2(exponent.neg().to(torch.float32))
    codes = _f32_to_floatx_unpacked(scaled, ebits=2, mbits=1).flatten(-2)
    shape = codes.shape
    assert shape[-1] % 2 == 0
    codes = codes.contiguous().view(-1)
    packed = (codes[::2] | codes[1::2] << 4).view(*shape[:-1], shape[-1] // 2)
    scales = torch.where(is_finite, biased_exponent, 0xFF).to(torch.uint8).squeeze(-1)
    return packed, scales


def _init_mxfp4_weight(weight, scale, rng):
    """Quantize logical BF16 fan-in weights into packed buffers one expert at a time."""
    codes = weight.view(torch.uint8)
    experts, rows, packed_columns = codes.shape
    columns = 2 * packed_columns
    for expert in range(experts):
        plain = torch.empty((rows, columns), dtype=torch.bfloat16, device=weight.device)
        FanInNormal(fan_in=columns)(plain, generator=rng)
        packed, exponents = _quantize_mxfp4(plain)
        codes[expert].copy_(
            packed.view(rows // 16, 16, columns // 64, 2, 16)
            .permute(0, 2, 3, 1, 4)
            .reshape(rows, columns // 2)
        )
        scale[expert].copy_(
            exponents.view(rows // 32, 2, 16, columns // 256, 2, 4)
            .permute(0, 3, 5, 2, 4, 1)
            .reshape(rows, columns // 32)
        )


@dataclass(frozen=True)
class Mxfp4MoeWeightInit(Initializer):
    """The MoE ABI's paired E2M1/E8M0 encoding and expert-local shuffles."""

    def validate(self, weight, scale):
        if weight.ndim != 3 or weight.dtype != getattr(torch, "float4_e2m1fn_x2", None):
            raise ValueError("expected native packed FP4 expert weights")
        experts, rows, packed = weight.shape
        if rows <= 0 or packed <= 0 or rows % 256 or (2 * packed) % 256:
            raise ValueError(
                "MXFP4 MoE logical rows/columns must be positive multiples of 256"
            )
        if scale.shape != (experts, rows, packed // 16) or scale.dtype != torch.uint8:
            raise ValueError("expected one uint8 E8M0 scale per 32 logical weights")

    def initialize(self, weight, scale, *, generator):
        _init_mxfp4_weight(weight, scale, generator)


@dataclass(frozen=True)
class TopKInit(Initializer):
    """Generate distinct expert IDs and their weights from the same router."""

    num_experts: int
    renormalize: bool = False

    def validate(self, ids, weights=None):
        if type(self.num_experts) is not int or self.num_experts <= 0:
            raise ValueError("num_experts must be a positive integer")
        if type(self.renormalize) is not bool:
            raise ValueError("renormalize must be bool")
        if ids.ndim < 1 or ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("topk IDs must have an integer last axis")
        if not 0 < ids.shape[-1] <= self.num_experts:
            raise ValueError("expected 0 < topk <= num_experts")
        if self.num_experts - 1 > torch.iinfo(ids.dtype).max:
            raise ValueError("num_experts exceeds ID dtype range")
        if weights is not None:
            if weights.shape != ids.shape or weights.dtype not in (
                torch.float16,
                torch.bfloat16,
                torch.float32,
                torch.float64,
            ):
                raise ValueError(
                    "topk weights must be floating point and match ID shape"
                )

    def initialize(self, ids, weights=None, *, generator):
        logits = torch.randn(
            (*ids.shape[:-1], self.num_experts),
            dtype=torch.float32,
            device=ids.device,
            generator=generator,
        )
        values, indices = logits.softmax(dim=-1).topk(ids.shape[-1], dim=-1)
        ids.copy_(indices)
        if weights is not None:
            if self.renormalize:
                values = values / values.sum(dim=-1, keepdim=True)
            weights.copy_(values)


def initialize_fused_moe_inputs(inputs, *, seed=0):
    """Initialize canonical BF16/MXFP4 MoE buffers in place with a local seeded RNG."""
    names = (
        "hidden_states",
        "w1",
        "w2",
        "topk_weights",
        "topk_ids",
        "w1_scale",
        "w2_scale",
    )
    check_init_buffers(inputs, names, seed)
    if inputs.keys() != set(names) | {"activation", "doweight_stage1"}:
        raise ValueError("expected the complete canonical MoE inputs")
    activation, stage = inputs["activation"], inputs["doweight_stage1"]
    if type(activation) is not int or activation not in (0, 1):
        raise ValueError("activation must be Python int 0 (silu) or 1 (gelu)")
    if type(stage) is not bool:
        raise ValueError("doweight_stage1 must be a Python bool")
    hidden, w1, w2, ids = (
        inputs["hidden_states"],
        inputs["w1"],
        inputs["w2"],
        inputs["topk_ids"],
    )
    if hidden.ndim != 2 or w1.ndim != 3 or w2.ndim != 3 or ids.ndim != 2:
        raise ValueError("invalid hidden_states/weight/topk_ids ranks")
    tokens, dim = hidden.shape
    experts, inter_dim, topk = w1.shape[0], 2 * w2.shape[2], ids.shape[1]
    if dim == 0 or dim % 256 or inter_dim == 0 or inter_dim % 256:
        raise ValueError("model_dim/inter_dim must be positive multiples of 256")
    if not 0 < topk <= experts:
        raise ValueError("expected 0 < topk <= num_experts")
    fp4 = getattr(torch, "float4_e2m1fn_x2", None)
    if fp4 is None:
        raise RuntimeError(
            "native torch.float4_e2m1fn_x2 is required; no uint8 fallback"
        )
    shapes = {
        "hidden_states": ((tokens, dim), torch.bfloat16),
        "w1": ((experts, 2 * inter_dim, dim // 2), fp4),
        "w2": ((experts, dim, inter_dim // 2), fp4),
        "topk_weights": ((tokens, topk), torch.float32),
        "topk_ids": ((tokens, topk), torch.int32),
        "w1_scale": ((experts, 2 * inter_dim, dim // 32), torch.uint8),
        "w2_scale": ((experts, dim, inter_dim // 32), torch.uint8),
    }
    for name, (shape, dtype) in shapes.items():
        if inputs[name].shape != shape or inputs[name].dtype != dtype:
            raise ValueError(f"{name}: expected {dtype} buffer with shape {shape}")
    return InputInitializer(
        {
            "hidden_states": NormalInit(),
            ("topk_ids", "topk_weights"): TopKInit(experts),
            ("w1", "w1_scale"): Mxfp4MoeWeightInit(),
            ("w2", "w2_scale"): Mxfp4MoeWeightInit(),
        }
    )(inputs, seed=seed)


_callable = initialize_fused_moe_inputs


def run(*args, **kwargs):
    return _callable(*args, **kwargs)
