from __future__ import annotations
from abc import ABC as ABC
from abc import abstractmethod as abstractmethod
from dataclasses import dataclass as dataclass
from dataclasses import field as field
from functools import partial as partial
import math as math
import torch as torch

@dataclass(frozen=True)
class ComparisonResult:
    passed: bool
    metrics: dict[str, float]
    message: str | None = None


def _exact_result(actual, expected):
    if expected.dtype == getattr(torch, "float4_e2m1fn_x2", None):
        actual, expected = actual.view(torch.uint8), expected.view(torch.uint8)
    bad = int((actual != expected).sum().item())
    ratio = 1 - bad / actual.numel() if actual.numel() else 1.0
    return ComparisonResult(
        bad == 0,
        {"matched_ratio": ratio},
        f"{bad}/{actual.numel()} elements differ" if bad else None,
    )


def validate_comparison(actual, expected, *, allow_packed=False):
    """Invalid references are ValueError; candidate contract failures are AssertionError."""
    if (
        not isinstance(expected, torch.Tensor)
        or expected.layout != torch.strided
        or expected.device.type not in ("cpu", "cuda")
        or expected.is_complex()
        or expected.is_quantized
    ):
        raise ValueError("invalid reference: expected a dense real CPU/CUDA tensor")
    packed = expected.dtype == getattr(torch, "float4_e2m1fn_x2", None)
    if packed and not allow_packed:
        raise ValueError(
            "packed FP4 outputs must be decoded before numerical comparison"
        )
    if (
        expected.is_floating_point()
        and not packed
        and not torch.isfinite(expected.to(torch.float64)).all().item()
    ):
        raise ValueError("invalid reference: non-finite output")
    if not isinstance(actual, torch.Tensor) or actual.layout != torch.strided:
        raise AssertionError("candidate must return one dense Tensor")
    if actual.shape != expected.shape:
        raise AssertionError(f"shape {actual.shape}, expected {expected.shape}")
    if actual.dtype != expected.dtype:
        raise AssertionError(f"dtype {actual.dtype}, expected {expected.dtype}")
    if actual.device != expected.device:
        raise AssertionError(f"device {actual.device}, expected {expected.device}")
    if (
        actual.is_floating_point()
        and not packed
        and not torch.isfinite(actual.to(torch.float64)).all().item()
    ):
        raise AssertionError("candidate contains NaN or Inf")


class Comparison(ABC):
    """Callable assertion plus a non-raising numerical report for valid tensor pairs."""

    allow_packed = False

    @torch.no_grad()
    def evaluate(self, actual, expected):
        validate_comparison(actual, expected, allow_packed=self.allow_packed)
        with torch.autocast(expected.device.type, enabled=False):
            if not expected.is_floating_point():
                return _exact_result(actual, expected)
            return self.measure(actual, expected)

    def __call__(self, actual, expected):
        result = self.evaluate(actual, expected)
        if not result.passed:
            raise AssertionError(
                result.message or f"comparison failed: {result.metrics}"
            )

    @abstractmethod
    def measure(self, actual, expected) -> ComparisonResult:
        """Measure tensors already validated by evaluate."""


@dataclass(frozen=True)
class Tolerances:
    atol: float
    rtol: float
    min_cosine_similarity: float


class TolerancePolicy(ABC):
    @abstractmethod
    def resolve(self, compute_dtype) -> Tolerances:
        """Resolve a mathematical compute format, which may differ from output dtype."""


@dataclass(frozen=True)
class DTypeTolerancePolicy(TolerancePolicy):
    """Legacy tolerances for elementwise and cosine comparisons."""

    def resolve(self, compute_dtype):
        name = str(getattr(compute_dtype, "value", compute_dtype)).removeprefix(
            "torch."
        )
        aliases = {
            "fp64": "float64",
            "fp32": "float32",
            "fp16": "float16",
            "bf16": "bfloat16",
            "mxfp4": "fp4",
            "nvfp4": "fp4",
            "float4_e2m1": "fp4",
            "float4_e2m1fn_x2": "fp4",
        }
        name = aliases.get(name, name)
        if name in (
            "float8_e4m3fn",
            "float8_e5m2",
            "float8_e4m3fnuz",
            "float8_e5m2fnuz",
        ):
            name = "fp8"
        table = {
            "float64": (1e-12, 1e-12, 1 - 1e-12),
            "float32": (1e-5, 1e-5, 1 - 1e-6),
            "float16": (1e-3, 1e-3, 0.9999),
            "bfloat16": (1e-2, 1e-2, 0.999),
            "fp8": (0.125, 0.125, 0.99),
            "fp4": (0.25, 0.25, 0.98),
        }
        if name not in table:
            raise ValueError(
                f"no tolerance profile for compute dtype {compute_dtype!r}"
            )
        return Tolerances(*table[name])


def _check_threshold(name, value, *, lower=0, upper=None):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or (lower is not None and value < lower)
        or (upper is not None and value > upper)
    ):
        raise ValueError(f"invalid {name}: {value!r}")


@dataclass(frozen=True)
class DefaultCompare(Comparison):
    atol: float | None = None
    rtol: float | None = None
    compute_dtype: object = None
    policy: TolerancePolicy = field(default_factory=DTypeTolerancePolicy)
    mode: str = "either"
    required_matched_ratio: float = 1.0

    def __post_init__(self):
        if self.mode not in ("either", "additive"):
            raise ValueError("mode must be 'either' or 'additive'")
        _check_threshold("required_matched_ratio", self.required_matched_ratio, upper=1)
        for name in ("atol", "rtol"):
            if getattr(self, name) is not None:
                _check_threshold(name, getattr(self, name))

    def measure(self, actual, expected):
        atol, rtol = self.atol, self.rtol
        if atol is None or rtol is None:
            defaults = self.policy.resolve(
                expected.dtype if self.compute_dtype is None else self.compute_dtype
            )
            atol = defaults.atol if atol is None else atol
            rtol = defaults.rtol if rtol is None else rtol
        _check_threshold("atol", atol)
        _check_threshold("rtol", rtol)
        if not expected.numel():
            return ComparisonResult(
                True,
                {
                    "max_absolute_error": 0.0,
                    "max_relative_error": 0.0,
                    "matched_ratio": 1.0,
                },
            )
        dtype = torch.float64 if expected.dtype == torch.float64 else torch.float32
        x, y = actual.to(dtype), expected.to(dtype)
        absolute = (x - y).abs()
        relative = absolute / (y.abs() + 1e-8)
        matches = (
            ((absolute <= atol) | (relative <= rtol))
            if self.mode == "either"
            else (absolute <= atol + rtol * y.abs())
        )
        matched = int(matches.sum().item())
        ratio = matched / matches.numel()
        metrics = {
            "max_absolute_error": absolute.max().item(),
            "max_relative_error": relative.max().item(),
            "matched_ratio": ratio,
        }
        passed = ratio >= self.required_matched_ratio
        return ComparisonResult(
            passed,
            metrics,
            (
                None
                if passed
                else f"{matches.numel() - matched}/{matches.numel()} elements fail {self.mode} tolerance (atol={atol}, rtol={rtol}); {metrics}"
            ),
        )


def compare_mhc_outputs(actual, expected, *, op):
    """Use AITER mHC's 1e-2 allclose tolerances, requiring every element to pass.

    Benchmark callbacks receive one Tensor for post, or a named output dict
    for pre/fused post→pre. FP32 mix storage does not imply FP32 input precision.
    Invalid references raise ValueError; invalid candidates raise AssertionError.
    """
    if op == "mhc_post":
        dtypes = {"out": torch.bfloat16}
        actual, expected = {"out": actual}, {"out": expected}
    elif op == "mhc_pre":
        dtypes = {
            "post_mix": torch.float32,
            "comb_mix": torch.float32,
            "layer_input": torch.bfloat16,
        }
    elif op == "mhc_fused_post_pre":
        dtypes = {
            "next_post_mix": torch.float32,
            "next_comb_mix": torch.float32,
            "layer_input": torch.bfloat16,
            "next_residual": torch.bfloat16,
        }
    else:
        raise ValueError(f"unknown mHC operation: {op}")
    if not isinstance(expected, dict) or expected.keys() != dtypes.keys():
        raise ValueError("invalid mHC reference: output names differ")
    if not isinstance(actual, dict) or actual.keys() != dtypes.keys():
        raise AssertionError("mHC candidate output names differ")
    for name, dtype in dtypes.items():
        reference = expected[name]
        if not isinstance(reference, torch.Tensor) or reference.dtype != dtype:
            raise ValueError(f"invalid mHC reference {name}: expected {dtype} Tensor")
        try:
            DefaultCompare(atol=1e-2, rtol=1e-2, mode="additive")(
                actual[name], reference
            )
        except (AssertionError, ValueError) as error:
            raise type(error)(f"{name}: {error}") from error


_callable = partial(compare_mhc_outputs, **{'op': 'mhc_fused_post_pre'})


def run(*args, **kwargs):
    return _callable(*args, **kwargs)
