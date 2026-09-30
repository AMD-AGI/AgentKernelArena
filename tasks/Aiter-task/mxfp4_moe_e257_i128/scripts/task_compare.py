# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
"""Correctness comparison for this workload, copied from the schema bundle.

Everything below this docstring is the definition's ``compare`` callback, taken
without edit from schema v2 so that this task and the acceptance run that
verifies its result apply one implementation rather than two that agree today.
Do not edit it here: change it in the bundle and copy it down again, or the two
silently diverge -- which is exactly the failure this file exists to prevent.

``run`` is the entry point the bundle exports.
"""

from __future__ import annotations
from abc import ABC as ABC
from abc import abstractmethod as abstractmethod
from dataclasses import dataclass as dataclass
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


def _check_threshold(name, value, *, lower=0, upper=None):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or (lower is not None and value < lower)
        or (upper is not None and value > upper)
    ):
        raise ValueError(f"invalid {name}: {value!r}")


def compute_error(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Return SQNR in dB using x as the reference and the input dtype."""
    signal = torch.linalg.vector_norm(x)
    noise = torch.linalg.vector_norm(x - y)
    return 20 * torch.log10(signal / noise)


@dataclass(frozen=True)
class SQNRCompare(Comparison):
    """Require SQNR >= min_sqnr in dB."""

    min_sqnr: float

    def __post_init__(self):
        _check_threshold("min_sqnr", self.min_sqnr, lower=None)

    def measure(self, actual, expected):
        sqnr = compute_error(expected, actual).item()
        # Exact zero/empty pairs produce 0/0 but still satisfy the comparison.
        if math.isnan(sqnr) and torch.equal(actual, expected):
            sqnr = math.inf
        passed = sqnr >= self.min_sqnr
        return ComparisonResult(
            passed,
            {"sqnr_db": sqnr},
            None if passed else f"sqnr_db={sqnr:.6g} < {self.min_sqnr}",
        )


def compare_fused_moe_outputs(actual, expected):
    """Assert output-dtype SQNR is at least 13 dB against the reference."""
    return SQNRCompare(min_sqnr=13)(actual, expected)


_callable = compare_fused_moe_outputs


def run(*args, **kwargs):
    return _callable(*args, **kwargs)
