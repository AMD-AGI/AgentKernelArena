# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Error bounds adapted from AKLR (aklr/core/contract/semantics/_compare.py, MIT)
# for AgentKernelArena's self-contained task evaluator.
"""Compare one written tensor with the high-precision reference.

``baseline_relative`` (the default) allows each element twice the baseline
kernel's own error, or its per-row RMS, plus one output rounding; the mean error
must also stay unbiased relative to the baseline's. Integral, boolean and packed
outputs compare exactly. Non-finite reference elements must be matched exactly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .values import UNIT_ROUNDOFF, tensor_bytes, is_exact_dtype

OUTPUT_ROUNDING_MULTIPLE = 2.0
MIN_ROW_ELEMENTS = 8
VIOLATION_PEAK_LIMIT = 16.0
DEFAULT_MAX_ERROR_MULTIPLE = 2.0


@dataclass(frozen=True)
class CompareResult:
    passed: bool
    bound: str
    violation_fraction: float = 0.0
    peak_bound_multiple: float = 0.0
    bias_z_score: float = 0.0
    nonfinite_mismatches: int = 0
    mismatched_elements: int = 0
    sqnr_db: float | None = None

    def to_mapping(self) -> dict:
        value = {"passed": self.passed, "bound": self.bound,
                 "violation_fraction": self.violation_fraction,
                 "peak_bound_multiple": _finite(self.peak_bound_multiple),
                 "bias_z_score": _finite(self.bias_z_score),
                 "nonfinite_mismatches": self.nonfinite_mismatches,
                 "mismatched_elements": self.mismatched_elements}
        if self.sqnr_db is not None:
            value["sqnr_db"] = _finite(self.sqnr_db)
        return value


def _finite(value: float) -> float:
    return value if math.isfinite(value) else 1e300


def arrange(order: str, actual: torch.Tensor, reference: torch.Tensor):
    if order == "fixed":
        return actual, reference
    if order == "any_in_full_rows":
        is_full = (reference >= 0).all(dim=-1, keepdim=True)
        return (torch.where(is_full, actual.sort(dim=-1).values, actual),
                torch.where(is_full, reference.sort(dim=-1).values, reference))
    raise ValueError(f"Unknown element order {order!r}")


def _exact(actual: torch.Tensor, reference: torch.Tensor) -> CompareResult:
    expected = reference.to(actual.dtype)
    if expected.shape != actual.shape:
        return CompareResult(False, "exact", 1.0, mismatched_elements=actual.numel())
    if actual.dtype.is_floating_point:
        a, b = actual.double(), expected.double()
        same = (a == b) | (a.isnan() & b.isnan())
        mismatched = int((~same).sum())
    else:
        width = actual.element_size() if actual.numel() else 1
        left = tensor_bytes(actual).reshape(actual.numel(), -1) if actual.numel() else tensor_bytes(actual)
        right = tensor_bytes(expected).reshape(expected.numel(), -1) if expected.numel() else tensor_bytes(expected)
        mismatched = int((left != right).any(-1).sum()) if width else 0
    fraction = mismatched / actual.numel() if actual.numel() else 0.0
    return CompareResult(mismatched == 0, "exact", fraction, mismatched_elements=mismatched)


def _nonfinite(a: torch.Tensor, rounded: torch.Tensor) -> int:
    finite = rounded.isfinite()
    same = (a == rounded) | (a.isnan() & rounded.isnan())
    return int((~finite & ~same).sum()) + int((finite & ~a.isfinite()).sum())


def _bias(errors: torch.Tensor, compared: torch.Tensor) -> tuple[bool, float]:
    error = errors[compared]
    n = error.numel()
    rms = error.square().mean().sqrt().item() if n else 0.0
    z = abs(error.mean().item()) / rms if rms > 0 else 0.0
    return n == 0 or z <= 5 / math.sqrt(n), z


def _elementwise(name: str, a, rounded, reference, limit, max_violation_fraction,
                 bias_check) -> CompareResult:
    finite = rounded.isfinite()
    errors = a - reference
    error = errors[finite]
    bound = limit[finite]
    fraction = peak = 0.0
    mismatched = 0
    if error.numel():
        exceeded = error.abs() > bound
        mismatched = int(exceeded.sum())
        fraction = mismatched / error.numel()
        multiples = (error.abs() / bound).masked_fill(error == 0, 0.0)
        peak = multiples.max().item()
        if math.isnan(peak):
            peak = math.inf
    unbiased, z = bias_check(errors, finite, fraction)
    nonfinite = _nonfinite(a, rounded)
    # max_violation_fraction admits rare elements over the bound (accumulation-order outliers), never an
    # arbitrary error: a skipped tail or a zeroed tile lands far beyond VIOLATION_PEAK_LIMIT bounds.
    within_peak = fraction == 0 or peak <= VIOLATION_PEAK_LIMIT
    passed = nonfinite == 0 and fraction <= max_violation_fraction and unbiased and within_peak
    return CompareResult(passed, name, fraction, peak, z, nonfinite, mismatched)


def compare_tensor(declaration: dict, actual: torch.Tensor, reference: torch.Tensor, *,
                   dtype_name: str, baseline: torch.Tensor | None = None,
                   element_order: str = "fixed") -> CompareResult:
    """``actual`` against the reference under the declared bound."""
    if tuple(actual.shape) != tuple(reference.shape):
        return CompareResult(False, declaration.get("bound", "exact"), 1.0,
                             mismatched_elements=actual.numel())
    actual, reference = arrange(element_order, actual, reference)
    if baseline is not None:
        baseline, _ = arrange(element_order, baseline, reference)
    kind = declaration.get("bound", "baseline_relative")
    if kind == "exact" or is_exact_dtype(dtype_name):
        return _exact(actual, reference)
    u = UNIT_ROUNDOFF[dtype_name]
    a = actual.double()
    y = reference.double()
    rounded = reference.to(actual.dtype).double()
    max_violation = float(declaration.get("max_violation_fraction", 0.0))
    if kind == "sqnr":
        kept = rounded.isfinite() & a.isfinite()
        noise = (a - rounded)[kept].norm().item()
        signal = rounded[kept].norm().item()
        allowed = signal * 10 ** (-float(declaration["min_db"]) / 20)
        peak = 0.0 if not noise else (noise / allowed if allowed else math.inf)
        nonfinite = _nonfinite(a, rounded)
        sqnr = math.inf if not noise else (20 * math.log10(signal / noise) if signal else -math.inf)
        return CompareResult(nonfinite == 0 and peak <= 1, "sqnr", float(peak > 1), peak, 0.0,
                             nonfinite, sqnr_db=sqnr)
    if kind == "output_rounding":
        limit = OUTPUT_ROUNDING_MULTIPLE * u * y.abs()
        return _elementwise(kind, a, rounded, y, limit, max_violation, lambda e, c, f: _bias(e, c))
    if kind == "tolerance":
        limit = float(declaration["atol"]) + (float(declaration["rtol"]) + u) * y.abs()
        return _elementwise(kind, a, rounded, y, limit, max_violation,
                            lambda e, c, f: (_bias(e, c)[0] or f == 0, _bias(e, c)[1]))
    if kind == "baseline_relative":
        if baseline is None:
            raise ValueError("baseline_relative bound needs the baseline output")
        k = float(declaration.get("k", DEFAULT_MAX_ERROR_MULTIPLE))
        b = baseline.double()
        baseline_error = b - y
        compared = y.isfinite() & b.isfinite()
        squares = torch.where(compared, baseline_error.square(), torch.zeros_like(baseline_error))
        # Elements where the reference and the baseline are both exactly zero (the structural zeros of a
        # triangular or sparse output) carry no error sample, so they do not dilute the RMS.
        sampled = compared & ((y != 0) | (b != 0))
        count = sampled.sum(-1) if y.dim() else sampled.reshape(1).sum(-1)
        rms = (squares.sum(-1) / count).sqrt() if y.dim() else (squares / count).sqrt()
        if y.dim():
            # A row of a few elements gives a one- or few-sample error estimate, so a correct candidate
            # fails wherever the baseline happened to be exact. Such rows use the whole output's RMS.
            short = count < MIN_ROW_ELEMENTS
            if bool(short.any()):
                total = sampled.sum()
                pooled = (squares.sum() / total).sqrt() if bool(total) else torch.full_like(rms, math.nan)
                rms = torch.where(short, pooled, rms)
        rms_b = rms[..., None] if y.dim() else rms
        relative = k * torch.maximum(baseline_error.abs(), rms_b)
        rounding = u * y.abs()
        is_relative = compared & (rms_b.isfinite() if y.dim() else rms.isfinite())
        limit = torch.where(is_relative, relative + rounding, OUTPUT_ROUNDING_MULTIPLE * rounding)

        def bias_check(errors, finite, fraction):
            mask = finite & compared
            unbiased, z = _bias(errors, mask)
            if not unbiased:
                error = errors[mask]
                error_rms = error.square().mean().sqrt().item()
                baseline_bias = abs(baseline_error[mask].mean().item())
                unbiased = abs(error.mean().item()) <= k * baseline_bias + 5 * error_rms / math.sqrt(error.numel())
            return unbiased, z

        return _elementwise(kind, a, rounded, y, limit, max_violation, bias_check)
    raise ValueError(f"Unknown bound {kind!r}")


def sqnr_db(actual: torch.Tensor, reference: torch.Tensor) -> float:
    """Signal-to-noise ratio of ``actual`` against the reference, for baseline sanity."""
    a = actual.double()
    y = reference.double()
    kept = y.isfinite() & a.isfinite()
    noise = (a - y)[kept].norm().item()
    signal = y[kept].norm().item()
    if not noise:
        return math.inf
    if not signal:
        return -math.inf
    return 20 * math.log10(signal / noise)
