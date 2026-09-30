from __future__ import annotations
import torch as torch

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


def compare_topk_outputs(actual, expected):
    """Compare unordered selections; dual outputs must preserve slot/raw pairing."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise AssertionError("output names differ")
        for name in expected:
            validate_comparison(actual[name], expected[name])
        actual_raw, expected_raw = (
            actual["out_raw_indices"],
            expected["out_raw_indices"],
        )
        actual_order = actual_raw.argsort(dim=-1)
        expected_order = expected_raw.argsort(dim=-1)
        for name in expected:
            if not torch.equal(
                actual[name].gather(1, actual_order),
                expected[name].gather(1, expected_order),
            ):
                raise AssertionError("top-k sets or slot/raw pairing differ")
    else:
        validate_comparison(actual, expected)
        if not torch.equal(actual.sort(dim=-1).values, expected.sort(dim=-1).values):
            raise AssertionError("top-k sets differ")
    # Padding belongs at the end, not just anywhere in the unordered set.
    actual_out = actual["out_page_indices"] if isinstance(actual, dict) else actual
    expected_out = (
        expected["out_page_indices"] if isinstance(expected, dict) else expected
    )
    if not torch.equal(actual_out == -1, expected_out == -1):
        raise AssertionError("padding positions differ")
    # A padded row is necessarily on the short-row path, whose order is fixed.
    short_rows = (expected_out == -1).any(dim=-1)
    if isinstance(expected, dict):
        for name in expected:
            if not torch.equal(actual[name][short_rows], expected[name][short_rows]):
                raise AssertionError("short-row order differs")
    elif not torch.equal(actual_out[short_rows], expected_out[short_rows]):
        raise AssertionError("short-row order differs")


_callable = compare_topk_outputs


def run(*args, **kwargs):
    return _callable(*args, **kwargs)
