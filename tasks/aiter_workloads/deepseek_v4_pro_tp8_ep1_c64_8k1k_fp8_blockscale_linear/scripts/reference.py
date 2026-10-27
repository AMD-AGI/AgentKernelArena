"""Protected FP8 block-scaled linear reference; no numerical policy defaults.

The oracle is an fp32 matmul of dequantized operands built from the canonical
(unshuffled) weight, independent of AITER and of the candidate. Every enrolled
carrier supplies fp8 x and required fp32 x_scale; the oracle dequantizes them
directly. The producer's activation quantization is outside the enrolled call.
The matmul follows run_torch in ROCm/aiter
op_tests/test_gemm_a8w8_blockscale.py:32-58 at
2cb3d17589b9253ea3b3ef5222fc7f810a8f3ef0.

The retained BF16 helper path first applies the per-1x128 quantization rule of
aiter.ops.quant.per_group_quant_hip on gfx950, all in fp32: amax seeded with
the 1e-10 floor (csrc/kernels/quant_kernels.cu:289, csrc/include/quant_amax_floor.h:18),
scale = amax * fp32(1 / fp8_max) (:231, :359), values = x * fp32(1 / scale)
(:416) clamped to +-fp8_max and cast to e4m3fn (aiter_opus_plus.h:44-60).
That helper path is outside the enrolled FP8 cases.
"""
from __future__ import annotations

import math

_EXPECTED = {
    "reference_kind": "fp32_dequantized_matmul",
    "activation_quant": "per_1x128_floored_amax_times_inverse_fp8_max_fp32_scale",
    "fp8_dtype": "float8_e4m3fn",
    "fp8_max": 448.0,
    "block_size": 128,
    "weight_layout": "aiter_shuffle_weight_16x16",
    "x_scale_layout": "fp32_m_by_k_div_128_column_major",
    "output_dtype": "bfloat16",
}


def validate_contract(contract):
    """Fail before importing Torch/AITER when numerical acceptance is unbound."""
    if not isinstance(contract, dict) or contract.get("binding_status") != "frozen":
        raise ValueError("Numerical contract is unresolved; task execution is not qualified")
    for key, value in _EXPECTED.items():
        if type(contract.get(key)) is not type(value) or contract[key] != value:
            raise ValueError(f"Unsupported or unresolved numerical contract field: {key}")
    if not isinstance(contract.get("evidence"), list) or not contract["evidence"] or not all(
        isinstance(item, str) and item.strip() for item in contract["evidence"]
    ):
        raise ValueError("Frozen numerical contract requires its evidence references")
    acceptance = contract.get("acceptance")
    if not isinstance(acceptance, dict) or acceptance.get("method") != "allclose":
        raise ValueError("Only an explicitly frozen allclose acceptance is implemented")
    for name in ("rtol", "atol"):
        value = acceptance.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Acceptance requires an explicit finite nonnegative {name}")


def quantize_1x128(x, contract):
    """Torch statement of the bound activation quantization: (fp8 [m,k], fp32 [m,k/128])."""
    import torch

    m, k = x.shape
    block, fp8_max = contract["block_size"], contract["fp8_max"]
    groups = x.float().reshape(m, k // block, block)
    fp32 = lambda value: torch.tensor(value, dtype=torch.float32, device=x.device)
    scale = torch.maximum(groups.abs().amax(-1), fp32(1e-10)) * fp32(1.0 / fp8_max)
    values = (groups * (1.0 / scale).unsqueeze(-1)).clamp(-fp8_max, fp8_max)
    return values.to(getattr(torch, contract["fp8_dtype"])).reshape(m, k), scale


def reference(payload, contract):
    """Evaluate the bound quantized semantics, independent of candidate source."""
    validate_contract(contract)
    import torch

    if torch.backends.cuda.matmul.allow_tf32:
        raise ValueError("The fp32 reference requires TF32 matmul to be disabled")
    x, _, weight_scale = payload["args"]
    x_scale = payload["kwargs"].get("x_scale")
    q, scale = quantize_1x128(x, contract) if x_scale is None else (x, x_scale)
    block = contract["block_size"]
    m, k = q.shape
    activation = (q.float().reshape(m, k // block, block) * scale.float().unsqueeze(-1)).reshape(m, k)
    canonical = payload["reference"]["canonical_weight"]
    expanded = weight_scale.float().repeat_interleave(block, 0).repeat_interleave(block, 1)
    weight = canonical.float() * expanded
    return (activation @ weight.t()).to(torch.bfloat16)


def check_output(actual, expected, contract):
    """Require complete, finite outputs and the protected acceptance rule."""
    validate_contract(contract)
    import torch

    if not isinstance(expected, torch.Tensor) or not torch.isfinite(expected).all():
        raise ValueError("Reference produced an invalid output; this is not a candidate mismatch")
    if not isinstance(actual, torch.Tensor):
        raise AssertionError("Candidate must return a tensor")
    if actual.shape != expected.shape or actual.dtype != expected.dtype or actual.device != expected.device:
        raise AssertionError("Candidate output shape, dtype or device differs from the contract")
    if not torch.isfinite(actual).all():
        raise AssertionError("Candidate output contains nonfinite values")
    policy = contract["acceptance"]
    torch.testing.assert_close(actual, expected, rtol=policy["rtol"], atol=policy["atol"], equal_nan=False)
    error = (actual.float() - expected.float()).abs()
    return {"compared_elements": expected.numel(), "max_absolute_error": error.max().item() if error.numel() else 0.0}


def check_separated(first, second, contract):
    """Require a valid pair that rejects the previous oracle output."""
    # Invalid references are binding failures, not evidence of changed values.
    for output in (first, second):
        try:
            check_output(output, output, contract)
        except AssertionError as error:
            raise ValueError("Reference produced an invalid output for replay qualification") from error
    if first.shape != second.shape or first.dtype != second.dtype or first.device != second.device:
        raise ValueError("Reference output geometry differs between replay states")
    import torch
    policy = contract["acceptance"]
    try:
        torch.testing.assert_close(first, second, rtol=policy["rtol"], atol=policy["atol"], equal_nan=False)
    except AssertionError:
        return {"old_oracle_rejected": True}
    raise ValueError("Captured replay pair cannot distinguish stale output under the frozen numerical rule")


def check_reference_controls(payload, contract):
    """Known zero-input identity and a comparator rejection control.

These check actual math/comparator behavior; they do not establish that this
reference recipe matches the served kernel's nonzero rounding semantics.
"""
    validate_contract(contract)
    import torch

    x, weight = payload["args"][0], payload["args"][1]
    zero_payload = dict(payload)
    zero_payload["args"] = (torch.zeros_like(x),) + payload["args"][1:]
    actual = reference(zero_payload, contract)
    expected = torch.zeros((x.shape[0], weight.shape[0]), dtype=torch.bfloat16, device=x.device)
    if actual.shape != expected.shape or actual.dtype != expected.dtype or not torch.equal(actual, expected):
        raise AssertionError("Reference failed the zero-activation identity")
    real = reference(payload, contract)
    amplitude = max(1.0, 2.0 * contract["acceptance"]["atol"])
    wrong = real + amplitude
    try:
        check_output(wrong, real, contract)
    except AssertionError:
        pass
    else:
        raise AssertionError("Comparator accepted deliberately incorrect output")
    return {"zero_activation": "PASS", "wrong_output_rejected": "PASS"}
