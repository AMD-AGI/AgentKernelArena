"""Protected torch reference for one mHC post+pre boundary call.

mhc_post_ref, mhc_pre_ref and mhc_post_pre_ref are copied from ROCm/aiter
op_tests/test_mhc.py at 2cb3d17589b9253ea3b3ef5222fc7f810a8f3ef0 (lines
683-690, 365-429 without the unused hc_head branch, 735-766); the copy keeps
this module importable without that test file's global device side effect.
Default tolerances there are aiter/test_common.py:520-525 (rtol=atol=1e-2),
used for all four outputs at test_mhc.py:937-950.

Public functions take explicit arguments and keep no module state; Torch is
imported lazily so the contract can be checked on a host without Torch.
"""
from __future__ import annotations

import math

OUTPUTS = ("post_mix", "comb_mix", "layer_input_out", "next_residual")
CALL_KWARGS = ("rms_eps", "hc_pre_eps", "hc_sinkhorn_eps", "hc_post_mult_value",
               "sinkhorn_repeat", "norm_eps")


def validate_contract(contract):
    """Fail before importing Torch/AITER when numerical acceptance is unbound."""
    if not isinstance(contract, dict) or contract.get("reference_kind") != "aiter_op_tests_mhc_post_pre_ref":
        raise ValueError("Unsupported or unresolved numerical contract")
    kwargs = contract.get("call_kwargs")
    if not isinstance(kwargs, dict) or set(kwargs) != set(CALL_KWARGS):
        raise ValueError("Call constants must be bound exactly")
    if type(kwargs["sinkhorn_repeat"]) is not int or kwargs["sinkhorn_repeat"] <= 0:
        raise ValueError("sinkhorn_repeat must be a positive integer")
    for name in CALL_KWARGS[:-2] + ("norm_eps",):
        value = kwargs[name]
        if type(value) is not float or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Call constant must be a positive finite float: {name}")
    if contract.get("hc_mult") != 4 or contract.get("hidden_size") != 7168:
        raise ValueError("Model geometry must be bound (hc_mult=4, hidden_size=7168)")
    acceptance = contract.get("acceptance")
    if not isinstance(acceptance, dict) or acceptance.get("method") != "allclose" or set(acceptance.get("outputs", {})) != set(OUTPUTS):
        raise ValueError("An explicit allclose policy is required for every output")
    for output, policy in acceptance["outputs"].items():
        for name in ("rtol", "atol"):
            value = policy.get(name) if isinstance(policy, dict) else None
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Acceptance for {output} requires a finite nonnegative {name}")
    if not isinstance(contract.get("evidence"), list) or not contract["evidence"]:
        raise ValueError("The numerical contract requires its evidence references")


def mhc_post_ref(x, residual, post_layer_mix, comb_res_mix):
    import torch

    term2 = torch.bmm(comb_res_mix.mT, residual.float())
    return (x.float().unsqueeze(-2) * post_layer_mix + term2).bfloat16()


def mhc_pre_ref(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                hc_post_mult_value, sinkhorn_repeat, norm_weight=None, norm_eps=1e-6):
    import torch

    hc_mult = residual.shape[-2]
    residual_flat = residual.flatten(-2, -1).float()
    sqrsum = residual_flat.square().sum(-1)
    out = residual_flat @ fn.T
    mixes = out * (sqrsum.unsqueeze(-1) / fn.shape[-1] + rms_eps).rsqrt()
    hc_scale = torch.cat([
        hc_scale[0].expand(hc_mult),
        hc_scale[1].expand(hc_mult),
        hc_scale[2].expand(hc_mult * hc_mult),
    ])
    mixes = mixes * hc_scale + hc_base
    pre_mix = mixes[:, :hc_mult].sigmoid().unsqueeze(-1) + hc_pre_eps
    post_mix = (mixes[:, hc_mult:2 * hc_mult].sigmoid() * hc_post_mult_value).unsqueeze(-1)
    res_mix = mixes[:, 2 * hc_mult:].view(-1, hc_mult, hc_mult)
    res_mix = res_mix.softmax(-1) + hc_sinkhorn_eps
    res_mix = res_mix / (res_mix.sum(-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        res_mix = res_mix / (res_mix.sum(-1, keepdim=True) + hc_sinkhorn_eps)
        res_mix = res_mix / (res_mix.sum(-2, keepdim=True) + hc_sinkhorn_eps)
    layer_input = (residual * pre_mix).sum(-2)
    if norm_weight is not None:
        x = layer_input
        rms = torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + norm_eps)
        layer_input = (x.float() * rms * norm_weight.float()).bfloat16()
    return post_mix, res_mix, layer_input.bfloat16()


def reference(payload, contract):
    """Unfused post then pre, in SGLang's order; returns the four public outputs."""
    validate_contract(contract)
    layer_input, residual_in, post_layer_mix, comb_res_mix, fn, hc_scale, hc_base = payload["args"]
    kwargs = payload["kwargs"]
    # SGLang passes the 2-D squeezed post mix; the upstream reference broadcasts [m, hc, 1].
    next_residual = mhc_post_ref(layer_input, residual_in, post_layer_mix.unsqueeze(-1), comb_res_mix)
    post_mix, comb_mix, layer_input_out = mhc_pre_ref(
        next_residual, fn, hc_scale, hc_base,
        kwargs["rms_eps"], kwargs["hc_pre_eps"], kwargs["hc_sinkhorn_eps"],
        kwargs["hc_post_mult_value"], kwargs["sinkhorn_repeat"],
        norm_weight=kwargs["norm_weight"], norm_eps=kwargs["norm_eps"],
    )
    return post_mix, comb_mix, layer_input_out, next_residual


def check_output(actual, expected, contract):
    """Require four complete, finite, contiguous outputs within the declared policy."""
    validate_contract(contract)
    import torch

    if not isinstance(expected, tuple) or len(expected) != len(OUTPUTS):
        raise ValueError("Reference produced an invalid result; this is not a candidate mismatch")
    if any(not isinstance(t, torch.Tensor) or not torch.isfinite(t).all() for t in expected):
        raise ValueError("Reference produced a nonfinite output; this is not a candidate mismatch")
    if not isinstance(actual, (tuple, list)) or len(actual) != len(OUTPUTS):
        raise AssertionError("Candidate must return (post_mix, comb_mix, layer_input_out, next_residual)")
    metrics = {}
    for name, got, want in zip(OUTPUTS, actual, expected):
        if not isinstance(got, torch.Tensor):
            raise AssertionError(f"{name}: candidate output is not a tensor")
        if got.shape != want.shape or got.dtype != want.dtype or got.device != want.device:
            raise AssertionError(f"{name}: shape, dtype or device differs from the contract")
        if not got.is_contiguous():
            raise AssertionError(f"{name}: output must be contiguous")
        if not torch.isfinite(got).all():
            raise AssertionError(f"{name}: output contains nonfinite values")
        policy = contract["acceptance"]["outputs"][name]
        try:
            torch.testing.assert_close(got, want, rtol=policy["rtol"], atol=policy["atol"], equal_nan=False)
        except AssertionError as error:
            raise AssertionError(f"{name}: {error}") from None
        error = (got.float() - want.float()).abs()
        metrics[name] = {"compared_elements": want.numel(),
                         "max_absolute_error": error.max().item() if error.numel() else 0.0}
    return metrics


def check_separated(first, second, contract):
    """Require a valid pair that rejects the previous oracle output tuple."""
    # Invalid references are binding failures, not evidence of changed values.
    for output in (first, second):
        try:
            check_output(output, output, contract)
        except AssertionError as error:
            raise ValueError("Reference produced an invalid output for replay qualification") from error
    if any(a.shape != b.shape or a.dtype != b.dtype or a.device != b.device
           for a, b in zip(first, second)):
        raise ValueError("Reference output geometry differs between replay states")
    import torch
    moved = {}
    for name, a, b in zip(OUTPUTS, first, second):
        policy = contract["acceptance"]["outputs"][name]
        try:
            torch.testing.assert_close(a, b, rtol=policy["rtol"], atol=policy["atol"], equal_nan=False)
        except AssertionError:
            moved[name] = True
        else:
            moved[name] = False
    if any(moved.values()):
        return {"old_oracle_rejected": True, "per_output_moved": moved}
    raise ValueError("Captured replay pair cannot distinguish stale output under the frozen numerical rule")


def check_reference_controls(payload, contract):
    """Known-answer identity for the post step and comparator rejection controls.

These check reference math and comparator behavior; they do not establish
that the pinned kernel agrees with the reference on these inputs.
"""
    validate_contract(contract)
    import torch

    layer_input, residual_in, post_layer_mix, comb_res_mix = payload["args"][:4]
    identity = dict(payload)
    eye = torch.eye(comb_res_mix.shape[-1], dtype=comb_res_mix.dtype, device=comb_res_mix.device)
    identity["args"] = (layer_input, residual_in, torch.zeros_like(post_layer_mix),
                        eye.expand_as(comb_res_mix).contiguous()) + tuple(payload["args"][4:])
    if not torch.equal(reference(identity, contract)[3], residual_in):
        raise AssertionError("Reference failed the zero-post/identity-comb residual identity")
    expected = reference(payload, contract)
    controls = {"zero_post_identity_comb": "PASS"}
    for index, name in enumerate(OUTPUTS):
        policy = contract["acceptance"]["outputs"][name]
        amplitude = max(1.0, 2.0 * policy["atol"])
        wrong = [t.clone() for t in expected]
        wrong[index].view(wrong[index].shape[0], -1)[0] += amplitude  # one token row only
        try:
            check_output(tuple(wrong), expected, contract)
        except AssertionError:
            controls[f"wrong_row_{name}_rejected"] = "PASS"
        else:
            raise AssertionError(f"Comparator accepted a deliberately incorrect {name} row")
    return controls
