"""Independent fp32 recurrence and complete observable-state checks."""
from __future__ import annotations

from cases import tensor_identity


def expected(payload):
    from kda_ref import chunk_kda_ref

    kw = payload["kwargs"]
    initial = kw["initial_state"][kw["ssm_state_indices"].long()].clone()
    return chunk_kda_ref(
        **{name: kw[name].float() for name in ("q", "k", "v", "g", "beta")},
        A_log=kw["A_log"], dt_bias=kw["dt_bias"], scale=kw["scale"],
        initial_state=initial, output_final_state=True, state_v_first=True,
        use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True, safe_gate=True, lower_bound=-5.0,
    )


def _byte_equal(actual, wanted):
    import torch

    return torch.equal(actual.contiguous().view(torch.uint8),
                       wanted.contiguous().view(torch.uint8))


def _relative_error(actual, wanted, label, limit):
    import torch

    if actual.shape != wanted.shape or not torch.isfinite(actual).all():
        raise AssertionError(f"{label}: wrong shape or nonfinite values")
    # Do not let one request's error disappear in another request's norm.
    diff = (actual.float() - wanted.float()).reshape(actual.shape[0], -1)
    gold = wanted.float().reshape(wanted.shape[0], -1)
    ratios = diff.square().mean(1).sqrt() / (gold.square().mean(1).sqrt() + 1e-8)
    worst = float(ratios.max().item())
    if not worst < limit:
        raise AssertionError(f"{label}: per-sequence RMS error {worst} >= {limit}")
    return worst


def check_result(payload, result, gold, before, *, limit=0.005):
    kw = payload["kwargs"]
    if not isinstance(result, tuple) or len(result) != 2:
        raise AssertionError("The public call must return (out, final_state)")
    output, state = result
    if tensor_identity(output) != before["identities"]["out"]:
        raise AssertionError("Returned output must use the caller's output buffer")
    if tensor_identity(state) != before["identities"]["initial_state"]:
        raise AssertionError("Returned final state must alias the input state pool")
    for name, identity in before["identities"].items():
        if tensor_identity(kw[name]) != identity:
            raise AssertionError(f"Public tensor identity/layout changed: {name}")
    for name, value in before["read_only"].items():
        if not _byte_equal(kw[name], value):
            raise AssertionError(f"Read-only input changed: {name}")
    active = kw["ssm_state_indices"].long()
    metrics = {
        "output_max_sequence_rms_ratio": _relative_error(output, gold[0], "output", limit),
        "state_max_sequence_rms_ratio": _relative_error(state[active], gold[1], "state", limit),
    }
    rows = payload["raw"].view(payload["slots"], payload["stride"])
    original = before["raw"].view_as(rows)
    if not _byte_equal(rows[payload["unused"]], original[payload["unused"]]):
        raise AssertionError("An unselected state slot changed")
    if not _byte_equal(rows[:, payload["plane"]:], original[:, payload["plane"]:]):
        raise AssertionError("State-slot padding changed")
    metrics.update(untouched_state_bytes_equal=True, read_only_bytes_equal=True)
    return metrics
