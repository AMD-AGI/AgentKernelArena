"""Independent per-coordinate bounds for the declared BF16 split intermediates.

The mathematical attention reference remains the original FP32 softmax/V
calculation. This additional bound does not read candidate outputs or call its
implementation. It accounts for the existing two BF16 partial vectors and BF16
log-sum-exp values, without requiring a candidate to use the same tiling.
"""
import torch


class NumericalMismatch(AssertionError):
    """A completed numerical comparison rejected the candidate output."""


def split_rounding_bounds(means, absolute_means, log_sums):
    """Enclose rounding of two independently computed attention partitions.

    BF16 round-to-nearest has unit roundoff u=2^-8. Rounding unnormalized
    exponentials before the V product contributes at most u*E[abs(V)].
    Storing that partial mean adds u*(abs(mean)+previous_error). Rounding the
    two log sums perturbs their difference by at most u*(abs(L0)+abs(L1));
    sigmoid is monotone, so its two endpoints bound the mixing weight.
    The final BF16 store is bounded after combining the partial intervals.
    Ordinary FP32 accumulation and output comparison still use the original
    0.01 absolute/relative tolerance in the caller.
    """
    assert means.shape[-2] == 2 and log_sums.shape[-1] == 2
    assert means.shape == absolute_means.shape
    u = 2.0 ** -8
    probability_error = u * absolute_means
    partial_error = probability_error + u * (means.abs() + probability_error)
    delta = log_sums[..., 0] - log_sums[..., 1]
    log_error = u * log_sums.abs().sum(dim=-1)
    weight_lo = torch.sigmoid(delta - log_error).unsqueeze(-1)
    weight_hi = torch.sigmoid(delta + log_error).unsqueeze(-1)
    partial_lo = means - partial_error
    partial_hi = means + partial_error
    lower = torch.minimum(
        weight_lo * partial_lo[..., 0, :] + (1-weight_lo) * partial_lo[..., 1, :],
        weight_hi * partial_lo[..., 0, :] + (1-weight_hi) * partial_lo[..., 1, :])
    upper = torch.maximum(
        weight_lo * partial_hi[..., 0, :] + (1-weight_lo) * partial_hi[..., 1, :],
        weight_hi * partial_hi[..., 0, :] + (1-weight_hi) * partial_hi[..., 1, :])
    final_error = u * torch.maximum(lower.abs(), upper.abs())
    return lower - final_error, upper + final_error


def attention_rounding_bounds(inputs):
    """Compute the bound using pristine inputs and standalone PyTorch math."""
    assert inputs['num_kv_splits'] == 2
    assert inputs['attn_logits'].dtype == torch.bfloat16
    query = inputs['q'].float()
    batch = query.shape[0]
    length = inputs['kv_indices'].numel() // batch
    ids = inputs['kv_indices'].long()
    keys = inputs['k_input'][ids, 0].float().reshape(batch, length, -1)
    values = inputs['v_input'][ids, 0].float().reshape(batch, length, -1)
    scores = torch.einsum('bhd,btd->bht', query, keys) * inputs['sm_scale']
    width = (length + 1) // 2
    means, absolute_means, log_sums = [], [], []
    for start in (0, width):
        logits = scores[..., start:start+width]
        v = values[:, start:start+width]
        weights = torch.softmax(logits, dim=-1)
        means.append(torch.einsum('bht,btd->bhd', weights, v))
        absolute_means.append(torch.einsum('bht,btd->bhd', weights, v.abs()))
        log_sums.append(torch.logsumexp(logits, dim=-1))
    return split_rounding_bounds(torch.stack(means, dim=-2),
                                 torch.stack(absolute_means, dim=-2),
                                 torch.stack(log_sums, dim=-1))


def check_rounding_bounds(actual, expected, lower, upper):
    """Every coordinate must be bounded; no fraction of arbitrary outliers."""
    assert lower.shape == upper.shape == expected.shape == actual.shape
    assert torch.isfinite(lower).all() and torch.isfinite(upper).all()
    tolerance = 1e-2 + 1e-2 * expected.float().abs()
    # Include the ideal reference itself, so an implementation with more
    # accurate intermediates remains valid as well.
    lower = torch.minimum(lower, expected.float()) - tolerance
    upper = torch.maximum(upper, expected.float()) + tolerance
    if not (actual.float() >= lower).all():
        raise NumericalMismatch('MLA coordinate below BF16 rounding bound')
    if not (actual.float() <= upper).all():
        raise NumericalMismatch('MLA coordinate above BF16 rounding bound')
