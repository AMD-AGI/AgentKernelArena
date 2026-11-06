"""Independent PyTorch reference for allowed-token, logit-bias and minimum-length masking."""
import torch


def reference(*, logits, expanded_idx_mapping, pos, num_allowed_token_ids, allowed_token_ids,
              num_logit_bias, logit_bias_token_ids, logit_bias, min_lens, num_stop_token_ids,
              stop_token_ids):
    """Apply, in order and per logit row: allowlist, additive bias, minimum-length stop masking.

    The result is float32 like the logits; the bias is one correctly rounded float32 addition, which
    the kernel performs identically, so the final logits are compared bit for bit.
    """
    out = logits.clone()
    slots = expanded_idx_mapping.long().tolist()
    positions = pos.long().tolist()
    n_allowed = num_allowed_token_ids.long().tolist()
    n_bias = num_logit_bias.long().tolist()
    n_stop = num_stop_token_ids.long().tolist()
    lengths = min_lens.long().tolist()
    for row, slot in enumerate(slots):
        if n_allowed[slot] > 0:
            keep = allowed_token_ids[slot, :n_allowed[slot]].long()
            kept = out[row, keep].clone()
            out[row].fill_(float("-inf"))
            out[row, keep] = kept
        if n_bias[slot] > 0:
            tokens = logit_bias_token_ids[slot, :n_bias[slot]].long()
            out[row, tokens] = out[row, tokens] + logit_bias[slot, :n_bias[slot]]
        if n_stop[slot] > 0 and positions[row] + 1 < lengths[slot]:
            out[row, stop_token_ids[slot, :n_stop[slot]].long()] = float("-inf")
    return out
