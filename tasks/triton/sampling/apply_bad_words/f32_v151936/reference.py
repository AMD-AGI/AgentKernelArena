"""Independent PyTorch reference for bad-words masking (exact: only -inf stores)."""
import torch


def reference(*, logits, expanded_idx_mapping, bad_word_token_ids, bad_word_offsets, num_bad_words,
              all_token_ids, prompt_len, total_len, input_ids, expanded_local_pos, max_num_bad_words):
    """Return logits with the last token of every matched bad word set to -inf.

    Logit row i belongs to request slot s = expanded_idx_mapping[i] at speculative position
    p = expanded_local_pos[i]. The tokens generated so far are all_token_ids[s, prompt_len:total_len];
    the p speculative input tokens input_ids[i - p : i] follow them. A bad word [w0 .. wk] masks wk when
    the last k tokens of that effective sequence equal [w0 .. w(k-1)].
    """
    out = logits.clone()
    n_rows = out.shape[0]
    width = bad_word_token_ids.shape[1]
    history = all_token_ids.shape[1]
    rows = torch.arange(n_rows, device=out.device)
    slot = expanded_idx_mapping.long()
    pos = expanded_local_pos.long()
    first = rows - pos
    prompt = prompt_len.long()[slot]
    output_len = total_len.long()[slot] - prompt
    effective = output_len + pos
    count = num_bad_words.long()[slot]
    spec_tokens = input_ids.long()
    for j in range(int(max_num_bad_words)):
        start = bad_word_offsets[slot, j].long()
        end = bad_word_offsets[slot, j + 1].long()
        prefix = end - start - 1
        match = (j < count) & (prefix <= effective)
        longest = int(prefix[match].max()) if bool(match.any()) else 0
        for t in range(longest):
            expected = bad_word_token_ids[slot, (start + t).clamp(0, width - 1)].long()
            actual_pos = effective - prefix + t
            from_spec = actual_pos >= output_len
            spec = spec_tokens[(first + actual_pos - output_len).clamp(0, n_rows - 1)]
            generated = all_token_ids[slot, (prompt + actual_pos).clamp(0, history - 1)].long()
            actual = torch.where(from_spec, spec, generated)
            match = match & ((t >= prefix) | (expected == actual))
        last = bad_word_token_ids[slot, (end - 1).clamp(0, width - 1)].long()
        out[rows[match], last[match]] = float("-inf")
    return out
