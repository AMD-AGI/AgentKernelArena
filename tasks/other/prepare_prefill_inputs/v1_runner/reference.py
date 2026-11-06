"""Exact integer reference of the prefill input preparation (all scheduled requests are prefilling)."""
import torch


def reference(*, idx_mapping, query_start_loc, all_token_ids, prefill_len, num_computed_tokens,
              next_prefill_tokens):
    total_tokens = int(query_start_loc[-1])
    out_ids = torch.zeros(total_tokens, dtype=all_token_ids.dtype, device=all_token_ids.device)
    out_next = next_prefill_tokens.clone()
    num_lookahead = next_prefill_tokens.shape[0]
    slots = idx_mapping.tolist()
    starts = query_start_loc.tolist()
    for b, slot in enumerate(slots):
        total = int(prefill_len[slot])
        computed = int(num_computed_tokens[slot])
        if computed >= total:
            raise ValueError("every scheduled request must still be prefilling")
        begin, end = starts[b], starts[b + 1]
        length = end - begin
        out_ids[begin:end] = all_token_ids[slot, computed:computed + length]
        for k in range(num_lookahead):
            position = computed + length + k
            out_next[k, slot] = all_token_ids[slot, position] if position < total else 0
    return {"input_ids": out_ids, "next_prefill_tokens": out_next}
