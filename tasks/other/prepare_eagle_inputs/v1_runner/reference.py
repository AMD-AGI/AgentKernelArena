"""Exact integer reference of the EAGLE draft-input preparation."""
import torch


def reference(*, target_input_ids, target_positions, idx_mapping, last_sampled, next_prefill_tokens,
              num_sampled, num_rejected, query_start_loc, seq_lens, draft_input_ids, draft_positions):
    device = target_input_ids.device
    max_num_reqs = last_sampled.shape[0]
    ids = draft_input_ids.clone()
    pos = draft_positions.clone()
    last_index = torch.zeros(max_num_reqs, dtype=torch.int64, device=device)
    q_start = torch.zeros(max_num_reqs + 1, dtype=torch.int32, device=device)
    lengths = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
    starts = query_start_loc.tolist()
    slots = idx_mapping.tolist()
    rejected = num_rejected.tolist()
    sampled = num_sampled.tolist()
    num_reqs = len(slots)
    for b in range(num_reqs):
        begin = starts[b]
        length = starts[b + 1] - begin - rejected[b]
        if sampled[b] > 0:
            next_token = int(last_sampled.reshape(-1)[slots[b]])
        else:
            next_token = int(next_prefill_tokens[slots[b]])
        # Shift the target ids by one, then the next token closes the request.
        ids[begin:begin + length - 1] = target_input_ids[begin + 1:begin + length]
        ids[begin + length - 1] = next_token
        pos[begin:begin + length] = target_positions[begin:begin + length]
        last_index[b] = begin + length - 1
        q_start[b] = begin
    lengths[:num_reqs] = seq_lens
    q_start[num_reqs:] = starts[num_reqs]
    return {"last_token_indices": last_index, "draft_current_step": torch.zeros(1, dtype=torch.int64, device=device),
            "draft_query_start_loc": q_start, "draft_seq_lens": lengths,
            "draft_input_ids": ids, "draft_positions": pos}
