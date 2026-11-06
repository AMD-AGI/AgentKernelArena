"""Exact integer reference of the position / sequence-length preparation."""
import torch


def reference(*, idx_mapping, query_start_loc, num_computed_tokens):
    device = idx_mapping.device
    max_num_reqs = num_computed_tokens.shape[0]
    total = int(query_start_loc[-1])
    start = query_start_loc.long()
    lens = start[1:] - start[:-1]
    computed = num_computed_tokens.long()[idx_mapping.long()]
    request_of_token = torch.repeat_interleave(torch.arange(lens.numel(), device=device), lens,
                                               output_size=total)
    offset = torch.arange(total, device=device) - start[:-1][request_of_token]
    positions = computed[request_of_token] + offset
    lengths = torch.zeros(max_num_reqs, dtype=torch.int64, device=device)
    lengths[:lens.numel()] = computed + lens
    return {"pos": positions.to(torch.int64), "seq_lens": lengths.to(torch.int32)}
