"""Exact integer reference of the request-to-token pool writer."""
import torch


def reference(*, req_to_token, req_pool_indices, prefix_indices, pre_lens, seq_lens, extend_lens,
              out_cache_loc):
    result = req_to_token.clone()
    rows = req_pool_indices.tolist()
    pre = pre_lens.tolist()
    seq = seq_lens.tolist()
    extend = extend_lens.tolist()
    start = 0
    for i, row in enumerate(rows):
        if extend[i] != seq[i] - pre[i]:
            raise ValueError("extend_lens must equal seq_lens - pre_lens")
        result[row, :pre[i]] = prefix_indices[i, :pre[i]].to(torch.int32)
        result[row, pre[i]:seq[i]] = out_cache_loc[start:start + extend[i]].to(torch.int32)
        start += extend[i]
    return result
