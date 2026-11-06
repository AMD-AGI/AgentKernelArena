"""Exact integer reference for request-local to global slot index translation."""
import torch

_CHUNK_ROWS = 8192
_BLOCK_SIZE = 64


def reference(*, req_id, block_table, token_indices):
    num_tokens = token_indices.shape[0]
    max_blocks = block_table.shape[1]
    table = block_table.long()
    out = torch.empty(token_indices.shape, dtype=torch.int64, device=token_indices.device)
    for start in range(0, num_tokens, _CHUNK_ROWS):
        stop = min(start + _CHUNK_ROWS, num_tokens)
        tok = token_indices[start:stop].long()
        block = torch.div(tok, _BLOCK_SIZE, rounding_mode="floor")
        offset = tok - block * _BLOCK_SIZE
        valid = (tok >= 0) & (block < max_blocks)
        base = table[req_id[start:stop].long().unsqueeze(1), block.clamp(0, max_blocks - 1)]
        out[start:stop] = torch.where(valid, base * _BLOCK_SIZE + offset, torch.full_like(tok, -1))
    return out.to(torch.int32)
