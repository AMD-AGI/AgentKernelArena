"""Sparse top-k token indices of a sparse-attention (DSA-style) decode/prefill batch.

Every token row holds ``topk`` request-local token positions chosen by the indexer. Most
positions lie inside the request's block table; about 10% are -1 (rows with fewer valid
positions than ``topk``), and a small fraction lie beyond the table (block index out of range),
which the kernel must also map to -1.
"""
import torch


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    n_tokens, topk = axes["n_tokens"], axes["topk"]
    limit = axes["max_blocks_per_req"] * axes["block_size"]
    tok = torch.randint(0, limit, (n_tokens, topk), generator=generator, device=device, dtype=torch.int32)
    draw = torch.rand((n_tokens, topk), generator=generator, device=device)
    beyond = torch.randint(limit, 2 * limit, (n_tokens, topk), generator=generator, device=device,
                           dtype=torch.int32)
    tok = torch.where(draw < 0.005, beyond, tok)
    tok = torch.where((draw >= 0.005) & (draw < 0.105), torch.full_like(tok, -1), tok)
    return {"token_indices": tok}
