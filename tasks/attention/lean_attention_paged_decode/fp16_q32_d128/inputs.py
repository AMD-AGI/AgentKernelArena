"""Ragged request lengths and per-head KV block tables for the paged decode scenario."""
import torch

BLOCK_N = 16


def request_blocks(batch, total_blocks):
    """Blocks per request: deterministic ragged split of total_blocks, each at least one."""
    weights = [1.0 + 0.5 * (i % 3) for i in range(batch)]
    scale = (total_blocks - batch) / sum(weights)
    blocks = [1 + int(w * scale) for w in weights]
    remainder = total_blocks - sum(blocks)
    for i in range(remainder):
        blocks[i % batch] += 1
    return blocks


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    batch, heads = int(axes["batch"]), int(axes["num_heads"])
    total_blocks = int(axes["total_blocks"])
    blocks = request_blocks(batch, total_blocks)
    cumulative = torch.tensor(blocks, dtype=torch.int64).cumsum(0).to(torch.int32)
    table = torch.rand(heads, total_blocks, generator=generator, device=device).argsort(dim=1)
    return {"kv_block_tables": table.to(torch.int32), "batch_num_block_n": cumulative.to(device)}
