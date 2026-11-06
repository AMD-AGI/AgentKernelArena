"""MXFP4 operands, routing and the expert-sorted token layout for the MXFP4 MoE GEMM task.

Activations and expert weights are drawn as standard-normal bfloat16 (as the previous task did) and
quantized to MXFP4 (mxfp4_numerics.py). The router draws standard-normal logits and keeps the top-k
softmax entries, so every token picks topk distinct experts. The sorted layout is what vLLM's
moe_align_block_size produces for the kernel's tile height ``align_block`` (the BLOCK_SIZE_M of AITER's
MX_FP4 MoE config for this token count): the (token, slot) indices ``t * topk + k`` grouped by expert in
ascending order, each expert's group padded to a multiple of align_block with the sentinel
``n_tokens * topk``, one expert id per tile (-1 for the unused tail), and the padded total.
"""
import torch

from mxfp4_numerics import quantize


def _sorted_layout(topk_ids, num_experts, block, sorted_len, num_blocks):
    device = topk_ids.device
    flat = topk_ids.reshape(-1).long()
    numel = flat.numel()
    counts = torch.bincount(flat, minlength=num_experts)
    padded = (counts + block - 1) // block * block
    starts = torch.cumsum(padded, 0) - padded
    order = torch.argsort(flat, stable=True)
    grouped = flat[order]
    rank = torch.arange(numel, device=device) - (torch.cumsum(counts, 0) - counts)[grouped]
    sorted_ids = torch.full((sorted_len,), numel, dtype=torch.int64, device=device)
    sorted_ids[starts[grouped] + rank] = order
    tiles = padded // block
    expert_ids = torch.full((num_blocks,), -1, dtype=torch.int64, device=device)
    used = int(tiles.sum())
    expert_ids[:used] = torch.repeat_interleave(torch.arange(num_experts, device=device), tiles)
    total = padded.sum().reshape(1)
    return sorted_ids.to(torch.int32), expert_ids.to(torch.int32), total.to(torch.int32)


def make_inputs(case, inputs, generator, device):
    axes = case["axes"]
    tokens, hidden, out_features = axes["n_tokens"], axes["hidden_size"], axes["intermediate_size"]
    num_experts, topk = axes["num_experts"], axes["topk"]
    x = torch.randn((tokens, hidden), generator=generator, device=device, dtype=torch.float32)
    a, a_scale = quantize(x.to(torch.bfloat16))
    b = torch.empty((num_experts, out_features, hidden // 2), dtype=torch.uint8, device=device)
    b_scale = torch.empty((num_experts, out_features, hidden // 32), dtype=torch.uint8, device=device)
    for expert in range(num_experts):
        w = torch.randn((out_features, hidden), generator=generator, device=device, dtype=torch.float32)
        b[expert], b_scale[expert] = quantize(w.to(torch.bfloat16))
    logits = torch.randn((tokens, num_experts), generator=generator, device=device, dtype=torch.float32)
    topk_ids = logits.softmax(dim=-1).topk(topk, dim=-1).indices.to(torch.int32).contiguous()
    sorted_ids, expert_ids, total = _sorted_layout(topk_ids, num_experts, axes["align_block"],
                                                   axes["sorted_len"], axes["num_blocks"])
    return {"a": a, "a_scale": a_scale, "b": b, "b_scale": b_scale, "topk_ids": topk_ids,
            "sorted_token_ids": sorted_ids, "expert_ids": expert_ids, "num_tokens_post_padded": total}
