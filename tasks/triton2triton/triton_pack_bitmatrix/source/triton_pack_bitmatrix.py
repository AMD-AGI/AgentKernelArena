"""Pack bitmatrix kernel using Triton, adapted from vLLM gpt_oss_triton_kernels_moe.py.

Packs topk expert IDs into a bitmatrix format where each bit indicates
whether a token is assigned to a particular expert.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def pack_bitmatrix(
    bitmatrix,
    topk_ids,
    n_rows,       # n_rows in bitmatrix / topk_ids
    bm_cols: tl.constexpr,  # n int32_t bitpacks in bitmatrix
    n_expts_act,  # num_topk
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Pack topk_ids into a bitmatrix representation.
    For each row (token), sets bit expert_id in the corresponding uint32 column.
    """
    pid_m = tl.program_id(0)
    offsets_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offsets_k = tl.arange(0, BLOCK_SIZE_K)
    offsets = offsets_m[:, None] * n_expts_act + offsets_k[None, :]
    mask = (offsets_m < n_rows)[:, None] & (offsets_k < n_expts_act)[None, :]
    indices = tl.load(topk_ids + offsets, mask=mask, other=-1)
    div = indices // 32
    rem = indices % 32
    one = tl.cast(1, tl.uint32)

    for i in range(bm_cols):
        offs = tl.arange(0, BLOCK_SIZE_K // 32) + i * (BLOCK_SIZE_K // 32)
        x = tl.where(
            div[:, :, None] == offs[None, None, :], (one << rem)[:, :, None], 0
        )
        y = tl.reduce_or(x, axis=1)
        bitmatrix_ptrs = bitmatrix + offsets_m[:, None] * bm_cols + offs[None, :]
        tl.store(bitmatrix_ptrs, y, mask=offsets_m[:, None] < n_rows)


def pack_topk_to_bitmatrix(
    topk_ids: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """
    Pack topk_ids into a bitmatrix.

    Args:
        topk_ids: [n_rows, topk] expert assignments (int16)
        num_experts: total number of experts
    Returns:
        bitmatrix: [n_rows, bm_cols] uint32 tensor where each bit indicates
                   if token is assigned to that expert
    """
    topk_ids = topk_ids.to(torch.int16)
    n_rows, num_topk = topk_ids.shape

    BLOCK_SIZE_M = 512
    BLOCK_SIZE_K = 32

    bm_cols = triton.cdiv(num_experts, BLOCK_SIZE_K)
    bitmatrix = torch.zeros(
        (n_rows, bm_cols), dtype=torch.uint32, device=topk_ids.device
    )

    grid = (triton.cdiv(n_rows, BLOCK_SIZE_M),)
    pack_bitmatrix[grid](
        bitmatrix,
        topk_ids,
        n_rows,
        bm_cols,
        num_topk,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return bitmatrix
