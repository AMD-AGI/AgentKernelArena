"""Lightning Attention - KV reduce kernel."""
import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kv_reduce(
    S,
    KV,
    KV_HISTORY,
    b: tl.constexpr,
    h: tl.constexpr,
    n,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK,
    D_FBLOCK: tl.constexpr,
    E_FBLOCK: tl.constexpr,
):
    # This kernel reduces the key-value outer products
    # across blocks and updates the KV history
    off_bh = tl.program_id(0)  # batch-head index
    off_h = off_bh % h  # head index

    kv_offset = off_bh * NUM_BLOCK * d * e

    # Calculate pointer to the key-value tensor
    KV_block_ptr = (
        KV
        + kv_offset
        + tl.arange(0, D_FBLOCK)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )

    # Load the decay rate for the current head
    s_ptrs = S + off_h
    s = tl.load(s_ptrs)

    # Calculate pointer to the key-value history tensor
    kv_history_offset = off_bh * d * e
    KV_HISTORY_block_ptr = (
        KV_HISTORY
        + kv_history_offset
        + tl.arange(0, D_FBLOCK)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )

    # Load the previous key-value history
    kv_pre = tl.load(KV_HISTORY_block_ptr).to(tl.float32)

    # Process all blocks in reverse order to compute the prefix sum
    for i in range(NUM_BLOCK):
        block_size = min(n - i * BLOCK, BLOCK)
        # Compute decay factor for the current block
        block_decay = tl.exp(-s.to(tl.float32) * block_size)

        # Load the current key-value outer product
        kv_cur = tl.load(KV_block_ptr).to(tl.float32)
        # Store the previous key-value history to the current block
        tl.store(KV_block_ptr, kv_pre.to(KV_block_ptr.dtype.element_ty))

        # Update the key-value history with the current block
        kv_pre = block_decay * kv_pre + kv_cur
        KV_block_ptr += d * e

    # Store the updated key-value history
    tl.store(KV_HISTORY_block_ptr, kv_pre)


def lightning_attn_kv_reduce_forward(s, kv, kv_history, n, BLOCK=256):
    """
    Standalone wrapper for the KV reduce kernel.

    Args:
        s: Slope/decay tensor [H] or [1, H, 1, 1]
        kv: KV outer product tensor [B, H, NUM_BLOCK, D, E] (float32)
        kv_history: KV history tensor [B, H, D, E] (float32, modified in-place)
        n: Sequence length
        BLOCK: Block size (default 256)

    Returns:
        kv: Modified KV tensor (prefix-summed, in-place)
        kv_history: Updated KV history (in-place)
    """
    if s.dim() == 4:
        s = s.squeeze(0).squeeze(-1).squeeze(-1)
    s = s.contiguous()

    b_size, h_size, NUM_BLOCK_val, d, e = kv.shape

    NUM_FBLOCK = 1
    D_FBLOCK = d // NUM_FBLOCK
    E_FBLOCK = e // NUM_FBLOCK

    grid = (b_size * h_size, NUM_FBLOCK)
    _fwd_kv_reduce[grid](
        s, kv, kv_history,
        b_size, h_size, n, d, e,
        BLOCK=BLOCK,
        NUM_BLOCK=NUM_BLOCK_val,
        D_FBLOCK=D_FBLOCK,
        E_FBLOCK=E_FBLOCK,
    )

    return kv, kv_history
