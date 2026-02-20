"""Lightning Attention - Parallel KV outer product kernel."""
import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kv_parallel(
    K,
    V,
    K_decay,
    KV,
    b: tl.constexpr,
    h: tl.constexpr,
    n,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK,
    D_FBLOCK: tl.constexpr,
    E_FBLOCK: tl.constexpr,
    NUM_FBLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
    NUM_CBLOCK: tl.constexpr,
):
    # This kernel computes the key-value outer
    # products for each block in parallel
    off_bh = tl.program_id(0)  # batch-head index
    off_block = tl.program_id(1)  # block index

    off_h = off_bh % h  # head index

    block_offset = off_block * BLOCK

    # Calculate offsets for the current block
    k_block_offset = block_offset * d
    v_block_offset = block_offset * e
    kv_block_offset = off_block * d * e

    # Calculate base offsets for the current batch and head
    k_offset = off_bh * n * d
    v_offset = off_bh * n * e
    kv_offset = off_bh * NUM_BLOCK * d * e

    # Calculate pointers to the key, value, and key-value tensors
    K_trans_block_ptr = (
        K
        + k_offset
        + k_block_offset
        + tl.arange(0, CBLOCK)[None, :] * d
        + tl.arange(0, D_FBLOCK)[:, None]
    )
    V_block_ptr = (
        V
        + v_offset
        + v_block_offset
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )
    KV_block_ptr = (
        KV
        + kv_offset
        + kv_block_offset
        + tl.arange(0, D_FBLOCK)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )

    # Load the decay factors for the current head and block
    k_decay_ptr = K_decay + off_h * BLOCK + tl.arange(0, CBLOCK)

    kv_index = tl.arange(0, CBLOCK)

    # Initialize the key-value outer product accumulator
    kv = tl.zeros([D_FBLOCK, E_FBLOCK], dtype=tl.float32)

    # Handle the last block which might be smaller than BLOCK
    split_n = n - (NUM_BLOCK - 1) * BLOCK if off_block == NUM_BLOCK - 1 else BLOCK
    left_shift = tl.cdiv(split_n, CBLOCK) * CBLOCK - split_n
    num_blocks = min(tl.cdiv(split_n, CBLOCK), NUM_CBLOCK)
    k_decay_ptr += (NUM_CBLOCK - num_blocks) * CBLOCK

    # Process all sub-blocks in the current block
    for j in range(num_blocks):
        left_bound = (1 - j) * left_shift
        # Load key and value, handling boundary conditions
        k_trans = tl.load(
            K_trans_block_ptr - left_shift * d,
            mask=kv_index[None, :] >= left_bound,
            other=0.0,
        )
        v = tl.load(
            V_block_ptr - left_shift * e,
            mask=kv_index[:, None] >= left_bound,
            other=0.0,
        )

        # Load decay factor and compute weighted key-value outer product
        k_decay = tl.load(k_decay_ptr)

        # NOTE: Need to add the extra dim here due to AMD MLIR lowering error.
        k_decay = k_decay[None, :]

        kv += tl.dot((k_trans * k_decay).to(tl.float16), v)

        # Move to the next sub-block
        K_trans_block_ptr += CBLOCK * d
        V_block_ptr += CBLOCK * e
        k_decay_ptr += CBLOCK

    # Store the result
    tl.store(KV_block_ptr, kv.to(KV_block_ptr.dtype.element_ty))


def lightning_attn_kv_parallel_forward(k, v, s, n, BLOCK=256, CBLOCK=64):
    """
    Standalone wrapper for the parallel KV outer product kernel.

    Args:
        k: Key tensor [B, H, N, D]
        v: Value tensor [B, H, N, E]
        s: Slope/decay tensor [H] or [1, H, 1, 1]
        n: Sequence length
        BLOCK: Block size (default 256)
        CBLOCK: Sub-block size (default 64)

    Returns:
        kv: KV outer product tensor [B, H, NUM_BLOCK, D, E]
    """
    k = k.contiguous()
    v = v.contiguous()

    if s.dim() == 4:
        s = s.squeeze(0).squeeze(-1).squeeze(-1)
    s = s.contiguous()

    b_size, h_size, seq_n, d = k.shape
    e = v.shape[-1]

    NUM_BLOCK = triton.cdiv(seq_n, BLOCK)
    NUM_CBLOCK = BLOCK // CBLOCK
    assert BLOCK % CBLOCK == 0, "BLOCK must be a multiple of CBLOCK"

    NUM_FBLOCK = 1
    D_FBLOCK = d // NUM_FBLOCK
    E_FBLOCK = e // NUM_FBLOCK

    # Compute decay factors for keys
    array = torch.arange(0, BLOCK, device=k.device) + 1
    k_decay = torch.exp(-s.view(-1, 1) * (BLOCK - array.reshape(1, -1)))

    kv = torch.empty((b_size, h_size, NUM_BLOCK, d, e), dtype=torch.float32, device=k.device)

    grid = (b_size * h_size, NUM_BLOCK)
    _fwd_kv_parallel[grid](
        k, v, k_decay, kv,
        b_size, h_size, seq_n, d, e,
        BLOCK=BLOCK,
        NUM_BLOCK=NUM_BLOCK,
        D_FBLOCK=D_FBLOCK,
        E_FBLOCK=E_FBLOCK,
        NUM_FBLOCK=NUM_FBLOCK,
        CBLOCK=CBLOCK,
        NUM_CBLOCK=NUM_CBLOCK,
    )

    return kv
