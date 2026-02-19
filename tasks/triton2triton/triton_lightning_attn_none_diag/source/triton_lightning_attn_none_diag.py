"""Lightning Attention - Non-diagonal (cross-block) attention kernel."""
import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_none_diag_kernel(
    Q,
    Out,
    S,
    KV,
    b: tl.constexpr,
    h: tl.constexpr,
    n,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK,
    E_FBLOCK: tl.constexpr,
    CBLOCK: tl.constexpr,
    NUM_CBLOCK: tl.constexpr,
):
    # This kernel computes the non-diagonal blocks of the attention matrix
    # Each non-diagonal block represents attention
    # where queries attend to keys in different blocks
    off_bh = tl.program_id(0)  # batch-head index
    off_h = off_bh % h  # head index

    off_nc = tl.program_id(1)
    off_n = off_nc // NUM_CBLOCK  # block index
    off_c = off_nc % NUM_CBLOCK  # sub-block index
    off_e = tl.program_id(2)  # output feature block index

    n_offset = off_n * BLOCK
    c_offset = off_c * CBLOCK
    e_offset = off_e * E_FBLOCK
    block_offset = n_offset + c_offset

    # Calculate offsets for the current batch, head, and block
    q_offset = off_bh * n * d + (n_offset + c_offset) * d
    o_offset = off_bh * n * e + (n_offset + c_offset) * e + e_offset
    kv_offset = off_bh * NUM_BLOCK * d * e + off_n * d * e + e_offset

    # Calculate pointers to the query, output, and key-value tensors
    Q_block_ptr = (
        Q + q_offset + tl.arange(0, CBLOCK)[:, None] * d + tl.arange(0, d)[None, :]
    )
    O_block_ptr = (
        Out
        + o_offset
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, E_FBLOCK)[None, :]
    )
    KV_block_ptr = (
        KV + kv_offset + tl.arange(0, d)[:, None] * e + tl.arange(0, E_FBLOCK)[None, :]
    )

    # Load the decay rate for the current head
    S_block_ptr = S + off_h
    s = tl.load(S_block_ptr)

    c_array = tl.arange(0, CBLOCK)

    # Load the key-value outer product for the current block
    kv = tl.load(KV_block_ptr).to(tl.float32)
    q_index = block_offset + tl.arange(0, CBLOCK)

    # Load query values
    q = tl.load(Q_block_ptr, mask=q_index[:, None] < n, other=0.0).to(tl.float32)

    # Compute decay factors for the current sub-block
    q_decay = tl.exp(-s.to(tl.float32) * (off_c * CBLOCK + c_array[:, None]))

    # Compute non-diagonal attention output
    qkv_none_diag = tl.dot(q, kv) * q_decay

    # Load diagonal attention output (computed by _fwd_diag_kernel)
    qkv_diag = tl.load(O_block_ptr, mask=q_index[:, None] < n, other=0.0).to(tl.float32)

    # Combine diagonal and non-diagonal attention outputs
    qkv = qkv_diag + qkv_none_diag

    # Store the result
    tl.store(
        O_block_ptr, qkv.to(O_block_ptr.dtype.element_ty), mask=q_index[:, None] < n
    )


def lightning_attn_none_diag_forward(q, o, s, kv, BLOCK=256, CBLOCK=64):
    """
    Standalone wrapper for the non-diagonal block attention kernel.

    This kernel adds cross-block attention contribution to the output tensor.
    The output tensor `o` should already contain the diagonal attention result.

    Args:
        q: Query tensor [B, H, N, D]
        o: Output tensor [B, H, N, E] (modified in-place, should contain diag result)
        s: Slope/decay tensor [H] or [1, H, 1, 1]
        kv: KV state tensor [B, H, NUM_BLOCK, D, E] (prefix-summed)
        BLOCK: Block size (default 256)
        CBLOCK: Sub-block size (default 64)

    Returns:
        o: Updated output tensor (in-place)
    """
    q = q.contiguous()

    if s.dim() == 4:
        s = s.squeeze(0).squeeze(-1).squeeze(-1)
    s = s.contiguous()

    b_size, h_size, seq_n, d = q.shape
    e = o.shape[-1]

    NUM_BLOCK = triton.cdiv(seq_n, BLOCK)
    NUM_CBLOCK = BLOCK // CBLOCK
    assert BLOCK % CBLOCK == 0

    NUM_FBLOCK = 1
    E_FBLOCK = e // NUM_FBLOCK

    grid = (b_size * h_size, NUM_BLOCK * NUM_CBLOCK)
    _fwd_none_diag_kernel[grid](
        q, o, s, kv,
        b_size, h_size, seq_n, d, e,
        BLOCK=BLOCK,
        NUM_BLOCK=NUM_BLOCK,
        E_FBLOCK=E_FBLOCK,
        CBLOCK=CBLOCK,
        NUM_CBLOCK=NUM_CBLOCK,
    )

    return o
