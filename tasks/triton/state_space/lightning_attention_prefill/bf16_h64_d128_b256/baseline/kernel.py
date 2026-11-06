# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Adapted from vllm-project/vllm, vllm/model_executor/layers/lightning_attn.py
# (_fwd_diag_kernel, _fwd_kv_parallel, _fwd_kv_reduce, _fwd_none_diag_kernel and the forward of
# _attention / lightning_attention).
"""MiniMax lightning-attention prefill (block-parallel linear attention) in Triton.

Four kernels, as in vLLM's lightning_attention for head size 128 and block size 256:

1. _fwd_diag_kernel      attention inside every 256-token block (causal, exponentially decayed);
2. _fwd_kv_parallel      the decayed K^T V outer product of every block;
3. _fwd_kv_reduce        the prefix scan of those states across blocks, starting from kv_history
                         and leaving the final state in kv_history;
4. _fwd_none_diag_kernel the contribution of all earlier blocks (q times the block's prefix state).

The decay table of the keys, which vLLM builds with torch.exp on the host, is computed inside
_fwd_kv_parallel. _fwd_kv_parallel also widens its key and value loads to float32, as
_fwd_diag_kernel does: vLLM runs it on float32 activations, and with this task's bfloat16 keys and
values its tl.dot would multiply float32 decayed keys by bfloat16 values, which Triton 3.8.0
rejects ("Both operands must be same dtype").
"""
import torch
import triton
import triton.language as tl

BLOCK = 256
DIAG_CBLOCK = 32
KV_CBLOCK = 64


@triton.jit
def _fwd_diag_kernel(
    Q,
    K,
    V,
    Out,
    S,
    b: tl.constexpr,
    h: tl.constexpr,
    n,
    d: tl.constexpr,
    e: tl.constexpr,
    BLOCK: tl.constexpr,
    NUM_BLOCK,
    CBLOCK: tl.constexpr,
):
    # This kernel computes the diagonal blocks of the attention matrix
    # Each diagonal block represents attention
    # where queries attend to keys in the same block
    off = tl.program_id(0)
    off_bh = off // NUM_BLOCK  # batch-head index
    off_block = off % NUM_BLOCK  # block index within the sequence
    off_cblock = tl.program_id(1)  # sub-block index within a block

    off_h = off_bh % h  # head index

    # Calculate base offsets for the current batch and head
    qk_offset = off_bh * n * d
    v_offset = off_bh * n * e
    o_offset = off_bh * n * e

    # Calculate offsets for the current block
    block_offset = off_block * BLOCK
    qk_block_offset = block_offset * d
    v_block_offset = block_offset * e
    o_block_offset = block_offset * e

    # Calculate offsets for the current sub-block
    cblock_offset = off_cblock * CBLOCK
    q_cblock_offset = cblock_offset * d
    o_cblock_offset = cblock_offset * e

    # Calculate pointers to the query, key, value, and output tensors
    Q_block_ptr = (
        Q
        + qk_offset
        + qk_block_offset
        + q_cblock_offset
        + tl.arange(0, CBLOCK)[:, None] * d
        + tl.arange(0, d)[None, :]
    )
    K_trans_block_ptr = (
        K
        + qk_offset
        + qk_block_offset
        + tl.arange(0, CBLOCK)[None, :] * d
        + tl.arange(0, d)[:, None]
    )
    V_block_ptr = (
        V
        + v_offset
        + v_block_offset
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, e)[None, :]
    )
    O_block_ptr = (
        Out
        + o_offset
        + o_block_offset
        + o_cblock_offset
        + tl.arange(0, CBLOCK)[:, None] * e
        + tl.arange(0, e)[None, :]
    )

    # Load the decay rate for the current head
    S_block_ptr = S + off_h
    s = tl.load(S_block_ptr)

    i = off_cblock
    q_index = tl.arange(0, CBLOCK) + i * CBLOCK

    # Load query values
    q = tl.load(Q_block_ptr, mask=block_offset + q_index[:, None] < n, other=0.0).to(
        tl.float32
    )

    # Initialize output accumulator
    qkv = tl.zeros([CBLOCK, e], dtype=tl.float32)

    # Process all sub-blocks up to and
    # including the current one (causal attention)
    for j in range(i + 1):
        kv_index = tl.arange(0, CBLOCK) + j * CBLOCK
        diff = q_index[:, None] - kv_index[None, :]
        s_index = s * diff
        # Apply causal mask: only attend to positions before the current one
        s_index = tl.where(diff >= 0, -s_index, float("-inf"))
        decay = tl.exp(s_index)

        # Load key and value
        k_trans = tl.load(
            K_trans_block_ptr,
            mask=block_offset + kv_index[None, :] < n,
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            V_block_ptr,
            mask=block_offset + kv_index[:, None] < n,
            other=0.0,
        ).to(tl.float32)

        # Compute attention scores and apply decay
        qk = tl.dot(q, k_trans) * decay

        # Compute weighted values and accumulate
        qkv += tl.dot(qk, v)

        # Move to the next sub-block
        K_trans_block_ptr += CBLOCK * d
        V_block_ptr += CBLOCK * e

    # Store the result
    tl.store(
        O_block_ptr,
        qkv.to(O_block_ptr.dtype.element_ty),
        mask=block_offset + q_index[:, None] < n,
    )


@triton.jit
def _fwd_kv_parallel(
    K,
    V,
    S,
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

    # Decay of every key to the end of its block: exp(-s * (BLOCK - 1 - position)).
    # (vLLM precomputes this table with torch.exp on the host; it is computed in-kernel here.)
    s = tl.load(S + off_h).to(tl.float32)

    kv_index = tl.arange(0, CBLOCK)

    # Initialize the key-value outer product accumulator
    kv = tl.zeros([D_FBLOCK, E_FBLOCK], dtype=tl.float32)

    # Handle the last block which might be smaller than BLOCK
    split_n = n - (NUM_BLOCK - 1) * BLOCK if off_block == NUM_BLOCK - 1 else BLOCK
    left_shift = tl.cdiv(split_n, CBLOCK) * CBLOCK - split_n
    num_blocks = min(tl.cdiv(split_n, CBLOCK), NUM_CBLOCK)
    k_decay_pos = (NUM_CBLOCK - num_blocks) * CBLOCK + tl.arange(0, CBLOCK)

    # Process all sub-blocks in the current block
    for j in range(num_blocks):
        left_bound = (1 - j) * left_shift
        # Load key and value, handling boundary conditions.
        # (vLLM feeds float32 activations; the bfloat16 keys and values of this task are widened
        # to float32 exactly, as _fwd_diag_kernel does, so that tl.dot sees two float32 operands.)
        k_trans = tl.load(
            K_trans_block_ptr - left_shift * d,
            mask=kv_index[None, :] >= left_bound,
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            V_block_ptr - left_shift * e,
            mask=kv_index[:, None] >= left_bound,
            other=0.0,
        ).to(tl.float32)

        # Decay factor and weighted key-value outer product
        k_decay = tl.exp(-s * (BLOCK - 1 - k_decay_pos).to(tl.float32))

        # NOTE: Need to add the extra dim here due to AMD MLIR lowering error.
        # Please don't move it back until issue is resolved.
        # Issue: https://github.com/ROCm/triton/issues/907
        k_decay = k_decay[None, :]

        kv += tl.dot(k_trans * k_decay, v)

        # Move to the next sub-block
        K_trans_block_ptr += CBLOCK * d
        V_block_ptr += CBLOCK * e
        k_decay_pos += CBLOCK

    # Store the result
    tl.store(KV_block_ptr, kv.to(KV_block_ptr.dtype.element_ty))


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


def run(*, q, k, v, slope_rate, kv_history, out):
    h, n, d = q.shape
    e = v.shape[-1]
    num_block = triton.cdiv(n, BLOCK)

    # Step 1: diagonal blocks of the attention matrix
    _fwd_diag_kernel[(h * num_block, BLOCK // DIAG_CBLOCK)](
        q, k, v, out, slope_rate, 1, h, n, d, e,
        BLOCK=BLOCK, NUM_BLOCK=num_block, CBLOCK=DIAG_CBLOCK,
    )

    # Step 2: key-value outer products of every block
    kv = torch.empty((h, num_block, d, e), dtype=torch.float32, device=q.device)
    _fwd_kv_parallel[(h, num_block)](
        k, v, slope_rate, kv, 1, h, n, d, e,
        BLOCK=BLOCK, NUM_BLOCK=num_block, D_FBLOCK=d, E_FBLOCK=e, NUM_FBLOCK=1,
        CBLOCK=KV_CBLOCK, NUM_CBLOCK=BLOCK // KV_CBLOCK,
    )

    # Step 3: prefix scan of the block states, updating the history in place
    _fwd_kv_reduce[(h, 1)](
        slope_rate, kv, kv_history, 1, h, n, d, e,
        BLOCK=BLOCK, NUM_BLOCK=num_block, D_FBLOCK=d, E_FBLOCK=e,
    )

    # Step 4: contribution of the earlier blocks
    _fwd_none_diag_kernel[(h, num_block * (BLOCK // KV_CBLOCK))](
        q, out, slope_rate, kv, 1, h, n, d, e,
        BLOCK=BLOCK, NUM_BLOCK=num_block, E_FBLOCK=e, CBLOCK=KV_CBLOCK,
        NUM_CBLOCK=BLOCK // KV_CBLOCK,
    )
