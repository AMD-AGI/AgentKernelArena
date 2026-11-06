# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pack variable-length sequences [N, D] into a padded [B, Lmax, D] tensor (vLLM pack_seq_triton)."""
import triton
import triton.language as tl


@triton.jit
def _pack_seq_kernel(
    x_ptr,  # [N, D]
    out_ptr,  # [B, Lmax, D]
    lengths_ptr,  # *i32, [B]
    N: tl.constexpr,
    D: tl.constexpr,
    Lmax: tl.constexpr,
    PAD_VALUE: tl.constexpr,
    PAD_IS_UINT8: tl.constexpr,
    BLOCK_T: tl.constexpr,  # timesteps per program
    BLOCK_D: tl.constexpr,  # features per program
):
    pid_b = tl.program_id(0)  # batch id
    pid_t = tl.program_id(1)  # block over time dimension
    pid_d = tl.program_id(2)  # block over feature dimension
    off_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)  # [BLOCK_T]
    off_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)  # [BLOCK_D]

    # Compute start index and sequence length from cumulative lengths
    in_start = 0
    for i in range(pid_b):
        in_start += tl.load(lengths_ptr + i)
    seq_len = tl.load(lengths_ptr + pid_b)

    # valid time positions for this block
    t_mask = off_t < Lmax

    # compute input row indices for valid (b, t)
    in_row = in_start + off_t
    valid_row = (off_t < seq_len) & t_mask

    # x_ptr: row-major [N, D]
    x_row_ptr = x_ptr + in_row[:, None] * D + off_d[None, :]

    # out_ptr: row-major [B, Lmax, D]
    out_row_ptr = out_ptr + (pid_b * Lmax + off_t)[:, None] * D + off_d[None, :]

    # Initialize with PAD. PAD_IS_UINT8 selects the pad tensor's dtype so
    # integer-typed outputs (e.g. MXFP4 packed nibbles, ue8m0 scale bytes)
    # get an exact-byte pad rather than going through an fp32->uint8 cast
    # that's implementation-defined outside of value 0.
    d_mask = off_d[None, :] < D
    if PAD_IS_UINT8:
        pad_vals = tl.full([BLOCK_T, BLOCK_D], PAD_VALUE, tl.uint8)
    else:
        pad_vals = tl.full([BLOCK_T, BLOCK_D], PAD_VALUE, tl.float32)
    tl.store(out_row_ptr, pad_vals, mask=t_mask[:, None] & d_mask)

    # Load & write only where within seq_len
    x_vals = tl.load(x_row_ptr, mask=valid_row[:, None] & d_mask)
    tl.store(out_row_ptr, x_vals, mask=valid_row[:, None] & d_mask)


def run(*, x, lengths, pad_value, out):
    n_tokens, hidden = x.shape
    batch, max_len, _ = out.shape
    block_t = 64
    block_d = 64
    grid = (batch, triton.cdiv(max_len, block_t), triton.cdiv(hidden, block_d))
    _pack_seq_kernel[grid](
        x,
        out,
        lengths,
        n_tokens,
        hidden,
        max_len,
        PAD_VALUE=float(pad_value),
        PAD_IS_UINT8=False,
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )
