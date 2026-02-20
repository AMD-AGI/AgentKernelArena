"""
Triton decode attention stage2 kernel: reduces partial attention outputs from
stage1 across KV splits using logsumexp.

Each program handles one (batch, head) and combines the partial outputs from
all kv_splits to produce the final attention output.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fwd_kernel_stage2(
    Mid_O,
    o,
    lse,
    B_Seqlen,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    stride_lse_bs,
    NUM_KV_SPLITS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)

    cur_batch_seq_len = tl.load(B_Seqlen + cur_batch)

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + offs_d
    offs_logic = cur_batch * stride_mid_ob + cur_head * stride_mid_oh + Lv

    for split_kv_id in range(0, NUM_KV_SPLITS):
        kv_len_per_split = tl.cdiv(cur_batch_seq_len, NUM_KV_SPLITS)
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_batch_seq_len)

        if split_kv_end > split_kv_start:
            tv = tl.load(
                Mid_O + offs_v + split_kv_id * stride_mid_os, mask=mask_d, other=0.0
            )
            tlogic = tl.load(Mid_O + offs_logic + split_kv_id * stride_mid_os)
            n_e_max = tl.maximum(tlogic, e_max)

            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(tlogic - n_e_max)
            acc += exp_logic * tv

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    tl.store(
        o + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        acc / e_sum,
        mask=mask_d,
    )
    lse_val = e_max + tl.log(e_sum)
    tl.store(
        lse + cur_batch * stride_lse_bs + cur_head,
        lse_val,
    )


def decode_softmax_reducev_fwd(
    mid_o,
    q,
    o,
    lse,
    v_buffer,
    b_seq_len,
    num_kv_splits,
):
    """
    Standalone wrapper for _fwd_kernel_stage2.

    Args:
        mid_o: [batch, num_heads, num_kv_splits, head_dim+1] — partial outputs from stage1
               (first head_dim elements are partial output, last element is logsumexp)
        q: [batch, num_heads, head_dim] — only used for shape info
        o: [batch, num_heads, head_dim] — final output
        lse: [batch, num_heads] — final logsumexp
        v_buffer: used for Lv shape info only
        b_seq_len: [batch] — sequence lengths
        num_kv_splits: int
    """
    batch, head_num = q.shape[0], q.shape[1]
    Lv = v_buffer.shape[-1]
    BLOCK_DV = triton.next_power_of_2(Lv)

    NUM_KV_SPLITS = num_kv_splits

    grid = (batch, head_num)
    _fwd_kernel_stage2[grid](
        mid_o,
        o,
        lse,
        b_seq_len,
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        o.stride(0),
        o.stride(1),
        lse.stride(0),
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        BLOCK_DV=BLOCK_DV,
        Lv=Lv,
        num_warps=4,
        num_stages=2,
    )
