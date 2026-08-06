"""Local editable copy of vLLM fused_moe_kernel_gptq_awq (int4 W4A16 MoE GEMM).
Extracted verbatim from vllm/model_executor/layers/fused_moe/fused_moe.py for
PerfSkills kernel-level optimization. Optimize the @triton.jit bodies here.
"""
from vllm.triton_utils import tl, triton

@triton.jit
def write_zeros_to_output(
    c_ptr,
    stride_cm,
    stride_cn,
    pid_n,
    N,
    offs_token,
    token_mask,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    compute_type,
):
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def fused_moe_kernel_gptq_awq(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    b_scale_ptr,
    b_zp_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    N: tl.constexpr,
    K: tl.constexpr,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_bse,
    stride_bsk,
    stride_bsn,
    stride_bze,
    stride_bzk,
    stride_bzn,
    block_k_diviable: tl.constexpr,
    group_size: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    SPLIT_K: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    has_zp: tl.constexpr,
    use_int4_w4a16: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    # Cast to int64 to prevent overflow in stride*offset products
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id).to(tl.int64)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        # -----------------------------------------------------------
        # Write back zeros to the output when the expert is not
        # in the current expert parallel rank.
        write_zeros_to_output(
            c_ptr,
            stride_cm,
            stride_cn,
            pid_n,
            N,
            offs_token,
            token_mask,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            compute_type,
        )
        return

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (
        offs_token[:, None] // top_k * stride_am + offs_k[None, :] * stride_ak
    )

    if use_int4_w4a16:
        # L10: load each packed int4 byte ONCE. The K dimension is packed two
        # nibbles per byte; iterate over BLOCK_SIZE_K//2 byte rows and unpack
        # both nibbles in-register instead of loading every byte twice.
        offs_kh = tl.max_contiguous(
            tl.multiple_of(tl.arange(0, BLOCK_SIZE_K // 2), BLOCK_SIZE_K // 2),
            BLOCK_SIZE_K // 2,
        )
        # group index for the half-tile (row i -> K position 2*i; both nibbles
        # of a byte fall in the same quant group since group_size is even).
        offs_kg = (2 * offs_kh)[:, None]
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + offs_kh[:, None] * stride_bk
            + offs_bn[None, :] * stride_bn
        )
    elif use_int8_w8a16:
        b_ptrs = (
            b_ptr
            + off_experts * stride_be
            + offs_k[:, None] * stride_bk
            + offs_bn[None, :] * stride_bn
        )

    if not has_zp and use_int4_w4a16:
        b_zp_num = 8
    if not has_zp and use_int8_w8a16:
        b_zp_num = 128
    elif has_zp and use_int4_w4a16:
        b_zp_shifter = (offs_bn[None, :] % 2) * 4

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.

        if not block_k_diviable:
            k_mask = offs_k[:, None] < K - k * BLOCK_SIZE_K
            k_other = 0.0
        else:
            k_mask = None
            k_other = None

        if block_k_diviable:
            a = tl.load(a_ptrs, mask=token_mask[:, None], other=0.0)
        else:
            a = tl.load(
                a_ptrs,
                mask=token_mask[:, None] & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                other=0.0,
            )

        if use_int4_w4a16:
            # ---- L10: byte loaded once, both nibbles unpacked in-register ----
            if not block_k_diviable:
                kh_mask = offs_kg < K - k * BLOCK_SIZE_K
                b8 = tl.load(b_ptrs, mask=kh_mask, other=0)
            else:
                b8 = tl.load(b_ptrs)
            lo = (b8 & 0xF).to(tl.float32)   # nibble at K position 2*i
            hi = ((b8 >> 4) & 0xF).to(tl.float32)  # nibble at K position 2*i+1

            # L11: scale/zp are constant within a quant group. With BLOCK_SIZE_K
            # and group_size constexpr, only NG = BLOCK_SIZE_K//group_size
            # distinct group rows exist per K-block. Load [NG, BN] once and
            # broadcast across the (group_size//2) half-tile rows per group.
            NG: tl.constexpr = BLOCK_SIZE_K // group_size
            GH: tl.constexpr = group_size // 2  # half-tile rows per group
            offs_ng = tl.arange(0, NG)
            gidx_g = (offs_ng + NG * k)[:, None]
            b_scale_ptrs = (
                b_scale_ptr
                + off_experts * stride_bse
                + offs_bn[None, :] * stride_bsn
                + gidx_g * stride_bsk
            )
            b_scale_g = tl.load(b_scale_ptrs).to(tl.float32)  # [NG, BN]
            b_scale = tl.reshape(
                tl.broadcast_to(b_scale_g[:, None, :], (NG, GH, BLOCK_SIZE_N)),
                (BLOCK_SIZE_K // 2, BLOCK_SIZE_N),
            )

            if has_zp:
                b_zp_ptrs = (
                    b_zp_ptr
                    + off_experts * stride_bze
                    + (offs_bn[None, :] // 2) * stride_bzn
                    + gidx_g * stride_bzk
                )
                b_zp_g = tl.load(b_zp_ptrs)
                b_zp_g = ((b_zp_g >> b_zp_shifter) & 0xF).to(tl.float32)  # [NG, BN]
                b_zp = tl.reshape(
                    tl.broadcast_to(b_zp_g[:, None, :], (NG, GH, BLOCK_SIZE_N)),
                    (BLOCK_SIZE_K // 2, BLOCK_SIZE_N),
                )
                lo = (lo - b_zp) * b_scale
                hi = (hi - b_zp) * b_scale
            else:
                lo = (lo - b_zp_num) * b_scale
                hi = (hi - b_zp_num) * b_scale

            # Interleave back to [BK, BN]: row 2i = lo[i], row 2i+1 = hi[i].
            lo = lo.to(compute_type)
            hi = hi.to(compute_type)
            j = tl.join(lo, hi)                    # [BK//2, BN, 2]
            j = tl.trans(j, 0, 2, 1)              # [BK//2, 2, BN]
            b = tl.reshape(j, (BLOCK_SIZE_K, BLOCK_SIZE_N))
            accumulator = tl.dot(a, b, acc=accumulator)
            b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk
        else:
            b = tl.load(b_ptrs)
            b_scale_ptrs = (
                b_scale_ptr
                + off_experts * stride_bse
                + offs_bn[None, :] * stride_bsn
                + ((offs_k[:, None] + BLOCK_SIZE_K * k) // group_size) * stride_bsk
            )
            b_scale = tl.load(b_scale_ptrs, mask=k_mask, other=k_other)
            b_scale = b_scale.to(tl.float32)

            if has_zp and use_int8_w8a16:
                offs_k_true = (offs_k[:, None] + BLOCK_SIZE_K * k) // group_size
                b_zp_ptrs = (
                    b_zp_ptr
                    + off_experts * stride_bze
                    + offs_bn[None, :] * stride_bzn
                    + offs_k_true * stride_bzk
                )
                b_zp = tl.load(b_zp_ptrs, mask=k_mask, other=k_other)
                b_zp = b_zp.to(tl.float32)

            if has_zp:
                b = ((b.to(tl.float32) - b_zp) * b_scale).to(compute_type)
            else:
                b = ((b.to(tl.float32) - b_zp_num) * b_scale).to(compute_type)
            accumulator = tl.dot(a, b, acc=accumulator)
            b_ptrs += BLOCK_SIZE_K * stride_bk

        # Advance the A ptr to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)
