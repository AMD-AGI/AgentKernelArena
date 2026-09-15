# test_flashattention_fwd

The on-disk starting functions contain implemented Triton code. The v2 declaration
therefore uses `implemented` and a frozen `initial_candidate` baseline, regardless
of this directory's historical suite name. Edit only the declared function scopes
(and permitted implementation helpers) in `test_flashattention_fwd.py`. Preserve signatures,
references, input generation, assertions, test parameters and timing policy.
Deliver edits to those files; a fenced code block alone is not a submission.

## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
with one role and one action. Submitted checks use `ARENA_EVAL_PHASE=candidate_evaluation`.
The adapter emits `arena-eval-v1`; Arena owns final score/validation reports.
`workloads.json` retains 12 original collected cases, including 6 performance cases.
Collection is checked against this independent manifest. Original correctness
functions run unchanged. Performance inputs additionally run the task-local
oracle in `_arena_reference.py`, before timing and against observed timed output.
Seeds, case parameters, original assertions/tolerances, launch parameters,
prepare/reset callbacks, warmups and sample counts are unchanged.

Arena times the same Triton path in its independently frozen baseline workspace
and the edited candidate workspace. The old benchmark helper's optional PyTorch
peer timing is not an Arena baseline and is omitted by this adapter. Candidate
measurements still use the canonical helper and its original mean device latency.
Do not edit `performance_utils_pytest.py` or generated benchmark helpers.

Existing skip conditions are retained as visible failures for the complete
manifest: missing hardware features, unsupported combinations, missing references
or incomplete execution cannot qualify this task. These require explicit task
qualification/repair before a campaign; the migration is not a GPU validation.
A missing/empty final kernel never falls back to a reference or starting kernel.

## Original operator instructions

The historical instructions below retain the operator semantics and interface;
the v2 on-disk edit and evaluation contract above governs submission format.


You are an expert in triton programming language. You will be given the function definition for the `flash_fwd_kernel` kernels. Your task is to complete the kernel code. Only complete the kernel code in the function definition, DONT remove any python imports or helper utils in the instruction/code provided, DONT change/interfere with the provided function definition and parameter list.

This kernel, `flash_fwd_kernel`,  performs forward pass of the FlashAttention algorithm
**Your objective is to implement the body of both the kernels `flash_fwd_kernel`.**

You must ensure that:
1.  All arguments received by `flash_fwd_kernel` are kept intact and not modified.
2. Provide you final code in ```python code block. 
Example:
```python
<YOUR-CODE-HERE>
```
The full definitions for `flash_fwd_kernel` and relevant helper utilities are provided in the context below. You only need to complete the code for `flash_fwd_kernel` whilst keeping other things intact. DONT remove Imports and HELPER utils.

######################################## Imports ######################################## 

# import numpy as np
import pytest
import torch

import triton
import triton.language as tl
######################################## Imports ######################################## 


@triton.jit
def flash_fwd_kernel(
    Q, K, V, sm_scale,  # Input tensors and softmax scale
    L, M,  # Intermediate tensors for online softmax
    Out,  # Output tensor
    stride_qz, stride_qh, stride_qm, stride_qk,  # Strides for Q
    stride_kz, stride_kh, stride_kn, stride_kk,  # Strides for K
    stride_vz, stride_vh, stride_vk, stride_vn,  # Strides for V
    stride_oz, stride_oh, stride_om, stride_on,  # Strides for Out
    Z, H, N_CTX, D0,  # Tensor dimensions
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr  # Block sizes
):
    """
    Computes the forward pass of FlashAttention.

    Args:
        Q (Tensor): Query tensor with shape (Z, H, N_CTX, D_HEAD). This kernel expects a pointer to the beginning of this tensor.
        K (Tensor): Key tensor with shape (Z, H, N_CTX, D_HEAD). This kernel expects a pointer to the beginning of this tensor.
        V (Tensor): Value tensor with shape (Z, H, N_CTX, D_HEAD). This kernel expects a pointer to the beginning of this tensor.
        sm_scale (float): Scaling factor applied to the QK^T product before softmax. Typically 1/sqrt(D_HEAD).
        L (Tensor): Output tensor of shape (Z*H, N_CTX) used to store the row-wise sum of `exp(scores - max_scores)` for the online softmax calculation.
                    It acts as the normalizer `l_i` in the FlashAttention algorithm.
        M (Tensor): Output tensor of shape (Z*H, N_CTX) used to store the row-wise maximum of QK^T scores (`m_i` in FlashAttention) for numerically stable online softmax.
        Out (Tensor): Output tensor of shape (Z, H, N_CTX, D_HEAD) where the attention output is stored.
        stride_qz (int): Stride for the Z (batch) dimension of the Q tensor, in terms of number of elements.
        stride_qh (int): Stride for the H (head) dimension of the Q tensor, in terms of number of elements.
        stride_qm (int): Stride for the M (query sequence length, N_CTX) dimension of the Q tensor, in terms of number of elements.
        stride_qk (int): Stride for the K (head dimension, D_HEAD) dimension of the Q tensor, in terms of number of elements.
        stride_kz (int): Stride for the Z (batch) dimension of the K tensor.
        stride_kh (int): Stride for the H (head) dimension of the K tensor.
        stride_kn (int): Stride for the N (key sequence length, N_CTX) dimension of the K tensor.
        stride_kk (int): Stride for the K (head dimension, D_HEAD) dimension of the K tensor.
        stride_vz (int): Stride for the Z (batch) dimension of the V tensor.
        stride_vh (int): Stride for the H (head) dimension of the V tensor.
        stride_vk (int): Stride for the K (key/value sequence length, N_CTX) dimension of the V tensor. (Note: `_vk` here refers to the sequence dim for V).
        stride_vn (int): Stride for the N (head dimension, D_HEAD) dimension of the V tensor. (Note: `_vn` here refers to the head dim for V).
        stride_oz (int): Stride for the Z (batch) dimension of the Out tensor.
        stride_oh (int): Stride for the H (head) dimension of the Out tensor.
        stride_om (int): Stride for the M (query sequence length, N_CTX) dimension of the Out tensor.
        stride_on (int): Stride for the N (head dimension, D_HEAD) dimension of the Out tensor.
        Z (int): Batch size.
        H (int): Number of attention heads.
        N_CTX (int): Sequence length (context length). Assumed to be the same for Q, K, and V for simplicity in this kernel's structure, particularly for causal masking and L, M storage.
        D0 (int): This parameter represents the sequence length dimension (N_CTX) for a single head's data matrix. It is used in `tl.make_block_ptr` for the `shape` argument's first dimension when viewing a head's Q, K, or V data. It should be equal to N_CTX.
        BLOCK_M (tl.constexpr): The size of the block along the query sequence length dimension (M). Queries are processed in blocks of this size.
        BLOCK_DMODEL (tl.constexpr): The head dimension size (D_HEAD). The kernel processes the full head dimension.
        BLOCK_N (tl.constexpr): The size of the block along the key/value sequence length dimension (N). Keys and values are loaded and processed in blocks of this size.
    """
    # Your code here




