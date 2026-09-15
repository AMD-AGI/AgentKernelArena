# test_chained_dot_fp8

The on-disk starting functions contain implemented Triton code. The v2 declaration
therefore uses `implemented` and a frozen `initial_candidate` baseline, regardless
of this directory's historical suite name. Edit only the declared function scopes
(and permitted implementation helpers) in `test_chained_dot_fp8.py`. Preserve signatures,
references, input generation, assertions, test parameters and timing policy.
Deliver edits to those files; a fenced code block alone is not a submission.

## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
with one role and one action. Submitted checks use `ARENA_EVAL_PHASE=candidate_evaluation`.
The adapter emits `arena-eval-v1`; Arena owns final score/validation reports.
`workloads.json` retains 24 original collected cases, including 20 performance cases.
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


You are an expert in triton programming language. You will be given the function definition for the `_chained_dot`. Your task is to complete the kernel code. Only optimize the kernel code in the function definition, DONT remove any python imports or helper utils in the instruction/code provided, DONT change/interfere with the provided function definition and parameter list.

This kernel, `_chained_dot`,  is designed to perform "chained dot product" operation.

**Your objective is to optimize the body of `_chained_dot`.**

You must ensure that:
1.  All arguments received by `_chained_dot` are kept intact and not modified.
2. Provide you final code in ```python code block. 
Example:
```python
<YOUR-CODE-HERE>
```


The full definition for `_chained_dot` and relevant helper utilities are provided in the context below. You only need to optimize the code for `_chained_dot` whilst keeping other things intact.

"""
Testing the (FP8) case of a dot op that consumes the output (MFMA) of
another dot op as an input.

"""
#Imports

import math
import pytest
import torch

import triton
import triton.language as tl

########################## HELPER utils ##########################
TORCH_HAS_FP8E4 = hasattr(torch, 'float8_e4m3fnuz')
float8: tl.constexpr = None if not TORCH_HAS_FP8E4 else torch.float8_e4m3fnuz
########################## HELPER utils ##########################

@triton.jit
def _chained_dot(
    Q,
    K,
    V,
    Out,
    q_desc,
    k_desc,
    v_desc,
    s_sc,
    s_desc,
    o_sc,
    stride_qz,
    stride_qm,
    stride_qd,
    stride_kz,
    stride_kn,
    stride_kd,
    stride_vz,
    stride_vd,
    stride_vn,
    stride_oz,
    stride_om,
    stride_od,
    Z,
    M,
    N,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_FP8: tl.constexpr,
):
    """
    This Triton kernel computes a "chained dot product" operation,
    effectively performing (Q @ K.T) @ V in a tiled manner.
    This is a core component of attention mechanisms.
    The kernel is parallelized across the M dimension of Q (target sequence length)
    and the Z dimension (batch size * number of heads).
    It iteratively loads blocks of K and V to compute partial results for the output.
    FP8 support allows for reduced precision computation with scaling factors.

    Parameters:
    -----------
    Q : tl.tensor
        Pointer to the Q (query) tensor. Expected shape (Z, M, D).
    K : tl.tensor
        Pointer to the K (key) tensor. Expected shape (Z, N, D).
    V : tl.tensor
        Pointer to the V (value) tensor. Expected shape (Z, N, D).
    Out : tl.tensor
        Pointer to the O (output) tensor. Expected shape (Z, M, D).
    q_desc : float
        Dequantization scale for the Q tensor (used if USE_FP8 is True).
    k_desc : float
        Dequantization scale for the K tensor (used if USE_FP8 is True).
    v_desc : float
        Dequantization scale for the V tensor (used if USE_FP8 is True).
    s_sc : float
        Scaling factor applied to the intermediate S (QK^T) tensor before dot product with V (used if USE_FP8 is True).
        This can be thought of as a quantization scale if S were to be stored in FP8.
    s_desc : float
        Dequantization scale for the intermediate S (QK^T) tensor when it's used in S@V (used if USE_FP8 is True).
    o_sc : float
        Quantization scale for the O (output) tensor (used if USE_FP8 is True).
    stride_qz : int
        Stride of the Q tensor along the Z (batch/head) dimension.
    stride_qm : int
        Stride of the Q tensor along the M (sequence length of Q / rows) dimension.
    stride_qd : int
        Stride of the Q tensor along the D (feature/embedding) dimension.
    stride_kz : int
        Stride of the K tensor along the Z (batch/head) dimension.
    stride_kn : int
        Stride of the K tensor along the N (sequence length of K / rows) dimension.
    stride_kd : int
        Stride of the K tensor along the D (feature/embedding) dimension.
    stride_vz : int
        Stride of the V tensor along the Z (batch/head) dimension.
    stride_vd : int
        Stride of the V tensor along the D (feature/embedding) dimension.
    stride_vn : int
        Stride of the V tensor along the N (sequence length of V / rows) dimension.
    stride_oz : int
        Stride of the Out tensor along the Z (batch/head) dimension.
    stride_om : int
        Stride of the Out tensor along the M (sequence length of Out / rows) dimension.
    stride_od : int
        Stride of the Out tensor along the D (feature/embedding) dimension.
    Z : int
        Size of the Z dimension (e.g., batch_size * num_heads).
    M : int
        Size of the M dimension (e.g., sequence length of Q, number of rows in Q).
    N : int
        Size of the N dimension (e.g., sequence length of K and V, number of columns in K.T / rows in V).
    BLOCK_D : tl.constexpr
        Tile size for the D dimension (feature/embedding dimension). Compile-time constant.
    BLOCK_M : tl.constexpr
        Tile size for the M dimension (rows of Q). Compile-time constant.
    BLOCK_N : tl.constexpr
        Tile size for the N dimension (columns of K.T / rows of V). Compile-time constant.
    USE_FP8 : tl.constexpr
        Boolean flag indicating whether to use FP8 E4M3 precision and apply scaling. Compile-time constant.
    """
    # Your code here





