# test_iv_dependent_matmul

The on-disk starting functions contain implemented Triton code. The v2 declaration
therefore uses `implemented` and a frozen `initial_candidate` baseline, regardless
of this directory's historical suite name. Edit only the declared function scopes
(and permitted implementation helpers) in `test_iv_dependent_matmul.py`. Preserve signatures,
references, input generation, assertions, test parameters and timing policy.
Deliver edits to those files; a fenced code block alone is not a submission.

## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
with one role and one action. Submitted checks use `ARENA_EVAL_PHASE=candidate_evaluation`.
The adapter emits `arena-eval-v1`; Arena owns final score/validation reports.
`workloads.json` retains 2105 original collected cases, including 2100 performance cases.
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


You are an expert in triton programming language. You will be given the function definition for the `iv_dependent_matmul`. Your task is to complete the kernel code. Only complete the kernel code in the function definition, DONT remove any python imports or helper utils in the instruction/code provided, DONT change/interfere with the provided function definition and parameter list.

This kernel, `iv_dependent_matmul`,  is designed to perform  tiled matrix multiplication (C = A @ B).

**Your objective is to implement the body of `iv_dependent_matmul`.**

You must ensure that:
1.  All arguments received by `iv_dependent_matmul` are kept intact and not modified.
2. Provide you final code in ```python code block. 
Example:
```python
<YOUR-CODE-HERE>
```
The full definition for `iv_dependent_matmul` and relevant helper utilities are provided in the context below. You only need to complete the code for `iv_dependent_matmul` whilst keeping other things intact. DONT remove Imports and HELPER utils.

######################################## Imports ######################################## 
import numpy as np
import pytest
import torch
from numpy.random import RandomState

import triton
import triton.language as tl
######################################## Imports ######################################## 

@triton.jit
def iv_dependent_matmul(a_ptr, b_ptr, c_ptr,
                        M, N, K,
                        stride_am, stride_ak,
                        stride_bk, stride_bn,
                        stride_cm, stride_cn,
                        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                        type: tl.constexpr):
    """
    Performs a tiled matrix multiplication (C = A @ B) with various strategies
    for managing and advancing pointers to input matrices A and B within the
    inner K-loop. The specific strategy for pointer updates is determined by
    the `type` parameter. This kernel is designed to explore different
    instruction scheduling and memory access patterns related to pointer
    arithmetic within the loop.

    Parameters:
    -----------
    a_ptr : tl.pointer_type
        Pointer to the first input matrix A.
    b_ptr : tl.pointer_type
        Pointer to the second input matrix B.
    c_ptr : tl.pointer_type
        Pointer to the output matrix C.
    M : int
        Number of rows in matrix A and C.
    N : int
        Number of columns in matrix B and C.
    K : int
        Number of columns in matrix A and rows in matrix B (the dimension being reduced).
    stride_am : int
        Stride for the M dimension (rows) of matrix A.
    stride_ak : int
        Stride for the K dimension (columns) of matrix A.
    stride_bk : int
        Stride for the K dimension (rows) of matrix B.
    stride_bn : int
        Stride for the N dimension (columns) of matrix B.
    stride_cm : int
        Stride for the M dimension (rows) of matrix C.
    stride_cn : int
        Stride for the N dimension (columns) of matrix C.
    BLOCK_SIZE_M : tl.constexpr
        The size of the block used for tiling along the M dimension.
    BLOCK_SIZE_N : tl.constexpr
        The size of the block used for tiling along the N dimension.
    BLOCK_SIZE_K : tl.constexpr
        The size of the block used for tiling along the K dimension (inner loop).
    type : tl.constexpr (str)
        A string literal controlling the pointer update strategy for `a_ptr` and `b_ptr`
        within the K-loop. Affects how `a_ptrs` and `b_ptrs` are calculated in each
        iteration of the K-loop.
        Possible values:
        - "pre_load": Pointers are calculated at the beginning of each K-loop iteration.
        - "post_load": Pointers are calculated at the end of each K-loop iteration for the next iteration.
        - "post_pre_mixed": `a_ptrs` is calculated at the beginning, `b_ptrs` at the end for the next iteration.
        - "post_load_two_iters": Pointers are advanced to prefetch/prepare for two iterations ahead.
        - "post_load_three_iters": Pointers are advanced to prefetch/prepare for three iterations ahead.
    """
    # Your code here




