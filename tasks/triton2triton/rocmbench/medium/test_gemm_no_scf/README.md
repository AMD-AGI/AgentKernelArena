# test_gemm_no_scf

The on-disk starting functions contain implemented Triton code. The v2 declaration
therefore uses `implemented` and a frozen `initial_candidate` baseline, regardless
of this directory's historical suite name. Edit only the declared function scopes
(and permitted implementation helpers) in `test_gemm_no_scf.py`. Preserve signatures,
references, input generation, assertions, test parameters and timing policy.
Deliver edits to those files; a fenced code block alone is not a submission.

## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
with one role and one action. Submitted checks use `ARENA_EVAL_PHASE=candidate_evaluation`.
The adapter emits `arena-eval-v1`; Arena owns final score/validation reports.
`workloads.json` retains 52 original collected cases, including 20 performance cases.
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


You are an expert in triton programming language. You will be given the function definition for the `matmul_no_scf_kernel` kernels. Your task is to optimize the kernel code for better performance while preserving correctness. Only optimize the kernel code in the function definition, DONT remove any python imports or helper utils in the instruction/code provided, DONT change/interfere with the provided function definition and parameter list.

This kernel, `matmul_no_scf_kernel`,  performs single block of matrix multiplication (C = A @ B) without Structured Control Flow (SCF)
**Your objective is to optimize the body of both the kernels `matmul_no_scf_kernel`.**

You must ensure that:
1.  All arguments received by `matmul_no_scf_kernel` are kept intact and not modified.
2. Provide you final code in ```python code block. 
Example:
```python
<YOUR-CODE-HERE>
```
The full definitions for `matmul_no_scf_kernel` and relevant helper utilities are provided in the context below. You only need to optimize the code for `matmul_no_scf_kernel` whilst keeping other things intact. DONT remove Imports and HELPER utils.

######################################## Imports ######################################## 
import itertools
import os
import re

import pytest
import torch
from torch.testing import assert_close

import triton
import triton.language as tl

######################################## Imports ######################################## 




@triton.jit
def matmul_no_scf_kernel(
    a_ptr,  # tl.pointer_type(dtype)
    b_ptr,  # tl.pointer_type(dtype)
    c_ptr,  # tl.pointer_type(dtype)
    M: int,
    N: int,
    K: int,
    stride_am: int,
    stride_ak: int,
    stride_bk: int,
    stride_bn: int,
    stride_cm: int,
    stride_cn: int,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    FLOAT16_OUTPUT: tl.constexpr,
    USE_TMA_EPILOGUE: tl.constexpr
):
    """
    Computes a single block of matrix multiplication (C = A @ B) without explicit
    iteration over the K dimension (i.e., no `tl.for_loop` for K accumulation,
    implying K must be equal to BLOCK_K). This kernel is  named "no_scf"
    because performing the full matmul accumulation over K would typically
    introduce Structured Control Flow (SCF) in the generated MLIR/LLVM IR,
    which this kernel avoids by processing only one K-block.

    It loads one block of matrix A and one block of matrix B, performs a dot
    product, and stores the resulting block to matrix C. The kernel should support
    optional casting of the output to float16 and an optional TMA-based epilogue
    for storing the result.

    Parameters:
    -----------
    a_ptr : tl.pointer_type
        Pointer to the input matrix A.
    b_ptr : tl.pointer_type
        Pointer to the input matrix B.
    c_ptr : tl.pointer_type
        Pointer to the output matrix C.
    M : int
        Number of rows in matrix A and C. Expected to be equal to BLOCK_M.
    N : int
        Number of columns in matrix B and C. Expected to be equal to BLOCK_N.
    K : int
        Number of columns in matrix A and rows in matrix B (the common dimension).
        Expected to be equal to BLOCK_K.
    stride_am : int
        Stride of matrix A along the M dimension (row stride).
    stride_ak : int
        Stride of matrix A along the K dimension (column stride).
    stride_bk : int
        Stride of matrix B along the K dimension (row stride).
    stride_bn : int
        Stride of matrix B along the N dimension (column stride).
    stride_cm : int
        Stride of matrix C along the M dimension (row stride).
    stride_cn : int
        Stride of matrix C along the N dimension (column stride).
    BLOCK_M : tl.constexpr
        Compile-time constant defining the height of the blocks to be processed from
        matrices A and C.
    BLOCK_N : tl.constexpr
        Compile-time constant defining the width of the blocks to be processed from
        matrices B and C.
    BLOCK_K : tl.constexpr
        Compile-time constant defining the width of the block from matrix A and
        the height of the block from matrix B (common dimension for dot product).
    FLOAT16_OUTPUT : tl.constexpr
        Compile-time boolean constant. If True, the output matrix C will be cast
        to float16 before storing. Otherwise, it will be stored in the compute
        precision (e.g., float32).
    USE_TMA_EPILOGUE : tl.constexpr
        Compile-time boolean constant. If True, the epilogue (storing the result C)
        will use Tensor Memory Access (TMA) operations via `tl.make_block_ptr`
        and `tl.store`. If False, it will use a more traditional epilogue by
        calculating destination pointers manually with `tl.arange` and `tl.store`.
    """
    # Your code here




