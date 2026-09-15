# No-SCF single-tile GEMM

The implemented initial Triton kernel is frozen independently as the baseline.
Edit only the declared `matmul_no_scf_kernel` and permitted helpers. A missing
candidate cannot delegate to or fall back to the baseline/reference.

A[M,K] and B[K,N] are read-only float16 matrices; the output C is float16 or
float32 as declared. The kernel computes the complete single-tile product,
including the pointer-store and block-pointer epilogue modes. Input strides
may reflect transposes. The independent oracle uses **private float32 input
copies** and the original `atol=1e-3, rtol=1e-2, check_dtype=False` numeric rule.
Output dtype is separately required to match the declared C buffer, alongside
shape/device/stride, non-aliasing, finiteness and full-value checks.

## Manifest and unsupported arguments

All 52 original case IDs/parameters remain (32 functional, 20 scored); four
additional unscored cases cover transposed A with both output types and
both epilogues. Original functional rows with NUM_CTAS=4 previously skipped on
HIP. When the actual backend reports multi-CTA launch unsupported, those eight
rows now invoke the real candidate and require precisely its backend ValueError
`num_ctas > 1 not supported on <arch>`. Success, another exception/message, or
input/output mutation fails. This tests rejection, not a skipped numerical pass.
If a backend supports multi-CTA, the same row executes normal numerical checks.
On the pinned gfx950 runtime the eight are rejection controls. Every scored
case remains NUM_CTAS=1 and must run complete numerical and timing checks.

## Replay and timing

The original preallocated-output wrapper, launch arguments, seed42,
warmup10/repetition100, canonical graph/event fallback and mean device latency
are preserved. Every performance input is numerically checked, as is the actual
`TimedRun` output. Untimed replay changes valid floating input values while
preserving pointers/layouts, recomputes the private oracle, poisons C, and replays
the timed invocation. Original inputs/output are restored in finally, including
failure paths. Both frozen baseline and candidate use identical work and cases.
Original functional oracle evaluation is protected from candidate input mutation.
The optional historical PyTorch peer timing is not Arena's independent baseline.
Do not edit helper stubs or generated performance helpers.

Use `python3 _arena_eval.py validate-task`, or
`python3 _arena_eval.py baseline|candidate compile|correctness|performance`.
The task emits `arena-eval-v1`; Arena alone writes official reports/scores.
Final submitted checks use `ARENA_EVAL_PHASE=candidate_evaluation`. Deliver files
within the configured edit boundary, not merely a fenced code block.

## Original operator instructions

The historical instructions below retain the operator semantics and interface;
the v2 on-disk edit and evaluation contract above governs submission format.


You are an expert in triton programming language. You will be given the function definition for the `matmul_no_scf_kernel` kernels. Your task is to complete the kernel code. Only complete the kernel code in the function definition, DONT remove any python imports or helper utils in the instruction/code provided, DONT change/interfere with the provided function definition and parameter list.

This kernel, `matmul_no_scf_kernel`,  performs single block of matrix multiplication (C = A @ B) without Structured Control Flow (SCF)
**Your objective is to implement the body of both the kernels `matmul_no_scf_kernel`.**

You must ensure that:
1.  All arguments received by `matmul_no_scf_kernel` are kept intact and not modified.
2. Provide you final code in ```python code block. 
Example:
```python
<YOUR-CODE-HERE>
```
The full definitions for `matmul_no_scf_kernel` and relevant helper utilities are provided in the context below. You only need to complete the code for `matmul_no_scf_kernel` whilst keeping other things intact. DONT remove Imports and HELPER utils.

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




