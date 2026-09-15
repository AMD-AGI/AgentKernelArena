# Cast-before-matmul contract

The initial Triton kernel is implemented. Arena freezes it as the independent
`initial_candidate` baseline; final submissions must implement the declared
Triton symbol, with no baseline or reference fallback. Only the configured
`matmul_kernel` scope and permitted implementation helpers are editable.

Both A[M,K] and B[K,N] are cast to C's dtype **before** multiplication.
Input dtypes are float16/float32/float64; output dtypes float16/float32.
The declared dot accumulator dtype remains a separate kernel parameter.
Inputs are read-only; C is the complete M-by-N output, and the protected public
wrapper returns that buffer. All output elements, metadata and finiteness are
checked against the independent PyTorch product of private input snapshots.
The original numerical rule remains `atol=0.3, rtol=0.01`.

## Cases and timing

`workloads.json` retains all 54 original collected cases (36 functional and
18 scored performance cases), their identities and parameters. Three additional unscored controls cover odd
M/N/K tails, partial GROUP_M, stride-two inputs, output prefix/padding and
noncontiguous output strides, yielding 57 total correctness cases. The original
same-input-dtype skips were a test selection preference, not an invalid kernel
argument: conversion is defined when operands share a dtype too. Those 24 rows
now execute real kernels and original numerical checks; none is a skipped pass.
No shape, seed, kernel, block/grid parameter or accumulator mode was changed.
Original functional assertions remain, with input-integrity checks and restoration.

Every performance input is checked, then the actual canonical `TimedRun` output
is checked in full. Untimed replay reverses/perturbs private input values,
recomputes the oracle, poisons C with NaNs, and replays the captured invocation.
Inputs and output are restored in `finally`, including failure paths. Timed work
remains the original preallocated-output wrapper, warmup10/repetition100 and mean
device latency with canonical graph/event fallback. The optional old PyTorch
peer benchmark is not Arena's frozen baseline and is not scored. Do not edit
`performance_utils_pytest.py` or generated helpers. Newly executed formerly
skipped rows have no historical timing; do not claim an unchanged historical score.

## Running checks

Use `python3 _arena_eval.py validate-task`, or
`python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(one role/action). Final submission checks use `ARENA_EVAL_PHASE=candidate_evaluation`.
Every action emits `arena-eval-v1`; only Arena writes final score/validation reports.
Deliver files under the configured edit boundary, not just a fenced code block.
Missing kernels, altered manifests, skips, failed checks and invalid timing fail.

## Original operator instructions

The historical instructions below retain the operator semantics and interface;
the v2 on-disk edit and evaluation contract above governs submission format.


You are an expert in triton programming language. You will be given the function definition for the `matmul_kernel`. Your task is to complete the kernel code. Only optimize the kernel code in the function definition, DONT remove any python imports or helper utils in the instruction/code provided, DONT change/interfere with the provided function definition and parameter list.

This kernel, `matmul_kernel`,  is designed to perform a matrix multiplication (C = A @ B) using a tiled approach.

**Your objective is to optimize the body of `matmul_kernel`.**

You must ensure that:
1.  All arguments received by `matmul_kernel` are kept intact and not modified.
2. Provide you final code in ```python code block. 
Example:
```python
<YOUR-CODE-HERE>
```
The full definition for `matmul_kernel` and relevant helper utilities are provided in the context below. You only need to optimize the code for `matmul_kernel` whilst keeping other things intact.

######################################## Imports#######################################
import pytest
import torch

import triton
import triton.language as tl
######################################## Imports#######################################

######################################## HELPERS utils ########################################
input_dtypes = ["float16", "float32", "float64"]
out_dtypes = ["float16", "float32"]
######################################## HELPERS utils ########################################


@triton.jit
def matmul_kernel(
    A,  # Pointer to the first input matrix (M x K).
    B,  # Pointer to the second input matrix (K x N).
    C,  # Pointer to the output matrix (M x N). The element type of C also dictates the type to which input tiles `a` and `b` are cast before `tl.dot`.
    M,  # Number of rows in matrix A and C.
    N,  # Number of columns in matrix B and C.
    K,  # Number of columns in matrix A and rows in matrix B (the shared dimension).
    stride_am,  # Stride for matrix A along the M dimension (row stride).
    stride_ak,  # Stride for matrix A along the K dimension (column stride).
    stride_bk,  # Stride for matrix B along the K dimension (row stride).
    stride_bn,  # Stride for matrix B along the N dimension (column stride).
    stride_cm,  # Stride for matrix C along the M dimension (row stride).
    stride_cn,  # Stride for matrix C along the N dimension (column stride).
    dot_out_dtype: tl.constexpr,  # The data type used for the accumulator in the `tl.dot` operation. This is a compile-time constant.
    BLOCK_M: tl.constexpr,  # Tile size for the M dimension (rows per block). This is a compile-time constant.
    BLOCK_N: tl.constexpr,  # Tile size for the N dimension (columns per block). This is a compile-time constant.
    BLOCK_K: tl.constexpr,  # Tile size for the K dimension (inner dimension per block). This is a compile-time constant.
    GROUP_M: tl.constexpr,  # Grouping factor for the M dimension to improve L2 cache performance. This is a compile-time constant.
):
    """
    Performs a matrix multiplication (C = A @ B) using a tiled approach.

    This kernel is designed to test mixed precision capabilities, specifically focusing
    on how `tl.dot` interacts with `tl.to` (cast) operations.
    Input tiles from matrices A and B are loaded, then explicitly cast to the
    element type of the output matrix C before the `tl.dot` operation.
    The accumulation within `tl.dot` is performed using the specified `dot_out_dtype`.
    Finally, the accumulated tile is cast to the element type of matrix C before
    being stored.

    The tiling strategy involves dividing the M and N dimensions into blocks of
    size `BLOCK_M` and `BLOCK_N` respectively. The K dimension is processed in
    chunks of `BLOCK_K`. Program IDs are re-ordered using `GROUP_M` to
    potentially improve L2 cache locality.
    """
    # Your code here



