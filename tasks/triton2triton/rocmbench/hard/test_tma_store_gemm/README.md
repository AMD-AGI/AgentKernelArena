# test_tma_store_gemm

The on-disk starting functions contain implemented Triton code. The v2 declaration
therefore uses `implemented` and a frozen `initial_candidate` baseline, regardless
of this directory's historical suite name. Edit only the declared function scopes
(and permitted implementation helpers) in `test_tma_store_gemm.py`. Preserve signatures,
references, input generation, assertions, test parameters and timing policy.
Deliver edits to those files; a fenced code block alone is not a submission.

## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
with one role and one action. Submitted checks use `ARENA_EVAL_PHASE=candidate_evaluation`.
The adapter emits `arena-eval-v1`; Arena owns final score/validation reports.
`workloads.json` retains 19 original collected cases, including 11 performance cases.
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


You are an expert in triton programming language. You will be given the function definition for the `matmul_tma_load_store`. Your task is to complete the kernel code. Only optimize the kernel code in the function definition, DONT remove any python imports or helper utils in the instruction/code provided, DONT change/interfere with the provided function definition and parameter list.

This kernel, `matmul_tma_load_store`,  performs a single block matrix multiplication (C = A @ B) using Triton's block pointers using TMA (Tensor Memory Accelerator).

**Your objective is to optimize the body of `matmul_tma_load_store`.**

You must ensure that:
1.  All arguments received by `matmul_tma_load_store` are kept intact and not modified.
2. Provide you final code in ```python code block.
Example:
```python
<YOUR-CODE-HERE>
```
The full definition for `matmul_tma_load_store` and relevant helper utilities are provided in the context below. You only need to optimize the code for `matmul_tma_load_store` whilst keeping other things intact. DONT remove Imports and HELPER utils.

######################################## Imports ########################################

import pytest
import torch
from torch.testing import assert_close

import triton
import triton.language as tl
######################################## Imports ########################################


@triton.jit
def matmul_tma_load_store(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    OUTPUT_F16: tl.constexpr
):
    """
    Performs a single block matrix multiplication (C = A @ B) using Triton's
    block pointers, potentially leveraging TMA (Tensor Memory Accelerator)
    for efficient loads and stores on compatible hardware.

    This kernel is designed to compute one `BLOCK_M x BLOCK_N` tile of the output matrix C.
    Specifically, it loads a `BLOCK_M x BLOCK_K` tile from matrix A (starting from `a_ptr`
    at offset (0,0)) and a `BLOCK_K x BLOCK_N` tile from matrix B (starting from `b_ptr`
    at offset (0,0)). It then computes their dot product and stores the resulting
    `BLOCK_M x BLOCK_N` tile into matrix C (starting at `c_ptr` at offset (0,0)).

    The kernel uses `tl.load` and `tl.store` with block pointers configured as follows:
    - Matrix A's tile is loaded assuming a row-major layout within the block (`order=(1,0)`).
    - Matrix B's tile is loaded assuming a column-major layout within the block (`order=(0,1)`),
      which is often beneficial for dot product operations.
    - Matrix C's tile is stored assuming a row-major layout within the block (`order=(1,0)`).

    Input matrices A and B are expected to have data types suitable for `tl.dot`
    (e.g., tl.float16, tl.bfloat16, tl.float32). The accumulation for the dot
    product is typically performed in tl.float32.

    Args:
        a_ptr: Pointer to the base of the input matrix A in global memory.
        b_ptr: Pointer to the base of the input matrix B in global memory.
        c_ptr: Pointer to the base of the output matrix C in global memory.
        M (int): The total number of rows in the full matrix A and matrix C. Used for boundary checks.
        N (int): The total number of columns in the full matrix B and matrix C. Used for boundary checks.
        K (int): The total number of columns in the full matrix A and rows in matrix B
                 (the common dimension for matrix multiplication). Used for boundary checks.
        stride_am (int): Stride in number of elements to move from one row to the next in matrix A.
        stride_ak (int): Stride in number of elements to move from one column to the next in matrix A.
        stride_bk (int): Stride in number of elements to move from one row to the next in matrix B.
        stride_bn (int): Stride in number of elements to move from one column to the next in matrix B.
        stride_cm (int): Stride in number of elements to move from one row to the next in matrix C.
        stride_cn (int): Stride in number of elements to move from one column to the next in matrix C.
        BLOCK_M (tl.constexpr): The height (number of rows) of the tile to be processed from matrix A
                                and written to matrix C. This defines the M-dimension of the block.
        BLOCK_N (tl.constexpr): The width (number of columns) of the tile to be processed from matrix B
                                and written to matrix C. This defines the N-dimension of the block.
        BLOCK_K (tl.constexpr): The width (number of columns) of the tile from matrix A, and height
                                (number of rows) of the tile from matrix B. This defines the
                                K-dimension of the blocks used in the dot product.
        OUTPUT_F16 (tl.constexpr): A boolean flag. If True, the resulting C tile is cast to
                                   `tl.float16` before being stored. Otherwise, it is stored
                                   in the accumulation data type (typically `tl.float32`).
    """
    # Your code here
