# test_cast_matmul

The on-disk starting functions contain implemented Triton code. The v2 declaration
therefore uses `implemented` and a frozen `initial_candidate` baseline, regardless
of this directory's historical suite name. Edit only the declared function scopes
(and permitted implementation helpers) in `test_cast_matmul.py`. Preserve signatures,
references, input generation, assertions, test parameters and timing policy.
Deliver edits to those files; a fenced code block alone is not a submission.

## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
with one role and one action. Submitted checks use `ARENA_EVAL_PHASE=candidate_evaluation`.
The adapter emits `arena-eval-v1`; Arena owns final score/validation reports.
`workloads.json` retains 54 original collected cases, including 18 performance cases.
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


You are an expert in triton programming language. You will be given the function definition for the `matmul_kernel`. Your task is to complete the kernel code. Only complete the kernel code in the function definition, DONT remove any python imports or helper utils in the instruction/code provided, DONT change/interfere with the provided function definition and parameter list.

This kernel, `matmul_kernel`,  is designed to perform a matrix multiplication (C = A @ B) using a tiled approach.

**Your objective is to implement the body of `matmul_kernel`.**

You must ensure that:
1.  All arguments received by `matmul_kernel` are kept intact and not modified.
2. Provide you final code in ```python code block. 
Example:
```python
<YOUR-CODE-HERE>
```
The full definition for `matmul_kernel` and relevant helper utilities are provided in the context below. You only need to complete the code for `matmul_kernel` whilst keeping other things intact.

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



