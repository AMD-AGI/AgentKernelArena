# test_load_reduce

The on-disk starting functions contain implemented Triton code. The v2 declaration
therefore uses `implemented` and a frozen `initial_candidate` baseline, regardless
of this directory's historical suite name. Edit only the declared function scopes
(and permitted implementation helpers) in `test_load_reduce.py`. Preserve signatures,
references, input generation, assertions, test parameters and timing policy.
Deliver edits to those files; a fenced code block alone is not a submission.

## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
with one role and one action. Submitted checks use `ARENA_EVAL_PHASE=candidate_evaluation`.
The adapter emits `arena-eval-v1`; Arena owns final score/validation reports.
`workloads.json` retains 43 original collected cases, including 42 performance cases.
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


You are an expert in triton programming language. You will be given the function definition for the `load_reduce_kernel` kernels. Your task is to complete the kernel code. Only complete the kernel code in the function definition, DONT remove any python imports or helper utils in the instruction/code provided, DONT change/interfere with the provided function definition and parameter list.

These kernels, `load_reduce_kernel`,  performs a block-wise load followed by a row-wise maximum reduction.

**Your objective is to implement the body of both the kernels `load_reduce_kernel`.**

You must ensure that:
1.  All arguments received by `matmul_kernel and mxfp_to_bf16_kernel` are kept intact and not modified.
2. Provide you final code in ```python code block. 
Example:
```python
<YOUR-CODE-HERE>
```
The full definitions for `load_reduce_kernel` and relevant helper utilities are provided in the context below. You only need to complete the code for `load_reduce_kernel` whilst keeping other things intact. DONT remove Imports and HELPER utils.

######################################## Imports ########################################

import pytest
import torch
from torch.testing import assert_close

import triton
import triton.language as tl
######################################## Imports ########################################

dtype_mapping = {
    'float16': torch.float16,
    'float32': torch.float32,
}


@triton.jit
def load_reduce_kernel(
    x_ptr,
    y_ptr,
    stride_xm,
    stride_xn,
    stride_y,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    This Triton kernel loads a 2D block of data from an input tensor `x_ptr`
    and performs a reduction operation (maximum) along one of its dimensions.

    Parameters
    ----------
    x_ptr
        Pointer to the input tensor X from which data will be loaded.
        This tensor is expected to be at least 2D.
    y_ptr
        Pointer to the output tensor Y where the reduced results will be stored.
        This tensor is expected to be 1D (or have a shape compatible with storing BLOCK_M elements).
    stride_xm
        Stride of the input tensor X along the M dimension (typically rows).
        It indicates the number of elements to skip in memory to move from one
        element to the next in the M dimension (e.g., from X[i, j] to X[i+1, j]).
    stride_xn
        Stride of the input tensor X along the N dimension (typically columns).
        It indicates the number of elements to skip in memory to move from one
        element to the next in the N dimension (e.g., from X[i, j] to X[i, j+1]).
    stride_y
        Stride of the output tensor Y.
        It indicates the number of elements to skip in memory to move from one
        element to the next in the output tensor Y (e.g., from Y[i] to Y[i+1]).
    BLOCK_M: tl.constexpr
        The size of the tile (or block) in the M dimension. This defines how many
        "rows" of the input data are processed by this kernel instance and consequently
        the number of output elements produced.
    BLOCK_N: tl.constexpr
        The size of the tile (or block) in the N dimension. This defines how many
        "columns" of the input data are processed for each "row" in the M dimension,
        and it's the dimension over which the reduction (max) is performed.
    """
    # Your code here




