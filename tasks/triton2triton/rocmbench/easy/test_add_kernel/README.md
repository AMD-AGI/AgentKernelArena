# test_add_kernel

The on-disk starting functions contain implemented Triton code. The v2 declaration
therefore uses `implemented` and a frozen `initial_candidate` baseline, regardless
of this directory's historical suite name. Edit only the declared function scopes
(and permitted implementation helpers) in `test_add_kernel.py`. Preserve signatures,
references, input generation, assertions, test parameters and timing policy.
Deliver edits to those files; a fenced code block alone is not a submission.

## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
with one role and one action. Submitted checks use `ARENA_EVAL_PHASE=candidate_evaluation`.
The adapter emits `arena-eval-v1`; Arena owns final score/validation reports.
`workloads.json` retains 6 original collected cases, including 4 performance cases.
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


You are an expert in triton programming language. You will be given the function definition for the `add_kernel` kernels. Your task is to complete the kernel code. Only optimize the kernel code in the function definition, DONT remove any python imports or helper utils in the instruction/code provided, DONT change/interfere with the provided function definition and parameter list.

These kernels, `add_kernel`,  performs element-wise addition of two 1D tensors.

**Your objective is to optimize the body of both the kernels `add_kernel`.**

You must ensure that:
1.  All arguments received by `matmul_kernel and mxfp_to_bf16_kernel` are kept intact and not modified.
2. Provide you final code in ```python code block. 
Example:
```python
<YOUR-CODE-HERE>
```
The full definitions for `add_kernel` and relevant helper utilities are provided in the context below. You only need to optimize the code for `add_kernel` whilst keeping other things intact. DONT remove Imports and HELPER utils.

######################################## Imports ########################################
import pytest
import torch
from torch.testing import assert_close

import triton
import triton.language as tl

dtype_mapping = {
    'float16': torch.float16,
    'float32': torch.float32,
}
######################################## Imports ########################################


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Performs element-wise addition of two 1D tensors, `x` and `y`,
    and stores the result in `output`. Boundary checks are handled using a mask to ensure that
    operations are only performed on valid elements within the tensor's
    `n_elements` bounds.

    Parameters
    ----------
    x_ptr
        Pointer to the first input tensor (x).
        The elements from this tensor will be added.
    y_ptr
        Pointer to the second input tensor (y).
        The elements from this tensor will be added.
    output_ptr
        Pointer to the output tensor where the result of x + y is stored.
    n_elements
        The total number of elements in the input and output tensors.
        This is used to ensure that memory accesses are within bounds.
    BLOCK_SIZE : tl.constexpr
        The size of the block that each program instance will process.
        This is a compile-time constant and dictates how many elements
        are loaded, processed, and stored together by a single program
        instance in the Triton kernel. It should typically be a power of two.
    """
    # Your code here




