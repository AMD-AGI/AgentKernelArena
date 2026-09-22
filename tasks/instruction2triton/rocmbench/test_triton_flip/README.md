# test_triton_flip

The on-disk starting functions contain implemented Triton code. The v2 declaration
therefore uses `implemented` and a frozen `initial_candidate` baseline, regardless
of this directory's historical suite name. Edit only the declared function scopes
(and permitted implementation helpers) in `test_triton_flip.py`. Preserve signatures,
references, input generation, assertions, test parameters and timing policy.
Deliver edits to those files; a fenced code block alone is not a submission.

## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
with one role and one action. Submitted checks use `ARENA_EVAL_PHASE=candidate_evaluation`.
The adapter emits `arena-eval-v1`; Arena owns final score/validation reports.
`workloads.json` retains 40 original collected cases, including 24 performance cases.
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


You are an expert in triton programming language. You will be given the function definition for the `flip_kernel`. Your task is to complete the kernel code. Only complete the kernel code in the function definition, DONT remove any python imports or helper utils in the instruction/code provided, DONT change/interfere with the provided function definition and parameter list.

This kernel, `flip_kernel`,  Flips each row of a 2D tensor horizontally.

**Your objective is to implement the body of `flip_kernel`.**

You must ensure that:
1.  All arguments received by `flip_kernel` are kept intact and not modified.
2. Provide you final code in ```python code block.
Example:
```python
<YOUR-CODE-HERE>
```
The full definition for `flip_kernel` and relevant helper utilities are provided in the context below. You only need to complete the code for `flip_kernel` whilst keeping other things intact. DONT remove Imports and HELPER utils.

######################################## Imports ########################################
import pytest
import torch
import triton
import triton.language as tl
import numpy as np
######################################## Imports ########################################

@triton.jit
def flip_kernel(X, Z, N: tl.constexpr, M: tl.constexpr):
    """
    Processes 2D blocks of data, flipping each block horizontally.

    This kernel loads a 2D block of data of shape (N, M) from an input tensor `X`.
    It then flips this block along its second dimension (columns), meaning each
    row within the block is reversed. The resulting flipped block is then
    stored into an output tensor `Z` at the corresponding offset.

    Parameters
    ----------
    X
        Pointer to the input tensor. Each kernel instance will load an (N, M) block from this tensor.
    Z
        Pointer to the output tensor. Each kernel instance will store the flipped (N, M) block to this tensor.
        It can be the same as X for an in-place operation if memory layout and access patterns allow.
    N : tl.constexpr
        A compile-time constant specifying the size of the first dimension
        (e.g., number of rows) of the 2D data block to be processed by each kernel instance.
    M : tl.constexpr
        A compile-time constant specifying the size of the second dimension
        (e.g., number of columns) of the 2D data block to be processed by each kernel instance.
        The flip operation occurs along this dimension.
    """
    # Your code here
