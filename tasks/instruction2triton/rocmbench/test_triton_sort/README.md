# test_triton_sort

The on-disk starting functions contain implemented Triton code. The v2 declaration
therefore uses `implemented` and a frozen `initial_candidate` baseline, regardless
of this directory's historical suite name. Edit only the declared function scopes
(and permitted implementation helpers) in `test_triton_sort.py`. Preserve signatures,
references, input generation, assertions, test parameters and timing policy.
Deliver edits to those files; a fenced code block alone is not a submission.

## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
with one role and one action. Submitted checks use `ARENA_EVAL_PHASE=candidate_evaluation`.
The adapter emits `arena-eval-v1`; Arena owns final score/validation reports.
`workloads.json` retains 120 original collected cases, including 88 performance cases.
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


You are an expert in triton programming language. You will be given the function definition for the `sort_kernel`. Your task is to complete the kernel code. Only complete the kernel code in the function definition, DONT remove any python imports or helper utils in the instruction/code provided, DONT change/interfere with the provided function definition and parameter list.

This kernel, `sort_kernel`,  Sorts each row of a 2D input tensor independently.

**Your objective is to implement the body of `sort_kernel`.**

You must ensure that:
1.  All arguments received by `sort_kernel` are kept intact and not modified.
2. Provide you final code in ```python code block.
Example:
```python
<YOUR-CODE-HERE>
```
The full definition for `sort_kernel` and relevant helper utilities are provided in the context below. You only need to complete the code for `sort_kernel` whilst keeping other things intact. DONT remove Imports and HELPER utils.

######################################## Imports ########################################
import pytest
import torch
import triton
import triton.language as tl
import numpy as np
######################################## Imports ########################################

@triton.jit
def sort_kernel(X, Z, N: tl.constexpr, M: tl.constexpr, descending: tl.constexpr):
    """
    Sorts each row of a 2D input tensor independently.

    This kernel loads a 2D block of data of shape (N, M) from the input tensor X.
    It then sorts each of the N rows (each of length M) independently.
    The sorted rows are stored in the output tensor Z.
    The sorting order (ascending or descending) is determined by the `descending` flag.

    Parameters
    ----------
    X : tl.pointer_type
        Pointer to the input tensor. The kernel expects to read a 2D block of data
        logically arranged as N rows and M columns.
    Z : tl.pointer_type
        Pointer to the output tensor. The sorted 2D block of data (N rows, M columns)
        will be stored here.
    N : tl.constexpr
        A compile-time constant representing the number of rows in the 2D block to be processed.
        Each of these N rows will be sorted independently.
    M : tl.constexpr
        A compile-time constant representing the number of columns (elements per row)
        in the 2D block. Sorting is performed along this dimension for each row.
    descending : tl.constexpr
        A compile-time constant boolean. If True, each row is sorted in descending order.
        If False, each row is sorted in ascending order.
    """

    # Your code here
