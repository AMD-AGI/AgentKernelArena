# test_kernel_sub

The on-disk starting functions contain implemented Triton code. The v2 declaration
therefore uses `implemented` and a frozen `initial_candidate` baseline, regardless
of this directory's historical suite name. Edit only the declared function scopes
(and permitted implementation helpers) in `test_kernel_sub.py`. Preserve signatures,
references, input generation, assertions, test parameters and timing policy.
Deliver edits to those files; a fenced code block alone is not a submission.

## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
with one role and one action. Submitted checks use `ARENA_EVAL_PHASE=candidate_evaluation`.
The adapter emits `arena-eval-v1`; Arena owns final score/validation reports.
`workloads.json` retains 49 original collected cases, including 42 performance cases.
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


You are an expert in triton programming language. You will be given the function definition for the `kernel_sub`. Your task is to complete the kernel code. Only complete the kernel code in the function definition, DONT remove any python imports or helper utils in the instruction/code provided, DONT change/interfere with the provided function definition and parameter list.

This kernel, `kernel_sub`,  performs  element-wise operation: output[i] = a[i] - (b[i] * 777)

**Your objective is to implement the body of `kernel_sub`.**

You must ensure that:
1.  All arguments received by `kernel_sub` are kept intact and not modified.
2. Provide you final code in ```python code block. 
Example:
```python
<YOUR-CODE-HERE>
```
The full definition for `kernel_sub` and relevant helper utilities are provided in the context below. You only need to complete the code for `kernel_sub` whilst keeping other things intact. DONT remove Imports and HELPER utils.

######################################## Imports ########################################
import multiprocessing
import shutil
import tempfile
import os
import pytest

import triton
import triton.language as tl
from triton.backends.compiler import AttrsDescriptor
from triton.compiler import ASTSource
######################################## Imports ########################################

import triton
import triton.language as tl

@triton.jit
def kernel_sub(a, b, o, N: tl.constexpr):
    """
    Performs an element-wise operation: output[i] = a[i] - (b[i] * 777)
    for N elements.

    Parameters
    ----------
    a
        Pointer to the first input tensor (minuend).
        Its elements are loaded and used in the subtraction.
    b
        Pointer to the second input tensor (subtrahend).
        Its elements are loaded, multiplied by 777, and then subtracted from 'a'.
    o
        Pointer to the output tensor.
        The results of a[i] - (b[i] * 777) are stored here.
    N : tl.constexpr
        A compile-time constant specifying the number of elements
        to process in tensors a, b, and o. This typically corresponds
        to the block size or a dimension of the tensors.
    """
    # Your code here




