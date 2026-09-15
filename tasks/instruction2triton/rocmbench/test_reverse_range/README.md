# test_reverse_range

The on-disk starting functions contain implemented Triton code. The v2 declaration
therefore uses `implemented` and a frozen `initial_candidate` baseline, regardless
of this directory's historical suite name. Edit only the declared function scopes
(and permitted implementation helpers) in `test_reverse_range.py`. Preserve signatures,
references, input generation, assertions, test parameters and timing policy.
Deliver edits to those files; a fenced code block alone is not a submission.

## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
with one role and one action. Submitted checks use `ARENA_EVAL_PHASE=candidate_evaluation`.
The adapter emits `arena-eval-v1`; Arena owns final score/validation reports.
`workloads.json` retains 10 original collected cases, including 9 performance cases.
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


You are an expert in triton programming language. You will be given the function definition for the `reverse_range`. Your task is to complete the kernel code. Only complete the kernel code in the function definition, DONT remove any python imports or helper utils in the instruction/code provided, DONT change/interfere with the provided function definition and parameter list.

This kernel, `reverse_range`,  is designed to reverse a specific 512-element segment from an input tensor and stores it into an output tensor.

**Your objective is to implement the body of `reverse_range`.**

You must ensure that:
1.  All arguments received by `reverse_range` are kept intact and not modified.
2. Provide you final code in ```python code block.
Example:
```python
<YOUR-CODE-HERE>
```
The full definition for `reverse_range` and relevant helper utilities are provided in the context below. You only need to complete the code for `reverse_range` whilst keeping other things intact. DONT remove Imports and HELPER utils.

######################################## Imports ########################################
import numpy as np
import pytest
import torch
from numpy.random import RandomState

import triton
import triton.language as tl
######################################## Imports ########################################

@triton.jit
def reverse_range(in_ptr, out_ptr):
    """
    Reverses a specific 512-element segment from an input tensor and stores it
    into an output tensor.

    This kernel operates on fixed-size blocks of 512 elements.
    It performs the following operation for each element `i` in the range `[0, 511]`:
    `out_ptr[i] = in_ptr[512 - i]`

    This means the kernel reads elements from `in_ptr + 1` up to `in_ptr + 512`
    (inclusive) and writes them in reversed order to `out_ptr + 0` up to
    `out_ptr + 511` (inclusive).

    Parameters
    ----------
    in_ptr : tl.pointer_type
        A pointer to the input tensor. The kernel reads a block of 512 elements
        from this tensor. Specifically, it reads from memory locations
        `in_ptr + 512` down to `in_ptr + 1`. For example, the value at
        `in_ptr + 512` is read first (for `x0=0` in the original implementation)
        and stored at `out_ptr + 0`.
    out_ptr : tl.pointer_type
        A pointer to the output tensor. The kernel writes the 512 reversed
        elements to this tensor. Specifically, it writes to memory locations
        `out_ptr + 0` through `out_ptr + 511`. For example, `out_ptr + 0`
        receives the value from `in_ptr + 512`.
    """
    # Your code here
