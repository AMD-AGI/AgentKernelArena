# test_random_int

The on-disk starting functions contain implemented Triton code. The v2 declaration
therefore uses `implemented` and a frozen `initial_candidate` baseline, regardless
of this directory's historical suite name. Edit only the declared function scopes
(and permitted implementation helpers) in `test_random_int.py`. Preserve signatures,
references, input generation, assertions, test parameters and timing policy.
Deliver edits to those files; a fenced code block alone is not a submission.

## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
with one role and one action. Submitted checks use `ARENA_EVAL_PHASE=candidate_evaluation`.
The adapter emits `arena-eval-v1`; Arena owns final score/validation reports.
`workloads.json` retains 136 original collected cases, including 64 performance cases.
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


You are an expert in triton programming language. You will be given the function definition for the `randint_kernel_runtime_seed,randint_kernel_const_seed` kernels. Your task is to complete the kernel code. Only complete the kernel code in the function definition, DONT remove any python imports or helper utils in the instruction/code provided, DONT change/interfere with the provided function definition and parameter list.

These kernels, `randint_kernel_runtime_seed,randint_kernel_const_seed`,  tests both ways Triton can handle arguments: Ensuring that tl.randint works correctly whether the seed is provided as a runtime variable or a compile-time constant

**Your objective is to implement the body of both the kernels `randint_kernel_runtime_seed,randint_kernel_const_seed`.**

You must ensure that:
1.  All arguments received by `randint_kernel_runtime_seed and randint_kernel_const_seed` are kept intact and not modified.
2. Provide you final code in ```python code block.
Example:
```python
<YOUR-CODE-HERE>
```
The full definitions for `randint_kernel_runtime_seed,randint_kernel_const_seed` and relevant helper utilities are provided in the context below. You only need to complete the code for `randint_kernel_runtime_seed,randint_kernel_const_seed` whilst keeping other things intact. DONT remove Imports and HELPER utils.

######################################## Imports ########################################
import numpy as np
import pytest
import torch

import triton
import triton.language as tl
BLOCK: tl.constexpr = 1024
######################################## Imports ########################################

#####################################
# Triton Kernels for randint
#####################################

@triton.jit
def randint_kernel_runtime_seed(X, N, seed_val):
    """
    Generates random integers and stores them in the output tensor X,
    using a seed value provided as a runtime argument.

    Each program instance computes a block of random numbers.
    The kernel is designed to test the tl.randint function with a runtime seed.

    Args:
        X (tl.tensor): Pointer to the output tensor where random integers will be stored.
                       The data type of X's elements is used for pid consistency if X is an integer type.
        N (int): The total number of random integers to generate. This is used to mask
                 out-of-bounds writes.
        seed_val (int): The seed value for the random number generator, passed as a runtime argument.
                        This seed is used by tl.randint to initialize its state.
    """
    # Your code here


@triton.jit
def randint_kernel_const_seed(X, N, seed_val: tl.constexpr):
    """
    Generates random integers and stores them in the output tensor X,
    using a seed value provided as a compile-time constant.

    Each program instance computes a block of random numbers.
    The kernel is designed to test the tl.randint function with a compile-time constant seed.

    Args:
        X (tl.tensor): Pointer to the output tensor where random integers will be stored.
                       The data type of X's elements is used for pid consistency if X is an integer type.
        N (int): The total number of random integers to generate. This is used to mask
                 out-of-bounds writes.
        seed_val (tl.constexpr): The seed value for the random number generator, passed as a
                                 compile-time constant. This seed is used by tl.randint
                                 to initialize its state.
    """
    # Your code here





The additional RNG oracle follows the public [Triton Philox counter and uniform conversion definitions](https://github.com/triton-lang/triton/blob/main/python/triton/language/random.py). Original statistical/exact checks remain active.
