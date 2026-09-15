# test_randn

The on-disk starting functions contain implemented Triton code. The v2 declaration
therefore uses `implemented` and a frozen `initial_candidate` baseline, regardless
of this directory's historical suite name. Edit only the declared function scopes
(and permitted implementation helpers) in `test_randn.py`. Preserve signatures,
references, input generation, assertions, test parameters and timing policy.
Deliver edits to those files; a fenced code block alone is not a submission.

## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
with one role and one action. Submitted checks use `ARENA_EVAL_PHASE=candidate_evaluation`.
The adapter emits `arena-eval-v1`; Arena owns final score/validation reports.
`workloads.json` retains 120 original collected cases, including 104 performance cases.
Collection is checked against this independent manifest. The 16 original statistical cases retain their range and KS assertions and now
also compare every element with the independent Philox oracle in
`_arena_reference.py`. The same exact oracle covers all 104 performance inputs,
before timing, against the actual timed output, and on bound replay after
output poisoning. Output state is restored in `finally`.
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

The actual operation is **uniform `tl.rand(seed, offset)` in [0, 1)**, not
normally distributed `tl.randn`; `randn` in the filename and kernel names is a
historical label. For every element the required value is Triton's Philox4x32-10
at the original scalar seed and linear offset, followed by the float32 uniform
conversion. Keep both runtime-seed and constexpr-seed interfaces intact.

The original 100000-element tests keep seeds 0, 42, 124, 54, int32/int64 indexing,
both seed bindings, inclusive range assertion and KS statistic < 0.01. The 104
scored cases keep sizes 2**10 through 2**22, seeds 0/42, both index widths and
seed bindings, four warps, ten warmups and 100 samples. No case/seed was removed
or substituted. Statistical plausibility alone cannot establish the seeded
sequence: permuting a correct sequence preserves KS but must fail exact checking.

Output-poisoned replay uses the **same seed and same bound timed invocation**.
It checks full output overwrite and exact values. It does not change scalar
arguments inside a captured graph or by itself rule out copying a cached correct
sequence. Any graph path that cannot expose its timed invocation fails; explicit
observable event timing remains supported. Oracles, poisoning and restoration
are outside timing. Baseline and candidate use the same policy.

The historical interface description below is subordinate to the actual source
signatures and v2 submission contract above (including on-disk submission).


You are an expert in triton programming language. You will be given the function definition for the `randn_kernel_runtime_seed,randn_kernel_const_seed` kernels. Your task is to optimize the kernel code for better performance while preserving correctness. Only complete the kernel code in the function definition, DONT remove any python imports or helper utils in the instruction/code provided, DONT change/interfere with the provided function definition and parameter list.

These kernels, `randn_kernel_runtime_seed,randn_kernel_const_seed`,  tests both ways Triton can handle arguments: Ensuring that tl.rand works correctly whether the seed is provided as a runtime variable or a compile-time constant

**Your objective is to optimize the body of both the kernels `randn_kernel_runtime_seed,randn_kernel_const_seed`.**

You must ensure that:
1.  All arguments received by `randn_kernel_runtime_seed and randn_kernel_const_seed` are kept intact and not modified.
2. Provide you final code in ```python code block. 
Example:
```python
<YOUR-CODE-HERE>
```
The full definitions for `randn_kernel_runtime_seed,randn_kernel_const_seed` and relevant helper utilities are provided in the context below. You only need to optimize the code for `randn_kernel_runtime_seed,randn_kernel_const_seed` whilst keeping other things intact. DONT remove Imports and HELPER utils.

######################################## Imports ######################################## 
import numpy as np
import pytest
import torch

import triton
import triton.language as tl
######################################## Imports ######################################## 

#####################################
# Triton Kernels for randn
#####################################

BLOCK: tl.constexpr = 1024

@triton.jit
def randn_kernel_runtime_seed(X: tl.pointer_type, N: tl.int32, seed: tl.int66, dtype: tl.constexpr):
    """
    Generates random numbers and stores them into an output tensor.
    The seed for the random number generator is passed at runtime.

    :param X: Pointer to the output tensor where random numbers will be stored (e.g., *tl.float32).
    :param N: Total number of elements to generate.
    :param seed: Seed for the random number generator (runtime value, typically int64 or int66).
    :param dtype: Data type used for kernel indexing/calculations (compile-time constant, e.g., tl.int32).
    """
    # Your code here


@triton.jit
def randn_kernel_const_seed(X: tl.pointer_type, N: tl.int32, seed: tl.constexpr, dtype: tl.constexpr):
    """
    Generates random numbers and stores them into an output tensor.
    The seed for the random number generator must be a compile-time constant.

    :param X: Pointer to the output tensor where random numbers will be stored (e.g., *tl.float32).
    :param N: Total number of elements to generate.
    :param seed: Seed for the random number generator (must be a compile-time constant).
    :param dtype: Data type used for kernel indexing/calculations (compile-time constant, e.g., tl.int32).
    """
    # Your code here





The additional RNG oracle follows the public [Triton Philox counter and uniform conversion definitions](https://github.com/triton-lang/triton/blob/4cff872ced001ea92d9fcf05b3f6517e2b486d19/python/triton/language/random.py). Original statistical/exact checks remain active.

The independent CPU regression also checks the [Random123 Philox4x32-10 known answers](https://github.com/DEShawResearch/random123/blob/v1.14.0/tests/kat_vectors). Task evaluation is self-contained and does not download either upstream source.
