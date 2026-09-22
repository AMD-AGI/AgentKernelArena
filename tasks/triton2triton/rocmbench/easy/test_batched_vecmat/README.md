# test_batched_vecmat

The on-disk starting functions contain implemented Triton code. The v2 declaration
therefore uses `implemented` and a frozen `initial_candidate` baseline, regardless
of this directory's historical suite name. Edit only the declared function scopes
(and permitted implementation helpers) in `test_batched_vecmat.py`. Preserve signatures,
references, input generation, assertions, test parameters and timing policy.
Deliver edits to those files; a fenced code block alone is not a submission.

## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
with one role and one action. Submitted checks use `ARENA_EVAL_PHASE=candidate_evaluation`.
The adapter emits `arena-eval-v1`; Arena owns final score/validation reports.
`workloads.json` retains 31 original collected cases, including 30 performance cases.
Collection is checked against this independent manifest. Original NumPy correctness
case and gate are retained, with private input snapshots. Performance inputs additionally run the task-local
oracle in `_arena_reference.py`, before timing, against actual TimedRun output and after fresh-input replay.
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


You are an expert in triton programming language. You will be given the function definition for the `batched_vecmat`. Your task is to complete the kernel code. Only optimize the kernel code in the function definition, DONT remove any python imports or helper utils in the instruction/code provided, DONT change/interfere with the provided function definition and parameter list.

This kernel, `batched_vecmat`,  is designed to perform batched element-wise multiplication and sum operation.

**Your objective is to optimize the body of `batched_vecmat`.**

You must ensure that:
1.  All arguments received by `batched_vecmat` are kept intact and not modified.
2. Provide you final code in ```python code block.
Example:
```python
<YOUR-CODE-HERE>
```
The full definition for `batched_vecmat` and relevant helper utilities are provided in the context below. You only need to optimize the code for `batched_vecmat` whilst keeping other things intact. DONT remove Imports and HELPER utils.

######################################## Imports ########################################
import numpy as np
import pytest
import torch
from numpy.random import RandomState

import triton
import triton.language as tl

######################################## Imports ########################################

@triton.jit
def batched_vecmat(
    A,
    B,
    dim_m, dim_n, dim_k,
    output,
    block_m: tl.constexpr, block_n: tl.constexpr, block_k: tl.constexpr
):
    """
    Performs a batched element-wise multiplication and sum operation.
    Effectively, for each m in dim_m, it computes the element-wise product of
    A[m, :] with each row of B[m, :, :], and sums the results.
    This can be expressed as: output[m, n] = sum_k (A[m, k] * B[m, n, k]).
    This is equivalent to `torch.sum(A.unsqueeze(1) * B, dim=2)`.

    Parameters
    ----------
    A
        Input tensor of shape [dim_m, dim_k].
        Interpreted as a batch of `dim_m` vectors, each of size `dim_k`.
    B
        Input tensor of shape [dim_m, dim_n, dim_k].
        Interpreted as a batch of `dim_m` groups of vectors. Each group `B[i, :, :]`
        contains `dim_n` vectors, each of size `dim_k`.
    dim_m
        The size of the first dimension of A and B (batch dimension).
    dim_n
        The size of the second dimension of B and the output tensor.
        Represents the number of vectors in each batch entry of B.
    dim_k
        The size of the last dimension of A and B (the dimension over which the sum is performed).
    output
        Output tensor of shape [dim_m, dim_n] where the results are stored.
    block_m
        tl.constexpr: Tiling size for the m dimension.
    block_n
        tl.constexpr: Tiling size for the n dimension.
    block_k
        tl.constexpr: Tiling size for the k dimension (reduction dimension).
    """
    # Your code here






## Reduction and tail repair

The numerical gate remains `atol=1e-3, rtol=1e-2`, including the original NumPy
comparison on seeded FP32 integer inputs. FP16 multiplication still rounds to
FP16; the reduction and loop accumulator now use FP32, with final storage in
the input dtype. This matches the existing task-local product-then-sum oracle
without relaxing its gate. The previous implicit FP16 reduction and accumulator
rounded each tree/block sum, causing the declared random FP16 cases to fail.
This is an explicit repaired baseline revision, not a fitted tolerance.

The original two N=32/block_n=64 scored cases are now actually executed. Operand
and output M/N masks, a ceiling K loop and final K masks make tail accesses safe;
no shape, case ID, block setting, seed or scored work is removed. All 31 original
case IDs and 30 original performance rows remain. Two unscored signed fractional
controls add simultaneous M/N/K tails (17,19,35), for 33 correctness cases.

The original allocating wrapper and all warmup/sample/timer settings remain.
Private snapshots protect A/B; full outputs are checked against the numerical
oracle after ordinary execution, actual timed execution, and changed-input
replay. Inputs and poisoned output snapshots are restored in `finally` on
success or failure. Historical timings describe the prior kernel revision;
current baseline and candidate both use the repaired revision.
