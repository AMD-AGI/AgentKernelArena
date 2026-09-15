# test_chained_matmul

The on-disk starting functions contain implemented Triton code. The v2 declaration
therefore uses `implemented` and a frozen `initial_candidate` baseline, regardless
of this directory's historical suite name. Edit only the declared function scopes
(and permitted implementation helpers) in `test_chained_matmul.py`. Preserve signatures,
references, input generation, assertions, test parameters and timing policy.
Deliver edits to those files; a fenced code block alone is not a submission.

## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
with one role and one action. Submitted checks use `ARENA_EVAL_PHASE=candidate_evaluation`.
The adapter emits `arena-eval-v1`; Arena owns final score/validation reports.
`workloads.json` retains 151 original collected cases, including 150 performance cases.
Collection is checked against this independent manifest. Original binary-input correctness
case and exact equality gate are retained, with private inputs and full-output checks. Performance inputs additionally run the task-local
oracle in `_arena_reference.py`, before timing, against actual TimedRun output, and after fresh inputs/NaN output poisoning.
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


You are an expert in triton programming language. You will be given the function definition for the `chained_matmul_kernel`. Your task is to complete the kernel code. Only complete the kernel code in the function definition, DONT remove any python imports or helper utils in the instruction/code provided, DONT change/interfere with the provided function definition and parameter list.

This kernel, `chained_matmul_kernel`,  is designed to perform chained matrix multiplication of the form `(A @ B.T) @ C` on a GPU.

**Your objective is to implement the body of `chained_matmul_kernel`.**

You must ensure that:
1.  All arguments received by `chained_matmul_kernel` are kept intact and not modified.
2. Provide you final code in ```python code block.
Example:
```python
<YOUR-CODE-HERE>
```
The full definition for `chained_matmul_kernel` and relevant helper utilities are provided in the context below. You only need to complete the code for `chained_matmul_kernel` whilst keeping other things intact.

######################################## Imports ########################################
import numpy as np
import pytest
import torch
from numpy.random import RandomState

import triton
import triton.language as tl

######################################## Imports ########################################



@triton.jit
def chained_matmul_kernel(A,  # Pointer to the first input tensor `A`. Expected shape: (m, k). This tensor provides the first operand in the (A @ B.T) operation.
                            B,  # Pointer to the second input tensor `B`. Expected shape: (n, k). The transpose of this tensor (B.T, shape (k, n)) is used as the second operand in the (A @ B.T) operation.
                            C,  # Pointer to the third input tensor `C`. Expected shape: (n, k). This tensor is the second operand in the ((A @ B.T) @ C) operation.
                            out,  # Pointer to the output tensor `out`. Expected shape: (m, k). This tensor will store the result of the chained matrix multiplication (A @ B.T) @ C.
                            m,    # Integer representing the 'm' dimension. This is the number of rows in matrix `A` and the output matrix `out`.
                            n,    # Integer representing the 'n' dimension. This is the number of rows in matrices `B` and `C`. It also becomes the shared inner dimension after the A @ B.T operation (i.e., A @ B.T results in an m x n matrix). The kernel iterates over this dimension in `block_n` sized chunks.
                            k: tl.constexpr,  # Compile-time constant integer for the 'k' dimension. This is the number of columns in matrices `A`, `B`, `C`, and `out`. It's also the shared inner dimension for B.T @ C if C were (k,p) and for A @ B.T if B.T were (k,n). In this kernel, it's the common feature dimension.
                            block_m: tl.constexpr,  # Compile-time constant integer defining the tile size for the 'm' dimension. Each kernel instance (program) will process `block_m` rows of matrix `A` (and write `block_m` rows to `out`) at a time.
                            block_n: tl.constexpr,  # Compile-time constant integer defining the tile size for the 'n' dimension. The kernel will iterate through the 'n' dimension in steps of `block_n` when processing matrices `B` and `C`.
                            block_k: tl.constexpr  # Compile-time constant integer defining the tile size for the 'k' dimension. For this specific kernel, there's a constraint `block_k == k`, meaning the entire 'k' dimension is processed at once within the dot products, rather than being tiled itself for reduction.
                           ):
    """
    Brief description of the kernel:
    This Triton JIT-compiled kernel, `chained_matmul_kernel`, is designed to efficiently compute
    a chained matrix multiplication of the form `(A @ B.T) @ C` on a GPU.
    It takes three input matrices: `A` of shape `(m, k)`, `B` of shape `(n, k)`, and `C` of shape `(n, k)`.
    The transpose of `B` is used in the first multiplication, resulting in an intermediate
    matrix of shape `(m, n)`. This intermediate result is then multiplied by `C` (after `C` is
    effectively processed column-wise due to the dot product with the intermediate, or rather,
    the accumulation logic results in an `(m,k)` output from `(m,n) @ (n,k)` where the second `(n,k)`
    is `C`). The final output matrix `out` has the shape `(m, k)`.
    The kernel utilizes a tiled approach for parallel processing.


    The user should implement the logic for (A @ B.T) @ C using Triton programming constructs.
    This typically involves:
    - Calculating program IDs and offsets for the current block.
    - Loading tiles of A, B, and C.
    - Performing the dot products and accumulations in a loop over the 'n' dimension.
    - Handling boundary conditions carefully with masks.
    - Storing the resulting tile to the output tensor `out`

    """
    # Your code here






## Numerical and replay contract

The original binary-input case still requires exact equality. The original 150
random FP16 performance cases had no numerical check before Arena migration;
copying the binary exact rule onto arbitrary random inputs incorrectly requires
identical floating-point reduction orders. Their new numerical contract is the
same two GEMMs with the kernel's **mandatory FP16 intermediate** and FP16 output.
An independent FP64 oracle bounds both FP32 dot accumulations using
`gamma(n)=n*2^-24/(1-n*2^-24)`, with `n=2*K` and `2*N`, propagates the intermediate
FP16 rounding interval through the second product, then rounds both endpoints to
FP16. Every output must lie inside that interval and be finite. Bounds depend on
input values and dimensions, never on a baseline's measured error. The original
binary equality assertion remains in addition to these checks.

All 151 original identities, 150 scored workloads, kernel code, source wrappers,
seed, tiles, stages, warmup 10, repetition 100, canonical timer and mean reduction
are unchanged. One unscored signed fractional case adds a masked final M tile,
for 152 correctness cases. It does not create a new timing workload. Read-only
inputs are snapshotted before evaluation; the actual timed result and a new
reference after changed inputs/poisoned output are checked in full. Inputs and
output are restored in `finally`, including failed replay paths. Oracle, poison,
and restoration work runs outside the measured callable for both roles.
