# test_gemm_fusion

The on-disk starting functions contain implemented Triton code. The v2 declaration
therefore uses `implemented` and a frozen `initial_candidate` baseline, regardless
of this directory's historical suite name. Edit only the declared function scopes
(and permitted implementation helpers) in `test_gemm_fusion.py`. Preserve signatures,
references, input generation, assertions, test parameters and timing policy.
Deliver edits to those files; a fenced code block alone is not a submission.

## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
with one role and one action. Submitted checks use `ARENA_EVAL_PHASE=candidate_evaluation`.
The adapter emits `arena-eval-v1`; Arena owns final score/validation reports.
`workloads.json` retains 517 original collected cases, including 516 performance cases.
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


You are an expert in triton programming language. You will be given the function definition for the `gemm_fusion_kernel`. Your task is to complete the kernel code. Only optimize the kernel code in the function definition, DONT remove any python imports or helper utils in the instruction/code provided, DONT change/interfere with the provided function definition and parameter list.

This kernel, `gemm_fusion_kernel`,  is designed to perform a fused matrix multiplication operation.

**Your objective is to optimize the body of `gemm_fusion_kernel`.**

You must ensure that:
1.  All arguments received by `gemm_fusion_kernel` are kept intact and not modified.
2. Provide you final code in ```python code block. 
Example:
```python
<YOUR-CODE-HERE>
```
The full definition for `gemm_fusion_kernel` and relevant helper utilities are provided in the context below. You only need to optimize the code for `gemm_fusion_kernel` whilst keeping other things intact.


######################################## Imports #######################################
import pytest
import torch

import triton
import triton.language as tl

######################################## Imports #######################################



@triton.jit
def gemm_fusion_kernel(A, B, C, E,  #
                       M, N, K,  #
                       stride_am, stride_ak, stride_bn, stride_bk, stride_cn, stride_ck, stride_em, stride_ek,  #
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """
    This Triton kernel is designed to perform a fused matrix multiplication operation.
    It computes E = (A @ B^T) @ C, where A, B, C, and E are matrices.
    The computation is performed in a tiled manner to optimize for memory access patterns
    and leverage the parallelism of GPU architectures.

    Each program instance (kernel launch) processes a tile of the output matrix E,
    corresponding to a `BLOCK_M` strip of rows from matrix A. It iterates over
    tiles of B and C along their N dimension to compute the intermediate product
    (A_tile @ B_tile^T) and then accumulates the result with C_tile into E_tile.

    Args:
        A: Pointer to the input matrix A. Expected shape (M, K_common).
        B: Pointer to the input matrix B. Expected shape (N, K_common).
        C: Pointer to the input matrix C. Expected shape (N, K_out).
        E: Pointer to the output matrix E, where the result E = (A @ B^T) @ C is stored. Expected shape (M, K_out).
        M: The number of rows in matrix A and matrix E.
        N: The number of rows in matrix B and matrix C. This is also the dimension
           over which the product (A @ B^T) and C are contracted.
        K: This parameter represents two potentially different dimensions depending on context,
           but given the block shapes, it is used as K_common for A and B, and K_out for C and E.
           Specifically:
           - For A and B, it's the common dimension (K_common) for A @ B^T.
           - For C and E, it's the output column dimension (K_out).
           The kernel structure (BLOCK_K used for all) implies K_common == K_out.
        stride_am: The stride (in number of elements) for matrix A along the M dimension (row stride).
        stride_ak: The stride (in number of elements) for matrix A along the K dimension (column stride).
        stride_bn: The stride (in number of elements) for matrix B along the N dimension (row stride).
        stride_bk: The stride (in number of elements) for matrix B along the K dimension (column stride).
        stride_cn: The stride (in number of elements) for matrix C along the N dimension (row stride).
        stride_ck: The stride (in number of elements) for matrix C along the K dimension (column stride).
        stride_em: The stride (in number of elements) for matrix E along the M dimension (row stride).
        stride_ek: The stride (in number of elements) for matrix E along the K dimension (column stride).
        BLOCK_M: tl.constexpr, the tile size for the M dimension. Each kernel instance
                   processes a block of `BLOCK_M` rows from A and E.
        BLOCK_N: tl.constexpr, the tile size for the N dimension. The kernel iterates
                   over B and C in blocks of `BLOCK_N` along their N dimension.
        BLOCK_K: tl.constexpr, the tile size for the K dimension. This is the block size
                   for the common dimension in A@B^T and the output column dimension
                   for C and E.
    """
    # Your code here





