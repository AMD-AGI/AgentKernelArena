# gemm

The on-disk starting functions contain implemented Triton code. The v2 declaration
therefore uses `implemented` and a frozen `initial_candidate` baseline, regardless
of this directory's historical suite name. Edit only the declared function scopes
(and permitted implementation helpers) in `gemm.py`. Preserve signatures,
references, input generation, assertions, test parameters and timing policy.
Deliver edits to those files; a fenced code block alone is not a submission.

## Workload scope and implementation interface

This Arena task measures **row-major FP16 GEMM**, `C = A @ B`, with FP16
inputs and output, `APPLY_SCALE=None` and `ACTIVATION=""`. Its workload is the
original protected pytest selection in `gemm.py` and `workloads.json`: eight
square `(M,N,K)=(1024*v,1024*v,1024*v)` shapes for `v=1..8`, plus
`(4864,4096,8192)`, `(9728,8192,65536)` and `(4864,8192,4160)`. These cover
different matrix sizes, rectangular GEMM and both long and tile-dependent tail
reductions. Each shape retains its original correctness and performance case.
The numerical gate is unchanged: `atol=5e-3`, `rtol=1e-2`, with the original
FP16 PyTorch result and the additional protected tensor-contract checks.

The implementation API is broader than this selected workload. Its dtype,
stride, scaling and activation arguments and existing behavior must be preserved.
The historical API documentation below explains those arguments; it does not
claim that this task benchmarks every supported combination. The original pytest
selection excludes alternate dtypes, column-major layouts and activation/scaling
modes; the optional standalone plotting CLI is separate from Arena's configured
pytest evaluation. A PASS for this task certifies the declared workload only,
not those additional API paths or arbitrary M/N-tail shapes. No original case,
assertion or implementation branch is removed by this scope clarification.

## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
with one role and one action. Submitted checks use `ARENA_EVAL_PHASE=candidate_evaluation`.
The adapter emits `arena-eval-v1`; Arena owns final score/validation reports.
`workloads.json` retains 22 original collected cases, including 11 performance cases.
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


You are an expert in triton programming language. You will be given the function definition for the `matmul_kernel`. Your task is to complete the kernel code. Only complete the kernel code in the function definition, DONT remove any python imports or helper utils in the instruction/code provided, DONT change/interfere with the provided function definition and parameter list ,

This kernel, `matmul_kernel`,  is designed to perform matrix multiplication C = A x B using a tiled approach.

**Your objective is to implement the body of `matmul_kernel`.**

You must ensure that:
1.  All arguments received by `matmul_kernel` are kept intact and not modified.
2. Provide you final code in ```python code block. 
Example:
```python
<YOUR-CODE-HERE>
```


The full definition for `matmul_kernel` and relevant helper utilities are provided in the context below. You only need to wrcomplete the the code for `matmul_kernel` whilst keeping other things intact.


import torch
import triton
import triton.language as tl
import sys
import argparse
import pytest
import re

# This is a Triton kernel for matrix multiplication (GEMM) with support for various data types and scaling modes.

#################### Helper utils functions ####################
# Activation function.  
@triton.jit  
def leaky_relu(x):  
    x = x + 1  
    return tl.where(x >= 0, x, 0.01 * x)  
#################### Helper utils functions ####################



@triton.autotune(  
    configs=[  
        triton.Config(  
            {  
                'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2,  
                'kpack': 2, 'matrix_instr_nonkdim': 16  
            }, num_warps=4, num_stages=2),  
        triton.Config(  
            {  
                'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2,  
                'kpack': 2, 'matrix_instr_nonkdim': 16  
            }, num_warps=8, num_stages=2),  
        triton.Config(  
            {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 4, 'waves_per_eu': 0},  
            num_warps=8, num_stages=2),  
        triton.Config(  
            {  
                'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2,  
                'kpack': 1, 'matrix_instr_nonkdim': 16  
            }, num_warps=8, num_stages=2),  
        triton.Config(  
            {  
                'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'waves_per_eu': 0,  
                'kpack': 1  
            }, num_warps=8, num_stages=2),  
        triton.Config(  
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 4, 'waves_per_eu': 0},  
            num_warps=8, num_stages=2),  
        triton.Config(  
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2},  
            num_warps=8, num_stages=2),  
    ],  
    key=['M', 'N', 'K'],  
    use_cuda_graph=True,  
)  
@triton.heuristics({  
    'EVEN_K':  
    lambda args: args['K'] % args['BLOCK_SIZE_K'] == 0, 'GRID_MN':  
    lambda args: triton.cdiv(args['M'], args['BLOCK_SIZE_M']) * triton.cdiv(args['N'], args['BLOCK_SIZE_N'])  
})  
@triton.jit  
def matmul_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    a_scale_ptr,
    b_scale_ptr,
    stride_ascale_m,
    stride_ascale_k,
    stride_bscale_k,
    stride_bscale_n,
    # Meta-parameters
    GROUP_K: tl.constexpr,
    GROUP_N: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    EVEN_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    APPLY_SCALE: tl.constexpr,
    ACTIVATION: tl.constexpr,
    GRID_MN: tl.constexpr,
):
    """
    Computes the matrix multiplication C = A x B using a tiled approach.

    This kernel is designed for efficient matrix multiplication on GPUs,
    incorporating features like block-wise computation, optional per-tensor or
    per-block scaling, optional activation functions, and program ID (PID)
    remapping for improved L2 cache utilization, particularly on multi-XCD
    (Cross-Chip Die) hardware.

    Parameters:
        a_ptr: Pointer to the input matrix A.
        b_ptr: Pointer to the input matrix B.
        c_ptr: Pointer to the output matrix C.
        M: The number of rows in matrix A and matrix C.
        N: The number of columns in matrix B and matrix C.
        K: The number of columns in matrix A and rows in matrix B (the reduction dimension).
        stride_am: Stride for matrix A along the M dimension (row stride).
        stride_ak: Stride for matrix A along the K dimension (column stride).
        stride_bk: Stride for matrix B along the K dimension (row stride).
        stride_bn: Stride for matrix B along the N dimension (column stride).
        stride_cm: Stride for matrix C along the M dimension (row stride).
        stride_cn: Stride for matrix C along the N dimension (column stride).
        a_scale_ptr: Pointer to scale factors for matrix A. Used if `APPLY_SCALE` is 'tensor' or 'block'.
                     If `APPLY_SCALE` is 'tensor', this points to a single scalar.
                     If `APPLY_SCALE` is 'block', this points to a tensor of scales.
        b_scale_ptr: Pointer to scale factors for matrix B. Used if `APPLY_SCALE` is 'tensor' or 'block'.
                     If `APPLY_SCALE` is 'tensor', this points to a single scalar.
                     If `APPLY_SCALE` is 'block', this points to a tensor of scales.
        stride_ascale_m: Stride for A's scale tensor along its M-dimension (if `APPLY_SCALE` is 'block' and A scales are per-row-block).
        stride_ascale_k: Stride for A's scale tensor along its K-dimension (if `APPLY_SCALE` is 'block' and A scales are per-K-group).
        stride_bscale_k: Stride for B's scale tensor along its K-dimension (if `APPLY_SCALE` is 'block' and B scales are per-K-group).
        stride_bscale_n: Stride for B's scale tensor along its N-dimension (if `APPLY_SCALE` is 'block' and B scales are per-N-group).
        GROUP_K (tl.constexpr): Grouping factor for the K dimension when `APPLY_SCALE` is 'block'.
                                Scales for A and B are loaded based on K-groups of this size.
        GROUP_N (tl.constexpr): Grouping factor for the N dimension when `APPLY_SCALE` is 'block' for matrix B.
                                Scales for B are loaded based on N-groups of this size.
        BLOCK_SIZE_M (tl.constexpr): The tile size for the M dimension processed by each kernel instance.
        BLOCK_SIZE_N (tl.constexpr): The tile size for the N dimension processed by each kernel instance.
        BLOCK_SIZE_K (tl.constexpr): The tile size for the K dimension (reduction dimension) processed in each inner loop.
        EVEN_K (tl.constexpr): Boolean flag. If True, K is assumed to be perfectly divisible by `BLOCK_SIZE_K`.
                               If False, boundary checks (masking) are applied when loading data along the K dimension.
        GROUP_SIZE_M (tl.constexpr): Number of M-dimension blocks to group together for program ID mapping.
                                     This influences L2 data reuse.
        APPLY_SCALE (tl.constexpr): Specifies how scaling is applied.
                                    - `None`: No scaling.
                                    - `'tensor'`: A single scale factor is applied to matrix A and another to matrix B.
                                    - `'block'`: Scale factors are loaded and applied per block of A and/or B.
        ACTIVATION (tl.constexpr): Specifies the activation function to apply after the accumulation and scaling.
                                   Example: "leaky_relu". If `None`, no activation is applied.
        GRID_MN (tl.constexpr): The total number of program instances (PIDs) launched for the M and N dimensions.
                                This is typically `tl.cdiv(M, BLOCK_SIZE_M) * tl.cdiv(N, BLOCK_SIZE_N)`.
                                Used for PID remapping across XCDs.
    """
    # Your code here.




