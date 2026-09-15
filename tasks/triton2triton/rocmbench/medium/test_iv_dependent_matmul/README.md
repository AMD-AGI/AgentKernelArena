# IV-dependent tiled GEMM

The implemented initial Triton kernel is frozen independently as the baseline.
Edit only the declared `iv_dependent_matmul` and permitted implementation helpers.
All five pointer-induction variants compute the same GEMM. The original kernel
rounds its accumulator to float16 before storing; functional tests use a float32
output buffer, performance uses float16. Inputs are read-only float16/float32
matrices. The wrapper returns the preallocated full M-by-N output buffer.

Private input snapshots supply the independent torch matrix-product oracle.
Original functional assertions stay `atol=rtol=0.01`. Performance checks retain
the product cast to float16 with that same gate, with separate output metadata,
no-input-aliasing, full-value and finite checks. No failure-fitted tolerance,
baseline exception or alternate candidate path is used.

## Cases and compilation

All 2,105 original identities/parameters remain, including 2,100 scored cases
covering shapes, seven block configurations, both dtypes, five addressing variants,
three stage counts and two warp counts. A fixed 64KB estimate previously skipped
800 rows before the real compiler ran. This heuristic is removed: each declared
specialization must actually compile and pass, with unchanged launch options.
A real resource/compile error is still a failure, never an expected scored pass.
The redundant BLOCK_K>K skip is removed too: masked K loads define a partial first
tile. Ten additional unscored controls (all variants and both dtypes) check odd
M/N/K tails and K<BLOCK_K, with explicit shape/block/stage/warp identities.

## Timing and replay

The original preallocated-output callable, seed42, stage/warp/block values,
warmup10/repetition100, canonical graph/event timing and mean device latency
are retained. Full outputs are checked before timing and from the actual
`TimedRun`. Untimed replay changes valid floating input values, recomputes the
independent oracle, poisons the output and replays the captured invocation.
A finally block restores original inputs/output on success and failure. Frozen
baseline and candidate use identical workloads and timing boundaries. Newly
executed formerly skipped rows have no prior valid timing; historical aggregate
scores are not directly comparable. The old optional PyTorch peer timing is not
Arena's baseline. Do not modify generated benchmark helpers or stubs.

Use `python3 _arena_eval.py validate-task`, or
`python3 _arena_eval.py baseline|candidate compile|correctness|performance`.
Final submission checks use `ARENA_EVAL_PHASE=candidate_evaluation`; every action
emits `arena-eval-v1`, and Arena alone writes final validation/score reports.
Deliver files inside the configured boundary, not merely a fenced code block.
Missing/empty candidates never fall back to a reference or frozen implementation.

## Original operator instructions

The historical instructions below retain the operator semantics and interface;
the v2 on-disk edit and evaluation contract above governs submission format.


You are an expert in triton programming language. You will be given the function definition for the `iv_dependent_matmul`. Your task is to complete the kernel code. Only optimize the kernel code in the function definition, DONT remove any python imports or helper utils in the instruction/code provided, DONT change/interfere with the provided function definition and parameter list.

This kernel, `iv_dependent_matmul`,  is designed to perform  tiled matrix multiplication (C = A @ B).

**Your objective is to optimize the body of `iv_dependent_matmul`.**

You must ensure that:
1.  All arguments received by `iv_dependent_matmul` are kept intact and not modified.
2. Provide you final code in ```python code block.
Example:
```python
<YOUR-CODE-HERE>
```
The full definition for `iv_dependent_matmul` and relevant helper utilities are provided in the context below. You only need to optimize the code for `iv_dependent_matmul` whilst keeping other things intact. DONT remove Imports and HELPER utils.

######################################## Imports ########################################
import numpy as np
import pytest
import torch
from numpy.random import RandomState

import triton
import triton.language as tl
######################################## Imports ########################################

@triton.jit
def iv_dependent_matmul(a_ptr, b_ptr, c_ptr,
                        M, N, K,
                        stride_am, stride_ak,
                        stride_bk, stride_bn,
                        stride_cm, stride_cn,
                        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                        type: tl.constexpr):
    """
    Performs a tiled matrix multiplication (C = A @ B) with various strategies
    for managing and advancing pointers to input matrices A and B within the
    inner K-loop. The specific strategy for pointer updates is determined by
    the `type` parameter. This kernel is designed to explore different
    instruction scheduling and memory access patterns related to pointer
    arithmetic within the loop.

    Parameters:
    -----------
    a_ptr : tl.pointer_type
        Pointer to the first input matrix A.
    b_ptr : tl.pointer_type
        Pointer to the second input matrix B.
    c_ptr : tl.pointer_type
        Pointer to the output matrix C.
    M : int
        Number of rows in matrix A and C.
    N : int
        Number of columns in matrix B and C.
    K : int
        Number of columns in matrix A and rows in matrix B (the dimension being reduced).
    stride_am : int
        Stride for the M dimension (rows) of matrix A.
    stride_ak : int
        Stride for the K dimension (columns) of matrix A.
    stride_bk : int
        Stride for the K dimension (rows) of matrix B.
    stride_bn : int
        Stride for the N dimension (columns) of matrix B.
    stride_cm : int
        Stride for the M dimension (rows) of matrix C.
    stride_cn : int
        Stride for the N dimension (columns) of matrix C.
    BLOCK_SIZE_M : tl.constexpr
        The size of the block used for tiling along the M dimension.
    BLOCK_SIZE_N : tl.constexpr
        The size of the block used for tiling along the N dimension.
    BLOCK_SIZE_K : tl.constexpr
        The size of the block used for tiling along the K dimension (inner loop).
    type : tl.constexpr (str)
        A string literal controlling the pointer update strategy for `a_ptr` and `b_ptr`
        within the K-loop. Affects how `a_ptrs` and `b_ptrs` are calculated in each
        iteration of the K-loop.
        Possible values:
        - "pre_load": Pointers are calculated at the beginning of each K-loop iteration.
        - "post_load": Pointers are calculated at the end of each K-loop iteration for the next iteration.
        - "post_pre_mixed": `a_ptrs` is calculated at the beginning, `b_ptrs` at the end for the next iteration.
        - "post_load_two_iters": Pointers are advanced to prefetch/prepare for two iterations ahead.
        - "post_load_three_iters": Pointers are advanced to prefetch/prepare for three iterations ahead.
    """
    # Your code here






## FP32 resource scheduling repair

The original FP32 128×64×64 / 64×128×64 specializations with four pipeline
stages required 196,608 bytes of shared memory, exceeding MI355X's 163,840-byte
limit. The same public specializations failed on pinned Triton 3.6 and 3.8.
No cases, stage counts, warp counts, public block arguments or numerical gates
are removed or changed. The kernel now splits an internal FP32 K tile in half
when `(BLOCK_SIZE_M + BLOCK_SIZE_N) * BLOCK_SIZE_K > 8192`. This keeps the
declared tile as the maximum logical K span while loading/accumulating smaller
physical chunks. All five induction-variable modes advance by that internal
chunk; they still cover exactly K, accumulate in FP32 and store FP16.

The twelve representative previously failing launches retain their exact public
options and now compile to 98,304 shared bytes and pass the unchanged 1e-2 gates.
Full qualification still requires all 2,115 correctness and 2,100 performance
cases. Baseline freezing captures this repaired kernel revision for both roles;
previous timings belong to the older baseline. Warmups, samples, output
allocation and timer boundaries remain unchanged.
