# triton_multreduce_matmul_kernel

The on-disk starting functions contain implemented Triton code. The v2 declaration
therefore uses `implemented` and a frozen `initial_candidate` baseline, regardless
of this directory's historical suite name. Edit only the declared function scopes
(and permitted implementation helpers) in `triton_multreduce_matmul_kernel.py`. Preserve signatures,
references, input generation, assertions, test parameters and timing policy.
Deliver edits to those files; a fenced code block alone is not a submission.

## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
with one role and one action. Submitted checks use `ARENA_EVAL_PHASE=candidate_evaluation`.
The adapter emits `arena-eval-v1`; Arena owns final score/validation reports.
`workloads.json` retains 80 original collected cases, including 66 performance cases.
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


You are an expert in triton programming language. You will be given the function definition for the `triton_multreduce_matmul_kernel`. Your task is to complete the kernel code. Only complete the kernel code in the function definition, DONT remove any python imports or helper utils in the instruction/code provided, DONT change/interfere with the provided function definition and parameter list ,

This kernel, `triton_multreduce_matmul_kernel`,  is designed to perform matrix multiplication by explicitly using element-wise multiplication followed by a reduction (summation), instead of relying on Triton's `tl.dot` intrinsic.

**Your objective is to implement the body of `triton_multreduce_matmul_kernel`.**

You must ensure that:
1.  All arguments received by `triton_multreduce_matmul_kernel` (i.e., `a_ptr`, `b_ptr`, `c_ptr`, `bias_ptr`, `M`, `N`, `K`, all stride arguments, `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, `BLOCK_SIZE_K`, `USE_BIAS`, and `EVEN_K`) are kept intact and not modified.
2. Provide you final code in ```python code block.
Example:
```python
<YOUR-CODE-HERE>
```


The full definition for `triton_multreduce_matmul_kernel` and relevant helper utilities are provided in the context below. You only need to complete the code for `triton_multreduce_matmul_kernel` whilst keeping other things.


# Imports:
# --------

import argparse
import itertools
import os
import sys
from typing import Any, Callable, Optional

import pytest
import torch
from torch import Tensor

import triton
import triton.language as tl


# Triton GEMM:
# ------------

######################## HELPER UTILS #####################
# Autotune configurations for Triton GEMM implemented with explicit dot product.
def get_triton_multreduce_autotune_configs() -> list[triton.Config]:
    block_size_k_range: list[int] = [128, 256, 512]
    kpack_range: list[int] = [1, 2]
    return [
        triton.Config(
            {"BLOCK_SIZE_M": 1, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": block_size_k, "waves_per_eu": 0, "kpack": kpack},
            num_warps=8, num_stages=2) for block_size_k, kpack in itertools.product(block_size_k_range, kpack_range)
    ]


def get_triton_autotune_key() -> list[str]:
    return ["M", "N", "K"]


def get_triton_heuristics() -> dict[str, Callable[[dict[str, Any]], Any]]:
    return {"EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0}

######################## HELPER UTILS #####################


# Triton GEMM kernel implemented with explicit dot product.
@triton.autotune(configs=get_triton_multreduce_autotune_configs(), key=get_triton_autotune_key())
@triton.heuristics(get_triton_heuristics())
@triton.jit
def triton_multreduce_matmul_kernel(a_ptr, b_ptr, c_ptr, bias_ptr,  #
                                    M: int, N: int, K: int,  #
                                    stride_am: int, stride_ak: int,  #
                                    stride_bk: int, stride_bn: int,  #
                                    stride_cm: int, stride_cn: int,  #
                                    stride_bias: int,  #
                                    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                                    BLOCK_SIZE_K: tl.constexpr,  #
                                    USE_BIAS: tl.constexpr, EVEN_K: tl.constexpr  #
                                    ):
    """
    Performs matrix multiplication (C = A @ B + bias) using an explicit
    element-wise multiplication followed by a reduction (summation) strategy,
    instead of `tl.dot()`.

    This kernel is a wrapper around `triton_matmul_kernel`, configured
    to use the non-`tl.dot` path by setting `USE_DOT=False`.

    Args:
        a_ptr: Pointer to the first input matrix A.
        b_ptr: Pointer to the second input matrix B.
        c_ptr: Pointer to the output matrix C.
        bias_ptr: Pointer to the bias vector/matrix.
        M: Number of rows in matrix A and C.
        N: Number of columns in matrix B and C.
        K: Number of columns in matrix A and rows in matrix B.
        stride_am: Stride for the M dimension of matrix A.
        stride_ak: Stride for the K dimension of matrix A.
        stride_bk: Stride for the K dimension of matrix B.
        stride_bn: Stride for the N dimension of matrix B.
        stride_cm: Stride for the M dimension of matrix C.
        stride_cn: Stride for the N dimension of matrix C.
        stride_bias: Stride for the bias.
        BLOCK_SIZE_M (tl.constexpr): Tile size for the M dimension.
        BLOCK_SIZE_N (tl.constexpr): Tile size for the N dimension.
        BLOCK_SIZE_K (tl.constexpr): Tile size for the K dimension.
        USE_BIAS (tl.constexpr): If True, add bias to the result.
        EVEN_K (tl.constexpr): If True, K is evenly divisible by BLOCK_SIZE_K,
                               allowing for unmasked loads in the K loop.
    """
    # Your code here.






The scored workload directly launches `triton_matmul_kernel` with the declared
fixed block sizes, warps and stages and `USE_DOT=False`. This actual entrypoint
is explicitly declared alongside `triton_multreduce_matmul_kernel`, the autotuned
wrapper exercised by the original correctness cases. Optimizing only wrapper
autotuning does not improve the fixed-launch measurements. Keep both interfaces
implemented; the wrapper remains part of correctness coverage. This declaration
clarifies the original harness behavior without changing its invocations, cases,
numerical gates, tuning parameters, warmups or sample counts. Configuration
helpers remain editable implementation helpers, not separate entrypoints.
