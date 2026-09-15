# multreduce_matmul_dot_kernel

The on-disk starting functions contain implemented Triton code. The v2 declaration
therefore uses `implemented` and a frozen `initial_candidate` baseline, regardless
of this directory's historical suite name. Edit only the declared function scopes
(and permitted implementation helpers) in `multreduce_matmul_dot_kernel.py`. Preserve signatures,
references, input generation, assertions, test parameters and timing policy.
Deliver edits to those files; a fenced code block alone is not a submission.

## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
with one role and one action. Submitted checks use `ARENA_EVAL_PHASE=candidate_evaluation`.
The adapter emits `arena-eval-v1`; Arena owns final score/validation reports.
`workloads.json` retains 38 original collected cases, including 24 performance cases.
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


You are an expert in triton programming language. You will be given a instruction/function definition of the required kernel : `triton_dot_matmul_kernel`, your task is to complete the kernel code for the corresponding operator/function definition using triton programming language. This kernel should implement a General Matrix Multiplication (GEMM) specifically using the tl.dot operation in triton and add necessary logic to use it. Only complete the kernel code in the function definition, DONT remove any python imports or helper utils in the instruction/code provided, DONT change/interfere with the provided function definition and parameter list ,only add if required. :
Provide you final code in ```python code block. 
Example:
```python
<YOUR-CODE-HERE>
```

The full definition for `triton_dot_matmul_kernel` and relevant helper utilities are provided in the context below. You only need to complete the code for `triton_dot_matmul_kernel` whilst keeping other things.


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

######################## HELPER UTILS #####################

Autotune configurations for Triton GEMM implemented with tl.dot.

def get_triton_dot_autotune_configs() -> list[triton.Config]:
block_size_n_range: list[int] = [16, 32]
block_size_k_range: list[int] = [128, 256, 512]
kpack_range: list[int] = [1, 2]
num_warps_range: list[int] = [1, 2]
return [
triton.Config(
{
"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": block_size_n, "BLOCK_SIZE_K": block_size_k, "waves_per_eu": 0,
"matrix_instr_nonkdim": 16, "kpack": kpack
}, num_warps=num_warps, num_stages=2) for block_size_n, block_size_k, kpack, num_warps in itertools.product(
block_size_n_range, block_size_k_range, kpack_range, num_warps_range)
]

def get_triton_autotune_key() -> list[str]:
return ["M", "N", "K"]

def get_triton_heuristics() -> dict[str, Callable[[dict[str, Any]], Any]]:
return {"EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0}

###############################################################

@triton.autotune(configs=get_triton_dot_autotune_configs(), key=get_triton_autotune_key())
@triton.heuristics(get_triton_heuristics())
@triton.jit
def triton_dot_matmul_kernel(a_ptr, b_ptr, c_ptr, bias_ptr,  #
M: int, N: int, K: int,  #
stride_am: int, stride_ak: int,  #
stride_bk: int, stride_bn: int,  #
stride_cm: int, stride_cn: int,  #
stride_bias: int,  #
BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
USE_BIAS: tl.constexpr, EVEN_K: tl.constexpr  #
):
"""
Performs a General Matrix Multiplication (GEMM) of the form C = A @ B + bias.
This kernel is specifically designed to use the tl.dot operation for the
core matrix multiplication.

Parameters:
- a_ptr: Pointer to the A matrix (input).
- b_ptr: Pointer to the B matrix (input).
- c_ptr: Pointer to the C matrix (output).
- bias_ptr: Pointer to the bias vector/matrix. Used only if USE_BIAS is True.
- M: Number of rows in matrix A and C.
- N: Number of columns in matrix B and C.
- K: Number of columns in matrix A and rows in matrix B (common dimension).
- stride_am: Stride for matrix A along the M dimension (row stride).
- stride_ak: Stride for matrix A along the K dimension (column stride).
- stride_bk: Stride for matrix B along the K dimension (row stride).
- stride_bn: Stride for matrix B along the N dimension (column stride).
- stride_cm: Stride for matrix C along the M dimension (row stride).
- stride_cn: Stride for matrix C along the N dimension (column stride).
- stride_bias: Stride for the bias. Interpretation depends on bias dimensions.
- BLOCK_SIZE_M: tl.constexpr, tile size for the M dimension during computation.
- BLOCK_SIZE_N: tl.constexpr, tile size for the N dimension during computation.
- BLOCK_SIZE_K: tl.constexpr, tile size for the K dimension during computation.
- USE_BIAS: tl.constexpr, boolean flag indicating whether to add the bias term.
- EVEN_K: tl.constexpr, boolean flag indicating if K is perfectly divisible by BLOCK_SIZE_K,
            allowing for potentially more efficient, unmasked loads along the K dimension.
"""
# Your code here.




