# layernorm

The on-disk starting functions contain implemented Triton code. The v2 declaration
therefore uses `implemented` and a frozen `initial_candidate` baseline, regardless
of this directory's historical suite name. Edit only the declared function scopes
(and permitted implementation helpers) in `layernorm.py`. Preserve signatures,
references, input generation, assertions, test parameters and timing policy.
Deliver edits to those files; a fenced code block alone is not a submission.

## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
with one role and one action. Submitted checks use `ARENA_EVAL_PHASE=candidate_evaluation`.
The adapter emits `arena-eval-v1`; Arena owns final score/validation reports.
`workloads.json` retains 32 original collected cases, including 21 performance cases.
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


You are an expert in triton programming language. You will be given the function definition for the `layernorm_kernel`. Your task is to complete the kernel code. Only optimize the kernel code in the function definition, DONT remove any python imports or helper utils in the instruction/code provided, DONT change/interfere with the provided function definition and parameter list ,

This kernel, `layernorm_kernel`,  is designed to perform layer normalization on the input tensor.

**Your objective is to optimize the body of `layernorm_kernel`.**

You must ensure that:
1.  All arguments received by `layernorm_kernel` are kept intact and not modified.
2. Provide you final code in ```python code block.
Example:
```python
<YOUR-CODE-HERE>
```


The full definition for `layernorm_kernel` and relevant helper utilities are provided in the context below. You only need to optimize the code for `layernorm_kernel` whilst keeping other things intact.


import argparse
import sys
import pytest

import torch
import triton
import triton.language as tl
import os
import json
import math
from itertools import product

######################################## HELPERS utils ########################################
def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def get_cuda_autotune_config():
    return [
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=16, num_stages=1),
    ]


def get_hip_autotune_config():
    return [
        triton.Config({'waves_per_eu': we}, num_warps=wa, num_stages=1) for we, wa in product([1, 2, 4], [4, 8, 16])
    ]


def get_autotune_config():
    if is_cuda():
        return get_cuda_autotune_config()
    else:
        return get_hip_autotune_config()

######################################## HELPERS utils ########################################
@triton.autotune(configs=get_autotune_config(), key=['n_rows', 'n_cols'], use_cuda_graph=True)
@triton.jit
def layernorm_kernel(x_ptr, y_ptr, w_ptr, b_ptr, mean_ptr, rstd_ptr, x_row_stride, y_row_stride, n_rows, n_cols, eps,
                     BLOCK_SIZE: tl.constexpr):
  """
  Performs Layer Normalization on an input tensor.

  This kernel normalizes each row of the input tensor `x` independently.
  For each row, it calculates the mean and variance across its columns (features).
  It then normalizes the row using these statistics, applies a learnable affine
  transformation (scale `w` and bias `b`), and stores the result in `y`.
  The per-row mean and reciprocal standard deviation (rstd) are also stored.

  Args:
      x_ptr (triton.language.tensor): Pointer to the input tensor of shape (n_rows, n_cols).
      y_ptr (triton.language.tensor): Pointer to the output tensor of shape (n_rows, n_cols),
                                      where the normalized and transformed data will be stored.
      w_ptr (triton.language.tensor): Pointer to the weight tensor (gamma) of shape (n_cols).
                                      Used for scaling the normalized input.
      b_ptr (triton.language.tensor): Pointer to the bias tensor (beta) of shape (n_cols).
                                      Used for shifting the normalized input.
      mean_ptr (triton.language.tensor): Pointer to a tensor of shape (n_rows) where the
                                         calculated mean for each row will be stored.
      rstd_ptr (triton.language.tensor): Pointer to a tensor of shape (n_rows) where the
                                         calculated reciprocal standard deviation
                                         (1/sqrt(variance + eps)) for each row will be stored.
      x_row_stride (int): The stride (number of elements) to move from one row
                          to the next in the `x_ptr` tensor.
      y_row_stride (int): The stride (number of elements) to move from one row
                          to the next in the `y_ptr` tensor.
      n_rows (int): The number of rows in the input tensor `x`. Each row is
                    processed independently by a separate program instance.
      n_cols (int): The number of columns (features) in the input tensor `x`.
                    Normalization is performed across these columns for each row.
      eps (float): A small constant added to the variance for numerical stability
                   before calculating the reciprocal square root.
      BLOCK_SIZE (tl.constexpr): A compile-time constant defining the size of blocks
                                 used to process columns. This influences how data is
                                 loaded and processed in parallel within a row.
  """
    # Your code here.
