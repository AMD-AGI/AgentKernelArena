# rmsnorm_fwd

The on-disk starting functions contain implemented Triton code. The v2 declaration
therefore uses `implemented` and a frozen `initial_candidate` baseline, regardless
of this directory's historical suite name. Edit only the declared function scopes
(and permitted implementation helpers) in `rmsnorm_fwd.py`. Preserve signatures,
references, input generation, assertions, test parameters and timing policy.
Deliver edits to those files; a fenced code block alone is not a submission.

## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
with one role and one action. Submitted checks use `ARENA_EVAL_PHASE=candidate_evaluation`.
The adapter emits `arena-eval-v1`; Arena owns final score/validation reports.
`workloads.json` retains 182 original collected cases, including 126 performance cases.
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


You are an expert in triton programming language. You will be given the function definition for the `rms_kernel`. Your task is to complete the kernel code. Only complete the kernel code in the function definition, DONT remove any python imports or helper utils in the instruction/code provided, DONT change/interfere with the provided function definition and parameter list.

This kernel, `rms_kernel`,  is designed to perform Root Mean Square (RMS) Normalization.

**Your objective is to implement the body of `rms_kernel`.**

You must ensure that:
1.  All arguments received by `rms_kernel` are kept intact and not modified.
2. Provide you final code in ```python code block. 
Example:
```python
<YOUR-CODE-HERE>
```
The full definition for `rms_kernel` and relevant helper utilities are provided in the context below. You only need to complete the code for `rms_kernel` whilst keeping other things intact. DONT remove Imports and HELPER utils.

####################### Imports #####################
import argparse
import torch
import sys
import pytest
from itertools import product

import triton
import triton.language as tl
####################### Imports #####################

############################ HELPER utils ############################


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def get_num_sms():
    current_device_index = torch.cuda.current_device()
    current_device = torch.cuda.get_device_properties(current_device_index)
    num_sms = current_device.multi_processor_count
    return num_sms


def get_cuda_autotune_config():
    return [
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=16, num_stages=1),
    ]


def get_hip_autotune_config():
    return [triton.Config({'waves_per_eu': we}, num_warps=nw) for (we, nw) in product([0, 1, 2, 4], [4, 8, 16])]


def get_autotune_config():
    if is_cuda():
        return get_cuda_autotune_config()
    else:
        return get_hip_autotune_config()
############################ HELPER utils ############################


@triton.autotune(configs=get_autotune_config(), key=['n_rows', 'n_cols'], use_cuda_graph=True)
@triton.jit
def rms_kernel(output_ptr, input_ptr, g_ptr, rsigma_ptr, input_row_stride, output_row_stride, n_rows, n_cols, epsilon,
               ZERO_CENTERED_GAMMA: tl.constexpr, BLOCK_SIZE: tl.constexpr, USE_BLOCKED: tl.constexpr,
               NUM_PRGMS: tl.constexpr):
    """
    Triton kernel for performing Root Mean Square (RMS) Normalization.

    This kernel normalizes each row of the input tensor by its RMS value,
    applies a learnable scaling factor (gamma), and stores the result.
    It also stores the reciprocal of the standard deviation (rsigma) for each row.
    The kernel supports two modes of operation: a simple row-wise processing
    and a blocked processing for potentially better performance on wider rows.
    It is designed as a persistent kernel where each program instance can handle
    multiple rows.

    Parameters:
    output_ptr: Pointer to the output tensor where the normalized values will be stored.
                Shape: (n_rows, n_cols)
    input_ptr: Pointer to the input tensor.
               Shape: (n_rows, n_cols)
    g_ptr: Pointer to the gamma (scale) tensor. This is a 1D tensor.
           Shape: (n_cols,)
    rsigma_ptr: Pointer to store the reciprocal of the standard deviation (or equivalent normalization factor)
                for each row. This is a 1D tensor.
                Shape: (n_rows,)
    input_row_stride: Stride in number of elements to move from one row to the next in the input_ptr.
    output_row_stride: Stride in number of elements to move from one row to the next in the output_ptr.
    n_rows: The number of rows in the input and output tensors.
    n_cols: The number of columns in the input and output tensors.
    epsilon: A small float value added to the variance to prevent division by zero during normalization.
    ZERO_CENTERED_GAMMA: tl.constexpr
                         A compile-time boolean constant. If True, 1.0 is added to the gamma values
                         before applying them, effectively making the provided gamma values adjustments
                         around a mean of 1.0.
    BLOCK_SIZE: tl.constexpr
                A compile-time integer constant representing the size of blocks used for processing
                columns. This is relevant for both the blocked and non-blocked execution paths
                (e.g., for `tl.arange`).
    USE_BLOCKED: tl.constexpr
                 A compile-time boolean constant. If True, the kernel uses a blocked algorithm
                 to iterate over columns, potentially improving cache utilization and performance
                 for rows with many columns. If False, a simpler, direct row-wise computation is performed.
    NUM_PRGMS: tl.constexpr
               A compile-time integer constant. This represents the number of program instances
               (effectively, persistent thread blocks) launched. Rows are distributed among these
               program instances. For example, program `pid` handles rows `pid, pid + NUM_PRGMS, ...`.
    """
    # Your code here





All 182 declared correctness cases now execute, including the 84 previously skipped
performance cases with different input and output dtypes. The harness allocates the
requested output dtype. The independent oracle selects the original numerical gate
by output dtype, exactly as test_rmsnorm does: fp16/bf16 use atol=1e-3, rtol=1e-2;
fp32 uses atol=rtol=1e-5. Previously executed same-dtype gates, the full case table,
seeds, candidate kernels, warmups and device timing policy remain unchanged.

The protected direct RMS reference now includes its `epsilon` argument inside
the square root, matching the candidate's formula. Direct correctness retains
epsilon 1e-6; performance retains its explicit 1e-5 and existing independent
reference. Near-zero CPU known answers detect the old omission, including the
old nonfinite result on an all-zero row. All 182 required cases, 126 scored
cases, dtype combinations, seeds, output gates and candidate timing remain
unchanged. This reference correction requires fresh GPU qualification.
