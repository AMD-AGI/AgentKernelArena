# softmax

The on-disk starting functions contain implemented Triton code. The v2 declaration
therefore uses `implemented` and a frozen `initial_candidate` baseline, regardless
of this directory's historical suite name. Edit only the declared function scopes
(and permitted implementation helpers) in `softmax.py`. Preserve signatures,
references, input generation, assertions, test parameters and timing policy.
Deliver edits to those files; a fenced code block alone is not a submission.

## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
with one role and one action. Submitted checks use `ARENA_EVAL_PHASE=candidate_evaluation`.
The adapter emits `arena-eval-v1`; Arena owns final score/validation reports.
`workloads.json` retains 31 original collected cases, including 21 performance cases.
Collection is checked against this independent manifest. The 10 original FP32 correctness cases keep their original allclose assertion,
now using an independent pre-invocation reference snapshot. All 21 performance
cases also check full output and read-only input before timing, against the
actual measured output, and after changed-input replay.
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

## Numerical and replay contract

The mathematical operator is row-wise softmax. Preserve the allocating public
wrapper, kernel, launch/autotune policy, all 10 FP32 correctness cases, and all
21 performance cases (seven shapes times fp16/bf16/fp32). No shape, seed, warmup,
sample count or timed allocation boundary changes.

Original `test_softmax` creates FP32 input, computes `torch.softmax`, and uses
`torch.allclose` with its defaults: `atol=1e-8`, `rtol=1e-5`. It does not contain
an FP16/BF16 numerical rule: those dtypes originally appeared only in timing.
The migrated wrapper incorrectly applied the FP32 rule directly to two already
rounded low-precision outputs. This repair keeps the FP32 gate and explicitly
extends it through the required output storage conversion:

1. From a pristine input snapshot, compute `r = torch.softmax(x.float(), dim=1)`.
2. For FP32 output, require the original full-tensor allclose rule.
3. For FP16/BF16, compute `d = 1e-8 + 1e-5 * abs(r)` in FP32. Every output element
   must lie between `(r-d).to(output_dtype)` and `(r+d).to(output_dtype)`.

Thus a low-precision value must be obtainable by rounding a value within the
original FP32 accuracy interval. This is an explicit extension of the original
FP32 contract, not a historical low-precision threshold, a dtype-wide tolerance
chosen from baseline failures, or permission to return arbitrary nearby values.
Shape, dtype, device, finite output and a separate output allocation are checked.
The same rule applies to initial baseline, final candidate, and timed replay.
[PyTorch allclose](https://docs.pytorch.org/docs/2.11/generated/torch.allclose.html)
defines the preserved FP32 interval;
[softmax dtype](https://docs.pytorch.org/docs/2.11/generated/torch.nn.functional.softmax.html)
defines input casting for the independent reference.

Input bytes are read-only and references use private pre-call snapshots. The
canonical `TimedRun` exposes the actual measured tensor, including the original
wrapper's allocation behavior. Outside timing, replay changes column values by
reversing/negating the input and adding column-varying offsets (a constant row
shift would not change softmax). It recomputes the independent reference,
poisons the measured output, and reruns the bound measured invocation. Input and
poisoned output are restored in `finally`, including failures. The same original
inputs are used for baseline/candidate timing with ten warmups, 100 samples and
unchanged graph calibration/batching defaults. Explicit observable event timing
is supported; unobservable automatic graph fallback fails. Single-column
correctness cases retain their mathematically constant output of one.

## Original operator instructions

The historical instructions below retain the operator semantics and interface;
the v2 on-disk edit and evaluation contract above governs submission format.


You are an expert in triton programming language. You will be given the function definition for the `softmax_kernel_online`. Your task is to complete the kernel code. Only complete the kernel code in the function definition, DONT remove any python imports or helper utils in the instruction/code provided, DONT change/interfere with the provided function definition and parameter list.

This kernel, `softmax_kernel_online`,  is designed to perform softmax function on the input tensor in an online, numerically stable manner.

**Your objective is to implement the body of `softmax_kernel_online`.**

You must ensure that:
1.  All arguments received by `softmax_kernel_online` are kept intact and not modified.
2. Provide you final code in ```python code block.
Example:
```python
<YOUR-CODE-HERE>
```


The full definition for `softmax_kernel_online` and relevant helper utilities are provided in the context below. You only need to complete the code for `softmax_kernel_online` whilst keeping other things intact.


#Imports
import argparse
import torch
import sys
import pytest

import triton
import triton.language as tl

######################################## HELPERS utils ########################################
def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def is_cdna():
    return is_hip() and triton.runtime.driver.active.get_current_target().arch in ('gfx940', 'gfx941', 'gfx942',
                                                                                   'gfx90a', 'gfx908')


def get_cuda_autotune_config():
    return [
        triton.Config({}, num_warps=4, num_stages=1),
        triton.Config({}, num_warps=8, num_stages=1),
        triton.Config({}, num_warps=16, num_stages=1),
    ]


def get_hip_autotune_config():
    return [
        triton.Config({'waves_per_eu': 1}, num_warps=4, num_stages=1),
        triton.Config({'waves_per_eu': 1}, num_warps=8, num_stages=1),
        triton.Config({'waves_per_eu': 1}, num_warps=16, num_stages=1),
        triton.Config({'waves_per_eu': 2}, num_warps=4, num_stages=1),
        triton.Config({'waves_per_eu': 2}, num_warps=8, num_stages=1),
        triton.Config({'waves_per_eu': 2}, num_warps=16, num_stages=1),
        triton.Config({'waves_per_eu': 4}, num_warps=4, num_stages=1),
        triton.Config({'waves_per_eu': 4}, num_warps=8, num_stages=1),
        triton.Config({'waves_per_eu': 4}, num_warps=16, num_stages=1),
    ]


def get_autotune_config():
    if is_cuda():
        return get_cuda_autotune_config()
    else:
        return get_hip_autotune_config()

######################################## HELPERS utils ########################################


@triton.autotune(configs=get_autotune_config(), key=['n_rows', 'n_cols'], use_cuda_graph=True)
@triton.jit
def softmax_kernel_online(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols,
                          BLOCK_SIZE: tl.constexpr):
    """
    Computes the softmax function for each row of the input tensor in an online, numerically stable manner.

    This Triton kernel processes each row of the input tensor independently. For each row,
    it iterates through the columns in blocks. In the first pass over the blocks,
    it computes the row-wise maximum value and the sum of exponentials (scaled by the
    running maximum) in an online fashion. This online update of the sum involves
    rescaling previous sums if a new, larger maximum is found.
    In the second pass, it subtracts the final row-wise maximum, exponentiates,
    divides by the final sum of exponentials, and stores the result.
    This approach is numerically stable, especially for inputs with large variations in magnitude.
    Each program instance (kernel launch) is responsible for processing a single row.

    Parameters:
    -----------
    output_ptr : tl.pointer_type
        Pointer to the output tensor where the softmax results will be stored.
        Expected to be of a floating-point type (e.g., float32).
    input_ptr : tl.pointer_type
        Pointer to the input tensor.
        Expected to be of a floating-point type (e.g., float32).
    input_row_stride : int
        The stride (in number of elements) to move from one row to the next in the input tensor.
    output_row_stride : int
        The stride (in number of elements) to move from one row to the next in the output tensor.
    n_rows : int
        The total number of rows in the input (and output) tensor. The kernel is typically
        launched with `n_rows` program instances in the first dimension.
    n_cols : int
        The total number of columns (features) in each row of the input tensor.
    BLOCK_SIZE : tl.constexpr
        The size of the blocks into which each row is divided for processing during the
        online computation. This is a compile-time constant and should ideally be a
        power of 2 for efficiency (e.g., 1024, 2048).
    """
    # Your code here.
