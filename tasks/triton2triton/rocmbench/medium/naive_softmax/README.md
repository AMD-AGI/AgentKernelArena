# naive_softmax

The on-disk starting functions contain implemented Triton code. The v2 declaration
therefore uses `implemented` and a frozen `initial_candidate` baseline, regardless
of this directory's historical suite name. Edit only the declared function scopes
(and permitted implementation helpers) in `naive_softmax.py`. Preserve signatures,
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


You are an expert in triton programming language. You will be given the function definition for the `softmax_kernel_naive` kernel. Your task is to optimize the kernel code for better performance while preserving correctness. Only optimize the kernel code in the function definition, DONT remove any python imports or helper utils in the instruction/code provided, DONT change/interfere with the provided function definition and parameter list.

This kernel, `softmax_kernel_naive`,  naive softmax operation.

**Your objective is to optimize the body of  the kernel `softmax_kernel_naive`.**

You must ensure that:
1.  All arguments received by `softmax_kernel_naive` are kept intact and not modified.
2. Provide your final code in ```python code block. 
Example:
```python
<YOUR-CODE-HERE>
```
The full definitions for `softmax_kernel_naive` and relevant helper utilities are provided in the context below. You only need to optimize the code for `softmax_kernel_naive` whilst keeping other things intact. DONT remove Imports and HELPER utils.

######################################## Imports ########################################
import pytest
import torch
from torch.testing import assert_close

import triton
import triton.language as tl

dtype_mapping = {
    'float16': torch.float16,
    'float32': torch.float32,
}
######################################## Imports ########################################


 @triton.jit
def softmax_kernel_naive(in_ptr, output_ptr, row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    """
    Computes the softmax function over the last dimension of a 2D input tensor.

    Each program instance is responsible for processing a single row of the input tensor.

    Parameters
    ----------
    in_ptr
        Pointer to the 2D input tensor.
    output_ptr
        Pointer to the 2D output tensor where the result is stored. The dimensions
        of this tensor should match the input tensor.
    row_stride
        The number of elements to skip in memory to move from the start of one
        row to the start of the next. This is used to correctly index into the
        input and output tensors.
    n_cols
        The size of the last dimension of the tensor (i.e., the number of columns
        in each row).
    BLOCK_SIZE : tl.constexpr
        A compile-time constant that defines the size of the data block that each
        instance processes in a single operation. This is used to tile the
        computation over the columns of a row.
    """    # Each program instance processes a single row of the input tensor.
    # 1. Get the row index
    row_idx = tl.program_id(axis=0)

    # 2. Compute offsets for the current row.
    # The naive kernel assumes that the number of columns is a power of 2.
    # and that `BLOCK_SIZE` is equal to `n_cols`.
    row_start_ptr = in_ptr + row_idx * row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets

    # 3. Load the row into a 1D block.
    # `mask` is used to handle rows where `n_cols` is not a power of 2.
    mask = col_offsets < n_cols
    # load the input data; use `other=-float('inf')` to ensure correct `max` calculation
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

    # 4. Compute softmax.
    #    a. Subtract the maximum value for numerical stability.
    row_minus_max = row - tl.max(row, axis=0)
    #    b. Compute the numerator.
    numerator = tl.exp(row_minus_max)
    #    c. Compute the denominator.
    denominator = tl.sum(numerator, axis=0)
    #    d. Normalize.
    softmax_output = numerator / denominator

    # 5. Write the result to the output tensor.
    output_row_start_ptr = output_ptr + row_idx * row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)




