# rmsnorm_bwd

The on-disk starting functions contain implemented Triton code. The v2 declaration
therefore uses `implemented` and a frozen `initial_candidate` baseline, regardless
of this directory's historical suite name. Edit only the declared function scopes
(and permitted implementation helpers) in `rmsnorm_bwd.py`. Preserve signatures,
references, input generation, assertions, test parameters and timing policy.
Deliver edits to those files; a fenced code block alone is not a submission.

## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
with one role and one action. Submitted checks use `ARENA_EVAL_PHASE=candidate_evaluation`.
The adapter emits `arena-eval-v1`; Arena owns final score/validation reports.
`workloads.json` retains 82 original collected cases, including 42 performance cases.
Collection is checked against this independent manifest. Original forward/autograd correctness
assertions and dtype gates remain, with pristine input snapshots. Performance inputs additionally run the task-local
oracle in `_arena_reference.py`, before timing, against actual timed backward buffers, and after changed-input replay.
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


You are an expert in triton programming language. You will be given the function definition for the `rms_bwd_kernel`. Your task is to complete the kernel code. Only complete the kernel code in the function definition, DONT remove any python imports or helper utils in the instruction/code provided, DONT change/interfere with the provided function definition and parameter list.

This kernel, `rms_bwd_kernel`,  is designed to calculate the backward pass for RMS Normalization

**Your objective is to implement the body of `rms_bwd_kernel`.**

You must ensure that:
1.  All arguments received by `rms_bwd_kernel` are kept intact and not modified.
2. Provide you final code in ```python code block.
Example:
```python
<YOUR-CODE-HERE>
```
The full definition for `rms_bwd_kernel` and relevant helper utilities are provided in the context below. You only need to complete the code for `rms_bwd_kernel` whilst keeping other things intact. DONT remove Imports and HELPER utils.

Okay, here's the setup for implementing rms_bwd_kernel, along with descriptions of the related kernels.

Provided Kernels (Assumed to be already implemented):

rms_fwd_kernel:

Purpose: This kernel performs the forward pass of RMS Normalization.

Given an input tensor x and a learnable scaling vector g, it computes the normalized output y and the reciprocal of the root mean square rsigma for each row.

The formula is roughly y_i = (x_i / sqrt(mean(x^2) + epsilon)) * effective_g_i, where effective_g_i is g_i or g_i + 1 depending on ZERO_CENTERED_GAMMA. rsigma is 1.0 / sqrt(mean(x^2) + epsilon).

_rmsnorm_bwd_dg_reduce:

Purpose: This kernel performs a reduction operation specifically for the gradient of the scaling parameter g.

It takes an intermediate gradient dg_tmp (which has dimensions n_rows x n_cols and is computed by rms_bwd_kernel) and sums it along the row dimension.

The result is the final gradient dg (with dimensions n_cols) for the learnable scaling parameter g.

Kernel to Implement: rms_bwd_kernel (function definition below)

####################### Imports #####################
import argparse
import torch
import sys
import pytest
from itertools import product

import triton
import triton.language as tl
####################### Imports #####################

@triton.jit
def rms_bwd_kernel(
    grad_output_ptr,    # Pointer to the gradient of the loss w.r.t. the output of RMSNorm (dL/dy).
                        # Shape: (n_rows, n_cols)
    input_ptr,          # Pointer to the input tensor 'x' from the forward pass.
                        # Shape: (n_rows, n_cols)
    g_ptr,              # Pointer to the learnable scaling parameter 'g'.
                        # Shape: (n_cols,)
    rsigma_ptr,         # Pointer to the reciprocal of the root mean square (1 / sqrt(mean(x^2) + eps))
                        # computed in the forward pass. Shape: (n_rows,)
    dx_ptr,             # Pointer to store the computed gradient of the loss w.r.t. the input 'x' (dL/dx).
                        # Shape: (n_rows, n_cols)
    dg_ptr,             # Pointer to store the computed intermediate gradient of the loss w.r.t. the
                        # scaling parameter 'g' (dL/dg_tmp). This is before reduction across rows.
                        # Shape: (n_rows, n_cols)
    input_row_stride,   # Stride of the 'input_ptr' and 'dx_ptr' tensors along the row dimension.
    output_row_stride,  # Stride of the 'grad_output_ptr' tensor along the row dimension.
                        # Note: 'dg_ptr' also uses 'input_row_stride' if it has the same layout as 'x'.
    n_rows,             # Total number of rows to process (e.g., batch_size * sequence_length).
    n_cols,             # Total number of columns (features) per row (e.g., hidden_dimension).
    ZERO_CENTERED_GAMMA: tl.constexpr, # Compile-time boolean. If True, effective gamma is (g + 1), otherwise it's 'g'.
    BLOCK_SIZE: tl.constexpr,          # Compile-time constant. Defines the size of blocks used for processing columns.
                                       # This is typically a power of 2, e.g., 1024.
    USE_BLOCKED: tl.constexpr,         # Compile-time boolean. If True, indicates a specialized blocked algorithm
                                       # should be used for computing sums over columns, potentially involving
                                       # multiple passes. This is often beneficial for large 'n_cols'.
    NUM_PRGMS: tl.constexpr            # Compile-time constant. The number of program instances launched by Triton.
                                       # Used for distributing row processing across different programs.
):
    """
    Computes the backward pass for RMS Normalization, calculating the gradients
    with respect to the input 'x' (dL/dx) and an intermediate gradient
    with respect to the scaling parameter 'g' (dL/dg_tmp).

    The core computations for dL/dx_i (grad_input) and dL/dg_i (dg) are:
    Let norm_factor = rsigma
    Let effective_g = g (or g + 1 if ZERO_CENTERED_GAMMA)

    1. grad_sum_per_row = sum_cols(grad_output * input * effective_g)
    2. dL/dx = grad_output * norm_factor * effective_g - (norm_factor^3 * input / n_cols) * grad_sum_per_row
    3. dL/dg_intermediate = grad_output * input * norm_factor

    This kernel handles parallelization over rows and, depending on USE_BLOCKED,
    may use a blocked approach for iterating over columns to manage memory and
    computation efficiently, especially for the `grad_sum_per_row` calculation.

    The `dg_ptr` output of this kernel (dL/dg_intermediate) will typically be
    further processed by `_rmsnorm_bwd_dg_reduce` to sum contributions across
    all rows to get the final dL/dg.
    """
    # Your code here






## Actual backward workload and output contract

All 82 case identities, 42 scored shape/dtype/gamma combinations, seeds and
numerical thresholds remain. The task's editable target is `rms_bwd_kernel`.
Both suite variants now measure that backward launch on precomputed protected
forward `rsigma`, with identical warmup 10, repetition 100 and canonical event
timing. The instruction variant already had this workload. The triton variant
previously measured only the protected forward function; its timing workload is
explicitly corrected to backward using the existing instruction harness.
Those historical forward times are not comparable to this revised task.

The public backward outputs are full `dx` and per-row FP32 `dg_tmp`. Both are
checked elementwise from private x/g/grad_output/rsigma snapshots, using the
unchanged FP16/BF16 1e-3 absolute / 1e-2 relative gate or FP32 1e-5 / 1e-5 gate.
The prior wrapper instead summed dg_tmp outside the timed kernel and compared
against a differently associated expression; cancellation in that extra
reduction created failures while leaving per-row output errors unexamined.
The original complete autograd tests, including their reduced-gradient gates,
remain separately required and unchanged in numerical criteria.

The protected forward rsigma input is independently checked against the RMS
formula. Actual timed dx and dg_tmp buffers are both checked, poisoned and
replayed after fresh x/g/grad_output plus a consistent newly computed rsigma.
Inputs and both outputs are restored in `finally`. The Triton launch's Python
return is a kernel handle; the checked outputs are its bound device buffers.
No oracle, poison, reduction or reset is inserted into the timed callable.
Candidate kernel code is unchanged. Each revised task freezes its own initial
baseline and evaluates both roles under the same current workload.
