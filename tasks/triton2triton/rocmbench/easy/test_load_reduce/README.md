# test_load_reduce

The on-disk starting functions contain implemented Triton code. The v2 declaration
therefore uses `implemented` and a frozen `initial_candidate` baseline, regardless
of this directory's historical suite name. Edit only the declared function scopes
(and permitted implementation helpers) in `test_load_reduce.py`. Preserve signatures,
references, input generation, assertions, test parameters and timing policy.
Deliver edits to those files; a fenced code block alone is not a submission.

## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
with one role and one action. Submitted checks use `ARENA_EVAL_PHASE=candidate_evaluation`.
The adapter emits `arena-eval-v1`; Arena owns final score/validation reports.
`workloads.json` retains 43 original collected cases, including 42 performance cases.
Collection is checked against this independent manifest. The original correctness case now computes its oracle from an independent
pre-invocation input snapshot, so modifying the input cannot rewrite the expected
answer. All 42 performance inputs use the same protected reference, checked
before timing, against the actual measured output, and on changed-input replay.
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

## Reduction and replay contract

The operation is the maximum of each full input row: `y[i] = max(x[i, :])`.
The input is read-only. The single original correctness case is float16 128x64;
the 42 scored cases retain all 14 original block shapes and fp16/fp32/bf16.
All input generation uses the original seed and remains unchanged. The kernel,
launch wrapper and performance function are unchanged; this task still uses a
single-program grid and contiguous output. No general strided-output support
is claimed merely because the signature contains a stride argument.

Every full output comparison retains `rtol=1e-2`, `atol=1e-3`, `check_dtype=False`;
shape/device and finite-value checks remain active. Independent input snapshots
and byte comparisons reject input mutation, including mutations to non-maxima
that happen to leave the row maximum unchanged.

The canonical timer exposes the actual measured output through `TimedRun`.
Outside timing, the harness reverses/negates columns and adds deterministic,
row-varying signed offsets to the input at the same device addresses. It computes
a new oracle from an independent copy, poisons the whole output with NaNs and
replays the bound measured invocation. This challenges cached old maxima,
unwritten output, all-negative rows and input mutation. Input and output are
restored in `finally`, including failures. Baseline and candidate therefore time
the same original input values with unchanged ten warmups, 100 samples, graph
batching/calibration defaults and mean device latency. Replay control creates no
new scored cases. Unobservable graph-to-event fallback fails; explicitly selected
observable event timing retains its timing metadata.

## Original operator instructions

The historical instructions below retain the operator semantics and interface;
the v2 on-disk edit and evaluation contract above governs submission format.


You are an expert in triton programming language. You will be given the function definition for the `load_reduce_kernel` kernels. Your task is to optimize the kernel code for better performance while preserving correctness. Only optimize the kernel code in the function definition, DONT remove any python imports or helper utils in the instruction/code provided, DONT change/interfere with the provided function definition and parameter list.

These kernels, `load_reduce_kernel`,  performs a block-wise load followed by a row-wise maximum reduction.

**Your objective is to optimize the body of both the kernels `load_reduce_kernel`.**

You must ensure that:
1.  All arguments received by `load_reduce_kernel` are kept intact and not modified.
2. Provide you final code in ```python code block. 
Example:
```python
<YOUR-CODE-HERE>
```
The full definitions for `load_reduce_kernel` and relevant helper utilities are provided in the context below. You only need to optimize the code for `load_reduce_kernel` whilst keeping other things intact. DONT remove Imports and HELPER utils.

######################################## Imports ########################################

import pytest
import torch
from torch.testing import assert_close

import triton
import triton.language as tl
######################################## Imports ########################################

dtype_mapping = {
    'float16': torch.float16,
    'float32': torch.float32,
}


@triton.jit
def load_reduce_kernel(
    x_ptr,
    y_ptr,
    stride_xm,
    stride_xn,
    stride_y,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    This Triton kernel loads a 2D block of data from an input tensor `x_ptr`
    and performs a reduction operation (maximum) along one of its dimensions.

    Parameters
    ----------
    x_ptr
        Pointer to the input tensor X from which data will be loaded.
        This tensor is expected to be at least 2D.
    y_ptr
        Pointer to the output tensor Y where the reduced results will be stored.
        This tensor is expected to be 1D (or have a shape compatible with storing BLOCK_M elements).
    stride_xm
        Stride of the input tensor X along the M dimension (typically rows).
        It indicates the number of elements to skip in memory to move from one
        element to the next in the M dimension (e.g., from X[i, j] to X[i+1, j]).
    stride_xn
        Stride of the input tensor X along the N dimension (typically columns).
        It indicates the number of elements to skip in memory to move from one
        element to the next in the N dimension (e.g., from X[i, j] to X[i, j+1]).
    stride_y
        Stride of the output tensor Y.
        It indicates the number of elements to skip in memory to move from one
        element to the next in the output tensor Y (e.g., from Y[i] to Y[i+1]).
    BLOCK_M: tl.constexpr
        The size of the tile (or block) in the M dimension. This defines how many
        "rows" of the input data are processed by this kernel instance and consequently
        the number of output elements produced.
    BLOCK_N: tl.constexpr
        The size of the tile (or block) in the N dimension. This defines how many
        "columns" of the input data are processed for each "row" in the M dimension,
        and it's the dimension over which the reduction (max) is performed.
    """
    # Your code here




