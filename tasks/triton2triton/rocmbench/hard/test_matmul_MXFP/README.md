# test_matmul_MXFP

The on-disk starting functions contain implemented Triton code. The v2 declaration
therefore uses `implemented` and a frozen `initial_candidate` baseline, regardless
of this directory's historical suite name. Edit only the declared function scopes
(and permitted implementation helpers) in `test_matmul_MXFP.py`. Preserve signatures,
references, input generation, assertions, test parameters and timing policy.
Deliver edits to those files; a fenced code block alone is not a submission.

## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
with one role and one action. Submitted checks use `ARENA_EVAL_PHASE=candidate_evaluation`.
The adapter emits `arena-eval-v1`; Arena owns final score/validation reports.
`workloads.json` retains 9 original collected cases, including 6 performance cases.
Collection is checked against this independent manifest. Original seeded correctness
cases and gates are retained, with pristine input snapshots and additional replay checks. Performance inputs additionally run the task-local
oracle in `_arena_reference.py`, before timing, against actual TimedRun output, and after typed input changes and NaN poisoning.
Seeds, case parameters, original assertions/tolerances, launch parameters,
prepare/reset callbacks, warmups and sample counts are unchanged.

Arena times the same Triton path in its independently frozen baseline workspace
and the edited candidate workspace. The old benchmark helper's optional PyTorch
peer timing is not an Arena baseline and is omitted by this adapter. Candidate
measurements still use the canonical helper and its original mean device latency.
Do not edit `performance_utils_pytest.py` or generated benchmark helpers.

Every declared case must execute. Missing hardware features, unsupported
combinations, missing references or incomplete execution cannot qualify this task.
A missing/empty final kernel never falls back to a reference or starting kernel.

## Original operator instructions

The historical instructions below retain the operator semantics and interface;
the v2 on-disk edit and evaluation contract above governs submission format.


You are an expert in triton programming language. You will be given the function definition for the `matmul_kernel,mxfp_to_bf16_kernel` kernels. Your task is to optimize the kernel code for better performance while preserving correctness. Only optimize the kernel code in the function definition, DONT remove any python imports or helper utils in the instruction/code provided, DONT change/interfere with the provided function definition and parameter list.

These kernels, `matmul_kernel,mxfp_to_bf16_kernel`,   performs matrix multiplication (C = A @ B) with optional support for scaled MXFP (Microscaling Format) inputs.

**Your objective is to optimize the body of both the kernels `matmul_kernel,mxfp_to_bf16_kernel`.**

You must ensure that:
1.  All arguments received by `matmul_kernel and mxfp_to_bf16_kernel` are kept intact and not modified.
2. Provide you final code in ```python code block.
Example:
```python
<YOUR-CODE-HERE>
```
The full definitions for `matmul_kernel,mxfp_to_bf16_kernel` and relevant helper utilities are provided in the context below. You only need to optimize the code for `matmul_kernel,mxfp_to_bf16_kernel` whilst keeping other things intact. DONT remove Imports and HELPER utils.

import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, scale_ptr, b_ptr, output_ptr,
    M, N, K_MXFP,
    stride_am, stride_ak,
    stride_sm, stride_sk,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr, a_type: tl.constexpr, b_type: tl.constexpr
):
    """
    Performs a matrix multiplication (C = A @ B) with optional support for scaled MXFP (Microscaling Format) inputs.
    If `a_type` and `b_type` are provided, it performs a scaled dot product where matrix A is scaled
    by `scale_ptr`. Otherwise, it performs a standard matrix multiplication.
    The computation is tiled and can be pipelined for efficiency.

    Parameters:
    ----------
    a_ptr: tl.pointer_type
        Pointer to the first input matrix (A).
    scale_ptr: tl.pointer_type
        Pointer to the scale tensor for matrix A. Used only if `a_type` and `b_type` are not None.
    b_ptr: tl.pointer_type
        Pointer to the second input matrix (B).
    output_ptr: tl.pointer_type
        Pointer to the output matrix (C).
    M: int
        Number of rows in matrix A and output matrix C.
    N: int
        Number of columns in matrix B and output matrix C.
    K_MXFP: int
        The inner dimension. If performing scaled MXFP matmul, this represents the number of MXFP vectors
        (groups of elements, e.g., 32 for FP8, or 16 for FP4 if DIV_FACTOR is 2 due to packing)
        along the K dimension of matrix A. Otherwise, it's the standard K dimension.
    stride_am: int
        Stride of matrix A along the M dimension (row stride).
    stride_ak: int
        Stride of matrix A along the K dimension (column/inner dimension stride).
    stride_sm: int
        Stride of the scale tensor along its M dimension (row stride for scales).
    stride_sk: int
        Stride of the scale tensor along its K dimension (stride between scale values).
    stride_bk: int
        Stride of matrix B along the K dimension (row stride).
    stride_bn: int
        Stride of matrix B along the N dimension (column/inner dimension stride).
    stride_cm: int
        Stride of the output matrix C along the M dimension (row stride).
    stride_cn: int
        Stride of the output matrix C along the N dimension (column/inner dimension stride).
    BLOCK_M: tl.constexpr
        Tile size for the M dimension.
    BLOCK_N: tl.constexpr
        Tile size for the N dimension.
    BLOCK_K: tl.constexpr
        Tile size for the K dimension (primarily for loading B and influencing A loads,
        also relates to how many elements are processed in the inner loop iteration).
    NUM_STAGES: tl.constexpr
        Number of stages for software pipelining.
    a_type: tl.constexpr (str or None)
        String specifying the MXFP type for matrix A (e.g., "e2m1", "e4m3", "e5m2").
        If None, standard matrix multiplication is assumed for A.
    b_type: tl.constexpr (str or None)
        String specifying the MXFP type for matrix B (e.g., "e4m3", "e5m2").
        If None, standard matrix multiplication is assumed for B.
    """
    # Your code here


@triton.jit
def mxfp_to_bf16_kernel(
    x_ptr,
    scale_ptr,
    mxfp_ptr,
    N,
    e_bits: tl.constexpr,
    m_bits: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Converts input data `x_ptr` (assumed to be in a packed MXFP format stored as uint8)
    to bfloat16 format, by applying scaling factors from `scale_ptr`.
    The kernel handles different MXFP types (e.g., FP8, FP4 variants) based on
    `e_bits` and `m_bits`. The output is stored in `mxfp_ptr`.

    Parameters:
    ----------
    x_ptr: tl.pointer_type
        Pointer to the input MXFP data, stored as uint8.
        Expected shape for processing: (N, 32) for FP8 or (N, 16) for FP4 (where 16 means 32 FP4 numbers packed).
    scale_ptr: tl.pointer_type
        Pointer to the scale values, stored as uint8. Expected shape: (N,).
    mxfp_ptr: tl.pointer_type
        Pointer to the output tensor where the bfloat16 results will be stored.
        Expected shape after processing: (N, 32).
    N: int
        The number of scale values, which typically corresponds to the number of rows or groups
        in the `x_ptr` data that are independently scaled.
    e_bits: tl.constexpr
        Number of exponent bits in the input MXFP format.
    m_bits: tl.constexpr
        Number of mantissa bits in the input MXFP format.
    BLOCK_SIZE: tl.constexpr
        The total number of bfloat16 elements in the output `mxfp_ptr` that a single
        program instance (or a block of threads launched by `tl.program_id(0)`) will compute and store.
        This is used to tile the processing of the output tensor.
    """
    # Your code here





The unscaled performance reference multiplies the original operands in FP32 and
casts the result to FP16, matching the kernel's accumulation/output contract.
It does not round FP32 inputs to FP16 before multiplication. The six unscaled
performance cases and their existing FP16 comparison defaults are unchanged.
The separately declared scaled pipeline case now attempts its real Triton
compilation on the selected backend. Unsupported lowering is a real failure;
the historical blanket non-CUDA skip no longer suppresses this required case.

The scaled reference reshapes packed operands into their 32-element scale
groups before decoding, then reconstructs the matrix. This fixes the matrix
path's previously invalid broadcasting without changing the decoder or gate.
The original seeded scaled comparison remains. An additional unscored control
uses exact one-valued decoded operands and ordinary E8M0 exponent 127, whose
answer is the logical K. It rejects zero-output implementations which the
original tiny-scale inputs alone could accept at the original 1e-2 tolerance.
Scored cases, launch parameters, warmups, samples and timing are unchanged.


## Scoring and branch coverage

The six original scored cases measure the **unscaled** branch of `matmul_kernel`
(three shapes × FP16/FP32 inputs; FP16 output). They retain all original tiles,
stages, warps, seed, warmup 10, repetition 100, device timer and mean reduction.
There is no separate performance score for conversion or scaled multiplication.
Those public branches remain required correctness contracts; an implementation
that removes either cannot pass. All nine existing case identities are retained;
three unscored converter controls cover E2M1/E4M3/E5M2, signed finite values,
E8M0 exponents 126/127/128 and the masked final row block, for 12 total cases.

The decoder's exact gate, scaled pipeline's 1e-2 absolute/relative gate, and
unscaled dtype-aware gates are unchanged. The original tiny-scale case and the
existing ordinary-scale all-one known answer both remain. Scaled replay uses
valid packed signs and channel-varying ordinary scales; converter replay uses
valid encoded sign flips. These inputs exercise real nonzero outputs without
changing numerical thresholds or any scored workload.

All oracles consume private snapshots before candidate execution; every declared
input is read-only and the full output's shape/dtype/device is checked. The
unscaled timed callable is checked again after fresh floating inputs and poisoned
output, using a newly computed FP32 matmul→FP16 reference. Inputs and output are
restored in `finally` after success or failure. These are untimed checks; no
reference or replay reset is inserted into the measured callable.
