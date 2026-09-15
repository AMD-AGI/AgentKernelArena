# test_chained_dot_fp8

The on-disk starting functions contain implemented Triton code. The v2 declaration
therefore uses `implemented` and a frozen `initial_candidate` baseline, regardless
of this directory's historical suite name. Edit only the declared function scopes
(and permitted implementation helpers) in `test_chained_dot_fp8.py`. Preserve signatures,
references, input generation, assertions, test parameters and timing policy.
Deliver edits to those files; a fenced code block alone is not a submission.

## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
with one role and one action. Submitted checks use `ARENA_EVAL_PHASE=candidate_evaluation`.
The adapter emits `arena-eval-v1`; Arena owns final score/validation reports.
`workloads.json` retains 24 original collected cases, including 20 performance cases.
Collection is checked against this independent manifest. Original collected FP8 correctness
cases and their absolute gate remain, with immutable inputs. Performance inputs additionally run the task-local
oracle in `_arena_reference.py`, before timing, against actual TimedRun output and after fresh-input replay.
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


You are an expert in triton programming language. You will be given the function definition for the `_chained_dot`. Your task is to complete the kernel code. Only optimize the kernel code in the function definition, DONT remove any python imports or helper utils in the instruction/code provided, DONT change/interfere with the provided function definition and parameter list.

This kernel, `_chained_dot`,  is designed to perform "chained dot product" operation.

**Your objective is to optimize the body of `_chained_dot`.**

You must ensure that:
1.  All arguments received by `_chained_dot` are kept intact and not modified.
2. Provide you final code in ```python code block. 
Example:
```python
<YOUR-CODE-HERE>
```


The full definition for `_chained_dot` and relevant helper utilities are provided in the context below. You only need to optimize the code for `_chained_dot` whilst keeping other things intact.

"""
Testing the (FP8) case of a dot op that consumes the output (MFMA) of
another dot op as an input.

"""
#Imports

import math
import pytest
import torch

import triton
import triton.language as tl

########################## HELPER utils ##########################
TORCH_HAS_FP8E4 = hasattr(torch, 'float8_e4m3fnuz')
float8: tl.constexpr = None if not TORCH_HAS_FP8E4 else torch.float8_e4m3fnuz
########################## HELPER utils ##########################

@triton.jit
def _chained_dot(
    Q,
    K,
    V,
    Out,
    q_desc,
    k_desc,
    v_desc,
    s_sc,
    s_desc,
    o_sc,
    stride_qz,
    stride_qm,
    stride_qd,
    stride_kz,
    stride_kn,
    stride_kd,
    stride_vz,
    stride_vd,
    stride_vn,
    stride_oz,
    stride_om,
    stride_od,
    Z,
    M,
    N,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_FP8: tl.constexpr,
):
    """
    This Triton kernel computes a "chained dot product" operation,
    effectively performing (Q @ K.T) @ V in a tiled manner.
    This is a core component of attention mechanisms.
    The kernel is parallelized across the M dimension of Q (target sequence length)
    and the Z dimension (batch size * number of heads).
    It iteratively loads blocks of K and V to compute partial results for the output.
    FP8 support allows for reduced precision computation with scaling factors.

    Parameters:
    -----------
    Q : tl.tensor
        Pointer to the Q (query) tensor. Expected shape (Z, M, D).
    K : tl.tensor
        Pointer to the K (key) tensor. Expected shape (Z, N, D).
    V : tl.tensor
        Pointer to the V (value) tensor. Expected shape (Z, N, D).
    Out : tl.tensor
        Pointer to the O (output) tensor. Expected shape (Z, M, D).
    q_desc : float
        Dequantization scale for the Q tensor (used if USE_FP8 is True).
    k_desc : float
        Dequantization scale for the K tensor (used if USE_FP8 is True).
    v_desc : float
        Dequantization scale for the V tensor (used if USE_FP8 is True).
    s_sc : float
        Scaling factor applied to the intermediate S (QK^T) tensor before dot product with V (used if USE_FP8 is True).
        This can be thought of as a quantization scale if S were to be stored in FP8.
    s_desc : float
        Dequantization scale for the intermediate S (QK^T) tensor when it's used in S@V (used if USE_FP8 is True).
    o_sc : float
        Quantization scale for the O (output) tensor (used if USE_FP8 is True).
    stride_qz : int
        Stride of the Q tensor along the Z (batch/head) dimension.
    stride_qm : int
        Stride of the Q tensor along the M (sequence length of Q / rows) dimension.
    stride_qd : int
        Stride of the Q tensor along the D (feature/embedding) dimension.
    stride_kz : int
        Stride of the K tensor along the Z (batch/head) dimension.
    stride_kn : int
        Stride of the K tensor along the N (sequence length of K / rows) dimension.
    stride_kd : int
        Stride of the K tensor along the D (feature/embedding) dimension.
    stride_vz : int
        Stride of the V tensor along the Z (batch/head) dimension.
    stride_vd : int
        Stride of the V tensor along the D (feature/embedding) dimension.
    stride_vn : int
        Stride of the V tensor along the N (sequence length of V / rows) dimension.
    stride_oz : int
        Stride of the Out tensor along the Z (batch/head) dimension.
    stride_om : int
        Stride of the Out tensor along the M (sequence length of Out / rows) dimension.
    stride_od : int
        Stride of the Out tensor along the D (feature/embedding) dimension.
    Z : int
        Size of the Z dimension (e.g., batch_size * num_heads).
    M : int
        Size of the M dimension (e.g., sequence length of Q, number of rows in Q).
    N : int
        Size of the N dimension (e.g., sequence length of K and V, number of columns in K.T / rows in V).
    BLOCK_D : tl.constexpr
        Tile size for the D dimension (feature/embedding dimension). Compile-time constant.
    BLOCK_M : tl.constexpr
        Tile size for the M dimension (rows of Q). Compile-time constant.
    BLOCK_N : tl.constexpr
        Tile size for the N dimension (columns of K.T / rows of V). Compile-time constant.
    USE_FP8 : tl.constexpr
        Boolean flag indicating whether to use FP8 E4M3 precision and apply scaling. Compile-time constant.
    """
    # Your code here







## Scaled FP8 and FP16 contract

The four original collected correctness cases are FP8. Their `atol=1e-2, rtol=0`
reference comparison remains, as does the scaled FP8 performance gate. Actual
output dtype/shape/device are checked **before** floating conversion for the
comparison, so returning an unquantized FP32 answer cannot satisfy FP8 output.
Both batches and every output element are checked.

The original performance suite also added random FP16 cases without a numerical
gate. Those use a private FP64 two-GEMM oracle with the mandatory FP16
intermediate/output. FP32 accumulation bounds are gamma(2*D) and gamma(2*N),
where gamma(n)=n*2^-24/(1-n*2^-24); intermediate rounding intervals propagate
through abs(V), then final endpoints round to FP16. This is the same explicit
rounding model as chained_matmul, never a tolerance fitted to observed errors.
Any original FP16 correctness branch's existing absolute assertion is retained.

All 24 original identities and 20 scored workloads, batch counts, RNG seeds,
scale selection, shapes, tile/warp/stage/instruction settings, allocating wrapper,
warmup 10, repetition 100 and canonical timer remain. Two unscored controls add
BATCH=2/M=17/N=35/D=16, signed values and meaningful non-unit FP8 scale products,
for 26 correctness cases. Block-pointer boundary checks and zero padding repair
unsafe public M/N tails without changing valid arithmetic or launch parameters.
Both roles freeze/evaluate this repaired kernel revision; old timings identify
an earlier baseline version.

Timed output is checked in full and replayed after fresh input plus NaN output
poison. FP8 replay negates Q exactly through its dtype, preserving range with
fixed scales; FP16 replay changes Q/K and recomputes the reference. All inputs
are read-only and inputs/poisoned outputs restore in `finally`, outside timing.

The added tail controls use signed deterministic Q/K/V with nonzero K/V offsets.
The independent answer is checked to be nonzero before launching the candidate,
so a cached zero result cannot make the sign-flip replay control vacuous.

### FP8 zero-padding compiler compatibility

Pinned Triton's block-pointer `padding_option="zero"` lowers to an integer
constant, which cannot be numerically cast directly to E4M3FNUZ. FP8 input
loads therefore reinterpret the same byte addresses as uint8, apply the same
M/N bounds with byte zero padding, and bitcast the result back to the original
FP8 element type. Byte zero encodes FP8 +0; all valid input encodings are
preserved exactly. FP16 loads, arithmetic, scale operations, public launch
arguments, cases, references and numerical gates are unchanged. This corrects
the previous masked-load revision's compilation failure; a fresh baseline is
required for this revision.
