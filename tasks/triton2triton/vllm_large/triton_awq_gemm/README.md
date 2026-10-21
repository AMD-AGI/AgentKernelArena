# triton_awq_gemm

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton AWQ GEMM kernel `awq_gemm_kernel` for maximum GPU throughput
while maintaining numerical correctness.

The kernel performs matrix multiplication with 4-bit AWQ quantized weights,
dequantizing on-the-fly during the GEMM computation with split-K parallelism.

Key optimization opportunities:
- Tile size tuning (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)
- Split-K factor optimization
- Memory access pattern optimization

Constraints:
- Must maintain the same function signature for `awq_gemm_triton`
- Output must match the original harness gate, atol=1e-1 and rtol=1e-1 for float16.
  The previous README's 1e-2 values did not describe the executable comparison.


## LLM-shape variant

This is the LLM-shape variant of `triton2triton/vllm/triton_awq_gemm`. Its cases are
decode-time linear layers of Llama-3.1-70B with AWQ int4 weights (group size 128) on one
GPU, as (M, N, K): qkv (1, 10240, 8192), o (1, 8192, 8192), gate_up (1, 57344, 8192),
down (1, 8192, 28672) and down (32, 8192, 28672). split_k is 8, the value vLLM passes;
on ROCm vLLM runs this kernel for batches below 256 tokens. The weights are a Gaussian
quantized the way AWQ does it, so scales, zero points and outputs have a real layer's
magnitudes, and outputs must also stay within 5e-3 normwise relative error of the
reference. The task is validated on gfx942 only.

Compute the operator in Triton kernels the candidate defines and launches. Do not
call a PyTorch/ATen matrix-product, softmax or attention operator, `torch.ops`, or
another kernel library (rocBLAS, hipBLASLt, aiter, flash-attn); PyTorch may only
allocate outputs, check inputs, and make views or contiguous copies. The harness
guard refuses a candidate that adds such a call.

## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all 5 cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.

## Output and timed replay checks

The protected checks use an independent CPU AWQ unpacking and GEMM
reference (the original loops, vectorized) with the original numerical gate
and a 5e-3 normwise relative error bound. They require the complete FP16
output on the input device and reject changes to activations, packed weights,
scales or packed zero points. Unscored correctness probes additionally exercise
35-row/24-column tails and split-K factors 1, 2 and 4, using deterministic inputs.
The scored shapes and weights are this variant's; the input seeds are the original's.

Performance observes the result of the actual timed public wrapper, including
its output allocation and split reduction. It compares that result against the
pristine-input reference, then changes all four input buffers, poisons the output
and numerically checks the same bound invocation on replay. Inputs and wrapper
hooks are restored on success or failure. The original ten warmups, 100 samples
and canonical graph/event timing parameters are retained. References, checks and
the extra replay run outside the measured interval. A fresh complete GPU task
validation is required for this task package.


### Editable device implementation boundary

The editable device functions are `awq_gemm_kernel`.
Their bodies and additional implementation helpers may be optimized. Module
imports, initialization, host dispatch and pointer preparation remain protected,
along with the existing numerical checks and benchmark protocol. The structural
guard enforces this boundary; it does not isolate arbitrary Python execution.

The protected wrapper also fixes the launch configuration: the grid and the tile
sizes, `num_warps` and `num_stages` it passes at launch. Optimization notes above
that mention tuning those values refer to the original whole-file contract; under
this boundary, optimize the kernel bodies and implementation helpers instead.
