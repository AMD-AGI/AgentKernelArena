# triton_w8a8_block_scaled_mm

The starting candidate is implemented Triton. Improve the declared device kernel in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton W8A8 block-scaled FP8 matmul kernel
`_w8a8_triton_block_scaled_mm` for maximum GPU throughput.

The kernel performs block-wise quantized matrix multiplication where both
A and B have per-block scaling factors.

Constraints:
- Must maintain the same function signature for `w8a8_triton_block_scaled_mm`
- Output must match reference within atol=1e-1, rtol=1e-1


## LLM-shape variant

This is the LLM-shape variant of `triton2triton/vllm/triton_w8a8_block_scaled_mm`. Its
cases are linear layers of DeepSeek-V3 with 128x128 FP8 blocks, the six (N, K) pairs
DeepGEMM tests: (2112, 7168), (24576, 1536), (32768, 512), (7168, 16384), (4096, 7168)
and (7168, 2048), each at M = 128 (decode) and M = 4096 (prefill). The task is validated
on gfx942 only.

Compute the operator in Triton kernels the candidate defines and launches. Do not
call a PyTorch/ATen matrix-product, softmax or attention operator, `torch.ops`, or
another kernel library (rocBLAS, hipBLASLt, aiter, flash-attn); PyTorch may only
allocate outputs, check inputs, and make views or contiguous copies. The harness
guard refuses a candidate that adds such a call.

## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all 12 cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.


## Complete outputs and measured replay

Correctness checks every output element, shape, dtype, device and finiteness
against the CPU block-dequantization/matmul oracle
(the original loops, vectorized), using pristine
operands and scales. All four input tensors must remain unchanged. The original
`atol=rtol=1e-1` rule is retained. Unscored controls add a contiguous higher-rank
A, partial K/N blocks, a 64-by-64 scale block, zero A and FP32/BF16 output.
The FP8 wrapper also checks a noncontiguous B supported by its strides.

The scored shapes are this variant's; seeds and input magnitudes are the
original's, including the larger performance scales/operands. Ten warmups and 100 device-timed samples
still measure device work replayed from the original public operator call.
Graph capture executes the protected host wrapper and allocates its output; graph
replay measures the captured device work and does not repeat that Python allocation.
The event fallback measures the callable's GPU-stream event interval and can include
stream gaps; it is not an independent CPU wall-clock or allocation-cost metric.
The protected wrapper collects that output from the exact measured graph,
checks it against the reference, then changes A/B/As/Bs, poisons the captured
output and checks the same graph's replay. These checks are outside timing.
Input buffers are verified read-only and restored even on replay failure.
Numerical failures or FP16 overflow in any scored case remain failures;
the check does not substitute an easier timing workload.


## Editable implementation boundary

Only `_w8a8_triton_block_scaled_mm` and permitted new implementation helpers are
editable. Existing imports, dtype selection, host launch configuration and public
wrapper remain protected with the numerical checker and timer. This keeps module
execution and wrapper changes from replacing the protected oracle or measured
call. Device tiling arithmetic, memory operations and compiler-approved helpers
remain available inside the declared kernel boundary. This is a reproducibility
boundary, not a security sandbox for arbitrary code.
