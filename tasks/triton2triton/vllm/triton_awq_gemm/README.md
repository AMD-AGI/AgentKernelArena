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


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.

## Output and timed replay checks

The protected checks use the original independent CPU AWQ unpacking and GEMM
reference, with the original numerical gate. They require the complete FP16
output on the input device and reject changes to activations, packed weights,
scales or packed zero points. Unscored correctness probes additionally exercise
35-row/24-column tails and split-K factors 1, 2 and 4, using deterministic inputs.
The original five scored shapes, input seeds and parameter table are unchanged.

Performance observes the result of the actual timed public wrapper, including
its output allocation and split reduction. It compares that result against the
pristine-input reference, then changes all four input buffers, poisons the output
and numerically checks the same bound invocation on replay. Inputs and wrapper
hooks are restored on success or failure. The original ten warmups, 100 samples
and canonical graph/event timing parameters are retained. References, checks and
the extra replay run outside the measured interval. A fresh complete GPU task
validation is required for this task package.
