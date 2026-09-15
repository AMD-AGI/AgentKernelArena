# triton_fused_moe_gptq_awq

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton `fused_moe_kernel_gptq_awq` for maximum GPU throughput.
This kernel performs fused MoE GEMM with GPTQ/AWQ quantized weights,
supporting 4-bit and 8-bit weight-only quantization with scales and zero points.

Key optimization opportunities:
- Block size tuning
- Efficient dequantization pipeline
- Memory access patterns for packed weights

Constraints:
- Must maintain the same function signature for `fused_moe_gptq_awq`
- Output must match reference within atol=1.0, rtol=0.5


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.

