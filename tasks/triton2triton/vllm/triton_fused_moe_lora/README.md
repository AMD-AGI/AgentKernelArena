# triton_fused_moe_lora

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton `fused_moe_lora_kernel` for maximum GPU throughput.
This kernel fuses MoE expert routing with LoRA adapter computation,
performing both shrink (A) and expand (B) LoRA operations within the
MoE expert computation pipeline.

The kernel supports both naive block assignment (flat expert_ids) and
sorted token assignment, per-expert LoRA weights, optional routing
weight multiplication, split-K reduction, and L2 cache hints for
weight loading.

Note: The distributed communication paths (all_gather/all_reduce for
fully_sharded mode) are excluded; only the local compute path is tested.

Key optimization opportunities:
- Block size tuning (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)
- GROUP_SIZE_M for L2 cache reuse
- SPLIT_K factor for K-dimension parallelism
- USE_B_L2_CACHE for weight caching strategy
- Memory access patterns for grouped GEMM with pointer indirection

Constraints:
- Must maintain the same function signature for `fused_moe_lora`
- Output must match reference within atol=5e-2, rtol=5e-2 for float16


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.

