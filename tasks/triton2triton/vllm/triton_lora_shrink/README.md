# triton_lora_shrink

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton `_lora_shrink_kernel` for maximum GPU throughput.
This is the LoRA A (shrink) kernel that projects input tokens from
hidden_size down to lora_rank using per-adapter LoRA A weight matrices.

The kernel computes output[slice, token] = scaling * input[token] @ lora_a[lora_id].T
for each token assigned to a LoRA adapter, with split-K reduction for
large hidden dimensions and support for multiple slices.

Key optimization opportunities:
- Block size tuning (BLOCK_M, BLOCK_N, BLOCK_K)
- SPLIT_K factor for K-dimension parallelism
- GROUP_SIZE_M for L2 cache reuse
- Memory access patterns for gather-based token indexing

Constraints:
- Must maintain the same function signature for `lora_shrink`
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

