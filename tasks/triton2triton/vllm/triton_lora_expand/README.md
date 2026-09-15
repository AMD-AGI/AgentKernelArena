# triton_lora_expand

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton `_lora_expand_kernel` for maximum GPU throughput.
This is the LoRA B (expand) kernel that multiplies the low-rank intermediate
activations by LoRA B weight matrices and accumulates into the output tensor.

The kernel computes output[token] += input[token] @ lora_b[lora_id].T for
each token assigned to a LoRA adapter, with support for multiple slices
(e.g., QKV projections) and optional input addition.

Key optimization opportunities:
- Block size tuning (BLOCK_M, BLOCK_N, BLOCK_K)
- Memory access patterns for the gather-based token indexing
- Efficient handling of the SAME_STRIDE vs per-slice stride path
- Compute type selection and casting

Constraints:
- Must maintain the same function signature for `lora_expand`
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

