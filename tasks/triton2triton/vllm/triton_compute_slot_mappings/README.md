# triton_compute_slot_mappings

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton kernel `_compute_slot_mappings_kernel` for maximum GPU throughput
while maintaining correctness.

The kernel computes slot mappings for paged KV cache. For each token position, it
computes: block_index = position // block_size, block_offset = position % block_size,
then looks up block_number from the block table and computes
slot_id = block_number * block_size + block_offset. The last program pads remaining
slots with PAD_ID (-1) for CUDA graph compatibility.

This is a simplified version with no context parallelism (TOTAL_CP_WORLD_SIZE=1).

Key optimization opportunities:
- Block size tuning
- Vectorized loads/stores
- Memory coalescing for block table lookups

Constraints:
- Must maintain the same function signature for `compute_slot_mappings`
- Output must match reference exactly (integer slot IDs)


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.


Protected checks preserve the original exact integer oracle and five scored
workloads. They additionally enforce output shape/dtype/device and validate
unscored diagnostic copies with reversed request-state mappings, a shifted
request boundary (unequal segment lengths), and nonzero positions crossing block
boundaries. The number of requests/tokens, allocated launch bound and scored input
seeds are unchanged. Returned tensors follow the actual wrapper's `num_tokens`
slice; internal padding capacity does not change the public result shape.

The actual timed output must match a pristine reference. After timing, the same
captured graph is replayed with those perturbed inputs and a poisoned output.
Both correctness and replay reject modified input buffers; perturbations are
restored in `finally`, including on failure. Original kernel, protected runner,
warmup/sample policy, case manifest and generated helpers remain unchanged.
