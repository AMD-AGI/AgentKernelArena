# triton_convert_req_to_global_index

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton kernel `_convert_req_index_to_global_index_kernel` for
maximum GPU throughput while maintaining numerical correctness.

This kernel translates request-local token indices to global cache slot indices
using a block table lookup:
  out[i,j] = block_table[req_id[i], tok[i,j] // BLOCK_SIZE] * BLOCK_SIZE
             + tok[i,j] % BLOCK_SIZE

Invalid tokens (== -1) or out-of-bounds block indices produce -1 in the output.
Optionally counts valid (non -1) entries per row via atomic add.

Key optimization opportunities:
- Coalesced memory access patterns
- Tile width (BLOCK_N) tuning
- Minimizing atomic contention for valid count tracking

Constraints:
- Must maintain the same function signature for `convert_req_to_global_index`
- Output must exactly match reference (int32 indices)


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.


Protected checks validate shape, dtype, device and exact values for the ordinary
output and for **both** outputs when `return_valid_counts=True`. Each original
correctness case additionally uses cloned inputs with shifted request/table
mapping and negative, exactly out-of-range and farther out-of-range token indices.
These unscored diagnostics preserve original scored shapes, seeds and exact gates.

Performance retains the existing invocation, warmups and samples. It checks the
actual timed result against a pristine scalar-reference snapshot, then perturbs
the same input buffers with those boundary values and poisons the output before
replaying the exact captured graph. Inputs are checked for mutation and restored
in `finally`. Original kernel and generated helper sources remain unchanged.
