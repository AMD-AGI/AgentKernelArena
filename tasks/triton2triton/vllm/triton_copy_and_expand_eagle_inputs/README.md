# triton_copy_and_expand_eagle_inputs

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton `copy_and_expand_eagle_inputs_kernel` for maximum GPU throughput.
This kernel copies and expands inputs from the target model to drafting buffers for
EAGLE speculative decoding, handling padding slots, parallel drafting tokens, and
rejected token regions.

Key optimization opportunities:
- Vectorized memory access
- Minimizing branch divergence across regions
- Block size tuning for token dimension

Constraints:
- Must maintain the same function signature for `copy_and_expand_eagle_inputs`
- Output must match reference exactly (integer computation)


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.


Protected checks require **all six** returned tensors with exact original values,
shapes, devices and dtypes: token IDs/positions/new indices/hidden mapping are
int32; rejected/masked flags are bool. They include the overallocated zero tails.
Both existing correctness shift modes remain exercised for every original case.

The original performance invocation still measures `shift_input_ids=False` with
unchanged shapes, seeds, launch bounds, allocation, warmups and samples. All six
actual timed outputs are checked, then poisoned and rechecked through the same
captured graph after changing tokens, positions and rejection boundaries.
References precede candidate execution; caller-owned inputs are checked for
mutation, and replay perturbations are restored in `finally` even on failure.
No scored workload, numerical rule, kernel or generated helper is changed.
