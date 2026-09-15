# triton_prepare_eagle_inputs

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton `_prepare_eagle_inputs_kernel` for maximum GPU throughput.
This kernel shifts target input_ids by one position, copies positions, and
computes last token indices for EAGLE speculative decoding.

Key optimization opportunities:
- Block size tuning for the copy loops
- Vectorized memory access
- Minimizing serial dependencies

Constraints:
- Must maintain the same function signature for `prepare_eagle_inputs`
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

