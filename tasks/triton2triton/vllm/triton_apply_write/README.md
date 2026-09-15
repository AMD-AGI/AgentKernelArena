# triton_apply_write

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton kernel `_apply_write_kernel` for maximum GPU throughput
while maintaining correctness.

The kernel applies staged writes to a 2D output tensor. Each program (one per write)
copies a variable-length segment from a flat write_contents buffer into a specific
row and column offset of the output tensor. Cumulative lengths (write_cu_lens) define
the boundaries within write_contents for each write operation.

Key optimization opportunities:
- Block size tuning for different content lengths
- Vectorized loads/stores
- Memory coalescing

Constraints:
- Must maintain the same function signature for `apply_write`
- Output must match reference exactly


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.

