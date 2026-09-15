# triton_batch_memcpy

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton `batch_memcpy_kernel` for maximum GPU throughput.
This kernel performs batched memory copies: each program instance copies
one (src, dst, size) triple using byte-level pointer access.

Key optimization opportunities:
- Block size tuning
- Wider data types for aligned copies
- Memory coalescing patterns

Constraints:
- Must maintain the same function signature for `batch_memcpy`
- Output must match reference exactly (byte-level copy)


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.

The runner retains all destination buffers from the measured graph and compares
every byte with its corresponding source. After timing it changes every source,
clears the destinations and verifies the exact same graph replay. Pointer tables,
variable copy lengths, seeds, allocations, warmups and sample counts stay unchanged
for the original five timed workloads; the replay checks run outside device timing.
