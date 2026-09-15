# triton_pack_seq

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton kernel `_pack_seq_kernel` for maximum GPU throughput
while maintaining numerical correctness.

The kernel packs variable-length sequences from a flat [N, D] tensor into a
batched [B, Lmax, D] tensor, filling unused positions with a pad value.

Key optimization opportunities:
- Block size tuning (BLOCK_T, BLOCK_D) for the target GPU
- Memory access coalescing
- Reducing redundant stores (pad then overwrite pattern)
- Warp scheduling and occupancy tuning

Constraints:
- Must maintain the same function signature for `pack_seq`
- Output must match reference exactly for valid positions
- Padding positions must contain the specified pad value


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.


## Protected evaluation controls

The original scored path launches the kernel directly into a preallocated output with pad=0; replay checks that exact path. Public-wrapper correctness also checks its default -inf and explicit padding; allocation remains outside original scored timing.

The task-local `_arena_contract.py` and `_arena_replay.py` are protected evaluation code. Original cases, seeds, tolerances, warmups, sample counts, allocations and preparation boundaries remain in `scripts/task_runner.py`. The extra `contract_controls` manifest row is correctness-only. Both the frozen baseline and candidate receive the same checks. The measured graph exposes its real outputs; an untimed replay changes a domain-valid input, recomputes the CPU oracle and restores all input buffers in `finally`. For the zero operator the replay control instead poisons its output. References and snapshots are outside device timing. Failure to observe or replay the measured invocation is an error, never an accepted timing sample.

Additional unscored public-branch controls from PR105: All-empty/high-rank and explicit packing block/dtype boundaries.
These use explicit `control-upstream-*` manifest rows. Original scored inputs,
numerical gates, seeds, warmups and sample counts remain unchanged.
