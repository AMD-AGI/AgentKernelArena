# triton_prepare_prefill_inputs

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton kernel `_prepare_prefill_inputs_kernel` for maximum GPU throughput
while maintaining correctness.

The kernel copies token IDs from a 2D all_token_ids buffer into a flat input_ids tensor
for each prefilling request. It uses idx_mapping to map batch indices to request state
indices and query_start_loc for per-request output offsets. It also stores the next
prefill token if the prefill is not yet complete.

Key optimization opportunities:
- Block size tuning for memory coalescing
- Vectorized loads/stores
- Memory access pattern optimization

Constraints:
- Must maintain the same function signature for `prepare_prefill_inputs`
- Output must match reference exactly (integer token IDs)


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

Controls include routed partial prefill, exactly completed prefill and decode requests. Slots not written by the operator must preserve their prior contents.

The task-local `_arena_contract.py` and `_arena_replay.py` are protected evaluation code. Original cases, seeds, tolerances, warmups, sample counts, allocations and preparation boundaries remain in `scripts/task_runner.py`. The extra `contract_controls` manifest row is correctness-only. Both the frozen baseline and candidate receive the same checks. The measured graph exposes its real outputs; an untimed replay changes a domain-valid input, recomputes the CPU oracle and restores all input buffers in `finally`. For the zero operator the replay control instead poisons its output. References and snapshots are outside device timing. Failure to observe or replay the measured invocation is an error, never an accepted timing sample.

Additional unscored public-branch controls from PR105: Ragged prefill, zero lengths, completed requests and token-block tails.
These use explicit `control-upstream-*` manifest rows. Original scored inputs,
numerical gates, seeds, warmups and sample counts remain unchanged.
