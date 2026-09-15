# triton_prepare_pos_seq_lens

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton kernel `_prepare_pos_seq_lens_kernel` for maximum GPU throughput
while maintaining correctness.

The kernel computes position IDs and sequence lengths for each request in a batch.
For each request, it writes positions starting from num_computed_tokens into pos[],
and stores seq_len = num_computed_tokens + query_len. The last thread block pads
unused seq_lens entries with zeros for CUDA graph compatibility.

Key optimization opportunities:
- Block size tuning
- Memory coalescing for position writes
- Efficient padding of unused slots

Constraints:
- Must maintain the same function signature for `prepare_pos_seq_lens`
- Output must match reference exactly (integer positions and lengths)


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

Controls cover heterogeneous query lengths, remapped requests and zero-filled inactive sequence-length slots.

The task-local `_arena_contract.py` and `_arena_replay.py` are protected evaluation code. Original cases, seeds, tolerances, warmups, sample counts, allocations and preparation boundaries remain in `scripts/task_runner.py`. The extra `contract_controls` manifest row is correctness-only. Both the frozen baseline and candidate receive the same checks. The measured graph exposes its real outputs; an untimed replay changes a domain-valid input, recomputes the CPU oracle and restores all input buffers in `finally`. For the zero operator the replay control instead poisons its output. References and snapshots are outside device timing. Failure to observe or replay the measured invocation is an error, never an accepted timing sample.

The position/sequence replay also fills both output buffers with `-1` after timing,
so an implementation that omits inactive zero writes cannot reuse the initially
zero sequence buffer. The original timed inputs remain unchanged, and all buffers
are restored even when this additional check fails.
