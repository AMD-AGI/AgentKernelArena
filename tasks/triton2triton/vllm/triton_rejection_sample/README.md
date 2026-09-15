# triton_rejection_sample

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton `_rejection_sample_kernel` for maximum GPU throughput.
This kernel performs sequential rejection sampling for speculative decoding:
compares target-sampled tokens against draft tokens, accepting until the first
mismatch. Uses num_warps=1 due to the sequential nature.

Key optimization opportunities:
- Memory access patterns
- Early termination efficiency
- Note: sequential nature limits parallelism within each request

Constraints:
- Must maintain the same function signature for `rejection_sample`
- Must preserve num_warps=1
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


## Protected evaluation controls

Controls cover full acceptance with bonus token, early/late rejection and zero speculative steps. Output metadata and accepted counts are strict; only each returned accepted prefix is defined by this operator, so unused output tail values are not invented as a new contract.

The task-local `_arena_contract.py` and `_arena_replay.py` are protected evaluation code. Original cases, seeds, tolerances, warmups, sample counts, allocations and preparation boundaries remain in `scripts/task_runner.py`. The extra `contract_controls` manifest row is correctness-only. Both the frozen baseline and candidate receive the same checks. The measured graph exposes its real outputs; an untimed replay changes a domain-valid input, recomputes the CPU oracle and restores all input buffers in `finally`. For the zero operator the replay control instead poisons its output. References and snapshots are outside device timing. Failure to observe or replay the measured invocation is an error, never an accepted timing sample.
