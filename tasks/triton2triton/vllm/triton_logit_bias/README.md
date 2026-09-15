# triton_logit_bias

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize additive logit-bias throughput on the five original scored workloads.
Those workloads disable allowlist and stop-token filtering and use identity request
mapping; their measured speedup applies to bias-only execution. The full public
`apply_logit_bias` API must still implement allowed-token filtering, additive biases
and minimum-length stop masking correctly. The original targeted correctness case
and the new routed controls enforce this broader compatibility. This score does not
claim an improvement in combined filtering throughput.

Constraints:
- Must maintain the same function signature for `apply_logit_bias`
- Output must match reference within atol=1e-2, rtol=1e-2 for float outputs or exactly for integer outputs


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.


The manifest also declares 1 original targeted correctness-only cases beyond the main five-shape table. Their original checks and seeds remain active; no performance score is assigned to those cases.

## Protected evaluation controls

Controls combine remapped requests, allowlists, nonzero heterogeneous biases and stop-token minimum lengths. Original scored cases remain the original bias-only workload.

The task-local `_arena_contract.py` and `_arena_replay.py` are protected evaluation code. Original cases, seeds, tolerances, warmups, sample counts, allocations and preparation boundaries remain in `scripts/task_runner.py`. The extra `contract_controls` manifest row is correctness-only. Both the frozen baseline and candidate receive the same checks. The measured graph exposes its real outputs; an untimed replay changes a domain-valid input, recomputes the CPU oracle and restores all input buffers in `finally`. For the zero operator the replay control instead poisons its output. References and snapshots are outside device timing. Failure to observe or replay the measured invocation is an error, never an accepted timing sample.
