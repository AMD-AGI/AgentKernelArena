# triton_topk_topp

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize top-k masking throughput for the five original scored workloads (k=50,
top-p disabled), using the direct kernel launch and reusable scratch buffers.
The public wrapper must retain combined top-k/top-p and top-p-only correctness,
including the original and added compatibility cases. The score is a top-k-only
result and does not claim combined top-k/top-p throughput.

Constraints:
- Must maintain the same function signature for `apply_top_k_top_p_triton`
- Common finite logits use atol=1e-4, rtol=1e-4. Mask disagreement is limited to
  one token per row for top-k, or max(4, vocab_size // 500) for top-p/combined paths.
  NaN and positive infinity are invalid; masked values are negative infinity.


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

Controls add top-p-only and heterogeneous k/p. Keep original numerical rules: finite-value atol/rtol=1e-4, top-k mask mismatch <=1, top-p/combined <=max(4,vocab//500). NaN/+inf cannot masquerade as a removed token. Original scored path remains direct preallocated top-k with k=50.

The task-local `_arena_contract.py` and `_arena_replay.py` are protected evaluation code. Original cases, seeds, tolerances, warmups, sample counts, allocations and preparation boundaries remain in `scripts/task_runner.py`. The extra `contract_controls` manifest row is correctness-only. Both the frozen baseline and candidate receive the same checks. The measured graph exposes its real outputs; an untimed replay changes a domain-valid input, recomputes the CPU oracle and restores all input buffers in `finally`. For the zero operator the replay control instead poisons its output. References and snapshots are outside device timing. Failure to observe or replay the measured invocation is an error, never an accepted timing sample.
