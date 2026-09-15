# triton_bad_words

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the original single-token bad-word filtering workload while preserving the
full `apply_bad_words` operator, including multi-token prefix matching. The five
scored cases use one-token bad-word lists, identity request mapping and local_pos=0;
they measure masking throughput as batch, vocabulary and list sizes grow. Their
speedup is specifically a single-token filtering result, not a claim about general
prefix-search or speculative-decoding throughput.

Multi-token words, generated-output history, request remapping and speculative
positions remain required correctness behavior. The three original targeted cases
and the additional `contract_controls` group check those paths independently; they
have never contributed performance samples to this task. A candidate cannot remove
those paths merely because the score uses the single-token workload.

Constraints:
- Must maintain the same function signature for `apply_bad_words`
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


The manifest also declares 3 original targeted correctness-only cases beyond the main five-shape table. Their original checks and seeds remain active; no performance score is assigned to those cases.

## Protected evaluation controls

Controls cover multi-token bad words using output history plus speculative input positions, matching/nonmatching prefixes and remapped requests.

The task-local `_arena_contract.py` and `_arena_replay.py` are protected evaluation code. Original cases, seeds, tolerances, warmups, sample counts, allocations and preparation boundaries remain in `scripts/task_runner.py`. The extra `contract_controls` manifest row is correctness-only. Both the frozen baseline and candidate receive the same checks. The measured graph exposes its real outputs; an untimed replay changes a domain-valid input, recomputes the CPU oracle and restores all input buffers in `finally`. For the zero operator the replay control instead poisons its output. References and snapshots are outside device timing. Failure to observe or replay the measured invocation is an error, never an accepted timing sample.
