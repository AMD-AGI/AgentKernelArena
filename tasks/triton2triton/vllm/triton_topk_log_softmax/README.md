# triton_topk_log_softmax

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton top-k log softmax kernel that computes log-probabilities for a set of token IDs using a numerically stable two-pass algorithm.

Constraints:
- Must maintain the same function signature for `compute_token_logprobs`
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


The public output is an FP32 tensor shaped like `token_ids`; logits and token
IDs are read-only. Protected checks use pristine inputs, require full metadata
and finite values, and retain the original log-softmax/gather reference and
atol=rtol=1e-2. An unscored 1031-token vocabulary control exercises the second
partial block, duplicated and boundary token IDs, seven selected tokens, and
large positive/negative logits. The original five scored cases, seeds, ten
warmups, and 100 timing samples remain unchanged.

Both the original timed output and its exact `TimedRun` replay are numerically
checked. Replay uses changed logits and token IDs and poisons the captured
output outside timing. Both caller input buffers are restored in `finally`,
including on replay failure.
