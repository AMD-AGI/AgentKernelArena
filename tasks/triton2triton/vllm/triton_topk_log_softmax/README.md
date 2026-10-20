# triton_topk_log_softmax

The starting candidate is implemented Triton. Improve the declared kernel and new implementation helpers;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton top-k log softmax kernel that computes log-probabilities for a set of token IDs using a numerically stable two-pass algorithm.

Constraints:
- Must maintain the same function signature for `compute_token_logprobs`
- Output must match reference within atol=1e-2, rtol=1e-2 for float outputs or exactly for integer outputs


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original scored cases and explicit correctness-only controls; the adapter verifies it against the
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

Additional unscored public-branch controls from PR105: Singleton vocab plus below-block and FP16 above-block numerical edges.
They have independent `control-upstream-*` manifest rows; existing scored
inputs, numerical gates and timing remain unchanged.

The wrapper, imports, allocations and layout normalization are protected; only
`_topk_log_softmax_kernel` and permitted new implementation helpers are editable.
The kernel uses unit column stride and dense token-ID offsets. The wrapper
normalizes both logits and INT64 token IDs to contiguous storage, supporting
strided caller views while preserving their values. Three explicit unscored
`control-layout-*` rows cover last-dimension stride-two logits, stride-two INT64
IDs and both together, with a 1033-token vocabulary and seven selected tokens.
Their FP64 log-partition oracle is independent of the kernel's two-pass reduction.

All original scored inputs are contiguous and already use INT64 IDs, so both
normalizations preserve their pointers, allocation boundary and device work.
The five original scored cases, seeds, tolerances, ten warmups and 100 samples
are unchanged. Copies needed for strided callers belong to the same protected
wrapper for baseline and candidate; no strided case contributes a timing score.
