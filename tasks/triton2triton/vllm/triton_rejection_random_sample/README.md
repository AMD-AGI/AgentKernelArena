# triton_rejection_random_sample

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton rejection random sample kernel for speculative decoding. The kernel performs probabilistic accept/reject on draft tokens using target and draft probabilities.

Constraints:
- Must maintain the same function signature for `rejection_random_sample`
- Output must match reference exactly for integer outputs or within atol=1e-2, rtol=1e-2 for float outputs


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.


The protected manifest requires the declared kernel symbols to remain Triton JIT
functions, including kernels originally decorated with `@triton.jit()`. Removing
the decorator is rejected before compilation. This structural check supplements
the numerical and timed-path checks; it does not by itself attest every dispatch.

Protected checks compare every int32 output token and every untouched output cell
against a pristine-input scalar accept/reject reference. The original condition
`draft_prob > 0 and target_prob / draft_prob >= uniform_prob` remains unchanged;
all five original cases, distributions, seeds (42+i correctness, 0 performance),
10 warmups and 100 samples are preserved. All probability, token, cumulative
length, uniform and greedy buffers are read-only. Both the returned tensor and
the caller's output are checked.

Unscored controls add ragged lengths including empty requests, greedy rows that
must remain untouched, acceptance equality, early/late rejection, zero draft
probability, and `draft_probs=None` (implicit draft probability one). Nonzero
output sentinels check that cells following rejection and unused capacity are
preserved. These diagnostics do not replace the original scored workloads.

Performance validates the output returned by the actual `TimedRun`, then changes
all input/routing/probability buffers in place and replays the same measured
invocation. Only cells that must be written are poisoned; untouched cells retain
their caller-provided values. This introduces no reset or allocation inside the
original timed wrapper. Every input and output buffer is restored in `finally`,
including replay errors.
