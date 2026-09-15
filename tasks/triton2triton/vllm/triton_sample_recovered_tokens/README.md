# triton_sample_recovered_tokens

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton kernel that samples recovered tokens for speculative decoding rejection sampling using the Gumbel-max trick on adjusted probability distributions.

Constraints:
- Must maintain the same function signature for `sample_recovered_tokens`
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

Protected checks require exact token values, dtype, shape and device from the
original independent CPU reference. Correctness retains both probability paths
and jagged per-request lengths for every original case. Scored performance retains
the original draft-probability path and full per-request lengths; the no-draft
path is a correctness requirement, not an additional scored measurement.
All routing, token, probability and exponential-race inputs are read-only.
The actual timed return value is checked, then a consistent vocabulary permutation
changes the data, the captured token output is filled with -1, and the same timed
invocation is replayed and checked. Inputs are restored on success or failure.
The five scored cases, seeds, full-wrapper timing, 10 warmups and 100 samples are
unchanged; neither branch substitutes a different output contract.
