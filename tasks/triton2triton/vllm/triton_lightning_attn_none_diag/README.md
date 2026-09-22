# triton_lightning_attn_none_diag

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton Lightning Attention non-diagonal block kernel
`_fwd_none_diag_kernel` for maximum GPU throughput while maintaining
numerical correctness.

The kernel applies accumulated KV state to queries for cross-block attention.
For each block, it computes O += Q @ KV_accumulated with proper exponential
decay. It reads the diagonal attention output and adds the cross-block
contribution.

Key parameters:
- BLOCK: main block size (default 256)
- CBLOCK: sub-block size (default 64)
- E_FBLOCK: output feature block dimension

Constraints:
- Must maintain the same function signature for `lightning_attn_none_diag_forward`
- Output must match reference within atol=1e-2, rtol=1e-2 for float16
- The kernel must handle arbitrary sequence lengths


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.

Protected checks require the supplied output buffer to be updated and returned,
with the original FP16 numerical gate and unchanged query, slope and KV inputs.
An additional unscored 273-token case checks a second main block, a partial
sub-block and four-dimensional slopes. The original five scored inputs, seeds,
10 warmups, 100 samples and public wrapper remain unchanged.

The existing `prepare_fn` copies the original diagonal result into the output
before each measured invocation; this reset remains outside timing and retains
one call per graph replay. Checks now inspect the actual timed output, perturb
query/slope/KV/diagonal inputs, and validate the same prepared replay against the
independent block formula. All checks run outside timing, and input/output state
is restored even when replay or validation fails.
