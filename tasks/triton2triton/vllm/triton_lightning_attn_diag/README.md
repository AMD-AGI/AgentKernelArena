# triton_lightning_attn_diag

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton Lightning Attention diagonal block kernel `_fwd_diag_kernel`
for maximum GPU throughput while maintaining numerical correctness.

The kernel computes diagonal (local) block attention for the Lightning Attention
algorithm. For each block, it computes Q @ K.T with a causal mask and exponential
decay factor, then accumulates O = attn @ V. The slope parameter S controls the
per-head exponential decay rate.

Key parameters:
- BLOCK: main block size (default 256)
- CBLOCK: sub-block size (default 32)
- Each program instance handles one sub-block within one block

Constraints:
- Must maintain the same function signature for `lightning_attn_diag_forward`
- FP16 output, compared in FP32 against the original FP32 reference, must match
  within atol=5e-2, rtol=5e-3 (the protected harness's existing gate).
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

`_arena_checks.py` checks output shape, dtype, device, finiteness, and numerical
values against pristine inputs; Q, K, V and slope are read-only. Additional
unscored checks cover 273 and 545 tokens (second and third main blocks with
partial sub-blocks), zero/nonzero decay, the public 4D slope form, and
BLOCK=64/CBLOCK=16 on the 273-token case.
All five scored cases, seeds, 10 warmups and 100 timing samples are unchanged.
Both roles use explicit device-Event timing to invoke the allocating public
wrapper for every measured sample. Graph `after_sample` would expose a captured
output after each replay, but replay would omit a fresh wrapper invocation and
a batched graph could leave earlier outputs unchecked. The benchmark checks
every measured output, changes all four inputs, poisons the final output, and
numerically checks an eager rerun of the same callable outside timing. Caller
inputs are restored even when rerun or verification fails. These Event times
are not directly comparable with historical graph-replay times.
