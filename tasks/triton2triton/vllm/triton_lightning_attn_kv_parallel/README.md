# triton_lightning_attn_kv_parallel

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton Lightning Attention parallel KV outer product kernel
`_fwd_kv_parallel` for maximum GPU throughput while maintaining numerical correctness.

The kernel computes key-value outer products K.T @ V with decay weighting for each
block position in parallel. The output is an accumulated KV state tensor of shape
[B, H, NUM_BLOCK, D, E].

Key parameters:
- BLOCK: main block size (default 256)
- CBLOCK: sub-block size (default 64)
- D_FBLOCK, E_FBLOCK: feature block dimensions

Constraints:
- Must maintain the same function signature for `lightning_attn_kv_parallel_forward`
- Output must match reference within atol=1e-2, rtol=1e-2 for float32
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

Protected evaluation validates the complete FP32 output tensor and preserves
the original atol=rtol=1e-2. An additional unscored 273-token case covers a second
main block, a partial sub-block and four-dimensional slopes, checking every
block against the independent CPU formula. All five original scored cases,
seeds, inputs, wrapper calls, 10 warmups and 100 samples remain unchanged.

The actual timed output is checked. A replay of the same measured invocation
after changing key/value/slope inputs and poisoning the output must also match
the reference. Read-only inputs are checked and restored even on failure; these
checks execute outside timing.
