# triton_lightning_attn_kv_reduce

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton Lightning Attention KV reduce kernel `_fwd_kv_reduce`
for maximum GPU throughput while maintaining numerical correctness.

The kernel reduces partial KV outer products from parallel computation into
final accumulated KV state using a prefix sum with exponential decay. It also
updates the KV history carry-over tensor.

Key parameters:
- BLOCK: main block size (default 256)
- D_FBLOCK, E_FBLOCK: feature block dimensions
- The kernel processes blocks sequentially to compute prefix sums

Constraints:
- Must maintain the same function signature for `lightning_attn_kv_reduce_forward`
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



## Prefix carry, in-place outputs and timed state

All five original shapes, seeds, FP32 inputs and `atol=rtol=0.01` comparisons
remain. An additional unscored 273-token diagnostic covers two blocks, a
17-token partial final block, nonzero history, zero/nonzero decay, and the
public four-dimensional slope input. Both returned buffers must preserve the
specified shapes, dtypes, devices and in-place storage; the slope is read-only.

Timing retains the original 10 warmups, 100 samples and original `prepare_fn`
that copies the source KV/history into the work buffers before each invocation.
Checks observe the actual timed in-place result. An unscored exact replay uses
changed slope, source KV and nonzero source history through the same preparation.
Both updated buffers must match the independent recurrence. In-place state is
also input, so this diagnostic changes valid input state instead of poisoning
values the operator must read. The caller-owned sources and work buffers are
restored even after a failed replay. No additional case is added to scoring.
