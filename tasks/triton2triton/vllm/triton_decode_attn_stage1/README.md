# triton_decode_attn_stage1

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton decode attention stage1 kernel `_fwd_kernel_stage1` for
maximum GPU throughput while maintaining numerical correctness.

The kernel implements the first stage of split-KV decode attention. Each program
instance handles one (batch, head, kv_split) and computes partial attention
over a contiguous range of KV tokens from a paged KV cache:
- Loads Q vector for current (batch, head)
- Iterates over KV tokens in the assigned split, loading K/V via paged
  Req_to_tokens mapping
- Computes Q @ K^T scaled dot-product with online softmax
- Outputs partial attention output and logsumexp for later reduction

Key optimization opportunities:
- BLOCK_N tuning for the target GPU
- Memory access coalescing for paged KV cache reads
- Warp scheduling and occupancy tuning (num_warps, num_stages)
- Reducing masking overhead
- Loop unrolling

Constraints:
- Must maintain the same function signature for `decode_att_m_fwd`
- Output must match reference within atol=1e-2, rtol=1e-2 for float16
- Must handle arbitrary sequence lengths, head dimensions, and page sizes


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.

