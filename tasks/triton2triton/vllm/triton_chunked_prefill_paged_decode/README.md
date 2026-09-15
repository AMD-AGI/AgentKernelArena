# triton_chunked_prefill_paged_decode

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton kernel `kernel_paged_attention_2d` for maximum GPU throughput
while maintaining numerical correctness.

This kernel implements paged attention for the decode path of chunked prefill.
It reads from a 5D K cache [num_blocks, num_kv_heads, head_size//x, block_size, x]
and 4D V cache [num_blocks, num_kv_heads, head_size, block_size], supporting
non-standard physical block sizes and GQA.

Key features:
- 5D K cache addressing with x-factor interleaving
- 4D V cache with slot-innermost layout
- GQA support with padded query-per-kv groups
- Optional sliding window and ALiBi slopes
- Decode-only filtering via query_start_len_ptr

Constraints:
- Must maintain the same function signature for `chunked_prefill_paged_decode`
- Output must match reference within atol=1e-2, rtol=1e-2 for float16


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.

