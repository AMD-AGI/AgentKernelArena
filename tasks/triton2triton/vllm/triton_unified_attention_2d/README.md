# triton_unified_attention_2d

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton kernel `kernel_unified_attention_2d` for maximum GPU throughput
while maintaining numerical correctness.

This kernel implements paged attention with a 2D block table layout. It performs
scaled dot-product attention with causal masking over variable-length packed
sequences, reading K/V from a paged cache indexed by a 2D block table
[num_seqs, max_blocks_per_seq].

Key features:
- Paged KV cache with configurable block size
- Causal masking with optional sliding window
- Optional softcap (tanh-based logit capping)
- GQA support (num_queries_per_kv grouping)
- Variable-length sequence packing via cu_seqlens_q

Constraints:
- Must maintain the same function signature for `unified_attention_2d`
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

