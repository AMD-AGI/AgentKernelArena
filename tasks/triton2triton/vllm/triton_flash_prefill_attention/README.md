# triton_flash_prefill_attention

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton flash attention prefill kernel `_fwd_kernel` for maximum
GPU throughput while maintaining numerical correctness.

The kernel implements memory-efficient flash attention for the prefill stage of
LLM inference. It processes variable-length packed sequences with support for:
- Grouped-query attention (GQA) and multi-query attention (MQA)
- Causal masking
- Bidirectional sliding window masking
- Online softmax with exp2 for numerical stability

Key optimization opportunities:
- Tile size tuning (BLOCK_M, BLOCK_N) for the target GPU
- Memory access pattern optimization (coalescing, prefetching)
- Warp scheduling and occupancy tuning (num_warps, num_stages)
- Reducing unnecessary masking overhead for common cases
- Loop unrolling and instruction-level parallelism

Constraints:
- Must maintain the same function signature for `context_attention_fwd`
- Output must match reference within atol=1e-2, rtol=1e-2 for float16
- The kernel must handle arbitrary sequence lengths and head dimensions


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.

