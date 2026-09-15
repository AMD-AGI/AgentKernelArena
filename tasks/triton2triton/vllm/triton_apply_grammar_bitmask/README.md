# triton_apply_grammar_bitmask

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton kernel `_apply_grammar_bitmask_kernel` for maximum GPU throughput
while maintaining correctness.

The kernel applies a packed grammar bitmask to logits for structured output generation.
The bitmask is stored as int32 values where each int32 encodes 32 bits. The kernel
unpacks the bitmask using bit shifting and sets logits to -inf where the bit is 0
(token not allowed by grammar).

The 2D grid is (num_masks, num_vocab_blocks) where num_vocab_blocks = ceil(vocab_size / BLOCK_SIZE).

Key optimization opportunities:
- Block size tuning
- Efficient bitmask unpacking
- Vectorized stores for masked positions
- Memory coalescing

Constraints:
- Must maintain the same function signature for `apply_grammar_bitmask`
- Bitmask format: int32 packed, bit 0 = disallowed (set to -inf), bit 1 = allowed
- Output must match reference (exact -inf placement, unchanged non-masked values)


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.

