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


Protected correctness checks include the complete output tensor, including rows
not selected by the mask mapping. They retain the exact negative-infinity pattern
and original finite-value atol=1e-8, rtol=1e-5, while checking dtype/shape/device.
Cloned diagnostic inputs permute the row mapping and complement all 32 mask bits,
including the sign bit omitted by the original positive-only mask generator.

Performance retains the original scored inputs, shapes, seed, warmups, samples,
`target_ms=20.0` and preparation that resets working logits. The actual timed
output and the same captured replay must match pristine references after changing
logits, mapping and signed mask words. Working/source buffers are restored even
on replay failure. All original scored cases, kernel and generated helpers remain
unchanged; masked negative infinity is expected rather than a nonfinite error.

Additional unscored public-branch controls from PR105: Masked vocab tails below/above 8192 with nonidentity selected rows.
They have independent `control-upstream-*` manifest rows; existing scored
inputs, numerical gates and timing remain unchanged.
