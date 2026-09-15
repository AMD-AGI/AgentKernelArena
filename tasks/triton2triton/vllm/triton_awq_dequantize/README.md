# triton_awq_dequantize

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton AWQ dequantization kernel `awq_dequantize_kernel` for maximum
GPU throughput while maintaining numerical correctness.

The kernel unpacks 4-bit AWQ quantized weights, applies zero-point subtraction
and scale multiplication to produce dequantized float16 weights.

Key optimization opportunities:
- Tile size tuning (BLOCK_SIZE_X, BLOCK_SIZE_Y) for the target GPU
- Memory access pattern optimization
- Warp scheduling and occupancy tuning

Constraints:
- Must maintain the same function signature for `awq_dequantize_triton`
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

The protected `_arena_checks.py` validates the exact captured output against the
original unpacking reference with `atol=1e-2, rtol=1e-2`. After timing, it changes
the packed weights (including the sign/high bits), packed zeros and scales,
poisons the output, and validates the same graph's replay. Shape, dtype, device
and finiteness are checked as well. Original timed inputs, seeds, cases and timing
parameters are unchanged. Unobservable event fallback fails explicitly.
