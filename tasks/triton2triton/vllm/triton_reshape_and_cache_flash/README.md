# triton_reshape_and_cache_flash

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton kernel `reshape_and_cache_kernel_flash` for maximum GPU
throughput while maintaining numerical correctness.

The kernel scatters K/V tokens into paged KV-cache slots. Each token's key and
value vectors are written to the appropriate block and slot in the paged cache
based on a slot_mapping tensor.

Key optimization opportunities:
- Tile size tuning for the target GPU
- Memory access coalescing for scattered writes
- Warp scheduling and occupancy tuning

Constraints:
- Must maintain the same function signature for `reshape_and_cache_flash`
- Output must match reference (exact copy for auto dtype)


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.


Protected checks preserve the original atol=rtol=1e-3 comparisons and inspect
the entire cache, including untouched slots, with pristine key/value/mapping
references and read-only scale checks. Unscored diagnostics cover negative slots,
partial tiles, nonzero cache sentinels, head-major layout, and FP8 scaling/already-FP8 input.
The FP8 diagnostic uses float8_e4m3fnuz; this does not certify every FP8 format.
All five original scored cases, seeds, 10 warmups, 100 samples and full-wrapper
timing remain unchanged. The original zero-reset runs unchanged for every warmup
and measured sample. Only the unscored exact replay poisons written cache slots
after that reset, using changed K/V inputs and reversed slot routing; the full
result is checked numerically. All caller-owned inputs/caches are restored on exit.
