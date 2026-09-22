# triton_bmm

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton batched matrix multiplication kernel `bmm_kernel` for maximum
GPU throughput while maintaining numerical correctness.

The kernel implements batched GEMM: (B, M, K) x (B, K, N) -> (B, M, N) with support
for large tensors and contiguous memory hints.

Key optimization opportunities:
- Tile size tuning (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)
- Memory access pattern optimization
- Warp scheduling and occupancy tuning

Constraints:
- Must maintain the same function signature for `bmm_triton`
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

Correctness and performance checks retain pristine input snapshots and require
that the candidate preserves caller-owned inputs. The runner checks the actual
timed BMM output, then changes the captured graph's input tensors, poisons the
output with NaNs and checks the same graph replay against a reference computed
before replay. Output shape, dtype, device and finite values are required.
The original FP32 BMM reference, FP16 output and `atol=1e-2`, `rtol=1e-2` stay
unchanged. All added reference work and replay checks run outside device timing;
the original five cases, seeds, warmups, samples and timed operator are preserved.
