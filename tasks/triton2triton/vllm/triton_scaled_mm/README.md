# triton_scaled_mm

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton scaled matrix multiplication kernel `scaled_mm_kernel` for
maximum GPU throughput while maintaining numerical correctness.

The kernel performs A @ B with per-token/per-channel scaling and optional bias.

Key optimization opportunities:
- Tile size tuning for the target GPU
- Memory access pattern optimization
- Scale application strategy

Constraints:
- Must maintain the same function signature for `triton_scaled_mm`
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


Protected checks require the declared output shape/dtype/device and finite values,
using the original FP32 reference and atol=rtol=1e-2. The oracle reads pristine
operands, scales and bias before candidate execution. Unscored diagnostics cover
partial M/N/K blocks, legal padded/transposed inputs, scalar-A/per-channel-B
scales, explicit tiles and FP32 output. All five scored cases retain their original
seeds and distributions, including the larger performance input magnitudes.
Both roles time the same public wrapper with 10 warmups and 100 samples. The
actual timed result is checked, poisoned and replayed after changing both operands,
scales and bias; every caller-owned input is checked and restored even on failure.
