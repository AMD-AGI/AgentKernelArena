# triton_swiglustep_and_mul

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton SwiGLU-step-and-mul activation kernel `_swiglustep_and_mul_kernel`
for maximum GPU throughput while maintaining numerical correctness.

The kernel computes: silu(x[:,:d]).clamp(max=limit) * x[:,d:].clamp(-limit, limit)
where d = input_dim // 2.

Key optimization opportunities:
- Block size tuning
- Memory access pattern optimization
- Fused computation optimizations

Constraints:
- Must maintain the same function signature for `swiglustep_and_mul`
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


## Clamp coverage and measured output

Full output shape, dtype, device and finiteness are checked against a pristine
input's original reference at `atol=rtol=1e-2`; input must remain read-only.
Unscored controls exercise both signs of the up-input clamp, the gate's upper
clamp (with no lower gate clamp), limits 7.0 and 0.1, a partial second 1024-column
tile, and row-strided inputs with contiguous columns.

The five original scored cases, random seeds, input distribution, default limit,
10 warmups and 100 device samples remain unchanged. Timing still measures the
public wrapper including allocation. After timing, its captured output is
checked, input is multiplied by -3 and the output poisoned with NaN. The same
captured graph must produce the new reference output on replay. Checks are
outside timing and original input is restored even when replay fails.
