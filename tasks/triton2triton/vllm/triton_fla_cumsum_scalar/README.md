# triton_fla_cumsum_scalar

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton kernel `chunk_local_cumsum_scalar_kernel` for maximum GPU throughput
while maintaining numerical correctness.

Chunk-local cumulative sum for scalar gates.

Constraints:
- Must maintain the same function signature for `chunk_local_cumsum_scalar`
- Output must match reference within atol=1e-4, rtol=1e-4


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.


The protected manifest requires the declared kernel symbols to remain Triton JIT
functions, including kernels originally decorated with `@triton.jit()`. Removing
the decorator is rejected before compilation. This structural check supplements
the numerical and timed-path checks; it does not by itself attest every dispatch.

The scored workload remains the five original FP32 seeds at the original shape
and chunk size 64. Protected checks now enforce complete output shape, FP32
dtype, input device and finite values at the original atol=rtol=0.0001.
An unscored contiguous-slice diagnostic changes batch/head counts, uses a
partial final chunk, halves the chunk size, and exercises reverse accumulation.
The vector task also exercises a feature dimension not divisible by its tile.
No original input, scored case or tolerance is replaced by these diagnostics.

Performance still times the original full wrapper with 10 warmups and 100
samples. Both the actual captured output and a poisoned replay with changed
input values must satisfy the same numerical gate. Pristine inputs and the
public callable are restored even on failure; validation is outside timing.
