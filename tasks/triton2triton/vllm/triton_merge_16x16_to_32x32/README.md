# triton_merge_16x16_to_32x32

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton kernel `merge_16x16_to_32x32_inverse_kernel` for maximum GPU throughput
while maintaining numerical correctness.

Merge 16x16 block inverses into 32x32 for flash linear attention.

Constraints:
- Must maintain the same function signature for `merge_16x16_to_32x32`
- Output must match reference within atol=1e-2, rtol=1e-2


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


## Full inverse outputs and exact measured replay

The returned inverse must preserve the input shape and device, with FP32 output.
The original CPU inverse reference and `atol=rtol=0.01` comparison remain. Checks
snapshot the input before execution and reject changes to this read-only buffer.
The five original seeds, shapes and distributions are unchanged. Additional
unscored correctness probes cover a full 32-row tile followed by three rows,
three heads, and the zero-input identity case; the public wrapper already uses
ceiling division and the kernel's boundary checks support partial tiles.

The benchmark still measures the original public wrapper, including its zeroed
output allocation, with 10 warmups and 100 samples. Its actual captured output
is compared with the pristine-input reference, then poisoned and replayed with a
changed strictly-lower-triangular input. All input buffers and the public wrapper
are restored in `finally`, including failed replay paths. The scored case list,
allocation boundary and canonical helper are unchanged. New GPU qualification
requires a complete finalized validator report for this task package.
