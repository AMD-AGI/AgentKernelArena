# triton_fla_l2norm

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton kernel `l2norm_fwd_kernel` for maximum GPU throughput
while maintaining numerical correctness.

L2 normalization over the last dimension.

Constraints:
- Must maintain the same function signature for `l2norm_fwd`
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
the numerical and timed-path checks. The task-local checker requires a genuine
Triton JIT/Autotuner chain and observes compiled launches of the declared
`l2norm_fwd_kernel` for each scored wrapper invocation, including all 100
reported samples and the perturbed rerun. Compile-only warmup calls do not count.
A correctness-only
control suppresses that launch and rejects a wrapper whose correct result is
independent of the declared kernel. The main-kernel width controls also require
that launch to contribute to their output. Wide-width controls use the separate
wide kernel and retain their numerical checks.

The scored workload remains the five original FP32 (512, 128) seed cases with
atol=rtol=1e-4. The protected checker requires the original output shape, dtype
and device and finite values, and checks input immutability. An extra unscored
(2, 3, 17) case covers flattening and feature/row tails; zero and near-zero rows
with nondefault epsilon make the epsilon behavior observable.
Further unscored three-row FP32 inputs cover every main-kernel power-of-two
block width from 256 through 16,384, with controls at the width and its first
feature tail. They use an independent CPU vector-norm oracle, varied rows and
a nondefault epsilon. Unscored widths 16,385, 32,768 and 32,769 cross the
one-block cap and check complete and partial wide tiles. The wide source path
accumulates the squared norm across every feature tile before writing the
normalized row. These controls run only during correctness. They do not change
the five scored 512-by-128 cases, tolerance or timing method.

The original full wrapper is timed with 10 warmups and 100 samples. Both roles
use explicit device-Event timing so every measured sample invokes the public
wrapper, including its output allocation. Graph `after_sample` can inspect a
captured output after each replay, but replay does not invoke that wrapper and
a batched graph exposes only its final returned output. Every measured output
and a poisoned eager rerun with perturbed inputs must satisfy the same reference
gate. Input restoration runs even when rerun fails. These checks execute outside
the measured region and retain all scored cases. Event times are not directly
comparable with the historical graph-replay times.
Launch observation adds the same Python bookkeeping to baseline and candidate
wrapper calls; fresh GPU qualification is required before comparing timings
with earlier results.
