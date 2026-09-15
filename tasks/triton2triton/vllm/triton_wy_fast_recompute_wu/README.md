# triton_wy_fast_recompute_wu

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton kernel `recompute_w_u_fwd_kernel` for maximum GPU throughput
while maintaining numerical correctness.

Recompute w and u with scalar gating for WY-fast delta rule.

Constraints:
- Must maintain the same function signature for `wy_fast_recompute_wu`
- Output must match reference within atol=5e-2, rtol=5e-2


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

Protected checks require exactly two tensors `(w, u)`, with the declared shapes,
dtypes, devices, finite values, and the original FP32 reference comparison at
atol=rtol=5e-2. All five input tensors are read-only and the oracle uses pristine
copies. An unscored `(B,T,H,K,V,BT)=(2,35,3,65,70,32)` control covers the second
partial block, both partial feature tiles, unequal output widths, and nonzero
outputs large enough to reject zero fills with the existing tolerance.

The original five seeds, input distributions, full wrapper allocations, 10
warmups, and 100 timing samples are unchanged. Both actual `TimedRun` outputs
are checked, poisoned, and checked again after changing all five inputs and
replaying that exact invocation. Verification and perturbation occur outside
timing; all input buffers are restored even on a failed replay.
