# triton_causal_conv1d_update

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton kernel `_causal_conv1d_update_kernel` for maximum GPU throughput.
Single-step conv update using conv state buffer. Updates state and computes output.
Supports SiLU activation, bias, and variable conv widths.
Constraints:
- Must maintain the same function signature for `causal_conv1d_update`
- Output must match reference within atol=1e-1, rtol=1e-1


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.


The five original cases exercise single-token FP32 updates with widths 3/4,
non-padding cache indices, optional bias and SiLU. The existing equal-dtype API
writes the result into `x` in place and shifts the convolution cache, appending
the original input token. Protected checks compare that cache exactly (it is a
copy operation) and preserve the original output atol=1e-1, rtol=1e-1. They also
check tensor shape, dtype, device, finiteness and unchanged weights/indices.
References are computed from pristine inputs before invoking the candidate.

Performance preserves the original wrapper, `target_ms=20.0`, warmups, repetitions,
and preparation that resets both working input and state before each invocation.
Both actual timed outputs are verified. A replay after changing the source input,
state and weights must produce the new correct output and updated history.
Diagnostic perturbations and working buffers are restored even on failure.
This adds no scored cases and does not change any kernel or generated helper.

The protected manifest requires the declared kernel symbols to remain Triton JIT
functions, including kernels originally decorated with `@triton.jit()`. Removing
the decorator is rejected before compilation. This structural check supplements
the numerical and timed-path checks; it does not by itself attest every dispatch.
