# triton_causal_conv1d_fwd

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton kernel `_causal_conv1d_fwd_kernel` for maximum GPU throughput.
1D causal convolution over variable-length sequences with continuous batching.
Supports SiLU activation, bias, and conv state caching.
Constraints:
- Must maintain the same function signature for `causal_conv1d_fwd`
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


The protected checks validate both the returned tensor and the updated convolution
cache. For the original five workloads (widths 3/4, non-padding cache slots and
sequences at least `width-1` tokens long), each cache slot must contain exactly
the corresponding input sequence's last `width-1` tokens. Output shape, dtype,
device and finiteness are checked; output arithmetic keeps atol=1e-1, rtol=1e-1.
References are computed from pristine inputs before invoking the candidate.

Performance still launches the original kernel directly with its precomputed
launch map and restores the initial state using the original `prepare_fn`.
After timing, protected checks verify both actual output buffers, perturb inputs
and initial state, poison outputs, then verify the same captured graph again.
Diagnostic changes to inputs and initial state are restored even on failure.
No case, seed, warmup, repetition, launch parameter or timed allocation changes.

The protected manifest requires the declared kernel symbols to remain Triton JIT
functions, including kernels originally decorated with `@triton.jit()`. Removing
the decorator is rejected before compilation. This structural check supplements
the numerical and timed-path checks; it does not by itself attest every dispatch.
