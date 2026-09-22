# triton_ep_gather

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton `_fwd_kernel_ep_gather` kernel for maximum GPU throughput.
This kernel gathers expert computation results back to original token order
with weighted accumulation across topk experts.

Key optimization opportunities:
- Block size tuning (BLOCK_D)
- Token-level parallelism
- Memory access patterns

Constraints:
- Must maintain the same function signature for `ep_gather`
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


Protected checks compare the complete in-place FP16 output against the original
weighted-sum reference at atol=rtol=5e-2, including shape, dtype, device, and
finiteness. The expert buffer, IDs, weights, and routing indices are read-only;
the oracle uses pristine copies. Unscored controls cover three routes, disabled
negative expert IDs (their indices must not be accessed), zero-active-route
rows, signed weights, 1025 tokens, 2048 hidden columns, and output row strides.

The original five cases, correctness seeds 42+i, performance seed 0, initial
output allocation, ten warmups and 100 samples are unchanged. Both actual
`TimedRun` output and its poisoned exact replay are checked after changing all
four inputs. Checks and perturbations are outside timing. All input buffers and
the caller's original output state are restored after timing, even on failure.
