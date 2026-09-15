# triton_expand

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton expand kernel that broadcasts a [batch_size] tensor to [num_tokens] based on cumulative token counts, with optional value replacement.

Constraints:
- Must maintain the same function signature for `expand_batch_to_tokens`
- Output must match reference within atol=1e-2, rtol=1e-2 for float outputs or exactly for integer outputs


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

The original five scored uniform-count cases and exact integer gate are kept.
An extra unscored control checks legal ragged cumulative counts (including an
empty request) and nondefault value replacement. Per-request counts must stay
within the implementation's MAX_SPEC_LEN=128 bound. The complete output shape,
input dtype and device are required, with a reference computed from pristine
source/count buffers before candidate invocation.

The original full public wrapper, seeds, 10 warmups and 100 samples remain the
performance workload. Its actual captured output is checked, then poisoned
and replayed with changed values and one redistributed token at the same total
size. Both input buffers must remain unchanged by the candidate and are restored
even on replay failure. Added reference and replay checks are outside timing.
