# triton_eagle_prepare_next_token_padded

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton `eagle_prepare_next_token_padded_kernel` for maximum GPU throughput.
This kernel computes per-request valid sampled token counts and next token IDs for
EAGLE speculative decoding, handling discarded requests and finding the last valid
token among sampled tokens.

Key optimization opportunities:
- Efficient reduction for valid token counting
- Minimizing branch divergence

Constraints:
- Must maintain the same function signature for `eagle_prepare_next_token_padded`
- Output must match reference exactly (integer computation)


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.


Protected checks require exactly two int32 output tensors of the documented
shape on the input device. Both are compared with the original CPU reference
computed from pristine inputs; mutating any source tensor fails. The actual
timed outputs are checked, poisoned and replayed against changed valid input
values. The replay exercises changed rejection/acceptance state and restores
all input buffers and callables in `finally`. Original five cases, input seeds,
exact comparisons, complete wrapper/allocation boundary, 10 warmups and 100
samples are unchanged. Supplemental replay work is outside measured time.
