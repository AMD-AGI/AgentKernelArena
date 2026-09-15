# triton_update_eagle_inputs

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton `_update_eagle_inputs_kernel` for maximum GPU throughput.
This kernel updates input IDs from draft tokens, copies hidden states from
output to input buffers, and increments positions and seq_lens (clamped to
max_model_len) between EAGLE speculative decoding steps.

Key optimization opportunities:
- Block size tuning for hidden state copy
- Vectorized memory access
- Fusing scalar operations

Constraints:
- Must maintain the same function signature for `update_eagle_inputs`
- Hidden state output must match within atol=1e-5, rtol=1e-5
- Integer outputs must match exactly


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.

