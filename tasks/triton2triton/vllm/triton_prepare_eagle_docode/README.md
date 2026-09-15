# triton_prepare_eagle_docode

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton `_prepare_eagle_docode_kernel` for maximum GPU throughput.
This kernel prepares decode-step inputs for EAGLE speculative decoding: copies
draft tokens to input IDs, copies hidden states from output to input buffers,
computes positions and seq_lens, and initializes query_start_loc for CUDA graphs.
Note: "docode" preserves the original vLLM spelling.

Key optimization opportunities:
- Block size tuning for hidden state copy
- Vectorized memory access for hidden states
- Efficient padding loop for query_start_loc

Constraints:
- Must maintain the same function signature for `prepare_eagle_decode`
- Output must match reference within atol=1e-5, rtol=1e-5 for hidden states
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

