# triton_pack_bitmatrix

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton `pack_bitmatrix` kernel for maximum GPU throughput.
This kernel packs topk expert IDs into a bitmatrix format where each bit
indicates whether a token is assigned to that expert. Used for OAI Triton
kernels MoE routing.

Key optimization opportunities:
- Block size tuning (BLOCK_SIZE_M, BLOCK_SIZE_K)
- Efficient bit manipulation
- Reduction optimization

Constraints:
- Must maintain the same function signature for `pack_topk_to_bitmatrix`
- Output must exactly match the reference bitmatrix


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.

