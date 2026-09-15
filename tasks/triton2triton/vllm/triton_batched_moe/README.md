# triton_batched_moe

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton `batched_triton_kernel` for maximum GPU throughput.
This kernel performs batched MoE GEMM with all experts in a single launch
using a 2D grid (expert_id, M_blocks*N_blocks). It skips experts with
zero tokens and handles variable token counts per expert.

Key optimization opportunities:
- Block size tuning for the 2D grid
- K-loop pipelining
- Expert-level parallelism

Constraints:
- Must maintain the same function signature for `batched_moe_gemm`
- Output must match reference within atol=5e-2, rtol=5e-2 for float16


## Evaluation contract

Run `python3 _arena_eval.py validate-task`, or `python3 _arena_eval.py baseline|candidate compile|correctness|performance`
(with one role and one action). Use `ARENA_EVAL_PHASE=candidate_evaluation` for submitted candidates.
`workloads.json` declares all five original cases; the adapter verifies it against the
protected harness table. Original input seeds, comparisons, tolerances, warmups, sample
counts and graph/event timing remain in `scripts/task_runner.py`. Compilation includes
syntax and import/interface checks. Missing candidates, incomplete measurements and
invalid timing fail; commands emit `arena-eval-v1`, never final Arena score reports.
Canonical benchmark helpers must be materialized by Arena; do not edit their generated regions.

