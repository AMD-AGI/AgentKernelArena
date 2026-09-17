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


## Protected kernel and validation boundaries

Only the declared Triton kernel and permitted implementation helpers are editable.
The host launch, routing, output allocation and standalone wrapper remain protected,
so the task always invokes the declared implementation. All five scored shapes,
seed 42 + case index, input distributions, atol=0.05/rtol=0.05, 10 warmups,
100 samples and original graph timing are retained.

Every correctness check uses a private reference computed before the candidate,
checks exact output shape/dtype/device and finite values, and detects mutation of
all read-only inputs. Performance checks both the original measured output and
the same captured graph with changed activations and poisoned outputs; all checks
and copies occur outside timing, and inputs are restored even on failure.

Additional manifest cases are correctness-only: zero_experts, ragged_tail.
They exercise inactive output zeros and/or partial matrix tiles with deterministic
integer-valued operands. The original five score rows remain unchanged.
The shipped kernel now masks weight columns beyond N in the final tile; previously
a partial N tile loaded outside that expert matrix. The original N-multiple-of-64
workloads are unchanged.
