# triton_expert_kernel

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton `expert_triton_kernel` for maximum GPU throughput.
This is a per-expert GEMM kernel that computes C = A @ B for one expert's tokens.

Key optimization opportunities:
- Block size tuning (BLOCK_M, BLOCK_N, BLOCK_K)
- K-loop pipelining
- Memory access patterns

Constraints:
- Must maintain the same function signature for `expert_gemm`
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


Protected checks validate the complete FP16 output shape, device, dtype, and
finite values against the original FP32-matmul-then-FP16 reference at
atol=rtol=5e-2. Both operands are read-only and the reference uses pristine
copies. An unscored `(M,K,N)=(67,35,71)` control covers partial tiles and the
wrapper's explicit stride support using noncontiguous A and B.

The five original correctness seeds and performance seed 0, original input
scales, full allocating wrapper, ten warmups, and 100 timing samples are
unchanged. The actual `TimedRun` output is checked and poisoned before an exact
replay with both operands changed. Verification is outside timing; caller inputs
are restored even if replay or checking fails.

Additional unscored public-branch controls from PR105: Sub-tile/multi-tile M/N/K and BF16 strided public GEMM.
They have independent `control-upstream-*` manifest rows; existing scored
inputs, numerical gates and timing remain unchanged.
