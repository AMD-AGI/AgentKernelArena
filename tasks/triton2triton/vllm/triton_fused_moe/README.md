# triton_fused_moe

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton `fused_moe_kernel` for maximum GPU throughput.
This is the main MoE GEMM kernel that multiplies each token by its assigned
expert weight matrix using sorted token IDs and expert IDs.

The kernel computes C[token] = A[token // topk] @ B[expert].T with grouped
block scheduling for L2 cache reuse, and optional routing weight multiplication.

Key optimization opportunities:
- Block size tuning (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)
- GROUP_SIZE_M for L2 cache reuse
- Memory access patterns and prefetching
- Compute type selection

Constraints:
- Must maintain the same function signature for `fused_moe`
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
Host routing and output allocation remain protected; the measured callable still
includes the complete original public wrapper. All five scored workloads, input
distributions, seed 42 + case index, original numerical tolerance, 10 warmups and
100 samples remain unchanged. Timing stays explicit GPU-event fallback for host
routing/dynamic allocation; it never changes because of a candidate failure.

Correctness uses private pristine inputs and checks exact shape, dtype, device,
finiteness and the original all-element numerical rule. All input tensors, including
quantization/routing metadata, are read-only. Performance retains and validates the
last actual event-measured output, then checks the same callable with changed
activations and poisoned output. References and input checks stay outside timing;
inputs are restored in a finally block.

Correctness-only manifest controls: optional_weights, unweighted, invalid_experts.
They include repeated routes, absent/invalid experts, optional routing weights,
partial matrix dimensions, and deterministic basis activations. Invalid expert
assignments produce zero rows. Missing routing weights mean unit weights.
