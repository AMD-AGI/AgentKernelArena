# triton_fused_moe_lora

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton `fused_moe_lora_kernel` for maximum GPU throughput.
This kernel fuses MoE expert routing with LoRA adapter computation,
performing both shrink (A) and expand (B) LoRA operations within the
MoE expert computation pipeline.

The kernel supports both naive block assignment (flat expert_ids) and
sorted token assignment, per-expert LoRA weights, optional routing
weight multiplication, split-K reduction, and L2 cache hints for
weight loading.

Note: The distributed communication paths (all_gather/all_reduce for
fully_sharded mode) are excluded; only the local compute path is tested.

Key optimization opportunities:
- Block size tuning (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)
- GROUP_SIZE_M for L2 cache reuse
- SPLIT_K factor for K-dimension parallelism
- USE_B_L2_CACHE for weight caching strategy
- Memory access patterns for grouped GEMM with pointer indirection

Constraints:
- Must maintain the same function signature for `fused_moe_lora`
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


The protected manifest requires the declared kernel symbols to remain Triton JIT
functions, including kernels originally decorated with `@triton.jit()`. Removing
the decorator is rejected before compilation. This structural check supplements
the numerical and timed-path checks; it does not by itself attest every dispatch.

The public wrapper adds LoRA contributions to the supplied output; it does not
clear that output. With `mul_routed_weight=False`, input activations have M rows
and each token is shared across its top-k routes. With `mul_routed_weight=True`,
activations have M*top_k rows in flattened route order; the expand stage applies
the corresponding routing weight. Both use `topk_weights.shape[0] == M` and
output shape `[M, top_k, output_columns]`. `offset` selects the first output
column. Disabled adapters, missing adapters/experts and surrounding columns
preserve the caller's original values.

Unscored controls exercise both weighted/unweighted and sorted/naive modes,
83 tokens, two slices, K35/rank19/N67 tails, offset5, nonzero initial output,
negative/zero routing weights, disabled adapters and empty expert groups. Sorted
inputs use per-adapter/expert groups padded to 64-token blocks, including a
multi-block group and nonsequential token order. These controls use the actual
public wrapper and the original prepared shrink/expand pair.

The original five scored cases, input distributions, seeds (42+i correctness,
0 performance), and full-output `atol=rtol=5e-2` comparison remain. The original
FP32 two-matmul reference remains used for every scored case; the weighted
extension follows the same arithmetic on flattened route activations. Output
metadata, finite values, every route and untouched columns are checked. Input
activations, weight-list membership, both weight stacks and all routing tables
are read-only, including during pointer construction.

Timing retains the original two raw launches with prebuilt pointers and
intermediate storage, SPLIT_K=1, 10 warmups and 100 samples. The original
`output.zero_` preparation stays outside timing. Actual `TimedRun` outputs are
checked before and after changing operands, expert/adapter routing and enabled
adapters in place. Replay poisons output and intermediate storage, then runs the
same captured pair with the original preparation. Inputs, pointer tables,
intermediate storage and output are restored even when replay fails. No new
reset, reference work or allocation is added to the measured pair.

Additional unscored public-branch controls from PR105: BF16 sorted routed LoRA and split-K2/no-L2-cache specialization.
These use explicit `control-upstream-*` manifest rows. Original scored inputs,
numerical gates, seeds, warmups and sample counts remain unchanged.
