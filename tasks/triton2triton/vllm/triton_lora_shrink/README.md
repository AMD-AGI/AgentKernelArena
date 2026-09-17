# triton_lora_shrink

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton `_lora_shrink_kernel` for maximum GPU throughput.
This is the LoRA A (shrink) kernel that projects input tokens from
hidden_size down to lora_rank using per-adapter LoRA A weight matrices.

The kernel computes output[slice, token] = scaling * input[token] @ lora_a[lora_id].T
for each token assigned to a LoRA adapter, with split-K reduction for
large hidden dimensions and support for multiple slices.

Key optimization opportunities:
- Block size tuning (BLOCK_M, BLOCK_N, BLOCK_K)
- SPLIT_K factor for K-dimension parallelism
- GROUP_SIZE_M for L2 cache reuse
- Memory access patterns for gather-based token indexing

Constraints:
- Must maintain the same function signature for `lora_shrink`
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

### Public operator and scored unit

Correctness checks the public `lora_shrink` operator, including its output reset
and supported input forms. Performance scores the declared `_lora_shrink_kernel`
invocation with prebuilt pointer/stride tables and a prepared output buffer.
It does not report the end-to-end latency of the Python `lora_shrink` wrapper.
Testing that wrapper for correctness does not add its pointer construction or
reset cost to the explicitly declared scored unit.

This is the original benchmark boundary for both baseline and candidate.
`output_tensor.zero_` prepares split-K accumulation on the measurement stream
outside the measured interval; all projection, scaling and accumulation work
remains in the timed kernel. Candidates may not move that work into pointer
construction, preparation or cached answers. The full-output, read-only-input
and changed-input replay checks still apply to the actual measured invocation.
Assess timing fairness against this declared unit and its symmetric preparation,
while continuing to reject omitted computation, different role boundaries or
incorrect timed/replayed outputs.


Protected checks retain the original full-output comparison on `.float()` values
at `atol=rtol=5e-2`, with shape/dtype/device and finite-output checks. Inputs,
weights, weight-list membership and routing tables are read-only. Performance
snapshots them in the protected data factory before the editable pointer-builder
runs. The original raw kernel, prebuilt pointer/stride tensors, five scored
cases, correctness seeds 42+i, performance seed 0, input scales, 10 warmups and
100 samples are preserved.

The actual `TimedRun` output is compared against pristine data. Replay changes
activations, weight values, adapter IDs and token routing in place while keeping
the original pointer tables and launch configuration. All input, output and
pointer-table state is restored, including pointer-builder and replay failures.

Shrink retains its original FP32 output buffer and FP16-rounded reference;
comparisons still convert both to float. Its original `output_tensor.zero_`
prepare callback remains outside measured split-K accumulation. Unscored
controls cover 83 shuffled tokens (including a 65-token adapter), disabled/empty
adapter groups, 4D weights, hidden515/rank19 tails, scaling1.25 and a nonzero
initial output to check the public wrapper's reset.

Additional unscored public-branch controls from PR105: Split-K first/last partial blocks, BF16 4D weights and inactive LoRA.
These use explicit `control-upstream-*` manifest rows. Original scored inputs,
numerical gates, seeds, warmups and sample counts remain unchanged.
