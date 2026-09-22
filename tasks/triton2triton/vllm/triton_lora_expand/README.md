# triton_lora_expand

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton `_lora_expand_kernel` for maximum GPU throughput.
This is the LoRA B (expand) kernel that multiplies the low-rank intermediate
activations by LoRA B weight matrices and accumulates into the output tensor.

The kernel computes output[token] += input[token] @ lora_b[lora_id].T for
each token assigned to a LoRA adapter, with support for multiple slices
(e.g., QKV projections) and optional input addition.

Key optimization opportunities:
- Block size tuning (BLOCK_M, BLOCK_N, BLOCK_K)
- Memory access patterns for the gather-based token indexing
- Efficient handling of the SAME_STRIDE vs per-slice stride path
- Compute type selection and casting

Constraints:
- Must maintain the same function signature for `lora_expand`
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

Expand retains its scored `ADD_INPUTS=False` overwrite path without a new reset
inside timing. Unscored controls cover 83 shuffled tokens (including a 65-token
adapter), disabled/empty adapter groups, 4D weights, partial rank19, offset9,
different slice widths17/33, and both overwrite/addition behavior with nonzero
initial output. Heterogeneous slices use cumulative offsets, as the original
pointer-builder does; the original equal-width reference remains used for all
scored cases. Unassigned tokens and surrounding output columns are preserved.
