# Top-k boundary checks

The packing kernel must include every selected expert in every row, including
assignments after slot 31 and in intermediate tiles. For valid expert IDs in
`[0, num_experts)`, set bit `id % 32` in output word `id // 32`. Duplicate IDs
set a single bit. The output has shape `(rows, ceil(num_experts / 32))` and dtype
`uint32` on the input device. Masked tile-padding slots must not set any bits.
Valid two-dimensional `int16` inputs may have row padding, column gaps, or
both. Packing uses their logical values and leaves the input unchanged.

The first three `control-upstream-*` cases retain the PR105 controls for
duplicates, expert-word boundaries and row tails. The next two cases are
repository-authored top-k controls, identified by their `source` metadata in
`workloads.json`; their existing case IDs are retained for compatibility.
They use multiple rows, top-k 33 and 65, and 70 experts spanning three output
words. Each tile contributes expert bits that cannot be supplied by the other
tiles, and rows have different memberships. The native correctness action
rejects candidates that skip later tiles, omit the middle tile, or read another
row's later assignments. Partial final tiles and output-word tails are checked.
Three more repository-authored native controls use noncontiguous GPU inputs
with shape `(3, 2)` and 34 experts. Their row, column and combined-gap strides
make the old row-major linear load return incorrect expert bits; the controls
compare against the same independent membership oracle.

The original five scored shapes, exact-equality checks, 10 warmups and 100
samples remain unchanged. Graph timing can batch multiple wrapper calls in one
replay and report time per call. The benchmark guard checks every output from
the final captured batch, including after poisoning every output and replaying
the same graph with changed input IDs. Outputs from outstanding calls must not
touch the same bytes; disjoint views of one allocation, including interleaved
views, are allowed. This prevents
a wrapper from doing work only for selected calls while returning one shared
result for the others. Event timing still checks its one measured call and replay.
These output checks do not prove that every possible cached or copied
implementation performs independent work for identical inputs.

The implementation keeps the original single-tile branch for the existing
scored workloads. The wrapper makes only noncontiguous inputs contiguous
before that kernel reads them; contiguous scored inputs need no copy.
Simplifying the single-tile branch requires a new paired timing check;
equivalent expressions can produce different generated code.

This task freezes the initial candidate as its performance baseline. The
top-k packing repair changes that baseline's source version. Record the task
revision and baseline fingerprint when comparing runs; results from different
baseline versions must not be pooled as measurements against one baseline.

The existing implementation is adapted from the Apache-2.0-licensed
[vLLM packing kernel](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/model_executor/layers/fused_moe/gpt_oss_triton_kernels_moe.py).
Copyright contributors to the vLLM project.
