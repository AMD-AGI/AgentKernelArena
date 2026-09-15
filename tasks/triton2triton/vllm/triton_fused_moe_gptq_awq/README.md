# triton_fused_moe_gptq_awq

The starting candidate is implemented Triton. Improve the declared source files in place;
the framework freezes the initial implementation as the baseline. Baseline and candidate
actions execute only this workspace, with no fallback to another implementation.

Optimize the Triton `fused_moe_kernel_gptq_awq` for maximum GPU throughput.
This kernel performs fused MoE GEMM with GPTQ/AWQ quantized weights,
supporting 4-bit and 8-bit weight-only quantization with scales and zero points.

Key optimization opportunities:
- Block size tuning
- Efficient dequantization pipeline
- Memory access patterns for packed weights

Constraints:
- Must maintain the same function signature for `fused_moe_gptq_awq`
- Original random cases must match reference within atol=1.0, rtol=0.5;
  the exact arithmetic correctness-only controls require equality.


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

Correctness-only manifest controls: int4_explicit, int4_default, int8_explicit, int8_default.
They include repeated routes, absent/invalid experts, optional routing weights,
partial matrix dimensions, and deterministic basis activations. Invalid expert
assignments produce zero rows. Missing routing weights mean unit weights.
INT4/INT8 controls exercise explicit and default zero points and signed routing
weights. Each activation row has one nonzero integer value, and the integer
weights/zero points, power-of-two scales and dyadic routing weights yield exactly
representable FP16 intermediates and outputs. These controls therefore require
exact numerical equality, independently of the unchanged atol=1.0/rtol=0.5 rule
for the five original random cases. The original relative tolerance alone admits
a uniformly half-scaled answer; the exact controls reject that error, wrong
packing/zero points/routing, and incorrect zero rows. They add no score rows or
fitted baseline thresholds; the scoring timer is unchanged.

The shipped quantized kernel now masks weight loads in a partial K block, matching
its existing activation/scale/zero-point masks. Previously K=48 reached an unmasked
weight load beyond the expert matrix. All original scored K dimensions divide 32.
