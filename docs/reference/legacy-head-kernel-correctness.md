# Correctness in the older head-kernel suite

This source review refers to the older `head_kernels` branch at commit
`f3a79f16eb03e03781caf631b687b916f4c63aa3`. It is an implementation review,
not a new GPU validation or a reconstruction of a historical experiment.

The inspected tasks generate deterministic synthetic inputs at model-derived
dimensions, then compare candidate outputs with either an original kernel copy
or an independent mathematical reference. They do not need the new suite's
33.14 GB capture archive to perform those checks.

| Task | Input generation | Reference and comparison |
| --- | --- | --- |
| [FP8 block-scale GEMM](https://github.com/AMD-AGI/AgentKernelArena/blob/f3a79f16eb03e03781caf631b687b916f4c63aa3/tasks/head_kernels/gemm_a8w8_blockscale/scripts/harness_run.py#L55) | Seeded FP8 operands and FP32 scales | Independent FP32 dequantization and matrix multiplication. Allows up to 5% of elements outside `atol=rtol=0.05`, with an additional normalized similarity check. |
| [Fused MoE](https://github.com/AMD-AGI/AgentKernelArena/blob/f3a79f16eb03e03781caf631b687b916f4c63aa3/tasks/head_kernels/fused_moe_kernel/scripts/harness_run.py#L94) | Seeded operands/scales and deterministic round-robin expert routing | Separate original and editable kernels run on identically regenerated inputs. Checks output shape, cosine similarity and relative/absolute error. The original copy establishes implementation equivalence; it is not an independent mathematical derivation. |
| [Per-token FP8 quantization](https://github.com/AMD-AGI/AgentKernelArena/blob/f3a79f16eb03e03781caf631b687b916f4c63aa3/tasks/head_kernels/_per_token_group_quant_fp8/scripts/harness_run.py#L99) | Seeded BF16 values, allocated quantized outputs and scales | Candidate versus original kernel through the positional launcher. Compares both scales and quantized values. |
| [INT4 fused MoE](https://github.com/AMD-AGI/AgentKernelArena/blob/f3a79f16eb03e03781caf631b687b916f4c63aa3/tasks/head_kernels/fused_moe_int4_w4a16/test_harness.py#L53) | Generated activations, routing, packed weights, scales and optional zero points | Original vLLM kernel comparison with `allclose(rtol=0.02, atol=0.02)` over correctness cases broader than the timed subset. |

The shape/signature JSON files are metadata, not serialized numerical inputs.
Several inspected harnesses directly encode sequence length 1024 and
concurrencies 2, 32 and 64. Those regimes must not be substituted for the newer
workload's actual captured operator shapes.

## Numerical correctness does not establish valid speedup

The old mechanisms can detect ordinary wrong outputs, but the surrounding
benchmark has additional responsibilities:

- The FP8 GEMM correctness call checks a newly returned tensor, while its timing
  call supplies an output buffer. The actual timed output is not compared.
- The variable-length KKT harness launches a grid with a `B * H` second
  dimension, but that kernel branch uses only the head component. Repeated
  work can produce the correct output while inflating baseline latency.
- Correctness and performance generally run separately; a single correct
  launch does not prove correctness of repeated or graph-captured execution.
- Some old comparisons lack an explicit finite-value check. Rejection written
  only as `error > threshold` or `similarity < threshold` can miss NaN values.
- The original kernel, generators, launch interface, cases, tolerances, timer
  and reports all need enforced protection. Calling a file a golden reference
  does not itself prevent an optimizing agent from changing it.

Large captured tensors do not fix these issues. Generated inputs can support
strong checks when a protected evaluator chooses reproducible seeds, creates
valid layouts and routing, computes trusted expected outputs, and validates the
actual timed operation with changed inputs and correctly restored mutable state.

The current new suite still has 14 capture-dependent tasks and four tasks with
generated inputs. See [artifact preparation](../how-to/prepare-head-kernel-artifacts.md)
for its present requirements and the distinction between a generated-input
contract and replaying captured values.
