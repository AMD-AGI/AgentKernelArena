# Shape and benchmark catalog: kimi-k3__moe_gemm2_stage2

This is a single-GPU MI355X (gfx950) callable benchmark. The serving capture used ISL 8192, OSL 1024, concurrency 64 and TP=8; those settings are context, not a substitute for the tensor shapes below.

The complete case IDs, argument inventories, original metadata, scalars, tensor attributes and source hashes are in [SHAPES.json](SHAPES.json). The task environment and benchmark commands are in [config.yaml](config.yaml).

The inventory has **4 shape records** and **4 benchmark cases**. The protected metadata's `num_cases` field is `not declared`; that field can count a different capture scope. Histogram observations and random value draws are not added to the benchmark count.

Callable: `aiter.ops.flydsl.moe_kernels:flydsl_moe_stage2`. Baseline recorded in metadata: `baseline_src/flydsl/moe_kernels.py:flydsl_moe_stage2`. Composite operator: a complete call may launch multiple physical kernels.

grouped per-expert GEMM + routing

Caller supplies out[M,3584], which is written and returned. Atomic mode requires zero initialization; reduce mode writes per-slot intermediates then reduces.

| Benchmark case ID | Regime | Tensor geometry | Shape source |
| --- | --- | --- | --- |
| `prefill_M16384` | prefill | inter_states [16384×16×384] float8_e4m3fn | `ut/meta.json#/case_specs/0`; JSON record 0 |
| `prefill_M8192` | prefill | inter_states [8192×16×384] float8_e4m3fn | `ut/meta.json#/case_specs/1`; JSON record 1 |
| `decode_M1` | decode | inter_states [1×16×384] float8_e4m3fn | `ut/meta.json#/case_specs/2`; JSON record 2 |
| `decode_M64` | decode | inter_states [64×16×384] float8_e4m3fn | `ut/meta.json#/case_specs/3`; JSON record 3 |

Benchmark IDs above follow the protected timing builder and common adapter. All capture-only and boundary records remain in the JSON inventory with their original IDs and explicit benchmark membership. The `tensor_overrides` entries distinguish timing storage from captured storage: MiniMax prefill remaps selected request rows and changes `slot_ids` to int32; recurrent decode generates contiguous timing tensors.

| Argument / output | Recorded geometry and layout |
| --- | --- |
| `inter_states` (input) | [1, 16, 384], [16384, 16, 384], [64, 16, 384], [8192, 16, 384]; float8_e4m3fn; strides [6144, 384, 1]. contiguous generated operand at frozen case shape. |
| `w2` (input) | [896, 3584, 192]; float4_e2m1fn_x2; strides [688128, 192, 1]. contiguous generated operand at frozen case shape. |
| `w2_scale` (input) | [3211264, 16]; float8_e8m0fnu; strides [16, 1]. contiguous generated operand at frozen case shape. |
| `a2_scale` (input) | [188416, 16], [28672, 16], [29696, 16], [319488, 16]; float8_e8m0fnu; strides [16, 1]. contiguous generated operand at frozen case shape. |
| `sorted_token_ids` (input) | unknown; int32; strides unknown. frozen real routing; exact shape/stride only in tensor oracle. |
| `sorted_expert_ids` (input) | unknown; int32; strides unknown. frozen real routing; exact shape/stride only in tensor oracle. |
| `num_valid_ids` (input) | unknown; int32; strides unknown. frozen real routing; exact shape/stride only in tensor oracle. |
| `sorted_weights` (input) | unknown; None; strides unknown. optional frozen routing operand; presence/shape/stride only in tensor oracle. |
| `expert_mask` (input) | unknown; None; strides unknown. optional frozen routing operand; presence/shape/stride only in tensor oracle. |
| `topk_ids` (input) | unknown; None; strides unknown. optional frozen routing operand; presence/shape/stride only in tensor oracle. |
| `out` (input_output) | [1, 3584], [16384, 3584], [64, 3584], [8192, 3584]; bfloat16; strides [3584, 1]. caller-owned contiguous; written in place. |
| `return` (output) | [1, 3584], [16384, 3584], [64, 3584], [8192, 3584]; bfloat16; strides [3584, 1]. aliases supplied output. |

Each JSON tensor record cites the metadata field or protected builder that establishes its geometry. A `null` shape, dtype or stride means it is not established by readable metadata; it is not a wildcard or permission to replace the fixture. Packed weights, sparse paging maps and routing-dependent lengths must retain the protected artifact representation.

**Evidence limit:** 24 tensor records have at least one unknown physical field. The machine catalog enumerates each field and its source. Exact opaque-fixture details require the hash-pinned tensor artifacts; this static catalog does not claim that they were inspected.

| Protected tensor artifact | Expected SHA-256 |
| --- | --- |
| `ut/reference_io.pt` | `5a3fefb2af1c8a2981980d68e4f90085985fd5f0301b69814d85de5c1d801a53` |

Source pointers in the JSON are relative to this task directory. Tensor artifacts were not deserialized, and no task code or GPU benchmark was run to produce this catalog. Fresh validation remains governed by the task contract.
