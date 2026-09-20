# Shape and benchmark catalog: deepseek-v4-pro__moe_stage1_grouped_gemm_silu_flydsl

This is a single-GPU MI355X (gfx950) callable benchmark. The serving capture used ISL 8192, OSL 1024, concurrency 64 and TP=8; those settings are context, not a substitute for the tensor shapes below.

The complete case IDs, argument inventories, original metadata, scalars, tensor attributes and source hashes are in [SHAPES.json](SHAPES.json). The task environment and benchmark commands are in [config.yaml](config.yaml).

The inventory has **5 shape records** and **5 benchmark cases**. The protected metadata's `num_cases` field is `5`; that field can count a different capture scope. Histogram observations and random value draws are not added to the benchmark count.

Callable: `aiter.ops.flydsl:flydsl_moe_stage1`. Baseline recorded in metadata: `same production seam, selected through frozen baseline binding`. Selected callable seam; kernel launch count is not asserted by this catalog.

grouped per-expert GEMM + routing: out[t,k,:] = quant_fp8( silu(gate)*up ) where [gate|up] = a[t,:] @ w1[e(t,k)]^T (mxfp8 act x mxfp4 weights, e8m0 block scales), scattered by the moe_sorting block map; returns (inter_states_fp8, out_scale_sorted_e8m0)

Returns intermediate FP8 states plus sorted E8M0 scales. The first-component call wrapper and whole-tuple correctness probe have distinct surfaces.

| Benchmark case ID | Regime | Tensor geometry | Shape source |
| --- | --- | --- | --- |
| `capture signature (record 0)` | prefill | a [2048×7168] float8_e4m3fn | `ut/meta.json#/cases/0`; JSON record 0 |
| `capture signature (record 1)` | decode | a [256×7168] float8_e4m3fn | `ut/meta.json#/cases/1`; JSON record 1 |
| `capture signature (record 2)` | decode | a [64×7168] float8_e4m3fn | `ut/meta.json#/cases/2`; JSON record 2 |
| `capture signature (record 3)` | decode | a [1×7168] float8_e4m3fn | `ut/meta.json#/cases/3`; JSON record 3 |
| `capture signature (record 4)` | prefill | a [32768×7168] float8_e4m3fn | `ut/meta.json#/cases/4`; JSON record 4 |

Benchmark IDs above follow the protected timing builder and common adapter. All capture-only and boundary records remain in the JSON inventory with their original IDs and explicit benchmark membership. The `tensor_overrides` entries distinguish timing storage from captured storage: MiniMax prefill remaps selected request rows and changes `slot_ids` to int32; recurrent decode generates contiguous timing tensors.

| Argument / output | Recorded geometry and layout |
| --- | --- |
| `a` (input) | 5 exact variants in JSON; float8_e4m3fn; strides unknown. unknown: physical strides are not recorded in this metadata. |
| `a1_scale` (input) | 5 exact variants in JSON; float8_e8m0fnu; strides unknown. unknown: physical strides are not recorded in this metadata. |
| `num_valid_ids` (input) | [2]; int32; strides unknown. unknown: physical strides are not recorded in this metadata. |
| `sorted_expert_ids` (input) | 5 exact variants in JSON; int32; strides unknown. unknown: physical strides are not recorded in this metadata. |
| `sorted_token_ids` (input) | 5 exact variants in JSON; int32; strides unknown. unknown: physical strides are not recorded in this metadata. |
| `topk_ids` (input) | 5 exact variants in JSON; int32; strides unknown. unknown: physical strides are not recorded in this metadata. |
| `w1` (input) | [384, 768, 3584]; float4_e2m1fn_x2; strides unknown. unknown: physical strides are not recorded in this metadata. |
| `w1_scale` (input) | [294912, 224]; float8_e8m0fnu; strides unknown. unknown: physical strides are not recorded in this metadata. |
| `return[0]` (output) | 5 exact variants in JSON; float8_e4m3fn; strides unknown. unknown: physical strides are not recorded in this metadata. |
| `return[1]` (output) | unknown; float8_e8m0fnu; strides unknown. sorted/padded scale layout; exact returned shape only in tensor oracle. |

Each JSON tensor record cites the metadata field or protected builder that establishes its geometry. A `null` shape, dtype or stride means it is not established by readable metadata; it is not a wildcard or permission to replace the fixture. Packed weights, sparse paging maps and routing-dependent lengths must retain the protected artifact representation.

**Evidence limit:** 50 tensor records have at least one unknown physical field. The machine catalog enumerates each field and its source. Exact opaque-fixture details require the hash-pinned tensor artifacts; this static catalog does not claim that they were inspected.

| Protected tensor artifact | Expected SHA-256 |
| --- | --- |
| `ut/reference_io.pt` | `722c170ae8e3c12d91c047d6acfd852ffc16beefecfd6b943fef51c6959159f4` |

Source pointers in the JSON are relative to this task directory. Tensor artifacts were not deserialized, and no task code or GPU benchmark was run to produce this catalog. Fresh validation remains governed by the task contract.
