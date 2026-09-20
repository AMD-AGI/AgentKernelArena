# Shape and benchmark catalog: deepseek-v4-pro__moe_stage2_down_proj_reduce_opus_a8w4

This is a single-GPU MI355X (gfx950) callable benchmark. The serving capture used ISL 8192, OSL 1024, concurrency 64 and TP=8; those settings are context, not a substitute for the tensor shapes below.

The complete case IDs, argument inventories, original metadata, scalars, tensor attributes and source hashes are in [SHAPES.json](SHAPES.json). The task environment and benchmark commands are in [config.yaml](config.yaml).

The inventory has **5 shape records** and **5 benchmark cases**. The protected metadata's `num_cases` field is `5`; that field can count a different capture scope. Histogram observations and random value draws are not added to the benchmark count.

Callable: `aiter.ops.opus.moe_stage2_a8w4_fused_adapter:opus_a8w4_stage2_wrapper`. Baseline recorded in metadata: `aiter.ops.opus.moe_stage2_a8w4_fused_adapter:opus_a8w4_stage2_wrapper`. Composite operator: a complete call may launch multiple physical kernels.

stage-2 of the a8w4 fused MoE, as ONE segment: out[t,:] = sum_k topk_w[t,k] * (inter_states[t,k,:] @ w2[e(t,k)]^T) with mxfp8 activations x mxfp4 expert weights and e8m0 block scales, scattered through the moe_sorting block map. decode (route_out=False): the down-GEMM atomically accumulates the topk partials directly into a ZEROED out. prefill (route_out=True): the down-GEMM emits per-token-slot partials, then the reduce kernel sums the topk slots into out. Returns the same bf16 [token, model_dim] `out` tensor it was handed (written in place).

Returns the same supplied out buffer. Decode atomically accumulates into zeroed out; prefill emits per-slot values and reduces into out.

| Benchmark case ID | Regime | Tensor geometry | Shape source |
| --- | --- | --- | --- |
| `capture signature (record 0)` | prefill | inter_states [2048×6×384] float8_e4m3fn | `ut/meta.json#/cases/0`; JSON record 0 |
| `capture signature (record 1)` | decode | inter_states [256×6×384] float8_e4m3fn | `ut/meta.json#/cases/1`; JSON record 1 |
| `capture signature (record 2)` | decode | inter_states [64×6×384] float8_e4m3fn | `ut/meta.json#/cases/2`; JSON record 2 |
| `capture signature (record 3)` | decode | inter_states [1×6×384] float8_e4m3fn | `ut/meta.json#/cases/3`; JSON record 3 |
| `capture signature (record 4)` | prefill | inter_states [32768×6×384] float8_e4m3fn | `ut/meta.json#/cases/4`; JSON record 4 |

Benchmark IDs above follow the protected timing builder and common adapter. All capture-only and boundary records remain in the JSON inventory with their original IDs and explicit benchmark membership. The `tensor_overrides` entries distinguish timing storage from captured storage: MiniMax prefill remaps selected request rows and changes `slot_ids` to int32; recurrent decode generates contiguous timing tensors.

| Argument / output | Recorded geometry and layout |
| --- | --- |
| `inter_states` (input) | 5 exact variants in JSON; float8_e4m3fn; strides unknown. unknown: physical strides are not recorded in this metadata. |
| `w1` (input) | [384, 768, 3584]; float4_e2m1fn_x2; strides unknown. unknown: physical strides are not recorded in this metadata. |
| `w2` (input) | [384, 7168, 192]; float4_e2m1fn_x2; strides unknown. unknown: physical strides are not recorded in this metadata. |
| `sorted_token_ids` (input) | 5 exact variants in JSON; int32; strides unknown. unknown: physical strides are not recorded in this metadata. |
| `sorted_expert_ids` (input) | 5 exact variants in JSON; int32; strides unknown. unknown: physical strides are not recorded in this metadata. |
| `num_valid_ids` (input) | [2]; int32; strides unknown. unknown: physical strides are not recorded in this metadata. |
| `out` (input_output) | 5 exact variants in JSON; bfloat16; strides unknown. unknown: physical strides are not recorded in this metadata. |
| `a2_scale` (input) | 5 exact variants in JSON; float8_e8m0fnu; strides unknown. unknown: physical strides are not recorded in this metadata. |
| `sorted_weights` (input) | 5 exact variants in JSON; float32; strides unknown. unknown: physical strides are not recorded in this metadata. |
| `w2_scale` (input) | [2752512, 16]; float8_e8m0fnu; strides unknown. unknown: physical strides are not recorded in this metadata. |
| `return` (output) | 5 exact variants in JSON; bfloat16; strides unknown. unknown: physical strides are not recorded in this metadata. |

Each JSON tensor record cites the metadata field or protected builder that establishes its geometry. A `null` shape, dtype or stride means it is not established by readable metadata; it is not a wildcard or permission to replace the fixture. Packed weights, sparse paging maps and routing-dependent lengths must retain the protected artifact representation.

**Evidence limit:** 55 tensor records have at least one unknown physical field. The machine catalog enumerates each field and its source. Exact opaque-fixture details require the hash-pinned tensor artifacts; this static catalog does not claim that they were inspected.

| Protected tensor artifact | Expected SHA-256 |
| --- | --- |
| `ut/reference_io.pt` | `3232c6110462fa4a7cb9d057f9c9f36ebda27e7a19cbf379194198e2cb509337` |

Source pointers in the JSON are relative to this task directory. Tensor artifacts were not deserialized, and no task code or GPU benchmark was run to produce this catalog. Fresh validation remains governed by the task contract.
