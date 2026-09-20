# Shape and benchmark catalog: deepseek-v4-pro__dsa_sparse_mla_attn

This is a single-GPU MI355X (gfx950) callable benchmark. The serving capture used ISL 8192, OSL 1024, concurrency 64 and TP=8; those settings are context, not a substitute for the tensor shapes below.

The complete case IDs, argument inventories, original metadata, scalars, tensor attributes and source hashes are in [SHAPES.json](SHAPES.json). The task environment and benchmark commands are in [config.yaml](config.yaml).

The inventory has **6 shape records** and **4 benchmark cases**. The protected metadata's `num_cases` field is `4`; that field can count a different capture scope. Histogram observations and random value draws are not added to the benchmark count.

Callable: `sglang.kernels.ops.attention.dsa.tilelang_kernel:dpsk_v4_fp8_attention_fwd`. Baseline recorded in metadata: `same production seam, selected through frozen baseline binding`. Composite operator: a complete call may launch multiple physical kernels.

sparse paged MLA: for each query row, gather topk_length[i] fp8 KV blocks named by indices[i] (and extra_indices_in_kvcache[i] from the second/compressed pool), compute softmax(Q.K^T * softmax_scale) . V over the gathered set with a per-head attention sink, split-K over the KV axis (dpsk_v4_fp8_partial_kernel) then LSE-combine the splits (dpsk_v4_combine_kernel); returns (out[T,1,H,head_dim_v] bf16, lse[T,1,H] fp32). Both device kernels are emitted as `main_kernel`.

Callable returns (out, lse); protected call exposes out and applies the oracle-derived undefined mask only where prescribed. Both partial and combine kernels remain in the call.

| Benchmark case ID | Regime | Tensor geometry | Shape source |
| --- | --- | --- | --- |
| `prefill_m8192_x64` | prefill_x64 | q [8192×1×64×512] bfloat16 | `ut/meta.json#/case_specs/0`; JSON record 0 |
| `prefill_m8192_x1024` | prefill_x1024 | q [8192×1×64×512] bfloat16 | `ut/meta.json#/case_specs/1`; JSON record 1 |
| `decode_m64_x128` | decode_x128 | q [64×1×64×512] bfloat16 | `ut/meta.json#/case_specs/2`; JSON record 2 |
| `decode_m64_x1024` | decode_x1024 | q [64×1×64×512] bfloat16 | `ut/meta.json#/case_specs/3`; JSON record 3 |

Benchmark IDs above follow the protected timing builder and common adapter. All capture-only and boundary records remain in the JSON inventory with their original IDs and explicit benchmark membership. The `tensor_overrides` entries distinguish timing storage from captured storage: MiniMax prefill remaps selected request rows and changes `slot_ids` to int32; recurrent decode generates contiguous timing tensors.

| Argument / output | Recorded geometry and layout |
| --- | --- |
| `attn_sink` (input) | [64]; float32; strides unknown. unknown: physical strides are not recorded in this metadata. |
| `extra_indices_in_kvcache` (input) | 6 exact variants in JSON; int32; strides unknown. unknown: physical strides are not recorded in this metadata. |
| `extra_k_cache` (input) | [40128, 2, 1, 584], [40128, 64, 1, 584]; float8_e4m3fn; strides unknown. unknown: physical strides are not recorded in this metadata. |
| `extra_topk_length` (input) | [1], [64], [8192]; int32; strides unknown. unknown: physical strides are not recorded in this metadata. |
| `indices` (input) | [1, 1, 128], [64, 1, 128], [8192, 1, 128]; int32; strides unknown. unknown: physical strides are not recorded in this metadata. |
| `k_cache` (input) | [4013, 256, 1, 584]; float8_e4m3fn; strides unknown. unknown: physical strides are not recorded in this metadata. |
| `q` (input) | [1, 1, 64, 512], [64, 1, 64, 512], [8192, 1, 64, 512]; bfloat16; strides unknown. unknown: physical strides are not recorded in this metadata. |
| `topk_length` (input) | [1], [64], [8192]; int32; strides unknown. unknown: physical strides are not recorded in this metadata. |
| `return[0]` (output) | [1, 1, 64, 512], [64, 1, 64, 512], [8192, 1, 64, 512]; bfloat16; strides unknown. unknown: physical strides are not recorded in this metadata. |
| `return[1]` (output) | [1, 1, 64], [64, 1, 64], [8192, 1, 64]; float32; strides unknown. unknown: physical strides are not recorded in this metadata. |

Each JSON tensor record cites the metadata field or protected builder that establishes its geometry. A `null` shape, dtype or stride means it is not established by readable metadata; it is not a wildcard or permission to replace the fixture. Packed weights, sparse paging maps and routing-dependent lengths must retain the protected artifact representation.

**Evidence limit:** 60 tensor records have at least one unknown physical field. The machine catalog enumerates each field and its source. Exact opaque-fixture details require the hash-pinned tensor artifacts; this static catalog does not claim that they were inspected.

| Protected tensor artifact | Expected SHA-256 |
| --- | --- |
| `ut/reference_io.pt` | `981eb9be0909a11f831800c31e6399cffb0970d31f1393469d4e7e3a91e35f5a` |

Source pointers in the JSON are relative to this task directory. Tensor artifacts were not deserialized, and no task code or GPU benchmark was run to produce this catalog. Fresh validation remains governed by the task contract.
