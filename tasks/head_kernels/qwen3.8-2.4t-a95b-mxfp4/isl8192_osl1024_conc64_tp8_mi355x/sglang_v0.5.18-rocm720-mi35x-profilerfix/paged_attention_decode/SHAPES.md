# Shape and benchmark catalog: qwen3.8-2.4t__paged_attention_decode

This is a single-GPU MI355X (gfx950) callable benchmark. The serving capture used ISL 8192, OSL 1024, concurrency 64 and TP=8; those settings are context, not a substitute for the tensor shapes below.

The complete case IDs, argument inventories, original metadata, scalars, tensor attributes and source hashes are in [SHAPES.json](SHAPES.json). The task environment and benchmark commands are in [config.yaml](config.yaml).

The inventory has **1 shape records** and **1 benchmark cases**. The protected metadata's `num_cases` field is `1`; that field can count a different capture scope. Histogram observations and random value draws are not added to the benchmark count.

Callable: `aiter.ops.attention:paged_attention_ragged`. Baseline recorded in metadata: `same production seam, selected through frozen baseline binding`. Selected callable seam; kernel launch count is not asserted by this catalog.

Paged softmax attention over the real ragged block metadata and referenced KV pages; the callable writes and returns the supplied BF16 output buffer while treating query, KV, metadata, and scales as immutable.

Returns the supplied out buffer; out and workspace_buffer may mutate. Query, KV caches, metadata and scales remain unchanged.

| Benchmark case ID | Regime | Tensor geometry | Shape source |
| --- | --- | --- | --- |
| `decode_m64_live` | decode | query [64×8×256] bfloat16 | `ut/meta.json#/workload/cases/0`; JSON record 0 |

Benchmark IDs above follow the protected timing builder and common adapter. All capture-only and boundary records remain in the JSON inventory with their original IDs and explicit benchmark membership. The `tensor_overrides` entries distinguish timing storage from captured storage: MiniMax prefill remaps selected request rows and changes `slot_ids` to int32; recurrent decode generates contiguous timing tensors.

| Argument / output | Recorded geometry and layout |
| --- | --- |
| `out` (input_output) | [64, 8, 256]; bfloat16; strides unknown. unknown: physical strides are not recorded in this metadata. |
| `workspace_buffer` (input_output) | [71926272]; uint8; strides unknown. unknown: physical strides are not recorded in this metadata. |
| `query` (input) | [64, 8, 256]; bfloat16; strides [2048, 256, 1]. recorded row-major. |
| `key_cache` (input) | [521351, 1, 1, 256]; bfloat16; strides unknown. NHD; lossless referenced-page compaction. |
| `value_cache` (input) | [521351, 1, 1, 256]; bfloat16; strides unknown. NHD; lossless referenced-page compaction. |
| `kv_indptr` (input) | [65]; int32; strides unknown. unknown: physical strides are not recorded in this metadata. |
| `kv_page_indices` (input) | [521351]; int32; strides unknown. unknown: physical strides are not recorded in this metadata. |
| `kv_last_page_lens` (input) | [198]; int32; strides unknown. unknown: physical strides are not recorded in this metadata. |
| `k_scale` (input) | [1]; float32; strides unknown. unknown: physical strides are not recorded in this metadata. |
| `v_scale` (input) | [1]; float32; strides unknown. unknown: physical strides are not recorded in this metadata. |
| `return` (output) | [64, 8, 256]; bfloat16; strides unknown. unknown: physical strides are not recorded in this metadata. |

Each JSON tensor record cites the metadata field or protected builder that establishes its geometry. A `null` shape, dtype or stride means it is not established by readable metadata; it is not a wildcard or permission to replace the fixture. Packed weights, sparse paging maps and routing-dependent lengths must retain the protected artifact representation.

Persistent live capture with compacted but lossless referenced K/V pages, real ragged/page metadata and scales, plus the complete returned output. Random parity changes query values only.

**Evidence limit:** 10 tensor records have at least one unknown physical field. The machine catalog enumerates each field and its source. Exact opaque-fixture details require the hash-pinned tensor artifacts; this static catalog does not claim that they were inspected.

| Protected tensor artifact | Expected SHA-256 |
| --- | --- |
| `ut/reference_io.pt` | `9c40d52c145027f8affca9e41bbf1100189fde724d42d085ac6296d57020ac56` |

Source pointers in the JSON are relative to this task directory. Tensor artifacts were not deserialized, and no task code or GPU benchmark was run to produce this catalog. Fresh validation remains governed by the task contract.
