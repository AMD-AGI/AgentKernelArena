# Shape and benchmark catalog: minimax-m3__gqa_share_sparse_decode_kernel

This is a single-GPU MI355X (gfx950) callable benchmark. The serving capture used ISL 8192, OSL 1024, concurrency 64 and TP=8; those settings are context, not a substitute for the tensor shapes below.

The complete case IDs, argument inventories, original metadata, scalars, tensor attributes and source hashes are in [SHAPES.json](SHAPES.json). The task environment and benchmark commands are in [config.yaml](config.yaml).

The inventory has **5 shape records** and **2 benchmark cases**. The protected metadata's `num_cases` field is `3`; that field can count a different capture scope. Histogram observations and random value draws are not added to the benchmark count.

Callable: `sglang.srt.layers.attention.minimax_sparse_ops.minimax_sparse:flash_decode_with_gqa_share_sparse`. Baseline recorded in metadata: `sglang.srt.layers.attention.minimax_sparse_ops.minimax_sparse:flash_decode_with_gqa_share_sparse`. Composite operator: a complete call may launch multiple physical kernels.

MiniMax-M3 sparse-attention decode MAIN pass: for each (kv head, batch) row, gather the topk=16 selected 128-token blocks of the paged K/V caches through req_to_token[slot_ids[b]], then run GQA-shared flash attention over those ~2048 keys for the gqa_group_size=8 query heads: o = softmax(q.K^T * sm_scale*k_scale + causal/len mask) . V * v_scale, with an optional sink logit. Split-K over the topk dimension writes (o_partial, lse_partial) which _merge_topk_attn_out_kernel reduces; the seam returns the merged o [batch, num_q_heads, head_dim] contiguous.

MiniMax-M3 sparse-attention decode MAIN pass: for each (kv head, batch) row, gather the topk=16 selected 128-token blocks of the paged K/V caches through req_to_token[slot_ids[b]], then run GQA-shared flash attention over those ~2048 keys for the gqa_group_size=8 query heads: o = softmax(q.K^T * sm_scale*k_scale + causal/len mask) . V * v_scale, with an optional sink logit. Split-K over the topk dimension writes (o_partial, lse_partial) which _merge_topk_attn_out_kernel reduces; the seam returns the merged o [batch, num_q_heads, head_dim] contiguous.

| Benchmark case ID | Regime | Tensor geometry | Shape source |
| --- | --- | --- | --- |
| `decode_m1_ctx8704` | decode | q [1×8×128] bfloat16 | `ut/meta.json#/cases/3`; JSON record 3 |
| `decode_m64_ctx8704` | decode | q [64×8×128] bfloat16 | `ut/meta.json#/cases/4`; JSON record 4 |

Benchmark IDs above follow the protected timing builder and common adapter. All capture-only and boundary records remain in the JSON inventory with their original IDs and explicit benchmark membership. The `tensor_overrides` entries distinguish timing storage from captured storage: MiniMax prefill remaps selected request rows and changes `slot_ids` to int32; recurrent decode generates contiguous timing tensors.

| Argument / output | Recorded geometry and layout |
| --- | --- |
| `k_cache` (input) | [4358330, 1, 128]; bfloat16; strides [128, 128, 1], unknown. contiguous generated timing input; unknown: physical strides are not recorded in this metadata. |
| `q` (input) | [1, 8, 128], [64, 8, 128], [7, 8, 128], [8, 8, 128]; bfloat16; strides [1024, 128, 1], unknown. contiguous generated timing input; unknown: physical strides are not recorded in this metadata. |
| `req_to_token` (input) | [1, 11268], [4097, 11268], [64, 11268]; int32; strides [11268, 1], unknown. contiguous generated timing input; unknown: physical strides are not recorded in this metadata. |
| `seq_lens` (input) | [1], [64], [7], [8]; int64; strides [1], unknown. contiguous generated timing input; unknown: physical strides are not recorded in this metadata. |
| `slot_ids` (input) | [1], [64], [7], [8]; int64; strides [1], unknown. contiguous generated timing input; unknown: physical strides are not recorded in this metadata. |
| `topk_idx` (input) | [1, 1, 16], [1, 64, 16], [1, 7, 16], [1, 8, 16]; int32; strides [1024, 16, 1], [16, 16, 1], unknown. generated selected blocks; unknown: physical strides are not recorded in this metadata. |
| `v_cache` (input) | [4358330, 1, 128]; bfloat16; strides [128, 128, 1], unknown. contiguous generated cache; unknown: physical strides are not recorded in this metadata. |
| `return` (output) | [1, 8, 128], [64, 8, 128], [7, 8, 128], [8, 8, 128]; bfloat16; strides unknown. merged attention output. |

Each JSON tensor record cites the metadata field or protected builder that establishes its geometry. A `null` shape, dtype or stride means it is not established by readable metadata; it is not a wildcard or permission to replace the fixture. Packed weights, sparse paging maps and routing-dependent lengths must retain the protected artifact representation.

**Evidence limit:** 26 tensor records have at least one unknown physical field. The machine catalog enumerates each field and its source. Exact opaque-fixture details require the hash-pinned tensor artifacts; this static catalog does not claim that they were inspected.

| Protected tensor artifact | Expected SHA-256 |
| --- | --- |
| `ut/reference_io.pt` | `835db1b0e1791b3be723288068630d702f78affc44334ee1a453651bd46c28e8` |
| `ut/timing_geometry.pt` | `53f447bdd43f30cdc8f7789c5a1a4693390b13272a1e53376d0207ea2eb530cd` |

Source pointers in the JSON are relative to this task directory. Tensor artifacts were not deserialized, and no task code or GPU benchmark was run to produce this catalog. Fresh validation remains governed by the task contract.
