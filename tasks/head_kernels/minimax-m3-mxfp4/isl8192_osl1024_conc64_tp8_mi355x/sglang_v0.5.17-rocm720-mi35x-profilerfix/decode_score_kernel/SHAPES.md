# Shape and benchmark catalog: minimax-m3__decode_score_kernel

This is a single-GPU MI355X (gfx950) callable benchmark. The serving capture used ISL 8192, OSL 1024, concurrency 64 and TP=8; those settings are context, not a substitute for the tensor shapes below.

The complete case IDs, argument inventories, original metadata, scalars, tensor attributes and source hashes are in [SHAPES.json](SHAPES.json). The task environment and benchmark commands are in [config.yaml](config.yaml).

The inventory has **7 shape records** and **2 benchmark cases**. The protected metadata's `num_cases` field is `5`; that field can count a different capture scope. Histogram observations and random value draws are not added to the benchmark count.

Callable: `sglang.kernels.ops.attention.minimax_sparse.decode.flash_with_topk_idx:flash_decode_with_topk_idx`. Baseline recorded in metadata: `sglang.kernels.ops.attention.minimax_sparse.decode.flash_with_topk_idx:flash_decode_with_topk_idx`. Composite operator: a complete call may launch multiple physical kernels.

MiniMax-M3 sparse-attention decode INDEX pass: per (index head, batch) compute the block score max_j (q . k_j) * sm_scale over each block_size=128 chunk of the paged K index cache gathered through req_to_token (score_type='max'), force-select the init_blocks/local_blocks, then top-k (k=16) the block scores -> topk_idx [num_q_heads, batch, topk] int32, front-packed, -1 padded. disable_index_value=True so no index VALUE attention runs and the seam returns (None, topk_idx, None).

MiniMax-M3 sparse-attention decode INDEX pass: per (index head, batch) compute the block score max_j (q . k_j) * sm_scale over each block_size=128 chunk of the paged K index cache gathered through req_to_token (score_type='max'), force-select the init_blocks/local_blocks, then top-k (k=16) the block scores -> topk_idx [num_q_heads, batch, topk] int32, front-packed, -1 padded. disable_index_value=True so no index VALUE attention runs and the seam returns (None, topk_idx, None).

| Benchmark case ID | Regime | Tensor geometry | Shape source |
| --- | --- | --- | --- |
| `decode_m1_ctx8554` | decode | q [1×1×128] bfloat16 | `ut/meta.json#/cases/5`; JSON record 5 |
| `decode_m64_ctx8554` | decode | q [64×1×128] bfloat16 | `ut/meta.json#/cases/6`; JSON record 6 |

Benchmark IDs above follow the protected timing builder and common adapter. All capture-only and boundary records remain in the JSON inventory with their original IDs and explicit benchmark membership. The `tensor_overrides` entries distinguish timing storage from captured storage: MiniMax prefill remaps selected request rows and changes `slot_ids` to int32; recurrent decode generates contiguous timing tensors.

| Argument / output | Recorded geometry and layout |
| --- | --- |
| `k_cache` (input) | [4358330, 1, 128]; bfloat16; strides [128, 128, 1], unknown. contiguous generated timing input; unknown: physical strides are not recorded in this metadata. |
| `q` (input) | [1, 1, 128], [64, 1, 128]; bfloat16; strides [128, 128, 1], unknown. contiguous generated timing input; unknown: physical strides are not recorded in this metadata. |
| `req_to_token` (input) | [1, 11268], [4097, 11268], [64, 11268]; int32; strides [11268, 1], unknown. contiguous generated timing input; unknown: physical strides are not recorded in this metadata. |
| `seq_lens` (input) | [1], [64]; int64; strides [1], unknown. contiguous generated timing input; unknown: physical strides are not recorded in this metadata. |
| `slot_ids` (input) | [1], [64]; int64; strides [1], unknown. contiguous generated timing input; unknown: physical strides are not recorded in this metadata. |
| `return[1]` (output) | [1, 1, 16], [1, 64, 16]; int32; strides unknown. front-packed block indices; -1 padding. |

Each JSON tensor record cites the metadata field or protected builder that establishes its geometry. A `null` shape, dtype or stride means it is not established by readable metadata; it is not a wildcard or permission to replace the fixture. Packed weights, sparse paging maps and routing-dependent lengths must retain the protected artifact representation.

**Evidence limit:** 32 tensor records have at least one unknown physical field. The machine catalog enumerates each field and its source. Exact opaque-fixture details require the hash-pinned tensor artifacts; this static catalog does not claim that they were inspected.

| Protected tensor artifact | Expected SHA-256 |
| --- | --- |
| `ut/reference_io.pt` | `12dcffd9ba5a3ac20fa2fbce3949f1789a57e87c5aed80de5e0176c60cea1cbe` |
| `ut/timing_geometry.pt` | `7a90e22e1248debfa30c40ab89b908399e394fc81de65f673d2ba83deee1fabb` |

Source pointers in the JSON are relative to this task directory. Tensor artifacts were not deserialized, and no task code or GPU benchmark was run to produce this catalog. Fresh validation remains governed by the task contract.
