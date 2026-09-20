# Shape and benchmark catalog: minimax-m3__gqa_share_sparse_fwd_kernel

This is a single-GPU MI355X (gfx950) callable benchmark. The serving capture used ISL 8192, OSL 1024, concurrency 64 and TP=8; those settings are context, not a substitute for the tensor shapes below.

The complete case IDs, argument inventories, original metadata, scalars, tensor attributes and source hashes are in [SHAPES.json](SHAPES.json). The task environment and benchmark commands are in [config.yaml](config.yaml).

The inventory has **3 shape records** and **1 benchmark cases**. The protected metadata's `num_cases` field is `3`; that field can count a different capture scope. Histogram observations and random value draws are not added to the benchmark count.

Callable: `sglang.kernels.ops.attention.minimax_sparse.prefill.topk_sparse:flash_prefill_with_gqa_share_sparse`. Baseline recorded in metadata: `sglang.kernels.ops.attention.minimax_sparse.prefill.topk_sparse:flash_prefill_with_gqa_share_sparse`. Selected callable seam; kernel launch count is not asserted by this catalog.

block-sparse GQA prefill: for each query, softmax over the topk selected KV blocks of (Q·Kᵀ)·sm_scale (+ causal mask, + optional attn sink) · V, paged K/V gathered through req_to_token

block-sparse GQA prefill: for each query, softmax over the topk selected KV blocks of (Q·Kᵀ)·sm_scale (+ causal mask, + optional attn sink) · V, paged K/V gathered through req_to_token

| Benchmark case ID | Regime | Tensor geometry | Shape source |
| --- | --- | --- | --- |
| `prefill_m8192_s1` | prefill | q [8192×8×128] bfloat16 | `ut/meta.json#/cases/2`; JSON record 2 |

Benchmark IDs above follow the protected timing builder and common adapter. All capture-only and boundary records remain in the JSON inventory with their original IDs and explicit benchmark membership. The `tensor_overrides` entries distinguish timing storage from captured storage: MiniMax prefill remaps selected request rows and changes `slot_ids` to int32; recurrent decode generates contiguous timing tensors.

| Argument / output | Recorded geometry and layout |
| --- | --- |
| `cu_seqblocks_q` (input) | [2]; int32; strides unknown. unknown: physical strides are not recorded in this metadata. |
| `cu_seqlens` (input) | [2]; int32; strides unknown. unknown: physical strides are not recorded in this metadata. |
| `k_cache` (input) | [4358330, 1, 128]; bfloat16; strides unknown. unknown: physical strides are not recorded in this metadata. |
| `prefix_lens` (input) | [1]; int32; strides unknown. unknown: physical strides are not recorded in this metadata. |
| `q` (input) | [1, 8, 128], [186, 8, 128], [8192, 8, 128]; bfloat16; strides unknown. unknown: physical strides are not recorded in this metadata. |
| `req_to_token` (input) | [4097, 11268]; int32; strides unknown. unknown: physical strides are not recorded in this metadata. |
| `seq_lens` (input) | [1]; int32; strides unknown. unknown: physical strides are not recorded in this metadata. |
| `slot_ids` (input) | [1]; int64; strides unknown. unknown: physical strides are not recorded in this metadata. |
| `topk_idx` (input) | [1, 1, 16], [1, 186, 16], [1, 8192, 16]; int32; strides unknown. unknown: physical strides are not recorded in this metadata. |
| `v_cache` (input) | [4358330, 1, 128]; bfloat16; strides unknown. unknown: physical strides are not recorded in this metadata. |
| `return` (output) | [1, 8, 128], [186, 8, 128], [8192, 8, 128]; bfloat16; strides unknown. merged attention output. |

Each JSON tensor record cites the metadata field or protected builder that establishes its geometry. A `null` shape, dtype or stride means it is not established by readable metadata; it is not a wildcard or permission to replace the fixture. Packed weights, sparse paging maps and routing-dependent lengths must retain the protected artifact representation.

**Evidence limit:** 33 tensor records have at least one unknown physical field. The machine catalog enumerates each field and its source. Exact opaque-fixture details require the hash-pinned tensor artifacts; this static catalog does not claim that they were inspected.

| Protected tensor artifact | Expected SHA-256 |
| --- | --- |
| `ut/reference_io.pt` | `55f55d92b743c71e0362c4c4e0e47d7a0120c9c19bdc03fac7d1972ac80be6ba` |
| `ut/timing_geometry.pt` | `d1ad8a11fd31d42d04a92aa7d9854db6dce4ba5ba0aa43168185b731f4471462` |

Source pointers in the JSON are relative to this task directory. Tensor artifacts were not deserialized, and no task code or GPU benchmark was run to produce this catalog. Fresh validation remains governed by the task contract.
