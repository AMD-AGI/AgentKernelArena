# Shape and benchmark catalog: kimi-k3__fwd_grouped_kernel_stage1

This is a single-GPU MI355X (gfx950) callable benchmark. The serving capture used ISL 8192, OSL 1024, concurrency 64 and TP=8; those settings are context, not a substitute for the tensor shapes below.

The complete case IDs, argument inventories, original metadata, scalars, tensor attributes and source hashes are in [SHAPES.json](SHAPES.json). The task environment and benchmark commands are in [config.yaml](config.yaml).

The inventory has **4 shape records** and **2 benchmark cases**. The protected metadata's `num_cases` field is `not declared`; that field can count a different capture scope. Histogram observations and random value draws are not added to the benchmark count.

Callable: `sglang.kernels.ops.attention.decode_attention:_decode_grouped_att_m_fwd`. Baseline recorded in metadata: `sglang.kernels.ops.attention.decode_attention:_decode_grouped_att_m_fwd`. Selected callable seam; kernel launch count is not asserted by this catalog.

MLA absorbed decode, split-KV stage 1: for each (batch b, q-head h, kv split s) over the paged latent KV named by kv_indices[kv_indptr[b]:kv_indptr[b+1]] (page_size=1), qk = Q[b,h,:576] . K[idx,0,:576] * sm_scale ; att_out[b,h,s,:512] = softmax_partial(qk) @ K[idx,0,:512]  (HAS_MLA: V is the leading 512 lanes of the same latent row, v = trans(k)) ; att_lse[b,h,s] = max(qk) + log(sum exp(qk-max)). Slots s >= num_kv_splits[b] are not written by the kernel and are compared as zero on both sides.

Native callable writes att_out and att_lse. The harness exposes these output buffers as a tuple; inactive split slots stay at their initial zero. Timing v_buffer is a view of k_buffer.

| Benchmark case ID | Regime | Tensor geometry | Shape source |
| --- | --- | --- | --- |
| `decode_bs1_ctx8704` | decode | q [1×12×576] bfloat16; k_buffer [8768×1×576] bfloat16 | `ut/meta.json#/cases/2`; JSON record 2 |
| `decode_bs64_ctx8704` | decode | q [64×12×576] bfloat16; k_buffer [557120×1×576] bfloat16 | `ut/meta.json#/cases/3`; JSON record 3 |

Benchmark IDs above follow the protected timing builder and common adapter. All capture-only and boundary records remain in the JSON inventory with their original IDs and explicit benchmark membership. The `tensor_overrides` entries distinguish timing storage from captured storage: MiniMax prefill remaps selected request rows and changes `slot_ids` to int32; recurrent decode generates contiguous timing tensors.

| Argument / output | Recorded geometry and layout |
| --- | --- |
| `q` (input) | [1, 12, 576], [64, 12, 576]; bfloat16; strides [6912, 576, 1], unknown. contiguous generated tensor; unknown: physical strides are not recorded in this metadata. |
| `k_buffer` (input) | [395, 1, 576], [524352, 1, 576], [557120, 1, 576], [8768, 1, 576]; bfloat16; strides [576, 576, 1], unknown. contiguous generated tensor; unknown: physical strides are not recorded in this metadata. |
| `v_buffer` (input) | [395, 1, 512], [524352, 1, 512], [557120, 1, 512], [8768, 1, 512]; bfloat16; strides [576, 576, 1], unknown. view of k_buffer[..., :512]. |
| `kv_indptr` (input) | [2], [65]; int32; strides [1], unknown. contiguous generated tensor; unknown: physical strides are not recorded in this metadata. |
| `kv_indices` (input) | [395], [524352], [557056], [8704]; int64; strides [1], unknown. contiguous generated tensor; unknown: physical strides are not recorded in this metadata. |
| `num_kv_splits` (input) | [1], [64]; int32; strides [1], unknown. contiguous generated tensor; unknown: physical strides are not recorded in this metadata. |
| `att_out` (output/output_buffer) | [1, 12, 256, 512], [64, 12, 256, 512]; float32; strides [1572864, 131072, 512, 1], unknown. caller-owned output; only live splits written; unknown: physical strides are not recorded in this metadata. |
| `att_lse` (output/output_buffer) | [1, 12, 256], [64, 12, 256]; float32; strides [3072, 256, 1], unknown. caller-owned output; only live splits written; unknown: physical strides are not recorded in this metadata. |

Each JSON tensor record cites the metadata field or protected builder that establishes its geometry. A `null` shape, dtype or stride means it is not established by readable metadata; it is not a wildcard or permission to replace the fixture. Packed weights, sparse paging maps and routing-dependent lengths must retain the protected artifact representation.

**Evidence limit:** 24 tensor records have at least one unknown physical field. The machine catalog enumerates each field and its source. Exact opaque-fixture details require the hash-pinned tensor artifacts; this static catalog does not claim that they were inspected.

| Protected tensor artifact | Expected SHA-256 |
| --- | --- |
| `ut/reference_io.pt` | `017f8ede884504695283291955fa8750be3e5d2779dd67b3914f13ab7ae85b98` |

Source pointers in the JSON are relative to this task directory. Tensor artifacts were not deserialized, and no task code or GPU benchmark was run to produce this catalog. Fresh validation remains governed by the task contract.
