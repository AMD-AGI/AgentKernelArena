# Shape and diagnostic catalog: fwd_grouped_kernel_stage1

This single-GPU MI355X (gfx950) catalog preserves callable shapes and unscored diagnostics. Historical serving sources share ISL 8192, OSL 1024, concurrency 64 and TP=8; these values do not establish one common capture scenario or replace the tensor shapes below.

The complete case IDs, argument inventories, original metadata, scalars, tensor attributes, extracted fixture layouts and source hashes are in [SHAPES.json](SHAPES.json). The task environment and benchmark commands are in [config.yaml](config.yaml).

The inventory has **4 shape records**, **zero enabled/scored benchmark cases**, and **2 unscored diagnostic timing cases**. All former timing IDs and all capture/boundary shapes are retained. Historical `workload.num_cases` counts remain unchanged and do not imply enabled scoring.

The attention regime does not declare a prefill chunk. Its historical analytic weight model uses 16384 but has zero prefill calls; this is not an observed attention prefill scenario. ISL 8192 denotes input sequence length. See `ut/workload.json` (`regime` and `serving_weight_model`) for the unchanged historical values; the catalog preserves those source records verbatim.

Callable: `sglang.kernels.ops.attention.decode_attention:_decode_grouped_att_m_fwd`. Baseline recorded in metadata: `sglang.kernels.ops.attention.decode_attention:_decode_grouped_att_m_fwd`. Selected callable seam; kernel launch count is not asserted by this catalog.

MLA absorbed decode, split-KV stage 1: for each (batch b, q-head h, kv split s) over the paged latent KV named by kv_indices[kv_indptr[b]:kv_indptr[b+1]] (page_size=1), qk = Q[b,h,:576] . K[idx,0,:576] * sm_scale ; att_out[b,h,s,:512] = softmax_partial(qk) @ K[idx,0,:512]  (HAS_MLA: V is the leading 512 lanes of the same latent row, v = trans(k)) ; att_lse[b,h,s] = max(qk) + log(sum exp(qk-max)). Slots s >= num_kv_splits[b] are not written by the kernel and are compared as zero on both sides.

Native callable writes att_out and att_lse. The harness exposes these output buffers as a tuple; inactive split slots stay at their initial zero. Timing v_buffer is a view of k_buffer.

| Unscored diagnostic case ID | Regime | Tensor geometry | Shape source |
| --- | --- | --- | --- |
| `decode_bs1_ctx8704` | decode | q [1×12×576] bfloat16; k_buffer [8768×1×576] bfloat16 | `ut/meta.json#/cases/2`; JSON record 2 |
| `decode_bs64_ctx8704` | decode | q [64×12×576] bfloat16; k_buffer [557120×1×576] bfloat16 | `ut/meta.json#/cases/3`; JSON record 3 |

Diagnostic IDs above preserve the former timing builder membership. `benchmark_cases` and every shape record's `benchmark_case_ids` are empty; `unscored_diagnostic_cases` and `diagnostic_case_ids` retain those links. The authoritative [scoring policy](ut/meta.json) has `workload_scoring.enabled=false`: proxy context/pool; exact original control mapping unavailable; zero-weight or boundary-only semantic case.

| Argument / output | Recorded geometry and layout |
| --- | --- |
| `q` (input) | [1, 12, 576], [64, 12, 576]; bfloat16; strides [6912, 576, 1]. contiguous generated tensor; preserved captured descriptor layout. |
| `k_buffer` (input) | [395, 1, 576], [524352, 1, 576], [557120, 1, 576], [8768, 1, 576]; bfloat16; strides [576, 576, 1]. contiguous generated tensor; serialized oracle tensor layout; reconstructed as prescribed by the protected builder. |
| `v_buffer` (input) | [395, 1, 512], [524352, 1, 512], [557120, 1, 512], [8768, 1, 512]; bfloat16; strides [576, 576, 1]. view k_buffer[:, :, :512] prescribed by _hydrate; view of k_buffer[..., :512]. |
| `kv_indptr` (input) | [2], [65]; int32; strides [1]. contiguous generated tensor; preserved captured descriptor layout. |
| `kv_indices` (input) | [395], [524352], [557056], [8704]; int64; strides [1]. contiguous generated tensor; serialized oracle tensor layout; reconstructed as prescribed by the protected builder. |
| `num_kv_splits` (input) | [1], [64]; int32; strides [1]. contiguous generated tensor; preserved captured descriptor layout. |
| `att_out` (output/output_buffer) | [1, 12, 256, 512], [64, 12, 256, 512]; float32; strides [1572864, 131072, 512, 1]. caller-owned output; only live splits written; fresh contiguous zero-filled caller buffer allocated by _make_call; same physical layout as supplied att_out; serialized oracle tensor layout; reconstructed as prescribed by the protected builder. |
| `att_lse` (output/output_buffer) | [1, 12, 256], [64, 12, 256]; float32; strides [3072, 256, 1]. caller-owned output; only live splits written; fresh contiguous zero-filled caller buffer allocated by _make_call; same physical layout as supplied att_lse; serialized oracle tensor layout; reconstructed as prescribed by the protected builder. |

Each JSON tensor record cites the metadata field or protected builder that establishes its geometry. A `null` shape, dtype or stride means it is not established by readable metadata; it is not a wildcard or permission to replace the fixture. Packed weights, sparse paging maps and routing-dependent lengths must retain the protected artifact representation.

| Optional archival tensor evidence | Expected SHA-256 |
| --- | --- |
| `ut/reference_io.pt` | `017f8ede884504695283291955fa8750be3e5d2779dd67b3914f13ab7ae85b98` |

Generated inputs use `ut/generated_cases.json`; the original tensor archive is optional historical evidence. The extraction description below records the earlier archive inspection, not a runtime dependency.

Source pointers in the JSON are relative to this task directory. Every declared artifact was SHA-256 verified and its tensor metadata extracted using PyTorch 2.9.1+cpu, `weights_only=True`, CPU mapping, mmap and FakeTensorMode. Tensor values were not read, and no GPU work was performed. The JSON preserves the parser, fixture hashes, descriptor layouts and serialized alias identities. Fresh GPU validation remains governed by the task contract.
