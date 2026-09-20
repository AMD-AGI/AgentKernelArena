# Shape and benchmark catalog: qwen3.8-2.4t__gemma_fused_add_rmsnorm

This is a single-GPU MI355X (gfx950) callable benchmark. The serving capture used ISL 8192, OSL 1024, concurrency 64 and TP=8; those settings are context, not a substitute for the tensor shapes below.

The complete case IDs, argument inventories, original metadata, scalars, tensor attributes and source hashes are in [SHAPES.json](SHAPES.json). The task environment and benchmark commands are in [config.yaml](config.yaml).

The inventory has **2 shape records** and **2 benchmark cases**. The protected metadata's `num_cases` field is `2`; that field can count a different capture scope. Histogram observations and random value draws are not added to the benchmark count.

Callable: `sglang.srt.layers.layernorm:rocm_triton_gemma_fused_add_rmsnorm`. Baseline recorded in metadata: `same production seam, selected through frozen baseline binding`. Selected callable seam; kernel launch count is not asserted by this catalog.

pre_norm_sum = bf16(x + residual); normed = bf16(float(pre_norm_sum) * rsqrt(mean(float(pre_norm_sum)^2) + eps) * (1 + float(weight))); return (normed, pre_norm_sum) as two fresh non-aliasing tensors without mutating x, residual, or weight.

Returns two fresh outputs that alias neither each other nor any input. Output shape/dtype/stride equal x. Inputs x, residual and weight remain unchanged.

| Benchmark case ID | Regime | Tensor geometry | Shape source |
| --- | --- | --- | --- |
| `prefill_m8192_live_shape` | prefill | x [8192×8192] bfloat16; residual [8192×8192] bfloat16; weight [8192] bfloat16 | `ut/meta.json#/workload/cases/0`; JSON record 0 |
| `decode_m64_live_shape` | decode | x [64×8192] bfloat16; residual [64×8192] bfloat16; weight [8192] bfloat16 | `ut/meta.json#/workload/cases/1`; JSON record 1 |

Benchmark IDs above follow the protected timing builder and common adapter. All capture-only and boundary records remain in the JSON inventory with their original IDs and explicit benchmark membership. The `tensor_overrides` entries distinguish timing storage from captured storage: MiniMax prefill remaps selected request rows and changes `slot_ids` to int32; recurrent decode generates contiguous timing tensors.

| Argument / output | Recorded geometry and layout |
| --- | --- |
| `x` (input) | [64, 8192], [8192, 8192]; bfloat16; strides [8192, 1]. explicit physical stride. |
| `residual` (input) | [64, 8192], [8192, 8192]; bfloat16; strides [8192, 1]. explicit physical stride. |
| `weight` (input) | [8192]; bfloat16; strides [1]. explicit physical stride. |
| `normed` (output) | [64, 8192], [8192, 8192]; bfloat16; strides [8192, 1]. fresh; output strides equal x strides. |
| `pre_norm_sum` (output) | [64, 8192], [8192, 8192]; bfloat16; strides [8192, 1]. fresh; output strides equal x strides. |

Each JSON tensor record cites the metadata field or protected builder that establishes its geometry. A `null` shape, dtype or stride means it is not established by readable metadata; it is not a wildcard or permission to replace the fixture. Packed weights, sparse paging maps and routing-dependent lengths must retain the protected artifact representation.

Deterministic nonzero inputs are generated at the two live shapes. Golden full-tuple outputs are produced at unittest runtime by the frozen baseline overlay in a separate process.

Source pointers in the JSON are relative to this task directory. Tensor artifacts were not deserialized, and no task code or GPU benchmark was run to produce this catalog. Fresh validation remains governed by the task contract.
