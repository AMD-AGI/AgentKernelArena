# Shape and benchmark catalog: glm-5.3-flash__elementwise_copy_cluster

This is a single-GPU MI355X (gfx950) callable benchmark. The serving capture used ISL 8192, OSL 1024, concurrency 64 and TP=8; those settings are context, not a substitute for the tensor shapes below.

The complete case IDs, argument inventories, original metadata, scalars, tensor attributes and source hashes are in [SHAPES.json](SHAPES.json). The task environment and benchmark commands are in [config.yaml](config.yaml).

The inventory has **8 shape records** and **8 benchmark cases**. The protected metadata's `num_cases` field is `7`; that field can count a different capture scope. Histogram observations and random value draws are not added to the benchmark count.

Callable: `sglang.srt.layers.quantization.fp8_utils:materialize_bpreshuffle_fp8_scale`. Baseline recorded in metadata: `same production seam, selected through frozen baseline binding`. Selected callable seam; kernel launch count is not asserted by this catalog.

out = scale for a [M,G] fp32 per-1x128 activation scale, but materialized with TRANSPOSED-CONTIGUOUS storage: same logical [M,G] shape and values, stride == (1, M). Non-2-D input passes through unchanged.

2-D scale keeps logical values/shape while changing physical layout to transpose-contiguous. Non-2-D input passes through unchanged. An already compatible degenerate layout may alias.

| Benchmark case ID | Regime | Tensor geometry | Shape source |
| --- | --- | --- | --- |
| `scale_m64x32_decode` | decode | scale [64×32] float32 | `ut/meta.json#/cases/3`; JSON record 3 |
| `scale_m64x2_decode` | decode | scale [64×2] float32 | `ut/meta.json#/cases/0`; JSON record 0 |
| `scale_m64x12_decode` | decode | scale [64×12] float32 | `ut/meta.json#/cases/1`; JSON record 1 |
| `scale_m16384x2_prefill` | prefill | scale [16384×2] float32 | `ut/meta.json#/cases/4`; JSON record 4 |
| `scale_m16384x12_prefill` | prefill | scale [16384×12] float32 | `ut/meta.json#/cases/5`; JSON record 5 |
| `scale_m16384x16_prefill` | prefill | scale [16384×16] float32 | `ut/meta.json#/cases/6`; JSON record 6 |
| `scale_m16384x32_prefill` | prefill | scale [16384×32] float32 | `ut/meta.json#/cases/7`; JSON record 7 |
| `scale_m64x16_decode` | decode | scale [64×16] float32 | `ut/meta.json#/cases/2`; JSON record 2 |

Benchmark IDs above follow the protected timing builder and common adapter. All capture-only and boundary records remain in the JSON inventory with their original IDs and explicit benchmark membership. The `tensor_overrides` entries distinguish timing storage from captured storage: MiniMax prefill remaps selected request rows and changes `slot_ids` to int32; recurrent decode generates contiguous timing tensors.

| Argument / output | Recorded geometry and layout |
| --- | --- |
| `scale` (input) | 8 exact variants in JSON; float32; strides per-case values in JSON. row-major generated timing input; captured oracle input stride is separate. |
| `return` (output) | 8 exact variants in JSON; float32; strides [1, 16384], [1, 64]. transpose-contiguous; logical values unchanged. |

Each JSON tensor record cites the metadata field or protected builder that establishes its geometry. A `null` shape, dtype or stride means it is not established by readable metadata; it is not a wildcard or permission to replace the fixture. Packed weights, sparse paging maps and routing-dependent lengths must retain the protected artifact representation.

| Protected tensor artifact | Expected SHA-256 |
| --- | --- |
| `ut/reference_io.pt` | `14e589b7296c667b170b39a5f962344152206ff954a84b4e16de4346567d12af` |

Source pointers in the JSON are relative to this task directory. Tensor artifacts were not deserialized, and no task code or GPU benchmark was run to produce this catalog. Fresh validation remains governed by the task contract.
