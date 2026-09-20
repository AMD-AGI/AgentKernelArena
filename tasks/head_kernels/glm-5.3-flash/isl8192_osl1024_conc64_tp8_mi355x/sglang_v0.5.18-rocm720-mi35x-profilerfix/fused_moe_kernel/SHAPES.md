# Shape and benchmark catalog: glm-5.3-flash__fused_moe_kernel

This is a single-GPU MI355X (gfx950) callable benchmark. The serving capture used ISL 8192, OSL 1024, concurrency 64 and TP=8; those settings are context, not a substitute for the tensor shapes below.

The complete case IDs, argument inventories, original metadata, scalars, tensor attributes and source hashes are in [SHAPES.json](SHAPES.json). The task environment and benchmark commands are in [config.yaml](config.yaml).

The inventory has **4 shape records** and **3 benchmark cases**. The protected metadata's `num_cases` field is `3`; that field can count a different capture scope. Histogram observations and random value draws are not added to the benchmark count.

Callable: `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe:fused_experts_impl`. Baseline recorded in metadata: `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe:fused_experts_impl`. Composite operator: a complete call may launch multiple physical kernels.

grouped per-expert GEMM + routing (GEMM1 w1 -> swiglu -> GEMM2 w2 -> top-k weighted reduce, one dispatcher call)

Protected unit cases force inplace=False and recompute the golden. The original production capture used inplace=True and therefore captured a post-call hidden_states buffer; this scope difference is documented in meta.notes.

| Benchmark case ID | Regime | Tensor geometry | Shape source |
| --- | --- | --- | --- |
| `moe_m64_decode` | decode | hidden_states [64×4096] bfloat16 | `ut/meta.json#/workload/cases/0`; JSON record 3 |
| `moe_m8192_prefill` | prefill | hidden_states [8192×4096] bfloat16 | `ut/meta.json#/cases/2`; JSON record 2 |
| `moe_m1_decode` | decode | hidden_states [1×4096] bfloat16 | `ut/meta.json#/cases/1`; JSON record 1 |

Benchmark IDs above follow the protected timing builder and common adapter. All capture-only and boundary records remain in the JSON inventory with their original IDs and explicit benchmark membership. The `tensor_overrides` entries distinguish timing storage from captured storage: MiniMax prefill remaps selected request rows and changes `slot_ids` to int32; recurrent decode generates contiguous timing tensors.

| Argument / output | Recorded geometry and layout |
| --- | --- |
| `hidden_states` (input) | [1, 4096], [19, 4096], [64, 4096], [8192, 4096]; bfloat16; strides unknown. captured activation; generated timing inputs are contiguous. |
| `w1` (input) | [288, 512, 4096]; fp8_e4m3; strides unknown. weight shapes recorded in meta.notes; scales require tensor oracle. |
| `w2` (input) | [288, 4096, 256]; fp8_e4m3; strides unknown. weight shapes recorded in meta.notes; scales require tensor oracle. |
| `w1_scale` (input) | unknown; float32; strides unknown. weight shapes recorded in meta.notes; scales require tensor oracle. |
| `w2_scale` (input) | unknown; float32; strides unknown. weight shapes recorded in meta.notes; scales require tensor oracle. |
| `topk_ids` (input) | [1, 8], [19, 8], [64, 8], [8192, 8]; None; strides unknown. real captured routing; timing bootstraps rows with replacement. |
| `topk_weights` (input) | [1, 8], [19, 8], [64, 8], [8192, 8]; None; strides unknown. real captured routing; timing bootstraps rows with replacement. |
| `return` (output) | [1, 4096], [19, 4096], [64, 4096], [8192, 4096]; bfloat16; strides unknown. fresh full dispatcher output. |

Each JSON tensor record cites the metadata field or protected builder that establishes its geometry. A `null` shape, dtype or stride means it is not established by readable metadata; it is not a wildcard or permission to replace the fixture. Packed weights, sparse paging maps and routing-dependent lengths must retain the protected artifact representation.

**Evidence limit:** 32 tensor records have at least one unknown physical field. The machine catalog enumerates each field and its source. Exact opaque-fixture details require the hash-pinned tensor artifacts; this static catalog does not claim that they were inspected.

| Protected tensor artifact | Expected SHA-256 |
| --- | --- |
| `ut/reference_io.pt` | `59040b163efbc9dbd3592a28dee01bcc2801bb669de170197524dfe335c5feee` |

Source pointers in the JSON are relative to this task directory. Tensor artifacts were not deserialized, and no task code or GPU benchmark was run to produce this catalog. Fresh validation remains governed by the task contract.
