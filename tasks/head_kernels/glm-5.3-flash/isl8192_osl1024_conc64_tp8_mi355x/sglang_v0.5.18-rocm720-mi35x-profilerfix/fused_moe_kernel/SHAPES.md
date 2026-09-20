# Shape and benchmark catalog: fused_moe_kernel

This is a single-GPU MI355X (gfx950) callable benchmark. The serving capture used ISL 8192, OSL 1024, concurrency 64 and TP=8; those settings are context, not a substitute for the tensor shapes below.

The complete case IDs, argument inventories, original metadata, scalars, tensor attributes, extracted fixture layouts and source hashes are in [SHAPES.json](SHAPES.json). The task environment and benchmark commands are in [config.yaml](config.yaml).

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
| `hidden_states` (input) | [1, 4096], [19, 4096], [64, 4096], [8192, 4096]; bfloat16; strides [4096, 1]. contiguous timing tensor generated with captured activation/routing dtype; serialized oracle tensor layout; reconstructed as prescribed by the protected builder. |
| `w1` (input) | [288, 512, 4096]; float8_e4m3fn; strides [2097152, 4096, 1]. serialized oracle tensor layout; reconstructed as prescribed by the protected builder. |
| `w2` (input) | [288, 4096, 256]; float8_e4m3fn; strides [1048576, 256, 1]. serialized oracle tensor layout; reconstructed as prescribed by the protected builder. |
| `w1_scale` (input) | [288, 4, 32]; float32; strides [128, 32, 1]. serialized oracle tensor layout; reconstructed as prescribed by the protected builder. |
| `w2_scale` (input) | [288, 32, 2]; float32; strides [64, 2, 1]. serialized oracle tensor layout; reconstructed as prescribed by the protected builder. |
| `topk_ids` (input) | [1, 8], [19, 8], [64, 8], [8192, 8]; int32; strides [8, 1]. contiguous timing tensor generated with captured activation/routing dtype; serialized oracle tensor layout; reconstructed as prescribed by the protected builder. |
| `topk_weights` (input) | [1, 8], [19, 8], [64, 8], [8192, 8]; float32; strides [8, 1]. contiguous timing tensor generated with captured activation/routing dtype; serialized oracle tensor layout; reconstructed as prescribed by the protected builder. |
| `return` (output) | [1, 4096], [19, 4096], [64, 4096], [8192, 4096]; bfloat16; strides [4096, 1]. fresh contiguous dispatcher output; serialized oracle tensor layout; reconstructed as prescribed by the protected builder. |

Each JSON tensor record cites the metadata field or protected builder that establishes its geometry. A `null` shape, dtype or stride means it is not established by readable metadata; it is not a wildcard or permission to replace the fixture. Packed weights, sparse paging maps and routing-dependent lengths must retain the protected artifact representation.

| Protected tensor artifact | Expected SHA-256 |
| --- | --- |
| `ut/reference_io.pt` | `59040b163efbc9dbd3592a28dee01bcc2801bb669de170197524dfe335c5feee` |

Source pointers in the JSON are relative to this task directory. Every declared artifact was SHA-256 verified and its tensor metadata extracted using PyTorch 2.9.1+cpu, `weights_only=True`, CPU mapping, mmap and FakeTensorMode. Tensor values were not read, and no GPU work was performed. The JSON preserves the parser, fixture hashes, descriptor layouts and serialized alias identities. Fresh GPU validation remains governed by the task contract.
