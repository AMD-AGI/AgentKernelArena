# Shape and benchmark catalog: fused_moe_2stage_mxfp4

This is a single-GPU MI355X (gfx950) callable benchmark. The serving capture used ISL 8192, OSL 1024, concurrency 64 and TP=8; those settings are context, not a substitute for the tensor shapes below.

The complete case IDs, argument inventories, original metadata, scalars, tensor attributes, extracted fixture layouts and source hashes are in [SHAPES.json](SHAPES.json). The task environment and benchmark commands are in [config.yaml](config.yaml).

The inventory has **2 shape records** and **2 benchmark cases**. The protected metadata's `num_cases` field is `2`; that field can count a different capture scope. Histogram observations and random value draws are not added to the benchmark count.

Callable: `aiter.fused_moe:fused_moe`. Baseline recorded in metadata: `same production seam, selected through frozen baseline binding`. Composite operator: a complete call may launch multiple physical kernels.

Full fused MoE dispatch: real top-k routing over compact MXFP4 expert weights, SiLU/mul stage 1, expert projection stage 2, and weighted reduction to a fresh BF16 output.

Returns a fresh BF16 result; input hidden_states, routing, packed weights, scales and expert mask remain unchanged. One call includes both MoE stages.

| Benchmark case ID | Regime | Tensor geometry | Shape source |
| --- | --- | --- | --- |
| `decode_m64_live` | decode | hidden_states [64×8192] bfloat16 | `ut/meta.json#/workload/cases/0`; JSON record 0 |
| `prefill_m8192_live` | prefill | hidden_states [8192×8192] bfloat16 | `ut/meta.json#/workload/cases/1`; JSON record 1 |

Benchmark IDs above follow the protected timing builder and common adapter. All capture-only and boundary records remain in the JSON inventory with their original IDs and explicit benchmark membership. The `tensor_overrides` entries distinguish timing storage from captured storage: MiniMax prefill remaps selected request rows and changes `slot_ids` to int32; recurrent decode generates contiguous timing tensors.

| Argument / output | Recorded geometry and layout |
| --- | --- |
| `expert_mask` (input) | [512]; int32; strides [1]. preserved captured descriptor layout. |
| `hidden_states` (input) | [64, 8192], [8192, 8192]; bfloat16; strides [8192, 1]. preserved captured descriptor layout. |
| `topk_ids` (input) | [64, 10], [8192, 10]; int32; strides [10, 1]. preserved captured descriptor layout. |
| `topk_weight` (input) | [64, 10], [8192, 10]; float32; strides [10, 1]. preserved captured descriptor layout. |
| `w1` (input) | [64, 4096, 4096]; float4_e2m1fn_x2; strides [16777216, 4096, 1]. preserved captured descriptor layout. |
| `w1_scale` (input) | [64, 4096, 256]; uint8; strides [1048576, 256, 1]. preserved captured descriptor layout. |
| `w2` (input) | [64, 8192, 1024]; float4_e2m1fn_x2; strides [8388608, 1024, 1]. preserved captured descriptor layout. |
| `w2_scale` (input) | [64, 8192, 64]; uint8; strides [524288, 64, 1]. preserved captured descriptor layout. |
| `return` (output) | [64, 8192], [8192, 8192]; bfloat16; strides [8192, 1]. preserved captured descriptor layout. |

Each JSON tensor record cites the metadata field or protected builder that establishes its geometry. A `null` shape, dtype or stride means it is not established by readable metadata; it is not a wildcard or permission to replace the fixture. Packed weights, sparse paging maps and routing-dependent lengths must retain the protected artifact representation.

Persistent live capture with real packed MXFP4 weights, uint8 scales, expert mask, top-k ids/weights, and full BF16 outputs. Random parity changes hidden values only and preserves live routing and expert state.

| Protected tensor artifact | Expected SHA-256 |
| --- | --- |
| `ut/reference_io.pt` | `09a71daa55bdd3683be45016f0e5139e2fd63b7c614ae5ba565f74cbd2c7e434` |

Source pointers in the JSON are relative to this task directory. Every declared artifact was SHA-256 verified and its tensor metadata extracted using PyTorch 2.9.1+cpu, `weights_only=True`, CPU mapping, mmap and FakeTensorMode. Tensor values were not read, and no GPU work was performed. The JSON preserves the parser, fixture hashes, descriptor layouts and serialized alias identities. Fresh GPU validation remains governed by the task contract.
