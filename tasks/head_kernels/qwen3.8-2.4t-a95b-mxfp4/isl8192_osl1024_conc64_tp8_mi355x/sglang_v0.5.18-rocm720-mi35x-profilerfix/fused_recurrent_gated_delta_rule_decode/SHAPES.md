# Shape and benchmark catalog: fused_recurrent_gated_delta_rule_decode

This is a single-GPU MI355X (gfx950) callable benchmark. The serving capture used ISL 8192, OSL 1024, concurrency 64 and TP=8; those settings are context, not a substitute for the tensor shapes below.

The complete case IDs, argument inventories, original metadata, scalars, tensor attributes, extracted fixture layouts and source hashes are in [SHAPES.json](SHAPES.json). The task environment and benchmark commands are in [config.yaml](config.yaml).

The inventory has **2 shape records** and **2 benchmark cases**. The protected metadata's `num_cases` field is `2`; that field can count a different capture scope. Histogram observations and random value draws are not added to the benchmark count.

Callable: `sglang.srt.layers.attention.linear.kernels.gdn_triton:fused_recurrent_gated_delta_rule_packed_decode`. Baseline recorded in metadata: `same production seam, selected through frozen baseline binding`. Selected callable seam; kernel launch count is not asserted by this catalog.

One packed decode gated-delta step writes out in place and updates initial_state at ssm_state_indices in place, returning the same two buffers as (out, initial_state).

Returns the same (out, initial_state) buffers. Both mutate in place; initial_state updates only ssm_state_indices. Other inputs are immutable.

| Benchmark case ID | Regime | Tensor geometry | Shape source |
| --- | --- | --- | --- |
| `decode_b64_live` | decode | a [64×16] bfloat16; mixed_qkv [64×2560] bfloat16 | `ut/meta.json#/cases/1`; JSON record 1 |
| `decode_b1_live` | decode | a [1×16] bfloat16; mixed_qkv [1×2560] bfloat16 | `ut/meta.json#/cases/0`; JSON record 0 |

Benchmark IDs above follow the protected timing builder and common adapter. All capture-only and boundary records remain in the JSON inventory with their original IDs and explicit benchmark membership. The `tensor_overrides` entries distinguish timing storage from captured storage: MiniMax prefill remaps selected request rows and changes `slot_ids` to int32; recurrent decode generates contiguous timing tensors.

| Argument / output | Recorded geometry and layout |
| --- | --- |
| `A_log` (input) | [16]; float32; strides [1]. preserved captured descriptor layout. |
| `a` (input) | [1, 16], [64, 16]; bfloat16; strides [16, 1]. preserved captured descriptor layout. |
| `b` (input) | [1, 16], [64, 16]; bfloat16; strides [16, 1]. preserved captured descriptor layout. |
| `dt_bias` (input) | [16]; bfloat16; strides [1]. preserved captured descriptor layout. |
| `initial_state` (input_output) | [199, 16, 128, 128]; float32; strides [262144, 16384, 128, 1]. preserved captured descriptor layout. |
| `mixed_qkv` (input) | [1, 2560], [64, 2560]; bfloat16; strides [2560, 1]. preserved captured descriptor layout. |
| `out` (input_output) | [1, 1, 16, 128], [64, 1, 16, 128]; bfloat16; strides [2048, 2048, 128, 1]. preserved captured descriptor layout. |
| `ssm_state_indices` (input) | [1], [64]; int32; strides [1]. preserved captured descriptor layout. |
| `return[0]` (output) | [1, 1, 16, 128], [64, 1, 16, 128]; bfloat16; strides [2048, 2048, 128, 1]. preserved captured descriptor layout. |
| `return[1]` (output) | [199, 16, 128, 128]; float32; strides [262144, 16384, 128, 1]. preserved captured descriptor layout. |

Each JSON tensor record cites the metadata field or protected builder that establishes its geometry. A `null` shape, dtype or stride means it is not established by readable metadata; it is not a wildcard or permission to replace the fixture. Packed weights, sparse paging maps and routing-dependent lengths must retain the protected artifact representation.

Live eager capture from the production callable is authoritative for values, routing, layout, output, and recurrent-state transition; deployment CUDA-graph behavior is checked separately by capture-once/replay-many in unittest.py.

| Protected tensor artifact | Expected SHA-256 |
| --- | --- |
| `ut/reference_io.pt` | `38a624a5f502279a13a09c012fababbe232cadedacd560056d8411640a04aaf1` |

Source pointers in the JSON are relative to this task directory. Every declared artifact was SHA-256 verified and its tensor metadata extracted using PyTorch 2.9.1+cpu, `weights_only=True`, CPU mapping, mmap and FakeTensorMode. Tensor values were not read, and no GPU work was performed. The JSON preserves the parser, fixture hashes, descriptor layouts and serialized alias identities. Fresh GPU validation remains governed by the task contract.
