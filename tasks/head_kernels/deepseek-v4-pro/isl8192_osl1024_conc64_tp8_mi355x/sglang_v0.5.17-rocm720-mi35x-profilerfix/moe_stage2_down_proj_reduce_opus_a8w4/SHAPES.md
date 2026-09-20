# Shape and benchmark catalog: moe_stage2_down_proj_reduce_opus_a8w4

**Qualification is blocked: inputs cover only 183/256 mandatory sequence calls.**
The unchanged 256-entry ledger requires 73 M=1920 calls at zero-based positions
183–255, but the original and mirrored archives contain only five records:
M2048, M256, M64, M1 and M32768. The original UT filtered unresolved signatures,
so its sequence check covered only the first 183 M2048 calls. That partial check
cannot establish full sequence correctness.

Correctness and performance now fail during CPU contract verification, before
runtime preflight or GPU workers. The failure report and
`build/sequence_coverage_report.json` identify every unresolved signature and
position. All 256 calls remain mandatory; no subset can produce a qualification
PASS. Authentic M1920 routing and tensor-layout inputs must be recovered before
promotion. The five benchmark cases, three random draws, M256-capture/M64-replay
boundary, tolerance, 10 warmups and 100 timing samples remain unchanged.

[ut/sequence_coverage_evidence.json](ut/sequence_coverage_evidence.json) records
the exact missing signature and positions, ledger hash, original UT hash,
archive hashes and recovery receipt hashes. The recovery was limited to the
scoped original and mirrored archives; it does not rule out unrelated captures
elsewhere. A shape descriptor alone is not a usable input record.

This is a single-GPU MI355X (gfx950) callable benchmark. The serving capture used ISL 8192, OSL 1024, concurrency 64 and TP=8; those settings are context, not a substitute for the tensor shapes below.

The complete case IDs, argument inventories, original metadata, scalars, tensor attributes, extracted fixture layouts and source hashes are in [SHAPES.json](SHAPES.json). The task environment and benchmark commands are in [config.yaml](config.yaml).

The inventory has **5 shape records** and **5 benchmark cases**. The protected metadata's `num_cases` field is `5`; that field can count a different capture scope. Histogram observations and random value draws are not added to the benchmark count.

Callable: `aiter.ops.opus.moe_stage2_a8w4_fused_adapter:opus_a8w4_stage2_wrapper`. Baseline recorded in metadata: `aiter.ops.opus.moe_stage2_a8w4_fused_adapter:opus_a8w4_stage2_wrapper`. Composite operator: a complete call may launch multiple physical kernels.

stage-2 of the a8w4 fused MoE, as ONE segment: out[t,:] = sum_k topk_w[t,k] * (inter_states[t,k,:] @ w2[e(t,k)]^T) with mxfp8 activations x mxfp4 expert weights and e8m0 block scales, scattered through the moe_sorting block map. decode (route_out=False): the down-GEMM atomically accumulates the topk partials directly into a ZEROED out. prefill (route_out=True): the down-GEMM emits per-token-slot partials, then the reduce kernel sums the topk slots into out. Returns the same bf16 [token, model_dim] `out` tensor it was handed (written in place).

Returns the same supplied out buffer. Decode atomically accumulates into zeroed out; prefill emits per-slot values and reduces into out.

| Benchmark case ID | Regime | Tensor geometry | Shape source |
| --- | --- | --- | --- |
| `capture signature (record 0)` | prefill | inter_states [2048×6×384] float8_e4m3fn | `ut/meta.json#/cases/0`; JSON record 0 |
| `capture signature (record 1)` | decode | inter_states [256×6×384] float8_e4m3fn | `ut/meta.json#/cases/1`; JSON record 1 |
| `capture signature (record 2)` | decode | inter_states [64×6×384] float8_e4m3fn | `ut/meta.json#/cases/2`; JSON record 2 |
| `capture signature (record 3)` | decode | inter_states [1×6×384] float8_e4m3fn | `ut/meta.json#/cases/3`; JSON record 3 |
| `capture signature (record 4)` | prefill | inter_states [32768×6×384] float8_e4m3fn | `ut/meta.json#/cases/4`; JSON record 4 |

Benchmark IDs above follow the protected timing builder and common adapter. All capture-only and boundary records remain in the JSON inventory with their original IDs and explicit benchmark membership. The `tensor_overrides` entries distinguish timing storage from captured storage: MiniMax prefill remaps selected request rows and changes `slot_ids` to int32; recurrent decode generates contiguous timing tensors.

| Argument / output | Recorded geometry and layout |
| --- | --- |
| `inter_states` (input) | 5 exact variants in JSON; float8_e4m3fn; strides [2304, 384, 1]. preserved captured descriptor layout. |
| `w1` (input) | [384, 768, 3584]; float4_e2m1fn_x2; strides [2752512, 3584, 1]. preserved captured descriptor layout. |
| `w2` (input) | [384, 7168, 192]; float4_e2m1fn_x2; strides [1376256, 192, 1]. preserved captured descriptor layout. |
| `sorted_token_ids` (input) | 5 exact variants in JSON; int32; strides [1]. preserved captured descriptor layout. |
| `sorted_expert_ids` (input) | 5 exact variants in JSON; int32; strides [1]. preserved captured descriptor layout. |
| `num_valid_ids` (input) | [2]; int32; strides [1]. preserved captured descriptor layout. |
| `out` (input_output) | 5 exact variants in JSON; bfloat16; strides [7168, 1]. preserved captured descriptor layout. |
| `a2_scale` (input) | 5 exact variants in JSON; float8_e8m0fnu; strides [16, 1]. preserved captured descriptor layout. |
| `sorted_weights` (input) | 5 exact variants in JSON; float32; strides [1]. preserved captured descriptor layout. |
| `w2_scale` (input) | [2752512, 16]; float8_e8m0fnu; strides [16, 1]. preserved captured descriptor layout. |
| `return` (output) | 5 exact variants in JSON; bfloat16; strides [7168, 1]. preserved captured descriptor layout. |

Each JSON tensor record cites the metadata field or protected builder that establishes its geometry. A `null` shape, dtype or stride means it is not established by readable metadata; it is not a wildcard or permission to replace the fixture. The generated-input draft retains sparse paging maps, routing-dependent lengths, dtype/stride/alias contracts and packed formats in `ut/generated_cases.json`; large numerical buffers are generated locally.

| Historical tensor artifact (not required by the generated draft) | Original SHA-256 |
| --- | --- |
| `ut/reference_io.pt` | `3232c6110462fa4a7cb9d057f9c9f36ebda27e7a19cbf379194198e2cb509337` |

Source pointers in the JSON are relative to this task directory. Every declared artifact was SHA-256 verified and its tensor metadata extracted using PyTorch 2.9.1+cpu, `weights_only=True`, CPU mapping, mmap and FakeTensorMode. Tensor values were not read, and no GPU work was performed. The JSON preserves the parser, fixture hashes, descriptor layouts and serialized alias identities. Fresh GPU validation remains governed by the task contract.

The current draft requires no external tensor archive. `ut/generated_cases.json` is the SHA-256-checked compact structural contract; `ut/generated_contract.py` supplies finite numerical values. The original archive hash remains provenance under `ut/meta.json:archival_capture`. Fresh GPU qualification is still required.
