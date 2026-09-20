# Shape and diagnostic catalog: moe_gemm2_stage2

This single-GPU MI355X (gfx950) catalog preserves callable shapes and unscored diagnostics. Historical serving sources share ISL 8192, OSL 1024, concurrency 64 and TP=8; these values do not establish one common capture scenario or replace the tensor shapes below.

The complete case IDs, argument inventories, original metadata, scalars, tensor attributes, extracted fixture layouts and source hashes are in [SHAPES.json](SHAPES.json). The task environment and benchmark commands are in [config.yaml](config.yaml).

The inventory has **4 shape records**, **zero enabled/scored benchmark cases**, and **4 unscored diagnostic timing cases**. All former timing IDs and all capture/boundary shapes are retained. Historical `workload.num_cases` counts remain unchanged and do not imply enabled scoring.

Stage-2 ut/workload.json records chunked_prefill_size=16384 from the original live server log. Both M=16384 and M=8192 shapes are retained. This is a distinct historical scenario from the 0828 stage-1 chunk of 8192; a case M value alone does not identify the configured chunk. ISL 8192 denotes input sequence length. See `ut/workload.json` (`regime` and `serving_weight_model`) for the unchanged historical values; the catalog preserves those source records verbatim.

Callable: `aiter.ops.flydsl.moe_kernels:flydsl_moe_stage2`. Baseline recorded in metadata: `baseline_src/flydsl/moe_kernels.py:flydsl_moe_stage2`. Composite operator: a complete call may launch multiple physical kernels.

grouped per-expert GEMM + routing

Caller supplies out[M,3584], which is written and returned. Atomic mode requires zero initialization; reduce mode writes per-slot intermediates then reduces.

| Unscored diagnostic case ID | Regime | Tensor geometry | Shape source |
| --- | --- | --- | --- |
| `prefill_M16384` | prefill | inter_states [16384×16×384] float8_e4m3fn | `ut/meta.json#/case_specs/0`; JSON record 0 |
| `prefill_M8192` | prefill | inter_states [8192×16×384] float8_e4m3fn | `ut/meta.json#/case_specs/1`; JSON record 1 |
| `decode_M1` | decode | inter_states [1×16×384] float8_e4m3fn | `ut/meta.json#/case_specs/2`; JSON record 2 |
| `decode_M64` | decode | inter_states [64×16×384] float8_e4m3fn | `ut/meta.json#/case_specs/3`; JSON record 3 |

Diagnostic IDs above preserve the former timing builder membership. `benchmark_cases` and every shape record's `benchmark_case_ids` are empty; `unscored_diagnostic_cases` and `diagnostic_case_ids` retain those links. The authoritative [scoring policy](ut/meta.json) has `workload_scoring.enabled=false`: workload control weight is not observed trace evidence. Individual stage-2 prefill eligibility does not override the disabled aggregate policy.

| Argument / output | Recorded geometry and layout |
| --- | --- |
| `inter_states` (input) | [1, 16, 384], [16384, 16, 384], [64, 16, 384], [8192, 16, 384]; float8_e4m3fn; strides [6144, 384, 1]. contiguous generated operand at frozen case shape. |
| `w2` (input) | [896, 3584, 192]; float4_e2m1fn_x2; strides [688128, 192, 1]. contiguous generated operand at frozen case shape. |
| `w2_scale` (input) | [3211264, 16]; float8_e8m0fnu; strides [16, 1]. contiguous generated operand at frozen case shape. |
| `a2_scale` (input) | [188416, 16], [28672, 16], [29696, 16], [319488, 16]; float8_e8m0fnu; strides [16, 1]. contiguous generated operand at frozen case shape. |
| `sorted_token_ids` (input) | [188400], [28672], [29680], [319472]; int32; strides [1]. serialized oracle tensor layout; reconstructed as prescribed by the protected builder. |
| `sorted_expert_ids` (input) | [2944], [4992], [896], [928]; int32; strides [1]. serialized oracle tensor layout; reconstructed as prescribed by the protected builder. |
| `num_valid_ids` (input) | [2]; int32; strides [1]. serialized oracle tensor layout; reconstructed as prescribed by the protected builder. |
| `sorted_weights` (input) | [188400], [28672], [29680], [319472]; float32; strides [1]. serialized oracle tensor layout; reconstructed as prescribed by the protected builder. |
| `out` (input_output) | [1, 3584], [16384, 3584], [64, 3584], [8192, 3584]; bfloat16; strides [3584, 1]. caller-owned contiguous; written in place. |
| `return` (output) | [1, 3584], [16384, 3584], [64, 3584], [8192, 3584]; bfloat16; strides [3584, 1]. serialized oracle tensor layout; reconstructed as prescribed by the protected builder. |

Each JSON tensor record cites the metadata field or protected builder that establishes its geometry. A `null` shape, dtype or stride means it is not established by readable metadata; it is not a wildcard or permission to replace the fixture. Packed weights, sparse paging maps and routing-dependent lengths must retain the protected artifact representation.

| Optional archival tensor evidence | Expected SHA-256 |
| --- | --- |
| `ut/reference_io.pt` | `5a3fefb2af1c8a2981980d68e4f90085985fd5f0301b69814d85de5c1d801a53` |

Generated inputs use `ut/generated_cases.json`; the original tensor archive is optional historical evidence. The extraction description below records the earlier archive inspection, not a runtime dependency.

Source pointers in the JSON are relative to this task directory. Every declared artifact was SHA-256 verified and its tensor metadata extracted using PyTorch 2.9.1+cpu, `weights_only=True`, CPU mapping, mmap and FakeTensorMode. Tensor values were not read, and no GPU work was performed. The JSON preserves the parser, fixture hashes, descriptor layouts and serialized alias identities. Fresh GPU validation remains governed by the task contract.
