# Shape and benchmark catalog: kimi-k3__moe_gemm1_stage1

This is a single-GPU MI355X (gfx950) callable benchmark. The serving capture used ISL 8192, OSL 1024, concurrency 64 and TP=8; those settings are context, not a substitute for the tensor shapes below.

The complete case IDs, argument inventories, original metadata, scalars, tensor attributes and source hashes are in [SHAPES.json](SHAPES.json). The task environment and benchmark commands are in [config.yaml](config.yaml).

The inventory has **3 shape records** and **3 benchmark cases**. The protected metadata's `num_cases` field is `not declared`; that field can count a different capture scope. Histogram observations and random value draws are not added to the benchmark count.

Callable: `aiter.ops.flydsl.moe_kernels:flydsl_moe_stage1`. Baseline recorded in metadata: `baseline_src/flydsl/moe_kernels.py:flydsl_moe_stage1`. Selected callable seam; kernel launch count is not asserted by this catalog.

grouped per-expert fused gate/up GEMM + activation

Caller supplies out[M,16,384], which is written and returned. Real stage-2 routing is inverted and re-sorted for stage 1; the decode launch variant is assumed from prefill, as recorded in provenance_note.

| Benchmark case ID | Regime | Tensor geometry | Shape source |
| --- | --- | --- | --- |
| `prefill_M8192` | prefill | a [8192×3584] bfloat16 | `ut/meta.json#/case_specs/0`; JSON record 0 |
| `decode_M1` | decode | a [1×3584] bfloat16 | `ut/meta.json#/case_specs/1`; JSON record 1 |
| `decode_M64` | decode | a [64×3584] bfloat16 | `ut/meta.json#/case_specs/2`; JSON record 2 |

Benchmark IDs above follow the protected timing builder and common adapter. All capture-only and boundary records remain in the JSON inventory with their original IDs and explicit benchmark membership. The `tensor_overrides` entries distinguish timing storage from captured storage: MiniMax prefill remaps selected request rows and changes `slot_ids` to int32; recurrent decode generates contiguous timing tensors.

| Argument / output | Recorded geometry and layout |
| --- | --- |
| `a` (input) | [1, 3584], [64, 3584], [8192, 3584]; bfloat16; strides [3584, 1]. contiguous generated operand at frozen case shape. |
| `w1` (input) | [896, 768, 1792]; float4_e2m1fn_x2; strides [1376256, 1792, 1]. contiguous generated operand at frozen case shape. |
| `w1_scale` (input) | [688128, 112]; float8_e8m0fnu; strides [112, 1]. contiguous generated operand at frozen case shape. |
| `sorted_token_ids` (input) | unknown; int32; strides unknown. frozen real routing; exact shape/stride only in tensor oracle. |
| `sorted_expert_ids` (input) | unknown; int32; strides unknown. frozen real routing; exact shape/stride only in tensor oracle. |
| `num_valid_ids` (input) | unknown; int32; strides unknown. frozen real routing; exact shape/stride only in tensor oracle. |
| `out` (input_output) | [1, 16, 384], [64, 16, 384], [8192, 16, 384]; bfloat16; strides [6144, 384, 1]. caller-owned contiguous; written in place. |
| `return` (output) | [1, 16, 384], [64, 16, 384], [8192, 16, 384]; bfloat16; strides [6144, 384, 1]. aliases supplied output. |

Each JSON tensor record cites the metadata field or protected builder that establishes its geometry. A `null` shape, dtype or stride means it is not established by readable metadata; it is not a wildcard or permission to replace the fixture. Packed weights, sparse paging maps and routing-dependent lengths must retain the protected artifact representation.

ROUTING IS REAL but SECOND-HAND: it is the served topk distribution captured at the stage-2 seam of the 0817 session (ut_stage2/reference_io.pt, same model / TP=8 / ISL 8192 / OSL 1024 / conc 64), inverted back to (topk_ids, topk_weights) and re-sorted by the production aiter moe_sorting at stage-1's own block size. It was NOT captured at the stage-1 seam. The launch variant is the REAL 0828 one (kernel name recorded live in tuning/work + the architect report), resolved through the frozen registry, but the DECODE variant is assumed identical to prefill's rather than observed. _capture_overlay1/ is the (unrun) hook that would replace both with a first-hand stage-1 capture. SEPARATELY: the golden is the elementwise MEDIAN of 21 frozen-baseline launches, not a single launch, because this kernel is not run-to-run reproducible at served routing spreads -- see nondeterminism_calibration.

**Evidence limit:** 9 tensor records have at least one unknown physical field. The machine catalog enumerates each field and its source. Exact opaque-fixture details require the hash-pinned tensor artifacts; this static catalog does not claim that they were inspected.

| Protected tensor artifact | Expected SHA-256 |
| --- | --- |
| `ut/reference_io.pt` | `3d175ea5803d3523001cac000aa5d6c7d54b4a06dfa97999cd8eb22c081b3e31` |

Source pointers in the JSON are relative to this task directory. Tensor artifacts were not deserialized, and no task code or GPU benchmark was run to produce this catalog. Fresh validation remains governed by the task contract.
