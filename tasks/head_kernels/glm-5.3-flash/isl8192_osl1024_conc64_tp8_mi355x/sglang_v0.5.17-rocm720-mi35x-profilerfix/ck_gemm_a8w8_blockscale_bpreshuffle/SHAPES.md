# Shape and benchmark catalog: glm-5.3-flash__ck_gemm_a8w8_blockscale_bpreshuffle

This is a single-GPU MI355X (gfx950) callable benchmark. The serving capture used ISL 8192, OSL 1024, concurrency 64 and TP=8; those settings are context, not a substitute for the tensor shapes below.

The complete case IDs, argument inventories, original metadata, scalars, tensor attributes and source hashes are in [SHAPES.json](SHAPES.json). The task environment and benchmark commands are in [config.yaml](config.yaml).

The inventory has **21 shape records** and **7 scored benchmark cases**. The protected metadata's `num_cases` field is `not declared`; that field can count a different capture scope. Histogram observations and random value draws are not added to the benchmark count.

All 21 cases remain mandatory correctness coverage; 14 inferred or unprofiled cases are unscored robustness/generalization. See [ut/meta.json](ut/meta.json) for each original source row, count status, and classification.

Callable: `aiter:gemm_a8w8_blockscale_bpreshuffle`. Baseline recorded in metadata: `aiter:gemm_a8w8_blockscale_bpreshuffle`. Selected callable seam; kernel launch count is not asserted by this catalog.

C[M,N] = (dequant(XQ)[M,K] @ dequant(WQ)[N,K]^T) -> bf16 ; fp8 a8w8 block-scale, act per-1x128 (x_scale[M,K/128], physically column-major per materialize_bpreshuffle_fp8_scale), weight per-128x128 (w_scale[N/128,K/128]), WQ PRESHUFFLED with shuffle_weight(w,(16,16))

| Scored benchmark case ID | Regime | Original source row | Observed invocation count |
| --- | --- | --- | --- |
| `nk512x4096_m64_decode` | decode | `ut/provenance/workload.json#/cases/1` | 4410 |
| `nk4096x256_m64_decode` | decode | `ut/provenance/workload.json#/cases/4` | 4410 |
| `nk2048x4096_m64_decode` | decode | `ut/provenance/workload.json#/cases/7` | 1155 |
| `nk2048x1536_m64_decode` | decode | `ut/provenance/workload.json#/cases/10` | 1155 |
| `nk4096x2048_m64_decode` | decode | `ut/provenance/workload.json#/cases/13` | 1155 |
| `nk3072x4096_m64_decode` | decode | `ut/provenance/workload.json#/cases/16` | 315 |
| `nk4096x1536_m64_decode` | decode | `ut/provenance/workload.json#/cases/19` | 315 |

Benchmark IDs above follow the protected timing builder and common adapter. All 21 declared geometries remain in the JSON inventory and correctness selector; the 14 unscored cases are listed separately under `unscored_correctness_cases`.

| Argument / output | Recorded geometry and layout |
| --- | --- |
| `XQ` (input) | 12 exact variants in JSON; float8_e4m3fn; strides per-case values in JSON. row-major contiguous. |
| `WQ` (input) | 7 exact variants in JSON; float8_e4m3fn; strides per-case values in JSON. contiguous storage; AITER (16,16) preshuffled values. |
| `x_scale` (input) | 12 exact variants in JSON; float32; strides per-case values in JSON. transpose-contiguous; column-major equivalent, degenerate M=1 shares contiguous layout. |
| `w_scale` (input) | 7 exact variants in JSON; float32; strides per-case values in JSON. row-major contiguous. |
| `return` (output) | 12 exact variants in JSON; bfloat16; strides per-case values in JSON. fresh contiguous result. |

Each JSON tensor record cites the metadata field or protected builder that establishes its geometry. A `null` shape, dtype or stride means it is not established by readable metadata; it is not a wildcard or permission to replace the fixture. Every declared GEMM operand retains its protected shape, dtype and physical layout in correctness coverage.

upstream op package: profile-derived families and serving M buckets; not frozen captured tensor values

Source pointers in the JSON are relative to this task directory. Tensor artifacts were not deserialized, and no task code or GPU benchmark was run to produce this catalog. Fresh validation remains governed by the task contract.

Original workload notes limit observed launches to M64. Source row counts/weights are provenance and do not change the shared Arena aggregation. Current per-case device dispatch and eager-versus-graph equivalence remain unproved.

The original GLM FP8 model tuning table and exact CK dispatch proof are missing. The task retains the stock `ut/dispatch.csv`, which has no matching GLM family rows and selects native CK defaults; the observed-shape classification does not establish historical dispatch equivalence.
