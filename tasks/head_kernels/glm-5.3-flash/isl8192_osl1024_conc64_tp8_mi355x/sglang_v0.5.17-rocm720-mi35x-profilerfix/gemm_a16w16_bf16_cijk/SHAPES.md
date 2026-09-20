# Shape and benchmark catalog: glm-5.3-flash__gemm_a16w16_bf16_cijk

This is a single-GPU MI355X (gfx950) callable benchmark. The serving capture used ISL 8192, OSL 1024, concurrency 64 and TP=8; those settings are context, not a substitute for the tensor shapes below.

The complete case IDs, argument inventories, original metadata, scalars, tensor attributes and source hashes are in [SHAPES.json](SHAPES.json). The task environment and benchmark commands are in [config.yaml](config.yaml).

The inventory has **27 shape records** and **7 scored benchmark cases**. The protected metadata's `num_cases` field is `not declared`; that field can count a different capture scope. Histogram observations and random value draws are not added to the benchmark count.

All 27 cases remain mandatory correctness coverage; 20 inferred or unprofiled cases are unscored robustness/generalization. See [ut/meta.json](ut/meta.json) for each original source row, count status, and classification.

Callable: `aiter.tuned_gemm:torch_gemm`. Baseline recorded in metadata: `aiter.tuned_gemm:torch_gemm`. Selected callable seam; kernel launch count is not asserted by this catalog.

C[M,N] = A[M,K] @ B[N,K]^T  (bf16 in / bf16 out, no bias, no scales) — sglang UnquantizedLinearMethod.apply -> tgemm.mm -> aiter.tuned_gemm.gemm_a16w16 -> per-shape DB lookup -> hipBLASLt Tensile Cijk_* / torch / skinny / triton solution

| Scored benchmark case ID | Regime | Original source row | Observed invocation count |
| --- | --- | --- | --- |
| `nk128x4096_m64_decode` | decode | `ut/provenance/workload.json#/cases/0` | Not recorded |
| `nk3072x4096_m64_decode` | decode | `ut/provenance/workload.json#/cases/1` | Not recorded |
| `nk288x4096_m64_decode` | decode | `ut/provenance/workload.json#/cases/2` | Not recorded |
| `nk1024x128_m64_decode` | decode | `ut/provenance/workload.json#/cases/3` | Not recorded |
| `nk8x4096_m64_decode` | decode | `ut/provenance/workload.json#/cases/5` | Not recorded |
| `nk4096x1024_m64_decode` | decode | `ut/provenance/workload.json#/cases/6` | Not recorded |
| `nk4096x1536_m64_decode` | decode | `ut/provenance/workload.json#/cases/12` | Not recorded |

Benchmark IDs above follow the protected timing builder and common adapter. All 27 declared geometries remain in the JSON inventory and correctness selector; the 20 unscored cases are listed separately under `unscored_correctness_cases`.

| Argument / output | Recorded geometry and layout |
| --- | --- |
| `inp` (input) | 15 exact variants in JSON; bfloat16; strides per-case values in JSON. row-major contiguous. |
| `weights` (input) | 9 exact variants in JSON; bfloat16; strides per-case values in JSON. row-major contiguous. |
| `return` (output) | 21 exact variants in JSON; bfloat16; strides per-case values in JSON. fresh contiguous result. |

Each JSON tensor record cites the metadata field or protected builder that establishes its geometry. A `null` shape, dtype or stride means it is not established by readable metadata; it is not a wildcard or permission to replace the fixture. Every declared GEMM operand retains its protected shape, dtype and physical layout in correctness coverage.

upstream op package: profile-derived families and serving M buckets; not frozen captured tensor values

Source pointers in the JSON are relative to this task directory. Tensor artifacts were not deserialized, and no task code or GPU benchmark was run to produce this catalog. Fresh validation remains governed by the task contract.

Original workload notes limit observed launches to M64. Source row counts/weights are provenance and do not change the shared Arena aggregation. Current per-case device dispatch and eager-versus-graph equivalence remain unproved.
