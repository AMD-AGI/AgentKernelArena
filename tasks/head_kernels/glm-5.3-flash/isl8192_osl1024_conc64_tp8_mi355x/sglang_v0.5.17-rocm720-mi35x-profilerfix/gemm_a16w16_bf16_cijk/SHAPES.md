# Shape and benchmark catalog: glm-5.3-flash__gemm_a16w16_bf16_cijk

This is a single-GPU MI355X (gfx950) callable benchmark. The serving capture used ISL 8192, OSL 1024, concurrency 64 and TP=8; those settings are context, not a substitute for the tensor shapes below.

The complete case IDs, argument inventories, original metadata, scalars, tensor attributes and source hashes are in [SHAPES.json](SHAPES.json). The task environment and benchmark commands are in [config.yaml](config.yaml).

The inventory has **27 shape records** and **27 benchmark cases**. The protected metadata's `num_cases` field is `not declared`; that field can count a different capture scope. Histogram observations and random value draws are not added to the benchmark count.

Callable: `aiter.tuned_gemm:torch_gemm`. Baseline recorded in metadata: `aiter.tuned_gemm:torch_gemm`. Selected callable seam; kernel launch count is not asserted by this catalog.

C[M,N] = A[M,K] @ B[N,K]^T  (bf16 in / bf16 out, no bias, no scales) — sglang UnquantizedLinearMethod.apply -> tgemm.mm -> aiter.tuned_gemm.gemm_a16w16 -> per-shape DB lookup -> hipBLASLt Tensile Cijk_* / torch / skinny / triton solution

C[M,N] = A[M,K] @ B[N,K]^T  (bf16 in / bf16 out, no bias, no scales) — sglang UnquantizedLinearMethod.apply -> tgemm.mm -> aiter.tuned_gemm.gemm_a16w16 -> per-shape DB lookup -> hipBLASLt Tensile Cijk_* / torch / skinny / triton solution

| Benchmark case ID | Regime | Tensor geometry | Shape source |
| --- | --- | --- | --- |
| `nk128x4096_m64_decode` | decode | inp [64×4096] bfloat16; weights [128×4096] bfloat16 | `ut/meta.json#/cases/0`; JSON record 0 |
| `nk3072x4096_m64_decode` | decode | inp [64×4096] bfloat16; weights [3072×4096] bfloat16 | `ut/meta.json#/cases/1`; JSON record 1 |
| `nk288x4096_m64_decode` | decode | inp [64×4096] bfloat16; weights [288×4096] bfloat16 | `ut/meta.json#/cases/2`; JSON record 2 |
| `nk1024x128_m64_decode` | decode | inp [64×128] bfloat16; weights [1024×128] bfloat16 | `ut/meta.json#/cases/3`; JSON record 3 |
| `nk128x4096_m8192_prefill` | prefill | inp [8192×4096] bfloat16; weights [128×4096] bfloat16 | `ut/meta.json#/cases/4`; JSON record 4 |
| `nk8x4096_m64_decode` | decode | inp [64×4096] bfloat16; weights [8×4096] bfloat16 | `ut/meta.json#/cases/5`; JSON record 5 |
| `nk4096x1024_m64_decode` | decode | inp [64×1024] bfloat16; weights [4096×1024] bfloat16 | `ut/meta.json#/cases/6`; JSON record 6 |
| `nk3072x4096_m8192_prefill` | prefill | inp [8192×4096] bfloat16; weights [3072×4096] bfloat16 | `ut/meta.json#/cases/7`; JSON record 7 |
| `nk288x4096_m8192_prefill` | prefill | inp [8192×4096] bfloat16; weights [288×4096] bfloat16 | `ut/meta.json#/cases/8`; JSON record 8 |
| `nk1024x128_m8192_prefill` | prefill | inp [8192×128] bfloat16; weights [1024×128] bfloat16 | `ut/meta.json#/cases/9`; JSON record 9 |
| `nk8x4096_m8192_prefill` | prefill | inp [8192×4096] bfloat16; weights [8×4096] bfloat16 | `ut/meta.json#/cases/10`; JSON record 10 |
| `nk4096x1024_m8192_prefill` | prefill | inp [8192×1024] bfloat16; weights [4096×1024] bfloat16 | `ut/meta.json#/cases/11`; JSON record 11 |
| `nk4096x1536_m64_decode` | decode | inp [64×1536] bfloat16; weights [4096×1536] bfloat16 | `ut/meta.json#/cases/12`; JSON record 12 |
| `nk4096x1536_m8192_prefill` | prefill | inp [8192×1536] bfloat16; weights [4096×1536] bfloat16 | `ut/meta.json#/cases/13`; JSON record 13 |
| `nk32x4096_m8192_prefill` | prefill | inp [8192×4096] bfloat16; weights [32×4096] bfloat16 | `ut/meta.json#/cases/14`; JSON record 14 |
| `nk4096x512_m8192_prefill` | prefill | inp [8192×512] bfloat16; weights [4096×512] bfloat16 | `ut/meta.json#/cases/15`; JSON record 15 |
| `nk32x4096_m64_decode` | decode | inp [64×4096] bfloat16; weights [32×4096] bfloat16 | `ut/meta.json#/cases/16`; JSON record 16 |
| `nk4096x512_m64_decode` | decode | inp [64×512] bfloat16; weights [4096×512] bfloat16 | `ut/meta.json#/cases/17`; JSON record 17 |
| `nk32x4096_m1_decode` | decode | inp [1×4096] bfloat16; weights [32×4096] bfloat16 | `ut/meta.json#/cases/18`; JSON record 18 |
| `nk4096x512_m1_decode` | decode | inp [1×512] bfloat16; weights [4096×512] bfloat16 | `ut/meta.json#/cases/19`; JSON record 19 |
| `nk4096x1536_m1_decode` | decode | inp [1×1536] bfloat16; weights [4096×1536] bfloat16 | `ut/meta.json#/cases/20`; JSON record 20 |
| `nk3072x4096_m1_decode` | decode | inp [1×4096] bfloat16; weights [3072×4096] bfloat16 | `ut/meta.json#/cases/21`; JSON record 21 |
| `nk4096x1024_m1_decode` | decode | inp [1×1024] bfloat16; weights [4096×1024] bfloat16 | `ut/meta.json#/cases/22`; JSON record 22 |
| `nk288x4096_m1_decode` | decode | inp [1×4096] bfloat16; weights [288×4096] bfloat16 | `ut/meta.json#/cases/23`; JSON record 23 |
| `nk128x4096_m1_decode` | decode | inp [1×4096] bfloat16; weights [128×4096] bfloat16 | `ut/meta.json#/cases/24`; JSON record 24 |
| `nk8x4096_m1_decode` | decode | inp [1×4096] bfloat16; weights [8×4096] bfloat16 | `ut/meta.json#/cases/25`; JSON record 25 |
| `nk1024x128_m1_decode` | decode | inp [1×128] bfloat16; weights [1024×128] bfloat16 | `ut/meta.json#/cases/26`; JSON record 26 |

Benchmark IDs above follow the protected timing builder and common adapter. All capture-only and boundary records remain in the JSON inventory with their original IDs and explicit benchmark membership. The `tensor_overrides` entries distinguish timing storage from captured storage: MiniMax prefill remaps selected request rows and changes `slot_ids` to int32; recurrent decode generates contiguous timing tensors.

| Argument / output | Recorded geometry and layout |
| --- | --- |
| `inp` (input) | 15 exact variants in JSON; bfloat16; strides per-case values in JSON. row-major contiguous. |
| `weights` (input) | 9 exact variants in JSON; bfloat16; strides per-case values in JSON. row-major contiguous. |
| `return` (output) | 21 exact variants in JSON; bfloat16; strides per-case values in JSON. fresh contiguous result. |

Each JSON tensor record cites the metadata field or protected builder that establishes its geometry. A `null` shape, dtype or stride means it is not established by readable metadata; it is not a wildcard or permission to replace the fixture. Packed weights, sparse paging maps and routing-dependent lengths must retain the protected artifact representation.

upstream op package: profile-derived families and serving M buckets; not frozen captured tensor values

Source pointers in the JSON are relative to this task directory. Tensor artifacts were not deserialized, and no task code or GPU benchmark was run to produce this catalog. Fresh validation remains governed by the task contract.
