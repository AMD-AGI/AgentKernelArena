# Shape and benchmark catalog: glm-5.3-flash__ck_gemm_a8w8_blockscale_bpreshuffle

This is a single-GPU MI355X (gfx950) callable benchmark. The serving capture used ISL 8192, OSL 1024, concurrency 64 and TP=8; those settings are context, not a substitute for the tensor shapes below.

The complete case IDs, argument inventories, original metadata, scalars, tensor attributes and source hashes are in [SHAPES.json](SHAPES.json). The task environment and benchmark commands are in [config.yaml](config.yaml).

The inventory has **21 shape records** and **21 benchmark cases**. The protected metadata's `num_cases` field is `not declared`; that field can count a different capture scope. Histogram observations and random value draws are not added to the benchmark count.

Callable: `aiter:gemm_a8w8_blockscale_bpreshuffle`. Baseline recorded in metadata: `aiter:gemm_a8w8_blockscale_bpreshuffle`. Selected callable seam; kernel launch count is not asserted by this catalog.

C[M,N] = (dequant(XQ)[M,K] @ dequant(WQ)[N,K]^T) -> bf16 ; fp8 a8w8 block-scale, act per-1x128 (x_scale[M,K/128], physically column-major per materialize_bpreshuffle_fp8_scale), weight per-128x128 (w_scale[N/128,K/128]), WQ PRESHUFFLED with shuffle_weight(w,(16,16))

C[M,N] = (dequant(XQ)[M,K] @ dequant(WQ)[N,K]^T) -> bf16 ; fp8 a8w8 block-scale, act per-1x128 (x_scale[M,K/128], physically column-major per materialize_bpreshuffle_fp8_scale), weight per-128x128 (w_scale[N/128,K/128]), WQ PRESHUFFLED with shuffle_weight(w,(16,16))

| Benchmark case ID | Regime | Tensor geometry | Shape source |
| --- | --- | --- | --- |
| `nk512x4096_m1_decode` | decode | XQ [1×4096] float8_e4m3fn; WQ [512×4096] float8_e4m3fn | `ut/meta.json#/cases/0`; JSON record 0 |
| `nk512x4096_m64_decode` | decode | XQ [64×4096] float8_e4m3fn; WQ [512×4096] float8_e4m3fn | `ut/meta.json#/cases/1`; JSON record 1 |
| `nk512x4096_m8192_prefill` | prefill | XQ [8192×4096] float8_e4m3fn; WQ [512×4096] float8_e4m3fn | `ut/meta.json#/cases/2`; JSON record 2 |
| `nk4096x256_m1_decode` | decode | XQ [1×256] float8_e4m3fn; WQ [4096×256] float8_e4m3fn | `ut/meta.json#/cases/3`; JSON record 3 |
| `nk4096x256_m64_decode` | decode | XQ [64×256] float8_e4m3fn; WQ [4096×256] float8_e4m3fn | `ut/meta.json#/cases/4`; JSON record 4 |
| `nk4096x256_m8192_prefill` | prefill | XQ [8192×256] float8_e4m3fn; WQ [4096×256] float8_e4m3fn | `ut/meta.json#/cases/5`; JSON record 5 |
| `nk2048x4096_m1_decode` | decode | XQ [1×4096] float8_e4m3fn; WQ [2048×4096] float8_e4m3fn | `ut/meta.json#/cases/6`; JSON record 6 |
| `nk2048x4096_m64_decode` | decode | XQ [64×4096] float8_e4m3fn; WQ [2048×4096] float8_e4m3fn | `ut/meta.json#/cases/7`; JSON record 7 |
| `nk2048x4096_m8192_prefill` | prefill | XQ [8192×4096] float8_e4m3fn; WQ [2048×4096] float8_e4m3fn | `ut/meta.json#/cases/8`; JSON record 8 |
| `nk2048x1536_m1_decode` | decode | XQ [1×1536] float8_e4m3fn; WQ [2048×1536] float8_e4m3fn | `ut/meta.json#/cases/9`; JSON record 9 |
| `nk2048x1536_m64_decode` | decode | XQ [64×1536] float8_e4m3fn; WQ [2048×1536] float8_e4m3fn | `ut/meta.json#/cases/10`; JSON record 10 |
| `nk2048x1536_m8192_prefill` | prefill | XQ [8192×1536] float8_e4m3fn; WQ [2048×1536] float8_e4m3fn | `ut/meta.json#/cases/11`; JSON record 11 |
| `nk4096x2048_m1_decode` | decode | XQ [1×2048] float8_e4m3fn; WQ [4096×2048] float8_e4m3fn | `ut/meta.json#/cases/12`; JSON record 12 |
| `nk4096x2048_m64_decode` | decode | XQ [64×2048] float8_e4m3fn; WQ [4096×2048] float8_e4m3fn | `ut/meta.json#/cases/13`; JSON record 13 |
| `nk4096x2048_m8192_prefill` | prefill | XQ [8192×2048] float8_e4m3fn; WQ [4096×2048] float8_e4m3fn | `ut/meta.json#/cases/14`; JSON record 14 |
| `nk3072x4096_m1_decode` | decode | XQ [1×4096] float8_e4m3fn; WQ [3072×4096] float8_e4m3fn | `ut/meta.json#/cases/15`; JSON record 15 |
| `nk3072x4096_m64_decode` | decode | XQ [64×4096] float8_e4m3fn; WQ [3072×4096] float8_e4m3fn | `ut/meta.json#/cases/16`; JSON record 16 |
| `nk3072x4096_m8192_prefill` | prefill | XQ [8192×4096] float8_e4m3fn; WQ [3072×4096] float8_e4m3fn | `ut/meta.json#/cases/17`; JSON record 17 |
| `nk4096x1536_m1_decode` | decode | XQ [1×1536] float8_e4m3fn; WQ [4096×1536] float8_e4m3fn | `ut/meta.json#/cases/18`; JSON record 18 |
| `nk4096x1536_m64_decode` | decode | XQ [64×1536] float8_e4m3fn; WQ [4096×1536] float8_e4m3fn | `ut/meta.json#/cases/19`; JSON record 19 |
| `nk4096x1536_m8192_prefill` | prefill | XQ [8192×1536] float8_e4m3fn; WQ [4096×1536] float8_e4m3fn | `ut/meta.json#/cases/20`; JSON record 20 |

Benchmark IDs above follow the protected timing builder and common adapter. All capture-only and boundary records remain in the JSON inventory with their original IDs and explicit benchmark membership. The `tensor_overrides` entries distinguish timing storage from captured storage: MiniMax prefill remaps selected request rows and changes `slot_ids` to int32; recurrent decode generates contiguous timing tensors.

| Argument / output | Recorded geometry and layout |
| --- | --- |
| `XQ` (input) | 12 exact variants in JSON; float8_e4m3fn; strides per-case values in JSON. row-major contiguous. |
| `WQ` (input) | 7 exact variants in JSON; float8_e4m3fn; strides per-case values in JSON. contiguous storage; AITER (16,16) preshuffled values. |
| `x_scale` (input) | 12 exact variants in JSON; float32; strides per-case values in JSON. transpose-contiguous; column-major equivalent, degenerate M=1 shares contiguous layout. |
| `w_scale` (input) | 7 exact variants in JSON; float32; strides per-case values in JSON. row-major contiguous. |
| `return` (output) | 12 exact variants in JSON; bfloat16; strides per-case values in JSON. fresh contiguous result. |

Each JSON tensor record cites the metadata field or protected builder that establishes its geometry. A `null` shape, dtype or stride means it is not established by readable metadata; it is not a wildcard or permission to replace the fixture. Packed weights, sparse paging maps and routing-dependent lengths must retain the protected artifact representation.

upstream op package: profile-derived families and serving M buckets; not frozen captured tensor values

Source pointers in the JSON are relative to this task directory. Tensor artifacts were not deserialized, and no task code or GPU benchmark was run to produce this catalog. Fresh validation remains governed by the task contract.
