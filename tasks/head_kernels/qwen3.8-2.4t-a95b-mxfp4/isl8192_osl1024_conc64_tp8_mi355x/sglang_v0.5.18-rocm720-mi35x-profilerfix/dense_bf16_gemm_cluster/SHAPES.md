# Shape and benchmark catalog: qwen3.8-2.4t__dense_bf16_gemm_cluster

This is a single-GPU MI355X (gfx950) callable benchmark. The serving capture used ISL 8192, OSL 1024, concurrency 64 and TP=8; those settings are context, not a substitute for the tensor shapes below.

The complete case IDs, argument inventories, original metadata, scalars, tensor attributes and source hashes are in [SHAPES.json](SHAPES.json). The task environment and benchmark commands are in [config.yaml](config.yaml).

The inventory has **5 shape records** and **5 benchmark cases**. The protected metadata's `num_cases` field is `5`; that field can count a different capture scope. Histogram observations and random value draws are not added to the benchmark count.

Callable: `aiter.tuned_gemm:gemm_a16w16`. Baseline recorded in metadata: `torch.nn.functional:linear`. Selected callable seam; kernel launch count is not asserted by this catalog.

C = A @ B.T for row-major BF16 A[M,K] and B[N,K], no bias or epilogue, returning a fresh BF16 C[M,N] without mutating A or B.

Returns fresh BF16 C with unchanged A and B.

| Benchmark case ID | Regime | Tensor geometry | Shape source |
| --- | --- | --- | --- |
| `prefill_m16384_n4608_k8192` | prefill | A [16384×8192] bfloat16; B [4608×8192] bfloat16 | `ut/meta.json#/workload/cases/0`; JSON record 0 |
| `decode_m64_n32_k8192` | decode | A [64×8192] bfloat16; B [32×8192] bfloat16 | `ut/meta.json#/workload/cases/1`; JSON record 1 |
| `decode_m64_n512_k8192` | decode | A [64×8192] bfloat16; B [512×8192] bfloat16 | `ut/meta.json#/workload/cases/2`; JSON record 2 |
| `decode_m64_n4608_k8192` | decode | A [64×8192] bfloat16; B [4608×8192] bfloat16 | `ut/meta.json#/workload/cases/3`; JSON record 3 |
| `decode_m64_n8192_k256` | decode | A [64×256] bfloat16; B [8192×256] bfloat16 | `ut/meta.json#/workload/cases/4`; JSON record 4 |

Benchmark IDs above follow the protected timing builder and common adapter. All capture-only and boundary records remain in the JSON inventory with their original IDs and explicit benchmark membership. The `tensor_overrides` entries distinguish timing storage from captured storage: MiniMax prefill remaps selected request rows and changes `slot_ids` to int32; recurrent decode generates contiguous timing tensors.

| Argument / output | Recorded geometry and layout |
| --- | --- |
| `A` (input) | [16384, 8192], [64, 256], [64, 8192]; bfloat16; strides [256, 1], [8192, 1]. row-major contiguous. |
| `B` (input) | [32, 8192], [4608, 8192], [512, 8192], [8192, 256]; bfloat16; strides [256, 1], [8192, 1]. row-major contiguous. |
| `return` (output) | 5 exact variants in JSON; bfloat16; strides per-case values in JSON. fresh contiguous result. |

Each JSON tensor record cites the metadata field or protected builder that establishes its geometry. A `null` shape, dtype or stride means it is not established by readable metadata; it is not a wildcard or permission to replace the fixture. Packed weights, sparse paging maps and routing-dependent lengths must retain the protected artifact representation.

At runtime the frozen torch.nn.functional.linear baseline generates full outputs for deterministic nonzero inputs at the five live dispatch shapes; no persistent tensor oracle is stored.

Source pointers in the JSON are relative to this task directory. Tensor artifacts were not deserialized, and no task code or GPU benchmark was run to produce this catalog. Fresh validation remains governed by the task contract.
