# Qwen captured native paged-attention decode

This task covers the original TP8/EP8, page-size-1 ragged-attention capture.
It does not represent the newer TP8/EP1, page-size-16 unified-attention workload.

The public ABI remains `aiter.ops.attention:paged_attention_ragged`. The one
recorded case has BF16 query/output `[64,8,256]`, compact BF16 K/V caches
`[521351,1,1,256]`, all 521351 referenced pages, and the complete original
ragged metadata and scale tensors. The 71,926,272-byte scratch buffer, 256-token
partitions, 44 maximum partitions, NHD layout, and all other captured arguments
remain unchanged. The full ABI is frozen in `ut/benchmark_contract.json`.

The runtime is the exact SGLang 0.5.20 image digest in `config.yaml`.
The capture and original oracle came from the documented 0.5.18 environment;
new qualification does not relabel that provenance or claim a new serving run.

## Native implementation and edit boundary

The actual image-native HIP implementation is included in
[`source/pa_kernels.cuh`](source/pa_kernels.cuh). Only these two device bodies
may be edited:

- `_paged_attention_kernel`
- `_paged_attention_ll4mi_reduce_kernel`

Their includes, preprocessing, declarations and signatures remain frozen.
The reused `ut/cpp_body_guard.py` rejects scope escapes, host constructors and
changed preprocessing before compilation. The Python ABI wrapper
`source/attention_candidate.py`, the source-routing alias, original baseline,
compile loader, launch template and support headers are immutable inputs.

The frozen loader compiles the same native GOLDEN specialization used by the
recorded head-size-256 case. Its binary/cache namespace includes a fresh process
nonce plus the submitted device-header and frozen compile-context hashes. It checks the actual copied
header before accepting a cached binary. The graph adapter separately verifies
the active Python wrapper, native build binding, and device-source digest.
The original image-native callable remains the correctness baseline.

[`ut/native_source_manifest.json`](ut/native_source_manifest.json) records the
exact image-source files and the explicit host-loading adaptations. This is a
native-source extraction, not a port or a claim that image Git metadata denotes
a clean upstream checkout.

## Original fixture and portable preparation

The original `ut/reference_io.pt` is a separate **536,480,065-byte** object with
SHA-256 `9c40d52c145027f8affca9e41bbf1100189fde724d42d085ac6296d57020ac56`.
Its object path is declared in
[`tools/headkernel-artifacts.json`](../../../tools/headkernel-artifacts.json).
Use the existing `src/tools/prepare_head_kernel_artifacts.py` helper from the
repository root with `--task qwen3.8-2.4t__paged_attention_decode` and a local
NVMe-backed `--suite-root`; `--download` verifies the original byte count,
SHA-256 and task metadata before installation. No regenerated golden or reduced
KV-page subset substitutes for the original fixture.

Use a verified assigned GPU and the exact declared image. Inside the prepared
task workspace, the configured commands remain:

```bash
python3 scripts/task_runner.py compile
python3 scripts/task_runner.py correctness
python3 scripts/task_runner.py performance
```

Compilation builds the real HIP specialization from the submitted header.
Correctness retains the frozen capture checks, random query parity, original
device engagement checks and output/input contracts, then runs independent CPU
mathematical challenges. Current selection evidence is written under
`ut/reports/`; the historical `ut/selection_validation.json` is not overwritten.

## Measured graph and CPU truth

The materialized canonical `_aka_benchmark.py` helper owns graph capture and
device events: **10 warmups, 100 ordered raw samples, one graph replay per
sample**. Graph failure is a failed run, with no event or median fallback.

Every preparation generates a new legal BF16 query from a CPU RNG with a fresh
run seed recorded in the report. Captured K/V pages, routing metadata and scale
tensors stay fixed. Expected softmax-attention outputs are computed and retained
on CPU. Only callable operands enter GPU memory; expected answers do not.

Before each replay, the query is copied into its stable buffer, output is
poisoned with NaN, and every scratch byte is set to `0xff` (NaNs in both its
FP32 statistics and BF16 partition-output regions). The preceding completed
replay is validated before its buffers are overwritten. Full immutable input
values, output alias/shape/dtype, finite values and the original mixed tolerance
`0.02` are checked outside the event interval.

The exact measured graph also runs zero-query, negated-query, small-query and
scaled-query challenges after timing. Reports retain all samples, the complete
case ABI, source/build identities, and 118 checked invocations including the
four challenges. They cannot omit a case, trim a sample, change the reported
mean, or substitute a different method.

Fresh compatible-GPU execution and a complete framework-finalized PASS are
still required for this repaired task. No optimization or serving gain is
claimed by the preparation work.
