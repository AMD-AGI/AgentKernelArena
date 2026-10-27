# MiniMax M3 dense MLP down projection

Optimize the complete rank-local BF16 GEMM at
`language_model.model.layers.0.mlp.down_proj`, starting from the public
`aiter.tuned_gemm.gemm_a16w16` dispatch at AITER
`2cb3d17589b9253ea3b3ef5222fc7f810a8f3ef0`.

This is a **configuration-derived fixed batching scenario** for the recorded
MiniMax M3 TP8, concurrency 64, input 8192/output 1024 settings. Its case weights
are declared modeling assumptions, not measured scheduler frequencies.
The task measures a single projection on one rank. It does not establish
whole-MLP performance, workload coverage, or serving throughput improvement.
Only one selected dense projection is measured; its result is not the fraction
of time spent in dense layers across the 60-layer main model.

## Workload and shape derivation

The checkpoint is `amd/MiniMax-M3-MXFP4` at revision
`b83d14e3d64bf373a207f3c2a7e9f0b0f1e7fc3a`. The complete retained checkpoint
configuration is [model_config.json](model_config.json), authenticated by
[source_identity.json](source_identity.json). It declares hidden size 6144,
dense intermediate size 12288, and a dense MLP in layer 0. TP8 splits the
down-projection input dimension into eight 1536-wide shards; each rank produces
all 6144 output channels. This task assumes that ordinary tensor-parallel dense
projection layout, with no dense-MLP data-parallel override.

Although the checkpoint uses Quark MXFP4 generally, its explicit exclusion list
contains this projection, and its unquantized tensor dtype is BF16. No MXFP4
dequantization, expert routing, or expert-parallel dimension is inferred here.
The recorded server arguments omit EP; it remains unspecified in the workload
metadata because it does not determine this dense projection.

| Case | A shape | Weight shape | Output shape | Scenario multiplicity |
| --- | --- | --- | --- | ---: |
| `prefill_m8192` | `[8192,1536]` | `[6144,1536]` | `[8192,6144]` | 64 |
| `decode_m64` | `[64,1536]` | `[6144,1536]` | `[64,6144]` | 1024 |

The declared scenario performs one 8192-row prefill projection for each of 64
requests and 1024 decode projection steps containing 64 rows each. This is a
deliberately fixed full-admission scenario derived from the request dimensions.
It does not reconstruct first-token attribution, scheduler admission, prefill
chunking, graph padding, partial batches, cache hits, or native invocation counts.
[frequency_provenance.json](frequency_provenance.json) records that distinction
and its exact relation to the weighted total-time objective. The upstream
model-named BF16 table corroborates N6144/K1536; its shape sweep and timings are
not enrolled.

## Interface and measurement boundary

Implement `source/kernel.py::mlp_down(x, weight)` and return
`(x.float() @ weight.float().T).to(torch.bfloat16)` numerically, using an available
GPU backend. Both inputs are contiguous BF16 tensors on the same GPU, including
views whose starting address is not 16-byte aligned. The
output must have shape `[M,6144]`, BF16 dtype and the same device. Inputs are
read-only. There is no bias, scale tensor, activation or residual addition.

The result is a **partial tensor-parallel product**. The following all-reduce
that sums the eight ranks is outside this task, as are the preceding activation
and gate/up projection. Do not return a full-MLP or all-reduced result.

Edit only `source/`. All value-dependent candidate work, including packing,
weight transformation, intermediate allocation and output allocation, belongs
inside the timed call. Code compilation and shape-only launch setup may be
cached. Do not cache input values or outputs between calls, read protected
reference/input code from the candidate, change dispatch tables, or alter the
reference, numerical rule, cases or timing policy.
The callable cannot express a candidate-specific one-time serving weight
preparation step. Such an integration requires a separate task boundary and
measurement of its amortization; its cost cannot disappear from this task.

## Inputs, correctness and timing

The input generator follows the normal BF16 operand construction in upstream
`op_tests/test_gemm_a16w16.py`. It generates new tensor values from seeds 42
and 1042, with independent weight generators seeded 1000000 higher. The weight
draw is equal across the two M cases for each seed. These are reproducible test
values, **not checkpoint weights or captured serving activations**. Neither old
optimized programs nor the historical MiniMax kernel-case corpus is used.

The protected oracle multiplies FP32 conversions of the BF16 inputs with TF32
disabled, then rounds to BF16. Every element must meet `rtol=0.01, atol=0.01`;
shape, dtype, device, finiteness and unchanged inputs are also checked. Both
operands are replaced in their existing storage for a second correctness draw.
For performance, that check replays the actual measured graph, detecting stale
activation, weight and output caches.
Correctness also covers unaligned activation and weight views separately and
together at both shapes, with new values replayed through each captured graph.
These layout checks do not add invocation weights to the fixed workload score.

Both roles use the same canonical graph timing helper, 10 warmup calls,
100 reported samples and a 1 ms target. All reported samples and the measured
graph replay are checked numerically outside the timed interval. The score is
the ratio of the two roles' weighted total times under the declared scenario.

The baseline uses the normal upstream tuned-table merge and dispatch, including
its default choice when no tuned row exists. It does not force Triton, select a
weak solution, or install an empty table. Each baseline action records the
selected per-case dispatch. Runtime checks bind the installed Python sources
and complete BF16 tuned-table source set to the pinned materialized AITER tree.
Compiled native artifacts and performance still require qualification in the
chosen runtime; these source checks do not certify them.
To satisfy the decode FlyDSL kernel's alignment requirement, the protected
baseline and initial candidate copy each operand whose pointer is not 16-byte aligned.
Those copies execute inside the callable and its captured graph; preparation is
charged whenever needed. Aligned scored inputs take the original upstream call
without copying, and neither path changes upstream kernel selection.

## Evaluation

The ordinary schema-v2 actions are implemented by `scripts/evaluate.py`:

```bash
python3 scripts/evaluate.py validate-task
python3 scripts/evaluate.py baseline compile
python3 scripts/evaluate.py baseline correctness
python3 scripts/evaluate.py baseline performance
python3 scripts/evaluate.py candidate compile
python3 scripts/evaluate.py candidate correctness
python3 scripts/evaluate.py candidate performance
```

Workspace setup requires installed AITER sources matching the declared pin.
The framework materializes `upstream/` and the canonical timing helper; that
source tree must not shadow the installed AITER package. Evaluation requires a
ROCm gfx950 GPU and fails if it is unavailable. CPU contract checks do not
constitute GPU qualification. A fresh framework task-validator PASS is required
before publication or optimization use.
