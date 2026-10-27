# GLM-5.2 MXFP4 TP4/EP4: one local M48 fused-MoE call

Optimize `source/kernel.py:fused_moe` for the complete rank-local operation
defined here. The starting candidate delegates to the installed, pinned AITER
public `fused_moe`; baseline actions independently call that same upstream
entrypoint. Only `source/` is editable. HIP, Triton, FlyDSL, PyTorch, and the
declared AITER runtime may be used. Candidate code may not import the protected
input generator, reference, runtime checker, or evaluator, read expected outputs,
or depend on files outside its task workspace and declared installed runtime.

The recorded serving configuration is `amd/GLM-5.2-MXFP4`, TP4/EP4, AgentX
concurrency 8, with EAGLE settings and trace-driven request lengths recorded in
`workload_source.json`. The sole case `local_ep4_m48` selects 48 tokens as a legal
local-call optimization scenario. Its weight is one invocation. This is a local
operator proxy: 48 is not a measured AgentX batch, an inferred EAGLE verification
width, or a deployment call frequency. No end-to-end gain is claimed.

The public checkpoint config at revision
`386bd0e4ec821f7b07975701cec3c3b953a5576a` confirms hidden size 6144, MoE
intermediate size 2048, 256 routed experts, one shared expert, and top-8 routed
selection. Its immutable URL, full-content hash and relevant projection are
retained in `workload_source.json`. Saved launch preparation identifies the
serving settings; it supplies no checkpoint tensors or timing baseline.
The selected model site is the first main-model MoE, `model.layers.3.mlp`,
after three dense layers. Its 195 rank-0 routed/shared projection modules have
no quantization exclusions or layer/type overrides. The config excludes MTP
layer 78; that layer is outside this task.

## Callable and local-expert semantics

```python
fused_moe(hidden_states, w1, w2, topk_weight, topk_ids, **kwargs)
```

| Input | Public shape and representation |
| --- | --- |
| `hidden_states` | Contiguous BF16 `[48, 6144]` |
| `w1` | Packed MXFP4 `[65, 4096, 3072]`, AITER `(16,16)` shuffled; logical `[65, 4096, 6144]` |
| `w2` | Packed MXFP4 `[65, 6144, 1024]`, AITER `(16,16)` shuffled; logical `[65, 6144, 2048]` |
| `topk_weight` | Contiguous FP32 `[48, 9]` |
| `topk_ids` | Contiguous INT32 `[48, 9]`, global expert IDs |
| `expert_mask` | INT32 `[257]` binary mask; resident IDs 0–63 and shared ID 256 |
| `w1_scale` | E8M0 `[266240, 192]`, AITER `e8m0_shuffle` layout |
| `w2_scale` | E8M0 `[399360, 64]`, AITER `e8m0_shuffle` layout |

The remaining keyword arguments are `activation=ActivationType.Silu`,
`quant_type=QuantType.per_1x32`, `gate_mode="separated"`, and
`doweight_stage1=False`. No bias or additional activation clamp is supplied.
The gate and up matrices occupy separate halves of logical `w1`.

The mask is the upstream binary residency contract, not a table of local IDs.
For a present global ID, its local ID is `cumsum(mask)[global_id] - 1`;
nonresident IDs map to `-1`. Rank 0 therefore owns local routed IDs 0–63 and
local shared ID 64. Routes to other ranks contribute zero to this local result.
The output is one finite BF16 `[48, 6144]` tensor containing the sum of all
resident routed and shared contributions. Routing weights are applied after
the second projection.

Sorting, activation quantization, both projections, SiLU multiplication,
intermediate quantization, route weighting, and local reduction belong to the
call. Routing generation and the distributed collective are outside this
operator. The task does not measure an entire EP exchange or a full model layer.

## Generated values and precision policy

The generator uses upstream-style independent BF16 standard normal hidden
states and expert weights. It quantizes each expert separately with the pinned
upstream `per_1x32_f4_quant(..., round_mode=2)` and prepares the public shuffled
weight/scale layouts outside timing. Seed 0 controls weight generation; seed 17
controls initial hidden states and routing. The generated values are neither
checkpoint bytes nor captured serving state.

Weight quantization uses group-32 E8M0 scales and nearest-even E2M1 values. The
explicit Even mode matches the checkpoint's recorded weight format policy.
The same config records EVEN for input activation quantization, but the pinned
public AITER BF16 call has its own activation behavior. This task binds that
upstream behavior: input activation scales use `ceil_pow2(amax / 6)`, and the
selected fused stage-1 epilogue quantizes FP32 `silu(gate) * up` directly. Its
intermediate scale exponent is obtained from
`((amax_f32_bits + 0x400000) & 0xFF800000) >> 23`, minus 2 and clamped at zero.
It does not round the intermediate to BF16 first. The rules can differ from
checkpoint-declared EVEN; the task does not claim a faithful replay of the
checkpoint's configured activation quantizer.

For routing, the generator draws FP32 standard-normal logits over the 256
routed global experts, applies softmax, takes eight distinct maxima, and
normalizes their selected weights to sum to one. It appends shared global ID
256 with coefficient 1. This shared coefficient is an explicit synthetic
local-call policy; no serving or all-reduce normalization is inferred from it.
No routing-frequency estimate is inferred from these values.

All public tensors, their full backing storage, metadata, and prepared-layout
attributes are read-only. The evaluator retains canonical packed weights and
scales separately for its reference. Candidate-specific conversions and scratch
preparation must remain inside the callable; no task-supplied untimed candidate
preparation hook exists. Compiled code may be cached, but outputs or transformed
inputs may not be cached across calls in place of computing the declared work.

## Reference, checks, and timing

`scripts/reference.py` implements an independent PyTorch oracle without
importing AITER or candidate code. It decodes E2M1 nibbles and E8M0 scales,
quantizes the hidden states, computes both projections and SiLU in FP32 with
TF32 disabled, follows the fused intermediate quantization above, applies route
weights, and sums the complete local result before conversion to BF16.
Only resident experts are evaluated.

Output dtype, shape, device and finiteness must match. The fixed comparison
accepts either all elements within `rtol=0.01, atol=0.01`, or normalized
difference

```text
D = 1 - 2 * sum(reference * output) / sum(reference^2 + output^2) <= 0.001
```

with zero difference when both tensors are zero. This is a whole-output
comparison; it never omits remote routes, difficult elements, or token rows.
The threshold uses the tighter warning-level difference bound from upstream's
checked-call policy. It is frozen before native qualification and is not
adjusted in response to a candidate's results. Reference controls cover zero
hidden states, all-remote routes, and deliberately corrupted output.

Both roles use the canonical materialized `_aka_benchmark` helper with CUDA
graph timing on ROCm: 10 warmups, 100 samples, a 1 ms graph target and a maximum
of 1000 graph repeats. The scoreable latency is the median per-invocation device
time. Graph failure is an error, not an implicit timing-method change. Input
generation and common upstream weight preparation are outside timing for both
roles. The callable includes its output and scratch work on every invocation.

The evaluator checks the actual measured outputs and exact captured graph
replay. It then copies seed-29 hidden states and routing into the same public
tensor objects and storage addresses, computes a fresh oracle, and replays
that graph against the fresh result. It requires the new expected result to
reject the stale initial output. Full public-storage and metadata checks run
around each phase. Weights remain fixed under this scenario's declared policy.

## Pinned baseline and qualification

`config.yaml` acquires upstream AITER at
`2cb3d17589b9253ea3b3ef5222fc7f810a8f3ef0`. Runtime actions call the installed
package and verify this task's selected source hashes against both the installed
package and the acquired source. The acquired tree is protected source context;
it does not shadow the installed Python package.

The required stack is PyTorch 2.11.0 ROCm, FlyDSL 0.4.0, the pinned AITER source,
and gfx950 with 256 CUs. The documented SGLang 0.5.20 build is an intended host
for this stack. The image digest in `workloads.json` is a reference runtime,
not the only usable image. The runner records the actual framework-provided
`AKA_SCORING_IMAGE_RUNTIME_REF` diagnostically and checks the required software,
source and device properties directly. Every new image needs its own GPU
task-validator qualification. The protected runner selects
`upstream/aiter/configs/model_configs/glm5_fp4_tuned_fmoe.csv`, rejects conflicting
dispatch overrides, verifies FP4 activation resolution, and checks the resolved
M64 table tier for M48:

```text
stage1: flydsl_moe1_afp4_wfp4_bf16_t32x128x256_w2_fp4
stage2: flydsl_moe2_afp4_wfp4_bf16_t32x128x256_atomic_bnt2
block_m: 32
fuse_quant: fp4
```

The upstream `op_tests/test_moe_glm5_fp4_ep_dispatch.py` supports this EP4 table
selection; its dispatch-only checks are not numerical evidence. The task's
native actions must exercise their own compiled path. Reports record resolved
baseline metadata and the hashes of loaded AITER native modules, without
asserting that loading alone proves kernel execution. No native artifacts,
historical task outputs or old timings supply this baseline.

The task exposes all seven schema-v2 actions through `scripts/evaluate.py`:
`validate-task`, and `baseline`/`candidate` with `compile`, `correctness`, or
`performance`. Failures exit nonzero and produce one `ARENA_EVAL_RESULT` envelope.
Compilation calls the selected function and synchronizes so JIT compilation and
execution errors are observed. Require native correctness and a framework-finalized
`task_validator` PASS for the exact task and runtime before PR submission.
Retain the reports with the tested source and runtime identities; CPU checks
establish neither GPU correctness nor timing.

Upstream source attribution: ROCm/AITER quantization and MoE reference helpers
are MIT licensed; the FlyDSL mixed-MoE fused epilogue is Apache-2.0, Copyright
2025–2026 FlyDSL Project Contributors. The pinned acquisition retains upstream
license files and source headers. Source identities and evidence paths are
recorded in `source_identity.json`.
