# Top-five-model head kernels

This suite updates the older `head_kernels` dataset with captured operator
workloads from MiniMax M3, Kimi K3, DeepSeek V4 Pro, GLM 5.3 Flash, and
Qwen3.8 2.4T. The serving capture context is ISL 8192, OSL 1024,
concurrency 64, and tensor parallelism 8. An isolated task replays an
individual rank's operator; it does not launch an eight-GPU model or
measure model end-to-end throughput.

The [coverage catalog](catalog.json) connects each profiled kernel row to an
optimization task or an explicit remaining capture/implementation requirement.
Several profiler symbols share one callable and therefore one task. Catalog-only
entries have no task configuration and must not be counted as runnable tasks.
The source table's reported GPU-time shares are historical selection evidence;
historical speedups and roofline estimates are not benchmark rewards.

## Exact workload and kernel tasks

There are **18 separate kernel tasks**, organized by complete model variant,
serving workload, exact capture image, and kernel:

```text
tasks/head_kernels/<model_slug>/
  isl8192_osl1024_conc64_tp8_mi355x/
    <image_name>_<exact_image_tag>/
      <kernel_slug>/
        config.yaml
        SHAPES.json
        SHAPES.md
        source/
        ut/
        scripts/
```

Each image group has a `workload.json` and README listing its exact serving
dimensions, image reference, kernel selectors, shapes, and per-kernel runtime
requirements. Each leaf remains a self-contained task with its own source,
frozen contract, runners, and environment preflight. `SHAPES.json` preserves
the complete case inventories and source evidence; `SHAPES.md` presents the
tensor shapes, dtypes, physical layouts, and explicitly unknown metadata.

| Model | Workload and image | Tasks | Principal targets |
| --- | --- | ---: | --- |
| MiniMax M3 MXFP4 | [ISL8192 / OSL1024 / CONC64 / TP8 / v0.5.17](minimax-m3-mxfp4/isl8192_osl1024_conc64_tp8_mi355x/sglang_v0.5.17-rocm720-mi35x-profilerfix/README.md) | 3 | Decode-score attention; GQA sparse decode; GQA sparse prefill |
| Kimi K3 | [ISL8192 / OSL1024 / CONC64 / TP8 / custom Kimi image](kimi-k3/isl8192_osl1024_conc64_tp8_mi355x/sglang-rocm-k3_rocm720-mi35x-k3-20260727-tl312-08011830/README.md) | 3 | Grouped MLA decode stage one; MoE stages one and two |
| DeepSeek V4 Pro | [ISL8192 / OSL1024 / CONC64 / TP8 / v0.5.17](deepseek-v4-pro/isl8192_osl1024_conc64_tp8_mi355x/sglang_v0.5.17-rocm720-mi35x-profilerfix/README.md) | 3 | Sparse MLA decode; MoE stages one and two |
| GLM 5.3 Flash | [ISL8192 / OSL1024 / CONC64 / TP8 / v0.5.17](glm-5.3-flash/isl8192_osl1024_conc64_tp8_mi355x/sglang_v0.5.17-rocm720-mi35x-profilerfix/README.md) | 2 | BF16 Cijk GEMM; FP8 blockscale preshuffle GEMM |
| GLM 5.3 Flash | [ISL8192 / OSL1024 / CONC64 / TP8 / v0.5.18](glm-5.3-flash/isl8192_osl1024_conc64_tp8_mi355x/sglang_v0.5.18-rocm720-mi35x-profilerfix/README.md) | 2 | Fused MoE; elementwise copy/scale |
| Qwen3.8 2.4T A95B MXFP4 | [ISL8192 / OSL1024 / CONC64 / TP8 / v0.5.18](qwen3.8-2.4t-a95b-mxfp4/isl8192_osl1024_conc64_tp8_mi355x/sglang_v0.5.18-rocm720-mi35x-profilerfix/README.md) | 5 | Two-stage MXFP4 MoE; dense BF16 GEMM; recurrent gated delta; RMSNorm; paged attention |

Task selectors are the full paths relative to `tasks/`. The imported flat IDs
remain in `headkernel.operation_id` and shape evidence as stable provenance.
The Qwen two-stage MXFP4 MoE is one captured composite stage-1 + stage-2
seam; its shared profiler rows do not supply independent stage oracles.

Some targets contain editable Triton, TileLang, or FlyDSL device bodies. Others
expose a dispatcher or replacement seam backed by an image-provided vendor
kernel. Each task identifies its production callable and source seed; a
dispatcher must not be presented as the assembly source of the vendor kernel.
The committed starting implementation is what Arena measures as its baseline.
In particular, a documented prior-candidate seed is not relabeled as an
untuned stock implementation.

## Prepare the exact inputs

The suite uses captured shapes, dtypes, physical strides, tensor attributes,
and reference contracts. **This revision is not yet fully fixture-free:** four
tasks generate inputs and references from code, while 14 tasks require captured
input/output fixtures totaling 33.14 GB across the whole suite. These files are
test data, not model checkpoints. They must be provisioned explicitly and
verified before a dependent task is copied into an evaluation workspace; use
`--task` to install only the selected tasks' files. See
[why this revision needs captures](../../docs/how-to/prepare-head-kernel-artifacts.md#why-this-revision-needs-captured-tensors).

```bash
python3 src/tools/prepare_head_kernel_artifacts.py --list
python3 src/tools/prepare_head_kernel_artifacts.py --mirror artifact_mirror
python3 src/tools/prepare_head_kernel_artifacts.py --verify
```

Run these commands from the repository root. The mirror keeps the original
`<operation-id>/ut/<artifact-name>` layout. The version-2 manifest declares
that location as `mirror_path` separately from each hierarchical destination
`path`; the reviewed flat mirror files remain usable. A content-addressed
cache is also supported.
See [artifact preparation](../../docs/how-to/prepare-head-kernel-artifacts.md)
and the exact [artifact manifest](artifacts.json).

The three MiniMax `timing_geometry.pt` files have recorded hashes and are
available in the current upstream delivery. Earlier suite prose describing
them as missing is stale. A locally absent or mismatched fixture still fails
preflight. Generated random-baseline caches and previous timing reports are
not persistent correctness oracles.

## Select the matching environment

Images are a task property. This suite contains distinct SGLang v0.5.17,
v0.5.18, and Kimi-specific image cohorts; selecting the suite does not make
one image compatible with every task.

Use the [per-kernel environment matrix](../../docs/reference/top5-head-kernel-environments.md)
and the [runtime guide](../../docs/how-to/top5-head-kernels-runtime.md).
For example, from the repository root:

```bash
python3 src/scripts/top5_head_kernels.py plan \
  --config example_configs/top5_validator_sglang_v0518_mi355x.yaml
python3 src/scripts/top5_head_kernels.py run \
  --config example_configs/top5_validator_sglang_v0518_mi355x.yaml
```

The launcher uses the supported Docker workflow and the selected cohort's
image. The task runtime checks architecture and dependency requirements
before correctness or performance. A tag string alone is not proof that the
intended device code executed.

## Benchmark integrity and completion

The candidate must run through the fixed callable contract with the complete
declared cases. Configuration, input geometry, reference data, harness code,
timing policy, and result parsing remain evaluator-owned. Correctness must
cover the actual timed output, including graph replay and mutable state.
Missing cases, failed required graph capture, inconsistent timing methods,
stale reports, changed interfaces, or benchmark-function tampering cannot
produce a valid speedup.

Arena measures the original and modified implementation through the same
protected entrypoint. Only the centralized evaluator writes scored task
results. A kernel source edit that passes the frozen interface and correctness
checks may earn a speedup; a harness edit or an incorrect result may not.

Every registered task requires a fresh, framework-finalized GPU
`validation_report.yaml` with `overall_status: PASS` on its matching runtime.
CPU tests, syntax checks, copied historical reports, partial runs, and skipped
tasks do not satisfy this completion criterion. See the repository's
[task validator contract](../../docs/how-to/task-validator.md).
