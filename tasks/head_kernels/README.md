# Top-five-model head kernels

This suite updates the older `head_kernels` dataset with captured operator
workloads from MiniMax M3, Kimi K3, DeepSeek V4 Pro, GLM 5.3 Flash, and
Qwen3.8 2.4T. The serving capture context is ISL 8192, OSL 1024,
concurrency 64, and tensor parallelism 8. An isolated task replays an
individual rank's operator; it does not launch an eight-GPU model or
measure model end-to-end throughput.

Read the [per-task workload-fidelity matrix](WORKLOAD_FIDELITY.md) and
[native validation results](VALIDATION.md) before interpreting a score. Some
paths are deliberately blocked because their original serving controls are
missing or reconstructed. The suite does not claim that every retained case
exactly reproduces its original HyperLoom execution.

The [coverage catalog](catalog.json) connects each profiled kernel row to an
optimization task or an explicit remaining capture/implementation requirement.
Several profiler symbols share one callable and therefore one task. Catalog-only
entries have no task configuration and must not be counted as runnable tasks.
The source table's reported GPU-time shares are historical selection evidence;
historical speedups and roofline estimates are not benchmark rewards.

## Run the native-verified subset

The [native-verified index](native_verified.json) identifies tasks that completed
all declared native compile/interface, correctness and performance phases on
MI355X with the recorded public runtime and matching task sources. It records
the source digest, source commit, evidence archive hash, and each required phase
report hash. These results do not claim a framework `task_validator` PASS or
complete original serving equivalence; see [validation status](VALIDATION.md)
and [workload fidelity](WORKLOAD_FIDELITY.md) for those separate limits.

The explicit configs select only indexed tasks and keep public runtimes separate:

| Public runtime | Initial verified tasks | Config |
| --- | --- | --- |
| SGLang v0.5.17 | GLM BF16 GEMM; GLM FP8 blockscale GEMM | [Native-verified v0.5.17](../../example_configs/top5_native_verified_sglang_v0517_mi355x.yaml) |
| SGLang v0.5.18 | Qwen dense BF16 GEMM; Qwen RMSNorm | [Native-verified v0.5.18](../../example_configs/top5_native_verified_sglang_v0518_mi355x.yaml) |

From the repository root, run the native checks without an agent:

```bash
python3 src/scripts/top5_head_kernels.py verify \
  --config example_configs/top5_native_verified_sglang_v0517_mi355x.yaml
python3 src/scripts/top5_head_kernels.py verify \
  --config example_configs/top5_native_verified_sglang_v0518_mi355x.yaml
```

Replace `verify` with `plan` to inspect the image and exact selection without
starting Docker. All 18 task directories and the full-cohort configs remain
available; omission from this subset means a complete matching native pass has
not been recorded in the index.

When extending the subset, add the completed native evidence and update the
corresponding explicit config. The source digest is SHA-256 over canonical JSON
of the native verifier's `source_identity.files`, after canonical performance
helpers are materialized. The focused index test reproduces that identity from
the current files. A changed task requires matching new evidence before its
record can describe the new source. Raw logs and generated workspaces remain
outside the shipping tree.

## Workload layout and kernel tasks

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

Each capture-image group has a `workload.json` and README listing its exact serving
dimensions, historical capture image, current public runtime, kernel selectors, shapes, and per-kernel runtime
requirements. Each leaf remains a self-contained task with its own source,
frozen contract, runners, and environment preflight. `SHAPES.json` preserves
the complete case inventories and source evidence; `SHAPES.md` presents the
tensor shapes, dtypes, physical layouts, and explicitly unknown metadata.

| Model | Workload and historical capture image | Tasks | Principal targets |
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

## Portable task inputs

All **18 kernel tasks require zero external tensor fixtures**. Their inputs are
constructed locally from committed structural JSON, deterministic generators,
and task-local metadata. Large numerical buffers use generated samples; each task
states which shapes, physical layouts, routing or paging controls, aliases and
mutable-state rules come from captured evidence. Expected outputs are produced
by an independent mathematical reference or protected original implementation.

A checkout and the declared public runtime are sufficient to construct the task
inputs. No model checkpoint, private registry, shared-NFS mirror, or original
capture node is required. List or check the external-artifact declaration from
the repository root:

```bash
python3 src/tools/prepare_head_kernel_artifacts.py --list
python3 src/tools/prepare_head_kernel_artifacts.py --verify
```

Both report **0 required fixtures and 0 required bytes**, with all 18 tasks
listed as requiring no persistent fixture. These commands check the external
artifact contract; the task runners separately verify committed input metadata,
generators, references and case definitions before executing candidates.

Original archive hashes remain optional provenance in [artifacts.json](artifacts.json)
and each task's metadata. See [portable inputs and optional archives](../../docs/how-to/prepare-head-kernel-artifacts.md)
for their role, and the [offline extraction guide](../../docs/how-to/extract-head-kernel-contracts.md)
for developer-only archive inspection.

Input portability does not establish exact-workload fidelity or GPU
qualification. Tasks retain their explicit scoring gates where required serving
controls or native runtime behavior have not been qualified. The current
[validation status](VALIDATION.md) records those limits.

## Select the matching environment

Images are a task property. All 18 tasks now select pinned public
`docker.io/rocm/hyperloom` images: 11 use SGLang v0.5.17, including Kimi, and
seven use v0.5.18. The original capture images remain historical provenance;
they are not contacted during execution. Directory groups identify the serving
capture, while `config.yaml` selects the current public runtime. A run must
select tasks sharing one public image.

Use the [per-kernel environment matrix](../../docs/reference/top5-head-kernel-environments.md)
and the [runtime guide](../../docs/how-to/top5-head-kernels-runtime.md).
The [public image catalog](../../docs/reference/top5-public-images.md) records
Docker Hub manifests and build pins. Historical private image names remain
provenance and are not contacted by the runtime.
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
