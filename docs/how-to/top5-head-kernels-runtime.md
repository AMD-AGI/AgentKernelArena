# Run the five-workload head-kernel suite

The tasks under `tasks/head_kernels` are isolated operator benchmarks from
five serving workloads. Their ISL 8192 / OSL 1024 / concurrency 64 / TP 8 metadata
describes the serving captures. Each task runs a local operator or one captured
tensor-parallel shard on an MI355X (`gfx950`); these commands do not launch the
five models or measure eight-GPU serving throughput.

The [suite index](../../tasks/head_kernels/README.md) lists **18 separate kernel
tasks** under `<model>/isl8192_osl1024_conc64_tp8_mi355x/<exact-image>/<kernel>`.
Each of the six model/image groups includes `workload.json` and a README with
the exact configuration and kernel list. Every kernel carries `SHAPES.json`,
`SHAPES.md`, its own `config.yaml`, source, frozen cases, and runtime preflight.
The full directory path relative to `tasks/` is its unique selector. Imported
flat identifiers remain stable `headkernel.operation_id` metadata.

The tasks span three capture runtimes. Use one cohort per run:

| Capture cohort | Validation config | Optimization config |
| --- | --- | --- |
| SGLang v0.5.17 | [Validator](../../example_configs/top5_validator_sglang_v0517_mi355x.yaml) | [Claude Code](../../example_configs/top5_claude_sglang_v0517_mi355x.yaml) |
| SGLang v0.5.18 | [Validator](../../example_configs/top5_validator_sglang_v0518_mi355x.yaml) | [Claude Code](../../example_configs/top5_claude_sglang_v0518_mi355x.yaml) |
| Kimi K3 capture build | [Validator](../../example_configs/top5_validator_kimi_k3_mi355x.yaml) | [Claude Code](../../example_configs/top5_claude_kimi_k3_mi355x.yaml) |

The image references live in each task's `headkernel.docker` field. The Kimi
build has a dated, specialized tag; its tag does not establish an SGLang release
number. Do not relabel it as v0.5.18 or replace it with a generic SGLang image.
GLM MoE and elementwise use v0.5.18. GLM's GEMM tasks retain their separately
declared capture images.

The [per-task environment matrix](../reference/top5-head-kernel-environments.md)
lists the image, ROCm/HIP requirement, known package pins, and required imports
for every kernel. It is generated from task metadata using the same contract
reader as the task preflight:

```bash
python3 src/scripts/top5_head_kernels.py matrix
```

The standard Docker runner selects its image through `AKA_DOCKER_IMAGE`; it
does not interpret `headkernel.docker`. The cohort launcher reads the selected
tasks, checks that they agree on exactly one versioned image reference, and
passes that reference to the existing runner. It rejects a conflicting
`AKA_DOCKER_IMAGE` before starting any container.

The launcher enables host-side image verification. The Docker runner inspects
the selected tag once before spawning workers, validates any captured image ID,
and launches all containers in that run by the resulting immutable local Docker
image/config ID. A tag change after inspection cannot replace the image used
for preflight, workers, or aggregation. The exact capture image must already be
available in the host Docker daemon; this path does not pull or rebuild it.

Inspect the exact task set, image, and underlying runner command without Docker:

```bash
CONFIG_PATH=example_configs/top5_validator_sglang_v0518_mi355x.yaml
python3 src/scripts/top5_head_kernels.py plan --config "$CONFIG_PATH"
```

On an allocated MI355X host with access to the capture image and an existing,
authenticated validator backend, check the environment and run validation:

```bash
python3 src/scripts/top5_head_kernels.py preflight --config "$CONFIG_PATH"
python3 src/scripts/top5_head_kernels.py check-agents --config "$CONFIG_PATH"
python3 src/scripts/top5_head_kernels.py run --config "$CONFIG_PATH"
```

The launcher forwards trailing arguments to `docker_benchmark.sh`. Resume and
parallel scheduling use the existing runner interfaces:

```bash
python3 src/scripts/top5_head_kernels.py run --config "$CONFIG_PATH" -- --resume-latest
GPU_IDS=0,1 python3 src/scripts/top5_head_kernels.py parallel-run --config "$CONFIG_PATH"
```

Parallel scheduling gives each worker one GPU and an independent task. The
number of workers does not change the captured TP shard shape. Run each of the
three validation configs separately and require framework-finalized
`validation_report.yaml` files with `overall_status: PASS` before PR submission;
see [task validation](task-validator.md). The launcher performs no registry
login, package installation, patch application, or GPU allocation.

## Runtime and dependency checks

The Docker runner exposes its selected reference as
`AGENT_KERNEL_ARENA_DOCKER_IMAGE`, the resolved local image/config ID as
`AGENT_KERNEL_ARENA_DOCKER_IMAGE_ID`, and registry manifest references, when
available, as `AGENT_KERNEL_ARENA_DOCKER_REPO_DIGESTS`. These values come from
host-side Docker inspection; the runner executes the resolved image ID.
Each task includes a local `scripts/runtime_preflight.py` helper that checks
the selected image, requires the host identity check, compares captured IDs,
and checks the
SGLang release when encoded in the image tag, ROCm/HIP 7.2, actual `gfx950`
device properties, required imports, and the production target callable.
It writes the identity, available registry digests, observed package versions,
and any environment failures to `build/runtime_preflight.json` before benchmark
execution. CPU syntax checks do not establish a usable GPU
runtime. The `preflight` launcher action also runs the existing arena/agent
environment preflight; full task checks occur when task correctness and
performance run.

Capture images supply PyTorch, Triton, AITER, and SGLang. Tasks using FlyDSL or
TileLang require the corresponding package from that same capture image. A
task may declare stricter known facts under `headkernel.runtime` using
`sglang_version`, `torch_version`, `hip_version`, `package_versions` (a mapping
of importable package names to exact versions), `required_modules`, and
`required_model_types`. `expected_image_id` optionally asserts a Docker
image/config ID. The helper also consumes captured IDs directly from the
protected `ut/meta.json` source/baseline provenance. Do not install floating
upgrades into a capture runtime.

The v0.5.18 cohort also requires `TVM_FFI_DISABLE_TORCH_C_DLPACK=1`, as documented
by the capture suite. That image ships a CPU-only TVM FFI addon; enabling the
Torch C DLPack path triggers repeated failing ROCm addon compilations and can
hang startup. The cohort launcher sets the flag, the Docker runner forwards
the explicit value, and task preflight requires and records it. Other runtimes
receive no new default. This is CPU FFI setup and does not change kernel work
or timing iteration counts.

The GLM MoE capture documents PyTorch `2.9.1+rocm7.2.0`, SGLang `0.5.18`, and HIP
`7.2.26015`. Its original model-serving bootstrap required `glm5_next`
architecture enablement. The isolated MoE task now publishes the captured
non-model settings through SGLang's `override_server_args` API and initializes
only a one-rank TP group. That operator path does not load a checkpoint or use
the architecture patch. A full model-serving run still needs independently
qualified `glm5_next` support; the unit benchmark does not establish it.

The five Qwen task metadata files record Docker image/config ID
`sha256:760dd38b9b6f2bd11c13011d470eb8e377c3f0d71284a090a710d64a23bd789f`,
and captured AITER or SGLang commits. That ID is enforced for a run selecting
any of those tasks. It is not a registry manifest digest. Other tasks do not
currently provide a captured image ID, and complete transitive dependency
locks are unavailable. Available registry `RepoDigests` are recorded from the
actual image rather than inferred from its ID or tag. The CPU checks for this
import cannot certify those images, serving architecture support, dependencies,
or GPU results. Preserve the runtime reports and actual registry metadata during
compatible GPU qualification.
The evaluation-tool sidecars qualified against the repository's default
SGLang v0.5.14 scoring image are not qualified for these capture images.
