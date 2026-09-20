# Run the five-workload head-kernel suite

Every task now selects a pinned public `docker.io/rocm/hyperloom` image. Runtime
launches use the local checkout and public Docker Hub; they do not contact the
historical capture registry or require the original compute node. The original
capture images and known Docker image IDs remain explicit historical provenance.
The new public runtime profiles have **GPU qualification pending** and are not
asserted to be identical to those capture images.

The [suite index](../../tasks/head_kernels/README.md) lists 18 operator tasks from
five serving workloads. ISL 8192 / OSL 1024 / concurrency 64 / TP 8 describes the
serving workload. Each task runs a local operator or one tensor-parallel shard
on an MI355X (`gfx950`); these are not eight-GPU model-serving measurements.
Task paths retain their capture-era grouping during the runtime migration.
`config.yaml`, rather than a directory name, selects the current runtime.

| Task cohort | Current public profile | Validator config | Optimization config |
| --- | --- | --- | --- |
| Original v0.5.17 tasks | SGLang v0.5.17 / ROCm 7.2 | [Validator](../../example_configs/top5_validator_sglang_v0517_mi355x.yaml) | [Claude Code](../../example_configs/top5_claude_sglang_v0517_mi355x.yaml) |
| Original v0.5.18 tasks | SGLang v0.5.18 / ROCm 7.2 | [Validator](../../example_configs/top5_validator_sglang_v0518_mi355x.yaml) | [Claude Code](../../example_configs/top5_claude_sglang_v0518_mi355x.yaml) |
| Kimi K3 operators | SGLang v0.5.17 / ROCm 7.2 | [Validator](../../example_configs/top5_validator_kimi_k3_mi355x.yaml) | [Claude Code](../../example_configs/top5_claude_kimi_k3_mi355x.yaml) |

## Pull and run the declared public runtime

On an MI355X host with working Docker/ROCm access, choose a run config and read
its exact image reference without starting a container:

```bash
CONFIG_PATH=example_configs/top5_validator_sglang_v0517_mi355x.yaml
python3 src/scripts/top5_head_kernels.py plan --config "$CONFIG_PATH"
IMAGE=$(python3 src/scripts/top5_head_kernels.py plan --config "$CONFIG_PATH" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["image"])')
docker pull "$IMAGE"
```

The launcher reads each selected task's `headkernel.docker` and
`headkernel.runtime.expected_image_id`. A run must select one manifest and one
Docker image/config ID. It rejects a conflicting `AKA_DOCKER_IMAGE`, an unknown
image substitution, or a mixture of the v0.5.17 and v0.5.18 profiles. The original
v0.5.17 and Kimi cohorts share the same public runtime and may be combined in a
custom run config.

With the configured validator backend installed and authenticated, run:

```bash
python3 src/scripts/top5_head_kernels.py preflight --config "$CONFIG_PATH"
python3 src/scripts/top5_head_kernels.py check-agents --config "$CONFIG_PATH"
python3 src/scripts/top5_head_kernels.py run --config "$CONFIG_PATH"
```

Each task must also satisfy its local input contract before execution. The
runtime launcher does not download model checkpoints or tensor fixtures. See
individual task documentation for its case/input construction and availability.

The standard Docker runner consumes `AKA_DOCKER_IMAGE`; it does not read task
metadata. The cohort launcher passes the declared public manifest to that runner.
The host resolves the local image once, compares its image/config ID with the
current public pin, then launches preflight, workers, and aggregation by that
resolved ID. No mutable tag can substitute different bytes after inspection.
The image must already be available locally; the launcher does not pull or
rebuild it automatically.

## Single-task GLM BF16 validation

The [single-task public validator config](../../example_configs/top5_validator_glm_bf16_public_mi355x.yaml)
uses the same public default as the other v0.5.17 tasks. Its 27 cases generate
synthetic BF16 tensors locally; it needs no captured tensor fixture or model
checkpoint:

```bash
CONFIG_PATH=example_configs/top5_validator_glm_bf16_public_mi355x.yaml
python3 src/scripts/top5_head_kernels.py plan --config "$CONFIG_PATH"
python3 src/scripts/top5_head_kernels.py run --config "$CONFIG_PATH"
```

The former `headkernel_validation_runtime` option is retired. The task's public
manifest and image/config ID are now its normal runtime contract. The original
capture reference remains under `headkernel.capture_runtime`. All 27 shapes,
references, source files, task-specific source/ABI checks, and benchmark controls
are preserved by this runtime change.

## Runtime checks and cache isolation

Task preflight requires the host-reported current image/config ID, the declared
SGLang version, ROCm/HIP 7.2, actual `gfx950` device properties, required package
imports, and the production target callable. It writes
`build/runtime_preflight.json`, including the current profile and image,
`qualification_status: pending`, observed package versions, registry RepoDigests,
and the separate historical capture image/ID. Existing task-specific native
source and ABI checks remain required; selecting a public image does not weaken
them. An environment preflight pass does not establish kernel correctness or
GPU benchmark qualification.

All cohorts enable `AKA_TOP5_ISOLATED_CACHES=1`. Each worker receives its own
ordinary user-owned temporary directories through `AITER_JIT_DIR` and
`FLYDSL_RUNTIME_CACHE_DIR`. The runner creates them before Python starts.
Installed caches and their permissions are untouched. First use may require
cold JIT compilation.

The v0.5.18 capture attempt in job 158486 confirmed that AITER's fallback tried
to copy an inaccessible installed FlyDSL cache into the isolated home. At the
public images' AITER revision `d9e5ef7ce08ee7045d583aed768cff41aa9210fe`,
[`get_user_jit_dir`](https://github.com/ROCm/aiter/blob/d9e5ef7ce08ee7045d583aed768cff41aa9210fe/aiter/jit/core.py#L438)
uses `AITER_JIT_DIR` directly, while
[`aiter.__init__`](https://github.com/ROCm/aiter/blob/d9e5ef7ce08ee7045d583aed768cff41aa9210fe/aiter/__init__.py#L68)
honors the FlyDSL cache override. The same isolation policy is configured for
both public profiles; the observed failure was on the old v0.5.18 capture image.
Unrelated custom-runtime defaults remain unchanged without the top-five opt-in.

The v0.5.18 profile also sets and checks `TVM_FFI_DISABLE_TORCH_C_DLPACK=1` to
avoid the documented CPU-only TVM FFI addon's failing ROCm compilation path.
These are runtime setup controls; they do not alter kernel work or timing counts.

## Capture provenance and qualification

`headkernel.docker` and `headkernel.runtime` describe the current public runtime.
`headkernel.capture_runtime` describes the historical serving capture. Protected
`ut/meta.json` records retain their historical image IDs and source revisions.
Those old IDs are checked for provenance consistency and are never substituted
for the new public image/config ID. In particular, Qwen's captured ID
`sha256:760dd38b9b6f2bd11c13011d470eb8e377c3f0d71284a090a710d64a23bd789f`
is historical; the public v0.5.18 profile enforces its own declared ID.

The [environment matrix](../reference/top5-head-kernel-environments.md) shows
current runtime requirements and historical capture identities separately. It
is generated from the same contract reader used by preflight:

```bash
python3 src/scripts/top5_head_kernels.py matrix
```

See the [public-image catalog](../reference/top5-public-images.md) for registry
and source-pin evidence. The Kimi attention pristine source matches public
SGLang v0.5.17, and its MoE entry functions match public AITER d9e5ef7c after
excluding annotation/docstring changes. These source comparisons support
qualification on the public v0.5.17 profile; they are not a GPU pass or proof
that the old specialized Kimi image was identical.

The GLM MoE operator publishes its captured non-model settings using SGLang's
`override_server_args` API and initializes a one-rank TP group. It does not
load a checkpoint or use the old `glm5_next` serving architecture patch. Full
model-serving support remains a separate deployment qualification.

Run each selected task through `task_validator` and require a fresh,
framework-finalized `validation_report.yaml` with `overall_status: PASS` before
claiming qualification. Preserve the runtime reports and GPU environment with
those results. Existing evaluator sidecars qualified for the repository's
v0.5.14 default image are not qualified for these public profiles.

The launcher forwards normal resume and scheduling arguments:

```bash
python3 src/scripts/top5_head_kernels.py run --config "$CONFIG_PATH" -- --resume-latest
GPU_IDS=0,1 python3 src/scripts/top5_head_kernels.py parallel-run --config "$CONFIG_PATH"
```

Each parallel worker gets one GPU and an independent task; worker count does
not change captured tensor-parallel shard shapes. Keep the runtime identity
fixed when resuming an existing run.
