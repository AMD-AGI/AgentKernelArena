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

Run the native compile, correctness and performance commands without an agent:

```bash
python3 src/scripts/top5_head_kernels.py verify --config "$CONFIG_PATH"
```

With the configured validator backend installed and authenticated, run:

```bash
python3 src/scripts/top5_head_kernels.py preflight --config "$CONFIG_PATH"
python3 src/scripts/top5_head_kernels.py check-agents --config "$CONFIG_PATH"
python3 src/scripts/top5_head_kernels.py run --config "$CONFIG_PATH"
```

All 18 tasks construct their inputs from the checkout and require zero external
tensor fixtures or model checkpoints. Each task must still satisfy its local
input and integrity contract before execution; see [portable input preparation](prepare-head-kernel-artifacts.md).
The runtime launcher performs no task-data downloads.

The standard Docker runner consumes `AKA_DOCKER_IMAGE`; it does not read task
metadata. The cohort launcher passes the declared public manifest to that runner.
The host resolves the local image once, verifies its manifest/config identity
against the public pins, then launches preflight, workers, and aggregation by
the resolved engine ID. No mutable tag can substitute different bytes after inspection.
The image must already be available locally; the launcher does not pull or
rebuild it automatically.

Docker engine image IDs and OCI config digests are recorded separately. The
`expected_image_id` task field pins the OCI config digest. With Docker's classic
store, `.Id` equals that digest. Docker's containerd store can instead report the
manifest digest, as observed with Docker 29.8.0. That form is accepted only when
`.Id` equals the selected pinned manifest, `RepoDigests` identifies the same
repository/manifest, `Descriptor` agrees on its digest/type/size, and the exact
committed manifest bytes hash to that pin and reference the expected config.
There is no arbitrary alternate-ID fallback and neither pin is changed.

The runner launches by the raw engine ID and exports it as
`AGENT_KERNEL_ARENA_DOCKER_IMAGE_ID`. It separately exports
`AGENT_KERNEL_ARENA_DOCKER_CONFIG_DIGEST`, `AGENT_KERNEL_ARENA_DOCKER_IMAGE_ID_ROLE`,
and the checked `AGENT_KERNEL_ARENA_DOCKER_IDENTITY` evidence. Both task preflight
and direct verification enforce the verified config pin. These checks use local
Docker inspection and committed public manifest metadata; no extra registry
request is required after the image has been pulled.

## Direct verification without an agent

For an interrupted single-task run with an authentic, fully completed compile
and correctness prefix, the optional [explicit prefix-resume route](native-verifier-prefix-resume.md)
can retain that pinned evidence and run the unchanged full performance command
in a fresh workspace. Source/runtime mismatches are rejected; reused phases are
identified as prior execution and performance is never reused.

### Parallel native verification

Use `parallel-verify` with explicit host GPU IDs to distribute one public-image
cohort across ordinary verifier containers:

```bash
GPU_IDS=0,1,2,3,4,5,6,7 \
  python3 src/scripts/top5_head_kernels.py parallel-verify \
  --config example_configs/top5_validator_sglang_v0518_mi355x.yaml
```

For the complete 18-task native matrix, the following exact-selector configs
cover every task once across two separate image runs:

```bash
# Eleven tasks, eight workers: the v0.5.17 cohort plus Kimi.
GPU_IDS=0,1,2,3,4,5,6,7 python3 src/scripts/top5_head_kernels.py parallel-verify \
  --config example_configs/top5_parallel_verify_public_v0517_mi355x.yaml

# Seven tasks, seven workers; the eighth requested GPU remains idle.
GPU_IDS=0,1,2,3,4,5,6,7 python3 src/scripts/top5_head_kernels.py parallel-verify \
  --config example_configs/top5_parallel_verify_public_v0518_mi355x.yaml
```

Use the physical GPU indexes actually assigned to the caller. The runner sets
the corresponding per-worker physical mask and logical GPU 0. Pull each config's
pinned image once before launching its workers. Use the repository's image
identity check: an outer driver must not compare Docker `.Id` directly with the
config digest, because supported containerd engines report the manifest ID.
Collect the entire `workspace_parallel_verification_*` tree, including partial
worker output if an external job deadline interrupts the command. Determine the
expected worker count from `parallel-plan.json`, not the allocation size, and
treat a missing final aggregate as incomplete verification.

The complete checks are not guaranteed to fit a 12-minute allocation. The GPU
qualification campaign hit that limit during several full correctness suites,
leaving later tasks in the same shard unstarted. Use the task's declared phase
timeouts when budgeting a full run, or select fewer tasks for a bounded attempt.
An external deadline does not reduce the mandatory cases, warmups, or samples;
interrupted work remains incomplete. See the [recorded outcomes](../../tasks/head_kernels/VALIDATION.md).

Tasks are assigned deterministically by position (`tasks[index::worker_count]`)
with no duplicates. Each active worker sees its selected physical GPU through
`ROCR_VISIBLE_DEVICES` and uses logical GPU 0 inside the container. The image
identity is verified once before workers start; each worker has separate home,
JIT/cache directories, and output paths. No agent CLI or authentication state
is mounted. GPU IDs must be distinct nonnegative indexes; there is no automatic
GPU discovery for this command. When GPUs outnumber tasks, the surplus devices
are recorded as unused and no empty workers are launched.

The command creates `workspace_parallel_verification_<unique>/` containing the
declared plan, per-worker stdout/stderr, and a separate `worker-NNN/` directory
with each worker's normal `direct-verification.json`, task workspaces, and all
retained native reports. The final `parallel-verification.json` checks worker
exit status, native task status, image/GPU/shard identity, and complete exactly-once
task coverage. Missing, failed, duplicated, or incomplete worker evidence makes
the overall command fail. A task-list digest prevents workers from silently
using a changed cohort selection. Existing output directories are never reused.

The low-level verifier also accepts `--shard-index`, `--shard-count`,
`--taskset-sha256`, and `--output-directory`; the host parallel route owns these
arguments. Ordinary `verify` remains a single full-cohort run by default.
Parallel execution preserves each task's commands, phase timeouts, correctness
checks, case set, and benchmark controls. Successful native execution is still
not a framework-finalized task-validator PASS. Select a config containing only
one runtime; the v0.5.17 and Kimi cohorts may share a combined config because
they use the same public image, while v0.5.18 must run separately.

For optional sampled baseline GPU symbols and launch metadata, use the separate
[device-trace diagnostic](head-kernel-device-trace.md). It profiles the existing
baseline/scored-case path and records its coverage and limits without changing
scoring or replacing correctness validation.

The `verify` action runs each selected task's declared compile, correctness,
and performance commands in order through the standard Docker runner. It uses
the full task case set, configured phase timeouts, and canonical benchmark
helpers. A failing command, timeout, missing native report, or native report
whose status is not `ok` makes the command exit nonzero. Later phases of that
task are skipped. Multiple selected tasks run sequentially; `AKA_VISIBLE_GPU`
can select one GPU using the runner's existing device visibility convention.

Each invocation creates a unique `workspace_direct_verification_*` directory
inside the repository. It contains `direct-verification.json`, per-task
`.direct.json` evidence, copied task workspaces, command stdout/stderr, and
snapshots of native reports in `direct-native-reports/`. Image/config identity,
available registry digests, runtime choice, source hashes, symlink targets,
phase return codes, and timeout status are retained. Internal source aliases
are preserved and checked before and after copying. Every ordinary file,
including any archived tensor fixtures, is copied independently so task
workspaces do not share mutable file inodes with their source. Existing source
tasks are unchanged.

These are direct execution results: `framework_task_validator` is `NOT_RUN`,
and no framework `validation_report.yaml`, `task_result.yaml`, or finalized
`PASS` is created. The compile phase checks Python syntax and fixed ABI;
correctness and performance supply device execution evidence. The public
runtime remains unqualified until it successfully completes GPU checks.
For the separate framework quality review and its PR gate, use the existing
`run` action with a configured validator backend as described in
[task validation](task-validator.md).


## Single-task GLM BF16 validation

The [single-task public validator config](../../example_configs/top5_validator_glm_bf16_public_mi355x.yaml)
uses the same public default as the other v0.5.17 tasks. Its 27 cases generate
synthetic BF16 tensors locally; it needs no captured tensor fixture or model
checkpoint:

```bash
CONFIG_PATH=example_configs/top5_validator_glm_bf16_public_mi355x.yaml
python3 src/scripts/top5_head_kernels.py plan --config "$CONFIG_PATH"
python3 src/scripts/top5_head_kernels.py verify --config "$CONFIG_PATH"
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

The same opt-in provides a separate container-owned tmpfs at
`/tmp/aiter_configs`, with the ordinary container user's UID/GID. AITER's native
configuration merger reads its default CSV and model-specific CSV inputs, then
writes the merged output and its lock at this hard-coded path. A fresh writable
output mount preserves that selection and merge behavior; the runner does not
set `AITER_CONFIG_*` tuning selectors or modify host/image permissions. This
also preserves the existing v0.5.14 runtime scratch behavior without duplicate
mounts when both settings apply.

Public v0.5.17 attempt 158739 verified the manifest/config identity chain and
built AITER's core JIT extension, then stopped before any correctness case on
`bf16_tuned_gemm.csv.lock` permissions. The tmpfs addresses that scratch-output
failure. That attempt observed Torch `2.9.1+rocm7.2.0.git7e1940d4`, HIP
`7.2.26015-fc0010cf6a`, Triton `3.6.0`, and SGLang `0.5.17`; it did not establish
kernel correctness or benchmark qualification.

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

## Worker preflight and source-overlay order

The common worker uses two preflight phases so native imports cannot cache a
stock target before an optimization overlay is installed:

1. `require_runtime(config, phase="environment")` validates the selected image,
   required environment and PyTorch/GPU properties. It defers native packages,
   model-architecture resolution and target imports.
2. The worker preloads declared trusted helper aliases and installs the integrity
   monitor over those actual objects, including the preflight implementation.
3. It rejects any overlay module already cached by a preload helper, then installs
   the source overlay.
4. `require_runtime(config)` performs complete native package/version/model/target
   validation under the monitor. The same attested entrypoint then executes.

Environment-only results are written to `runtime_preflight_environment.json` and
carry `native_resolution_complete: false`. Complete results remain in
`runtime_preflight.json`, with the resolved target's module file. An environment
phase alone cannot finalize a worker or establish runtime qualification.

Availability stubs in CPU worker tests must accept the `phase` keyword; production
code must provide both phases. The protocol is shared across tasks and does not
change kernel ABIs, tensor contracts, source identity checks or scoring policy.
