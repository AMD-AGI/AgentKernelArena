# MI355X runtime upgrade qualification

This record accompanies the task-schema refactor. It qualifies runtime changes
separately from task validation and agent optimization. A tag existing in a
registry, an import succeeding, or a runtime smoke passing does not establish
correctness or performance for all Arena tasks.

## Decision

Retain the verified SGLang 0.5.14 / ROCm 7.2 runtime for `gfx950`. The default in
[`docker_benchmark.sh`](../../src/scripts/docker_benchmark.sh) now selects its
immutable manifest reference, rather than its movable dated tag. This preserves
the existing software stack, cache handling, and evaluation-tool image gate.
Explicit `AKA_DOCKER_IMAGE` and per-architecture overrides remain available.

The two ROCm 10 candidates below are **not approved as default scoring images**.
Their runtime and GPU evidence is recorded separately below. The `gfx942` and
`gfx1201` defaults are unchanged. An MI355X qualification cannot establish
support on those other architectures.

The explicit SGLang 0.5.19 candidate now receives the same private AITER/FlyDSL
cache paths and writable AITER config directory as the existing gfx950 runtime.
GPU job 139224 exposed 3,795 root-owned, unreadable bundled FlyDSL cache files
when AITER used its default copy-on-import path. Setting `AITER_JIT_DIR` selects
its writable-cache path instead. The override is limited to the recorded image
references and isolated per worker; installed image packages remain unchanged.
Docker argument tests cover the dated tag, immutable digest, and worker suffix.
This cache fix needs a fresh GPU task validation; it neither promotes the default
image nor changes the separately pinned evaluation-tool compatibility gate.

## Registry evidence (2026-09-15 UTC)

The Docker Registry v2 API returned single-platform `linux/amd64` manifests for
all three references. SHA256 of the downloaded manifest bytes matched the
registry's `Docker-Content-Digest`. The image-config blob digest is distinct
from the pullable manifest digest. Record Docker's locally resolved `.Id`
separately: the observed Docker 29.7.2 daemon reports the manifest digest as
`.Id`, rather than the config-blob digest.

Repository for every reference: `lmsysorg/sglang-rocm`.

| Role | Dated tag | Manifest digest | Config-blob digest | Compressed layers |
| --- | --- | --- | --- | --- |
| Existing stack | `v0.5.14-rocm720-mi35x-20260705` | `sha256:b435b508b5aa696abb25c909341ce73e41574c4271cf716bed72418dcea86b78` | `sha256:0a78d51f2f1db80a1abfe23350fc2e5733ac5acb1528d6dc7ce3679bdb099aff` | 27,994,781,865 bytes |
| New candidate | `v0.5.19-rocm10-mi35x-20260913` | `sha256:106a7adbeec5554b6e66a4bda0b3694af442717b9fe92754a9885520077b6f93` | `sha256:83a0d6661fed28e45e241ef1c24f619b6aaa4f94dd179ede4e06171d0be0c4f7` | 13,699,345,131 bytes |
| Earlier candidate | `v0.5.18-rocm10-mi35x-20260904` | `sha256:916da507975f74e0a4c869bd7f5dceebf8838b2d065835bc7574032330ef4910` | `sha256:6f0335cb633fc0689bd229d4906647b5a09853f1bd9ccbb70d9337f5f15e71f2` | 13,525,716,969 bytes |

The upstream [SGLang v0.5.19 release](https://github.com/sgl-project/sglang/releases/tag/v0.5.19)
announces ROCm 10 images. The [registry](https://hub.docker.com/r/lmsysorg/sglang-rocm/tags)
and [upstream Docker recipe](https://github.com/sgl-project/sglang/blob/main/docker/rocm.Dockerfile)
provide context; the manifest and config blobs above establish the particular
images examined here. A live upstream recipe can change after qualification.

### Build metadata versus installed versions

The downloaded config histories describe these builds:

| Property | Existing stack | New candidate | Earlier candidate |
| --- | --- | --- | --- |
| Ubuntu label | 22.04 | 24.04 | 24.04 |
| Python build selection | 3.10 | 3.12 | 3.12 |
| ROCm build selection | 7.2 | SDK 10.0.0 | SDK 10.0.0 |
| PyTorch build selection | 2.9.1 | 2.11.0 | 2.11.0 |
| Triton ROCm 10 build selection | Not applicable | `3.8.0+git4cff872c` | `3.8.0+git4cff872c` |
| SGLang generated package version | `0.5.14.dev20260705+g3ea875fef4` | `0.5.19.dev20260913+g14b647cf27` | `0.5.18.dev20260904+g978cc228ca` |
| AITER checkout | `9127c94a18e4398e1eba91f6639e910f0994ad02` | `4ad99832823dde2315b361cbd3b54b1c5c12acd5` | `c16d44b93a528b2a4bfd6d8d3409116d465872a9` |

These are **build metadata**, not a substitute for importing the installed
packages. The daily images contain development versions and additional AITER
patches, so neither the SGLang tag prefix nor the AITER checkout alone identifies
the complete installed implementation. Preserve the image digest as well.

The candidates put the ROCm SDK beneath
`/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel` and create `/opt/rocm`
as a compatibility symlink. They retain `/opt/venv`. Actual compiler execution,
non-root cache permissions, profiler availability, and the FlyDSL APIs used by
tasks still require runtime checks; the presence of a symlink is insufficient.

## Observed container compatibility

CPU-only, non-root inspection of the existing stack on node 071 confirmed
Python `3.10.12`, PyTorch module `2.9.1+rocm7.2.0.git7e1940d4` (distribution
`2.9.1+rocm7.2.0.lw.git7e1940d4`), Triton module `3.6.0` (distribution
`3.6.0+git42270451`), FlyDSL `0.2.2`, and AITER distribution metadata
`0.1.17.dev110+g9127c94a1`. Its `rocprof-compute` executable is present.
Its AITER import also requires a GPU and fails in CPU-only inspection.

CPU-only, non-root inspection of the new candidate on the same node confirmed:

- Python `3.12.3`, PyTorch `2.11.0+rocm10.0.0`, Triton module `3.8.0`
  (distribution `3.8.0+git4cff872c.rocm10.0.0`), FlyDSL `0.3.2`.
- SGLang `0.5.19.dev20260913+g14b647cf27`; AITER distribution metadata
  `0.1.21.dev48+g4ad998328.d20260913`.
- `pytest`, PyYAML, NumPy, pandas, SciPy and jsonschema import successfully.
  Arena evaluator/performance/preprocessing modules and the unchanged
  `fp8_gemm_4wave_kernel` candidate module also import successfully.
  Importing that FlyDSL module does not compile or execute its GPU kernel.
- `hipcc`, `rocminfo`, `rocprofv3` and `amd-smi` are available through
  `/opt/venv/bin`; `/opt/rocm` resolves to the SDK directory. `hipcc --version`
  executes successfully and reports AMD clang `23.0.0git` at LLVM commit
  `8f497e0992fb7513f7f78a6f6b6f1056c375e961`.
- **`rocprof-compute` is missing.** In the initial inspection, the then-current
  `_container_smoke` entrypoint returned exit 1 with
  `missing command: rocprof-compute`, before reaching the GPU check. The bounded
  follow-up audits whether this is a core dependency and introduces an explicit
  [profiler capability policy](#profiler-capability-policy); the initial failure
  remains part of the evidence.
- AITER import requests GPU architecture using `rocminfo` and fails with no GPU
  mounted. This CPU result does not establish an AITER compatibility failure
  on a GPU, and is not a successful import qualification.
- `torch.version.hip` reports `7.15.26333`, despite the distribution and ROCm
  SDK version being `10.0.0`. Preserve both rather than inferring one from the
  other. GPU device count is zero by design in this CPU-only check.

The unchanged HIP `TanH` task compilation command completed successfully in
18.44 seconds with `PYTORCH_ROCM_ARCH=gfx950` and `MAX_JOBS=2`, producing the
compiled extension and a successful `build/compile_report.json` in a materialized
copy. This is a real host-side HIP extension compilation, **not GPU execution**;
no device was mounted, and no correctness or timing claim follows from it.

The earlier ROCm 10 image was also pulled and inspected as the host UID. It
contains Python `3.12.3`, PyTorch `2.11.0+rocm10.0.0`, Triton distribution
`3.8.0+git4cff872c.rocm10.0.0`, **FlyDSL `0.3.1`**, SGLang
`0.5.18.dev20260904+g978cc228ca`, and AITER distribution metadata
`0.1.19.post3.dev139+gc16d44b93.d20260904`. The report dependencies import, but
`rocprof-compute` is missing here too. AITER again requires a GPU. Thus the
earlier candidate does not avoid the profiler gap; its different FlyDSL/AITER
versions need their own GPU qualification. The actual task compilation and
492-test selection above were run on the newer candidate only.

The local daemon reports image ID
`sha256:106a7adbeec5554b6e66a4bda0b3694af442717b9fe92754a9885520077b6f93`
for the new candidate, matching its manifest digest. Raw image inspection is
preserved alongside the separate config-blob digest above.

## Qualification scope and artifacts

Artifacts belong under `logs/runtime-qualification/` in the worker checkout and
are intentionally not committed. They include:

- Exact registry manifest/config bytes and compact summaries.
- Slurm submission IDs, authoritative job-status snapshots, stdout and stderr.
- Image pull logs, local image inspection, package/import inventories.
- A GPU probe script that materializes task copies, leaving committed tasks
  unchanged, then executes their existing compile/correctness/performance
  commands without changing cases, tolerances, timing, or result parsing.

The initial, undispatched GPU probe selected HIP `TanH`, Triton `test_kernel_sub`,
FlyDSL `fp8_gemm_4wave_kernel`, SIKL `gemm_a16w16_nt_n32_k6144`, and SIKL
`mxfp4_moe_e65_i1024`. The SIKL stubs exercise production baseline behavior;
passing those paths would not establish a completed candidate. A per-command
180-second qualification cap is recorded as a timeout, never a successful task
validation. Full validator runs must honor the task's actual declared limits.

The CPU probes explicitly set writable temporary caches. They do not establish
that the candidate image works with every default cache path in an Arena agent
run; verify that behavior through the actual Docker runner before promotion.

No agent authentication or shared Hyperloom/GEAK installations are needed for
these probes. The registry pull uses public images; credentials are not written
to evidence. Probe containers bind the checkout read-only and only their log
and copied-task tree writable. The GPU probe requires an assigned GPU and sets its visibility mask; this is
a reproducibility boundary, not a security sandbox. CPU inspection uses no GPU
mounts or privileged-container flag. Instrumentation sidecars are not enabled.

## Slurm / Spur observations

The login host provides Spur 0.11.0 Slurm-compatible commands. Where the
`scontrol` and `sacctmgr` aliases are absent, use `spur show` and `spur accounts`.
Both requested QoS names exist: `amd-aicos-qos` has a configured group limit of
7 nodes and `amd-oai-qos` 8 nodes. These are limits, not free-node counts.
Snapshot availability must be checked again before the optimization campaign.

The initial attempts were serialized, with each failed pending job explicitly
cancelled before its replacement:

| Job | Request | Observed result |
| --- | --- | --- |
| 138832 | One typed MI355X GPU, amd-aicos | Launch confirmation failed: GPU/resource allocation mismatch |
| 138842 | One untyped GPU, amd-aicos | Same explicit launch failure |
| 138849 | One GPU pinned to idle node 214 | Same explicit launch failure |
| 138854 | Short exclusive node 214, no explicit GPU count | Same explicit launch failure; batch script never started |
| 138866 | One GPU pinned to mixed-use node 071 | Same explicit launch failure |
| 138870 | Two CPUs / 16 GB, nonexclusive node 071 | CPU inspection completed for all three images; no GPU reservation or device use |

The failed jobs never reached the batch script and produced no container
results. Their `spur show job` records report
`JobLaunchFailure (dispatch confirmation failed (0/1 confirmed): 1 gpu/resource allocation mismatch)`.
This is a scheduler/worker launch failure, not an Arena task failure or a Docker
image failure. Retrying because a polling request expires would be incorrect;
these replacements were based on explicit controller failures.

A CPU job starting on node 071 while GPU requests fail isolates the problem to
the resource/dispatch path. In the inspected upstream Spur source
([controller classification](https://github.com/ROCm/spur/blob/61de615cccdaad77f71b3566c3b2751fc5dac696/crates/spurctld/src/scheduler_loop.rs#L1016),
[node allocation check](https://github.com/ROCm/spur/blob/61de615cccdaad77f71b3566c3b2751fc5dac696/crates/spurd/src/agent_server.rs#L4334)),
this label means the controller's resource allocation is unavailable in the
node's local allocation table. This supports a controller/node state mismatch
as the diagnosis; the installed client build is `11e7fa6b`, so this source
inspection is not a claim that the deployed node binary was audited. It does not authorize using unallocated GPUs from a
CPU job. The campaign should not proceed by bypassing GPU accounting.

### Bounded GPU follow-up

Further observations on 2026-09-15 change the scheduler conclusion: **real GPU
allocations and Docker computation work on node `crsuse2-m2m-213`, including a
nonexclusive single-GPU allocation**. The earlier failures are not evidence of
cluster-wide GPU unavailability. Follow-up artifacts are preserved separately
under `logs/spur-followup/`, with one directory per started job.

The installed client is `0.11.0 (11e7fa6b)`. Its help, `spur show config`, current
node/resource views, user associations, and QoS limits were inspected read-only.
Both authorized account/QoS pairs are usable by this user. The runtime's existing
typed `-G mi355x:<count>` resource syntax is supported; no Slurm runner change is
justified by the observations.

| Job | Request | Observation |
| --- | --- | --- |
| 138913 | `amd-oai` / `amd-oai-qos`, mixed node 219, `--gres gpu:mi355x:1`, 8 CPUs / 32 GB / 15 min | `PENDING`, `QOSGrpNodeLimit`; never dispatched, then explicitly cancelled. This is quota admission, not a GPU launch failure. |
| 138914 | `amd-aicos` / `amd-aicos-qos`, idle node 213, `-G mi355x:8 --exclusive`, 16 CPUs / 64 GB / 15 min | Started and allocated all eight GPU IDs. Custom non-root probe omitted the host numeric `render` group and failed HIP initialization. Failed after 2m16s; no task correctness result. |
| 138917 | Same node/account and eight GPUs, corrected container groups, 10 min | Completed in 54s. All eight allocated GPUs executed checked tensors; actual Arena smoke and the two legacy representative task paths passed. |
| 138918 | Same node/account, `-G mi355x:1`, nonexclusive, 16 CPUs / 64 GB / 15 min, newer pinned image | Completed in 3m34s including image pull. GPU ID `0`, actual Arena smoke, checked computation and three legacy task paths passed. |

At most two requests were live at once. The unsuccessful OAI request was
cancelled rather than repeatedly resubmitted under an unchanged quota. Whole-node
jobs exited when their probes ended. No node was reset, no controller or shared
installation changed, and no GPU was accessed from a CPU allocation.

All four follow-up jobs are terminal; no GPU reservation is retained. The final
queue snapshot and per-job status are preserved in `logs/spur-followup/`.

Node 213 was newly idle (`CPUAlloc=0`, eight advertised MI355X devices). The first
successful request changed both the node and explicit GPU count, so that attempt
alone could not establish which difference mattered. Job 138918 subsequently
demonstrates that one typed GPU works there too. The failed earlier typed request
and successful new typed request do **not** establish a CLI syntax fix; node and
allocation state changed. Container group membership fixes the separate probe
permission error, not controller dispatch. The normal Arena Docker runner already
adds the correct host numeric `render` and `video` groups.

Job 138918 records `ReqTRES=cpu=16,mem=65536M,node=1,gres/gpu=1` and
`ReqGPUs=1`. Its allocated ID `0` is passed as `ROCR_VISIBLE_DEVICES=0`; the
container observes exactly one GPU and successfully computes and compares a
tensor result. The device name is `AMD Instinct MI355X`, architecture
`gfx950:sramecc+:xnack-`. Device-file presence or `device_count` alone would not
have established this result.

An administrator investigating the earlier dispatch failures has this bounded
evidence: jobs 138849/138854 on node 214 and 138866 on node 071 reported
`dispatch confirmation failed (0/1 confirmed): 1 gpu/resource allocation mismatch`,
before any batch script. CPU-only job 138870 ran on 071; GPU jobs 138917/138918
ran on 213. Compare the failed jobs' controller GPU assignments with those nodes'
local allocation owners and deployed binary versions. No request to reset a
node follows from these observations. OAI job 138913 instead stopped at its
configured eight-node group limit and needs quota availability, not the same
dispatch investigation.

### Legacy representative GPU scope

These probes use task files from parent base `d0d8122e`, with canonical benchmark
helpers materialized into copies. They do **not** exercise the parent's subsequent
SIKL v2 migration (`d88c9c55`), root runtime protocol integration, full
`task_validator`, or any agent campaign. They never write an authoritative
`validation_report.yaml` or claim a full validation pass.

Job 138917 uses the existing immutable image from the registry table. It passes
the actual Docker runner smoke, checked tensor computation on all eight assigned
GPUs, and these unchanged legacy task entrypoints on logical GPU 0:

| Task | Compile | Correctness | Performance |
| --- | --- | --- | --- |
| `triton2triton/geak_eval/L1/fused_append_shared_experts` | PASS | PASS | PASS |
| `SIKL-task/gemm_a16w16_nt_n32_k6144` | PASS | PASS, 13 cases | PASS, 13 cases |

Job 138918 uses the new candidate manifest
`sha256:106a7adbeec5554b6e66a4bda0b3694af442717b9fe92754a9885520077b6f93`.
The local Docker image ID matches that manifest. The actual runner smoke reports
Python `3.12.3`, PyTorch `2.11.0+rocm10.0.0`, Triton `3.8.0`, FlyDSL `0.3.2`,
`hipcc` and `rocprofv3` in `/opt/venv/bin`, and `rocprof-compute=optional-missing`.
The one allocated GPU executes checked tensor work, followed by:

| Task | Compile | Correctness | Performance |
| --- | --- | --- | --- |
| `triton2triton/geak_eval/L1/fused_append_shared_experts` | PASS | PASS | PASS |
| `SIKL-task/gemm_a16w16_nt_n32_k6144` | PASS | PASS, 13 cases | PASS, 13 cases |
| `flydsl2flydsl/fp8_gemm_4wave_kernel` | PASS | PASS, 5 shapes | PASS, 5 shapes |

These results support further qualification of the new image. They do not
establish an improvement in performance: command durations include JIT/setup,
and a matched repeated scoring campaign has not been run. HIP `TanH` was compiled
on the new image in the earlier CPU-only job, but HIP GPU correctness, MoE,
agent integration, sanitizer compatibility and the full migrated suite remain
outside this probe's coverage. The earlier ROCm 10 candidate has no GPU results.

The SIKL stub routes to the production AITER baseline. Its successful execution
validates that baseline path, not a generated FlyDSL candidate. Compile phases
mean the task's declared compile check; JIT compilation may occur in correctness.
The follow-up runs all original cases, with an outer 120-second per-command
probe cap; a cap expiry is recorded as a probe timeout, never a task pass. No
threshold, workload, warmup, timing boundary, or scoring policy is changed.

## Profiler capability policy

The executable audit found no call to `rocprof-compute` in core evaluation or
timing paths. Shared timing uses GPU events/graphs. The old mandatory executable
check was in Docker smoke; `src.preprocessing.check_environment()` also mentions
it but has no callers at this base. Task profiling and agent analysis can still
require it: availability must be declared and tested for those workflows.

Smoke and run preflight now report both `rocprof-compute` and `rocprofv3`, printing
`optional-missing` for an absent optional binary. `hipcc`, required Python imports,
GPU availability and the selected-versus-actual architecture checks remain
mandatory. A workflow that requires a profiler names it explicitly:

```bash
AKA_REQUIRED_PROFILERS=rocprof-compute make docker-smoke
```

The variable also accepts a comma/space-separated list and is forwarded into the
container. Missing requested binaries and unknown names fail preflight; finding
`rocprofv3` cannot satisfy a request for `rocprof-compute`. This tests executable
availability only. It neither qualifies hardware counters nor proves candidate
analysis. Evaluation-tool sidecar image, capability and attestation gates are
unchanged. The default image remains the existing pinned stack.

### Optional ROCm 10 profiler overlay, not installed

The [official Compute Profiler installation source](https://github.com/ROCm/rocm-systems/blob/develop/projects/rocprofiler-compute/docs/install/core-install.rst)
documents the `rocm[profiler]` pip extra. A moving installation example is not an
immutable overlay lock. The candidate's [AMD package index](https://stable.repo.amd.com/rocm/whl-next/rocm-profiler/)
contains `rocm_profiler-10.0.0-py3-none-linux_x86_64.whl`; read-only inspection
confirmed `_rocm_profiler/bin/rocprof-compute` and no `Requires-Dist` entries in
that wheel. The downloaded 64,883,923-byte wheel has SHA256
`ae05423eef63b2c57ea00f2de8f8208c94bdd148baf96f275ad6bd014989326d`.
Separately, `rocm==10.0.0` metadata pins `rocm-sdk-core==10.0.0` and the profiler
extra to `rocm-profiler==10.0.0`. None of this establishes a working installation.

If profiling is required, build a separate derivative of the candidate manifest,
pin every added artifact by version/hash, inspect the complete dependency plan,
and preserve the existing PyTorch/Triton/FlyDSL packages. Verify package consistency,
entrypoint discovery as the host UID, profiler version, and actual counter
collection from a small known kernel on an allocated MI355X. Then rerun unchanged
task checks and qualify any instrumentation independently. Do not install into
shared host environments or silently upgrade scoring packages to satisfy pip.
No overlay was installed or promoted in this qualification.

## Regression checks

The initial runtime patch pins the existing image. The bounded follow-up separates
profiler availability from core smoke and enforces explicitly requested profiler
binaries. Neither alters a harness, tolerance, timing method, agent or task schema.

Follow-up checks:

- `python3 tests/test_rdna4_runtime.py`: **12 tests PASS**, including optional
  profiler inventory, explicit missing/unknown capability rejection, compiler,
  GPU and architecture failures. The host lacks `pytest`, so the attempted
  `python3 -m pytest` invocation failed before collection; this test file's
  standard-library `unittest` entrypoint was run instead.
- `make check-docker-runner`: PASS, including forwarding the explicit profiler
  requirement and the existing cache, agent-selection and sidecar isolation tests.
- `make check-slurm-runner`: PASS; no resource-request code change was necessary.
- `git diff --check`: PASS.
- Real Docker/GPU smoke: PASS on the old pinned image (138917) and newer image
  (138918), with the representative legacy outcomes above. RDNA4 and gfx942
  profiler-policy coverage is mocked; no new hardware qualification is claimed
  for either architecture.

Initial image-pin checks:

- `make check-docker-runner`: PASS before and after the change. The regression
  now asserts that an implicit gfx950 launch uses the manifest digest, keeps
  writable cache configuration, and preserves an explicit dated-tag override.
- `make check-slurm-runner`: PASS.
- `git diff --check`: PASS.
- Inside the new candidate image, as the host UID with no GPU mounts:

  ```bash
  python3 -m pytest -q -p no:cacheprovider \
    tests/test_score.py tests/test_perf_helper_materialization.py \
    tests/test_evaluator_stub_guard.py tests/test_sikl_tasks.py
  ```

  Result: **492 passed, 21 skipped** in 12.88 seconds. The 21 skipped checks
  require `kernelforge.rewrite_by_flydsl.protocol`, which is not installed in
  the base image. This test selection does not execute GPU kernels or the
  refactor's future TaskSpec implementation.

## Promotion gate

Before changing the scoring default to either candidate:

1. Obtain a real compatible GPU reservation. The follow-up establishes that
   node 213 can allocate GPUs; earlier failures on other nodes do not imply
   cluster-wide unavailability. Recheck current resource and QoS availability.
2. Import the framework and each required task dependency as the actual host
   UID, with the same paths and cache behavior used by the Arena runner.
3. Run unchanged representative compile, correctness and device-timing paths
   on old and candidate images. Preserve failures and separate numerical
   baseline discrepancies from runtime incompatibility.
   If the campaign requires a profiler, require its named capability explicitly
   and validate actual profiling, not just executable presence.
4. Complete the refactor's full task-validator and agent optimization campaign
   on the selected stack. A few representative probes do not replace it.
5. Qualify sanitizer sidecars against the selected immutable scoring image
   separately. The current image gate intentionally rejects a different stack;
   do not remove it to make an upgrade pass.
6. Record the promoted manifest digest and resulting installed versions, keep
   an explicit old-image rollback, and compare experiments only when their
   complete runtime identities and timing policies match.

For a deliberate candidate smoke on an allocated MI355X, use an immutable
override (this does not promote the default):

```bash
AKA_DOCKER_IMAGE=lmsysorg/sglang-rocm@sha256:106a7adbeec5554b6e66a4bda0b3694af442717b9fe92754a9885520077b6f93 \
  make docker-smoke
```

See the [task authoring contract](../how-to/add-task.md),
[Slurm runner guide](../how-to/slurm-run.md), and
[evaluation-tool policy](../how-to/use-evaluation-tools.md) for the existing
validation, resource-isolation, and instrumentation requirements.
