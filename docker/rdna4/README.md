# RDNA runtime image (RDNA4 and Radeon PRO W7900)

This recipe supports `gfx1201` and, on the AMD Radeon PRO W7900 and W7900 Dual
Slot, `gfx1100`. Both architectures run the same image. The base image tag lives
in [Dockerfile](Dockerfile); the default local output tag lives in
[`docker_benchmark.sh`](../../src/scripts/docker_benchmark.sh).

From the repository root on the GPU host:

```bash
make docker-smoke
make docker-check-agents CONFIG=example_configs/quickstart_claude_rdna4.yaml
make docker-run CONFIG=example_configs/quickstart_claude_rdna4.yaml
```

On a W7900, use `example_configs/quickstart_claude_w7900.yaml`. On a multi-GPU
host, pin one board with `AKA_VISIBLE_GPU=<ROCr index>`.

Install and authenticate the selected agent separately as described in the
[installation guide](../../docs/install/install.md). The build and smoke steps
do not need an agent or its credentials. `AKA_DOCKER_IMAGE_GFX1201` overrides
both the build output tag and the `gfx1201` runtime selection, and
`AKA_DOCKER_IMAGE_GFX1100` overrides the `gfx1100` runtime selection;
`AKA_DOCKER_IMAGE` remains the global run override.

The runner automatically builds the default image when it is absent from the
Docker daemon, before launching the first container. This applies to smoke,
shell, agent checks, and task runs; parallel runs complete this step during
preflight before starting workers. The first build may download the base image
and locked packages. Later invocations reuse the existing image without checking
for recipe updates. Use `make docker-build-rdna4` to prebuild it or rebuild after
recipe changes. A build failure stops the command before the experiment starts.

Setting either image override disables automatic builds, even when its value
equals the default tag. Custom images use Docker's normal run/pull behavior;
build or publish them separately. To build this recipe under a custom tag, use
`AKA_DOCKER_IMAGE_GFX1201=<tag> make docker-build-rdna4`; on a W7900, also set
`AKA_DOCKER_IMAGE_GFX1100=<tag>` to run it.

## Radeon PRO W7900 board identity

`gfx1100` also names consumer and other workstation boards (RX 7900 XTX, XT and
GRE; Radeon PRO W7800 and W7800 48GB). `target_gpu_model: W7900` therefore
requires every visible GPU to report `gfx1100` **and** the rocminfo marketing
name `AMD Radeon Pro W7900` or `AMD Radeon Pro W7900 Dual Slot` (exact,
case-insensitive). Smoke and preflight print `gpu_marketing_name` and
`gpu_pci_device_id`, and `main.py` writes the observed name and PCI device ID to
`gpu_identity/<role>.json` in the run directory. Any other name, an empty name,
a missing `rocminfo`, or a `gfx1101`/`gfx1102` device is refused before a run
directory is created. A `gfx11` device under any other `target_gpu_model`
(including `RDNA4` and `MI300`) is refused as well. The PCI device ID is
recorded for provenance only; it is not used as identity.

## Runtime layout

The upstream image puts a uv-managed Python virtual environment in
`/opt/python`, with its base interpreter under `/root`. The recipe copies only
that base interpreter and standard library to `/opt/aka-python-runtime` and
updates the virtual environment's interpreter links and `pyvenv.cfg`. Installed
PyTorch, Triton, and ROCm SDK packages remain in their original locations.
The recipe also installs the missing plotting packages from
[`requirements.lock`](requirements.lock), using exact versions and wheel hashes
without dependency upgrades. Other plotting dependencies come from the base
image. Refresh this lock when changing the base or Python version.

Aliases `/opt/venv` and `/opt/rocm` expose the virtual environment and ROCm SDK
through the runner's existing PATH. The build checks Python imports, compiler
and profiler accessibility, and an executable `rocminfo` for the W7900 identity
check as an unprivileged UID, without exposing `/root`.
Experiments still run as the invoking host UID; no runtime root initialization
or host permission changes are required.
The runner supplies `USER` and `LOGNAME` for Python libraries that resolve a
username, since the host UID may have no entry in the image's passwd database.
On `gfx1201` and `gfx1100`, `AITER_ROOT_DIR` and `AITER_JIT_DIR` point to
separate container-local `/tmp` caches, with the runner's worker suffix when
set, so repository and installed-package builds do not write under the
unmounted host home directory. On `gfx1100`, `FLYDSL_RUNTIME_CACHE_DIR` moves
FlyDSL's kernel cache to `/tmp` the same way.

Smoke requires `hipcc`, the Python dependencies, and a visible GPU matching the
selected architecture. It reports `rocprofv3` and `rocprof-compute` availability
separately: core correctness and graph/event timing do not launch either tool.
For a workflow that requires profiling, use `AKA_REQUIRED_PROFILERS=rocprofv3`
(or a comma-separated list) to fail smoke if the requested binary is absent.
Finding a profiler does not establish hardware-counter support or imply that
a task or candidate was profiled; evaluation tools retain their separate gates.

## Agent architecture guidance

`target_gpu_model: RDNA4` selects the
[gfx1201 architecture context](../../src/prompts/cheatsheet/RDNA4_architecture.md)
and the existing HIP/Triton `knowledge_override` entries in
[the cheatsheet map](../../src/prompts/cheatsheet/default_cheatsheet.yaml).
This shared prompt route does not require a GEAK workflow. Other target
languages retain their default guides; selecting one does not qualify its
runtime dependencies on RDNA4.

The guides distinguish hardware WMMA formats from installed compiler support,
SIMD residency from CU/WGP totals, and tracing from hardware counters. Board
capacity and launch limits must come from the actual device. The guide commands
are for diagnostics inside this runtime; use the task's unchanged harness for
scored timing. These prompt corrections do not establish a performance gain.

`target_gpu_model: W7900` selects the
[gfx1100 architecture context](../../src/prompts/cheatsheet/RDNA3_architecture.md)
and the RDNA3 HIP and Triton `knowledge_override` entries in the same map. The
guides describe gfx11 WMMA (FP16, BF16, INT8 and INT4 inputs; no FP8, FP4,
sparse WMMA or MFMA), wave32 launch geometry, the 64 KiB per-workgroup LDS
limit, and hardware counters whose values depend on the operator-set perf level.
The device facts in these guides were checked on a W7900 in an image built from
this base and package lock.

## Validation and limits

Runtime checks on a 16 GB `gfx1201` GPU predate the schema-v3 kernel tasks;
Triton, Gluon, HIP and FlyDSL are verified on MI355X only, and RDNA4 is
implemented but unverified. A passing baseline does not validate a generated
target kernel. Runtime command checks also do not replace a framework-finalized
`qa_report.yaml` from the QA auditor.

On a W7900 Dual Slot (48 GB), smoke passed its identity and numerical checks,
and a scored Claude Code run of the schema-v2 W7900 quickstart accepted both the
HIP GELU and Triton `test_add_kernel` candidates and wrote the board's identity
record. Those single-run speedups are not performance claims. On the same host,
preflight and `main.py` refused replayed W7800, RX 7900 XTX and empty marketing
names, and the runner refused a W7900 config that enables evaluation tools.
These runs predate the schema-v3 kernel tasks; the W7900 is implemented but
unverified for them.

Known limits of existing tasks with this image include:

- Device capacity (16 GB tested): HIP Transpose's original correctness cases
  exceeded VRAM on the tested board. `gfx1201` covers boards with different
  memory capacities; see [AMD GPU specifications](https://rocm.docs.amd.com/en/docs-10.0.0/reference/gpu-specs.html).
  Larger-memory boards, including 32 GB models, have not been validated here.
- Per-workgroup LDS (`gfx1201`): vLLM persistent matmul can request 96 KiB of
  shared memory against the 64 KiB per-workgroup limit. Increasing board VRAM
  does not increase this separate LDS limit.
- Architecture restrictions: the shipped kernel tasks declare `required_arch`
  in `platform_support` (currently `gfx950`) and are skipped on `gfx1201` and
  `gfx1100`. FlyDSL may use APIs absent from the bundled version or require a
  CDNA architecture.
- The schema-v2 task sweep on a W7900 found baselines that fail on `gfx1100`
  for an architecture reason: MFMA, fnuz FP8, MXFP4 or scaled dot, missing
  AITER kernels or configs, wave64 assumptions, 64 KiB LDS, and compiler
  output the gfx1100 assembler rejects. Image-bound tasks need their own
  source images, so this image cannot run them.
- Evaluation-tool sidecars are not qualified here. The host runner refuses them
  on `gfx1201` and `gfx1100`, and `main.py` refuses a W7900 run config that
  enables `evaluation_tools`.

Keep task shapes, correctness tolerances, and timing policy intact when
investigating failures. Honor architecture restrictions, and qualify task
changes with the [QA auditor](../../docs/how-to/qa-auditor.md).
Only `gfx1201` and the W7900 are configured here; this does not establish
support for every task category, other Radeon architectures or `gfx11` boards,
full vLLM serving, or evaluation-tool sidecars. Existing task contracts and
benchmark helpers are unchanged. Agent availability and authentication are
separate preflight checks.

## Security and reproducibility review

External inputs are the upstream image, named by its versioned
`rocm/vllm` tag, and the hash-locked PyPI wheels. Both automatic and explicit
builds send only the Dockerfile, normalizer, and package lock from
`docker/rdna4/` as their context. They install only the locked wheels,
with no source builds, dependency resolution, or agent credentials. The normalizer
refuses to replace existing runtime aliases and leaves `/root` private. The runner
retains its existing host-UID execution, device mounts, privileged-container
policy, and opt-in evaluation-tool policy. These containers remain
reproducibility boundaries, not security sandboxes. Builds use the base image
already in the Docker daemon and pull it only when absent, so a tag that is
later re-pushed does not change an existing host. Record the derived local
image ID with run evidence; the base tag alone does not identify the base
image or changes to this recipe.
