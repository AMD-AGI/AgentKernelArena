# RDNA3 runtime image (Radeon PRO W7900)

This recipe supports `gfx1100` on the AMD Radeon PRO W7900 and W7900 Dual Slot.
The base image and immutable digest live in [Dockerfile](Dockerfile); the
default local output tag lives in
[`docker_benchmark.sh`](../../src/scripts/docker_benchmark.sh). The base, the
interpreter normalizer, and the package lock are the same as
[`docker/rdna4`](../rdna4/README.md); the copies here are kept byte-identical by
a test.

From the repository root on the GPU host:

```bash
make docker-smoke
make docker-check-agents CONFIG=example_configs/quickstart_claude_w7900.yaml
make docker-run CONFIG=example_configs/quickstart_claude_w7900.yaml
```

On a multi-GPU host, pin one board with `AKA_VISIBLE_GPU=<ROCr index>`.
Install and authenticate the selected agent separately as described in the
[installation guide](../../docs/install/install.md). The build and smoke steps
do not need an agent or its credentials. `AKA_DOCKER_IMAGE_GFX1100` overrides
both the build output tag and the architecture-specific runtime selection;
`AKA_DOCKER_IMAGE` remains the global run override.

The runner builds the default image when it is absent from the Docker daemon,
before launching the first container, exactly as for RDNA4. The first build may
download the base image and locked packages; later invocations reuse the image
without checking for recipe updates. Use `make docker-build-rdna3` to prebuild
it or rebuild after recipe changes. Setting either image override disables
automatic builds, even when its value equals the default tag. To build this
recipe under a custom tag, use
`AKA_DOCKER_IMAGE_GFX1100=<tag> make docker-build-rdna3`.

## Board identity

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

The layout matches the RDNA4 image: the uv-managed interpreter is copied to
`/opt/aka-python-runtime`, `/opt/venv` and `/opt/rocm` alias the virtual
environment and ROCm SDK, and the plotting packages in
[`requirements.lock`](requirements.lock) are installed by exact version and
wheel hash. The build checks Python imports, `hipcc`, `rocprofv3`, and an
executable `rocminfo` as an unprivileged UID. Experiments run as the invoking
host UID. On `gfx1100`, `AITER_ROOT_DIR` and `AITER_JIT_DIR` point to separate
container-local `/tmp` caches, and `FLYDSL_RUNTIME_CACHE_DIR` moves FlyDSL's
kernel cache out of the unwritable host-home path, each with the runner's
worker suffix when set.

## Agent architecture guidance

`target_gpu_model: W7900` selects the
[gfx1100 architecture context](../../src/prompts/cheatsheet/RDNA3_architecture.md)
and the RDNA3 HIP and Triton `knowledge_override` entries in
[the cheatsheet map](../../src/prompts/cheatsheet/default_cheatsheet.yaml). The
guides describe gfx11 WMMA (FP16, BF16, INT8 and INT4 inputs; no FP8, FP4,
sparse WMMA or MFMA), wave32 launch geometry, the 64 KiB per-workgroup LDS
limit, and hardware counters whose values depend on the operator-set perf level.
The device facts in these guides were checked in this runtime on a W7900.

## Validation and limits

On a W7900 Dual Slot (48 GB), smoke passed its identity and numerical checks,
and a scored Claude Code run of the schema-v2 W7900 quickstart accepted both the
HIP GELU and Triton `test_add_kernel` candidates and wrote the board's identity
record.
Those single-run speedups are not performance claims. On the same host,
preflight and `main.py` refused replayed W7800, RX 7900 XTX and empty marketing
names, and the runner refused a W7900 config that enables evaluation tools.
These runs and the task sweep below predate the schema-v3 kernel tasks;
Triton, Gluon, HIP and FlyDSL are verified on MI355X only, and the W7900 is
implemented but unverified for them. A passing baseline does not validate a
generated target kernel.

- Architecture restrictions: the shipped kernel tasks declare `required_arch`
  in `platform_support` (currently `gfx950`) and are skipped on `gfx1100`.
- The schema-v2 task sweep on a W7900 found baselines that fail on `gfx1100`
  for an architecture reason: MFMA, fnuz FP8, MXFP4 or scaled dot, missing
  AITER kernels or configs, wave64 assumptions, 64 KiB LDS, and compiler
  output the gfx1100 assembler rejects. Image-bound tasks need their own
  source images, so this image cannot run them.
- Evaluation-tool sidecars are qualified only for `gfx950`. The host runner
  refuses them on `gfx1100`, and `main.py` refuses a W7900 run config that
  enables `evaluation_tools`.
- Only the W7900 is qualified here. Other `gfx11` boards, full vLLM serving,
  and every task category are not established by this recipe.

Keep task shapes, correctness tolerances, and timing policy intact when
investigating failures. Honor architecture restrictions, and qualify task
changes with the [QA auditor](../../docs/how-to/qa-auditor.md).

## Security and reproducibility review

External inputs are the digest-pinned upstream image and the hash-locked PyPI
wheels. Both automatic and explicit builds send only the Dockerfile, normalizer,
and package lock from `docker/rdna3/` as their context, install only locked
wheels, and use no agent credentials. The runner keeps its host-UID execution,
device mounts, privileged-container policy, and opt-in evaluation-tool policy.
These containers are reproducibility boundaries, not security sandboxes. Record
the derived local image ID with run evidence; the base digest alone does not
identify changes to this recipe.
