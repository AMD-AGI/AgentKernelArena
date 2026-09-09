# RDNA4 runtime image

This recipe supports `gfx1201`. The base image and immutable digest live in
[Dockerfile](Dockerfile); the default local output tag lives in
[`docker_benchmark.sh`](../../src/scripts/docker_benchmark.sh).

From the repository root on the GPU host:

```bash
make docker-build-rdna4
make docker-smoke
make docker-check-agents CONFIG=example_configs/quickstart_claude_rdna4.yaml
make docker-run CONFIG=example_configs/quickstart_claude_rdna4.yaml
```

Install and authenticate the selected agent separately as described in the
[installation guide](../../docs/install/install.md). The build and smoke steps
do not need an agent or its credentials. `AKA_DOCKER_IMAGE_GFX1201` overrides
both the build output tag and the architecture-specific runtime selection;
`AKA_DOCKER_IMAGE` remains the global run override. Build the image on each
Docker host before selecting it. There is no automatic image build on a run.

## Runtime layout

The upstream image puts a uv-managed Python virtual environment in
`/opt/python`, with its base interpreter under `/root`. The recipe copies only
that base interpreter and standard library to `/opt/aka-python-runtime` and
updates the virtual environment's interpreter links and `pyvenv.cfg`. Installed
PyTorch, Triton, and ROCm SDK packages remain in their original locations.
The recipe also installs the missing plotting packages from
[`requirements.lock`](requirements.lock), using exact versions and wheel hashes
without dependency upgrades. Other plotting dependencies come from the pinned
base image. Refresh this lock when changing the base or Python version.

Aliases `/opt/venv` and `/opt/rocm` expose the virtual environment and ROCm SDK
through the runner's existing PATH. The build checks Python imports and compiler
and profiler accessibility as an unprivileged UID, without exposing `/root`.
Experiments still run as the invoking host UID; no runtime root initialization
or host permission changes are required.

The `gfx1201` smoke check requires `hipcc` and `rocprofv3`. Other architectures
retain the existing `hipcc` and `rocprof-compute` checks. Finding a profiler does
not imply that a task or candidate was profiled.

## Validation and limits

Runtime checks have exercised existing HIP GELU, Triton RMSNorm, and BF16 GEMM
tasks on a 16 GB `gfx1201` GPU. This is not a guarantee that every Arena task fits
the device. In particular, the current vLLM persistent-matmul task can request
96 KiB shared memory against the device's 64 KiB per-workgroup limit. Keep task
shapes, correctness tolerances, and timing policy intact when investigating such
failures; any tuning belongs in the permitted kernel implementation.

Only `gfx1201` is configured here. Other Radeon architectures, optional FlyDSL
and AITER task dependencies, full vLLM serving, and evaluation-tool sidecars need
separate qualification. Existing task contracts and benchmark helpers are
unchanged. Agent availability and authentication are separate preflight checks.

## Security and reproducibility review

External inputs are the digest-pinned upstream image and the hash-locked PyPI
wheels. The explicit build sends only the Dockerfile, normalizer, and package
lock from `docker/rdna4/` as its context. It installs only the locked wheels,
with no source builds, dependency resolution, or agent credentials. The normalizer
refuses to replace existing runtime aliases and leaves `/root` private. The runner
retains its existing host-UID execution, device mounts, privileged-container
policy, and opt-in evaluation-tool policy. These containers remain
reproducibility boundaries, not security sandboxes. Record the derived local
image ID with run evidence; the base digest alone does not identify changes to
this recipe.
