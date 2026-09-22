# Run the restored upstream head-kernel tasks

The original tasks live at `tasks/headkernel/<flat-task-name>/`. The five-model
selection contains 16 configured tasks and five `NOT_BUILT` placeholders. The
task files, sources, harnesses and original suite tools are restored from the
upstream delivery; [HEADKERNELS_UPSTREAM.md](../../HEADKERNELS_UPSTREAM.md) preserves
its README. The original tools also describe the wider six-model suite, including
GLM-5.2, which is outside this selection. Their full-suite counts therefore differ.

Verification of copied bytes establishes source identity. It does not establish
GPU correctness, performance, or a new framework `task_validator` PASS. No GPU
validation was run for this restoration. Historical upstream verdicts remain
historical evidence; preserve new run logs and reports separately.

## Select the original runtime

The original runtime declarations target MI355X / `gfx950`. Each task's
`headkernel.docker` preserves that source evidence. The authorized public overrides
are documented immediately below. These task-validator configs preserve
the original image grouping and select only configured tasks:

| Run config | Tasks | Exact original image |
| --- | ---: | --- |
| [upstream_headkernel_sglang_v0517_mi355x.yaml](../../example_configs/upstream_headkernel_sglang_v0517_mi355x.yaml) | 7 | `harbor.crusoe.primus-safe.amd.com/hyperloom-image/sglang:v0.5.17-rocm720-mi35x-profilerfix` |
| [upstream_headkernel_sglang_v0518_mi355x.yaml](../../example_configs/upstream_headkernel_sglang_v0518_mi355x.yaml) | 6 | `harbor.crusoe.primus-safe.amd.com/hyperloom-image/sglang:v0.5.18-rocm720-mi35x-profilerfix` |
| [upstream_headkernel_kimi_k3_mi355x.yaml](../../example_configs/upstream_headkernel_kimi_k3_mi355x.yaml) | 3 | `harbor.crusoe.primus-safe.amd.com/hyperloom-image/sglang-rocm-k3:rocm720-mi35x-k3-20260727-tl312-08011830` |

All 16 configs retain `ISL 8192 / OSL 1024 / CONC 64 / TP 8`. The UTs replay
captured rank-local operations; this serving provenance does not mean each UT
launches an eight-GPU model server.

**GLM elementwise has an upstream version discrepancy.** Its
[config.yaml](../../tasks/headkernel/glm-5.3-flash__elementwise_copy_cluster/config.yaml)
declares v0.5.17, while its
[ut/README.md](../../tasks/headkernel/glm-5.3-flash__elementwise_copy_cluster/ut/README.md)
documents v0.5.18. The v0.5.17 cohort preserves the original declaration. This
restoration does not resolve the disagreement by changing the image pin.

The GLM UT docs record Torch `2.9.1+rocm7.2.0`, SGLang `0.5.18`, HIP `7.2.26015`
and `gfx950`. Qwen metadata records AITER commit
`d9e5ef7ce08ee7045d583aed768cff41aa9210fe` or SGLang commit
`71de97b264b04dcd514cf904003028aefe9775c8`, according to the source repository,
and image/config ID
`sha256:760dd38b9b6f2bd11c13011d470eb8e377c3f0d71284a090a710d64a23bd789f`.
That ID is not a registry manifest digest. No equivalence to a public image is
established by these records.

### Image availability checked on 2026-09-22

The original Harbor hostname failed DNS resolution from both the CPU work host
and the source login host. Ordinary-user access to both Docker sockets was also
denied. No image was pulled, exported, uploaded to OCI, or run during this check.
Current original-image sizes and registry manifest digests could not be observed. The
Qwen image/config ID above remains historical source evidence.

Exact original-image export therefore still requires access to the original registry or an
existing authorized Docker daemon holding the exact images. Preserve a raw
registry manifest and its layer digests when exporting from the registry, or
record image IDs and repository digests alongside a `docker image save` archive.
Hash each completed archive before transferring it with `rclone`. Loading an
archive on another node must reproduce the recorded image ID. No OCI image
object is advertised as ready by this guide.

## Authorized public runtime overrides

The current runtime choice is the corresponding public `rocm/hyperloom` image.
Original task `headkernel.docker` values remain unchanged as historical source
metadata. On 2026-09-22, anonymous Docker Hub manifest reads verified both public
image digests and their configuration digests; no GPU execution was performed.
These are authorized public runtime choices, with no claim of byte identity to
Harbor images.

| Alias | Public tag | Pinned pull reference | Compressed layer bytes |
| --- | --- | --- | ---: |
| P17 | `docker.io/rocm/hyperloom:sglang-v0.5.17-rocm7.2.0-mi350x` | `docker.io/rocm/hyperloom@sha256:1f5464829559b086eb66f9b803cb9c7a817438c43edff2d5ef59b46a186745f6` | 28479323239 |
| P18 | `docker.io/rocm/hyperloom:sglang-v0.5.18-rocm7.2.0-mi350x` | `docker.io/rocm/hyperloom@sha256:da36f56f24cb2897a56be52dd43774c1a75b1500308ed7db3ba32fb8db4d259c` | 23161974193 |

Every workload below uses `TVM_FFI_DISABLE_TORCH_C_DLPACK=1`, the compiler cache
variables and setup mount shown in the direct Docker command below. The full
per-task image, original image, environment and patch mapping is machine readable
in [tools/headkernel-public-runtimes.json](../../tools/headkernel-public-runtimes.json).

| Workload / kernel task | Public runtime | Additional setup / status |
| --- | --- | --- |
| `deepseek-v4-pro__dsa_sparse_mla_attn` | P17 | No GPU validation |
| `deepseek-v4-pro__moe_stage1_grouped_gemm_silu_flydsl` | P17 | No GPU validation |
| `deepseek-v4-pro__moe_stage2_down_proj_reduce_opus_a8w4` | P17 | No GPU validation |
| `glm-5.3-flash__elementwise_copy_cluster` | P17 | Preserves original v0.5.17 grouping; UT docs record v0.5.18 |
| `minimax-m3__decode_score_kernel` | P17 | No GPU validation |
| `minimax-m3__gqa_share_sparse_decode_kernel` | P17 | No GPU validation |
| `minimax-m3__gqa_share_sparse_fwd_kernel` | P17 | No GPU validation |
| `glm-5.3-flash__fused_moe_kernel` | P18 | Original patch + GLM configuration bundle; patch compatibility must pass |
| `qwen3.8-2.4t__dense_bf16_gemm_cluster` | P18 | Task-local `live_dispatch_rows.csv` |
| `qwen3.8-2.4t__fused_moe_2stage_mxfp4` | P18 | No GPU validation |
| `qwen3.8-2.4t__fused_recurrent_gated_delta_rule_decode` | P18 | No GPU validation |
| `qwen3.8-2.4t__gemma_fused_add_rmsnorm` | P18 | No GPU validation |
| `qwen3.8-2.4t__paged_attention_decode` | P18 | No GPU validation |
| `kimi-k3__fwd_grouped_kernel_stage1` | P17 | Candidate only; custom Kimi build compatibility unverified |
| `kimi-k3__moe_gemm1_stage1` | P17 | Candidate only; custom Kimi build compatibility unverified |
| `kimi-k3__moe_gemm2_stage2` | P17 | Candidate only; custom Kimi build compatibility unverified |

The complete Docker Hub `rocm/hyperloom` tag listing contained no Kimi/K3-named
build on 2026-09-22. Generic public v0.5.17 is therefore only a candidate for the
three Kimi tasks; it is not established as the custom original Kimi build. Keep
that compatibility limitation in any handoff. Public images are pulled directly
from Docker Hub; no OCI mirror of the original Harbor images is claimed.

## Restore the original tensor files

The 16 tasks require **17 original tensor files**: 14 `ut/reference_io.pt` files
and the three MiniMax `ut/timing_geometry.pt` files. Source code alone is
insufficient to run their original harnesses. Obtain the original files through
the source delivery and verify their recorded hashes before running. Preserve
the original serialization and metadata; regenerated inputs change the setup.

The OCI object-storage destination is recorded in
[tools/headkernel-artifacts.json](../../tools/headkernel-artifacts.json):

```text
oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_ut_0914_full/20260922/tensors
```

Check its `oci_storage.publication_status` before a handoff. A configured
destination does not establish that every upload completed. Once publication is
verified, configure an `oci` rclone remote outside the repository and download
the complete original objects with the fixture helper:

```bash
python3 src/tools/prepare_head_kernel_artifacts.py --download
python3 src/tools/prepare_head_kernel_artifacts.py --verify
```

The helper verifies the declared byte count and SHA-256 before installing each
file. `--oci-remote REMOTE:bucket/prefix` selects another caller-configured
remote. Registry and object-storage credentials are not part of the delivery.

Paths below are relative to `tasks/headkernel/`:

| Task | Required files under `ut/` |
| --- | --- |
| `deepseek-v4-pro__dsa_sparse_mla_attn` | `reference_io.pt` |
| `deepseek-v4-pro__moe_stage1_grouped_gemm_silu_flydsl` | `reference_io.pt` |
| `deepseek-v4-pro__moe_stage2_down_proj_reduce_opus_a8w4` | `reference_io.pt` |
| `glm-5.3-flash__elementwise_copy_cluster` | `reference_io.pt` |
| `glm-5.3-flash__fused_moe_kernel` | `reference_io.pt` |
| `kimi-k3__fwd_grouped_kernel_stage1` | `reference_io.pt` |
| `kimi-k3__moe_gemm1_stage1` | `reference_io.pt` |
| `kimi-k3__moe_gemm2_stage2` | `reference_io.pt` |
| `minimax-m3__decode_score_kernel` | `reference_io.pt`, `timing_geometry.pt` |
| `minimax-m3__gqa_share_sparse_decode_kernel` | `reference_io.pt`, `timing_geometry.pt` |
| `minimax-m3__gqa_share_sparse_fwd_kernel` | `reference_io.pt`, `timing_geometry.pt` |
| `qwen3.8-2.4t__fused_moe_2stage_mxfp4` | `reference_io.pt` |
| `qwen3.8-2.4t__fused_recurrent_gated_delta_rule_decode` | `reference_io.pt` |
| `qwen3.8-2.4t__paged_attention_decode` | `reference_io.pt` |

Qwen dense GEMM and fused add+RMSNorm deliberately generate their deterministic
baseline at runtime and do not require `reference_io.pt`. Kimi MoE stores a
synthetic schema with captured routing in its frozen blob; the common config's
oracle label does not describe every package's internal format precisely.

The MiniMax geometry files now exist in the refreshed source inventory. The
older upstream README's statement that they are absent is stale. Keep each
task's `capture_telemetry.json`, `attempts/`, baseline files and overlays too:
the original provenance checks can open them during execution.

## Download the published setup assets

All ten external setup files are published under this OCI prefix. Download them
with the requested transfer/progress options:

```bash
mkdir -p .upstream-assets/setup
rclone copy \
  oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_ut_0914_full/20260922/setup \
  .upstream-assets/setup --transfers 64000 --progress
```

Verify the downloaded files against
[tools/headkernel-setup-artifacts.json](../../tools/headkernel-setup-artifacts.json).
From the repository root, verify the published checksums and restore the lock
script executable bit before using its wrapper:

```bash
(cd .upstream-assets/setup && sha256sum -c ../../tools/headkernel-setup.SHA256SUMS)
chmod +x .upstream-assets/setup/shared_nfs/hongtaom/qwen3_14B/hl_matrix_0824/deps/kimi-k3/GEAK/kernel_workflow/scripts/gpu_lock.sh
```
The object layout preserves their original paths below `shared_nfs/`. Mount
`.upstream-assets/setup/shared_nfs` at `/shared_nfs` in the container so the
unchanged `pre_run_patch`, `GEAK_MODEL_PATH` default and Kimi `gpu_lock.sh` paths
resolve. The setup upload is complete; the image-export limitation is separate.

## Complete the original environment

The original [tools/run_on_gpu.sh](../../tools/run_on_gpu.sh) and upstream README
define the suite launch environment. Read the script before using its cluster
driver; it relies on the upstream shared-filesystem layout and reservation tools.
Its actual syntax is `bash tools/run_on_gpu.sh <jobid> <task-id|all>`; it also
supports `pending`. The task ID is the flat directory name. Use an explicit task
ID for this subset. The original driver reads the original Harbor declaration;
it does not select the public override table. Use the direct Docker route below
for an authorized public image.

The suite sets `TVM_FFI_DISABLE_TORCH_C_DLPACK=1` inside its containers. The
v0.5.18 image ships a CPU build of the optional torch/DLPack addon; otherwise
fresh subprocesses can repeatedly try and fail to compile the ROCm addon. Keep
the original cache environment and warmup/timing settings from the launcher and
task runner when comparing results.

**GLM fused MoE requires an external architecture patch and checkpoint
configuration.** Its
[config.yaml](../../tasks/headkernel/glm-5.3-flash__fused_moe_kernel/config.yaml)
declares `headkernel.pre_run_patch`. Make that original patch available to the
container and apply it to the image's SGLang checkout as the upstream launcher
does. Stock v0.5.18 does not recognize `glm5_next` without the patch; stock
v0.5.17 also lacks the runtime-context API this UT uses.

The original
[sglang_bootstrap.py](../../tasks/headkernel/glm-5.3-flash__fused_moe_kernel/ut/sglang_bootstrap.py)
reads `GEAK_MODEL_PATH`, defaulting to the original shared model path, and passes
it as both `model_path` and `tokenizer_path` to real `ServerArgs`. Ensure the
checkpoint configuration and any files that this image reads from that path
are accessible inside the container. The bootstrap publishes the original
TP=8 configuration and initializes a world-size-1 process group to replay one
already-sharded MoE operation. Its declared patch and model assets remain
external setup requirements; a source-only checkout does not supply them.

The source SSH inventory on 2026-09-22 confirmed the original patch and the model configuration/tokenizer files.
The published bundle contains `config.json`, `generation_config.json`,
`processor_config.json`, `tokenizer_config.json`, `chat_template.jinja`,
`tokenizer.json`, the upstream model README and license, plus the patch and
GPU-lock script: 10 files totaling 21,170,934 bytes. It contains no checkpoint
weights or weight index; the exact minimum subset accessed by the original
image was not tested. The original `config.json` declares
`model_type: glm5_next` and architecture `Glm5NextForConditionalGeneration`.

| External setup file | Bytes | SHA-256 observed on source |
| --- | ---: | --- |
| Declared `001_glm5_next_arch_enablement.patch` | 850212 | `808c011b098cbc0087f303e4424f8e248984bbf8419879b058c4c35449883558` |
| GLM `config.json` | 69416 | `bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f` |
| Kimi `kernel_workflow/scripts/gpu_lock.sh` | 13048 | `6174729af74289cbe6575b08c87a5692e0a3e170b8a252c79c888a6dac41515a` |

The source inventory above is included in the published setup delivery. Use the
setup manifest to verify all ten downloaded files before launching a container.

**Kimi grouped attention requires its original baseline overlay.** Its
[ut/run_unittest.sh](../../tasks/headkernel/kimi-k3__fwd_grouped_kernel_stage1/ut/run_unittest.sh)
sets `PYTHONPATH` to `ut/baseline_overlay` and invokes
`GEAK_ROOT/kernel_workflow/scripts/gpu_lock.sh`. The original UT README identifies
that overlay as necessary to preserve the accepted tuning-round baseline. Supply
the original lock script when using this wrapper. The common task runner invokes
`unittest.py` directly and inherits the environment; inspect the actual launch
environment and both resolved legs before interpreting a comparison.

Qwen dense GEMM sets `AITER_CONFIG_GEMM_BF16` to its own
`ut/live_dispatch_rows.csv`. Preserve that local dispatch file. Historical
serving paths in selection evidence and oracle-capture scripts do not imply
that a full model server or those capture directories are needed for every UT.

## Direct Docker command with the public runtime

After installing the tensor files and setup bundle, select one task and its
public runtime from the table. This example runs GLM fused MoE and fails if the
original patch does not apply cleanly to the selected public image. A patch
failure is a compatibility issue to report; do not silently skip it.

```bash
TASK=tasks/headkernel/glm-5.3-flash__fused_moe_kernel
IMAGE=docker.io/rocm/hyperloom@sha256:da36f56f24cb2897a56be52dd43774c1a75b1500308ed7db3ba32fb8db4d259c
GPU=0
mkdir -p .upstream-assets/cache/{triton,flydsl,comgr,tilelang,torch_ext}
docker run --rm -i -u 0 --entrypoint /bin/bash \
  --ipc=host --network=host --shm-size 128G \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --security-opt seccomp=unconfined \
  -e HIP_VISIBLE_DEVICES="$GPU" -e PYTHONUNBUFFERED=1 \
  -e TRITON_CACHE_DIR=/cache/triton -e FLYDSL_RUNTIME_CACHE_DIR=/cache/flydsl \
  -e AMD_COMGR_CACHE_DIR=/cache/comgr -e TILELANG_CACHE_DIR=/cache/tilelang \
  -e TORCH_EXTENSIONS_DIR=/cache/torch_ext -e TVM_FFI_CACHE_DIR=/tmp/c/tvm_ffi \
  -e TVM_FFI_DISABLE_TORCH_C_DLPACK=1 \
  -v "$PWD:/workspace" \
  -v "$PWD/.upstream-assets/setup/shared_nfs:/shared_nfs:ro" \
  -v "$PWD/.upstream-assets/cache:/cache" \
  -w "/workspace/$TASK" "$IMAGE" -se <<'CONTAINER'
mkdir -p /tmp/c/tvm_ffi
patch_path=$(python3 -c 'import yaml; print(yaml.safe_load(open("config.yaml"))["headkernel"].get("pre_run_patch", ""))')
if [ -n "$patch_path" ]; then
  git -C /sgl-workspace/sglang apply --check "$patch_path"
  git -C /sgl-workspace/sglang apply --whitespace=nowarn "$patch_path"
fi
HK_TASK_TIMEOUT=240 timeout -k 120 300 python3 -u scripts/task_runner.py compile
HK_TASK_TIMEOUT=5340 timeout -k 120 5400 python3 -u scripts/task_runner.py correctness
HK_TASK_TIMEOUT=1140 timeout -k 120 1200 python3 -u scripts/task_runner.py performance
CONTAINER
```

The device, shared-memory, cache and TVM environment settings and phase budgets
come from the original driver. The container uses its original root-user mode;
outputs in the mounted checkout may consequently be root-owned. For Kimi grouped
attention also follow its baseline-overlay wrapper requirements above. These
commands are instructions for the next GPU node, not a record of a run performed
during this restoration.

## Execute and report

Inside the task's original Docker image, after completing its setup, run its
unchanged commands from the task directory:

```bash
python3 scripts/task_runner.py compile
python3 scripts/task_runner.py correctness
python3 scripts/task_runner.py performance
```

These commands write task-local `build/` reports. `HK_TASK_TIMEOUT` defaults to
1800 seconds in the original task runner. Preserve the original performance
path, including any explicitly reported fallback, and report failures without
relabeling them as environment-independent correctness verdicts.

The standard framework accepts the cohort configs above with its existing
`make docker-run CONFIG=...` interface and `AKA_DOCKER_IMAGE` override. For
example, its image-selection syntax for the first cohort is:

```bash
AKA_DOCKER_IMAGE=docker.io/rocm/hyperloom@sha256:1f5464829559b086eb66f9b803cb9c7a817438c43edff2d5ef59b46a186745f6 \
  make docker-run CONFIG=example_configs/upstream_headkernel_sglang_v0517_mi355x.yaml
```

**This framework command alone does not complete upstream setup.** The unchanged
[Docker runner](../../src/scripts/docker_benchmark.sh) does not consume
`headkernel.docker` or `headkernel.pre_run_patch`, mount the shared model tree, or
forward `TVM_FFI_DISABLE_TORCH_C_DLPACK` from the host. The image override selects
the image only. Use the original launcher for an original-setup run. A framework
reproduction additionally needs the same environment and assets supplied to its
own container; the cohort config and image override alone do not provide them.
Do not use the framework's default MI355X image for these cohorts.

The example configs select the existing `task_validator` agent, whose CLI and
authentication requirements are described in the
[validator guide](task-validator.md). They do not introduce a separate native
verifier. A fresh framework validation requires its finalized
`validation_report.yaml`; direct UT reports, copied-byte checks, and historical
upstream reports are distinct evidence. Record the exact image identity,
hardware, setup assets, command and resulting report for every new GPU run.
