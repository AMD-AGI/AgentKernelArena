# Public images for head-kernel qualification

The public [ROCm Hyperloom repository on Docker Hub](https://hub.docker.com/r/rocm/hyperloom/tags)
provides the following relevant images. Anonymous manifest and configuration
requests succeeded on 2026-09-20, and their content hashes were verified.
The [machine-readable record](top5-public-images.json) includes full digests,
build source pins, sizes and observation times. This is registry verification;
it is not a successful GPU correctness or performance run.

| Public tag under `rocm/hyperloom` | ROCm / target | AITER build commit | Use in this suite |
| --- | --- | --- | --- |
| `sglang-v0.5.17-rocm7.2.0-mi350x` | 7.2.0 / `gfx950` | `d9e5ef7ce08ee7045d583aed768cff41aa9210fe` | Public default for 11 tasks, including Kimi; [native results by task](../../tasks/head_kernels/native_verified.json) |
| `sglang-v0.5.18-rocm7.2.0-mi350x` | 7.2.0 / `gfx950` | `d9e5ef7ce08ee7045d583aed768cff41aa9210fe` | Public default for seven tasks; [native results by task](../../tasks/head_kernels/native_verified.json) |
| `sglang-v0.5.18-main20260825-rocm7.2.0-mi350x` | 7.2.0 / `gfx950` | `d9e5ef7ce08ee7045d583aed768cff41aa9210fe` | Separate main-build variant; not qualified |
| `sglang-v0.5.19-rocm7.2.0-mi350x` | 7.2.0 / `gfx950` | `c16d44b93a528b2a4bfd6d8d3409116d465872a9` | Different SGLang and AITER revisions; not a replacement for the frozen v0.5.17 task |

The `mi350x` public tag targets `gfx950`, as recorded in its build environment.
The task still requires an actual MI355X-compatible `gfx950` runtime; the tag
alone does not prove the GPU or the installed operator's behavior.

## Download the v0.5.17 candidate

Use the immutable registry manifest reference:

```bash
docker pull rocm/hyperloom@sha256:1f5464829559b086eb66f9b803cb9c7a817438c43edff2d5ef59b46a186745f6
```

This identifies `rocm/hyperloom:sglang-v0.5.17-rocm7.2.0-mi350x` as observed
on 2026-09-20. Its layers total **28,479,323,239 compressed bytes**, separate
from any task tensor fixtures. Its Docker image/config ID is:

```text
sha256:ffe4af630e49b05c812db4a468bfb411c3dbb0e93124801f28349bfa31352dea
```

The manifest digest and image/config ID are different identifiers. Both must
match the declared public validation runtime. A first-layer anonymous HEAD
also returned HTTP 200; the complete image was not downloaded during this
registry check.

All **18 task input contracts are self-contained**, with zero required tensor
archives and no shared cluster filesystem dependency. The GLM BF16 task, for
example, generates all **27 cases** locally. The image's AITER build pin matches
that task's required source revision, and the public `aiter/tuned_gemm.py` at that
revision has the task's exact required SHA-256. The task retains its installed
source hash check: a declared build argument is not a substitute for checking
the installed file and executing the GPU tests.

Use the dedicated public validation config and the cohort launcher's `verify`
action documented in [the runtime guide](../how-to/top5-head-kernels-runtime.md).
It runs the task's native commands without an LLM CLI or agent credentials.
Completed compile, correctness, and performance phases are recorded in the
[native-verified index](../../tasks/head_kernels/native_verified.json) for the
specific operator contracts, source identities, and runtimes tested. Direct
verification leaves `framework_task_validator: NOT_RUN` and does not create a
framework-finalized `validation_report.yaml` or establish full model-serving
equivalence. The separate framework acceptance route uses `run` with a configured
task-validator backend, as described in the runtime guide.

All 18 task defaults now select pinned public images. The original
capture runtime and the current public runtime have separate recorded
identities. The private Harbor image's exact build delta and byte identity
have not been established; this candidate is not presented as its exact mirror.

## Other public images are separate builds

`lmsysorg/sglang:v0.5.17-rocm720-mi35x` and
`lmsysorg/sglang:v0.5.18-rocm720-mi35x` are public stable-release images whose
build records also use AITER `d9e5ef7c...`. They have their own image identities.

The dated `lmsysorg/sglang-rocm` images are development builds. For example,
`v0.5.17-rocm720-mi35x-20260821` uses AITER `c16d44b9...`; its public
`tuned_gemm.py` differs from the GLM task's required file despite the v0.5.17
tag prefix. The September 13 v0.5.19 development image uses another AITER
revision again. Matching framework version labels do not establish equal
operator code.

The `rocm10` SGLang variants change the ROCm stack. The `rocm/vllm` images
also change the serving framework. None is an automatically admitted runtime
for these captured SGLang tasks.

Public image access and self-contained inputs do not resolve outstanding
correctness, workload-fidelity or benchmark qualification findings. Those remain
separate requirements in [the validation status](../../tasks/head_kernels/VALIDATION.md).
