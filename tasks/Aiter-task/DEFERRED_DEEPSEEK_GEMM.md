# Deferred DeepSeek GEMM task packages

These seven additional schema-v2 tasks are deferred for production-baseline
numerical failures: five BF16 GEMMs and two block-scaled FP8 GEMMs. They are
separate from the existing deferred GEMM benchmark upgrades and MXFP8 linear
task. Each package retains 13 workload rows, an implemented Python starting
candidate, a provided production baseline, and a Triton target.

All seven received **FAIL** in the previous MI355X (`gfx950`) run. The baseline
failed its required numerical comparison, so the complete candidate and
performance actions did not run. This does not establish whether the root cause
is the production implementation, dispatch, packing/scales, or reference
contract. No diagnostic exception or relaxed comparison has been introduced.

| Task | First failing M | Baseline result |
| --- | ---: | --- |
| `gemm_a16w16_nt_n1024_k4096` | 1 | FAIL |
| `gemm_a16w16_nt_n2048_k4096` | 1 | FAIL |
| `gemm_a16w16_nt_n256_k4096` | 4 | FAIL |
| `gemm_a16w16_nt_n512_k4096` | 2 | FAIL |
| `gemm_a16w16_nt_n64_k4096` | 8 | FAIL |
| `gemm_a8w8_blockwise_scaled_blk128x128_nt_obfloat16_bshuf16x16_aslogical_n1024_k4096` | 128 | FAIL |
| `gemm_a8w8_blockwise_scaled_blk128x128_nt_obfloat16_bshuf16x16_asraw_n1024_k4096` | 128 | FAIL |

BF16 failures occur at small M. Both FP8 variants first fail at M=128 while
using different activation-scale storage layouts. Reproduce these exact cases,
check production dispatch and scaling/packing against the independent reference,
and identify the mismatch before changing implementation behavior. The supplied
operator comparator requires its existing `atol=rtol=0.01` rule; the generic
policy fields do not replace that callback.

The unrelated bundle instructions, unavailable documentation links, and
imprecise failing-case labels also remain to be repaired. Preserve the oracle,
input distribution, workloads, and numerical thresholds while investigating.

## Reproduction and evidence scope

Historical image: `lmsysorg/sglang:v0.5.20-rocm10-mi35x`, pinned as:

```text
lmsysorg/sglang@sha256:e20849665c105d389ef91d23c0dc73931aaa6f02056dd10e7b43e4f16c79df69
```

The historical run used writable per-worker AITER and FlyDSL caches, with a
writable copy of AITER configuration files. PyTorch, AITER, SGLang, and their GPU
runtime dependencies came from that image. No package upgrade was performed.
The image and writable-cache setup are part of reproducing these findings.

Run the explicit task selection using the repository Docker workflow on MI355X:

```bash
AKA_DOCKER_IMAGE=lmsysorg/sglang@sha256:e20849665c105d389ef91d23c0dc73931aaa6f02056dd10e7b43e4f16c79df69 \
  make docker-run CONFIG=example_configs/task_validator_deferred_deepseek_gemm_mi355x.yaml
```

The image override pins the container, but writable-cache setup is still required
for the historical runtime. See the
[Docker workflow](../../../docs/install/install.md) and
[task-validator guide](../../../docs/how-to/task-validator.md).

The executable task files, configs, callbacks, and workloads were checked against
the historical GPU run's source hashes when packaged. The task selectors have
changed; those historical reports are not fresh qualification of this branch.
CPU packaging checks cover schema loading, case manifests, isolated materialization,
Python syntax, and failure reporting without a GPU. They cannot qualify numerical
correctness or performance. Obtain fresh framework-finalized reports before
merging; WARN is not a clean PASS and FAIL remains blocking.
