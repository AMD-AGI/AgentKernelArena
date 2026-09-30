# Draft DeepSeek task packages

These 18 schema-v2 task packages cover 13 block-scaled FP8 GEMMs, three MLA
variants, one mHC operator, and one Top-k operator. Each retains 13 workload rows,
an implemented Python starting candidate, a provided production baseline, and a
Triton target. The baseline, independent reference, initializer, comparator,
workloads, timing policy, and executable harness are preserved from the tested
packages. They do not depend on the task generator at runtime.

## Qualification still required

The previous MI355X (`gfx950`) run produced **2 PASS, 12 WARN, and 4 FAIL**
for this group. All seven runtime actions passed for 17 tasks; Top-k could not
load its production API. Successful execution does not replace formal task
qualification. This is a draft collection, not a claim that all 18 are ready.

| Task | Previous formal status | Remaining work |
| --- | --- | --- |
| `flash_mla_with_kvcache_dsv4_fp8_10011_q1_h64_d512_p256_k128` | FAIL | Add independent runtime-length and padding coverage |
| `flash_mla_with_kvcache_dsv4_fp8_11111_q1_h64_d512_p256_k128_ep2_ek8256` | FAIL | Add independent runtime-length and padding coverage |
| `flash_mla_with_kvcache_dsv4_fp8_11111_q1_h64_d512_p256_k128_ep64_ek512` | FAIL | Add independent runtime-length and padding coverage |
| `gemm_a8w8_blockwise_scaled_blk128x128_nt_obfloat16_bshuf16x16_aslogical_n4096_k256` | WARN | Replace unrelated bundle instructions and fix unavailable links |
| `gemm_a8w8_blockwise_scaled_blk128x128_nt_obfloat16_bshuf16x16_aslogical_n4096_k512` | WARN | Replace unrelated bundle instructions and fix unavailable links |
| `gemm_a8w8_blockwise_scaled_blk128x128_nt_obfloat16_bshuf16x16_aslogical_n512_k4096` | WARN | Replace unrelated bundle instructions and fix unavailable links |
| `gemm_a8w8_blockwise_scaled_blk128x128_nt_obfloat16_bshuf16x16_asraw_n1536_k4096` | WARN | Replace unrelated bundle instructions and fix unavailable links |
| `gemm_a8w8_blockwise_scaled_blk128x128_nt_obfloat16_bshuf16x16_asraw_n32768_k1024` | WARN | Replace unrelated bundle instructions and fix unavailable links |
| `gemm_a8w8_blockwise_scaled_blk128x128_nt_obfloat16_bshuf16x16_asraw_n4096_k1024` | WARN | Replace unrelated bundle instructions and fix unavailable links |
| `gemm_a8w8_blockwise_scaled_blk128x128_nt_obfloat16_bshuf16x16_asraw_n4096_k2048` | PASS | Fresh qualification after packaging |
| `gemm_a8w8_blockwise_scaled_blk128x128_nt_obfloat16_bshuf16x16_asraw_n4096_k256` | PASS | Fresh qualification after packaging |
| `gemm_a8w8_blockwise_scaled_blk128x128_nt_obfloat16_bshuf16x16_asraw_n4096_k4096` | WARN | Replace unrelated bundle instructions and fix unavailable links |
| `gemm_a8w8_blockwise_scaled_blk128x128_nt_obfloat16_bshuf16x16_asraw_n4096_k512` | WARN | Replace unrelated bundle instructions and fix unavailable links |
| `gemm_a8w8_blockwise_scaled_blk128x128_nt_obfloat16_bshuf16x16_asraw_n4096_k8192` | WARN | Replace unrelated bundle instructions and fix unavailable links |
| `gemm_a8w8_blockwise_scaled_blk128x128_nt_obfloat16_bshuf16x16_asraw_n512_k4096` | WARN | Replace unrelated bundle instructions and fix unavailable links |
| `gemm_a8w8_blockwise_scaled_blk128x128_nt_obfloat16_bshuf16x16_asraw_n8192_k1024` | WARN | Replace unrelated bundle instructions and fix unavailable links |
| `mhc_fused_post_pre_flat_rmsnorm_c4_d4096` | WARN | Replace unrelated bundle instructions and fix unavailable links |
| `topk_transform_paged_paged_k512_page_size64` | FAIL | Resolve production API, length coverage, and comparator boundary ordering |

The copied `BUNDLE_README.md` describes MLA and Top-k requirements and contains
links into the original bundle layout. It is not an adequate per-task guide for
GEMM or mHC. The MLA tasks additionally need explicit empty, short, intermediate,
and boundary length regimes, mixed rows, and independent negative padding in
both correctness and performance checks. Seed changes alone do not provide
this coverage. Top-k also needs a compatible production entrypoint and stricter
boundary-ordering checks. These findings remain unresolved in this draft.

## Reproduction and evidence scope

Historical image: `lmsysorg/sglang:v0.5.20-rocm10-mi35x`, pinned as:

```text
lmsysorg/sglang@sha256:e20849665c105d389ef91d23c0dc73931aaa6f02056dd10e7b43e4f16c79df69
```

The historical run used writable per-worker AITER and FlyDSL caches, with a
writable copy of AITER configuration files. PyTorch, AITER, SGLang, and their GPU
runtime dependencies came from that image. No package upgrade was performed.
The run config now pins this image using `docker_image`, and the runner applies
the writable-cache setup automatically for both the image tag and digest.

Run the explicit task selection using the repository Docker workflow on MI355X:

```bash
make docker-run CONFIG=example_configs/task_validator_deepseek_drafts_mi355x.yaml
```

Explicit environment image overrides take precedence over the config. See the
[Docker workflow](../../../docs/install/install.md) and
[task-validator guide](../../../docs/how-to/task-validator.md).

The executable task files, configs, callbacks, and workloads were checked against
the historical GPU run's source hashes when packaged. The task selectors have
changed; those historical reports are not fresh qualification of this branch.
CPU packaging checks cover schema loading, case manifests, isolated materialization,
Python syntax, and failure reporting without a GPU. They cannot qualify numerical
correctness or performance. Obtain fresh framework-finalized reports before
merging; WARN is not a clean PASS and FAIL remains blocking.
