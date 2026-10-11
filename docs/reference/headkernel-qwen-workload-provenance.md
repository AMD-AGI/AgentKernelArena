# Qwen workload provenance and proposed SG520 capture

The proposed Qwen3.8 TP8 serving capture on SGLang 0.5.20 **has not run**. Its
preparation is secondary to completing the 23 isolated kernel tasks. The
[runtime catalogue](../../tools/headkernel-runtime-targets.json) remains the
source of truth for task mappings and qualification; this document supplies
workload provenance, not a new task pass or end-to-end gain.

Two historical workloads need separate treatment. The latest measured Qwen
record in the catalogue preserved on October 5 used **TP8/EP1**. The original
callable fixtures came from **TP8/EP8**. Both used ISL8192, OSL1024 and C64, but
their parallelism, cache settings and dispatch routes differ.

| Evidence | Parallelism | Recorded settings | Limits |
| --- | --- | --- | --- |
| [Hyperloom record 103483][latest-record], session started September 18, 12:35:28 UTC | TP8 / **EP1** | Context 11264; memory fraction 0.8; page size 16; BF16 Mamba states; AITER unified attention and all-reduce fusion; checkpoint prefetch with four threads | Reports version 0.5.17; image and source revision absent. Raw run is inaccessible. Graph-capture traces and shape-profiler events were reported missing. |
| r11 serving contract, run `qwen38_hyperloom_local_kernel_20260913T081758Z_093_multikernel_r11` | TP8 / **EP8** | Context 11264; chunked prefill 16384; declared maximum model length 13312; memory fraction 0.8; 192 prompts and eight warmups | The launch recipe does not disable graphs. This serving contract differs from the later fixture extraction. |
| September 14 callable extraction, run `qwen38/HeadKernel/20260913T180016Z` | TP8 / **EP8** | Context 11264; chunked prefill 16384; page size 1; effective memory fraction **0.68**; graphs disabled; server/client warmup skipped; KV dtype `auto`; FP32 Mamba state from the model configuration | The script declared memory fraction 0.8, but `captures/moe_attention_bench/server.log` records 0.68. Its image was SGLang 0.5.18, not the SG520 target. |

The immutable checkpoint is
[`amd/Qwen3.8-2.4T-A95B-Quark-MXFP4` at revision `1122ff36878499eb1e44823999e7de10ddfabd3b`][checkpoint].
The pinned Hugging Face revision metadata was checked on October 11: **213 weight
shards, 287,119 tensor keys and 1,372,407,479,936 weight bytes**, plus 11 metadata
files. Every shard's upstream LFS SHA-256 matched its local cache blob name and
size; each metadata file matched its upstream Git/LFS identity. The full 1.37 TB
of shard contents was not rehashed during preparation. A new node must download
the pinned revision and verify every destination file. A mutable model alias or
`refs/main` is not a checkpoint identity.

The proposed image is the exact `images.sglang_v0520` target in the runtime
catalogue, with manifest digest
`sha256:3a78acc9d6c191f1a12c7c67631657580f06af3af562c9ee6d88282a71ec5e96`.
The package records 26 native source hashes. They came from a verified,
selected-layer projection and must be checked against the actual pinned image
before loading weights. The SG520 text-only entry is
`sglang.srt.models.qwen3_5_text:Qwen3_5MoeForCausalLM`; the inspected source does
not justify an architecture rewrite or source backport.

The primary proposed recipe, `latest-record-tp8-ep1`, retains record 103483's
exact server extras and reviewed non-secret workload switches. These include
`SGLANG_USE_AITER=1`, `SGLANG_MAMBA_SSM_DTYPE=bfloat16`,
`SGLANG_USE_AITER_UNIFIED_ATTN=1` and
`SGLANG_MOE_COPY_WEIGHT_VIEWS_BEFORE_H2D=1`. It retains the record's
`AITER_FLYDSL_FORCE=1`; SG520 recognizes that switch, and its source labels it
for debug use. These are recipe-specific settings, not universal requirements
for Qwen. The separate `historical-extraction-tp8-ep8` recipe reproduces the
extraction settings on a different image and does not reproduce r11's serving
configuration.

The latest record's benchmark seed, request count and warmup details were not
recovered. The new protocol therefore explicitly selects the current SG520
native random-ShareGPT procedure: seed 42, 64 requests of 8192/1024 at C64, and
an unscored 64-request 8192/16 warmup for the primary recipe. Those are new
protocol choices. The EP8 extraction variant skips client warmup. Profiling
requests eight steps per stage; report actual GPU-correlated coverage rather
than interpreting graph-launch counts as complete tracing.

The sanitized portable package is published at this immutable OCI prefix:

```text
oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_workload_handoffs/20261011/qwen-sg520-provenance/20261011T083812Z/
```

The archive and all **46 payload files** were downloaded and hash-verified.
The package contains launch/model/source metadata, bounded evidence excerpts,
portable command rendering and read-only verification scripts. It contains no
credentials, raw process environment, raw logs, full source trees, tensors or
model weights. Original evidence filenames and hashes remain in the package.

| Artifact | SHA-256 |
| --- | --- |
| [qwen-sg520-workload-provenance.tar.gz][archive] | `33bed7e2a410cc54eb39a670239a918549f79cc687c16d2b00f8f303b310b19b` |
| [MANIFEST.json][manifest] | `127e1b038b093b84f34840c9d0090a6a09dd0d89dd99caab81e6dd62e2e7e9e3` |

Download the bounded package using existing OCI credentials:

```bash
QWEN_PROVENANCE_DIR="$(mktemp -d)"
GOMAXPROCS=1 rclone copy \
  oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_workload_handoffs/20261011/qwen-sg520-provenance/20261011T083812Z/ \
  "$QWEN_PROVENANCE_DIR" \
  --max-depth 1 --transfers 64000 --progress --buffer-size 0
cd "$QWEN_PROVENANCE_DIR"
printf '%s  %s\n' \
  33bed7e2a410cc54eb39a670239a918549f79cc687c16d2b00f8f303b310b19b \
  qwen-sg520-workload-provenance.tar.gz | sha256sum --check
sha256sum --check SHA256SUMS
tar -xzf qwen-sg520-workload-provenance.tar.gz
```

Check the archive hash against the table above before extraction. Then follow
the extracted `README.md`, supply the downloaded model directory to
`verify_preflight.py --model-root <directory> --hash-weights`, and use
`render_launch.py` to inspect the exact server/client commands. Rendering does
not allocate resources or execute a capture. Staging groups contain at most
eight shard filenames; every rclone copy must retain `GOMAXPROCS=1`,
`--transfers 64000`, `--progress` and `--buffer-size 0`.

Execution still needs an assigned eight-MI355X lane, full destination hashes,
actual image/source/import checks and reviewed dispatch/tensor observers.
Planning estimates are 2 TiB host RAM, 128 CPUs, 1.6 TiB free NVMe and 60–90
minutes of GPU time after staging; these are not measured SG520 EP1 peaks or an
allocation. Completion requires 64 successful requests, zero failures, 524,288
input tokens, 65,536 output tokens and readable prefill/decode traces for ranks
0–7, with the actual sampled coverage stated.

Fresh capture must establish which of the four remaining task seams execute:
dense GEMM, fused MoE, recurrent GDN and paged attention. EP1 and unified
attention can change their dispatch. An absent old callable is an explicit
mapping/coverage gap; old EP8 fixture passes cannot qualify fresh EP1 workload
provenance. No all-kernel readiness, GEAK/Hyperloom speedup or historical +2.4%
attribution is established by this package.

[latest-record]: https://global.primus-safe.amd.com/hyperloom/api/v1/session-breakdowns/103483/breakdown
[checkpoint]: https://huggingface.co/amd/Qwen3.8-2.4T-A95B-Quark-MXFP4/tree/1122ff36878499eb1e44823999e7de10ddfabd3b
[archive]: oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_workload_handoffs/20261011/qwen-sg520-provenance/20261011T083812Z/qwen-sg520-workload-provenance.tar.gz
[manifest]: oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_workload_handoffs/20261011/qwen-sg520-provenance/20261011T083812Z/MANIFEST.json
