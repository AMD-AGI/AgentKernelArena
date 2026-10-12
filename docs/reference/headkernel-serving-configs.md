# Recorded serving configurations for the five models

This reference describes preserved serving configurations as of **2026-10-12**. DeepSeek, GLM, Kimi and MiniMax have the SGLang 0.5.20 capture records below. Qwen's latest recorded metadata reports 0.5.17, its callable fixtures came from 0.5.18, and its 0.5.20 capture remains proposed. These records do not assert that servers are currently running or add GPU results, task qualification or serving/E2E gains. See the [current task and evidence index](headkernel-ready-subsets.md) for qualification status.

The image source of truth is **`images.sglang_v0520.pull_reference` in the [runtime catalog](../../tools/headkernel-runtime-targets.json)**. The four SG0.5.20 captures used that pinned Linux amd64 / MI355X image. Qwen's proposed SG0.5.20 recipe targets it; that does not identify Qwen's older image. Preserve the capture-specific overlays and model metadata described below.

## Four recorded SG0.5.20 captures

All four selected runs used **64 prompts, ISL 8192, OSL 1024, concurrency 64, maximum running requests 64, context length 9218, TP8 / EP1 / DP1 and client seed 42**. Each first ran an unscored 64-request ISL8192/OSL16/C64 warmup, then flushed the prefix cache.

**C64 is a serving concurrency limit, not a fixed kernel M dimension.** Active request batches, prefill token counts, expert routing and rank-local operands vary. Task case manifests and captured invocation records define kernel shapes. A configured CUDA graph maximum likewise does not establish that the workload reached that batch size.

The table uses the immutable archive's Kimi run `194550`; it must not be assigned the later `194958` run's server seed. The sources below bind every column to its named run.

| Recorded field | DeepSeek V4 Pro | GLM 5.3 Flash | Kimi K3 | MiniMax M3 |
| --- | --- | --- | --- | --- |
| Run | `deepseek-capture-csr-v7-194603` | `glm-final-capture-v4-194841` | `kimi-actual-v4-no-stack-194550` | `minimax-original-head-v3-194969` |
| Chunked prefill | 8192 | 8192 | 16384 | 16384 |
| Maximum prefill tokens | 16384 | 16384 | 16384 | 16384 |
| Decode graph backend / resolved maximum batch | full / **512** | full / **64** | full / **64** | full / **512** |
| Prefill graph backend | disabled | disabled | disabled | disabled |
| Client/workload seed | **42** | **42** | **42** | **42** |
| Resolved server random seed | **324272933** | **730852689** | **409199678** | **47658906** |
| Attention backend | `dsv4` | `dsa`; TileLang prefill and decode | `triton` | `triton` |
| MoE runner field | `auto` | `aiter` | `auto` | `auto` |
| MoE all-to-all field | `none` | `none` | `none` | `none` |
| Quantization server field | `null` | `fp8` | `null` | `quark` |
| Dtype / KV cache dtype fields | `auto` / `fp8_e4m3` | `bfloat16` / `bfloat16` | `bfloat16` / `auto` | `bfloat16` / `bfloat16` |
| Page size | 256 | 64 | 1 | 1 |
| Memory fraction | 0.90 | 0.80 | 0.85 | 0.8626 |
| Radix cache disabled | yes | no | yes | no |

These are recorded server-info fields. `null` quantization means no explicit value in that field, not unquantized checkpoint weights; `auto` remains a runtime selection. Client and server seeds are separate. Kimi and GLM explicitly requested decode graph maximum 64; DeepSeek and MiniMax omitted that flag and resolved to 512 in `cuda_graph_config.decode.max_bs`.

The benchmark CLI used backend `sglang`, dataset `random` with **native ShareGPT prompt sampling**, random input/output lengths 8192/1024, random-range-ratio 1.0, 64 prompts, max-concurrency 64, request-rate `inf`, seed 42, tokenized prompts and internal warmup-requests 0. The separate OSL16 warmup above still occurred. All four benchmark logs record temperature 0.0, top-p 1.0 and `disable_ignore_eos=False`. The pinned SGLang request builder sends `temperature=0.0`, `max_new_tokens=1024` and `ignore_eos=True`; it omits top-p from the payload when the CLI value is 1.0. This is not a random-token-ID workload.

## Model identities and runtime switches

The archive guides bind the exact model metadata and weight identities. Their retrieval instructions avoid a dependency on the original node or shared model directory.

| Model | Checkpoint reference |
| --- | --- |
| DeepSeek | [`deepseek-ai/DeepSeek-V4-Pro` at `b5968e91`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/b5968e9190ef611bbf34a7229255be88a0e937c1) |
| Kimi | [`moonshotai/Kimi-K3` at `9f62e4e9`](https://huggingface.co/moonshotai/Kimi-K3/tree/9f62e4e9fffbd0a83ddd60e1c209d828994b3569) |
| MiniMax | [`amd/MiniMax-M3-MXFP4` at `b83d14e3`](https://huggingface.co/amd/MiniMax-M3-MXFP4/tree/b83d14e3d64bf373a207f3c2a7e9f0b0f1e7fc3a); apply the original model metadata from its guide |
| GLM | Use the [GLM guide][glm-guide] and its manifest-bound model/config/index references. A public repository revision is not established by these records; do not substitute an assumed one. |
| Qwen | [`amd/Qwen3.8-2.4T-A95B-Quark-MXFP4` at `1122ff36`](https://huggingface.co/amd/Qwen3.8-2.4T-A95B-Quark-MXFP4/tree/1122ff36878499eb1e44823999e7de10ddfabd3b) |

The recorded model-specific settings matter alongside the image:

- **DeepSeek:** `AITER_BF16_FP8_MOE_BOUND=0`, `SGLANG_DEFAULT_THINKING=1`, `SGLANG_DP_USE_GATHERV=1`, `SGLANG_DSV4_FP4_EXPERTS=true`, `SGLANG_DSV4_REASONING_EFFORT=max`, `SGLANG_HACK_FLASHMLA_BACKEND=unified_kv_triton`, `SGLANG_USE_ROCM700A=0`. The launch disabled shared-expert fusion, set SWA/full-token ratio 0.1 and selected parsers `deepseekv4` / `deepseek-v4`. Capture instrumentation is separate from image source; the guide does not prescribe a blanket backport.
- **GLM:** `SGLANG_USE_AITER=1`, AITER MoE, TileLang DSA, multithread model loading and watchdog 1200 seconds. **18 recorded backport/overlay files were used; stock-image equivalence is false.** Its 16 stage/rank traces are sampled, with eight profiler steps requested per stage; they do not establish every replay or shape/scalar. Eight inaccessible rank directories from failed v3 remain disclosed in the handoff. The [FP8 GEMM scoring reference](headkernel-ready-subsets.md#glm-fp8-gemm-primary-and-diagnostic-timings) explains the matched native-production/port score and the separate frozen-port diagnostic.
- **Kimi:** `AITER_FLYDSL_FORCE=1`, `AITER_SITUV2_A8W4=1`, `SGLANG_AITER_K3_OPT=1`, `SGLANG_USE_AITER=1`; both parsers `kimi_k3`, multithread model loading and watchdog 1200 seconds. The portable wrapper adds no extra model/kernel backport and preserves the image and capture-hook source hashes.
- **MiniMax:** `SGLANG_USE_AITER=1`, `USE_ROCM_AITER_ROPE_BACKEND=0`. The **archived model metadata overlay is required** because original `config.json` differs from the public default. Its SHA256 is `927ca3a58eb765bd250fc57202908d34879c3e193d64de7af809a95b1fe5bd47`. Primary run 194969 retained native capture records and did not start Torch profiling; earlier sampled traces remain separate.

## Qwen: recorded metadata, fixtures and proposed capture

The [Qwen provenance guide](headkernel-qwen-workload-provenance.md) and its [published bounded package][qwen-package] distinguish these three records. **No SG0.5.20 Qwen serving capture has run.**

| Field | Latest recorded metadata: 103483 | Historical callable fixtures | Proposed SG0.5.20 primary |
| --- | --- | --- | --- |
| Version/status | Reports **0.5.17**; exact image and source revision unknown | **0.5.18** extraction | **Prepared, not executed** |
| TP / EP | 8 / 1 | 8 / 8 | 8 / 1 proposed |
| ISL / OSL / concurrency | 8192 / 1024 / 64 | 8192 / 1024 / 64 provenance | 8192 / 1024 / 64 proposed |
| Context / chunked prefill | 11264 / 16384 | 11264 / 16384 | 11264 / 16384 proposed |
| Page / memory fraction | 16 / 0.8 | 1 / effective **0.68**; script declared 0.8 | 16 / 0.8 proposed |
| KV / Mamba state dtype | `auto` / BF16 | `auto` / FP32 | `auto` / BF16 proposed |
| Attention | AITER unified attention, AITER all-reduce fusion | AITER | Same EP1 extras proposed |
| Server random seed | **585727691**, recorded flag | **168105520**, recorded effective log | **585727691 requested**; no resolved server observed |
| Client seed / request count / warmup | **Unknown / unknown / unknown** | Client seed/count not asserted; server/client warmups skipped | **42 / 64 / separate 64-request OSL16 warmup**, new protocol choices |
| CUDA graphs | Maximum batch unknown; missing capture traces do not prove disabled | Disabled | No observed effective graph settings |

Record 103483 used maximum running requests 64, maximum prefill tokens 16384, four checkpoint-prefetch threads, watchdog 1800 seconds and disabled radix caching. Its reviewed environment includes `SGLANG_USE_AITER=1`, `AITER_FLYDSL_FORCE=1`, `SGLANG_MAMBA_SSM_DTYPE=bfloat16`, `SGLANG_USE_AITER_UNIFIED_ATTN=1` and `SGLANG_MOE_COPY_WEIGHT_VIEWS_BEFORE_H2D=1`. These are recipe-specific settings. The SG520 proposal retains those extras and adds no model/kernel source override.

The old record's sampling and ignore-EOS settings were not recovered. The proposed native random-ShareGPT command selects request-rate `inf`, random-range-ratio 1, tokenized prompts and internal warmup 0, without explicit temperature/top-p/ignore-EOS flags; it has not executed. Its client seed 42 and 64-request protocol must not be attributed to record 103483. The separate r11 serving contract used TP8/EP8, memory fraction 0.8, 192 prompts and eight warmups, and did not disable graphs; it differs from the fixture extraction.

## Portable evidence sources

These are rclone OCI object-store links. The [workload archive index](headkernel-ready-subsets.md#workload-archives-in-oci) explains retrieval using OCI credentials without the original Crusoe, Spur or node session. The immutable index-v2 remains unchanged; GLM and Qwen are separate publications. Model weights and dataset/tensor bodies remain external where the guides say so.

| Model | Guide | Recorded server info | Directory containing the original request and benchmark log |
| --- | --- | --- | --- |
| DeepSeek | [Guide][ds-guide] | [SERVER-INFO-BEFORE.json][ds-server] | `runs/deepseek-capture-csr-v7-194603/` in that package |
| GLM | [Guide][glm-guide], [publication hashes](glm-workload-handoff-20261012.json) | [SERVER-INFO-BEFORE.json][glm-server] | `capture/glm-final-capture-v4-194841/` in that package |
| Kimi | [Guide][kimi-guide] | [SERVER-INFO-BEFORE.json][kimi-server] | `run/` in the immutable `194550-r2` package |
| MiniMax | [Guide][minimax-guide] | [SERVER-INFO-BEFORE.json][minimax-server] | `retained/top5-completion-20261005/runs/minimax-original-head-v3-194969/` in that package |
| Qwen | [Provenance guide](headkernel-qwen-workload-provenance.md) | No complete current SG520 server record | [Published package][qwen-package]: sanitized record 103483, `LAUNCH-PLAN.json` and rendered commands |

[ds-guide]: oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_workload_handoffs/20261009/deepseek-v4-pro/snapshot-20261009T105053Z/GUIDE.md
[ds-server]: oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_workload_handoffs/20261009/deepseek-v4-pro/snapshot-20261009T105053Z/runs/deepseek-capture-csr-v7-194603/SERVER-INFO-BEFORE.json
[glm-guide]: oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_workload_handoffs/20261012/glm-5.3-flash/glm-final-capture-v4-194841/GUIDE.md
[glm-server]: oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_workload_handoffs/20261012/glm-5.3-flash/glm-final-capture-v4-194841/capture/glm-final-capture-v4-194841/SERVER-INFO-BEFORE.json
[kimi-guide]: oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_workload_handoffs/20261009/kimi-k3/lean-c39e94c8-kimi-actual-v4-no-stack-194550-r2/README.md
[kimi-server]: oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_workload_handoffs/20261009/kimi-k3/lean-c39e94c8-kimi-actual-v4-no-stack-194550-r2/run/SERVER-INFO-BEFORE.json
[minimax-guide]: oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_workload_handoffs/20261009/minimax-m3/ef7474dd-194969-20261009T110900Z/GUIDE.md
[minimax-server]: oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_workload_handoffs/20261009/minimax-m3/ef7474dd-194969-20261009T110900Z/retained/top5-completion-20261005/runs/minimax-original-head-v3-194969/SERVER-INFO-BEFORE.json
[qwen-package]: oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_workload_handoffs/20261011/qwen-sg520-provenance/20261011T083812Z/qwen-sg520-workload-provenance.tar.gz
