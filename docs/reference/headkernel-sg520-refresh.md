# SGLang 0.5.20 head-kernel refresh progress

> **Current status — 2026-10-09:** Four scoped ready entries are published across 23 active task mappings; 19 mappings still require qualification. See the [current ready-subset index](headkernel-ready-subsets.md) for pinned guides, the exact runtime image, scope limits, and outstanding tasks. The October 5 snapshot below retains its original date and scope.

**One task is qualified; the full four-model refresh remains incomplete.** DeepSeek native quant passed final native checks, the framework validator and the trusted six-phase measurement. The [progress catalog](../../tools/headkernel-sg520-refresh.json) records exact counts, timings and evidence hashes.

## Scope and image

Refresh/new validation covers MiniMax M3, Kimi K3, DeepSeek V4 Pro and GLM-5.3-Flash on **SGLang 0.5.20 only**, using the latest matching evidence.
Qwen3.8 2.4T is excluded; its mappings, tasks and workloads remain unchanged. The [runtime policy](../../tools/headkernel-runtime-targets.json) pins this exact image:

```text
docker.io/lmsysorg/sglang@sha256:3a78acc9d6c191f1a12c7c67631657580f06af3af562c9ee6d88282a71ec5e96
```

| Model | Current result | Still pending |
| --- | --- | --- |
| DeepSeek V4 Pro | Final native quant passed on 194130; framework validator 194224 passed all 12 checks | Remaining head tasks, traced tensor oracles and full refresh |
| MiniMax M3 | [194299 trace/shape review](headkernel-sg520-minimax.md): passed the full 64-request 8192/1024 workload; 16 prefill/decode traces verified after retirement, stock native JIT | Kernel contracts, tensor oracles and task qualification |
| Kimi K3 | 194131 verified 1.56 TB staging, then was preempted during loading at about 18/96 shards | Full workload/profile and task validation |
| GLM-5.3-Flash | [194292 trace/shape review](headkernel-sg520-glm.md): passed the same full workload; 16 prefill/decode traces verified after retirement, SGLang 0.5.20 plus 18 backport files | Kernel contracts, tensor oracles and task qualification |

The [native-quant task](../../tasks/headkernel_sg520/deepseek-v4-pro__per_group_quant_fp8/README.md) is the only qualified refreshed task; its PASS does not qualify other tasks or a full-model run.
Both legacy batches, 193902 and 194224, completed **11 non-Qwen tasks: 4 successful command exits / 7 failed exits**. All remain unqualified.
All three MiniMax candidates had binding/completion receipts, but the corrected parser rejects their duplicate timing IDs; a zero-store GPU negative also failed as required.
Fix `d425a3e9` prevents those duplicates from being scored; the earlier successful exits do not qualify that change.
MiniMax 194299 and GLM 194292 each completed **64 × 8192/1024**, **C64 / TP8 / context 9218**, seed 42, error-free, with ranks 0–7 in both stages.
Each requested **eight sampled steps per stage**; **all graph replays are not captured**, and live argument/shape completeness and tensor oracles remain pending.
MiniMax's earlier 194049 capture lacked DECODE; it remains historical. GLM's six native component passes are separate from its new full-model profile, which is not stock-image equivalent.

## DeepSeek workload and sampled coverage

Run `deepseek-sg520-context9218-193529` completed **64 × 8192/1024 requests**, **C64 / TP8**, seed 42, error-free: 524,288 input and 65,536 output tokens.
Context **9,218** includes one TP-worker slot and one scheduler slot (`8192 + 1024 + 2`), avoiding truncation to 1,022 output tokens.
This bounded profile did not replay the full online-CI accuracy/performance test.
Independent streaming review verified **16 complete traces**, original hashes and read stability, with ranks **0–7** in both stages.
The original native status remains `incomplete`: its 256 MiB summary bound rejected prefill traces. Independent review recovered sampled trace/shape evidence without rewriting that status.
Prefill records **eight `EXTEND` steps/rank**, each `bs=1, toks=8192`. Decode has **eight CPU `hipGraphLaunch` calls/rank**, but only **one/rank** with explicitly correlated GPU kernels; these are not eight fully traced GPU replays.
The 246,400 prefill and 21,558 decode kernel occurrences are unscaled observations, not proof of full-workload GPU-step coverage.

## Native per-group FP8 quantization

`aiter::per_group_quant_hip` calls native `aiter::dynamic_per_token_scaled_quant`. Inputs are contiguous BF16; outputs are same-shape FP8 `e4m3fn`. Group size is **128**, scales are **FP32**, and `transpose_scale=True`.
Wrapper-call counts are identical on ranks 0–7 and are not extrapolated to the full workload.

| Input/output shape | Native input view | Allocated scale shape; stride | Calls/rank | Calls/all 8 ranks |
| --- | --- | --- | ---: | ---: |
| `[8192,1536]` | `[98304,128]` | `[8192,12]`; `[12,1]` | 728 | 5,824 |
| `[8192,2048]` | `[131072,128]` | `[8192,16]`; `[16,1]` | 488 | 3,904 |
| `[8192,7168]` | `[458752,128]` | `[8192,56]`; `[56,1]` | 968 | 7,744 |

Physical scale writes are **`scale[group_column * M + row]`**, despite the allocated `[M,N/128]` shape/stride; row-major indexing changes the contract. Group-32 E8M0 scales (488 calls/rank) remain separate.
Decode's 212 quantization GPU occurrences/rank have no argument events; do not infer shapes from prefill.
Historical pre-hardening validation passed **3 cases × seeds 0/1 × production/candidate = 12 rows**; its two translation units compiled in **17.1 seconds**.
Both legs passed independent correctness on seeded legal values at observed dimensions; original traced tensors remain unavailable.
Historical graph timing used **10 warmups / 100 alternating paired samples**, with oracle-valid graphs: **0.0269248668 ms production / 0.0273288870 ms candidate**. Those timings do not qualify the revised harness or establish gains.
Final oracle **`ceac8edb`** passed native compile/correctness/performance on **194130**, all **12 shape/seed/leg checks**; the no-op failed with `invalid dynamic scale`.
Graph timing used **10 warmups / 100 samples**, fresh inputs and poisoned outputs per replay, sequential isolated workers and randomized leg order. Means were **0.0449486520 ms production / 0.0453507456 ms candidate**; no speedup is claimed.
Framework run **194224** then finalized **PASS for all 12 schema-v3 checks**, with three raw/parsed timing cases and no warnings, skips or timeouts.
The catalog binds that PASS to its tested hashes/report. Prior **194017 FAIL** (cache permissions and collapsed score cases) and 193855 native pass remain historical. The fixes and final oracle were validated; full-model E2E performance was not.
Trusted measurement `reference-194224-1` completed reference and candidate compile/correctness/performance with all three cases. Its arithmetic mean ratio was **0.9848393502471442×**, with identical reference/candidate source hashes: **no optimized-kernel or E2E gain is claimed**.

## Source differences requiring current dispatch binding

| Family | Difference from historical task source |
| --- | --- |
| MiniMax decode score | Math/ABI unchanged; fixed ROCm launch configuration and token-block selection replace autotuning. |
| DeepSeek FlyDSL MoE stage 1 | Adds `v2_output_layout` and a new implementation/grid path; migrate the consistent family. |
| DeepSeek Opus MoE stage 2 | `OpusA8W4LaunchConfig` validates kernel instance, packed weights and scales; not an old-adapter drop-in. |
| Kimi MLA | Adds `tune_mla`, `forced_kv_splits` and launch planning; old candidate launchers are not current stock code. |
| Kimi MoE stages 1/2 | Implementation, compiler/reduction and v2-layout changes; three old source paths are absent. |
| GLM fused MoE | Public signatures unchanged; HIP top-k-1 reduction bypass and clamp-path restrictions change internally. |

Historical gains are not reused. Source differences do not establish gains; remaining work requires current binding, kernel contracts, tensor oracles and finalized task validation.

## Published profile evidence

**22 OCI objects / 447,262,076 bytes** were read back and SHA-256 verified: 16 profiles, five trace-review files and `MANIFEST.json`. Download with:

```bash
rclone copy \
  oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_sg520_refresh/20261005/deepseek-sg520-context9218-193529 \
  .upstream-assets/sg520/deepseek-sg520-context9218-193529 --transfers 64000 --progress
```

Verify `MANIFEST.json` against `evidence.publication.manifest_sha256` in the catalog, then every listed file's size and SHA-256.
Native correctness/timing and framework/trusted receipts are in the separately verified validation archive below. Raw profiles/logs stay outside Git; no absolute local path is required.
Reproduce the quant framework check with the committed [MI355X validator config](../../example_configs/validate_headkernel_sg520_quant_mi355x.yaml), copied byte-for-byte from the successful run. The [task README](../../tasks/headkernel_sg520/deepseek-v4-pro__per_group_quant_fp8/README.md) gives the pinned-image `make docker-run` command with `AKA_AITER_JIT_SOURCE` and `AKA_RCLONE_BIN`, and the separate trusted-host command required for accepting agent speedups.

## Published validation evidence

The earlier validation archive was fully read back and SHA-256 verified: **172 evidence files plus `MANIFEST.json` (173 objects), 5,008,529 bytes**. It contains quant-only qualification and separate diagnostics, and predates the completed MiniMax/GLM profiles.

```bash
rclone copy \
  oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_sg520_refresh/20261005/validation-final-20261005T172445Z \
  .upstream-assets/sg520/validation-final-20261005T172445Z --transfers 64000 --progress --buffer-size 0 --multi-thread-streams 0 --checkers 4
```

Verify `MANIFEST.json` SHA-256 **`83660fc336def213fa374615cea2a96ed3c9687aed798b6f86172837d4bd31be`**, then every listed file's size and SHA-256. The catalog's `validation_evidence.publication.key_reports` gives exact portable OCI URIs and hashes for the finalized validator, trusted same-source control, native timing and no-op rejection. References absent from that archive remain explicitly marked.
The fresh profile archives are also fully read back and SHA-256 verified under the same OCI parent: **minimax-m3** `minimax-native-profile-194299` (27 objects / 373,279,062 bytes), manifest `6327d9c7cb072f10c60a4ee32020a5510419bc1edeaf34b2dd75aa737bf05aea`; **glm-5.3-flash** `glm-native-jit-194292` (52 objects / 330,462,859 bytes), manifest `047e532a6ab7449fc23d4a6dbbf62b487a17b0f8ee0603de6a83999d816ef3bb`.
The catalog's `profile_publications` records both complete OCI paths. Copy each with `rclone copy <oci_root> <destination> --transfers 64000 --progress`, then verify its manifest and file hashes. Profile completion does not qualify kernel contracts, tensor oracles, additional tasks or E2E gains.
