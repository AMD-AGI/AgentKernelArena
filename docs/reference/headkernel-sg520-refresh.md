# SGLang 0.5.20 head-kernel refresh progress

Native DeepSeek quantization correctness and graph timing passed. **The full four-model refresh remains incomplete.**
The [progress catalog](../../tools/headkernel-sg520-refresh.json) records exact counts, timings and evidence hashes.

## Scope and image

Refresh/new validation covers MiniMax M3, Kimi K3, DeepSeek V4 Pro and GLM-5.3-Flash on **SGLang 0.5.20 only**.
Use the latest matching evidence. Qwen3.8 2.4T is excluded; its mappings, tasks and workloads remain unchanged.
The [runtime policy](../../tools/headkernel-runtime-targets.json) pins this exact image:

```text
docker.io/lmsysorg/sglang@sha256:3a78acc9d6c191f1a12c7c67631657580f06af3af562c9ee6d88282a71ec5e96
```

| Model | Current result | Still pending |
| --- | --- | --- |
| DeepSeek V4 Pro | Full token gate; sampled trace review; native quantization correctness and graph timing passed | Framework task-validator PASS and remaining head-kernel refresh |
| MiniMax M3 | Retired incomplete at 14:38 UTC: 11/59 checkpoint shards after about two minutes; projected load exceeded preemption allowance | Completed startup and fresh capture |
| Kimi K3 | Fresh run prepared but not executed | Native run, capture and readiness review |
| GLM-5.3-Flash | P2 source correction applied; 39 CPU checks passed | Native build, GPU correctness and capture |

Existing tasks retain historical captures. The new
[experimental native-quant package](../../experimental/headkernel_sg520/deepseek-v4-pro__per_group_quant_fp8/README.md)
is outside automatic `tasks/` discovery while framework task validation is pending.

## DeepSeek workload and sampled coverage

Run `deepseek-sg520-context9218-193529` completed **64 requests**, each with **8,192 input and 1,024 output tokens**,
at **C64 / TP8**, seed 42, without request errors: 524,288 input and 65,536 output tokens in total.
Context **9,218** includes one TP-worker slot and one scheduler slot (`8192 + 1024 + 2`), avoiding truncation to 1,022 output tokens.
This bounded profile did not replay the full online-CI accuracy/performance test.

Independent streaming review verified **16 complete traces**, original hashes and read stability, with ranks **0–7** in both stages.
The original native status remains `incomplete`: its 256 MiB summary bound rejected prefill traces.
Independent review recovered sampled trace/shape evidence without rewriting that status.

Prefill records **eight `EXTEND` steps per rank**, each `bs=1, toks=8192`. Decode records **eight CPU `hipGraphLaunch`
calls per rank**, but only **one per rank** has explicitly correlated GPU graph kernels; these are not eight fully traced GPU replays.
The 246,400 prefill and 21,558 decode kernel occurrences are unscaled observations, not proof of full-workload GPU-step coverage.

## Native per-group FP8 quantization

`aiter::per_group_quant_hip` calls native `aiter::dynamic_per_token_scaled_quant`. Inputs are contiguous BF16;
outputs are same-shape FP8 `e4m3fn`. Group size is **128**, scales are **FP32**, and `transpose_scale=True`.
Wrapper-call counts are identical on ranks 0–7 and are not extrapolated to the full workload.

| Input/output shape | Native input view | Allocated scale shape; stride | Calls/rank | Calls/all 8 ranks |
| --- | --- | --- | ---: | ---: |
| `[8192,1536]` | `[98304,128]` | `[8192,12]`; `[12,1]` | 728 | 5,824 |
| `[8192,2048]` | `[131072,128]` | `[8192,16]`; `[16,1]` | 488 | 3,904 |
| `[8192,7168]` | `[458752,128]` | `[8192,56]`; `[56,1]` | 968 | 7,744 |

Physical scale writes are **`scale[group_column * M + row]`**, despite the allocated `[M,N/128]` shape/stride;
ordinary row-major indexing changes the contract. Group-32 E8M0 scales (488 calls/rank) remain a separate family.
Decode's 212 quantization GPU occurrences/rank have no argument events; do not infer shapes from prefill.

A native GPU check passed **3 cases × seeds 0/1 × production/candidate = 12 rows**. The candidate's two translation
units compiled in **17.1 seconds**. Inputs were seeded legal values at observed dimensions; original traced tensor values
remain unavailable. Production AITER and the separately compiled candidate both passed independent correctness.

Graph timing passed all three cases with **10 warmups and 100 alternating paired samples**, with both graphs oracle-valid.
Weighted means were **0.0269248668 ms production** and **0.0273288870 ms candidate**, using the same unoptimized source.
**No speedup or E2E gain is claimed.** Framework task validation remains pending.

## Source differences requiring current dispatch binding

| Family | Difference from historical task source |
| --- | --- |
| MiniMax decode score | Math/ABI unchanged; fixed ROCm launch configuration and token-block selection replace autotuning. |
| DeepSeek FlyDSL MoE stage 1 | Adds `v2_output_layout` and a new implementation/grid path; migrate the consistent family. |
| DeepSeek Opus MoE stage 2 | `OpusA8W4LaunchConfig` validates kernel instance, packed weights and scales; not an old-adapter drop-in. |
| Kimi MLA | Adds `tune_mla`, `forced_kv_splits` and launch planning; old candidate launchers are not current stock code. |
| Kimi MoE stages 1/2 | Implementation, compiler/reduction and v2-layout changes; three old source paths are absent. |
| GLM fused MoE | Public signatures unchanged; HIP top-k-1 reduction bypass and clamp-path restrictions change internally. |

Source differences and formatting-only changes do not establish gains. Historical gains are not reused.
Remaining work requires current binding, captures, correctness/performance evidence and finalized task validation.

## Published profile evidence

**22 OCI objects / 447,262,076 bytes** were read back and SHA-256 verified:
16 profiles, five curated trace-review files and `MANIFEST.json`. Download with:

```bash
rclone copy \
  oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_sg520_refresh/20261005/deepseek-sg520-context9218-193529 \
  .upstream-assets/sg520/deepseek-sg520-context9218-193529 --transfers 64000 --progress
```

Verify `MANIFEST.json` against `evidence.publication.manifest_sha256` in the catalog, then every listed file's size and SHA-256.
Native correctness/timing receipts are cataloged separately. Raw profiles/logs stay outside Git; no absolute local path is required.
