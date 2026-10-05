# SGLang 0.5.20 head-kernel refresh progress

**One task is qualified; the full four-model refresh remains incomplete.** DeepSeek native quant passed final native checks, the framework validator and the trusted six-phase measurement.
The [progress catalog](../../tools/headkernel-sg520-refresh.json) records exact counts, timings and evidence hashes.

## Scope and image

Refresh/new validation covers MiniMax M3, Kimi K3, DeepSeek V4 Pro and GLM-5.3-Flash on **SGLang 0.5.20 only**, using the latest matching evidence.
Qwen3.8 2.4T is excluded; its mappings, tasks and workloads remain unchanged.
The [runtime policy](../../tools/headkernel-runtime-targets.json) pins this exact image:

```text
docker.io/lmsysorg/sglang@sha256:3a78acc9d6c191f1a12c7c67631657580f06af3af562c9ee6d88282a71ec5e96
```

| Model | Current result | Still pending |
| --- | --- | --- |
| DeepSeek V4 Pro | Final native quant passed on 194130; framework validator 194224 passed all 12 checks | Remaining head tasks, traced tensor oracles and full refresh |
| MiniMax M3 | 194049 completed 64 full 8192/1024 requests at C64/TP8; eight EXTEND traces, no DECODE traces | Decode capture and complete fresh task qualification |
| Kimi K3 | 194131 verified 1.56 TB staging, then was preempted during loading at about 18/96 shards | Full workload/profile and task validation |
| GLM-5.3-Flash | Six native groups passed; 194223 verified 328 GB staging and reached native warmup after the help-timeout repair | Full model profile, head capture and task validation |

Existing tasks retain historical captures. The qualified [native-quant task](../../tasks/headkernel_sg520/deepseek-v4-pro__per_group_quant_fp8/README.md)
is the only qualified refreshed task; its PASS does not qualify other tasks or a full-model run.

Both legacy batches, 193902 and 194224, completed **11 non-Qwen tasks: 4 successful command exits / 7 failed exits**. All remain unqualified.
All three MiniMax candidates had binding/completion receipts, but the corrected parser rejects their duplicate timing IDs; a zero-store GPU negative also failed as required.
Fix `d425a3e9` prevents those duplicates from being scored; the earlier successful exits do not qualify that change.
MiniMax 194049 passed its full token gate, but only EXTEND traces cover ranks 0–7; no decode kernel or shape evidence was captured.
GLM groups `topk`, `transform`, `cache`, `logits`, `zero_rope` and `mhc` passed; earlier failures remain preserved. Component checks are not a full GLM profile.

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

Final oracle **`ceac8edb`** passed native compile/correctness/performance on **194130**: all **12 shape/seed/leg checks** passed;
the no-op failed with `invalid dynamic scale`. Graph timing used **10 warmups / 100 samples**, fresh inputs and poisoned outputs per replay,
and sequential isolated workers in randomized leg order. Weighted means were **0.0449486520 ms production / 0.0453507456 ms candidate**; no speedup is claimed.
Framework run **194224** then finalized **PASS for all 12 schema-v3 checks**, with three raw/parsed timing cases and no warnings, skips or timeouts.
The catalog binds that PASS to its tested source/package hashes and finalized report. Prior validator **194017 FAIL** (cache permissions and collapsed score cases)
and the earlier 193855 native pass remain historical evidence. The cache/score fixes and final native oracle were validated; full-model E2E performance was not.
Trusted measurement `reference-194224-1` completed reference and candidate compile/correctness/performance with all three cases.
Its arithmetic mean ratio was **0.9848393502471442×**, with identical reference/candidate source hashes: **no optimized-kernel or E2E gain is claimed**.

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
Native correctness/timing and framework/trusted receipts are in the separately verified validation archive below. Raw profiles/logs stay outside Git; no absolute local path is required.

Reproduce the quant framework check with the committed [MI355X validator config](../../example_configs/validate_headkernel_sg520_quant_mi355x.yaml), copied byte-for-byte from the successful run. The [task README](../../tasks/headkernel_sg520/deepseek-v4-pro__per_group_quant_fp8/README.md) gives the pinned-image `make docker-run` command with `AKA_AITER_JIT_SOURCE` and `AKA_RCLONE_BIN`, and the separate trusted-host command required for accepting agent speedups.

## Published validation evidence

The validation archive is **published and fully read back with SHA-256 verification**: **172 evidence files plus `MANIFEST.json` (173 objects)**, totaling **5,008,529 downloaded bytes**. It includes the clean framework PASS for the single quant task, the trusted six-phase same-source result, final native/no-op receipts, and separately labeled diagnostics. It does not establish a full four-model refresh: GLM full-model profiling and current MiniMax final runs remain pending.

```bash
SG520_VALIDATION_OCI=oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_sg520_refresh/20261005/validation-final-20261005T172445Z
SG520_VALIDATION_DIR=.upstream-assets/sg520/validation-final-20261005T172445Z
GOMAXPROCS=1 rclone copy "$SG520_VALIDATION_OCI" "$SG520_VALIDATION_DIR" \
  --transfers 64000 --progress --buffer-size 0 --multi-thread-streams 0 --checkers 4
python3 - "$SG520_VALIDATION_DIR" <<'PYVERIFY'
from pathlib import Path
import hashlib, json, sys
root = Path(sys.argv[1])
manifest = root / "MANIFEST.json"
assert hashlib.sha256(manifest.read_bytes()).hexdigest() == "83660fc336def213fa374615cea2a96ed3c9687aed798b6f86172837d4bd31be"
for item in json.loads(manifest.read_text())["files"]:
    path = root / item["path"]
    assert path.stat().st_size == item["bytes"], item["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"], item["path"]
print("All validation evidence hashes match the pinned manifest")
PYVERIFY
```

The manifest SHA-256 is `83660fc336def213fa374615cea2a96ed3c9687aed798b6f86172837d4bd31be`. The catalog's `validation_evidence.publication.key_reports` gives exact portable OCI URIs, byte counts and hashes. Key objects below are relative to the archive prefix:

| Evidence | Object | Meaning |
| --- | --- | --- |
| Finalized validator | `validator/snapshot-hardened-7/workspace_MI355X_task_validator/run_20261005_175201_sg520_quant_validator/headkernel_sg520_deepseek-v4-pro__per_group_quant_fp8_20261005_175201/validation_report.yaml` | Clean schema-v3 PASS, all 12 checks, quant task only |
| Trusted control | `trusted/reference-194224-1/measurement/trusted_measurement.json` | Six phases passed with identical source; ratio 0.9848393502 is not an optimization claim |
| Native graph timing | `gpu-runs/quant-final-194130/task/build/performance_report.json` | Three cases, 100 samples per native leg, generated inputs |
| Native no-op rejection | `gpu-runs/quant-final-noop-194130/task/build/performance_candidate_native.log` | Compiled no-op rejected by the oracle without a score |

Some older catalog references are not included in this bundle and are explicitly marked that way; they are not presented as downloadable objects here. Model statuses and incomplete stage coverage remain as recorded above.
