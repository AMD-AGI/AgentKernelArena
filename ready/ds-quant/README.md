# Ready starter: DeepSeek per-group FP8 quantization

The ready entry is **`headkernel_sg520/deepseek-v4-pro__per_group_quant_fp8`**.
Optimize only `dynamic_per_group_scaled_quant_kernel` in
[`source/quant_kernels.cu`](../../tasks/headkernel_sg520/deepseek-v4-pro__per_group_quant_fp8/source/quant_kernels.cu).
The source guard permits the declared GPU body and its local device helpers;
the translation unit, signature, launch logic, references and harness stay frozen.

This handoff uses measured commit
`0b098bd1beaa7c15e33cf6ecb35c00cfa990c07f`, source SHA-256
`5c9d1e56d978d45ccdc099806267d75aad9074ab837b07cca32b8ae21dcccfec`,
and task tree `b0019f08fcea66443cf73e2a6600bdbbbf75b231`.
The task is byte-identical to its full framework validation at `20c1a949`:
all 12 checks PASS, with no errors, warnings or policy findings. The fresh
repeat passed all six trusted phases and the timing-quality gate. Separate
no-op and wrong-output source variants compiled and were rejected by the
dynamic-scale oracle. Those invalid-source runs stop at the first failing
check; an exhaustive per-case source-mutant sweep is not claimed.

Only this task is selected by the [ready config](../../example_configs/ready_ds_quant_mi355x.yaml)
and [readiness receipt](READY.json). Other tasks, models and historical workload
coverage receive no readiness claim from this handoff. Earlier provisional task
metadata is preserved to retain the exact qualified task identity.

## Cases, metric and retained timings

The three mandatory BF16 input shapes and observed counts are unchanged.
Counts cover **eight sampled prefill steps per rank**, with matching counts
across the eight captured ranks; their sum is 2,184. They are not extrapolated
to the complete serving workload.

All values below are milliseconds from the completed, unchanged-source repeat:

| Input shape | `trace_call_count` | Protected reference submission | Protected candidate submission | Production native in reference run | Production native in candidate run | Raw reference/candidate ratio |
|---|---:|---:|---:|---:|---:|---:|
| `[8192,1536]` | 728 | 0.0517429501 | 0.0462269099 | 0.0512417500 | 0.0526122000 | 1.1193253073x |
| `[8192,2048]` | 488 | 0.0533714703 | 0.0432521298 | 0.0581658499 | 0.0535145800 | 1.2339616691x |
| `[8192,7168]` | 968 | 0.0687592399 | 0.0682228097 | 0.0709236299 | 0.0693220005 | 1.0078629170x |

The trusted comparison's raw metric is the **arithmetic mean of the three
per-case protected reference/candidate ratios**, `1.1203832978x`. It is not
weighted by `trace_call_count`. Trace-weighted means and the production-native
columns are retained diagnostics. Each protected submission freshly compiles
its submitted source into the same guarded candidate-native entrypoint;
production-native diagnostics run in separate processes.

**Both submissions have identical executable source.** The gate therefore
records `gain_eligible=false`, `accepted_gain=false`, and a null accepted
arithmetic-mean speedup. The raw ratio is a control observation and is not an
optimization or whole-workload/serving E2E gain.

Every native timing series retains all 100 raw samples: 1,200 samples across
the two outer performance phases, three cases and two native legs. All score
and production-native series pass the approved catastrophic-instability gate.
Four samples exceed twice their own series median; the largest is 3.8301x its
median. Moderate timing variation remains relevant when reviewing a future
gain. The earlier 68 ms outlier comparison remains archived with its original
rejection; this authorized single repeat does not erase or rescore that run.

## Exact environment and first run

The tested runtime is SGLang 0.5.20. Its verified tag and immutable image identity
are recorded in the [`sglang_v0520` runtime entry](../../tools/headkernel-runtime-targets.json).

Use Linux amd64, a MI355X (`gfx950`), ROCm-compatible Docker access, Git, Python 3
with PyYAML, and rclone. Put the checkout, generated workspaces, result folders
and scratch on local NVMe. The unchanged Docker runner seeds AITER under host
`/tmp`, so its backing storage and Docker's data directory need adequate local
space. Ordinary credentials for the chosen optimization agent are required.

No model weights or external captured tensors need downloading for this task.
Inputs are seeded legal BF16 values at the three traced shapes. The ABI keeps
group size 128, FP8 E4M3FN output, FP32 scales, `transpose_scale=True`, and absent
optional scale/row-count inputs. Scales are allocated as contiguous
`[M,N/128]` but written at physical offset `group*M+row`; the independent oracle
checks that layout, exact FP8 bytes, scale rounding and input immutability.

From the root of this handoff checkout:

```bash
python3 ready/ds-quant/verify.py
QUANT_REPO="$PWD"
QUANT_COMMIT=0b098bd1beaa7c15e33cf6ecb35c00cfa990c07f
QUANT_STORAGE=/path/to/local-nvme/quant-ready
mkdir -p "$QUANT_STORAGE/scratch" "$QUANT_STORAGE/tmp"
export TMPDIR="$QUANT_STORAGE/tmp"
export GOMAXPROCS=1
export RCLONE_BUFFER_SIZE=0
export AKA_DOCKER_IMAGE=docker.io/lmsysorg/sglang@sha256:3a78acc9d6c191f1a12c7c67631657580f06af3af562c9ee6d88282a71ec5e96
export AKA_AITER_JIT_SOURCE=/sgl-workspace/aiter/aiter/jit
export AKA_RCLONE_BIN="$(command -v rclone)"
docker pull "$AKA_DOCKER_IMAGE"
make docker-smoke
make docker-check-agents CONFIG=example_configs/ready_ds_quant_mi355x.yaml
make docker-run CONFIG=example_configs/ready_ds_quant_mi355x.yaml
```

The one-task config uses the existing Claude Code template. Configure it
normally, or change only the agent block to another supported integration.
For a fresh full task review, use the existing
[`validate_headkernel_sg520_quant_mi355x.yaml`](../../example_configs/validate_headkernel_sg520_quant_mi355x.yaml).
Its task-validator/Codex backend settings match the archived framework PASS.

Correctness preserves seeds `[0,1]` for both native legs and all three cases.
Performance preserves 10 warmups and 100 graph replays per case/native leg,
with fresh inputs, poisoned outputs, oracle checks and input-immutability
checks around every measured replay. No timing-method change accompanies this
handoff.

## Mandatory trusted comparison and quality handling

An ordinary Arena score alone is not acceptance evidence. After the optimization
worker stops, return to the trusted checkout and run the pinned native evaluator.
Choose an available allocated render device and new output directory:

```bash
cd "$QUANT_REPO"
QUANT_CANDIDATE=/path/to/stopped/task-workspace
QUANT_RENDER_DEVICE=/dev/dri/renderD128
python3 src/tools/trusted_native_eval.py \
  --repo "$QUANT_REPO" --commit "$QUANT_COMMIT" \
  --task tasks/headkernel_sg520/deepseek-v4-pro__per_group_quant_fp8 \
  --agent-workspace "$QUANT_CANDIDATE" \
  --candidate "$QUANT_CANDIDATE/source/quant_kernels.cu" \
  --render-device "$QUANT_RENDER_DEVICE" \
  --output "$QUANT_STORAGE/trusted-candidate" \
  --scratch-dir "$QUANT_STORAGE/scratch" --timeout 7200
```

The actual device number is host-specific. The evaluator supplies the protected
reference/harness from the qualified Git commit and admits only the guarded
candidate source. Each of six phases uses a fresh container/build directory and
verified private image cache. Preserve a budget for the complete protocol; the
qualified repeat reserved seven 7,200-second setup/phase allowances plus 120
seconds of task reserve and 900 seconds of lifecycle reserve. Do not shorten
the protocol to fit a lease.

The approved `benchmark_quality` implementation from `47aaa883` is already
built into this pinned evaluator. It runs only after all six reports and every
raw sample are retained, and it checks both comparison and production-native
diagnostic series. A rejected series rejects the whole comparison. Same-source
controls receive no accepted speedup.

Independently review the retained bundle before accepting a claimed gain:

```bash
python3 -m src.tools.benchmark_quality_gate \
  --repo "$QUANT_REPO" --commit "$QUANT_COMMIT" \
  --task tasks/headkernel_sg520/deepseek-v4-pro__per_group_quant_fp8 \
  --evidence "$QUANT_STORAGE/trusted-candidate" \
  --candidate-source "$QUANT_CANDIDATE/source/quant_kernels.cu" \
  --output "$QUANT_STORAGE/quality-reviewed.json"
```

This CPU-only check validates complete native reports, identities, source
boundary, report hashes and raw means, then writes a separate decision.
Exit 1 means rejected timing quality; exit 2 means an unchanged-source control;
neither permits a gain. An accepted gain also requires changed executable
source and an accepted arithmetic-mean ratio above one. The gate detects
catastrophic instability; it is not statistical proof of a speedup. Review
smaller noise effects too. Never trim samples, select favorable legs/cases,
or automatically retry. Any authorized repeat must retain every earlier result.

## Preserved evidence

[READY.json](READY.json) records exact URIs and SHA-256s for the fresh
measurement, independent audits, framework PASS and compiled source negatives.
The fresh archive contains **534 payload files / 25,222,982 bytes**, plus its
preservation receipt, including all 196
remaining files from the completed NVMe run, launch/binding/retirement receipts,
and the preserved earlier invalid comparison:

```text
oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_sg520_completion/20261008/ds-quant-0b098bd1-201262-one-repeat-20261008T190900Z/
```

Its `ARCHIVE-MANIFEST.json` SHA-256 is
`52ed6085db0614bb4f903f169d8079d25bad321fb656adadf5290a2df0e60dbb`.
Copies use `GOMAXPROCS=1` and
`rclone --transfers 64000 --progress --buffer-size 0`; the archive was downloaded
and hash-verified. The earlier compiled-negative binaries remain in the
separately verified original archive referenced by the readiness receipt.
