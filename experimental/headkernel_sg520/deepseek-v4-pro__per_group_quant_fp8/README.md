Experimental DeepSeek-V4 per-group FP8 quantization package for the pinned SGLang 0.5.20 runtime. Native GPU correctness and graph replay checks passed on one MI355X with the earlier harness recorded below. **The hardened harness requires fresh GPU validation, and framework `task_validator` qualification is pending**, so this package remains outside the automatically discovered `tasks/` tree.

The production baseline is AITER's native `per_group_quant_hip`. Baseline and candidate load in separate worker processes. Every candidate worker freshly compiles the current native source into a uniquely named extension in a new build directory; it cannot reuse a supplied binary or fall back to production code. The parent checks the current source/package hashes, worker identity, fresh request identifier, extension hash and complete results before publishing a report.

Only the body of `dynamic_per_group_scaled_quant_kernel` in `source/quant_kernels.cu` is editable. Its signature and surrounding translation unit are checked against `ut/native/quant_kernels.reference.cu`. Local device helpers inside that body are allowed. New includes, macros, host hooks and changes to the launcher are rejected before compilation. The stock, brace-free architecture type-selection block may remain unchanged or be removed with a rewritten kernel body. All other task files are frozen.

| BF16 input | Observed calls per rank |
|---|---:|
| `[8192, 1536]` | 728 |
| `[8192, 2048]` | 488 |
| `[8192, 7168]` | 968 |

The corrected serving run completed 64 requests with ISL 8192, OSL 1024, concurrency 64 and TP 8. The weights above cover **eight sampled prefill steps per rank**, with matching counts across all eight ranks; their sum is 2,184. They are not extrapolated full-workload counts. [cases.json](cases.json) records the ABI, weights and trace references. Original profiles and curated evidence are archived at:

`oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_sg520_refresh/20261005/deepseek-sg520-context9218-193529`

Inputs are seeded legal BF16 values at those observed shapes. No model weights or captured tensor artifacts are required. Group size is 128, output is FP8 E4M3FN, and scales are FP32 with `transpose_scale=True`; optional input scale and row-count tensors are absent. The scale allocation remains contiguous `[M,N/128]`, but native writes use flat offset `group*M+row`. The independent PyTorch oracle checks this physical layout, scale rounding, exact FP8 bytes, input immutability and negative controls.

Run from the repository root on a ROCm host with a MI355X and Docker. The writable copy keeps generated build/cache files outside the checkout; change `ROCR_VISIBLE_DEVICES` to an available allocated device:

```bash
QUANT_WORKDIR="$(mktemp -d)"
rclone copy experimental/headkernel_sg520/deepseek-v4-pro__per_group_quant_fp8 "$QUANT_WORKDIR" --transfers 64000 --progress
QUANT_IMAGE=docker.io/lmsysorg/sglang@sha256:3a78acc9d6c191f1a12c7c67631657580f06af3af562c9ee6d88282a71ec5e96
docker run --rm --network=none --device=/dev/kfd --device=/dev/dri \
  --ipc=private --shm-size=8g --memory=64g --cpus=16 --pids-limit=2048 \
  -e ROCR_VISIBLE_DEVICES=0 -e MAX_JOBS=8 -e PYTHONUNBUFFERED=1 \
  -e TORCH_EXTENSIONS_DIR=/task/build/extensions \
  -e XDG_CACHE_HOME=/task/build/xdg -e TRITON_CACHE_DIR=/task/build/triton \
  -v "$QUANT_WORKDIR:/task" -w /task --entrypoint /bin/bash "$QUANT_IMAGE" \
  -c 'python3 scripts/task_runner.py correctness && python3 scripts/task_runner.py performance'
```

The pinned image supplies AITER, Torch, the HIP compiler, rocPRIM and CK headers. The task builds its candidate locally without fetching third-party sources. Individual runner modes are `compile`, `correctness`, and `performance`; performance writes `build/performance_report.json` only after every case passes and all samples are finite and positive. Worker/compiler output is saved under `build/<mode>_<leg>.log` and is never parsed as timing evidence.

Correctness requires all three shapes and both seeds 0 and 1 for each native leg, with separate FP8-byte and scale negative controls. Performance keeps 10 warm-up calls and 100 measured graph replays per case per leg. Before every measured replay, it copies fresh generated values into the same input allocation and poisons both output buffers; after each replay it checks the independent oracle and input immutability. Input generation, copies, poisoning and validation stay outside the device timing interval. Both processes receive identical per-run seed sequences, and their execution order is randomized for each invocation. This replaces the earlier same-process alternating timing protocol and requires new timing evidence.

The report publishes a separate scoreable `test_cases` entry for each of the three mandatory shapes. Standard Arena scoring uses the arithmetic mean of the three matched baseline/candidate speedup ratios. The observed call weights 728, 488 and 968 remain attached to their cases, and `weighted_mean_ms` reports the observed-call-weighted timing for each native leg as a diagnostic only. That workload-weighted diagnostic does not determine the standard Arena score. Missing, duplicate, reshaped or reweighted cases, incomplete samples, mismatched timing summaries and stale worker results fail closed.

Historical native validation before hardening: **12 correctness checks passed** (three cases × two seeds × production/candidate). Separate production/candidate graphs passed the oracle before and after timing, with 10 warm-up calls and 100 measured replays per leg per case, alternating paired order. Weighted means were **0.0269248668 ms production** and **0.0273288870 ms candidate**. This was an unmodified-source comparison; no speedup or E2E gain is asserted. [NATIVE-VALIDATION.json](NATIVE-VALIDATION.json) retains the exact historical source/harness hashes, native extension identity, coverage and timing summaries; it does not qualify the revised harness. Raw logs, binaries, caches and tensor bodies are excluded.

Metadata paths were relocated for portability. Recorded hashes still identify the original source/evidence bytes. OCI references into `WORKLOAD-GATE.json#/source_refs/...` locate archived declarations of run artifacts rather than claim those individual artifacts were separately uploaded.

[AITER's MIT license](LICENSE) and [license notices](LICENSE-NOTICES.md) accompany the preserved source headers.
