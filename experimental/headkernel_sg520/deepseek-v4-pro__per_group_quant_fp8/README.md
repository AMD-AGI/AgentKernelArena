Experimental DeepSeek-V4 per-group FP8 quantization package for the pinned SGLang 0.5.20 runtime. Native GPU correctness and graph replay checks passed on one MI355X. **Framework `task_validator` qualification is pending**, so this package remains outside the automatically discovered `tasks/` tree.

The production baseline is AITER's native `per_group_quant_hip`. The candidate compiles the unchanged native `source/quant_kernels.cu` into a separate extension; the frozen wrapper, oracle and build path remain byte-identical to the GPU-tested versions. Only that native source file is editable.

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

The pinned image supplies AITER, Torch, the HIP compiler, rocPRIM and CK headers. The task builds its candidate locally without fetching third-party sources. Individual runner modes are `compile`, `correctness`, and `performance`; performance writes `build/performance_report.json` only after every case passes and all samples are finite and positive.

Recorded native validation: **12 correctness checks passed** (three cases × two seeds × production/candidate). Separate production/candidate graphs passed the oracle before and after timing, with 10 warm-up calls and 100 measured replays per leg per case, alternating paired order. Weighted means were **0.0269248668 ms production** and **0.0273288870 ms candidate**. This was an unmodified-source comparison; no speedup or E2E gain is asserted. [NATIVE-VALIDATION.json](NATIVE-VALIDATION.json) records tested source/harness hashes, native extension identity, coverage and timing summaries. Raw logs, binaries, caches and tensor bodies are excluded.

Metadata paths were relocated for portability. Recorded hashes still identify the original source/evidence bytes. OCI references into `WORKLOAD-GATE.json#/source_refs/...` locate archived declarations of run artifacts rather than claim those individual artifacts were separately uploaded.

[AITER's MIT license](LICENSE) and [license notices](LICENSE-NOTICES.md) accompany the preserved source headers.
