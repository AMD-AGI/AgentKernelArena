DeepSeek-V4 per-group FP8 quantization task for the pinned SGLang 0.5.20 runtime. **Framework `task_validator` qualification passed:** the finalized schema-v3 report has all 12 checks marked PASS, with no warnings, skips or failures. The package is promoted to `tasks/headkernel_sg520/` without changing its kernel, harness or configuration.

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
rclone copy tasks/headkernel_sg520/deepseek-v4-pro__per_group_quant_fp8 "$QUANT_WORKDIR" --transfers 64000 --progress
QUANT_IMAGE=docker.io/lmsysorg/sglang@sha256:3a78acc9d6c191f1a12c7c67631657580f06af3af562c9ee6d88282a71ec5e96
docker run --rm --network=none --device=/dev/kfd --device=/dev/dri \
  --ipc=private --shm-size=8g --memory=64g --cpus=16 --pids-limit=2048 \
  -e ROCR_VISIBLE_DEVICES=0 -e MAX_JOBS=8 -e PYTHONUNBUFFERED=1 \
  -e TORCH_EXTENSIONS_DIR=/task/build/extensions \
  -e XDG_CACHE_HOME=/task/build/xdg -e TRITON_CACHE_DIR=/task/build/triton \
  -v "$QUANT_WORKDIR:/task" -w /task --entrypoint /bin/bash "$QUANT_IMAGE" \
  -c 'python3 scripts/task_runner.py correctness && python3 scripts/task_runner.py performance'
```

The native Docker command above is a diagnostic. For accepting an agent's speedup, stop the optimization worker and use the [trusted host retest tool](../../../src/tools/trusted_native_eval.py) from a trusted checkout, following the [root README](../../../README.md). A privileged Arena workspace alone is not trusted acceptance evidence. The host retest takes only the candidate kernel source and supplies the committed harness/reference itself:

```bash
python3 src/tools/trusted_native_eval.py \
  --repo . --commit '<full-trusted-commit>' \
  --task tasks/headkernel_sg520/deepseek-v4-pro__per_group_quant_fp8 \
  --agent-workspace '<agent-workspace>' --candidate '<agent-workspace>/source/quant_kernels.cu' \
  --render-device /dev/dri/renderD128 --output '<new-trusted-output-directory>'
```

To reproduce framework validation from the repository root, use the [committed validator configuration](../../../example_configs/validate_headkernel_sg520_quant_mi355x.yaml). It is byte-identical to the successful run's configuration: task-validator agent, Codex backend, `gpt-6-astra`, effort `max`, and MI355X. Normal backend credentials/provider configuration are still required. The cache source is a path inside the pinned image:

```bash
AKA_DOCKER_IMAGE=docker.io/lmsysorg/sglang@sha256:3a78acc9d6c191f1a12c7c67631657580f06af3af562c9ee6d88282a71ec5e96 \
AKA_AITER_JIT_SOURCE=/sgl-workspace/aiter/aiter/jit \
AKA_RCLONE_BIN="$(command -v rclone)" \
make docker-run CONFIG=example_configs/validate_headkernel_sg520_quant_mi355x.yaml \
  RUN_ARGS="--run-suffix sg520_quant_validator"
```

The pinned image supplies AITER, Torch, the HIP compiler, rocPRIM and CK headers. The task builds its candidate locally without fetching third-party sources. Individual runner modes are `compile`, `correctness`, and `performance`; performance writes `build/performance_report.json` only after every case passes and all samples are finite and positive. Worker/compiler output is saved under `build/<mode>_<leg>.log` and is never parsed as timing evidence.

Correctness requires all three shapes and both seeds 0 and 1 for each native leg, with separate FP8-byte and scale negative controls. Performance keeps 10 warm-up calls and 100 measured graph replays per case per leg. Before every measured replay, it copies fresh generated values into the same input allocation and poisons both output buffers; after each replay it checks the independent oracle and input immutability. Input generation, copies, poisoning and validation stay outside the device timing interval. Both processes receive identical per-run seed sequences, and their execution order is randomized for each invocation. This separate-process, fresh-input protocol is covered by the final native `ceac8edb` run recorded below.

The report publishes a separate scoreable `test_cases` entry for each of the three mandatory shapes. Standard Arena scoring uses the arithmetic mean of the three matched baseline/candidate speedup ratios. The observed call weights 728, 488 and 968 remain attached to their cases, and `weighted_mean_ms` reports the observed-call-weighted timing for each native leg as a diagnostic only. That workload-weighted diagnostic does not determine the standard Arena score. Missing, duplicate, reshaped or reweighted cases, incomplete samples, mismatched timing summaries and stale worker results fail closed.

Final native validation (`quant-final-194130`, revision `ceac8edb`): **compile, correctness, and performance all passed** in the pinned image. Correctness covered all three cases and seeds 0 and 1 in separate production/candidate processes—12 case/seed/leg checks—with independent FP8-byte and scale negative controls. Each candidate worker freshly compiled the guarded current source. Graph timing covered all three cases with 10 warm-up calls and 100 finite, positive measured replays per case per leg. Fresh input copies, output poisoning, oracle checks and input-immutability checks were applied for every measured replay outside the timing interval. Weighted means were **0.0449486520 ms production** and **0.0453507456 ms candidate**. These are unmodified-source diagnostic timings; no speedup or E2E gain is asserted.

A separate completed no-op probe (`quant-final-noop-194130`) replaced the allowed GPU kernel body with `return;`. Its candidate compiled, then the initial graph replay failed the independent oracle with `invalid dynamic scale`. The run exited 1 and published neither a performance report nor a task result/score. This establishes an observed rejection by the final native harness; it is not a framework validator result.

Framework validation (`run-194224-1`) finalized **overall PASS** for task snapshot `4fd1e4d9`: all 12 schema-v3 checks passed. Compilation, correctness and performance each ran once within the declared timeout; all three cases were emitted and parsed. Correctness covered both seeds and native legs, and graph timing retained 10 warmups and 100 measured replays per case per leg. The previously failed unprivileged attempt is retained as history; private AITER cache seeding resolved its image-cache permission failure before this clean rerun.

The independent trusted retest (`reference-194224-1`) also passed all six phases: compile, correctness and performance for both reference and candidate submissions. Both submissions used the same source SHA-256, with complete three-case coverage. Its arithmetic mean reference/candidate ratio was **0.9848393502**. This is a same-source acceptance-path check, not an optimization or E2E gain claim. The native no-op rejection described above remains a distinct completed negative probe.

The promotion is a directory move with unchanged evaluation inputs. The package identity is `7346fd54b04df36d3a49a8ced82a612a431e2a80d2a73f3e860f54136d8c07c7`, matching both the finalized framework run and trusted retest.

[NATIVE-VALIDATION.json](NATIVE-VALIDATION.json) records the final report hashes, tested package/source/harness hashes, per-phase extension identities, case coverage, 100-sample timing summaries, completed no-op rejection, clean framework PASS, and independent trusted same-source result. Raw logs, binaries, caches and tensor bodies are excluded.

The validation archive is now published at:

`oci:ocieobject1/sapmajum/AgentKernelArena/headkernel_sg520_refresh/20261005/validation-final-20261005T172445Z`

All **172 evidence files and the manifest (173 objects)** were read back and SHA-256 verified. The manifest hash is `83660fc336def213fa374615cea2a96ed3c9687aed798b6f86172837d4bd31be`. [Portable download and verification instructions](../../../docs/reference/headkernel-sg520-refresh.md#published-validation-evidence) include the required rclone flags. Exact framework, trusted, native and no-op report URIs are also recorded in `NATIVE-VALIDATION.json`. This archive qualifies only this task; it does not complete the four-model refresh.

Metadata paths were relocated for portability. Recorded hashes still identify the original source/evidence bytes. OCI references into `WORKLOAD-GATE.json#/source_refs/...` locate archived declarations of run artifacts rather than claim those individual artifacts were separately uploaded.

[AITER's MIT license](LICENSE) and [license notices](LICENSE-NOTICES.md) accompany the preserved source headers.
