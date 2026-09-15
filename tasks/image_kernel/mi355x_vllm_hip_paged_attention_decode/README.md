# mi355x-rocm-paged-attention-decode-20260801

Image_kernel harness for the ROCm custom paged-attention decode pair
`paged_attention_ll4mi_QKV_mfma16_kernel` (per-partition attention) and
`paged_attention_ll4mi_reduce_kernel` (cross-partition softmax reduction), reached
through `aiter.paged_attention_rocm`.

This operator is the largest decode leaf in four MI355X Hyperloom 2026-08-01
sessions:

| Model | GQA | Q/KV heads | Session GPU share | Session MFMA variant |
| --- | --- | --- | --- | --- |
| Llama-3.1-8B-Instruct | 4:1 | 32 / 8 | 26.85% | MFMA4 |
| Qwen3-8B | 4:1 | 32 / 8 | 25.041% | MFMA4 |
| Qwen3-0.6B | 2:1 | 16 / 8 | 35.778% | MFMA4 |
| Qwen3-14B-FP8 | 5:1 | 40 / 8 | 10.544% | MFMA16 |

## Workload

All four sessions ran the same harness — TP=1, EP=1, concurrency 64, ISL=1024,
OSL=1024, `max_model_len=6144` — so the only thing that varies across models is
the head geometry. Every session reported `kv_cache_dtype=auto`, i.e. BF16
Q / KV cache / output, cross-checked against each kernel's own
`(vllm::Fp8KVCacheDataType)0 = kAuto` template argument. Qwen3-14B-FP8 quantizes
weights, not the KV cache. Head counts were read from each model's `config.json`
and cross-checked against the `GQA_RATIO` template argument in each trace.

Seven cases cover the three distinct geometries at `head_size=128`,
`block_size=16`, 64 decode sequences and 1024–2048 tokens of KV context. The
primary model is sampled at three context lengths to expose the memory-bound
scaling; the coverage models are sampled at the two endpoints of their decode
regime. `num_seqs` and `ctx_len` are reconstructed from that decode regime: the
graph-captured decode launches carry no recorded tensor shapes in the trace.

Qwen3-8B gets no dedicated cases: its geometry, dtypes and workload are identical
to Llama-3.1-8B-Instruct, so the `llama3_1-8b-*` cases already cover it exactly
and duplicate ids would only double suite runtime.

Correctness and performance sweep all seven cases. Profiling is a single-shape
probe pinned to `llama3_1-8b-decode-m64-ctx1024` via `profile_case` in
`session_cases.json` (surfaced as `PROFILE_CASE_ID` / `profile_case()` in the task
runner) — GQA 4:1 at `head_size=128` is the most common decode geometry here, and
pinning keeps the profiled kernel from drifting with measurement noise.

## Which kernel this measures

Three of the four sessions ran **vLLM's** MFMA4 variant from
`csrc/rocm/attention.cu`, selected because `gqa_ratio <= 4`. This task measures
**AITER's** MFMA16 implementation of the same logical operator at each session's
exact shapes and dtypes. Only the Qwen3-14B-FP8 cases (GQA 5:1) match the
session's MFMA variant as well.

The reason is that the runtime image ships no vLLM C++ source: vLLM is a
wheel-only install and `/app/vllm` contains only `benchmarks/`, `docker/` and
`examples/`, so there is nothing to seed and edit. AITER ships the same
`paged_attention_ll4mi` family with full source, but dispatches MFMA16 for every
GQA ratio — the MFMA4 launches are commented out in both
`csrc/kernels/attention.cu` and `csrc/cpp_itfs/pa/pa.cpp.jinja`. Confirmed by
symbol inspection: `mfma4` exists only in `vllm/_rocm_C.abi3.so`, while
`aiter/jit/module_pa*.so` carry `mfma16` only.

Treat wins here as transferable to the same operator, not as a direct measurement
of the MFMA4 leaf kernel three of these sessions executed.

## Editable surface and JIT

`aiter.paged_attention_rocm` goes through `csrc/cpp_itfs/pa/pa.py`, which renders
`pa.cpp.jinja` and compiles it per specialization via `compile_template_op`. The
four files listed in `config.yaml` (`pa_kernels.cuh`, `pa.cuh`, `pa_common.cuh`,
`pa.cpp.jinja`) are the editable device-code surface.

`compile_template_op` caches purely by template arguments, so an edited kernel
would otherwise keep serving a stale `lib.so`. Two things prevent that:

- `AITER_REBUILD=1` clears the template-op build cache at import.
  AgentKernelArena injects it per build subprocess (`src/jit_rebuild.py`); the
  task runner also sets it by default so standalone runs stay honest.
- `AITER_META_DIR` points the importable `csrc` package — and therefore
  `AITER_CORE_DIR`, the jinja template and every include — at the workspace copy.
  `_import_aiter()` asserts this resolved to the seeded tree and fails loudly
  rather than silently measuring the in-image source.

The cache key includes `gqa_ratio`, so the suite builds three specializations
(GQA 2 / 4 / 5) at roughly 15 s each. All cases give
`npar_loops = ceil(ceil(ctx_len/256) / 64) = 1`, matching the sessions'
`paged_attention_ll4mi_reduce_kernel<..., 128, 128, 256, 1>`. Profiling builds
only the one pinned specialization.

## Verified locally

Workspace materialized through `src.preprocessing.setup_workspace` on
MI355X/gfx950:

```text
compile      paged_attention_rocm compile smoke: PASS
correctness  PASS x7
performance  llama3_1-8b   ctx1024/1536/2048  0.055 / 0.076 / 0.128 ms
             qwen3-0_6b    ctx1024/2048       0.055 / 0.125 ms
             qwen3-14b-fp8 ctx1024/2048       0.057 / 0.130 ms
             benchmark_method=cuda_graph on every case
forge_driver --bench-mode  7x case_ms + mean_ms: 0.089140
forge_driver --profile-run exit 0, single specialization built
```

Edit propagation was checked end to end by scaling the reduce kernel's output by
2 in `pa_kernels.cuh`: correctness flipped to `allclose: False` and back to
`allclose: True` after restoring the file. The profile pin was checked against a
performance report whose slowest case was `qwen3-14b-fp8-decode-m64-ctx2048`; the
driver still selected the pinned Llama case.

Expected runtime image:

```text
harbor.crusoe.primus-safe.amd.com/sync/vllm-openai-rocm:v0.24.0
```

## Effective task instructions

Optimize the ROCm custom paged-attention decode kernels on MI355X/gfx950. The device kernels are paged_attention_ll4mi_QKV_mfma16_kernel (partial attention per KV partition) and paged_attention_ll4mi_reduce_kernel (cross-partition softmax reduction), reached through aiter.paged_attention_rocm. Both are JIT specialized per (gqa_ratio, head_size, npar_loops, dtype, block_size, ...) from csrc/cpp_itfs/pa/pa.cpp.jinja, whose device code lives in pa.cuh / pa_kernels.cuh / pa_common.cuh - all four files are editable and every edit is recompiled before each scoring step. The workload is BF16 decode with 64 sequences, head_size 128, block_size 16 and 1024-2048 tokens of KV context, across the three head geometries these models produce: GQA 2:1 (16 q / 8 kv), 4:1 (32 q / 8 kv) and 5:1 (40 q / 8 kv). That is three JIT specializations for the seven cases, so a rebuild costs three compiles. Do not tune one gqa_ratio at the expense of another. Cases come from four MI355X Hyperloom 2026-08-01 sessions (Llama-3.1-8B-Instruct, Qwen3-8B, Qwen3-0.6B, Qwen3-14B-FP8) and are stored in session_cases.json. Preserve all correctness cases and improve CUDA-graph measured performance. Do not change the signature of aiter.paged_attention_rocm or the launch contract in pa.cpp.jinja.

## Arena v2 contract

The candidate is the existing implementation in the declared image sources.
Its required final language and exact task-relative editable files are in
`config.yaml`; directory names do not select execution behavior. The framework
freezes this initial implementation into a separate baseline workspace. Both
roles run the same protected harness in their own workspace; an absent candidate
or missing image source is an error, never permission to use the installed copy.

Setup runs `python3 scripts/setup_task.py` after declared image materialization
and before baseline capture. It validates source paths and required build assets.
Do not edit `scripts/`, workload files or references. Additional source files
outside `candidate.editable` are dependencies, not editable implementation.
Preserve the original numerical gates, seeds, layouts, dispatch, state handling
and CUDA graph/event timing. `workloads.json` enumerates the complete manifest
independently of reported timings; `session_cases.json`, when present, retains
its original session provenance. Cases marked correctness-only are not scored.

Use the agent-neutral commands:

```bash
python3 scripts/evaluate.py validate-task
python3 scripts/evaluate.py baseline compile
python3 scripts/evaluate.py baseline correctness
python3 scripts/evaluate.py baseline performance
python3 scripts/evaluate.py candidate compile
python3 scripts/evaluate.py candidate correctness
python3 scripts/evaluate.py candidate performance
```

Baseline commands run in the framework's frozen workspace. Each command emits
one `ARENA_EVAL_RESULT=` envelope. A failed dependency, dispatch or output contract
is a failure, not an accepted baseline numerical diagnostic. The original
`task_runner.py` remains the protected operator implementation of these checks;
its generated performance region must be materialized by Arena. Developer
profiling drivers do not supply final evaluation evidence.
This migration has CPU regression coverage; formal GPU task validation and the
optimization campaign are coordinated separately. Runtime source availability
must be checked against the selected immutable image, not inferred from a tag.

HIP evaluation uses a fresh task-local JIT directory per action. The runner
requires a successful compilation whose inputs include a declared candidate
translation unit or template header. It records the covered files and rejects
unrelated/precompiled dispatch. This is build-source evidence, not exhaustive
proof that every launched GPU instruction belongs to every editable file.
No compiler-triggered repository cloning or checkout resets are permitted.

The qualified SGLang runtime stores AITER as a complete source repository at
`/sgl-workspace/aiter`, not as an installed `aiter_meta` wheel directory.
`workspace.sources` explicitly copies that repository to the task's metadata
root; the unified-attention task separately copies its `aiter/` Python package.
Editable task-relative paths and operator semantics remain unchanged. This fixes
source availability only; dispatch, compilation and numerical compatibility
still require full GPU validation on the selected immutable runtime.
