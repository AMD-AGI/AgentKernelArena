# mi355x-gemma4-26b-vllm-triton-unified-attention-20260801

Image_kernel harness for vLLM's Triton attention kernel `kernel_unified_attention`
(`vllm/v1/attention/ops/triton_unified_attention.py:179`).

Generated from the gemma-4-26B-A4B-it Hyperloom 2026-08-01 MI355X session, where
this kernel is the largest single leaf in the trace — 27.59% of GPU time across all
its shapes, with the sliding/head_size-256 decode shape alone at 19.550%.

## This is vLLM's kernel, not AITER's

The sibling task `mi355x_vllm_triton_unified_attention` covers AITER's
`aiter/ops/triton/_triton_kernels/attention/unified_attention.py`, which defines
only `kernel_unified_attention_2d` and `kernel_unified_attention_3d`. The trace
records the **unsuffixed** `kernel_unified_attention`, which exists only in the
vLLM file. AITER's variant is reachable solely from the
`ROCM_AITER_UNIFIED_ATTN` backend, and this session ran `TRITON_ATTN`. The two
tasks therefore measure different kernels.

## Why Triton at all

Gemma4 on ROCm has no hand-written paged-attention path:
`use_rocm_custom_paged_attention` accepts `head_size` 64/128 on gfx9, and this
model uses 256 for sliding layers and 512 for full layers. Every attention layer
falls back to this one Triton kernel, which is what makes it the trace's top
entry.

## Workload

TP=2, EP=1, concurrency 64, ISL=1024, OSL=1024, `max_model_len=6144`. BF16
throughout — `dtype=torch.bfloat16`, `quantization=None`, `kv_cache_dtype=auto`,
and the checkpoint carries no `quantization_config`. (The session workload label
says `precision: fp8`; that label is wrong, and TraceLens' own
`metadata/model_info.json` also records BF16.)

`config.json` gives 16 q heads / 8 kv heads / `head_dim` 256 for sliding layers
and `global_head_dim` 512 / `num_global_key_value_heads` 2 for full layers, with
`layer_types` = 5×sliding + 1×full repeated five times (25 sliding, 5 full),
`sliding_window` 1024 and `attn_logit_softcapping` null. Under TP=2 the per-rank
slice is:

| Layer type | Q heads | KV heads | head_size | Window | Session decode share |
| --- | --- | --- | --- | --- | --- |
| sliding | 8 | 4 | 256 | 1024 | 19.550% (k002) |
| full | 8 | 1 | 512 | none | 3.965% (k009) |

Both geometries go through the same kernel but with very different
queries-per-KV ratios (2 vs 8), which drive `BLOCK_M` / `BLOCK_Q`.
`_get_tile_size` also has a Gemma-specific branch keyed on
`(head_size, sliding_window)`.

Four cases cover both geometries at two context lengths. `ctx_len` is
reconstructed from the decode regime (context grows across 1024–2048 with
ISL/OSL 1024); the trace records query/output/step shapes but not the
per-sequence KV context length. ctx 2048 is where the 1024-token sliding window
actually clips.

Correctness and performance sweep all four cases. Profiling is a single-shape
probe pinned to `gemma4-sliding-decode-m64-ctx1024` via `profile_case` in
`session_cases.json` (surfaced as `PROFILE_CASE_ID` / `profile_case()` in the task
runner) — that is the session's largest single leaf, and pinning keeps the
profiled kernel from drifting with measurement noise.

### Prefill is deliberately not covered

The trace also holds four prefill entries for this kernel (7218- and 1080-token
steps at both head sizes, 4.078% combined). They are chunked-prefill steps whose
per-sequence query-length composition is not recorded — the slice is labelled
`mixed_steady_state_prefill_0_prefilldecode_2_decode_30_bs319_conc63` — so a case
would be guesswork about the batch mix rather than a reconstruction. The two
decode shapes are exactly determined (64 tokens = 64 sequences × 1 query token at
concurrency 64) and account for 23.5% of the trace.

## Editable surface and JIT

`triton_unified_attention.py` is seeded from the image and loaded from the
workspace copy, so edits to the `@triton.jit` kernel, its helpers, `_get_tile_size`
or the `unified_attention` host wrapper all take effect. Triton re-keys its JIT
cache on source, so no explicit rebuild step is needed. Every import in the file
is absolute, so the edited copy resolves against the installed `vllm`.

`seq_threshold_3D` / `num_par_softmax_segments` / `softmax_segm_*` are left unset,
so `unified_attention` takes the 2D path (`use_3d` False at
`triton_unified_attention.py:969`). That matches the trace, which records
`kernel_unified_attention` with no `reduce_segments` companion.

## Verified locally

Workspace materialized through `src.preprocessing.setup_workspace` on
MI355X/gfx950:

```text
compile      unified_attention compile smoke: PASS
correctness  PASS x4
performance  sliding ctx1024/2048  0.0597 / 0.0596 ms
             full    ctx1024/2048  0.0767 / 0.1508 ms
             benchmark_method=cuda_graph on every case
forge_driver --bench-mode  4x case_ms + mean_ms: 0.086725
forge_driver --profile-run exit 0
```

The sliding cases are flat across context length while the full cases roughly
double, which is the expected signature of a 1024-token window capping the work —
a useful check that the SWA path is genuinely active.

Edit propagation was checked end to end by scaling the kernel's output by 2
(`acc = acc / L[:, None]`): correctness flipped to `allclose: False` and back to
`allclose: True` after restoring the file. The profile pin was checked against a
performance report whose slowest case was `gemma4-full-decode-m64-ctx2048`; the
driver still selected the pinned sliding case.

Harness timings are not expected to reproduce the session's per-call time
(k002 averaged 161 µs/call there). The harness measures the kernel in isolation
with a compact, contiguous block table, whereas the session ran it against a
fully populated paged cache under TP=2.

Expected runtime image:

```text
harbor.crusoe.primus-safe.amd.com/sync/vllm-openai-rocm:v0.24.0
```

## Effective task instructions

Optimize the vLLM Triton attention kernel kernel_unified_attention (in triton_unified_attention.py) on MI355X/gfx950. This is vLLM's own kernel, not AITER's kernel_unified_attention_2d/_3d. Gemma4 on ROCm has no hand-written paged-attention path - use_rocm_custom_paged_attention only accepts head_size 64/128 on gfx9 - so every attention layer falls back to this Triton kernel and it becomes the largest single leaf in the trace. One kernel serves two geometries: sliding layers at head_size 256 with a 1024-token window (8 query heads over 4 KV heads) and full layers at head_size 512 with no window (8 query heads over 1 KV head), both BF16 decode with 64 sequences and 1024-2048 tokens of KV context. Note the very different queries-per-KV ratios (2 vs 8), which drive BLOCK_M/BLOCK_Q, and that _get_tile_size already has a Gemma-specific branch. Do not tune one geometry at the expense of the other. Cases come from the gemma-4-26B-A4B-it Hyperloom 2026-08-01 MI355X session and are stored in session_cases.json. Preserve all correctness cases and improve CUDA-graph measured performance. Do not change the public signature of unified_attention.

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
