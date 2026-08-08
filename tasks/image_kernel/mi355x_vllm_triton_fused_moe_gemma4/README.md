# mi355x-gemma4-26b-vllm-triton-fused-moe-20260801

Image_kernel harness for vLLM's unquantized Triton fused-MoE expert GEMM
`fused_moe_kernel` (`vllm/model_executor/layers/fused_moe/fused_moe.py:293`).

Generated from the gemma-4-26B-A4B-it Hyperloom 2026-08-01 MI355X session, where
this kernel is the second largest hot path — 16.75% of GPU time across its three
token counts, with the decode shape alone at 12.500%.

## This is the BF16 path, not the int4 one

The sibling task `mi355x_vllm_triton_fused_moe_gptq_awq` edits the same file but
targets `fused_moe_kernel_gptq_awq`, the int4 weight-only WNA16 kernel. That is a
different `@triton.jit` function with different operands. This task covers the
unquantized BF16 `fused_moe_kernel`.

## Why Triton at all

The ROCm AITER unquantized MoE backend rejects Gemma4's GELU_TANH activation, so
the oracle (`fused_moe/oracle/unquantized.py:156`) falls back to Triton.
`server.log` records it directly:

```text
unquantized.py:252 Unquantized MoE backend ROCm AITER does not support ...
                   MoEActivation.GELU_TANH activation. Falling back
unquantized.py:266 Using TRITON Unquantized MoE backend out of potential
                   backends: ['TRITON', 'BATCHED_TRITON']
```

## Workload

TP=2, EP=1, concurrency 64, ISL=1024, OSL=1024. BF16 and unquantized —
`dtype=torch.bfloat16`, `quantization=None`, no `quantization_config` in
`config.json`. (The session workload label says `precision: fp8`; that label is
wrong, and TraceLens' own `metadata/model_info.json` also records BF16.)

`config.json` gives `num_experts` 128, `top_k_experts` 8, `hidden_size` 2816,
`moe_intermediate_size` 704 and `hidden_activation` `gelu_pytorch_tanh`. EP=1
keeps all 128 experts on every rank while TP=2 shards the intermediate, so the
per-rank expert tensors are `w13 (128, 704, 2816)` and `w2 (128, 2816, 352)`.
Cross-checked against the trace: `gelu_tanh_and_mul` takes `(512, 704)` and
returns `(512, 352)`, where 512 = 64 tokens × top-8 and 352 = 704 / TP2.

| Case | Tokens | Phase | Session share |
| --- | --- | --- | --- |
| `gemma4-moe-decode-m64` | 64 | decode | 12.500% (k010) |
| `gemma4-moe-prefill-m1080` | 1080 | chunked prefill | 0.926% (k012) |
| `gemma4-moe-prefill-m7218` | 7218 | chunked prefill | 3.326% (k011) |

Unlike the attention kernels in this session, MoE is token-parallel: the shape is
fully determined by token count, expert geometry and top-k, with no hidden
per-sequence composition. All three trace token counts are reproduced directly
rather than reconstructed.

Each MoE layer launches `fused_moe_kernel` twice — once for the w13 gate_up GEMM
and once for w2 down — so the trace's 1800 decode calls are 30 layers × 2 GEMMs ×
30 steps. One harness call through `fused_experts_impl` covers both launches.

Correctness and performance sweep all three cases. Profiling is a single-shape
probe pinned to `gemma4-moe-decode-m64` via `profile_case` in
`session_cases.json` (surfaced as `PROFILE_CASE_ID` / `profile_case()` in the task
runner) — that is the session's hot entry, and pinning keeps the profiled kernel
from drifting with measurement noise.

## The missing tuned config is a real lever

The session ran with vLLM's fallback MoE config. `server.log` line 89:

```text
WARNING [fused_moe.py:1071] Using default MoE config. Performance might be
sub-optimal! Config file not found at .../configs/
E=128,N=352,device_name=AMD_Instinct_MI355_OAM.json
```

No such file ships in the image, so the harness reproduces that condition exactly.
`configs/` is part of the seeded editable tree, which makes supplying a tuned
config for this shape a legitimate — and probably the highest-leverage —
optimization alongside changing the kernel.

## Editable surface and JIT

`fused_moe.py` is seeded from the image and loaded from the workspace copy, so
edits to the `@triton.jit` kernel, `invoke_fused_moe_kernel`, `fused_experts_impl`
or `configs/` all take effect. Triton re-keys its JIT cache on source, so no
explicit rebuild step is needed. `_load_kernel_module` suppresses
`direct_register_custom_op` while executing the copy so its custom-op
registrations do not clash with the already-imported installed module.

## Benchmark stability

`target_ms` is raised to 10 ms for this task. These cases run 0.16–1.0 ms per
call, and at the default `target_ms=1.0` the captured repeat count collapses
toward 1 (exactly 1 for the m7218 case), which stops amortizing the fixed
graph-replay overhead. Measured consequence: run-to-run swings of 1.7x, and a
30x outlier on a cold first run — reported as a legitimate `cuda_graph` number.
At 10 ms the repeat count stays comfortably above 1 and four consecutive runs
agree to within 0.35%. The sibling attention tasks were checked and do not need
this: their repeat counts land between 6 and 17 at the default.

## Verified locally

Workspace materialized through `src.preprocessing.setup_workspace` on
MI355X/gfx950:

```text
compile      fused_moe compile smoke: PASS
correctness  PASS x3
performance  decode m64     0.1565 ms
             prefill m1080  0.2200 ms
             prefill m7218  0.9976 ms
             benchmark_method=cuda_graph on every case
forge_driver --bench-mode  3x case_ms + mean_ms: 0.458057
forge_driver --profile-run exit 0
```

Edit propagation was checked end to end by scaling the kernel's accumulator by 2
before the output cast: correctness flipped to `allclose: False` and back to
`allclose: True` after restoring the file. The profile pin was checked against a
performance report whose slowest case was `gemma4-moe-prefill-m7218`; the driver
still selected the pinned decode case.

Note that the harness times `fused_experts_impl`, i.e. the whole expert path,
which also includes `moe_align_block_size`, `moe_sum` and the activation. In the
session those are separate kernels worth another 4.35% on top of the 12.500%
attributed to `fused_moe_kernel` itself, so do not compare the harness numbers
against the 12.500% figure directly. This matches the sibling gptq_awq task,
which drives the same entry point.

Expected runtime image:

```text
harbor.crusoe.primus-safe.amd.com/sync/vllm-openai-rocm:v0.24.0
```
