# mi355x-kimi-k3-aiter-mxfp4-moe-2stage-20260728

An `image_kernel` task derived from the **Kimi-K3 routed-expert MoE 2-stage
GEMM** in Hyperloom session `20260728T091437Z` on MI355X/gfx950
(session archived at `_archive/Kimi-K3_20260728T091437Z_pod_restart_disk_quota`).

The current task benchmarks deterministic synthetic tensors at the documented
model geometry. The prefill shape was recorded; decode token=62 is a representative
reconstruction with the evidence and limits described below. Its measured latency
is for this declared workload, not an exact replay of an unavailable decode
trace. Both scored cases and all numerical gates are retained. Historical
session dispatch statements below describe that original build; current runs
record their own image/source identity and actual tuned kernel pair, which must
agree between correctness and performance.

## Which hot kernels this covers

k001/k002/k003/k006 are the **same aiter MoE 2-stage op** in different execution
modes and stages (one MoE layer = two GEMMs with an activation between them):

| kernel_id | trace op                       | mode            | stage          | backend            |
|-----------|--------------------------------|-----------------|----------------|--------------------|
| k001      | `hipGraphLaunch->moe_gemm1_0`  | decode (graph)  | stage1 gate/up | **FlyDSL** (a16w4) |
| k002      | `hipGraphLaunch->moe_gemm2_0`  | decode (graph)  | stage2 down    | **FlyDSL** (a16w4) |
| k003      | `pseudo_op::moe_flydsl_stage1` | prefill (eager) | stage1 gate/up | **FlyDSL** (named) |
| k006      | `pseudo_op::moe_flydsl_stage2` | prefill (eager) | stage2 down    | FlyDSL (named)     |

TraceLens reports k001/k002 as "not resolved" because they are hipGraph synthetic
ops with no launcher. They are not a separate backend: `moe_gemm1_0` is the device
symbol emitted by `compile_mixed_moe_gemm1_a16w4`
(`aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py:4951`, inner `def moe_gemm1`
at `:5300`; the `_0` suffix is FlyDSL's compiled-instance index). So all four
kernel_ids are the same FlyDSL pair, captured in a graph (decode) or launched
eagerly (prefill).

Two scored cases:
- `kimi-k3-prefill-flydsl-k003-k006` — `token=7211`, **exact** trace shape.
- `kimi-k3-decode-graph-k001-k002` — `token=62`, **reconstructed** (decode M is not
  recorded for hipGraph synthetic ops; 62 is the steady-state slice concurrency).
  Corroborated after the fact: the session's own `trace_split/` contains
  `decode_only_steady_state_..._bs64_conc64_...`, and `M=64` is one of the 14
  buckets dispatched in its `server.log` — so `token=62` lands on the kernel pair
  decode actually used.

Plus 12 `mbucket-*` correctness-only cases covering the other M buckets (see
[Correctness](#correctness)).

## Exact shapes / dtypes (authoritative)

From the forge `invocation_spec_pseudo_op_moe_flydsl_stage1.json` (trace-recorded):

```
act        (7211, 3584)      bf16              # model_dim = routed_expert_hidden = 3584
w1         (896, 768, 1792)  fp4               # 768 = inter*2 (g1u1), 1792 = 3584/2 (fp4 packed)
w2         (896, 3584, 192)  fp4               # 192 = inter/2 (fp4 packed) -> inter = 384
topk_w     (7211, 16)        fp32
topk_id    (7211, 16)        int32
w1_scale   (688128, 112)     Float8_e8m0fnu    # (896*768, 3584/32)
w2_scale   (3211264, 16)     Float8_e8m0fnu    # (896*3584, 12 -> padded to 16)
```

Per-rank (TP=8) config: `model_dim=3584`, `inter_dim=384` (`moe_intermediate 3072 / TP8`),
`experts=896`, `topk=16`, `quant_type=per_1x32` (mxfp4 group_size 32), bf16 activation,
`fp4x2` weight, `g1u1=True`, activation **SiTUv2** with `beta=4.0` / `linear_beta=25.0`.

## Dispatch fidelity (verified)

Running this harness on the session image reproduces the session's own aiter
dispatch lines **verbatim**, for both M buckets the cases land in:

```
token 7211 -> M=8192  kernelName1='flydsl_moe1_abf16_wfp4_bf16_t32x128x256_w2'
                      kernelName2='flydsl_moe2_abf16_wfp4_bf16_t32x256x256_atomic_bnt2_xcd4_persist'
token 62   -> M=64    kernelName1='flydsl_moe1_abf16_wfp4_bf16_t32x64x256_w3_xcd4_kw2'
                      kernelName2='flydsl_moe2_abf16_wfp4_bf16_t32x256x128_atomic_bnt2_persist'
```

Both appear in the session's own logs, so the harness exercises the same kernels.

## Three contract details that are easy to get wrong

All three were verified against the in-image sources, not assumed:

1. **The activation enum is `Situv2`, not `Situ`.** This build exposes
   `ActivationType.{Gelu,No,Silu,Situv2,Swiglu}` and the K3 dispatch branches all
   key on `Situv2` (`aiter/fused_moe.py:619,1202,2198,3104`). The session logs
   contain 3409 occurrences of `ActivationType.Situv2` and none of `Situ`.

2. **K3's SiTU a16w4 path runs `GateMode.SEPARATED`, so weights and scales must be
   shuffled with `gate_up=False`** (GGUU rows). `gate_up=True` produces the
   GUGU/INTERLEAVE layout used by the gpt-oss `use_mxfp4_w4a16` path; feeding that
   to the SEPARATED kernel yields output with the *right magnitude but cosine ~0*
   against the reference. Authority:
   `vllm/model_executor/layers/fused_moe/experts/rocm_aiter_moe.py:369-386`.

3. **The SiTUv2 beta parameters must come from the model config.** `fused_moe`
   defaults to `1.0/1.0` while `torch_moe_stage1` defaults to `2.0/1.5`
   (`aiter/fused_moe.py:676-677` vs `:2998-2999`) — neither is K3's value. K3
   `config.json text_config` sets `activation_situ_beta=4.0` and
   `activation_situ_linear_beta=25.0`; the harness drives both the kernel call and
   the reference from those, so the two sides cannot silently drift apart.

## Correctness

Compared against aiter's own dequantized `torch_moe_stage1`/`torch_moe_stage2`,
which unpack the mxfp4 nibbles, apply the per-1x32 e8m0 group scales and
accumulate in fp32 — a real independent implementation of the op, not a wrapper
around the kernel under test. Gate: `cos > 0.999` and relative norm error `< 0.05`,
taken as the **worst of 3 runs** (stage2 reduces with atomics, so a single pass
can be lucky).
The output must also have the reference's exact shape/device and BF16 dtype,
including on the captured path. Casting to FP32 for error metrics must not accept
an FP32 public output.

### Every M bucket is checked, at its real token count

The FlyDSL kernel pair is chosen **per M bucket** from the tuned CSV, and the 14
buckets the session actually dispatched (`1,2,4,…,8192`, all present in its
`server.log`) map to **14 distinct kernel pairs**. So correctness must run at the
same token count performance is measured at, or it validates a different kernel:

```
token=62   -> flydsl_moe1_..._t32x64x256_w3_xcd4_kw2  | flydsl_moe2_..._t32x256x128_atomic_bnt2_persist
token=7211 -> flydsl_moe1_..._t32x128x256_w2          | flydsl_moe2_..._t32x256x256_atomic_bnt2_xcd4_persist
```

Both scored cases therefore run correctness at their real token, and 12
`mbucket-*` cases (`correctness_only`, not scored) cover the remaining buckets.
All 14 pass; measured worst-of-3 on MI355X, whole suite in ~30 s:

| bucket | 1 | 2 | 4 | 8 | 16 | 32 | 62 | 128 | 256 | 512 | 1024 | 2048 | 4096 | 7211 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cos | .999997 | .999997 | .999997 | .999865 | .999919 | .999965 | .999971 | .999976 | .999969 | .999962 | .999984 | .999983 | .999983 | .999925 |
| rel_err | .0026 | .0026 | .0025 | .0165 | .0127 | .0083 | .0077 | .0070 | .0079 | .0087 | .0056 | .0058 | .0058 | .0123 |

There is no token clamp. It was previously 64, which is why the scored M=8192
kernel went unchecked. Measured: the torch reference costs ~0.4 s and ~37 GiB
peak at token=7211, and that cost is dominated by dequantizing all 896 expert
weights — token-independent (`token=64 -> 37.15 GiB`, `token=7211 -> 37.31 GiB`).
`moe_config.correctness_max_token` remains as an escape hatch; if it (or a
per-case `correctness_token`) shrinks the token, `run_correctness` re-derives the
dispatched pair at the performance token and **fails** unless the pair is
identical.

## Gates

Five checks beyond the numeric comparison, each verified to fire (negative-tested
on MI355X):

| gate | catches | where |
|---|---|---|
| M-bucket identity | correctness and performance landing on different kernel pairs | `run_correctness` |
| tuned-dispatch assertion | aiter falling back to the heuristic FlyDSL branch (`fused_moe.py:2272`) instead of the tuned path the session ran — i.e. a whole run spent optimising code that never executes | `run_compile`, `run_correctness`, `run_performance` |
| `w2_scale` deployment path | a patch that only works with the distinct `shuffle_scale_a16w4` layout; vLLM's loader reaches `w2_scale` through `e8m0_shuffle` (`is_guinterleave=False`), so the harness uses that exact path too | `_prepare` |
| `nLane == 16` | a kernel needing a different `nLane`; vLLM hardcodes 16 at `mxfp4.py:789,792` and would never reach it | `_prepare` |
| worst-of-3 | atomic non-determinism turning a marginal result into an intermittent pass | `run_correctness` |

The performance report also records `dispatched_stage1_kernel` /
`dispatched_stage2_kernel` per case, so the scored kernel is identifiable after
the fact rather than inferred.

### Why these matter for applying the patch to vLLM

The deliverable is a patch to **`aiter` only** — vLLM is not modified.
`rocm_aiter_ops.shuffle_weight_a16w4` / `shuffle_scale_a16w4` are pure forwarders
into `aiter.ops.shuffle` (`vllm/_aiter_ops.py:2727,2748`), so a patched aiter is
also what vLLM's weight loader uses and layouts stay consistent for free. The
harness explicitly pins the hardcoded `nLane=16` and drives `w2_scale` through
the same `e8m0_shuffle` entry point as vLLM.

## Edit surface and JIT freshness

The five editable paths include the FlyDSL builder, dispatch wrappers, shuffle
code, and tuned Kimi CSV; `config.yaml` lists their exact task-relative locations.
The image source is a **complete repository** seeded at `aiter/`, so the Python
package and config files live under `aiter/aiter/`, and headers under
`aiter/csrc/`. The repository's package directory must shadow the installed copy.
The task does not link to an external installed `aiter_meta` directory.

Setup captures `fused_moe.py` as the protected `scripts/_reference_fused_moe.py`
before the framework freezes the initial baseline. The mathematical stage
references come from that copy, so editing candidate dispatch does not overwrite
the reference. Their FP4 support dependencies remain outside the edit surface.
A resumed setup does not refresh this reference from an optimized candidate.

The strict tuned-dispatch gates remain mandatory. Availability of the five files
in another image does not establish API, numerical or tuned-dispatch compatibility.
Historical session validation statements above describe the original session
build, not a fresh v2 validation of either SGLang qualification image.

## Run

```
python3 scripts/evaluate.py candidate compile       # smoke: one MoE call
python3 scripts/evaluate.py candidate correctness   # cos > 0.999 vs the torch reference
python3 scripts/evaluate.py candidate performance   # CUDA-graph timed -> build/performance_report.json
```

## Effective task instructions

Optimize the Kimi-K3 routed-expert MoE 2-stage GEMM using the protected synthetic workloads derived from Hyperloom session 20260728T091437Z on MI355X/gfx950. This is the aiter MoE path behind hot kernels k001 (moe_gemm1_0, decode-graph stage1 gate/up), k002 (moe_gemm2_0, decode-graph stage2 down), k003 (moe_flydsl_stage1) and k006 (moe_flydsl_stage2, eager/prefill FlyDSL). Config: bf16 activation x fp4 (mxfp4-pack, group_size=32 -> QuantType.per_1x32) weight, g1u1, SiTUv2 activation with beta=4.0 / linear_beta=25.0, per-rank (TP=8) dims model_dim=3584 / inter_dim=384 / experts=896 / topk=16. The prefill shape and model geometry are trace-recorded in session_cases.json; decode token=62 is a documented reconstruction, not an exact archived trace shape. Current dispatch comes from the selected immutable runtime and must match between correctness and timing for each M bucket. The compute core is a FlyDSL MLIR builder (ops/flydsl/kernels/mixed_moe_gemm_2stage.py, compile_mixed_moe_gemm1_a16w4 / compile_mixed_moe_gemm2_a16w4) and is part of this task's edit surface. The full edit surface also includes the tuned Kimi-K3 CSV and the aiter Python dispatch in fused_moe.py, ops/flydsl/moe_kernels.py, and ops/shuffle.py. Useful levers are the MFMA pipeline, tile/block loop structure, LDS usage, stage2 atomic reduction, 2-stage kernel selection, and tuned-config lookup. The 864 a16w4 variants are keyed by tile_m/tile_n/tile_k plus w*/bnt*/xcd*/kw* suffixes. Preserve the per-M-bucket dispatch contract, block_m / ksplit semantics, and moe_sorting behavior. Do not edit the harness. The harness enforces five invariants that a valid patch must keep: correctness and performance must dispatch the same FlyDSL kernel pair; dispatch must stay on the tuned path rather than the heuristic fallback; w2_scale must use the same e8m0_shuffle path as vLLM; nLane must stay 16 (vLLM hardcodes it); and correctness is the worst of 3 runs because stage2 reduces with atomics. All 14 reachable M buckets are correctness-checked. Two contract details that must be preserved: K3's SiTU a16w4 path runs GateMode.SEPARATED, so weights and scales are shuffled with gate_up=False (gate_up=True is the gpt-oss INTERLEAVE layout and silently produces garbage); and the SiTUv2 beta/linear_beta must stay at the session values, since aiter's kernel and torch-reference defaults disagree with each other and with K3. Correctness compares against aiter's dequantized torch_moe_stage1/stage2 reference. Preserve all correctness cases and improve the CUDA-graph measured performance.

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
its generated performance region must be materialized by Arena. Optional
profiling does not supply final evaluation evidence. Agent CLI adaptation belongs
to the agent integration; use the declared v2 runner for task evaluation, with
the task's full numerical and workload checks.
This migration has CPU regression coverage; formal GPU task validation and the
optimization campaign are coordinated separately. Runtime source availability
must be checked against the selected immutable image, not inferred from a tag.
