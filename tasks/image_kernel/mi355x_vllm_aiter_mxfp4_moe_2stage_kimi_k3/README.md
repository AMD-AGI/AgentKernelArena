# mi355x-kimi-k3-aiter-mxfp4-moe-2stage-20260728

Faithful `image_kernel` reproduction of the **Kimi-K3 routed-expert MoE 2-stage
GEMM** as it actually ran in Hyperloom session `20260728T091437Z` on MI355X/gfx950
(session archived at `_archive/Kimi-K3_20260728T091437Z_pod_restart_disk_quota`).

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
| `w2_scale` layout invariant | a patch editing one branch of `shuffle_scale` but not the other; vLLM's loader reaches `w2_scale` via `e8m0_shuffle` (`is_guinterleave=False`) while this harness uses `shuffle_scale_a16w4` (`is_guinterleave=True`) — byte-identical today, and the harness cannot notice divergence on its own because it drives both sides | `_prepare` |
| `nLane == 16` | a kernel needing a different `nLane`; vLLM hardcodes 16 at `mxfp4.py:789,792` and would never reach it | `_prepare` |
| worst-of-3 | atomic non-determinism turning a marginal result into an intermittent pass | `run_correctness` |

The performance report also records `dispatched_stage1_kernel` /
`dispatched_stage2_kernel` per case, so the scored kernel is identifiable after
the fact rather than inferred.

### Why these matter for applying the patch to vLLM

The deliverable is a patch to **`aiter` only** — vLLM is not modified.
`rocm_aiter_ops.shuffle_weight_a16w4` / `shuffle_scale_a16w4` are pure forwarders
into `aiter.ops.shuffle` (`vllm/_aiter_ops.py:2727,2748`), so a patched aiter is
also what vLLM's weight loader uses and layouts stay consistent for free. The two
places that do **not** follow automatically are the hardcoded `nLane=16` and the
`e8m0_shuffle` entry point for `w2_scale`; those are exactly what the two
invariant gates pin down.

## Edit surface and JIT freshness

The compute core (FlyDSL asm / ck2stages fp4) is not an in-tree editable `.cu`, so
the source-resolved edit surface is the aiter Python dispatch: `fused_moe.py`,
`ops/flydsl/moe_kernels.py`, `ops/shuffle.py`.

The harness puts the workspace-seeded `aiter` copy first on `sys.path`, so an
agent's edits shadow the in-image install. Verified: scaling `fused_moe`'s return
by 1.5 in the workspace copy changes the measured output by exactly 1.5x. aiter's
JIT output is pinned to `<workspace>/build/jit` so no run can load another run's
compiled module.

`aiter` locates its C++ headers relative to the directory *containing* the package
(`aiter/utility/aiter_types.py:_find_aiter_enum_h` hardcodes `parents[2]` and
ignores `AITER_META_DIR`), so `_configure()` links the image's `aiter_meta` next to
the seeded copy. Those are C++ sources outside this task's edit surface, so they
are linked rather than duplicated per run.

## Run

```
python3 scripts/task_runner.py compile       # smoke: one MoE call
python3 scripts/task_runner.py correctness   # cos > 0.999 vs the torch reference
python3 scripts/task_runner.py performance   # CUDA-graph timed -> build/performance_report.json
```
