# mi355x-kimi-k3-aiter-mxfp4-moe-2stage-20260728

Faithful `image_kernel` reproduction of the **Kimi-K3 routed-expert MoE 2-stage
GEMM** as it actually ran in Hyperloom session `20260728T091437Z` on MI355X/gfx950
(session archived at `_archive/Kimi-K3_20260728T091437Z_pod_restart_disk_quota`).

## Which hot kernels this covers

k001/k002/k003/k006 are the **same aiter MoE 2-stage op** in different execution
modes and stages (one MoE layer = two GEMMs with an activation between them):

| kernel_id | trace op                       | mode            | stage          | backend (per trace) |
|-----------|--------------------------------|-----------------|----------------|---------------------|
| k001      | `hipGraphLaunch->moe_gemm1_0`  | decode (graph)  | stage1 gate/up | not resolved        |
| k002      | `hipGraphLaunch->moe_gemm2_0`  | decode (graph)  | stage2 down    | not resolved        |
| k003      | `pseudo_op::moe_flydsl_stage1` | prefill (eager) | stage1 gate/up | **FlyDSL** (named)  |
| k006      | `pseudo_op::moe_flydsl_stage2` | prefill (eager) | stage2 down    | FlyDSL (named)      |

Two cases:
- `kimi-k3-prefill-flydsl-k003-k006` — `token=7211`, **exact** trace shape.
- `kimi-k3-decode-graph-k001-k002` — `token=62`, **reconstructed** (decode M is not
  recorded for hipGraph synthetic ops; 62 is the steady-state slice concurrency).

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
around the kernel under test. Gate: `cos > 0.999` and relative norm error `< 0.05`.

Measured on MI355X: `cos = 0.99997`, `rel_err = 0.006-0.008`. The stage2 kernel
reduces with atomics, so results vary slightly run to run (observed cos spread
0.999968-0.999983); the gate leaves ~6x margin over that.

Only the *token count* is reduced for correctness (to 64). Expert count and all
dims stay at the session values so the same FlyDSL kernel pair is dispatched.

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
