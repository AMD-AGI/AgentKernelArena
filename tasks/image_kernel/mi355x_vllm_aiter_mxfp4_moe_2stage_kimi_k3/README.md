# mi355x-kimi-k3-aiter-mxfp4-moe-2stage-20260728

Faithful `image_kernel` reproduction of the **Kimi-K3 routed-expert MoE 2-stage
GEMM** as it actually ran in Hyperloom session `20260728T091437Z` on MI355X/gfx950
(session archived at `_archive/Kimi-K3_20260728T091437Z_pod_restart_disk_quota`).

Goal: match the session's real shapes/dtypes/config exactly. This task is **not
meant to run on stock aiter** — run it in the session build (see below).

## Which hot kernels this covers

k001, k002, k003 are the **same aiter MoE 2-stage op** in different execution
modes / stages (one MoE layer = two GEMMs with an activation in between; "fused"
means routing/gather/activation/scale are fused into the GEMMs, not that the two
GEMMs are merged):

| kernel_id | trace op                        | mode            | stage          | backend (per trace) |
|-----------|---------------------------------|-----------------|----------------|---------------------|
| k001      | `hipGraphLaunch->moe_gemm1_0`   | decode (graph)  | stage1 gate/up | not resolved        |
| k002      | `hipGraphLaunch->moe_gemm2_0`   | decode (graph)  | stage2 down    | not resolved        |
| k003      | `pseudo_op::moe_flydsl_stage1`  | prefill (eager) | stage1 gate/up | **FlyDSL** (named)  |
| (k006)    | `pseudo_op::moe_flydsl_stage2`  | prefill (eager) | stage2 down    | FlyDSL (named)      |

Only k003/k006 are confirmed FlyDSL by op name. k001/k002 are decode graph nodes
named generically (`moe_gemm1_0/2_0`); the trace does not resolve their backend.

Two cases:
- `kimi-k3-prefill-flydsl-k003-k006` — `token=7211`, **exact** trace shape. Covers k003 (+k006).
- `kimi-k3-decode-graph-k001-k002` — `token=62`, **reconstructed** (decode M not recorded). Covers k001+k002.

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

Logical per-rank (TP=8) MoE config: `model_dim=3584`, `inter_dim=384`
(= `moe_intermediate 3072 / TP8`), `experts=896`, `topk=16`,
`quant_type=per_1x32` (mxfp4 group_size 32), activation tensor `bf16`,
weight `fp4x2`, activation function **situ** (`situ_and_mul`), `g1u1=True`.

## Run environment (REQUIRED — this is the whole point)

Run in the **session build**: vLLM `0.1.dev19253+g5f76ae224.d20260727` plus its
matching aiter dev build, on MI355X/gfx950.

**Stock `amd-aiter 0.1.13.post1` cannot reproduce this** (verified on this host —
all three are concrete version/build differences, not spec errors):

1. **No `ActivationType.Situ`** — stock enum is only `Gelu/No/Silu/Swiglu`. The
   runner hard-fails with a clear message rather than substituting a wrong gate.
2. **No Kimi-K3 dispatch/tune rows** — stock `fused_moe` finds no tuned config for
   `(3584, 384, 896, 16)` and falls back to a generic ck2stages kernel that does
   not support `inter_dim=384` → `device_gemm ... does not support this GEMM
   problem`. (Bundled tune CSVs cover kimik2/dsv3/minimax/qwen3, not kimik3.)
3. **`shuffle_scale_a16w4` asserts on inter_dim=384** — 384/32 = 12 groups; stock
   asserts instead of padding to 16 (the session build pads 12→16, which is why
   the trace w2_scale is `(…,16)`).

The kernel *families* K3 used (`flydsl_moe1_afp4_wfp4_*`, `moe_ck2stages_gemm2_…
FP4X2`) do exist in stock aiter — what differs is the K3 dispatch/tuning, situ,
and scale padding. That is why the session ran fine and a stock-image extraction
does not: **different build**, confirmed by the vLLM version gap
(`0.1.dev19253` vs `0.24.0`).

## Notes

- The true device kernel is FlyDSL asm / ck2stages fp4, not an in-tree editable
  `.cu`; `config.yaml` points at `aiter/fused_moe.py` (the source-resolved dispatch
  wrapper / FlyDSL stage wrappers). Matches the session's "part-complete" (k003).
- Weights/scales are freshly generated (via aiter's own quant+shuffle helpers, so
  layouts stay build-consistent), not the real checkpoint — for kernel timing, not
  accuracy validation.
- `token=62` in the decode case is reconstructed (conc62 in the steady-state
  slice); the real graph-capture batch is not archived.

## Run

```
python3 scripts/task_runner.py compile       # smoke: one MoE call
python3 scripts/task_runner.py correctness   # cosine error < 0.03 vs torch_moe reference
python3 scripts/task_runner.py performance   # CUDA-graph timed, writes build/performance_report.json
```
