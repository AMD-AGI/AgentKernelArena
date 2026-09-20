# DeepSeek-V4-Pro — MoE stage-1 `moe_stage1_grouped_gemm_silu_flydsl`

| Model | ISL | OSL | CONC | Docker / Image | Kernel | GPU Time Share (%) | Current Roofline | HL Run Directory |
|---|---|---|---|---|---|---|---|---|
| DeepSeek-V4-Pro | 8192 | 1024 | 64 | harbor.crusoe.primus-safe.amd.com/hyperloom-image/sglang:v0.5.17-rocm720-mi35x-profilerfix | moe_stage1_grouped_gemm_silu_flydsl | 7.31% (blended, bigmoe3) / 12.54% (extraction run) | ~100% HBM — memory-saturated | provenance://shared-nfs/zihao/arena/exp/DeepSeek-V4-Pro-bigmoe/DeepSeek-V4-Pro/20260828T074807Z-9f06d44e |

Device symbols as the profiler saw them:

- `mfma_moe1_silu_mul_afp8_wfp4_fp8_t64x128x256_pm1_fp8q_sort_async_gui_xcd4_v32` — prefill tile
- `mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256_pm1_fp8q_sort_async_gui_xcd4_v32` — decode tile

One callable, two tile variants selected by M. The UT covers both (5 cases, 2 prefill + 3 decode).

## Kernel identity

| field | value |
|---|---|
| `target_callable` | `aiter.ops.flydsl:flydsl_moe_stage1` |
| `source_path_in_sglang` | `provenance://runtime-image/aiter/aiter/ops/flydsl/moe_kernels.py` |
| `op_kind` | moe |
| `backend` | sglang, via aiter flydsl |
| `candidate_backends` | triton, aiter, hip, flydsl |
| `regimes_captured` | decode, prefill |
| `graph_replayed` | True |
| `oracle_complete` | True |

Math contract (verbatim from `meta.json`):

> grouped per-expert GEMM + routing: `out[t,k,:] = quant_fp8( silu(gate)*up )` where `[gate|up] = a[t,:] @ w1[e(t,k)]^T`
> (mxfp8 act x mxfp4 weights, e8m0 block scales), scattered by the moe_sorting block map;
> returns `(inter_states_fp8, out_scale_sorted_e8m0)`

## Roofline

No automatic roofline entry exists at full confidence: the round-0 pass only models entries
>= 5% GPU time, and this kernel's byte model came back **infeasible**, so the skill refused to
emit a verdict.

| quantity | value | source |
|---|---|---|
| avg launch | 57.2 us, 61 launches/step (decode) | `bigmoe3` round-0 top-N |
| bytes_est | 669 MB/launch (243 of 384 experts hit, uniform-routing prior) | round-0 roofline |
| achieved | 11.71 TB/s = **146% of the 8 TB/s roof** ⇒ infeasible, model over-counts | round-0 roofline |
| flops_est | 4.23 GFLOP -> 74.0 TFLOP/s = 1.48% of the fp8 roof | round-0 roofline |
| AI / ridge | 6.32 vs 625.0 | round-0 roofline |
| verdict | `bound_type: memory`, `headroom_class: unknown` (L3 suspect), stage-C candidate | round-0 roofline |

What the infeasibility does prove: the kernel is on the memory axis and is streaming expert
weights at whatever HBM can deliver, and the true distinct-expert count must be <= ~166, not the
~243 a uniform-routing prior predicts — consistent with the group/hash-limited routing
(`noaux_tc`, `num_hash_layers=3`). At 166 experts the byte model lands at ~7.98 TB/s ≈ 100% of
the roof, which agrees independently with the stage-2 kernel's own over-count ratio.

**Lever: byte reduction / routing, not another tuning pass.**

- stream only routed experts (skip unrouted expert weight reads)
- tile/config tune the flydsl `t32x128x256` variant for M=64 decode (aiter tuned-config DB / flydsl variant bake-off)
- fuse the separate quant/sort prologues (`fused_mx_quant_moe_sort`, `dynamic_per_group_scaled_quant`) into the stage-1 launch to remove an activation round trip
- fuse stage-1 with stage-2 (`opus_moe_stage2_a8w4`, the sibling UT) to keep the intermediate in L2

## Extraction provenance

Copied from
`provenance://shared-nfs/zihao/arena/exp/DeepSeek-V4-Pro-bigmoe/DeepSeek-V4-Pro/20260828T074807Z-9f06d44e/geak/e2e_cycle0/kernels/moe_stage1_grouped_gemm_silu_flydsl_task`
(`__pycache__` and `.torch_ext` build caches excluded; nothing else changed).

Note this is the **bigmoe** run (2026-08-28), not the **bigmoe3** run the MLA UT came from. Same
model, same image, same ISL/OSL/CONC, different session — which is why the two `pct_gpu_time`
figures above differ (12.54% in its own profile vs 7.31% blended in bigmoe3).

Selection validation (`selection_validation.json`): `ok: true`, `deepest_verified: true`,
5 target-marker calls, 580 correlated external ids, evidence
`installed+live_nested_candidate_markers+torch_profiler_external_id_or_launch_correlation`.

## What is in this directory

| file | role |
|---|---|
| `unittest.py` | the UT entry point — correctness + timing for this kernel |
| `cases.py` | IMMUTABLE case definitions; `call(args)` is the seam both legs go through |
| `meta.json` | kernel geometry, `target_callable`, workload cases, `pct_gpu_time` |
| `harness_lib.py` | timing / correctness primitives (`time_op`, `correct`, ...) |
| `reference_io.pt` | the frozen oracle: real captured production inputs + outputs (1.5 GB) |
| `workload.json` | serving regime, quant scheme, case weights |
| `kernel_src/` | the editable kernel source — this is what an optimization patches |
| `baseline_ref/`, `baseline_overlay/` | the untouched reference implementation and its overlay |
| `_baseline_random.pt` | random-input baseline for the non-oracle timing leg |
| `overlay_setup.py`, `leg_runner.py` | overlay install + per-leg driver |
| `regime.json` | the serving regime the capture was taken under |
| `selection_validation*.json` | proof that the captured callable IS the profiled device kernel |

`unittest.py`, `cases.py`, `meta.json` and `reference_io.pt` are immutable — an optimization
edits `kernel_src/` or ships a candidate overlay.
