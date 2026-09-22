# DeepSeek-V4-Pro — MoE stage-2 `moe_stage2_down_proj_reduce_opus_a8w4`

| Model | ISL | OSL | CONC | Docker / Image | Kernel | GPU Time Share (%) | Current Roofline | HL Run Directory |
|---|---|---|---|---|---|---|---|---|
| DeepSeek-V4-Pro | 8192 | 1024 | 64 | harbor.crusoe.primus-safe.amd.com/hyperloom-image/sglang:v0.5.17-rocm720-mi35x-profilerfix | moe_stage2_down_proj_reduce_opus_a8w4 | 3.71% (blended, bigmoe3) / 8.65% (extraction run) | ~98% HBM — memory-saturated | /shared_nfs/zihao/arena/exp/DeepSeek-V4-Pro-bigmoe/DeepSeek-V4-Pro/20260828T074807Z-9f06d44e |

Device symbols as the profiler saw them — the head is the **sum of the two**, i.e. the stage-2
segment, not a single dispatch:

- `opus_moe_stage2_a8w4_decode_kernel_gfx950` — the down GEMM (decode atomic kid 2005, prefill route-out kid 2003)
- `opus_moe_stage2_reduce_token_slot_route_output_kernel_gfx950` — the prefill top-k reduce

Despite the `_decode_` in the name, the first kernel runs in BOTH regimes (61 launches/prefill-step
at 278 us, 61/decode-step at 12.4-29.0 us). The UT covers both (5 cases).

## Kernel identity

| field | value |
|---|---|
| `target_callable` | `aiter.ops.opus.moe_stage2_a8w4_fused_adapter:opus_a8w4_stage2_wrapper` |
| `source_path_in_sglang` | `/sgl-workspace/aiter/aiter/ops/opus/moe_stage2_a8w4_fused_adapter.py` |
| `op_kind` | moe |
| `backend` | sglang, via aiter opus |
| `candidate_backends` | triton, aiter, hip, flydsl |
| `regimes_covered` | decode, prefill |
| `graph_replayed` | True |
| `oracle_complete` | True |

Math contract (verbatim from `meta.json`):

> stage-2 of the a8w4 fused MoE, as ONE segment: `out[t,:] = sum_k topk_w[t,k] * (inter_states[t,k,:] @ w2[e(t,k)]^T)`
> with mxfp8 activations x mxfp4 expert weights and e8m0 block scales, scattered through the
> moe_sorting block map. decode (`route_out=False`): the down-GEMM atomically accumulates the topk.

## Roofline

The round-0 automatic pass did **not** model this entry — it only models entries >= 5% blended
GPU time, and this one sits at 3.71%. The numbers below are computed by hand with the same
byte/FLOP model the skill uses, at the same measured time.

| quantity | value | basis |
|---|---|---|
| avg launch | 29.0 us, 61 launches/step (decode) | `bigmoe3` round-0 top-N |
| per-expert local w2 | (3072/TP8) x 7168 x fp4(0.5 B) = 1.376 MB | model config |
| bytes_est | 243 experts hit x 1.376 MB = 334 MB/launch | uniform-routing prior |
| achieved | 11.53 TB/s = **144% of the 8 TB/s roof** ⇒ infeasible, model over-counts | derived |
| feasible ceiling | 8 TB/s x 29.0 us = 232 MB ⇒ <= ~168 distinct experts touched | derived |
| at 166 experts | 228 MB / 29.0 us = 7.87 TB/s = **~98% of the roof** | derived |
| flops_est | 2 x 384 pairs x 384 x 7168 = 2.11 GFLOP -> 72.9 TFLOP/s = 1.5% of the fp8 roof | derived |
| verdict | memory-bound, at the roof | derived |

This kernel's over-count ratio (144%) agrees with stage-1's (146%), and both back-solve to the
same ~166 distinct experts per launch rather than the ~243 uniform routing predicts. Two
independent kernels landing on the same correction is the strongest evidence available that the
routing really is group/hash-limited (`noaux_tc`, `num_hash_layers=3`) — and that both MoE stages
are genuinely streaming expert weights at HBM pin rate.

Confidence: stage A, hand-computed byte model, no FETCH_SIZE counters on this image. Treat as
routing evidence, not as a measurement.

**Lever: byte reduction / routing, not another tuning pass.**

- stream only routed experts
- fuse stage-1 and stage-2 to keep `inter_states` in L2 instead of round-tripping HBM
- the prefill `reduce_token_slot_route_output` dispatch is a separate launch — a fusion candidate
  in its own right, same shape of win as the MLA partial+combine fusion that produced 3.43x

## Extraction provenance

Copied from
`/shared_nfs/zihao/arena/exp/DeepSeek-V4-Pro-bigmoe/DeepSeek-V4-Pro/20260828T074807Z-9f06d44e/geak/e2e_cycle0/kernels/moe_stage2_down_proj_reduce_opus_a8w4_task`
(`__pycache__` and `.torch_ext` build caches excluded; nothing else changed).

Note this is the **bigmoe** run (2026-08-28), not the **bigmoe3** run the MLA UT came from. Same
model, same image, same ISL/OSL/CONC, different session — which is why the two `pct_gpu_time`
figures above differ (8.65% in its own profile vs 3.71% blended in bigmoe3).

Selection validation (`selection_validation.json`): `ok: true`, `deepest_verified: true`,
16 target-marker calls matching 16 kernel calls, 291 correlated external ids, evidence
`installed+live_nested_candidate_markers+torch_profiler_external_id_or_launch_correlation`.

## What is in this directory

| file | role |
|---|---|
| `unittest.py` | the UT entry point — correctness + timing for this kernel |
| `cases.py` | IMMUTABLE case definitions; `call(args)` is the seam both legs go through |
| `meta.json` | kernel geometry, `target_callable`, workload cases, `pct_gpu_time` |
| `harness_lib.py` | timing / correctness primitives (`time_op`, `correct`, ...) |
| `reference_io.pt` | the frozen oracle: real captured production inputs + outputs (2.2 GB) |
| `workload.json` | serving regime, quant scheme, case weights |
| `kernel_src/` | the editable kernel source — this is what an optimization patches |
| `baseline_ref/`, `baseline_overlay/` | the untouched reference implementation and its overlay |
| `_baseline_random.pt` | random-input baseline for the non-oracle timing leg (1.5 GB) |
| `_capture_shapes_enhanced.py` | the shape-capture driver used during extraction |
| `overlay_setup.py`, `leg_runner.py` | overlay install + per-leg driver |
| `regime.json` | the serving regime the capture was taken under |
| `selection_validation*.json` | proof that the captured callable IS the profiled device kernel |

`unittest.py`, `cases.py`, `meta.json` and `reference_io.pt` are immutable — an optimization
edits `kernel_src/` or ships a candidate overlay.
