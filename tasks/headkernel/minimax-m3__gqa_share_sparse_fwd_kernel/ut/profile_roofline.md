# Roofline headroom — round 0 baseline (stage A, gfx950 / MI355X, TP=8)

- peaks: HBM 8.0 TB/s; MFMA bf16 2.5 PFLOP/s, fp8 5.0, fp4 10.0 (peaks.md, confidence high)
- stage **A** (shapes from the profile, not yet from the extractor) => every entry is `confidence: low`: **advisory only, do not rank on it**
- heads = pct_gpu_time >= 5% (prefill) + serving_weighted_pct >= 5% (decode)

| kernel | regime | class | edit | %gpu | sw% | t/launch | hbm_util | comp_util | bound | roofline% | target | speedup | e2e gain% | headroom |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `_gqa_share_sparse_fwd_kernel` | prefill | attn | Y | 21.8 | 7.4 | 1307.8 us | 0.026 | 0.042 | latency | 0.042 | 0.5 | 11.89 | 6.81 | underperforming |
| `allreduce_prototype_twoshot` | prefill | comm | N | 21.2 | 7.2 | 599.9 us | - | - | unknown | None | None | None | None | unknown |
| `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x128x256_` | prefill | moe | Y | 9.8 | 3.3 | 606.4 us | 0.077 | 0.051 | latency | 0.077 | 0.9 | 11.69 | 3.05 | underperforming |
| `_flash_attn_fwd_with_block_score_kernel` | prefill | attn | Y | 6.8 | 2.3 | 409.1 us | 0.328 | 0.538 | latency | 0.538 | 0.9 | 1.67 | 0.93 | underperforming |
| `_ZN5aiter26cross_device_reduce_2stageIDF16bLi8` | decode | comm | N | 1.1 | 8.8 | 15.6 us | - | - | unknown | None | None | None | None | unknown |
| `_decode_score_kernel` | decode | attn | Y | 1.0 | 7.9 | 29.8 us | 0.598 | 0.015 | latency | 0.598 | 0.85 | 1.42 | 2.33 | moderate |
| `_ZN7ck_tile6kentryILi2ENS_15MoeFlatmmKernelINS` | decode | moe | Y | 0.9 | 7.3 | 27.5 us | 1.276 | 0.009 | memory | 1.276 | 0.9 | None | None | unknown |
| `_gqa_share_sparse_decode_kernel` | decode | attn | Y | 0.7 | 5.8 | 21.8 us | 0.385 | 0.01 | latency | 0.385 | 0.5 | 1.3 | 1.33 | moderate |

## Per-entry notes

- **_gqa_share_sparse_fwd_kernel** (underperforming): Stage A. Per-launch = 1 layer of a 16384-token chunked-prefill step (1881 calls / 60 layers = 31.4 steps). Sparse model: each query attends topk_blocks=16 x block=128 = 2048 keys; per-rank kv heads = 1 (4 kv heads replicated over TP=8), q heads/rank = 8, head_dim 128, KV bf16. Unique KV bytes per launch are small (chunk-local, L2-resident), so the memory axis is not the binding roof. ALTERNATIVE model (kernel actually walking the FULL 8192-token context) gives 4.2e14 FLOP/s = 17% of the bf16 roof -> still underperforming. Either way this is the #1 editable prefill head with real headroom; sharpen with stage-C counters.

- **allreduce_prototype_twoshot** (unknown): NOT roofline-modelable: QuickReduce two-shot custom all-reduce (INT4-quantized) is bound by inter-GPU link bandwidth + peer synchronization, not by an HBM or FLOP roof. Per-launch payload [16384,6144] bf16 = 201 MB; local HBM traffic ~402 MB in 600 us = 0.67 TB/s = 8% of the HBM roof, i.e. HBM is idle - the cost is the fabric. Per-call skew 0.97 => it is NOT spin-inflated; the time is real. Lever = CONFIG (AR algorithm/quant, ROCM_QUICK_REDUCE_*, custom-AR vs quickreduce vs rccl, comm/compute overlap, TP layout), not a kernel rewrite.

- **mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x128x256_vscale_fix3_fp** (underperforming): Stage A. FlyDSL MXFP4 MoE grouped-GEMM stage-2 (down proj), per-launch = 1 MoE layer of a 16384-token chunk. pairs = 16384*top_k4 = 65536 => all 128 experts hit. Per-rank w2 = (3072/8)x6144 MXFP4 (0.53 B/elem incl. e8m0 scales) = 1.25 MB/expert => 160 MB weights + 12.6 MB fp4 activations in + 201 MB bf16 output ([16384,6144], assuming the topk reduction happens in-kernel). Largest feasible byte model chosen per the feasibility rule (a per-pair 805 MB output model is infeasible). Peak taken as fp4 = 1e16.

- **_flash_attn_fwd_with_block_score_kernel** (underperforming): Stage A. sglang Triton full-context flash attention + block-score emission (feeds the sparse top-k selection). FLOP model: 4*M(16384)*qheads_per_rank(8)*keys_avg(8192)*head_dim(128); keys_avg = the mid-prefill average context, a real error source. 1.35 PFLOP/s = 54% of the bf16 MFMA roof - a credible number for a Triton FA prefill, which also corroborates the peak table.

- **_ZN5aiter26cross_device_reduce_2stageIDF16bLi8ELb0EEEvPNS_8R** (unknown): NOT roofline-modelable (collective). #1 cost of the DECODE steady state: 12.7% of the decode window, 8.8% serving-weighted. Payload per launch is tiny ([64,6144] bf16 = 786 KB => ~0.1 TB/s, 1.3% of the HBM roof) so this is pure latency/sync per hop, 2.3 all-reduces per layer x 60 layers x 15.6 us. Per-call skew 1.05 => real, not spin. Lever = CONFIG: fewer/cheaper collectives (AR backend + quant, one-shot vs two-shot at batch 64, comm/compute overlap). This is the single biggest decode config lever.

- **_decode_score_kernel** (moderate): Stage A. Sparse-attention DECODE block scoring over the FULL context: batch 64 x ctx ~8704 (isl+osl/2) x index_dim 128 x bf16, 1 index head per rank. The 4-index-head variant implies 19 TB/s (infeasible) so 1 head/rank is the only feasible model -> the feasibility rule picks it. 4.8 TB/s = 60% of the HBM roof: genuinely bandwidth-bound. target_eff 0.85 (streaming-like scan).

- **_ZN7ck_tile6kentryILi2ENS_15MoeFlatmmKernelINS_33GemmSpatial** (unknown): Stage A. CK-tile MoE flat-MM, decode. M=64, top_k=4 => 256 pairs => experts_hit = 128*(1-(1-1/128)^256) = 110.8 of 128. Modeled as the stage-1 (gate+up) launch: per-rank w1 = 6144 x (2*3072/8) MXFP4 = 2.36 MB/expert x 110.8 = 262 MB in 27.5 us = 9.5 TB/s, ABOVE the 8 TB/s pin rate => infeasible => suspect, no verdict (L3). Interpretation: this kernel is AT or within ~20% of the memory wall (the overcount is likely routing skew / MALL-L2 reuse / fewer distinct experts). Route it to BYTE REDUCTION + host-side instance selection, not to a micro-tuning rewrite. Stage-C counter measurement (FETCH_SIZE+WRITE_SIZE) is the way to settle it.

- **_gqa_share_sparse_decode_kernel** (moderate): Stage A. Sparse GQA decode attention over the selected 2048 keys (16 blocks x 128): batch 64 x 2048 keys x 1 kv head/rank x 128 x (K,V) x bf16 = 67 MB in 21.8 us = 3.1 TB/s = 38% of the HBM roof. target_eff 0.50 (paged/irregular decode attention).


## Rankings (emitted side by side, deliberately)

- by %%gpu: _gqa_share_sparse_fwd_kernel, allreduce_prototype_twoshot, mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x128x256_vscale_fix3_fp, _flash_attn_fwd_with_block_score_kernel, _ZN5aiter26cross_device_reduce_2stageIDF16bLi8ELb0EEEvPNS_8R, _decode_score_kernel, _ZN7ck_tile6kentryILi2ENS_15MoeFlatmmKernelINS_33GemmSpatial, _gqa_share_sparse_decode_kernel
- by expected e2e gain: _gqa_share_sparse_fwd_kernel, mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x128x256_vscale_fix3_fp, _decode_score_kernel, _gqa_share_sparse_decode_kernel, _flash_attn_fwd_with_block_score_kernel, allreduce_prototype_twoshot, _ZN5aiter26cross_device_reduce_2stageIDF16bLi8ELb0EEEvPNS_8R, _ZN7ck_tile6kentryILi2ENS_15MoeFlatmmKernelINS_33GemmSpatial

roofline_pct measures how well each kernel executes its CURRENT byte/FLOP budget, not whether that budget is necessary. Stage A = low confidence: display/annotate only, do NOT rank on it; pct_gpu_time / serving_weighted_pct remain the primary key.

## Caution on the two 12x rows

`_gqa_share_sparse_fwd_kernel` and `mfma_moe2_*` show both utilization axes low (latency/occupancy-bound). At stage A that almost always means the analytic byte/FLOP model under-counts the real work, not that a 12x win is on the table. They are correctly identified as the heads with unexplained headroom; size the prize with stage-C rocprofv3 counters (FETCH_SIZE+WRITE_SIZE, MfmaFlops, OccupancyPercent, MemUnitStalled) before committing budget.
