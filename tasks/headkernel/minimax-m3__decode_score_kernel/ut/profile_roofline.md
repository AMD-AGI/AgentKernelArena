# Roofline analysis — round `tuning` (stage A)

gfx950 · peaks: HBM 8.0 TB/s, bf16 2.5 PF/s, fp4 10 PF/s · source: `/wekafs/test_results/Minimax_m3_MXFP4_20260828/MiniMax-M3-MXFP4/20260828T105325Z-4d49e44f/geak/e2e_cycle0/profile/round_tuning/profile_topN.json`

Head = `pct_gpu_time` >= 5 (prefill heads) plus decode heads with `serving_weighted_pct` >= 5. Same 8 kernels as round 0.

| # | kernel | regime | %gpu | sw% | t/launch (ms) | vs r0 | AI | bound | roofline_pct | target | speedup | exp e2e gain % | class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `_gqa_share_sparse_fwd_kernel` | prefill | 21.98 | 7.61 | 1.32982 | +1.7% | 512.0 | latency | 0.041 | 0.5 | 12.09 | 6.98 | underperforming |
| 2 | `allreduce_prototype_twoshot` | prefill | 21.27 | 7.41 | 0.6064 | +1.1% | — | unknown | — | — | — | — | unknown |
| 3 | `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x128x256_vs` | prefill | 9.84 | 3.43 | 0.61435 | +1.3% | 827.7 | latency | 0.076 | 0.9 | 11.84 | 3.14 | underperforming |
| 4 | `_flash_attn_fwd_with_block_score_kernel` | prefill | 6.77 | 2.34 | 0.40933 | +0.1% | 512.0 | latency | 0.537 | 0.9 | 1.68 | 0.94 | moderate |
| 5 | `_ZN5aiter26cross_device_reduce_2stageIDF16bLi8EL` | decode | 1.34 | 10.73 | 0.01881 | +20.6% | — | unknown | — | — | — | — | unknown |
| 6 | `_decode_score_kernel` | decode | 1.0 | 7.98 | 0.02973 | -0.2% | 8.0 | latency | 0.6 | 0.85 | 1.42 | 2.35 | moderate |
| 7 | `_ZN7ck_tile6kentryILi2ENS_15MoeFlatmmKernelINS_3` | decode | 0.76 | 6.12 | 0.02277 | -17.2% | 8.6 | memory | 1.0 | 0.9 | — | — | unknown |
| 8 | `_gqa_share_sparse_decode_kernel` | decode | 0.73 | 5.85 | 0.0218 | +0.0% | 8.0 | latency | 0.385 | 0.5 | 1.3 | 1.35 | moderate |

## Round-over-round movement (per-launch time)

- `_gqa_share_sparse_fwd_kernel` (prefill): 1.3078 -> 1.32982 ms/launch (+1.7%)
- `allreduce_prototype_twoshot` (prefill): 0.5999 -> 0.6064 ms/launch (+1.1%)
- `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x128x256_vscale_fix` (prefill): 0.6064 -> 0.61435 ms/launch (+1.3%)
- `_flash_attn_fwd_with_block_score_kernel` (prefill): 0.4091 -> 0.40933 ms/launch (+0.1%)
- `_ZN5aiter26cross_device_reduce_2stageIDF16bLi8ELb0EEEvPN` (decode): 0.0156 -> 0.01881 ms/launch (+20.6%)
- `_decode_score_kernel` (decode): 0.0298 -> 0.02973 ms/launch (-0.2%)
- `_ZN7ck_tile6kentryILi2ENS_15MoeFlatmmKernelINS_33GemmSpa` (decode): 0.0275 -> 0.02277 ms/launch (-17.2%)
- `_gqa_share_sparse_decode_kernel` (decode): 0.0218 -> 0.0218 ms/launch (+0.0%)

## Notes per entry

### `_gqa_share_sparse_fwd_kernel`

Stage A (byte/FLOP model carried over from round 0; only t/launch was re-measured). Per-launch = 1 layer of a 16384-token chunked-prefill step (1881 calls / 60 layers = 31.4 steps). Sparse model: each query attends topk_blocks=16 x block=128 = 2048 keys; per-rank kv heads = 1 (4 kv heads replicated over TP=8), q heads/rank = 8, head_dim 128, KV bf16. Unique KV bytes per launch are small (chunk-local, L2-resident), so the memory axis is not the binding roof. ALTERNATIVE model (kernel actually walking the FULL 8192-token context) gives 4.2e14 FLOP/s = 17% of the bf16 roof -> still underperforming. Either way this is the #1 editable prefill head with real headroom; sharpen with stage-C counters.

> CAUTION: attainable_speedup 11.9x is NOT credible at face value: both utilization axes are low, which at stage A usually means the analytic byte/FLOP model UNDER-counts the work this kernel really does (sparse block selection / grouped-GEMM tail / expert padding), not that a 12x kernel win exists. Read it as "this head has real, unexplained headroom - measure it with stage-C counters before sizing the prize"; do not quote the number as a target.

### `allreduce_prototype_twoshot`

NOT roofline-modelable: QuickReduce two-shot custom all-reduce (INT4-quantized) is bound by inter-GPU link bandwidth + peer synchronization, not by an HBM or FLOP roof. Per-launch payload [16384,6144] bf16 = 201 MB; local HBM traffic ~402 MB in 600 us = 0.67 TB/s = 8% of the HBM roof, i.e. HBM is idle - the cost is the fabric. Per-call skew 0.97 => it is NOT spin-inflated; the time is real. Lever = CONFIG (AR algorithm/quant, ROCM_QUICK_REDUCE_*, custom-AR vs quickreduce vs rccl, comm/compute overlap, TP layout), not a kernel rewrite.

### `mfma_moe2_afp4_wfp4_bf16_cshuffle_t64x128x256_vscale_fix3_fp`

Stage A (byte/FLOP model carried over from round 0; only t/launch was re-measured). FlyDSL MXFP4 MoE grouped-GEMM stage-2 (down proj), per-launch = 1 MoE layer of a 16384-token chunk. pairs = 16384*top_k4 = 65536 => all 128 experts hit. Per-rank w2 = (3072/8)x6144 MXFP4 (0.53 B/elem incl. e8m0 scales) = 1.25 MB/expert => 160 MB weights + 12.6 MB fp4 activations in + 201 MB bf16 output ([16384,6144], assuming the topk reduction happens in-kernel). Largest feasible byte model chosen per the feasibility rule (a per-pair 805 MB output model is infeasible). Peak taken as fp4 = 1e16.

> CAUTION: attainable_speedup 11.7x is NOT credible at face value: both utilization axes are low, which at stage A usually means the analytic byte/FLOP model UNDER-counts the work this kernel really does (sparse block selection / grouped-GEMM tail / expert padding), not that a 12x kernel win exists. Read it as "this head has real, unexplained headroom - measure it with stage-C counters before sizing the prize"; do not quote the number as a target.

### `_flash_attn_fwd_with_block_score_kernel`

Stage A (byte/FLOP model carried over from round 0; only t/launch was re-measured). sglang Triton full-context flash attention + block-score emission (feeds the sparse top-k selection). FLOP model: 4*M(16384)*qheads_per_rank(8)*keys_avg(8192)*head_dim(128); keys_avg = the mid-prefill average context, a real error source. 1.35 PFLOP/s = 54% of the bf16 MFMA roof - a credible number for a Triton FA prefill, which also corroborates the peak table.

### `_ZN5aiter26cross_device_reduce_2stageIDF16bLi8ELb0EEEvPNS_8R`

NOT roofline-modelable (collective). #1 cost of the DECODE steady state: 12.7% of the decode window, 8.8% serving-weighted. Payload per launch is tiny ([64,6144] bf16 = 786 KB => ~0.1 TB/s, 1.3% of the HBM roof) so this is pure latency/sync per hop, 2.3 all-reduces per layer x 60 layers x 15.6 us. Per-call skew 1.05 => real, not spin. Lever = CONFIG: fewer/cheaper collectives (AR backend + quant, one-shot vs two-shot at batch 64, comm/compute overlap). This is the single biggest decode config lever.

### `_decode_score_kernel`

Stage A (byte/FLOP model carried over from round 0; only t/launch was re-measured). Sparse-attention DECODE block scoring over the FULL context: batch 64 x ctx ~8704 (isl+osl/2) x index_dim 128 x bf16, 1 index head per rank. The 4-index-head variant implies 19 TB/s (infeasible) so 1 head/rank is the only feasible model -> the feasibility rule picks it. 4.8 TB/s = 60% of the HBM roof: genuinely bandwidth-bound. target_eff 0.85 (streaming-like scan).

### `_ZN7ck_tile6kentryILi2ENS_15MoeFlatmmKernelINS_33GemmSpatial`

Stage A (byte/FLOP model carried over from round 0; only t/launch was re-measured). CK-tile MoE flat-MM, decode. M=64, top_k=4 => 256 pairs => experts_hit = 128*(1-(1-1/128)^256) = 110.8 of 128. Modeled as the stage-1 (gate+up) launch: per-rank w1 = 6144 x (2*3072/8) MXFP4 = 2.36 MB/expert x 110.8 = 262 MB in 27.5 us = 9.5 TB/s, ABOVE the 8 TB/s pin rate => infeasible => suspect, no verdict (L3). Interpretation: this kernel is AT or within ~20% of the memory wall (the overcount is likely routing skew / MALL-L2 reuse / fewer distinct experts). Route it to BYTE REDUCTION + host-side instance selection, not to a micro-tuning rewrite. Stage-C counter measurement (FETCH_SIZE+WRITE_SIZE) is the way to settle it.

### `_gqa_share_sparse_decode_kernel`

Stage A (byte/FLOP model carried over from round 0; only t/launch was re-measured). Sparse GQA decode attention over the selected 2048 keys (16 blocks x 128): batch 64 x 2048 keys x 1 kv head/rank x 128 x (K,V) x bf16 = 67 MB in 21.8 us = 3.1 TB/s = 38% of the HBM roof. target_eff 0.50 (paged/irregular decode attention).


## Doctrine

roofline_pct measures how well each kernel executes its CURRENT byte/FLOP budget, not whether that budget is necessary. Stage A = low confidence: display/annotate only, do NOT rank on it; pct_gpu_time / serving_weighted_pct remain the primary key.


## Degraded / not modeled

- `allreduce_prototype_twoshot`: collective: no HBM/FLOP roof applies (link-bandwidth + sync bound)
- `aiter cross_device_reduce_2stage`: collective: no HBM/FLOP roof applies (link-latency/sync bound)

## Round-`tuning` interpretation

- The tuning deploy (aiter dense-bf16 tuned-GEMM table, 20 rows) does NOT touch any of the 8 roofline heads: none of them is a dense bf16 GEMM. The heads moved <=1.7% except the two decode entries below, so the roofline picture is unchanged from round 0 and the same three levers remain: (1) collectives (config), (2) _gqa_share_sparse_fwd prefill attention (editable, unexplained headroom), (3) MoE grouped-GEMM byte reduction.
- cross_device_reduce_2stage got 20.6% SLOWER per launch (15.6 -> 18.8 us) and is now the #1 serving-weighted cost at 10.73% (was 8.75%). It is not roofline-modelable and the deploy cannot have changed the collective itself; the most likely cause is that the decode step got shorter elsewhere so the peer-wait/skew per all-reduce grew, i.e. the collective absorbed part of the compute win. This makes the AR config lever MORE valuable, not less.
- ck_tile MoeFlatmm decode stage-1 got 17.2% FASTER (27.5 -> 22.8 us). At the round-0 byte model this now implies 12.3 TB/s = 1.54x the HBM pin rate, i.e. the analytic byte model is now clearly over-counting (it was already suspect at 1.28x in round 0). Verdict withheld (L3, stage_c_candidate) - settle with FETCH_SIZE/WRITE_SIZE counters before sizing any MoE byte-reduction prize.
- The dense-bf16 GEMM seam the deploy actually targets sits BELOW the roofline head threshold in both rounds (largest single entry 3.55 %gpu / 2.92 sw%), which is why a +1.3% e2e is the right order of magnitude for this artifact and why the roofline head set is unchanged.
