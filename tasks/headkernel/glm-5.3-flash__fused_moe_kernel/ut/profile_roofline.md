# Roofline headroom - round 0 (stage A, gfx950 / MI355X, sglang TP=8, decode regime)

Peaks: HBM 8.0 TB/s, fp8 5.0 PFLOP/s, bf16 2.5 PFLOP/s (peaks.md, confidence high).
Scope: Top-N entries with pct_gpu_time >= 5%. Stage A = shapes from the trace/config, **display + annotate only, do not rank on it**.

| kernel | class | %gpu | t/unit us | bytes MB | achieved BW | hbm_util | comp_util | bound | roofline% | target | headroom | attainable | exp e2e gain% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `fused_moe_kernel` | moe | 23.32 | 96.3 | 754.9 | 7.84 TB/s | 0.980 | 0.0067 | memory | 98.0% | 0.90 | saturated | 1.00x | 0.00 |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x1024_MI` | gemm | 6.31 | 8.3 | 2.9 | 0.35 TB/s | 0.044 | 0.0073 | latency | 4.4% | 0.90 | unknown | 1.66x | 2.51 |
| `_ZN2ck59kernel_gemm_xdl_cshuffle_v3_multi_d_blockscale_b_pre` | gemm | 5.65 | 16.6 | 13.2 | 0.80 TB/s | 0.100 | 0.0194 | latency | 10.0% | 0.90 | underperforming | 3.32x | 3.95 |
| `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x32x512_MI1` | gemm | 5.55 | 12.2 | 26.1 | 2.14 TB/s | 0.268 | 0.0529 | latency | 26.8% | 0.90 | unknown | 2.44x | 3.27 |
| `main_kernel` | attn | 5.22 | 73.0 | 134.2 | 1.84 TB/s | 0.230 | 0.0471 | latency | 23.0% | 0.50 | underperforming | 2.18x | 2.82 |

## Per-entry notes

### fused_moe_kernel (moe, saturated)
- unit: one logical MoE layer = 2 launches (w1 gate/up + w2); 84.0 launches/step
- fp8 e4m3 experts, per-rank shard moe_intermediate 2048/TP8=256, E=288 top_k=8 M=64 (trace shapes [288,512,4096]+[288,4096,256] confirm). byte model used: experts_hit(239/288); all-expert=908 MB, experts_hit=755 MB. Stage A.
- byte-reduction levers (saturated => tuning has nothing left; move fewer bytes):
  - fuse the separate fp8 activation-quant epilogue (_per_token_group_quant_8bit, 2.04%gpu, 8820 launches) into the MoE kernel to remove an activation round-trip
  - stop streaming unrouted experts if the kernel reads all 288 (all-expert byte model is infeasible here, so it likely already skips - confirm at stage C)
  - fp8 -> fp4 (mxfp4) expert weights: halves the dominant weight stream. LOSSY - must pass the accuracy gate
  - raise arithmetic intensity by batching more tokens per expert visit - NOT available: conc/isl/osl are fixed by the measurement contract

### Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x1024_MI (gemm, unknown)
- unit: one launch, largest feasible listed shape [[64, 4096], [4096, 288]]; 132.0 launches/step
- decode GEMM at M=64 (skinny). dtype axis=bf16. Shape mixture under one Tensile/CK kernel; modeled on the largest listed case. Stage A. DISPATCH-BOUND: 8.3us/launch is within ~3x the ~5us launch-overhead floor, and the run is eager (--disable-cuda-graph, 1433 launches/step, 48% device idle per TraceLens) -> this launch is timed by dispatch, not by its bytes. No kernel-tuning verdict; lever is HIP-graph capture / fusion.

### _ZN2ck59kernel_gemm_xdl_cshuffle_v3_multi_d_blockscale_b_pre (gemm, underperforming)
- unit: one launch, largest feasible listed shape [[64, 4096], [3072, 4096], [64, 32], [24, 32], [64, 3072]]; 59.0 launches/step
- decode GEMM at M=64 (skinny). dtype axis=fp8. Shape mixture under one Tensile/CK kernel; modeled on the largest listed case. Stage A. LATENCY-BOUND (hbm_util 0.100, compute_util 0.0194 both <0.60 at 16.6us/launch): the lever is occupancy / tile+geometry at M=64 / fusion / HIP-graph capture, NOT byte reduction. The 9.03x roofline-formula speedup is an UPPER bound; capping at the ~5us launch floor gives a realistic 3.32x.

### Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x32x512_MI1 (gemm, unknown)
- unit: one launch, largest feasible listed shape [[64, 4096], [4096, 3072]]; 79.0 launches/step
- decode GEMM at M=64 (skinny). dtype axis=bf16. Shape mixture under one Tensile/CK kernel; modeled on the largest listed case. Stage A. DISPATCH-BOUND: 12.2us/launch is within ~3x the ~5us launch-overhead floor, and the run is eager (--disable-cuda-graph, 1433 launches/step, 48% device idle per TraceLens) -> this launch is timed by dispatch, not by its bytes. No kernel-tuning verdict; lever is HIP-graph capture / fusion.

### main_kernel (attn, underperforming)
- unit: one DSA attention launch (73us cluster); the 9.3us cluster is the indexer/kpool launch; 22.0 launches/step
- TileLang DSA attention (main_kernel = TVM/TileLang default device-fn name). MLA latent KV (kv_lora_rank=512, bf16 kv-cache, replicated across TP), index_topk=2048 selected tokens, batch=64. Shapes NOT in trace (tilelang launch carries no Input Dims) -> byte model from config.json. Stage A, LOW confidence. LATENCY-BOUND (hbm_util 0.230, compute_util 0.0471 both <0.60 at 73.0us/launch): the lever is occupancy / tile+geometry at M=64 / fusion / HIP-graph capture, NOT byte reduction. The 2.18x roofline-formula speedup is an UPPER bound; capping at the ~5us launch floor gives a realistic 2.18x.

## Rankings (emitted side by side deliberately)

- by pct_gpu_time: fused_moe_kernel, Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x1024_MI, _ZN2ck59kernel_gemm_xdl_cshuffle_v3_multi_d_blockscale_b_pre, Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x32x512_MI1, main_kernel
- by expected e2e gain: _ZN2ck59kernel_gemm_xdl_cshuffle_v3_multi_d_blockscale_b_pre, Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x32x512_MI1, main_kernel, Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT16x16x1024_MI, fused_moe_kernel

