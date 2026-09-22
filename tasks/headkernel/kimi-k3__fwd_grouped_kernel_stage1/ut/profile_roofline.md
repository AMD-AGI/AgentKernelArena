# Roofline headroom — round 0 baseline (gfx950, stage A)

- peaks: HBM 8.0 TB/s; bf16/fp16 2.5 PF/s; fp8 5.0 PF/s; fp4 10.0 PF/s (table, gfx950)
- scope: Top-N entries with `pct_gpu_time >= 5.0` (6 of 25). Advisory only — `pct_gpu_time` stays the primary key.

| # | kernel | regime | class | t/launch | launches/step | bytes | AI | hbm_util | comp_util | bound | roofline% | target | attainable | e2e gain% | verdict | conf |
|--|--------|--------|-------|----------|---------------|-------|----|----------|-----------|-------|-----------|--------|-----------|-----------|---------|------|
| 1 | `_fwd_grouped_kernel_stage1` | decode | attn | 0.436 ms | 24.0 | 0.64 GB | 22.7 | 0.184 | 0.013 | latency | 0.184 | 0.50 | 2.72x | 8.60 | **underperforming** | medium |
| 2 | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x2` | prefill | gemm | 0.873 ms | 439.82 | 0.52 GB | 2726.5 | 0.074 | 0.647 | compute | 0.647 | 0.90 | 1.39x | 2.57 | **moderate** | medium |
| 3 | `allreduce_prototype_twoshot` | prefill | unknown | 1.423 ms | 181.36 | - | - | - | - | unknown | - | - | - | - | **unknown** | low |
| 4 | `moe_gemm1_0` | prefill | moe | 1.767 ms | 89.21 | 3.59 GB | 401.8 | 0.254 | 0.082 | latency | 0.254 | 0.90 | 3.54x | 4.67 | **underperforming** | low |
| 5 | `moe_gemm2_0` | prefill | moe | 1.485 ms | 89.21 | 2.74 GB | 263.8 | 0.230 | 0.049 | latency | 0.230 | 0.90 | 3.91x | 4.07 | **underperforming** | low |
| 6 | `_score_kernel` | prefill | unknown | 0.662 ms | 180.36 | - | - | - | - | unknown | - | - | - | - | **unknown** | low |

## Ranking disagreement

- by `pct_gpu_time`: _fwd_grouped_kernel_stage1 (decode), Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x64_MI (prefill), allreduce_prototype_twoshot (prefill), moe_gemm1_0 (prefill), moe_gemm2_0 (prefill), _score_kernel (prefill)
- by `expected_e2e_gain_pct`: _fwd_grouped_kernel_stage1 (decode), moe_gemm1_0 (prefill), moe_gemm2_0 (prefill), Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x64_MI (prefill), allreduce_prototype_twoshot (prefill), _score_kernel (prefill)

## Per-entry notes

### _fwd_grouped_kernel_stage1 (decode, 13.61% gpu) — underperforming
MLA absorbed decode: KV latent (kv_lora_rank 512 + qk_rope 64) bf16, NOT TP-sharded (each rank reads the full latent KV). bs=64, seq_len taken as isl+osl/2=8704 (real error source: the captured step's true mean context is unknown). 12 q-heads/rank. One launch per full-attn layer, 24 of 93 layers. Unit = ONE launch.
byte-reduction levers: fp8 KV-cache dtype (halves latent bytes; lossy -> accuracy gate); longer page/tile for better coalescing of the paged latent reads

### Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x64_MI (prefill, 9.14% gpu) — moderate
Dominant captured case [16384,7168]x[7168,6016] bf16 (2852 of 14514 window launches, 873.5 us each); t_ms is that case, not the whole-kernel median. Unit = ONE launch. COMPUTE-AXIS CAVEAT: the bf16 MFMA peak is the least-validated (SKILL 8.2).

### allreduce_prototype_twoshot (prefill, 9.12% gpu) — unknown
L2 DEGRADE: collective all-reduce: no interconnect peak in peaks.md (op_class unknown). 

### moe_gemm1_0 (prefill, 6.51% gpu) — underperforming
MoE stage-1 gate+up, mxfp4 weights (Float4_e2m1fn_x2, group 32) + bf16 acts, TP-sharded intermediate (3072 -> 384/rank). pairs=M*top_k=262144; at prefill M=16384 every one of E=896 experts is touched so full weight streaming is unavoidable; at decode M=64 -> pairs=1024, experts_hit~896 of 896. Bytes = W 1.31 GB + A 1.88 GB + C 0.40 GB (feasibility cap 14.1 GB). Peak used = fp4 1.0e16 (CDNA4 native mxfp4); if the path dequantizes to bf16 the compute axis moves. Unit = ONE launch.
byte-reduction levers: skip unrouted experts at decode (experts_hit << E); keep the pair-major intermediate in LDS/registers to avoid the A/C round-trip (fuse stage1+stage2)

### moe_gemm2_0 (prefill, 5.47% gpu) — underperforming
MoE stage-2 down, mxfp4 weights (Float4_e2m1fn_x2, group 32) + bf16 acts, TP-sharded intermediate (3072 -> 384/rank). pairs=M*top_k=262144; at prefill M=16384 every one of E=896 experts is touched so full weight streaming is unavoidable; at decode M=64 -> pairs=1024, experts_hit~896 of 896. Bytes = W 0.66 GB + A 0.20 GB + C 1.88 GB (feasibility cap 11.9 GB). Peak used = fp4 1.0e16 (CDNA4 native mxfp4); if the path dequantizes to bf16 the compute axis moves. Unit = ONE launch.
byte-reduction levers: skip unrouted experts at decode (experts_hit << E); keep the pair-major intermediate in LDS/registers to avoid the A/C round-trip (fuse stage1+stage2)

### _score_kernel (prefill, 5.26% gpu) — unknown
L2 DEGRADE: no operand shapes captured for this kernel (stage-A unmodelable). 

