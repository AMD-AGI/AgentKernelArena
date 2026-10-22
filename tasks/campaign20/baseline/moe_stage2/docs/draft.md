# moe_stage2 (kda) draft

## Bound diagnosis (from PROFILE + stage1 precedent)
Under-utilized: ~36% HBM, ~19% MFMA → occupancy/latency-bound. K=768 short loop.
Our case: tag=`a8w8blkscale`, per_1x128, fp8, inter_dim(=K to dispatch)=768 (%256==0), out fp16.
Compiled gemm2 instances (a8w8_gemm2_blockscale_kernels_list):
  bm16: V1 256/16/128/256 1x4 ; bm16(alt): V1 128/16/128/128 1x2 ; bm32: V1 256/32/128/128 1x4 ; bm64: **V3** 256/64/128/128 1x4
Host block_m: token<=32 -> 16 ; token>32 -> 64. tokens/expert = token/32.
  t=256 -> 8 tok/expert, bm64 => ~87% padding (L16 candidate)
  t>=8192 -> bm64 V3 (2-LDS, 1 block/CU) => occupancy-bound (L15 candidate)

## Levers (order)
1. **L15: V1 pipeline for block_m=64** — add V1 256/64/128/128 1x4 instance; switch bm64 dispatch V3->V1.
   Lower LDS (single buffer) -> 2 blocks/CU -> +occupancy on large tokens. Won stage1 (+11-16%).
2. **L16: tokens-per-expert-aware block_m** — for sparse routing (t=256, ~8 tok/expert) use bm16/32
   to cut padding waste. Host edit in fused_moe.py get_2stage_cfgs block_m formula.
3. Atomic-add scatter contention (stage2-specific) — only if 1&2 leave headroom.

## Validation
err_ratio<=0.20 AND cos_diff<0.02 vs torch_moe_stage2. Build ~80s. GPU2, HIP_VISIBLE_DEVICES=2.
Keep negatives.

## Results log
- baseline geomean 0.6166 ms
- C1 (L15 V1 bm64): geomean 0.5125 ms = 1.203x. PASS. token256 1.19x, 2048 1.25x, 8192 1.27x, 16384 1.30x. KEEP.
- C2 (L15 V1 + L16 bm16@sparse): geomean 0.4959 ms = 1.243x. token256 1.47x.
- microbench: bm32 (V1) >= bm64 across ENTIRE prefill span (2048 0.272 vs 0.316; 4096 0.436 vs 0.480;
  8192 0.787 vs 0.806; 16384 1.454 vs 1.572). bm64 never best. bm16 wins only <=8 tok/expert.
- C3 (L15 V1 + L16: bm16 if tok/expert<=8 else bm32): geomean 0.4841 ms = 1.274x. PASS. KEEP.
  per-case: 16:0.070 256:0.167 2048:0.274 4096:0.440 8192:0.795 11264:1.059 16384:1.597(noisy) 17920:1.593
- NEG C4 (bm32 KPerBlock=256, K=768->3 iters): geomean 0.4956 vs 0.4841 -> REGRESSION. KPerBlock=256
  raises reg/LDS pressure -> cuts occupancy on this occupancy-bound kernel. Reverted to KPerBlock=128.
- FINAL: L15 V1 (bm64 dispatch + instance) + L16 (bm16 if tok/expert<=8 else bm32). 3 runs: 0.4816/0.4818/0.4838
  geomean ~0.482 ms = 1.28x vs baseline 0.6166. All correctness PASS (err_ratio 0, cos_diff ~1.5e-7).
