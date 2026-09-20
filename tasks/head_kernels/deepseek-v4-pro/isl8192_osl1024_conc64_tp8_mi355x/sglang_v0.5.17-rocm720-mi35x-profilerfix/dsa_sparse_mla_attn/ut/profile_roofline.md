# Roofline headroom analysis — round 0 baseline (stage A, advisory)

- gfx: `gfx950` — peaks: HBM **8.0 TB/s**, bf16 **2.5 PFLOP/s**, fp8 **5.0 PFLOP/s** (table, bf16==fp16 equality check holds)
- stage **A** (profile time, no captured operand shapes yet) ⇒ every entry `confidence: low` — **display/annotate only, do not rank on it**. `pct_gpu_time` stays the primary routing key.
- scope: only Top-N entries at **≥ 5.0% blended e2e GPU time** are modelled (2 of 25). Smaller entries are omitted by design, not degraded.
- e2e-critical regime = **decode** (96 decode steps vs ~1 chunked-prefill step in the capture; decode = 0.6292 of e2e time).

| # | kernel | class | regime | t/launch | hbm_util | compute_util | bound | roofline_pct | target | headroom | attainable | exp. e2e gain |
|--|--------|-------|--------|----------|----------|--------------|-------|--------------|--------|----------|------------|---------------|
| 1 | `main_kernel` | attn | decode / full-sparse-MLA layers | 206.8 µs | 0.388 | 0.0044 | latency | 39% | 0.50 | **moderate** | 1.29x | +10.1% |
| 2 | `mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256_pm1_f` | moe | decode | 57.2 µs | 1.464 | 0.0148 | memory | 100% (raw 146% ⚠ suspect) | 0.90 | **unknown** | — | — |

## Both rankings, side by side (they disagree — that is the point)

- by `pct_gpu_time`: `main_kernel`, `mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256`
- by expected e2e gain: `main_kernel`, `mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256`

> Doctrine: `roofline_pct` measures how well the kernel executes its CURRENT byte/FLOP budget — never whether that budget is necessary. A saturated head is a byte-reduction target, not a finished one.

## #1 `main_kernel` — 44.88% e2e GPU time, moderate

- unit: 1 launch, 30 launches/step, 206.8 µs each; bytes_est **642 MB**, flops_est **2.28 GFLOP**
- achieved **3.10 TB/s** (38.8% of HBM roof) / **11.0 TFLOP/s** (0.44% of compute roof); AI 3.56 vs ridge 312.5
- bound_type **latency**, editable=True
- levers:
  - fp8 / lower-precision KV latent (lossy - accuracy gate)
  - fuse the DSA indexer top-k pass into the attention launch (mhc_pre_big_fuse / topk_transform are separate Top-N rows)
  - page/layout the latent KV so the top-1024 gather is contiguous

Unit = ONE launch of the full-sparse-MLA decode layer (30 of 61 layers, 30 launches/step, avg 206.8us). Two candidate byte models; per SKILL sec.4 the LARGEST FEASIBLE one is used: (A) FULL-CONTEXT read -- batch64 x ctx(isl+osl/2=8704) x (kv_lora 512 + rope 64) x bf16 = 642 MB -> 3.10 TB/s = 39% of the HBM roof [USED: feasible, and the pessimistic/conservative headroom]. (B) TRUE-SPARSE read -- only the index_topk=1024 selected tokens per query = 75 MB -> 0.37 TB/s = 4.6% of the roof, which would imply ~11.0x attainable instead of the conservative value reported. The MLA latent KV is replicated on every TP rank (num_key_value_heads=1), so no /TP on bytes; FLOPs use 16 local q-heads (128/TP8) over the absorbed 576-dim latent. THE TWO MODELS DISAGREE BY 8.5x -- this entry is the #1 priority for a stage-B/C re-run (real captured shapes / rocprofv3 FETCH_SIZE) because it decides whether the run's dominant kernel has ~1.3x or ~10x of headroom. Corroborating evidence for the low end being wrong: the PREFILL regime of the same kernel (24.45 ms per full-sparse layer for an 8192-token chunk) works out to ~11 TFLOP/s and <0.5 TB/s -- ~0.4% of the compute roof and ~5% of the memory roof, i.e. far from BOTH roofs in the phase where attention should be compute-dense; that is a latency/occupancy signature, not a saturated kernel. Stage A: confidence LOW, advisory only -- do NOT rank on attainable_speedup.

## #2 `mfma_moe1_silu_mul_afp8_wfp4_fp8_t32x128x256_pm1_f` — 7.31% e2e GPU time, unknown

- unit: 1 launch, 61 launches/step, 57.2 µs each; bytes_est **669 MB**, flops_est **4.23 GFLOP**
- achieved **11.71 TB/s** (146.3% of HBM roof) / **74.0 TFLOP/s** (1.48% of compute roof); AI 6.32 vs ridge 625.0
- bound_type **memory**, editable=True
- ⚠ **L3 suspect**: raw roofline_pct 146% > 100% ⇒ byte model over-counts; verdict forced to `unknown`, flagged as a **stage-C counter-measurement candidate**. Feasible byte ceiling at this time = 457 MB.
- levers:
  - stream only routed experts (skip unrouted expert weight reads)
  - tile/config tune the flydsl t32x128x256 variant for M=64 decode (aiter tuned-config DB / flydsl variant bake-off)
  - fuse the separate quant/sort prologues (fused_mx_quant_moe_sort, dynamic_per_group_scaled_quant) into the stage-1 launch to remove an activation round-trip
  - fuse stage-1 and stage-2 (opus_moe_stage2_a8w4, rank 3) to keep the intermediate in L2

Unit = ONE launch (1 per layer, 61 launches/step, avg 57.16us) = MoE stage-1 (gate+up) grouped GEMM, fp8 activations x fp4 expert weights with fused SiLU*mul. Byte model: pairs = 64x6 = 384, experts_hit = E(1-(1-1/E)^pairs) = 243 of 384; per-expert local weight = 2 x (3072/TP8) x 7168 fp4 = 2.75 MB. This gives 669 MB in 57.16us = 11.7 TB/s = 146% of the 8 TB/s pin rate -- INFEASIBLE, so the model OVER-COUNTS (SKILL L3): headroom_class forced to unknown, suspect=true, stage-C counter-measurement candidate. What the infeasibility DOES prove: the kernel is squarely on the memory axis and is streaming as many expert bytes as HBM can deliver, and the true distinct-expert count must be <= ~166 (not the ~243 a uniform-routing prior predicts) -- consistent with group/hash-limited routing (noaux_tc, num_hash_layers=3). All-expert model would be 1057 MB (even more infeasible). Treat as memory-bound-and-near-saturated for routing: the lever is BYTE REDUCTION / routing, not another tuning pass -- but do not report a saturated verdict, the ratio is not a measurement. AI = 6.3 << ridge 625.

## Routing implication

1. **`main_kernel` (DSA sparse attention, tilelang, editable)** — 44.9% of e2e GPU time and **far from both roofs** (39% memory / 0.4% compute on the conservative byte model, ~5%/0.4% on the sparse one; the same ~5% signature shows up in prefill). Bound type **latency/occupancy**, so the lever is occupancy, dependency chains and fusion (and fewer KV bytes), not more MFMA tuning. It is the top target on BOTH rankings — no disagreement to resolve here.
2. **MoE stage-1 flydsl grouped GEMM** — squarely memory-axis and streaming expert weights at or beyond what HBM can deliver under a uniform-routing byte prior. No verdict is emitted (L3), but the routing is unambiguous: **byte-reduction / routing / config-tune track** (aiter tuned-config DB, flydsl tile bake-off, prologue fusion), not a from-scratch rewrite.
3. Everything else in the Top-N is < 4% e2e — ordinary Amdahl ordering applies; no roofline model was built for it.
