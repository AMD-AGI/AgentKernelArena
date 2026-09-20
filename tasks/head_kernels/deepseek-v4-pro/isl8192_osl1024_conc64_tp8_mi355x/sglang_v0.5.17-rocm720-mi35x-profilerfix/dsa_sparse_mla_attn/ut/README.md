> Historical capture documentation. The active generated-input draft uses `generated_cases.json` and `generated_contract.py`, with fresh separate frozen-source references. No external tensor archive is required; see the task README for current commands.

# DeepSeek-V4-Pro — head kernel `dsa_sparse_mla_attn`

| Model | ISL | OSL | CONC | Docker / Image | Head Kernel | GPU Time Share (%) | Current Roofline | HL Run Directory |
|---|---|---|---|---|---|---|---|---|
| DeepSeek-V4-Pro | 8192 | 1024 | 64 | harbor.crusoe.primus-safe.amd.com/hyperloom-image/sglang:v0.5.17-rocm720-mi35x-profilerfix | dsa_sparse_mla_attn | 44.88% | 38.79% / latency-bound | provenance://shared-nfs/zihao/arena/exp/DeepSeek-V4-Pro-bigmoe3/DeepSeek-V4-Pro/20260830T005045Z-433821d2 |

Device symbol as the profiler saw it: `main_kernel` (op_class `attn`). The row is named
for what the kernel does, because the emitted symbol does not say.

## Roofline detail (verbatim from GEAK's roofline skill, stage A)

Source: `provenance://shared-nfs/zihao/arena/exp/DeepSeek-V4-Pro-bigmoe3/DeepSeek-V4-Pro/20260830T005045Z-433821d2/geak/e2e_cycle0/profile/round_0/profile_roofline.json`
(copied here as `profile_roofline.json` / `.md`). **Not recomputed.**

| field | value |
|---|---|
| `name` | main_kernel |
| `regime` | decode / full-sparse-MLA layers |
| `op_class` | attn |
| `editable` | True |
| `pct_gpu_time` | 44.876 |
| `pct_gpu_time_decode` | 26.679 |
| `t_ms` | 0.2068 |
| `launches_per_step` | 30 |
| `bytes_est` | 641728512 |
| `flops_est` | 2281701376 |
| `achieved_bw_bytes_s` | 3103135938104.4487 |
| `achieved_flops` | 11033372224371.373 |
| `hbm_util` | 0.3879 |
| `compute_util` | 0.0044 |
| `arithmetic_intensity` | 3.56 |
| `ridge_point` | 312.5 |
| `bound_type` | latency |
| `roofline_pct_raw` | 0.3879 |
| `roofline_pct` | 0.3879 |
| `target_eff` | 0.5 |
| `attainable_speedup` | 1.289 |
| `expected_e2e_gain_pct` | 10.06 |
| `headroom_class` | moderate |
| `confidence` | low |
| `suspect` | False |
| `modeled` | True |

Peaks used as denominators (gfx950, from the skill's `peaks.md`):

```json
{
  "hbm_bw_bytes_s": 8000000000000.0,
  "flops": {
    "bf16": 2500000000000000.0,
    "fp8": 5000000000000000.0,
    "fp4": 1e+16
  },
  "source": "table",
  "confidence": "high"
}
```

## Byte-reduction levers

- fp8 / lower-precision KV latent (lossy - accuracy gate)
- fuse the DSA indexer top-k pass into the attention launch (mhc_pre_big_fuse / topk_transform are separate Top-N rows)
- page/layout the latent KV so the top-1024 gather is contiguous

## Skill notes

Unit = ONE launch of the full-sparse-MLA decode layer (30 of 61 layers, 30 launches/step, avg 206.8us). Two candidate byte models; per SKILL sec.4 the LARGEST FEASIBLE one is used: (A) FULL-CONTEXT read -- batch64 x ctx(isl+osl/2=8704) x (kv_lora 512 + rope 64) x bf16 = 642 MB -> 3.10 TB/s = 39% of the HBM roof [USED: feasible, and the pessimistic/conservative headroom]. (B) TRUE-SPARSE read -- only the index_topk=1024 selected tokens per query = 75 MB -> 0.37 TB/s = 4.6% of the roof, which would imply ~11.0x attainable instead of the conservative value reported. The MLA latent KV is replicated on every TP rank (num_key_value_heads=1), so no /TP on bytes; FLOPs use 16 local q-heads (128/TP8) over the absorbed 576-dim latent. THE TWO MODELS DISAGREE BY 8.5x -- this entry is the #1 priority for a stage-B/C re-run (real captured shapes / rocprofv3 FETCH_SIZE) because it decides whether the run's dominant kernel has ~1.3x or ~10x of headroom. Corroborating evidence for the low end being wrong: the PREFILL regime of the same kernel (24.45 ms per full-sparse layer for an 8192-token chunk) works out to ~11 TFLOP/s and <0.5 TB/s -- ~0.4% of the compute roof and ~5% of the memory roof, i.e. far from BOTH roofs in the phase where attention should be compute-dense; that is a latency/occupancy signature, not a saturated kernel. Stage A: confidence LOW, advisory only -- do NOT rank on attainable_speedup.

## What is in this directory

The GEAK kernel-task UT for this kernel, copied from
`provenance://shared-nfs/zihao/arena/exp/DeepSeek-V4-Pro-bigmoe3/DeepSeek-V4-Pro/20260830T005045Z-433821d2/geak/e2e_cycle0/kernels/main_kernel_task`.

| file | role |
|---|---|
| `unittest.py` | the UT entry point — correctness + timing for this kernel |
| `cases.py` | IMMUTABLE case definitions; `call(args)` is the seam both legs go through |
| `meta.json` | kernel geometry, `target_callable`, workload cases, `pct_gpu_time` |
| `harness_lib.py` | timing / correctness primitives (`time_op`, `correct`, ...) |
| `reference_io.pt` | the frozen oracle: real captured production inputs + outputs |
| `baseline_overlay/` | the baseline leg's overlay |
| `kernel_src/` | the kernel source GEAK was editing |
| `profile_roofline.{json,md}` | the roofline evidence quoted above |

`cases.py`, `meta.json`, `unittest.py` and `reference_io.pt` are IMMUTABLE — an
optimization edits `kernel_src/` (or a candidate overlay), never these.

## Verified runnable

Run `python3 unittest.py` from inside this directory, in the image above.

| check | result |
|---|---|
| when / where | 2026-08-31, node crsuse2-m2m-115 (jid 74733), container ds_probe, image sglang:v0.5.17-rocm720-mi35x-profilerfix, 1x MI355X (gfx950) |
| smoke (`_verify/ut_smoke.py`) | PASS (4/4: dir self-contained; target resolves to provenance://runtime-image/sglang/python/sglang/kernels/ops/attention/dsa/tilelang_kernel.py, i.e. OUTSIDE the task dir; reference_io.pt rehydrates and prefill_m8192_x64 runs in 4.78 ms; both legs build and provably differ) |
| full `unittest.py` | python3 unittest.py -> exit 0, CORRECTNESS: PASS, GEAK_WEIGHTED_SPEEDUP 1.0807 (geomean 1.1841): prefill_m8192_x64 4.563->2.820 ms (1.618x), decode_m64_x1024 0.2486->0.2037 ms (1.221x), prefill_m8192_x1024 0.999x, decode_m64_x128 0.996x |
| null-candidate control | kernel_src/ replaced by baseline_ref/tilelang_kernel.py.orig -> GEAK_WEIGHTED_SPEEDUP 0.9998, every bucket within 0.2%. So the 1.0807 above is a real candidate gain, not a warm-cache / leg-ordering artifact. |

Caveats:

- kernel_src/tilelang_kernel.py is NOT stock: it carries the tuned-launch-geometry patch (_DSA_TUNED_GEOMETRY) GEAK wrote at 00:53 on 2026-08-31, before the node was lost. The stock file is baseline_ref/tilelang_kernel.py.orig, which is byte-identical to the live stack. Start from baseline_ref if you want a clean slate.
- decode_m1_x1024 reports max_rel_err 0.11819 and still passes. That is by design: harness_lib.correct uses |out-ref| <= atol + tol*|ref| with atol = tol*RMS(ref), so a small-magnitude element of a spiky single-token output can carry a large RELATIVE error while sitting under the absolute floor. Every other case is exactly 0.0.
