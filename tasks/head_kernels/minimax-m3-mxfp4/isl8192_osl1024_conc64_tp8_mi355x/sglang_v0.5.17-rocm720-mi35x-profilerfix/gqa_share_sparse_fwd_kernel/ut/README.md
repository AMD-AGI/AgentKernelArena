# MiniMax-M3-MXFP4 — head kernel `_gqa_share_sparse_fwd_kernel`

| Model | ISL | OSL | CONC | Docker / Image | Head Kernel | GPU Time Share (%) | Current Roofline | HL Run Directory |
|---|---|---|---|---|---|---|---|---|
| MiniMax-M3-MXFP4 | 8192 | 1024 | 64 | harbor.crusoe.primus-safe.amd.com/hyperloom-image/sglang:v0.5.17-rocm720-mi35x-profilerfix | _gqa_share_sparse_fwd_kernel | 21.79% | 4.2% / latency-bound | provenance://shared-nfs/chaox/Minimax_m3_mxfp4/test_results/Minimax_m3_MXFP4_20260828/MiniMax-M3-MXFP4/20260828T105325Z-4d49e44f |

Backend: **Triton** (sparse GQA prefill attention in `topk_sparse.py`).

Device symbol: `_gqa_share_sparse_fwd_kernel` (op_class `attn`, regime `prefill`).

## GEAK optimization outcome

- **Accepted** into the final stack (e2e throughput +8.72%, isolated UT weighted speedup ~7.2×).
- Optimized source: `kernel_src/topk_sparse.py` (from team run `team__gqa_share_sparse_fwd_kernel_task_20260829_005746_1057070_12140`).
- Baseline reference: `baseline_ref/topk_sparse.py.orig`.

## Roofline detail (verbatim from GEAK stage A)

Source: `geak/e2e_cycle0/profile/round_0/profile_roofline.json` (copied as `profile_roofline.json` / `.md`). **Not recomputed.**

| field | value |
|---|---|
| `name` | _gqa_share_sparse_fwd_kernel |
| `regime` | prefill |
| `op_class` | attn |
| `editable` | True |
| `pct_gpu_time` | 21.79 |
| `t_ms` | 1.3078 |
| `launches_per_step` | 60 |
| `bound_type` | latency |
| `roofline_pct` | 0.042 (4.2%) |
| `compute_util` | 0.042 |
| `hbm_util` | 0.026 |
| `attainable_speedup` | 11.89 |
| `expected_e2e_gain_pct` | 6.81 |
| `headroom_class` | underperforming |

## What is in this directory

GEAK kernel-task UT copied from
`provenance://shared-nfs/chaox/Minimax_m3_mxfp4/test_results/Minimax_m3_MXFP4_20260828/MiniMax-M3-MXFP4/20260828T105325Z-4d49e44f/geak/e2e_cycle0/kernels/_gqa_share_sparse_fwd_kernel_task`.

| file | role |
|---|---|
| `unittest.py` | UT entry point — correctness + timing |
| `cases.py` | IMMUTABLE case definitions |
| `meta.json` | kernel geometry, target_callable, workload cases |
| `harness_lib.py` | timing / correctness primitives |
| `reference_io.pt` | frozen oracle: captured production inputs + outputs |
| `_baseline_random.pt` | random parity draw seeds |
| `baseline_overlay/` | baseline leg overlay |
| `_cand_overlay/` | candidate overlay used during optimization |
| `baseline_ref/` | stock kernel snapshot (`topk_sparse.py.orig`) |
| `kernel_src/` | **GEAK-optimized** `topk_sparse.py` |
| `leg_runner.py`, `overlay_setup.py` | harness plumbing |
| `regime.json`, `workload.json`, `selection_validation.json` | capture metadata |
| `_capture_meta.json` | capture telemetry |
| `profile_roofline.{json,md}` | roofline evidence quoted above |

`cases.py`, `meta.json`, `unittest.py` and `reference_io.pt` are IMMUTABLE — optimizations edit `kernel_src/` only.

## How to run

```bash
cd provenance://shared-nfs/zihao/headkernel_ut_0831/MiniMax-M3-MXFP4_gqa_share_sparse_fwd_kernel
python3 unittest.py
```

Run inside the sglang image above on MI355X (gfx950).
