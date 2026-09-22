# MiniMax-M3-MXFP4 — head kernel `_gqa_share_sparse_decode_kernel`

| Model | ISL | OSL | CONC | Docker / Image | Head Kernel | GPU Time Share (%) | Current Roofline | HL Run Directory |
|---|---|---|---|---|---|---|---|---|
| MiniMax-M3-MXFP4 | 8192 | 1024 | 64 | harbor.crusoe.primus-safe.amd.com/hyperloom-image/sglang:v0.5.17-rocm720-mi35x-profilerfix | _gqa_share_sparse_decode_kernel | 0.74% (5.76% serving-weighted) | 38.5% / latency-bound | /shared_nfs/chaox/Minimax_m3_mxfp4/test_results/Minimax_m3_MXFP4_20260828/MiniMax-M3-MXFP4/20260828T105325Z-4d49e44f |

Backend: **Triton** (sparse GQA decode attention in `topk_sparse.py`).

Device symbol: `_gqa_share_sparse_decode_kernel` (op_class `attn`, regime `decode`).

## GEAK optimization outcome

- **Isolated UT accepted** (director weighted speedup **1.65×**, correctness PASS).
- **E2E stack NOT accepted** — kernel is only ~0.74% of GPU time (Amdahl-limited).
- Optimized source: `kernel_src/topk_sparse.py` (from team run `team__gqa_share_sparse_decode_kernel_task_20260829_085117_1297003_28411`).
- Baseline reference: `baseline_ref/topk_sparse.py.orig`.

## Roofline detail (verbatim from GEAK stage A)

Source: `geak/e2e_cycle0/profile/round_0/profile_roofline.json` (copied as `profile_roofline.json` / `.md`). **Not recomputed.**

| field | value |
|---|---|
| `name` | _gqa_share_sparse_decode_kernel |
| `regime` | decode |
| `op_class` | attn |
| `editable` | True |
| `pct_gpu_time` | 0.74 |
| `serving_weighted_pct` | 5.76 |
| `t_ms` | 0.0218 |
| `launches_per_step` | 64 |
| `bound_type` | latency |
| `roofline_pct` | 0.385 (38.5%) |
| `compute_util` | 0.01 |
| `hbm_util` | 0.385 |
| `attainable_speedup` | 1.3 |
| `expected_e2e_gain_pct` | 1.33 |
| `headroom_class` | moderate |

## What is in this directory

GEAK kernel-task UT copied from
`/shared_nfs/chaox/Minimax_m3_mxfp4/test_results/Minimax_m3_MXFP4_20260828/MiniMax-M3-MXFP4/20260828T105325Z-4d49e44f/geak/e2e_cycle0/kernels/_gqa_share_sparse_decode_kernel_task`.

| file | role |
|---|---|
| `unittest.py` | UT entry point — correctness + timing |
| `cases.py` | IMMUTABLE case definitions |
| `meta.json` | kernel geometry, target_callable, workload cases |
| `harness_lib.py` | timing / correctness primitives |
| `reference_io.pt` | frozen oracle |
| `_baseline_random.pt` | random parity draw seeds |
| `baseline_overlay/`, `_cand_overlay/` | baseline / candidate overlays |
| `baseline_ref/` | stock kernel snapshot |
| `kernel_src/` | **GEAK-optimized** `topk_sparse.py` |
| `leg_runner.py`, `overlay_setup.py` | harness plumbing |
| `regime.json`, `workload.json`, `selection_validation.json` | capture metadata |
| `_capture_meta.json` | capture telemetry |
| `profile_roofline.{json,md}` | roofline evidence |

## How to run

```bash
cd /shared_nfs/zihao/headkernel_ut_0831/MiniMax-M3-MXFP4_gqa_share_sparse_decode_kernel
python3 unittest.py
```

Run inside the sglang image above on MI355X (gfx950).
