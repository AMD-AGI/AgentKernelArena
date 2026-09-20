# Kimi-K3 — head kernel `_fwd_grouped_kernel_stage1`

| Model | ISL | OSL | CONC | Docker / Image | Head Kernel | GPU Time Share (%) | Current Roofline | HL Run Directory |
|---|---|---|---|---|---|---|---|---|
| Kimi-K3 | 8192 | 1024 | 64 | harbor.crusoe.primus-safe.amd.com/hyperloom-image/sglang-rocm-k3:rocm720-mi35x-k3-20260727-tl312-08011830 | _fwd_grouped_kernel_stage1 | 13.61% | 18.40% / latency-bound | provenance://shared-nfs/hongtaom/qwen3_14B/hl_matrix_0824/exp/kimi-k3/Kimi-K3/20260829T041510Z-96caeb1a |

Device symbol as the profiler saw it: `_fwd_grouped_kernel_stage1` (op_class `attn`, Triton).
Unlike the DeepSeek row, the emitted symbol is already readable, so the row is named for it
verbatim. What it does: MLA **absorbed decode** stage 1 — one launch per full-attn layer,
24 of 93 layers.

Precision note: this run is **mxfp4** on routed-expert Linear, but this kernel is **bf16** —
the KV latent it streams is not quantized.

## Roofline detail (verbatim from GEAK's roofline skill, stage A)

Source: `provenance://shared-nfs/hongtaom/qwen3_14B/hl_matrix_0824/exp/kimi-k3/Kimi-K3/20260829T041510Z-96caeb1a/geak/e2e_cycle0/profile/round_0/profile_roofline.json`
(copied here as `profile_roofline.json` / `.md`). **Not recomputed.**

| field | value |
|---|---|
| `name` | _fwd_grouped_kernel_stage1 |
| `regime` | decode |
| `op_class` | attn |
| `editable` | True |
| `pct_gpu_time` | 13.61 |
| `t_ms` | 0.435845 |
| `launches_per_step` | 24.0 |
| `bytes_est` | 641728512 |
| `flops_est` | 14545846272 |
| `achieved_bw_bytes_s` | 1472377822391 |
| `achieved_flops` | 33373897307529 |
| `hbm_util` | 0.184 |
| `compute_util` | 0.0133 |
| `arithmetic_intensity` | 22.67 |
| `ridge_point` | 312.5 |
| `bound_type` | latency |
| `roofline_pct` | 0.184 |
| `target_eff` | 0.5 |
| `attainable_speedup` | 2.717 |
| `expected_e2e_gain_pct` | 8.6 |
| `headroom_class` | underperforming |
| `confidence` | medium |
| `suspect` | False |
| `modeled` | True |

`head_threshold_pct` for this run = 5.0. This kernel ranks **#1 by both `ranking_by_pct` and
`ranking_by_expected_gain`** — the two rankings agree here, which is the opposite of the
GLM-5.3-Flash case where they disagree.

Peaks used as denominators (gfx950, from the skill's `peaks.md`):

```json
{
  "hbm_bw_bytes_s": 8000000000000.0,
  "flops": {
    "fp64": 78600000000000.0,
    "fp32": 157000000000000.0,
    "bf16": 2500000000000000.0,
    "fp16": 2500000000000000.0,
    "fp8": 5000000000000000.0,
    "fp4": 1e+16
  },
  "source": "table",
  "confidence": "high",
  "note": "bf16==fp16 equality check in peaks.md passes; the bf16 MFMA peak is still the least-validated axis (SKILL 8.2)."
}
```

## Byte-reduction levers

- fp8 KV-cache dtype (halves latent bytes; lossy -> accuracy gate)
- longer page/tile for better coalescing of the paged latent reads

## Skill notes

MLA absorbed decode: KV latent (kv_lora_rank 512 + qk_rope 64) bf16, **NOT TP-sharded** (each
rank reads the full latent KV). bs=64, seq_len taken as isl+osl/2=8704 (real error source: the
captured step's true mean context is unknown). 12 q-heads/rank. One launch per full-attn layer,
24 of 93 layers. Unit = ONE launch.

Two `degraded` entries exist in the same report and are *not* this kernel:
`allreduce_prototype_twoshot` (collective, no interconnect peak in peaks.md) and `_score_kernel`
(no operand shapes captured, stage-A unmodelable). The all-reduce block is 12.56% of GPU time
and remains unmodeled — see the run's `geak/e2e_cycle0/final_report.md` §6.

## What is in this directory

The GEAK kernel-task UT for this kernel, copied from
`.../20260829T041510Z-96caeb1a/geak/e2e_cycle0/kernels/_fwd_grouped_kernel_stage1_task`,
plus the accepted overlay and the validation evidence.

| file | role |
|---|---|
| `unittest.py` | the UT entry point — correctness (oracle + random parity) |
| `run_unittest.sh` | wrapper that sets the LOAD-BEARING `PYTHONPATH` (see below) |
| `meta.json` | kernel geometry, `target_callable`, workload cases, `pct_gpu_time` |
| `harness_lib.py` | timing / correctness primitives (`time_op`, `correct`, `_time_graph`, ...) |
| `workload.json` / `regime.json` | case + regime definitions |
| `reference_io.pt` | the frozen oracle: real captured production inputs + outputs (0.95 GiB) |
| `selection_validation.json` | which captured process/rank the oracle was selected from |
| `opbench_result.json` | op-level bake-off result (see caveat 3) |
| `baseline_overlay/` | the baseline leg's overlay — **the denominator**, see below |
| `baseline_ref/` | `decode_attention.py.orig` (tuned) + `decode_attention.pristine.py.orig` (stock) |
| `kernel_src/geak_mla_stage1.py` | the kernel GEAK authored |
| `accepted_overlay/` | the overlay that was actually ACCEPTED (`c1_triton/_cand_overlay`) |
| `profile_roofline.{json,md}` | the roofline evidence quoted above |
| `_evidence/` | accept gate, director validation, patch, and the team's transferable notes |

`meta.json`, `unittest.py` and `reference_io.pt` are IMMUTABLE — an optimization edits
`kernel_src/` (or a candidate overlay), never these. Their checksums are enforced by the
validator; both were re-verified after this copy:

| artifact | sha256 | matches `meta.json` / accept record |
|---|---|---|
| `reference_io.pt` | `017f8ede884504695283291955fa8750be3e5d2779dd67b3914f13ab7ae85b98` | yes |
| `unittest.py` | `9bc5d7f57f3c03f094db2f78d6d738966e16258d23f7b6754fa36205b7f14648` | yes |

## `PYTHONPATH=baseline_overlay` is LOAD-BEARING

`run_unittest.sh` sets it, and running `python3 unittest.py` bare will give the wrong answer.
It makes `sglang.kernels.ops.attention.decode_attention` resolve to the ACCEPTED
`[geak-tune r2]` overlay (**BLOCK_N=32, waves_per_eu=2**). Without it the denominator is the
stock **BLOCK_N=16** kernel and every speedup is inflated by the ~**1.73x** already banked in
the earlier tuning round.

This differs from the two bigmoe3 UTs in this parent directory, where `python3 unittest.py`
is run directly. Use `bash run_unittest.sh <gpu_id>` here, with a gpu id from the
optimization pool, and set `GEAK_ROOT` if the GEAK checkout is not at
`provenance://shared-nfs/hongtaom/qwen3_14B/hl_matrix_0824/deps/kimi-k3/GEAK` (only `gpu_lock.sh` is
needed from it).

## Already validated in-session (this copy has NOT been re-run)

From `_evidence/director_validation.json` — Director re-ran the candidate from a fresh copy of
`KERNEL_PATH_ORIG + final_patch.diff` on 2026-08-30:

| check | result |
|---|---|
| correctness | **pass** — `RESULT PASS (oracle=True random_parity=True); [oracle] seq0/seq1 correct=True, output_independence correct=True` |
| primary metric | `GEAK_AB_WEIGHTED_SPEEDUP` (cross-process paired A/B, 3 reps, per-case median) |
| director weighted / geomean | **1.2591** / 1.1601 (TechLead reported 1.2616 / 1.1591 — within 0.2%) |
| per-case | `decode_bs64_ctx8704` 0.335843 -> 0.266723 ms = **1.2591x** (1024 calls); `decode_bs1_ctx8704` 0.03228 -> 0.0302 ms = 1.0689x |
| spreads | base <=0.006, cand <=0.019 |
| gates | `unittest_RESULT_PASS` / `AB_RESULT_PASS` / `weighted>1.02` / `decode_no_regress_bs1>=0.95` / `spreads<=0.05` — all True |
| timing basis | `device_graph_replay` (CUDA-graph capture, event timers, L2 flush per sample, legs in SEPARATE processes) |
| validation_status | **accepted** |

And the e2e accept, from `_evidence/integrate_result_c1_triton.json`:

| field | value |
|---|---|
| `winner_kind` | **env** — no kernel source change in the accepted delta |
| `apply_env` | `GEAK_MLA_DECODE_BLOCK_N=128 GEAK_MLA_DECODE_WAVES_PER_EU=1` |
| `isolated_speedup` | 1.2486 |
| ref runs / median | 803.001 / 803.616 / 806.863 -> **803.616** tok/s |
| cand | **823.218** tok/s |
| `e2e_delta_pct` | **+2.4392%** |
| TPOT | ref 60.763 / 60.73 / 60.41 -> cand **58.767** ms |
| TTFT | ref median 19467.099 -> cand 19432.012 ms |
| engagement | 8/8 TP ranks logged `[geak] _decode_grouped_att_m_fwd -> tuned launch geometry BLOCK_N=128 waves_per_eu=1 num_warps=4 num_stages=1 (Lk=576, gfx950=True)`; all 3 ref legs show BLOCK_N=32 waves_per_eu=2 |
| work identity | 192/192 completed, 1,572,864 in / 196,608 out, 0 errors — identical to every ref leg |
| `output_parity` | pass (`parity_kind: accuracy`) |
| `gate` | **accepted** |

## Caveats

1. **The accepted delta is env-only, not a kernel rewrite.** `winner_kind: env`. `kernel_src/geak_mla_stage1.py`
   is the kernel GEAK authored during the campaign, but what shipped is two env vars retuning the
   launch geometry of the kernel already in `accepted_overlay/_patched/`. Do not read
   `kernel_src/` as "the accepted kernel".
2. **`final_patch.diff` alone is not sufficient to reproduce.** The Director flagged this as a
   PACKAGING GAP: the workspace `.gitignore` excluded `_cand_overlay/**`, so the patch does not
   rebind `sglang.kernels.ops.attention.decode_attention`. That gap is closed in this package by
   `accepted_overlay/`, which is the real `_cand_overlay` — apply it together with `apply_env`.
3. **`opbench_result.json` reports `measured=false`** with `available: false, correct: false, ms: null`.
   That is by design, not a failure: attention-backend comparison is a server-level flag
   (`--attention-backend`) and was delegated to the Config Tuner fast path, so no op-level bake-off
   timing was taken. Correctness came from the oracle leg, speed from `ab_bench.py`.
4. **Never quote `unittest.py`'s speedup columns.** Per `_evidence/COMMANDMENT.md`: they are
   MEANINGLESS under a candidate overlay. `unittest.py` is the correctness judge; all speed claims
   must come from `ab_bench.py` / the A/B receipts above.
5. **`max_rel_err` up to 2.19 vs the oracle still passes, by design.** `harness_lib.correct` uses
   `|out-ref| <= atol + tol*|ref|` with `atol = tol*RMS(ref)`, so the online-softmax accumulation
   order is free; the *split partition* is not. See `_evidence/codebase_context.md` §6.
6. **No `GEAK_TIMING_RECEIPT`.** This task's frozen harness does not emit one and the receipt
   contract (`oracle_freezer.md`) is absent from that SKILL_DIR install.
7. **No null-candidate control was run** (the discipline used for the two bigmoe3 UTs). Given the
   `PYTHONPATH` denominator subtlety in this task, running one is the single highest-value
   verification to add.

## Structural differences vs the two bigmoe3 UTs in this directory

This task predates their harness generation, so four files they have are genuinely absent here —
not omitted from the copy:

| file | DeepSeek / Qwen | Kimi-K3 |
|---|---|---|
| `cases.py` | present | **absent** — case definitions live inside `meta.json` / `workload.json` |
| `leg_runner.py` | present | **absent** |
| `overlay_setup.py` | present | **absent** — overlays are pre-materialized in `baseline_overlay/` and `accepted_overlay/` |
| `_baseline_random.pt` | present | **absent** — random-parity draws are generated in-process; only the frozen `reference_io.pt` oracle is on disk |
| `_aiter_cfg/` | DeepSeek only | absent — K3's tuning table lives in the live tree as `aiter/configs/model_configs/geak_k3tune_bf16_tuned_gemm.csv` |

## Provenance

- Session: `20260829T041510Z-96caeb1a`, state `global_converged`, rc=0
- Run baseline -> best: 762.66 -> 821.41 tok/s/GPU (+7.70%), action `geak_e2e`
- This kernel's contribution: +2.4392% e2e on the same config (803.616 -> 823.218)
- Team run that produced it: `geak/e2e_cycle0/kernels/_exp/team__fwd_grouped_kernel_stage1_task_20260830_161821_238598_4096/`
  (2 rounds x 3 engineers; r1_d2 1.1902 weighted, r2_d0 1.2398, director 1.2591)
- Packaged 2026-09-01 by copy from the session tree; checksums re-verified (see table above).
  Staged here because `provenance://shared-nfs/zihao/headkernel_ut_0831/` is not writable by uid 91201 —
  move or symlink this directory in as `Kimi-K3_fwd_grouped_kernel_stage1/`.
