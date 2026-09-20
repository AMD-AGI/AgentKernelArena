# GLM-5.3-Flash — `fused_moe_kernel` head-kernel unit test

Head kernel by GPU share for GLM-5.3-Flash: **23.32% of GPU time**, 3.7x the next entry.

**Read this first: this UT was never used for an optimization.** The harness is complete and its
smoke test passes, but zero candidates were ever authored against it. There is no A/B result, no
accepted patch, and no e2e validation in this package — unlike the Kimi-K3 package in this
directory. What you are getting is a fully captured, ready-to-run harness with a **null candidate**
already installed. See "Status" below for the evidence.

## The op

| | |
|---|---|
| device kernel | `fused_moe_kernel` (Triton) |
| target callable | `sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe:fused_experts_impl` |
| source in image | `provenance://runtime-image/sglang/python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe.py` |
| editable | yes (`editable: true` in `meta.json`) |
| quant | fp8_e4m3 weights + fp8 act, blockscale `[128, 128]` |
| geometry | E=288, top_k=8, hidden=4096, intermediate_per_tp=256 (2048 / TP8) |
| shapes | A `[M, 4096]`, B `[512, 4096]` |
| unit | one logical MoE layer = 2 launches (w1 gate/up + w2), 84 launches/step |

Serving contract: sglang 0.5.18, MI355X / gfx950, TP=8, ISL 8192 / OSL 1024 / CONC 64, enforce-eager.

## Roofline (verbatim from `profile_roofline.md`, cycle1 round 0, stage A)

| %gpu | t/unit | bytes | achieved BW | hbm_util | comp_util | bound | roofline% | target | headroom | attainable | exp e2e gain |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 23.32 | 96.3 us | 754.9 MB | 7.84 TB/s | 0.980 | 0.0067 | memory | 98.0% | 0.90 | **saturated** | **1.00x** | **0.00%** |

**The two rankings disagree, and that disagreement is the point.** By %gpu this kernel dominates;
by recoverable headroom it is **last** of the five heads — it already runs at 98.0% of the 8 TB/s
memory roof, so there is nothing for per-kernel tuning to recover. The roofline's own conclusion:

> byte-reduction levers (saturated => tuning has nothing left; move fewer bytes)

The levers it lists, verbatim:

- fuse the separate fp8 activation-quant epilogue (`_per_token_group_quant_8bit`, 2.04%gpu, 8820
  launches) into the MoE kernel to remove an activation round-trip
- stop streaming unrouted experts if the kernel reads all 288 — **already ruled out**: the
  all-288-expert byte model gives 906 MB = 118% of peak (infeasible), so the kernel does skip them.
  The feasible model is experts_hit = 239.5/288 -> 755 MB/layer
- fp8 -> fp4 (mxfp4) expert weights: halves the dominant weight stream. LOSSY — must pass the
  accuracy gate
- raise arithmetic intensity by batching more tokens per expert visit — **NOT available**: conc/isl/osl
  are fixed by the measurement contract

For reference, the other four heads and their expected e2e gain: CK a8w8 blockscale bpreshuffle
(5.65%, +3.95%), `Cijk_..._MT32x32x512` (5.55%, +3.27%), `main_kernel` TileLang DSA attn (5.22%,
+2.82%), `Cijk_..._MT16x16x1024` (6.31%, +2.51%). The three GEMM families are non-editable
(Tensile solutions / precompiled CK instances); `main_kernel` is the only other editable head.

## Contents

Canonical file set, matching the DeepSeek-V4-Pro and MiniMax-M3 packages in this directory:

```
unittest.py              3,005     IMMUTABLE driver
harness_lib.py          69,688     IMMUTABLE timing + correctness primitives
cases.py                 8,207     IMMUTABLE oracle/random case construction
leg_runner.py            5,674     IMMUTABLE per-leg subprocess runner
meta.json               16,947     geometry, target_callable, cases, tol, oracle sha256
regime.json                534     enforce-eager / fp8_blockscale / bf16 kv
workload.json            4,037     timing cases + serving weights
overlay_setup.py        16,650     overlay installer
reference_io.pt  1,041,269,475     frozen oracle
_baseline_random.pt 203,395,745    random-parity baseline outputs
selection_validation.json 22,023   capture-target audit (ok: true)
capture_meta.json      123,283     capture provenance
profile_roofline.json/.md          stage-A roofline (source of the table above)

baseline_overlay/                  empty manifest — the unmodified tree IS the baseline
baseline_ref/                      fused_moe.py.orig
kernel_src/                        fused_moe.py            <- edit this
_cand_overlay/                     candidate overlay       <- currently a NULL candidate
_provenance/                       build_oracle.py, run_capture.sh, sglang_bootstrap.py,
                                   write_meta.py, selection_validation_full_probe.json
```

Total 1.2 GB, 34 files. Not included: `reference_io_raw.pt` (2.85 GB pre-oracle raw capture, not
part of the package spec) and the 24 `selection_trace.*.json` debug dumps.

## The candidate overlay is a provable no-op

`unittest.py` times `baseline_overlay` against `_cand_overlay` in interleaved fresh subprocesses.
All three copies of the kernel source are byte-identical:

```
e1bf07f3ed1aae13a595ac8f41291438  kernel_src/fused_moe.py
e1bf07f3ed1aae13a595ac8f41291438  baseline_ref/fused_moe.py.orig
e1bf07f3ed1aae13a595ac8f41291438  _cand_overlay/_patched/sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe.py
```

`_cand_overlay/_overlay_manifest.json` does bind the module, so the overlay machinery is exercised —
it just injects the original file. **Running the package as shipped measures the harness's own noise
floor**, which is the right first thing to do before trusting any candidate's number. Write your
candidate into `kernel_src/fused_moe.py` and re-run `overlay_setup.py` to install it.

## Running it

Inside `harbor.crusoe.primus-safe.amd.com/hyperloom-image/sglang:v0.5.18-rocm720-mi35x-profilerfix`,
on one GPU from the optimization pool (never the serving set):

```bash
python3 unittest.py
```

Exit codes, from the driver's own docstring: `0` pass, `1` correctness FAIL, `2` environment,
`3` regenerate UT (harness incomplete). Correctness is frozen oracle + random-value parity against
the live baseline leg, `tol = 0.02`, `random_draws = 3`. The metric printed as
`GEAK_WEIGHTED_SPEEDUP` is `serving_weighted_speedup` = measured baseline ms x analytic serving calls.

No `PYTHONPATH` wrapper is needed (unlike the Kimi-K3 package): `baseline_overlay` carries an empty
manifest, so the unmodified installed tree is the baseline.

## Status: never optimized — the evidence

| record | value |
|---|---|
| `kernel_journey.json` -> `kernels[]` | `[]` (empty, both cycles) |
| `hot_kernels[].selected_for_optimization` | `false` — for all 41 (cycle0) and 25 (cycle1) entries |
| `_capture/bench_runs.jsonl` | 0 bytes |
| `_capture/bench_summary.json` | `"runs": 0`, all metrics null |
| `geak/result.json` | `status: "no_gain"`, `accepted_kernels: []`, `accepted_heads: []` |
| `meta.json` -> `smoke` | `"PASS"` — the harness itself is sound |
| `selection_validation.json` -> `ok` | `true` — the capture target was validated |

So the stop was **not** a technical failure. The oracle was built (14:43), the random baseline was
built (15:07), the overlay was scaffolded, and then the task was left. The kernel lane's wall clock
went elsewhere: `fused_moe` got 1h46m, the fp8 bpreshuffle GEMM 3h51m, the bf16 a16w16 GEMM 8h09m
(which had already reported `isolated_speedup: 1.0` / "nothing to deploy" on its first opbench), and
the only author lane that ran was a 7.13% elementwise-copy cluster that is not a head kernel at all.

The cycle1 Architect had planned the opposite order and said so explicitly:

> **h0** — the 23.32% head and the long pole (an author lane). Start it early enough to finish
> inside the budget; do not let the roofline's `saturated` verdict defer it.

## Known gaps — read before trusting a number

**1. The dominant timing bucket has no frozen oracle case.**

| | cases |
|---|---|
| timing (`workload.json`) | `decode_M64` (weight 0.70), `prefill_M8192` (0.30), `decode_M1` (0.00) |
| oracle (`meta.json`) | `oracle_m19_decode`, `oracle_m1_decode`, `oracle_m8192_prefill` |

`decode_M64` carries 70% of the weighted metric, but the frozen oracle covers M=19 for decode, not
M=64. Correctness at the bucket that dominates the score therefore rests on the 3 random-parity
draws, not on the oracle. M=19 and M=64 are both real dispatched values (the server only ever
dispatches M in {1, 19, 64, 8192, 16384}), so this is a coverage gap, not a wrong shape.

**2. Two different GPU-share numbers are recorded, and they disagree by 6x.**

```
meta.json:  "pct_gpu_time": 3.97
            "pct_gpu_time_tracelens_prior": 23.32
```

23.32 is the TraceLens figure for the device kernel and is what the roofline, the profile top-N and
the head-kernel table all use. 3.97 is what this task measured live for the callable during capture
(`workload.json` repeats it as `live_pct_gpu`). The package ships both; it does not resolve which
scope is right. Anyone sizing an Amdahl ceiling from this UT should decide deliberately which one
they mean.

**3. The regime is launch-bound, so isolated wins may not transfer.**

The server runs `--disable-cuda-graph` with 48.3% device idle and 1433 kernels/step at 6.75 us mean.
Both GEMM heads in this run produced proven isolated speedups (1.564x and 1.4451x) that converted to
-0.13% and -0.438% e2e. A byte-reduction win on this MoE kernel is a better bet than a latency win
precisely because the kernel is memory-saturated rather than launch-starved — but the e2e ceiling is
still set by the launch bubble and by TP=8 collective peer-wait (8870.70 ms of 10691.82 ms raw
kernel time is de-inflated spin in `cross_device_reduce_2stage`).

**4. A tuned MoE config IS already loaded — do not re-derive from "zero coverage".**

The cycle1 `strategy.md` claims no tuned MoE config ships for this shape. The cycle0 Tuning
Specialist checked the baseline server log and found that claim **factually wrong on this box**: all
8 ranks log a fallback that successfully loads a tuned config from the triton 3.7.1 directory
(N=256, not N=512), plus "Down MoE config file not found ...; reusing the tuned up-projection config
without TMA". Any plan premised on zero coverage needs re-deriving.

**5. Stage A confidence.** The roofline row is stage A — shapes from the trace/config, no Extractor
capture, no rocprofv3 counters. Display and annotate only; `pct_gpu_time` remains the primary key.

## Provenance

| | |
|---|---|
| HL run | `provenance://shared-nfs/hongtaom/qwen3_14B/hl_matrix_0824/exp/glm53-flash/GLM-5.3-Flash/20260829T045924Z-d82dabe1` |
| task | `geak/e2e_cycle1/kernels/fused_moe_kernel_task` |
| roofline | `geak/e2e_cycle1/profile/round_0/profile_roofline.{json,md}` |
| image | `harbor.crusoe.primus-safe.amd.com/hyperloom-image/sglang:v0.5.18-rocm720-mi35x-profilerfix` |
| stack | torch 2.9.1+rocm7.2.0, sglang 0.5.18, hip 7.2.26015, gfx950 x8 |
| baseline | 716.237 tok/s (GEAK hot), TTFT 6011.206 ms, TPOT 83.599 ms |
| server flags | `--trust-remote-code --mem-fraction-static=0.8 --disable-radix-cache --disable-cuda-graph --dsa-prefill-backend tilelang --dsa-decode-backend tilelang --kv-cache-dtype bfloat16 --disable-shared-experts-fusion --reasoning-parser deepseek-r1 --context-length 11264 --watchdog-timeout 1800 --moe-runner-backend triton --stream-interval 10` |

Oracle integrity, verified at pack time and self-consistent with `meta.json`'s recorded hash:

```
sha256  59040b163efbc9dbd3592a28dee01bcc2801bb669de170197524dfe335c5feee  reference_io.pt
```

All eight immutable files (`unittest.py`, `harness_lib.py`, `cases.py`, `leg_runner.py`, `meta.json`,
`regime.json`, `workload.json`, `overlay_setup.py`) are md5-identical to the source task tree.
