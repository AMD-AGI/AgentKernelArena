# GLM-5.3-Flash — `elementwise_kernel_manual_unroll` copy-cluster unit test

Profile rows **#8 (3.24%)** and **#10 (2.85%)**; with row #24 the cluster is **7.13% of effective
GPU time** aggregate. TraceLens resolves both to `aten::copy_` from
`sglang/srt/layers/quantization/fp8_utils.py:112` and
`sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla_rocm.py:176`.

**This is not a head kernel** (below the 5% `head_threshold_pct`) — it is the one *kernel-track*
target that actually got an author lane in this session, and it is the only UT here with a real,
measured, non-null candidate. **Weighted speedup 1.4253x / 1.3691x over two runs.**

It is also the scale-materialization copy that *feeds* the fp8 bpreshuffle GEMM head, so if that
head ever changes, this cluster may shrink or move.

## The op

| | |
|---|---|
| device kernel | `at::native::elementwise_kernel_manual_unroll<128, 4, ... direct_copy_kernel_cuda ...>` |
| target callable | `sglang.srt.layers.quantization.fp8_utils:materialize_bpreshuffle_fp8_scale` |
| source in image | `provenance://runtime-image/sglang/python/sglang/srt/layers/quantization/fp8_utils.py` |
| editable | **yes** — `kernel_src/bpreshuffle_scale_impl.py` is the only writable path |
| bind | `kind: rebind` onto the live seam (`meta.candidate_bind`) |
| dtype | fp32 in / fp32 out |
| tol | **1e-6** — this op is bit-exact, not approximate |
| oracle | `reference_io.pt`, captured from the live TP=8 baseline, `oracle_complete: true` |
| `selection_validation.ok` | **true** |

### Math contract — it is a *layout* contract, not a value contract

> `out == scale` (fp32 `[M,G]`) but with **transposed-contiguous storage**:
> `out.stride() == (1, M)`, i.e. `out.t().is_contiguous()`. Non-2-D input passes through unchanged.

The per-1×128 fp8 activation scale leaves the producing quant kernel row-major `[M,G]`
(`stride == (G,1)`); the gfx950 CK `gemm_a8w8_blockscale_bpreshuffle` GEMM reads those bytes
column-major. The baseline body is literally `scale.t().contiguous().t()` — that is the copy the
profiler charges ~26k launches of `direct_copy_kernel_cuda` to.

`cases.call` **asserts the stride contract**. A value-correct but row-major return FAILS the
unittest rather than silently corrupting the GEMM end-to-end. `harness_lib.assert_independent_outputs`
also does a `data_ptr` check, so a cached persistent output buffer is rejected as cheating (it
measures 6.1–6.6 us and is illegal).

### Shapes

Captured live, 341 405 calls total, 7 cases:

```
T(64, 2)  fp32  164 061     T(64, 32) fp32  109 378     T(64, 12) fp32   33 209
T(64, 16) fp32   21 485     T(16384, 2)      5 208      T(16384, 32) …   (prefill tail)
```
Decode `[64, {2,12,16,32}]` · prefill `[16384, {2,12,16,32}]` · analytic calls 1024 decode / 64 prefill.

## What the author lane found — the whole point of this package

`_candidate_best/round_1_metrics.json` (rocprofv3, round 1 reprofile):

| | |
|---|---|
| **weighted speedup** | **1.4253x** (run 1) · **1.3691x** (run 2) |
| dominant-bucket speedup | 1.429 · 1.357 |
| host speedup measured | 1.68x |
| the op, before | `elementwise_kernel_manual_unroll<128,4,direct_copy_kernel_cuda>` @ **3.25 us** avg |
| the op, after | `bpsh_t_copy_v4(float const*, float*, int, int, int)` @ **2.207 us** avg |

**The bottleneck is `overhead`, not bandwidth, and the evidence is unusually clean:**

```
device time is FLAT across a 4000x range in bytes (512 B .. 2 MB):
  m64x2 9.7us   m64x12 10.1us   m64x16 9.9us   m64x32 8.7us
  m16384x2 9.6us  m16384x12 9.4us  m16384x16 8.4us  m16384x32 8.2us
  empty_event_floor_no_kernel = 4.4us
-> zero memory-bandwidth signal; the whole number is event floor + host dispatch latency
   visible as a GPU-timeline gap + ~2us of kernel.
```

Supporting counters: `hip_api_pct_of_profiled_time` **98.99%** vs `kernel_dispatch_pct` **1.01%**;
`hbm_pct_of_peak` **0.09**; `grid_size` 512, `ctas` 2 of 118 CU → `gpu_fill_pct` **1.7%**;
`wall_over_device` **2.51**; `kernarg_fixed_tax_us` 1.6.

So the lever is **the length of the host path**, and every device-side idea (tiling, warp count,
float4, BM×BG×warps sweep) is worth <0.2 us. Three independent round-1 arms converged on that same
conclusion and were hand-merged into the winning candidate:

| arm | approach | speedup |
|---|---|---|
| `r1_d0` host_runtime | allocate the final layout directly with `empty_strided((M,G),(1,M))` — alloc + one copy, never `.t().contiguous().t()`; resolve every hot callable once at import | 1.169x |
| `r1_d1` algorithm | ONE pybind entry point in HIP: `at::empty_strided` + ONE `hipLaunchKernelGGL` on the current stream, pybind function object bound **directly** as the seam — zero Python frames, no dispatcher hop, no `torch.library` registration. Includes an `M % 4 == 0` float4-store fast path | 1.355x |
| `r1_d2` compute | proved the lever is "C-level launch + fused allocation", **not** the device language — its Triton-hsaco arm and the HIP arm are within noise | 1.379x |

Remaining after the merge: `additive_host_floor_us` 4.10, `remaining_addressable_us` **0.45**
(5.7% of scored ms). There is almost nothing left in this op.

Probe results worth keeping (`_candidate_best/analysis.json`):

```
torch.empty_strided((M,G),(1,M)).copy_(x)      7.2 - 7.9 us   (1.10x - 1.31x)
cached persistent out buffer + copy_           6.1 - 6.6 us   ** ILLEGAL — data_ptr check rejects it **
triton.jit transpose-copy via kernel[grid](…)  16.7 - 17.0 us (0.5x — the Python launcher adds ~+9us)
```

**e2e: not measured.** The lane ended at round 2 when the session's wall clock expired. Your table's
"未测" for this row is correct.

## ⚠ What ships in `kernel_src/` is the NULL candidate

```
kernel_src/bpreshuffle_scale_impl.py   1 623 B  — the BASELINE body (scale.t().contiguous().t())
_candidate_best/bpreshuffle_scale_impl.py  7 761 B  — the 1.4253x HIP merge
```

Running the package as shipped measures the **harness's own noise floor**, which is the right first
thing to do. To score the real candidate:

```bash
cp _candidate_best/bpreshuffle_scale_impl.py kernel_src/
python3 overlay_setup.py        # reinstall the candidate overlay
python3 unittest.py
```

The winning candidate compiles a HIP extension once into `$TORCH_EXTENSIONS_DIR` under an
md5-of-source name, flock-guarded, then `dlopen`s the prebuilt `.so` directly via `importlib` (no
ninja, no `cpp_extension` baton) — because every timing bucket is a fresh subprocess.

## Contents

```
unittest.py              IMMUTABLE driver (h.measure_legs: baseline_overlay vs _cand_overlay,
                         interleaved fresh subprocesses)
harness_lib.py           IMMUTABLE timing + correctness primitives
cases.py                 IMMUTABLE oracle/random case construction + the stride-contract assert
leg_runner.py            IMMUTABLE per-leg subprocess runner
overlay_setup.py         overlay installer
meta.json                7 cases, capture_shape_counts, candidate_bind, math_contract
workload.json            timing cases + serving weights
regime.json              enforce-eager / fp8_blockscale / bf16 kv
selection_validation.json  rocprofv3 seam-engagement audit  (ok: true)
capture_meta.json        capture provenance
reference_io.pt          frozen oracle (captured from the live TP=8 baseline)
_baseline_random.pt      random-parity baseline outputs
kernel_src/              bpreshuffle_scale_impl.py   <- edit this  (currently the NULL candidate)
baseline_ref/            fp8_utils.py.orig
baseline_overlay/        empty manifest — the unmodified tree IS the baseline
_cand_overlay/           candidate overlay (currently binds the null candidate)
_candidate_best/         THE 1.4253x RESULT — winning impl + full author-lane record:
                           bpreshuffle_scale_impl.py   the merged HIP candidate
                           current_best.diff           the patch
                           round_1_metrics.json        rocprofv3 numbers quoted above
                           round_1_shift_analysis.md   why the bottleneck moved
                           analysis.json               probes, shapes, illegal-optimization notes
                           baseline_metrics.json       baseline rocprofv3 counters
                           baseline_timing.json
                           insight_log.md / profiling_summary.md / roadmap.md
_provenance/             build_meta.py, apply_weights.py, run_select.sh, run_kernel_selection.sh
profile_roofline.json/.md  stage-A roofline (this op is below the 5% scope, listed for context)
```

## Running it

Inside `harbor.crusoe.primus-safe.amd.com/hyperloom-image/sglang:v0.5.18-rocm720-mi35x-profilerfix`,
on one GPU from the optimization pool:

```bash
python3 unittest.py
```

Exit `0` pass · `1` correctness FAIL · `2` environment · `3` regenerate UT.
Prints `GEAK_PER_CASE` / `GEAK_WEIGHTED_DETAIL` / `GEAK_GEOMEAN_SPEEDUP` / `GEAK_WEIGHTED_SPEEDUP` /
`PASS|FAIL`. `tol = 1e-6` — bit-exact.

## Caveats

- **The profiler was degraded.** `rocprof-compute` exited 1 (ROCm 7.2.0 install missing 13 python
  packages: plotext, dash>=3.0.0, textual, sqlalchemy>=2.0.42, …). Fell back to `rocprofv3`
  deliberately, so **SoL / VALU% / VMEM% / LDS% / cache-hit are unavailable and reported `null`**
  rather than guessed. Not a renamed flag — no `RPC_PROFILE_ARGS` override fixes it.
- **118 CU, not 256.** `rocminfo` reports 118 CU on this MI355X against a nameplate of 256; every
  occupancy/fill figure above uses 118.
- **In the profile output, `vectorized_elementwise_kernel<4, FillFunctor<float>>` at 95.16% is the
  harness's own 128 MB L2-flush buffer, NOT the op.** The op is 3.20% of that trace. Anyone reading
  `profile_output_r1/` cold will misread this.
- **The serving weight collapses onto two buckets.** `serving_weighted_speedup` routes all of a
  regime's analytic calls to that regime's largest-M bucket; every decode bucket has M=64, so the
  tie resolves to the first workload case (`m64x32`), and prefill resolves to `m16384x2`. Those two
  carry essentially the whole weighted metric.
- **No e2e.** A 1.42x on ~3.2% of GPU time is ~0.95% of GPU time recovered, against a 0.5% e2e noise
  band on a launch-bound, TP-sync-bound decode. Treat it as unproven at e2e until someone runs the
  A/B — and note that on this box two *larger* proven isolated wins (1.4451x on 7.97%, 1.564x on
  7.97%) both converted to zero or negative.

## Provenance

| | |
|---|---|
| HL run | `provenance://shared-nfs/hongtaom/qwen3_14B/hl_matrix_0824/exp/glm53-flash/GLM-5.3-Flash/20260829T045924Z-d82dabe1` |
| task | `geak/e2e_cycle1/kernels/elementwise_copy_cluster_bpreshuffle_scale_and_mla_absorb_task` |
| author lane | `geak/e2e_cycle1/kernels/_exp/team_elementwise_copy_cluster_…_20260831_035342_1702329_396/` |
| image | `harbor.crusoe.primus-safe.amd.com/hyperloom-image/sglang:v0.5.18-rocm720-mi35x-profilerfix` |
| device | AMD Instinct MI355X / gfx950 / CDNA4, 118 CU, ~8 TB/s HBM, fp8=OCP |
| stack | torch 2.9.1+rocm7.2.0, sglang 0.5.18, hip 7.2.26015, gfx950 ×8 |
| session outcome | `status: no_gain`, `accepted_kernels: []`, `accepted_heads: []` |

Oracle integrity, verified at pack time and self-consistent with `meta.reference_io_sha256`:

```
sha256  14e589b7296c667b170b39a5f962344152206ff954a84b4e16de4346567d12af  reference_io.pt
```

All immutable files (`unittest.py`, `harness_lib.py`, `cases.py`, `leg_runner.py`,
`overlay_setup.py`, `meta.json`, `workload.json`, `regime.json`, `selection_validation.json`) are
md5-identical to the source task tree.
