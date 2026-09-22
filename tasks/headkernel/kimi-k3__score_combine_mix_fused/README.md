# kimi-k3__score_combine_mix_fused - NOT_BUILT

**Kimi-K3** - `_score_kernel (+ _combine_kernel)` (Triton - sglang srt/layers/attn_residual.py, two launches from one callable _mix_fused, 5.26% GPU).

This is a placeholder, **not a task**. It carries no `config.yaml`, so the arena
task scanner will not pick it up and it cannot be run or scored.

## Why

IMAGE-VERIFIED 2026-09-15: the editable source EXISTS in the row's own runtime image. sglang-rocm-k3:rocm720-mi35x-k3-20260727-tl312-08011830 (sglang 0e756912eb3c, aiter 68e42f5f4615) carries /sgl-workspace/sglang/python/sglang/srt/layers/attn_residual.py (17379 bytes) defining @triton.jit _score_kernel:103, @triton.jit _combine_kernel:143 and the _mix_fused:197 callable that launches both - all three targets. VERSION WARNING: the baseline_ref/attn_residual.py.orig the UT package ships is 19681 bytes, and the v0.5.17 image's copy is 19298 - neither matches the 17379 in the k3 image this row actually runs under. Confirm with the package owner which build the capture came from before vendoring. The package has no kernel_src, no frozen oracle (hand-written UT, fp32 reference + live parity), no candidate_bind and no overlay_setup.py, so an edit to source/ would be inert in the measurement until a bind is added. Note also that the k3 image has no kernels/ops/kimi_k3/attn_res_hip.py, i.e. the 1.4994x replacement path measured elsewhere is not in this image.

| field | value |
|---|---|
| GPU time share | 5.26% |
| empirical roofline | None |
| optimized roofline | - |
| e2e uplift measured | - |
| device symbol | `_score_kernel + _combine_kernel` |
| production seam | `sglang.srt.layers.attn_residual:_mix_fused` |
| info rows | K3-6 |
| upstream UT package | `Z/Kimi-K3_score_kernel` |

## To promote it into the suite

1. Get an editable implementation of the seam into `source/`. Read the Why above
   first - it says whether the source has to be written (the profiled symbol is a
   prebuilt vendor artifact with no Python behind it) or merely vendored (the
   package shipped an empty `kernel_src/` but the Triton source exists upstream,
   and for the `aiter.tuned_gemm` rows a stock copy already ships in this suite at
   `tasks/headkernel/qwen3.8-2.4t__dense_bf16_gemm_cluster/source/`).
2. Add `candidate_bind` to the package `meta.json` so the candidate leg actually
   shadows the production callable - without it both legs resolve to the same code
   and any measured speedup is noise. For the `aiter.tuned_gemm` rows note that a
   bare `setattr` on the module is a DEAD rebind: `solMap` is built at import time
   holding direct function objects, so the dispatcher keeps calling the original.
3. Re-capture parity against the live server and confirm
   `selection_validation.ok == true`.
4. Re-run `tools/build_suite.py`; flip `built` to true in `tools/manifest.json`.
