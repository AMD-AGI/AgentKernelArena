# glm-5.3-flash__ck_gemm_a8w8_blockscale_bpreshuffle - NOT_BUILT

**GLM-5.3-Flash** - `ck_gemm_..._blockscale_b_` (CK / CKTile via aiter, 5.65% GPU).

This is a placeholder, **not a task**. It carries no `config.yaml`, so the arena
task scanner will not pick it up and it cannot be run or scored.

## Why

IMAGE-VERIFIED 2026-09-15: a Python dispatcher for this seam EXISTS, which corrects the earlier note that nothing Python defines it. sglang:v0.5.17-rocm720-mi35x-profilerfix (sglang 2948168546, aiter d9e5ef7ce08e) carries /sgl-workspace/aiter/aiter/ops/gemm_op_a8w8.py with gemm_a8w8_blockscale_bpreshuffle:896 plus _ck:316, _cktile:331, _asm:397 and _flydsl:203 variants, and the production entry is sglang srt/layers/quantization/fp8_utils.py:1062. What remains genuinely non-editable is the profiled DEVICE symbol itself - a CK C++ template instantiation (ck::kernel_gemm_xdl_cshuffle_v3_multi_d_blockscale_b_preshuffle) with no Python behind it. So this row is optimizable only at the dispatch/tiling level or by replacing the CK call, not by editing the kernel. The package has 21 cases, no kernel_src and no frozen oracle (meta.oracle = 'synthesized_dequant_fp32'). Measured 1.4451x isolated -> -0.438% e2e.

| field | value |
|---|---|
| GPU time share | 5.65% |
| empirical roofline | 0.029 |
| optimized roofline | 0.029 |
| e2e uplift measured | -0.13% |
| device symbol | `ck_gemm_a8w8_blockscale_bpreshuffle` |
| production seam | `-` |
| info rows | G53-3 |
| upstream UT package | `H/GLM-5.3-Flash_ck_gemm_a8w8_blockscale_bpreshuffle_0913` |

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
