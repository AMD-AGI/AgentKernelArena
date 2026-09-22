# glm-5.3-flash__gemm_a16w16_bf16_cijk - NOT_BUILT

**GLM-5.3-Flash** - `Cijk_..._MT16x16x1024_` (hipBLASLt / Tensile, 6.31% GPU).

This is a placeholder, **not a task**. It carries no `config.yaml`, so the arena
task scanner will not pick it up and it cannot be run or scored.

## Why

IMAGE-VERIFIED 2026-09-15: the editable source EXISTS in the row's own runtime image. sglang:v0.5.17-rocm720-mi35x-profilerfix (sglang 2948168546, aiter d9e5ef7ce08e) carries /sgl-workspace/aiter/aiter/tuned_gemm.py (23148 bytes, gemm_a16w16:354, torch_gemm:479, class TunedGemm:683, solMap:672) - byte-size and aiter commit identical to the copy already vendored in this suite at tasks/headkernel/qwen3.8-2.4t__dense_bf16_gemm_cluster/source/tuned_gemm_candidate.py, so that copy is version-correct for this row. The package ships 27 cases and the clearest rebind recipe of any open row, but no kernel_src and no frozen oracle (meta.oracle is the string 'synthesized_fp32_matmul'). Remaining work: wire the source in, add candidate_bind (watch the import-time solMap - a bare setattr is a dead rebind), capture an oracle. Tuning it in cycle 0 cost -7.01% e2e, so verify the headroom before investing.

| field | value |
|---|---|
| GPU time share | 6.31% |
| empirical roofline | 0.024 |
| optimized roofline | 0.024 |
| e2e uplift measured | -7.01% |
| device symbol | `Cijk_..._MT16x16x1024_MI / Cijk_..._MT32x32x512_MI1` |
| production seam | `aiter.tuned_gemm:gemm_a16w16` |
| info rows | G53-2 |
| upstream UT package | `H/GLM-5.3-Flash_gemm_a16w16_bf16_Cijk_0913` |

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
