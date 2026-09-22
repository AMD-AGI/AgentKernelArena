# kimi-k3__dense_bf16_gemm_cijk - NOT_BUILT

**Kimi-K3** - `Cijk_..._MT256x256x64_MI` (hipBLASLt / Tensile via aiter.tuned_gemm, 9.14% GPU).

This is a placeholder, **not a task**. It carries no `config.yaml`, so the arena
task scanner will not pick it up and it cannot be run or scored.

## Why

IMAGE-VERIFIED 2026-09-15: the editable source EXISTS in the row's own runtime image. sglang-rocm-k3:rocm720-mi35x-k3-20260727-tl312-08011830 (sglang 0e756912eb3c, aiter 68e42f5f4615) carries /sgl-workspace/aiter/aiter/tuned_gemm.py (23412 bytes) defining gemm_a16w16:352, torch_gemm:480, class TunedGemm:684. Note it is a DIFFERENT aiter build from the copy already in this suite (23148 bytes, aiter d9e5ef7) - take K3-2's copy from the k3 image, not from qwen3.8-2.4t__dense_bf16_gemm_cluster/source/. The package itself ships a 210 MB frozen oracle but no kernel_src and no candidate_bind, so the remaining work is: vendor tuned_gemm.py from the k3 image, add candidate_bind, re-capture parity. Hazard: solMap:673 is built at import time holding direct function objects, so a bare setattr on the module is a DEAD rebind. Caveat on value: the 0831 audit found this is the one genuinely COMPUTE-bound head (AI 2356 >> ridge 312, 63.69% of roof) with hipBLASLt already optimal.

| field | value |
|---|---|
| GPU time share | 9.14% |
| empirical roofline | 0.647 |
| optimized roofline | 0.648 |
| e2e uplift measured | +5.479% |
| device symbol | `Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT256x256x64_MI16x16x1_...` |
| production seam | `aiter.tuned_gemm:torch_gemm` |
| info rows | K3-2 |
| upstream UT package | `Z/Kimi-K3_dense_bf16_gemm_cijk` |

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
