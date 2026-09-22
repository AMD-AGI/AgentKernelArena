# minimax-m3__gemm_afp4wfp4_kernel - NOT_BUILT

**MiniMax-M3-MXFP4** - `_gemm_afp4wfp4_kernel` (Triton, 5.14% GPU).

This is a placeholder, **not a task**. It carries no `config.yaml`, so the arena
task scanner will not pick it up and it cannot be run or scored.

## Why

IMAGE-VERIFIED 2026-09-15: the editable source EXISTS in the row's own runtime image. sglang:v0.5.17-rocm720-mi35x-profilerfix (sglang 2948168546, aiter d9e5ef7ce08e) carries the Triton kernel at /sgl-workspace/aiter/aiter/ops/triton/_triton_kernels/gemm/basic/gemm_afp4wfp4.py:36 (_gemm_afp4wfp4_kernel, 25526 bytes) and its wrapper at aiter/ops/triton/gemm/basic/gemm_afp4wfp4.py:259 (gemm_afp4wfp4, 21731 bytes); a gluon variant also exists. The UT package is the problem, not the source: kernel_src/ holds only .gitkeep, candidate_bind is the empty stub {'kind':'module','module':'','file':'kernel_src/'}, and there is no reference_io.pt - unittest.py prints the UT_HARNESS_INCOMPLETE sentinel and exits 3. Remaining work: vendor the two files, fill in candidate_bind, and CAPTURE AN ORACLE - that last one needs a live server run and is the real cost.

| field | value |
|---|---|
| GPU time share | 5.14% |
| empirical roofline | 24% |
| optimized roofline | - |
| e2e uplift measured | - |
| device symbol | `_gemm_afp4wfp4_kernel_BLOCK_SIZE_M_32_BLOCK_SIZE_N_32_BLOCK_SIZE_K_512_...` |
| production seam | `aiter.ops.triton.gemm_afp4wfp4:gemm_afp4wfp4` |
| info rows | MM-4 |
| upstream UT package | `Z/MiniMax-M3-MXFP4_gemm_afp4wfp4_kernel` |

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
