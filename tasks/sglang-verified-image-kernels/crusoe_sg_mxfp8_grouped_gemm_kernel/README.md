# mi355x-sg-mxfp8-grouped-gemm

SGLang `_mxfp8_grouped_gemm_kernel` image_kernel harness verified on MI355X/gfx950 (2026-07-28).

Source: `kernels/ops/moe/mxfp8_moe_amd_gfx95.py` (`origin/main`). Cases from 7-24 hot-kernel analysis (P0). Verified: `task_runner` compile/correctness/performance + SGLang `test_mxfp8_native_moe`.
