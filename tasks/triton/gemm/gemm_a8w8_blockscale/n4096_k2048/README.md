# fp8 block-scaled GEMM, N = 4096, K = 2048

`out = (a * a_scale) @ (b * b_scale)^T`: float8_e4m3fn activations `a` [m, 2048] with one float32 scale per 1x128
group along K, float8_e4m3fn weights `b` [4096, 2048] with one float32 scale per 128x128 block (`b_scale` [32, 16]).
Each 128-wide K block of the product is accumulated in float32, multiplied by the scales of its blocks and summed;
the result is rounded to bfloat16. This is the DeepSeek-style block-scaled linear layer used for fp8 serving.

**Layouts.** Production hands the kernel prepared operands, so the task declares the kernel-facing forms and
`inputs.py::to_kernel_layout` produces them outside timing:

- `b` is AITER's `shuffle_weight(layout=(16, 16))` tiling: `[N/16, 16, K/32, 2, 16]` reordered to
  `[N/16, K/32, 2, 16, 16]` (weight preparation done once at load time).
- `a_scale` is the column-major payload of the logical `[m, 16]` scale, stored as a contiguous `[m, 16]`
  tensor (element (row, group) at offset `group * m + row`), which is what an activation-quantization kernel
  writes. The activation quantization itself is upstream of this operator and not part of it.

The reference receives the logical operands (unshuffled `b`, plain `a_scale`) and dequantizes and multiplies in float64.

**Inputs.** `inputs.py` draws normal activations (std 1) and weights (std 0.02) and quantizes them with scale
`amax / 448` per 1x128 group and per 128x128 block, so codes and scales are consistent and the output
is meaningful.

**Workloads.** Derived from the old AITER FlyDSL task of the same shape (M of 1 to 4096). The weight is 8 MB; the estimated baseline time exceeds 10 us from about M = 1024 (fp8 flops / 2e15 plus a 4 us floor), so smaller M are correctness-only. Performance cases run from 1024 to 131072 tokens (128x) as prefill chunk sizes.

**Baseline.** AITER's Triton preshuffle kernel `_gemm_a8w8_blockscale_preshuffle_kernel` with its launcher and
the generic gfx950 tuned-config table inlined (see `provenance` in `task.yaml`; `baseline/LICENSE` is the AITER MIT
license). AITER has no N/K-specialized gfx950 config for this shape, so the table's tile choice depends only on M
(`M_LEQ_8` ... `any`); every row has no split-K. The old tasks' baseline was AITER's CK/assembly
`gemm_a8w8_blockscale_bpreshuffle`, which cannot be vendored as source; this Triton variant replaces it and is the
baseline here. It is slower than that assembly path, so speedups are relative to the Triton kernel. Upstream clamps
`num_stages` to 2 on Triton < 3.7 for gfx950 (a compiler limitation); the baseline keeps that clamp. The one
deviation from the launcher is that `stride_ck` is passed as `out.stride(0)` instead of 0: it is unused without split-K
and the kernel asserts it positive.

**GPU audit risks.** The smallest performance case (M = 1024) is estimated at about 12 us; verify it clears the 10 us floor on the GPU audit. The baseline is Triton, so it must also satisfy the Triton candidate rules (no PyTorch
compute in `run()`; the only host-side operations are views and the scale-transpose view).
