# fp8 block-scaled GEMM, N = 8192, K = 1024

`out = (a * a_scale) @ (b * b_scale)^T`: float8_e4m3fn activations `a` [m, 1024] with one float32 scale per 1x128
group along K, float8_e4m3fn weights `b` [8192, 1024] with one float32 scale per 128x128 block (`b_scale` [64, 8]).
Each 128-wide K block of the product is accumulated in float32, multiplied by the scales of its blocks and summed;
the result is rounded to bfloat16. This is the DeepSeek-style block-scaled linear layer used for fp8 serving.

**Layouts.** Production hands the kernel prepared operands, so the task declares the kernel-facing forms and
`inputs.py::to_kernel_layout` produces them outside timing:

- `b` is AITER's `shuffle_weight(layout=(16, 16))` tiling: `[N/16, 16, K/32, 2, 16]` reordered to
  `[N/16, K/32, 2, 16, 16]` (weight preparation done once at load time).
- `a_scale` is the column-major payload of the logical `[m, 8]` scale, stored as a contiguous `[m, 8]` tensor
  (element (row, group) at offset `group * m + row`), which is what an activation-quantization kernel writes. The
  activation quantization itself is upstream of this operator and not part of it.

The reference receives the logical operands (unshuffled `b`, plain `a_scale`) and dequantizes and multiplies in float64.

**Inputs.** `inputs.py` draws normal activations (std 1) and weights (std 0.02) and quantizes them with scale
`amax / 448` per 1x128 group and per 128x128 block, so codes and scales are consistent and the output is meaningful.

**Workloads.** The old task (a FlyDSL port of the AITER op) swept M from 1 to a few thousand tokens. Here M runs
to prefill chunk sizes so that 100x of span is reachable: counts up to 1000 are correctness-only (the weight is 8 MB;
fp8 flops / 2e15 plus a 4 us floor is about 12.6 us at M = 1024), performance cases run from 1024 to 131072 tokens
(128x). The largest case writes a 2 GB output and its float64 reference output is about 8.6 GB.

**Baseline.** AITER's Triton preshuffle kernel `_gemm_a8w8_blockscale_preshuffle_kernel` with its launcher and the
generic gfx950 tuned-config table inlined (see `provenance` in `task.yaml`; `baseline/LICENSE` is the AITER MIT
license; the inlined table equals `gfx950-GEMM-A8W8_BLOCKSCALE_PRESHUFFLED.json`). AITER has no N/K-specialized gfx950
config for this shape, so the tile choice depends only on M (`M_LEQ_8` ... `any`); every row has no split-K. The old
task's baseline was AITER's CK/assembly `gemm_a8w8_blockscale_bpreshuffle`, which cannot be vendored as source; this
Triton variant replaces it and is the baseline here. It is slower than that assembly path, so speedups are relative
to the Triton kernel. Upstream clamps `num_stages` to 2 on Triton < 3.7 for gfx950 (a compiler limitation); the
baseline keeps that clamp. The one deviation from the launcher is that `stride_ck` is passed as `out.stride(0)`
instead of 0: it is unused without split-K and the kernel asserts it positive. The baseline is identical to that of
the sibling N = 4096 / 32768 scenarios.

**GPU audit risks.** The smallest performance case (M = 1024) is estimated at about 12 us; verify it clears the 10 us
floor. Memory of the M = 131072 case (inputs, 2 GB output, 8.6 GB fp64 reference) is large but fits. The baseline is
Triton, so it must also satisfy the Triton candidate rules (no PyTorch compute in `run()`; the only host-side
operations are views and the scale-transpose view).
