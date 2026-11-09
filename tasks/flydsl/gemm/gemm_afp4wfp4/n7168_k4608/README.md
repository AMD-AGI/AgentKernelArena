# afp4wfp4 GEMM, N = 7168, K = 4608 (FlyDSL baseline)

`out = dequant(a) @ dequant(w)^T`: MXFP4 activations `a` and weights `w`, both stored as two e2m1 codes per byte
(the even K index is the low nibble) with uint8 e8m0 scales (`2^(s - 127)`), one per 32 values of K. The block products
are accumulated in float32 and the result is rounded to bfloat16. `m` is the only varying axis. The operator, the
logical inputs, the reference, the generator and the scenario are those of the Triton task
`tasks/triton/gemm/gemm_afp4wfp4/n7168_k4608`; only the kernel-facing layout, the workloads and the baseline differ.

## Scenario and config citations

The DeepSeek-V3 dense-layer MLP down projection at tensor parallelism 4: N = `hidden_size` 7168, K =
`intermediate_size` 18432 / 4 = 4608. Both fields are in the `deepseek` / `V3` entry of
`op_tests/op_benchmarks/triton/utils/model_configs.json` of ROCm/aiter at `652f2d4b768239aefd515c929e84817681ff598d`
(`hidden_size` 7168, `intermediate_size` 18432), and `bench_gemm_afp4wfp4.py` of the same commit derives the `fc2` layer
as `N, K = hidden_dim, intermediate_dim` with `K = ceil(K / tp)`. The tensor-parallel degree 4 is my reading of the
shape: AITER's test table `get_x_vals` in `op_tests/triton_tests/gemm/basic/test_gemm_afp4wfp4.py` lists
`(v, 7168, 4608)` without naming a model or a TP degree. The pinned FlyDSL, vLLM and SGLang contain no caller of this
kernel for this shape; it is the shape the Triton sibling uses, so one operator means the same thing in both
languages. FlyDSL requires N to be a multiple of the tile width 256 (7168 = 28 x 256) and K a multiple of 256
(4608 = 18 x 256).

## Kernel-facing layout

The FlyDSL preshuffle GEMM reads operands in a prepared layout that production produces once at load time (weights) or
in the preceding quantization kernel (activation scales). `to_kernel_layout` in `inputs.py` applies it outside timing,
so the baseline and every candidate receive:

- `a` unchanged, plain row-major `[m, 2304]`;
- `w` [7168, 2304] with each 16 x 64 byte tile permuted as FlyDSL's `shuffle_weight_w4(w, NLane=16)`: reshape to
  `(n/16, 16, k/128, 4, 16)` and permute to `(n/16, k/128, 4, 16, 16)`;
- `w_scales` [7168, 144] permuted as `shuffle_scale_w4(., 1, False)`: reshape to `(rows/32, 2, 16, k/256, 2, 4)` and
  permute to `(rows/32, k/256, 4, 16, 2, 2)`;
- `a_scales` [ceil(m / 32) * 32, 144]: rows padded to a multiple of 32 with scale 1.0 and then shuffled like `w_scales`
  (padded rows hold no activation codes).

Both permutations were compared on CPU against `shuffle_weight_w4` and `shuffle_scale_w4` of
`tests/kernels/utils/gemm_common_utils.py` at the pinned FlyDSL commit (see "Checks run" below). The reference receives
the logical tensors.

## Reference and inputs

`reference.py` dequantizes both operands from codes and scales and multiplies in float64, row-chunked; it is
independent of the baseline and of the quantizer. `inputs.py` draws normal activations (std 1) and weights (std 0.05)
and quantizes them like AITER's dynamic MXFP4 quantization: the scale of a 1x32 group is the group maximum rounded to a
power of two minus the exponent of the largest e2m1 value; codes round to nearest with ties to the even code,
saturating at 6. In production the activation quantization is a separate kernel, so it is part of the input, not timed.

## Baseline

`launch_gemm` of `kernels/gemm/mxfp4_preshuffle.py` (ROCm/FlyDSL `dd8373349e33638c8d158a9ea009b7b28598984a`, tag
v0.3.2, Apache-2.0; header `# Copyright (c) 2025 FlyDSL Project Contributors` kept in both files): MXFP4 activations
times MXFP4 weights through scaled 16x16x128 MFMAs, activations streamed through LDS. `kernel.py` follows the call
shape of `_mxfp4_launcher` in `tests/kernels/test_preshuffle_gemm.py`, with tile 64x256x256, the tile of that test's
large-M cases (`M = 1024` and `5133`, N = K = 8192) which include a ragged M. There is no tuned per-shape table for this
kernel in FlyDSL, so none is invented: every `m` uses this one tile. Third-party origin: neither file names one, and the
history of `mxfp4_preshuffle.py` starts with FlyDSL's own commit "Port fp4 gemm to new api and remove original
preshuffle gemm" (#754).

Why not `kernels/gemm/fp4_gemm_4wave.py`: it is hard-wired to 256 x 256 output tiles with an epilogue that has no bounds
checks (the kernel's own test asserts `M % 256 == 0`), so it cannot serve decode-size or ragged `m`, and it reads the
environment (`FP4_DMA_INTRINSIC`), which candidates may not do.

### FlyDSL 0.3.2 adaptation

- `mxfp4_preshuffle.py` imports `xcd_remap_bx_by` from the repository-level `kernels.common` package, which the
  installed FlyDSL release does not ship. The import is removed together with the `xcd_swizzle` constexpr parameter and
  its `if const_expr(xcd_swizzle > 0)` branch; the launch always used `xcd_swizzle` 0, so the compiled kernel is the
  same.
- The vendored FlyDSL is the version of the evaluation image; no other API port is needed.
- `launch_gemm` is a `@flyc.jit` function that caches compiled kernels per Constexpr configuration; the argument checks
  are cached per `(n, k)` with `functools.lru_cache`.

## Workloads

`m` is the token count of a step, from decode batches to batched prefill chunks. The weight is 16.5 MB. Estimated
baseline time (dense fp4 flops / 4.0e15 plus 4 us): 8.5 us at 256 tokens, 12.5 us at 512; the smallest performance case
is therefore 512 tokens, and the cases up to 5133 tokens that are not performance cases are correctness-only: 1, 3, 8,
17, 100, 192, 1000 and 5133 (not a multiple of the tile, as in the FlyDSL test). Performance cases run from 512 to 65536
tokens (128x); the largest is about 4.3e12 flops, estimated 1.1 ms. Its output is 0.94 GB and the float64 reference
3.8 GB.

## Kernel notes

A tile of 64
rows gives `ceil(m / 64) x 28` workgroups: 28 at one token, so the decode cases are launch-bound but correctness-only.
The kernel reads the activation matrix through a buffer resource bounded to the real `m` rows, so ragged `m` relies on
out-of-range reads returning zero; cases 1, 3, 8, 17, 100, 192, 1000 and 5133 exercise that. 512 tokens (112
workgroups on 256 CUs) is the estimated floor of the performance cases, near the 10 us floor. The first call per
configuration compiles.
