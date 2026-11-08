# Batched afp4wfp4 GEMM (FlyDSL baseline), 128 batch elements, N = 128, K = 512

`out[b] = dequant(x[b]) @ dequant(w[b])^T` for 128 batch elements. Activations `x` [batch, m, 256] and weights
`w` [batch, 128, 256] are MXFP4: two e2m1 codes per byte (the even K index is the low nibble) with uint8 e8m0
scales (`2^(s - 127)`), one per 32 values of K. The block products are accumulated in float32 and the result is rounded
to bfloat16. `m` is the only varying axis. The operator and its logical inputs are those of the Triton task
`tasks/triton/gemm/batched_gemm_afp4wfp4/b128_n128_k512` (same reference, same generator, same workloads); only
the kernel-facing layout and the baseline differ.

## Scenario and config citations

The scenario is the DeepSeek multi-head latent attention value up-projection (the absorbed `W_UV` batched
GEMM): each attention head multiplies its latent attention output by its slice of `W_UV`. DeepSeek-V3
`config.json` (Hugging Face, `deepseek-ai/DeepSeek-V3`): `num_attention_heads` 128 (batch), `kv_lora_rank` 512 (K),
`v_head_dim` 128 (N). 128 heads is the single-rank head count; with tensor parallelism the batch shrinks to 128 / TP.
The FlyDSL kernel requires K to be a multiple of 256 and N a multiple of the 128-wide tile, which this shape meets
(the transposed absorbed W_UK projection, K = 128, would not). At the pinned AITER commit the kernel is exposed as
`flydsl_batched_gemm_mxfp4`; no caller of it was found in the pinned vLLM and SGLang, so the shape is the
MLA absorbed projection, not an observed call.

## Kernel-facing layout

The FlyDSL preshuffle GEMM reads operands in a prepared layout that production produces once at load time (weights)
or in the preceding quantization kernel (activation scales). `to_kernel_layout` in `inputs.py` applies it outside timing,
so the baseline and every candidate receive:

- `x` unchanged, plain row-major `[batch, m, k // 2]`;
- `w` [batch, 128, 256] with each 16 x 64 byte tile permuted as FlyDSL's `shuffle_weight_w4(w, NLane=16)`:
  reshape to `(n/16, 16, k/128, 4, 16)` and permute to `(n/16, k/128, 4, 16, 16)` per batch element;
- `w_scales` [batch, 128, 16] permuted as `shuffle_scale_w4`: reshape the `[rows, k/32]` scales to
  `(rows/32, 2, 16, k/256, 2, 4)` and permute to `(rows/32, k/256, 4, 16, 2, 2)` per batch element;
- `x_scales` [batch, ceil(m / 32) * 32, 16]: the rows padded to a multiple of 32 with scale 1.0 (the kernel needs
  whole 32-row chunks; padded rows hold no activation codes, so they do not affect the output) and then shuffled like
  `w_scales`.

The shuffles were checked against the FlyDSL functions on CPU (`tests/kernels/utils/gemm_common_utils.py` at the
pinned commit). The reference receives the logical (unshuffled, unpadded) tensors. Weight codes and scales stay
read-only inputs.

## Reference and inputs

`reference.py` dequantizes both operands from codes and scales and multiplies per batch element in float64
(row-chunked); it is independent of the baseline and of the quantizer in `inputs.py`. `inputs.py` draws normal
activations (std 1) and weights (std 0.05) per batch element and quantizes them like AITER's dynamic MXFP4
quantization: the scale of a 1x32 group is the group maximum rounded to a power of two minus the exponent of the
largest e2m1 value, codes round to nearest with ties to the even code, saturating at 6. Outputs have std about 1.1.

## Workloads

Same as the Triton task: `m` is the number of tokens of a step, from decode batches to batched prefill chunks. The
performance cases are memory-bound (about 68 KB moved per token across the 128 heads: 272 B of fp4 activations and
scales plus 256 B of bfloat16 output per head) and start at 384 tokens, estimated at 10.8 us (bytes / 4.5e12 B/s plus
4 us; 256 tokens would be about 8.8 us and is correctness-only). 38400 tokens (estimated about 580 us) give the
required 100x span. The largest case holds 1.3 GB of inputs and a 5.0 GB float64 reference output. Boundary cases: 1
token, 3 (non power of two), 17, 100, 1000 and 3000 (not multiples of 16, above 16).

## Baseline

The FlyDSL `launch_gemm` of `kernels/gemm/mxfp4_preshuffle.py` (ROCm/FlyDSL v0.3.2, Apache-2.0: MXFP4 activations
times MXFP4 weights through scaled 16x16x128 MFMAs, activations streamed through LDS by double-buffered async DMA)
with the launcher of AITER's `flydsl_batched_gemm_mxfp4` (ROCm/aiter, MIT) for the gfx950, `bmn`, fp4-activation
path: tile 128x128x256 for every `m` (AITER's default; there is no tuned per-shape table for this kernel, so none is
invented), one `z` grid slice per batch element. `kernel.py` is the launcher and `mxfp4_preshuffle.py` the kernel;
each keeps its upstream SPDX/copyright header. `baseline/LICENSE` is the AITER MIT license and
`baseline/LICENSE-FlyDSL` the FlyDSL Apache-2.0 license. No third-party origin is recorded in either upstream file
or its upstream directory; the FlyDSL file was added to the FlyDSL history by the commit "Port fp4 gemm to new api and remove original
preshuffle gemm" (#754), a port within the same project.

### FlyDSL 0.3.2 adaptation

- `mxfp4_preshuffle.py` imports `xcd_remap_bx_by` from the repository-level `kernels.common` package, which the
  installed FlyDSL release does not ship. The import is removed with the `xcd_swizzle` constexpr parameter and its
  `if const_expr(xcd_swizzle > 0)` branch. The launcher always used `xcd_swizzle` 0, so the compiled kernel is the
  same.
- AITER's call predates the `b_dtype` argument that `launch_gemm` has in v0.3.2 (between `out_dtype` and `batch`);
  its positional call would shift every following argument. `kernel.py` passes `b_dtype="fp4"` and matches the signature of
  v0.3.2 (the call shape of that release's `tests/kernels/test_preshuffle_gemm.py`).
- AITER's `ptr_arg` shim is replaced by `flyc.from_c_void_p(fx.Uint8, t.data_ptr())`, the same call the FlyDSL test
  uses; the launcher's output allocation and layout switch (`mbn`) are not needed.
- `launch_gemm` is a `@flyc.jit` function that caches its compiled kernel per Constexpr configuration; the argument
  checks are cached per `(batch, n, k)` with `functools.lru_cache`.

## Kernel notes

Tile 128x128x256 with N = 128
gives one tile column, so the grid is `ceil(m / 128) x 1 x 128`; at 384 tokens that is 384 workgroups for the
256 CUs. The first call per configuration compiles (the evaluator's correctness phase precedes timing). The kernel reads
the activation matrix through a buffer resource bounded to the real `m` rows, so ragged `m` relies on out-of-range
reads returning zero; correctness cases 17, 100, 1000 and 3000 exercise that. M = 384 is the estimated floor of
the performance cases. The comparison keeps several float64 temporaries of the whole output (5 GB each at the largest case).
